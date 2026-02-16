DOCKER_USER=go2
DOCKER_HOME=/home/${DOCKER_USER}


# Default docker network
if [ -z "$DOCKER_NETWORK" ]; then
    export DOCKER_NETWORK="host"
fi
echo "# Using host DOCKER_NETWORK=$DOCKER_NETWORK"
NETWORK_SETTINGS="--network $DOCKER_NETWORK "

# x11
if [ -z "${DISPLAY:-}" ]; then
    echo "DISPLAY is not set on host. Cannot forward X11."
    exit 1
fi

if [ -d "/tmp/.X11-unix.$(id -un)" ]; then
    XSOCK="/tmp/.X11-unix.$(id -un)"
else
    XSOCK="/tmp/.X11-unix"
fi
XAUTH="/tmp/.docker.xauth.$(id -un)"
touch "$XAUTH"

if command -v xauth >/dev/null 2>&1; then
    XAUTH_SOURCE="${XAUTHORITY:-$HOME/.Xauthority}"
    xauth -f "$XAUTH_SOURCE" nlist "$DISPLAY" 2>/dev/null | \
        sed -e 's/^..../ffff/' | \
        xauth -f "$XAUTH" nmerge - >/dev/null 2>&1 || true
else
    echo "Warning: host 'xauth' not found; X11 auth may fail in container."
fi

# Allow container users with different UID to read the cookie file.
chmod a+r "$XAUTH" || true

# graphics
GRAPHICS_SETTINGS+="  --runtime=nvidia \
                      -e DISPLAY=$DISPLAY \
                      -e GPU=true \
                      -e NVIDIA_DRIVER_CAPABILITIES=all \
                      -e NVIDIA_VISIBLE_DEVICES=all \
                      --gpus all \
                      -e XAUTHORITY=$XAUTH \
                      -v $XSOCK:$XSOCK:rw \
                      -v $XAUTH:$XAUTH:ro"

# ros2 ws
export ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "# Using ROS2_WS=$ROS2_WS"

docker run --rm -it \
    $NETWORK_SETTINGS \
    $GRAPHICS_SETTINGS \
    --name go2_ros2 \
    go2_ros2 \
    "$@"
