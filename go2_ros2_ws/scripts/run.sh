DOCKER_USER=go2
DOCKER_HOME=/home/${DOCKER_USER}


# Default docker network
if [ -z "$DOCKER_NETWORK" ]; then
    export DOCKER_NETWORK="host"
fi
echo "# Using host DOCKER_NETWORK=$DOCKER_NETWORK"
NETWORK_SETTINGS="--network $DOCKER_NETWORK "

# x11
# if [ -d "/tmp/.X11-unix.$(id -un)" ]; then
#     XSOCK="/tmp/.X11-unix.$(id -un)"
# else
#     XSOCK="/tmp/.X11-unix"
# fi
# XAUTH=/tmp/.docker.xauth.$(id -un)
# touch $XAUTH

# graphics
GRAPHICS_SETTINGS+="  --runtime=nvidia \
                      -e DISPLAY=$DISPLAY \
                      -e GPU=true \
                      -e NVIDIA_DRIVER_CAPABILITIES=all \
                      -e NVIDIA_VISIBLE_DEVICES=all \
                      --gpus all \
                      -e XAUTHORITY=$XAUTHORITY \
                      -v /tmp/.X11-unix:/tmp/.X11-unix \
                      -v $XAUTHORITY:$XAUTHORITY"
                    #   -v $XSOCK:$XSOCK:rw \
                    #   -v $XAUTH:$XAUTH:rw \
                    #   -e XAUTHORITY=${XAUTH}"

# ros2 ws
export ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "# Using ROS2_WS=$ROS2_WS"

docker run --rm -it \
    $NETWORK_SETTINGS \
    $GRAPHICS_SETTINGS \
    --name go2_ros2 \
    go2_ros2 \
    "$@"
    