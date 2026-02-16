#!/usr/bin/env bash

WORKSPACE_NAME=simdist
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

ENV_FILE="${ROS2_WS}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "# Loading env file $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -n "${CYCLONEDDS_IFACE:-}" ]; then
    echo "# Using CYCLONEDDS_IFACE=$CYCLONEDDS_IFACE"
else
    echo "Error: Could not determine network interface for CycloneDDS."
    echo "Configure go2_ros2_ws/.env with the appropriate interface name."
    exit 1
fi

export HOST_WORKSPACE_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
export CONTAINER_WORKSPACE_ROOT="${CONTAINER_WORKSPACE_ROOT:-${DOCKER_HOME}/$WORKSPACE_NAME}"
export CONTAINER_ROS2_WS="${CONTAINER_WORKSPACE_ROOT}/go2_ros2_ws"

echo "# Mounting HOST_WORKSPACE_ROOT=$HOST_WORKSPACE_ROOT"
echo "# Into CONTAINER_WORKSPACE_ROOT=$CONTAINER_WORKSPACE_ROOT"

docker run --rm -it \
    $NETWORK_SETTINGS \
    $GRAPHICS_SETTINGS \
    -v "${HOST_WORKSPACE_ROOT}:${CONTAINER_WORKSPACE_ROOT}:rw" \
    -w "${CONTAINER_ROS2_WS}" \
    -e ROS2_WS="${CONTAINER_ROS2_WS}" \
    -e SIMDIST_WORKSPACE_ROOT="${CONTAINER_WORKSPACE_ROOT}" \
    -e CYCLONEDDS_IFACE="${CYCLONEDDS_IFACE:-}" \
    --name go2_ros2 \
    go2_ros2 \
    "$@"
