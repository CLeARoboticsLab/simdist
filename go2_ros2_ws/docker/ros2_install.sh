#!/usr/bin/env bash

export ROS_DISTRO="${ROS_DISTRO:-humble}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update && \
sudo apt upgrade -y && \
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-colcon-clean \
    python3-colcon-common-extensions \
    python3-colcon-lcov-result \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    ros-${ROS_DISTRO}-ament-cmake-black \
    ros-${ROS_DISTRO}-ament-cmake-clang-format \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-foxglove-bridge \
    ros-${ROS_DISTRO}-grid-map \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-rviz-2d-overlay-msgs \
    ros-${ROS_DISTRO}-rviz-2d-overlay-plugins \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-plotjuggler-ros \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-console \
    ros-${ROS_DISTRO}-rqt-py-common \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-test-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-tf2-kdl \
    ros-${ROS_DISTRO}-tf2-msgs \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-ros-py \
    ros-${ROS_DISTRO}-tf2-sensor-msgs \
    ros-${ROS_DISTRO}-vrpn-mocap \
    ros-${ROS_DISTRO}-xacro || exit 1
