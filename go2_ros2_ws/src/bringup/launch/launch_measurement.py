import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
measurement_config = os.path.join(config_dir, "config", "measurement.yaml")
transform_cloud_config = os.path.join(config_dir, "config", "transform_cloud.yaml")


def generate_launch_description():
    repub_body_imu = Node(
        package="measurement",
        executable="repub_body_imu",
        name="repub_body_imu",
        output="screen",
        parameters=[common_config, measurement_config],
    )
    state_transform_and_filter = Node(
        package="measurement",
        executable="state_transform_and_filter",
        name="state_transform_and_filter",
        output="screen",
        parameters=[common_config, measurement_config],
    )
    transform_cloud = Node(
        package="measurement",
        executable="transform_cloud",
        name="transform_cloud",
        output="screen",
        parameters=[common_config, transform_cloud_config],
    )
    observer = Node(
        package="measurement",
        executable="observer.py",
        name="observer",
        output="screen",
        parameters=[common_config, measurement_config],
    )
    imu_to_clock = Node(
        package="measurement",
        executable="imu_to_clock",
        name="imu_to_clock",
        output="screen",
        parameters=[common_config],
    )

    return LaunchDescription(
        [
            repub_body_imu,
            state_transform_and_filter,
            transform_cloud,
            observer,
            imu_to_clock,
        ]
    )
