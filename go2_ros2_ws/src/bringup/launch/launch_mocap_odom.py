import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
measurement_config = os.path.join(config_dir, "config", "measurement.yaml")


def generate_launch_description():
    mocap = Node(
        package="measurement",
        executable="mocap.py",
        name="mocap",
        output="screen",
        parameters=[common_config, measurement_config],
    )
    laser_mapping_simple = Node(
        package="measurement",
        executable="laser_mapping_simple",
        name="laser_mapping_simple",
        output="screen",
        parameters=[common_config],
    )
    path_viz = Node(
        package="measurement",
        executable="path_viz",
        name="path_viz",
        output="screen",
        parameters=[common_config],
    )

    return LaunchDescription(
        [
            mocap,
            laser_mapping_simple,
            path_viz,
        ]
    )
