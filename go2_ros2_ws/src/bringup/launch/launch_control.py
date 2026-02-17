import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
control_config = os.path.join(config_dir, "config", "control.yaml")


def generate_launch_description():
    cmd_vel_pub = Node(
        package="control",
        executable="cmd_vel_pub",
        name="cmd_vel_pub",
        output="screen",
        parameters=[common_config, control_config],
    )

    controller = Node(
        package="control",
        executable="simdist_controller_node.py",
        name="simdist_controller_node",
        output="screen",
        parameters=[common_config],
    )

    return LaunchDescription(
        [
            cmd_vel_pub,
            controller,
        ]
    )
