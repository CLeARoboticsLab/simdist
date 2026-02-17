import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
robot_desc_dir = get_package_share_directory("go2_description")
urdf = os.path.join(robot_desc_dir, "urdf", "go2_description.urdf.xacro")


def generate_launch_description():
    joint_state_pub = Node(
        package="visualization",
        executable="joint_state_pub.py",
        name="joint_state_pub",
        output="screen",
        parameters=[common_config],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[common_config, {"robot_description": Command(["xacro ", urdf])}],
        output="screen",
    )

    return LaunchDescription(
        [
            joint_state_pub,
            robot_state_publisher,
        ]
    )
