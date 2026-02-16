import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_dir = get_package_share_directory("config")
    common_config = os.path.join(config_dir, "config", "common.yaml")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rvizGA",
        output="screen",
        arguments=[
            "-d",
            os.path.join(get_package_share_directory("bringup"), "rviz", "rviz.rviz"),
        ],
        prefix="nice",
        parameters=[common_config],
    )

    return LaunchDescription(
        [
            rviz_node,
        ]
    )
