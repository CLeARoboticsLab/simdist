import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_dir = get_package_share_directory("config")
    common_config = os.path.join(
        config_dir,
        "config",
        "common.yaml",
    )
    lio_config = os.path.join(
        config_dir,
        "config",
        "lio.yaml",
    )

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_files",
                default_value="",
                description="List of parameter YAML files separated by semicolons",
            ),
            launch_ros.actions.Node(
                package="point_lio_unilidar",
                executable="pointlio_mapping",
                name="laserMapping",
                output="screen",
                parameters=[common_config, lio_config],
                remappings=[
                    ("/cloud_registered", "/registered_scan"),
                    ("/aft_mapped_to_init", "/state_estimation"),
                ],
            ),
        ]
    )
