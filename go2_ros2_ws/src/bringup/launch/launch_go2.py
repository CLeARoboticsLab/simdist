import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # launch files
    tf_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bringup"), "launch", "launch_tfs.py"
            )
        )
    )
    if os.environ.get("MOCAP").lower() == "true":
        lio_launch = IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bringup"),
                    "launch",
                    "launch_mocap_odom.py",
                )
            )
        )
    else:
        lio_launch = IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bringup"), "launch", "launch_lio.py"
                )
            )
        )

    elevation_mapping_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bringup"),
                "launch",
                "launch_elevation_mapping.py",
            )
        )
    )
    measurement_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bringup"),
                "launch",
                "launch_measurement.py",
            )
        )
    )
    robot_state_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("bringup"),
                "launch",
                "launch_robot_state.py",
            )
        )
    )

    return LaunchDescription(
        [
            tf_launch,
            lio_launch,
            elevation_mapping_launch,
            measurement_launch,
            # robot_state_launch,
        ]
    )
