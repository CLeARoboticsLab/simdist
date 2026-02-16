import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Configs
config_dir = get_package_share_directory("config")
common_config = os.path.join(config_dir, "config", "common.yaml")
control_config = os.path.join(config_dir, "config", "control.yaml")


def generate_launch_description():
    use_sim = os.environ.get("USE_SIM", "false").lower() == "true"

    prefix = "gnome-terminal --" if use_sim else ""

    state_machine = Node(
        package="control",
        executable="state_machine",
        parameters=[common_config, control_config],
        prefix=prefix,
    )

    return LaunchDescription(
        [
            state_machine,
        ]
    )
