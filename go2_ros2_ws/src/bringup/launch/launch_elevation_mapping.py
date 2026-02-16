import os
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_dir = get_package_share_directory("config")
    common_config = os.path.join(config_dir, "config", "common.yaml")
    elevation_mapping_config = os.path.join(
        config_dir, "config", "elevation_mapping.yaml"
    )
    terrain_analysis_simple_config = os.path.join(
        config_dir, "config", "terrain_analysis_simple.yaml"
    )

    body_aligned_z_tf_pub = Node(
        package="elevation_mapping",
        executable="body_aligned_z_tf_pub",
        name="body_aligned_z_tf_pub",
        output="screen",
        parameters=[common_config],
    )

    world_tf_pub = Node(
        package="elevation_mapping",
        executable="world_tf_pub",
        name="world_tf_pub",
        output="screen",
        parameters=[common_config, elevation_mapping_config],
    )

    pointcloud_to_gridmap = Node(
        package="elevation_mapping",
        executable="pointcloud_to_gridmap",
        name="pointcloud_to_gridmap",
        output="screen",
        parameters=[common_config, elevation_mapping_config],
    )

    flatten_gridmap = Node(
        package="elevation_mapping",
        executable="flatten_gridmap",
        name="flatten_gridmap",
        output="screen",
        parameters=[common_config, elevation_mapping_config],
    )

    terrain_analysis = Node(
        package="elevation_mapping",
        executable="terrain_analysis_simple",
        output="screen",
        parameters=[common_config, terrain_analysis_simple_config],
    )

    return launch.LaunchDescription(
        [
            body_aligned_z_tf_pub,
            world_tf_pub,
            pointcloud_to_gridmap,
            flatten_gridmap,
            terrain_analysis,
        ]
    )
