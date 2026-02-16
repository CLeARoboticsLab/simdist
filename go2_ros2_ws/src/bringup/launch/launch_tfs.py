from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    static_transform_publisher_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="loamInterfaceTransPubMap",
        arguments=["0", "0", "0", "0", "0", "0", "map", "camera_init"],
    )

    static_transform_publisher_body = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="loamInterfaceTransPubVehicle",
        arguments=["0", "0", "0", "0", "0", "0", "aft_mapped", "body"],
    )

    static_transform_publisher_base_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_static_tf_pub",
        arguments=["0", "0", "0", "0", "0", "0", "body", "odom"],
    )

    static_transform_publisher_body_corrected = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_static_tf_pub",
        arguments=["0", "0", "0.15", "0", "0", "0", "body", "body_corrected"],
    )

    static_transform_publisher_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_static_tf_pub",
        arguments=[
            "0.3",
            "0",
            "0",
            "0",
            "2.87820258505555555556",
            "0",
            "body",
            "lidar",
        ],
    )

    return LaunchDescription(
        [
            static_transform_publisher_map,
            static_transform_publisher_body,
            static_transform_publisher_base_link,
            static_transform_publisher_body_corrected,
            static_transform_publisher_lidar,
        ]
    )
