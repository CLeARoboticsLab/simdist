#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import tf_transformations as tf
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster
import numpy as np
import threading

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped


class MocapNode(Node):
    def __init__(self):
        super().__init__("mocap_node")

        self.g_world = np.array([0, 0, -9.81])
        self.rate = 100.0  # Hz TODO

        self.declare_parameter("topics.state", "/state_estimation")
        self.declare_parameter("topics.mocap_pose", "/mocap_pose")
        self.declare_parameter("topics.mocap_twist", "/mocap_twist")

        self.state_topic = (
            self.get_parameter("topics.state").get_parameter_value().string_value
        )
        self.mocap_pose_topic = (
            self.get_parameter("topics.mocap_pose").get_parameter_value().string_value
        )
        self.mocap_twist_topic = (
            self.get_parameter("topics.mocap_twist").get_parameter_value().string_value
        )

        self.mocap_rot = None
        self.base_lin_vel_mocap = None
        self.base_ang_vel_mocap = None

        self.need_to_init_mocap = True
        self.mocap_init_pos = None
        self.mocap_init_orient_inv = None
        self.latest_mocap_position = None
        self.latest_mocap_orientation = None

        self.mocap_lock = threading.Lock()
        self.tf_broadcaster = TransformBroadcaster(self)

        qos_profile = QoSProfile(depth=1)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        self.mocap_pose_sub_group = MutuallyExclusiveCallbackGroup()
        self.mocap_pose_sub = self.create_subscription(
            PoseStamped,
            self.mocap_pose_topic,
            self.mocap_pose_callback,
            qos_profile,
            callback_group=self.mocap_pose_sub_group,
        )
        self.mocap_twist_sub_group = MutuallyExclusiveCallbackGroup()
        self.mocap_twist_sub = self.create_subscription(
            TwistStamped,
            self.mocap_twist_topic,
            self.mocap_twist_callback,
            qos_profile,
            callback_group=self.mocap_twist_sub_group,
        )

        self.mocap_pub_group = MutuallyExclusiveCallbackGroup()
        self.mocap_odom_pub = self.create_publisher(
            Odometry,
            self.state_topic,
            100,
            callback_group=self.mocap_pub_group,
        )
        self.create_timer(
            1 / self.rate,
            self.publish_mocap,
            callback_group=self.mocap_pub_group,
        )

    def mocap_pose_callback(self, msg: PoseStamped):
        with self.mocap_lock:
            position = np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]
            )
            orientation = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )

            if self.need_to_init_mocap:
                self.mocap_init_pos = position
                self.mocap_init_orient_inv = tf.quaternion_inverse(orientation)
                self.need_to_init_mocap = False

            self.latest_mocap_position = position
            self.latest_mocap_orientation = orientation
            self.mocap_rot = tf.quaternion_matrix(orientation)[:3, :3]

    def mocap_twist_callback(self, msg: TwistStamped):
        if self.mocap_rot is None:
            return
        with self.mocap_lock:
            base_lin_vel_global = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
            )
            self.base_lin_vel_mocap = self.mocap_rot.T @ base_lin_vel_global

            base_ang_vel_global = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
            )
            self.base_ang_vel_mocap = self.mocap_rot.T @ base_ang_vel_global

    def publish_mocap(self):
        if (
            self.base_lin_vel_mocap is None
            or self.base_ang_vel_mocap is None
            or self.latest_mocap_position is None
            or self.latest_mocap_orientation is None
            or self.mocap_init_pos is None
            or self.need_to_init_mocap
        ):
            return

        with self.mocap_lock:
            # publish odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "camera_init"
            odom_msg.child_frame_id = "aft_mapped"
            delta_position = self.latest_mocap_position - self.mocap_init_pos
            R_body0_inv = tf.quaternion_matrix(self.mocap_init_orient_inv)[:3, :3]
            relative_position = R_body0_inv @ delta_position
            relative_orientation = tf.quaternion_multiply(
                self.mocap_init_orient_inv, self.latest_mocap_orientation
            )
            odom_msg.pose.pose.position.x = relative_position[0]
            odom_msg.pose.pose.position.y = relative_position[1]
            odom_msg.pose.pose.position.z = relative_position[2]
            odom_msg.pose.pose.orientation.x = relative_orientation[0]
            odom_msg.pose.pose.orientation.y = relative_orientation[1]
            odom_msg.pose.pose.orientation.z = relative_orientation[2]
            odom_msg.pose.pose.orientation.w = relative_orientation[3]
            odom_msg.twist.twist.linear.x = self.base_lin_vel_mocap[0]
            odom_msg.twist.twist.linear.y = self.base_lin_vel_mocap[1]
            odom_msg.twist.twist.linear.z = self.base_lin_vel_mocap[2]
            odom_msg.twist.twist.angular.x = self.base_ang_vel_mocap[0]
            odom_msg.twist.twist.angular.y = self.base_ang_vel_mocap[1]
            odom_msg.twist.twist.angular.z = self.base_ang_vel_mocap[2]
            self.mocap_odom_pub.publish(odom_msg)

            # Publish TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = "camera_init"
            tf_msg.child_frame_id = "aft_mapped"
            tf_msg.transform.translation.x = relative_position[0]
            tf_msg.transform.translation.y = relative_position[1]
            tf_msg.transform.translation.z = relative_position[2]
            tf_msg.transform.rotation.x = relative_orientation[0]
            tf_msg.transform.rotation.y = relative_orientation[1]
            tf_msg.transform.rotation.z = relative_orientation[2]
            tf_msg.transform.rotation.w = relative_orientation[3]
            self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MocapNode()

    # Use multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
