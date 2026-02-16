#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
import numpy as np
from tf_transformations import euler_from_quaternion


class PoseKfNode(Node):
    def __init__(self):
        super().__init__("pose_kf_node")

        self.vel_noise = 0.05
        self.pos_noise = 0.05
        self.init_P = 0.05**2

        self.pos_est = np.zeros((2,))
        self.P = np.ones_like(self.pos_est) * self.init_P
        self.latest_pos = None
        self.latest_vel_body = None
        self.latest_yaw = None

        self.obs_callback_group = MutuallyExclusiveCallbackGroup()
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            "/observation",
            self.obs_callback,
            1,
            callback_group=self.obs_callback_group,
        )

        self.odom_sub_group = MutuallyExclusiveCallbackGroup()
        self.odom_sub = self.create_subscription(
            Odometry,
            "/state_transform_and_filtered",
            self.odom_callback,
            1,
            callback_group=self.odom_sub_group,
        )

        self.odom_pub = self.create_publisher(Odometry, "/odom_kf", 1)

        self.timer = self.create_timer(1.0 / 50.0, self.update)  # 50Hz

    def obs_callback(self, msg: Float32MultiArray):
        self.latest_vel_body = np.array(msg.data[0:3])

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.latest_pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ]
        )
        orientation_q = msg.pose.pose.orientation
        quaternion = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        _, _, self.latest_yaw = euler_from_quaternion(quaternion)

    def update(self):
        if self.latest_pos is None or self.latest_vel_body is None:
            return

        # propagate
        dt = 1.0 / 50.0
        dx = self.latest_vel_body[0] * dt
        dy = self.latest_vel_body[1] * dt
        self.pos_est[0] += dx * np.cos(self.latest_yaw) - dy * np.sin(self.latest_yaw)
        self.pos_est[1] += dx * np.sin(self.latest_yaw) + dy * np.cos(self.latest_yaw)
        self.P += dt**2 * self.vel_noise

        # measurement update
        z = self.latest_pos
        R = self.pos_noise**2
        K = self.P / (self.P + R)
        self.pos_est += K * (z - self.pos_est)
        self.P = (1.0 - K) * self.P

        msg = self.latest_odom
        msg.pose.pose.position.x = self.pos_est[0]
        msg.pose.pose.position.y = self.pos_est[1]
        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseKfNode()

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
