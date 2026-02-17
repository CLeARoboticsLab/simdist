#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from unitree_go.msg import LowState


JOINT_NAMES = [
    "RF_hip_joint",
    "RF_upper_leg_joint",
    "RF_lower_leg_joint",
    "LF_hip_joint",
    "LF_upper_leg_joint",
    "LF_lower_leg_joint",
    "RR_hip_joint",
    "RR_upper_leg_joint",
    "RR_lower_leg_joint",
    "LR_hip_joint",
    "LR_upper_leg_joint",
    "LR_lower_leg_joint",
]


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__("joint_state_pub")
        self.load_params()
        self.latest_joint_pos = None

        self.js_pub = self.create_publisher(JointState, self.joint_states_topic, 10)
        self.low_state_sub_group = MutuallyExclusiveCallbackGroup()
        self.lowstate_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.lowstate_callback,
            1,
            callback_group=self.low_state_sub_group,
        )

        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

    def load_params(self):
        self.declare_parameter("topics.lowstate", "/lowstate")
        self.lowstate_topic = (
            self.get_parameter("topics.lowstate").get_parameter_value().string_value
        )
        self.declare_parameter("topics.joint_states", "/joint_states")
        self.joint_states_topic = (
            self.get_parameter("topics.joint_states").get_parameter_value().string_value
        )

    def lowstate_callback(self, msg: LowState):
        self.latest_joint_pos = [msg.motor_state[i].q for i in range(12)]

    def tick(self):
        if self.latest_joint_pos is None:
            return

        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = self.latest_joint_pos
        msg.header.stamp = self.get_clock().now().to_msg()
        self.js_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

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
