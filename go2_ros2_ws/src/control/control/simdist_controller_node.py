#!/usr/bin/env python3
import os
import threading

import numpy as np
import yaml

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist

from simdist.control.controller_base import ControllerInput
from simdist.control.mppi import MppiController
from simdist.utils import paths, model as model_utils, config
from control.episode_logger_hdf5 import HDF5EpisodeLogger as EpisodeLogger


class SimdistControllerNode(Node):
    def __init__(self):
        super().__init__("simdist_controller_node")

        # Load parameters
        self.declare_parameter("topics.robot_state", "/robot_state")
        self.declare_parameter("topics.observation", "/observation")
        self.declare_parameter("topics.elevation_vec", "/elevation_vec")
        self.declare_parameter("topics.joint_pos_cmd", "/joint_pos_cmd")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")
        self.robot_state_topic = (
            self.get_parameter("topics.robot_state").get_parameter_value().string_value
        )
        self.observation_topic = (
            self.get_parameter("topics.observation").get_parameter_value().string_value
        )
        self.elevation_topic = (
            self.get_parameter("topics.elevation_vec")
            .get_parameter_value()
            .string_value
        )
        self.output_topic = (
            self.get_parameter("topics.joint_pos_cmd")
            .get_parameter_value()
            .string_value
        )
        self.cmd_vel_topic = (
            self.get_parameter("topics.cmd_vel").get_parameter_value().string_value
        )

        # Load config
        config_dir = get_package_share_directory("config")
        yaml_path = os.path.join(config_dir, "config", "simdist_controller.yaml")
        with open(yaml_path, "r") as file:
            cfg = yaml.safe_load(file)
            self.get_logger().info(f"Config: {cfg}")
        self.height_scan_offset = cfg["height_scan_offset"]
        self.action_scale = cfg["action_scale"]
        self.rate = cfg["rate"]

        # Create controller
        mppi_cfg = cfg["mppi"]
        ckpt = cfg["model"]["checkpoint"]
        if ckpt.startswith("/"):
            ckpt_dir = ckpt
        else:
            ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), ckpt)
        self.get_logger().info(f"Checkpoint directory: {ckpt_dir}")
        if not os.path.exists(ckpt_dir):
            self.get_logger().error(f"Checkpoint directory does not exist: {ckpt_dir}")
            return
        model, model_cfg, _ = model_utils.load_model_from_ckpt(
            ckpt_dir, cfg["model"]["step"]
        )
        self.controller = MppiController(model, model_cfg, mppi_cfg)

        # Data storage
        self.robot_state = None
        self.latest_observation = None
        self.latest_elevation_vec = None
        self.latest_cmd_vel = None
        self.last_action = np.zeros((12,))
        self.first_action_taken = False
        self.logger = EpisodeLogger(cfg["logging"], self)

        # locks
        self.obs_lock = threading.Lock()
        self.elev_lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self.log_lock = threading.Lock()

        # create callback groups to ensure true multi-threading
        self.robot_state_sub_group = MutuallyExclusiveCallbackGroup()
        self.observation_sub_group = MutuallyExclusiveCallbackGroup()
        self.elevation_sub_group = MutuallyExclusiveCallbackGroup()
        self.cmd_vel_sub_group = MutuallyExclusiveCallbackGroup()
        self.publisher_group = MutuallyExclusiveCallbackGroup()
        self.update_group = MutuallyExclusiveCallbackGroup()

        # Subscribers
        self.robot_state_sub = self.create_subscription(
            String,
            self.robot_state_topic,
            self.robot_state_callback,
            1,
            callback_group=self.robot_state_sub_group,
        )
        self.observation_sub = self.create_subscription(
            Float32MultiArray,
            self.observation_topic,
            self.observation_callback,
            1,
            callback_group=self.observation_sub_group,
        )
        self.elevation_sub = self.create_subscription(
            Float32MultiArray,
            self.elevation_topic,
            self.elevation_callback,
            1,
            callback_group=self.elevation_sub_group,
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            1,
            callback_group=self.cmd_vel_sub_group,
        )

        # Publishers
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            100,
            callback_group=self.publisher_group,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimdistControllerNode()

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
