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
from simdist.utils import paths, model as model_utils
from simdist_controller.episode_logger_hdf5 import HDF5EpisodeLogger as EpisodeLogger
from utils.loop_timer import LoopTimer


DEFAULT_JOINT_POS_UNITREE = np.array([0.0, 0.67, -1.3] * 4, dtype=np.float32)
DEFAULT_JOINT_POS_ISAAC = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5], dtype=np.float32
)
DEFAULT_JOINT_POS = DEFAULT_JOINT_POS_ISAAC


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
        self.control_group = MutuallyExclusiveCallbackGroup()

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

        # Timers
        dt = 1.0 / self.rate
        self.loop_timer = LoopTimer(
            print_interval_sec=0.5,
            node=self,
            name="Update Timer",
            window_size=50,
            print_info=False,
            warn_rate=0.98 * self.rate,
        )
        self.loop_timer.disable_print()
        self.create_timer(dt, self.run_control, callback_group=self.control_group)

    def run_control(self):
        self.loop_timer.tick()

        if self.controller is None:
            return

        if self.controller.is_creating:
            return

        if (
            self.latest_observation is None
            or self.latest_elevation_vec is None
            or self.latest_cmd_vel is None
        ):
            return

        if not self.controller.is_initialized:
            # initialize the controller
            self.get_logger().warn("Initializing controller...")
            self.controller.initialize(
                self._make_ctrl_input(
                    self.latest_observation,
                    self.latest_elevation_vec,
                    self.last_action,
                ),
                self.latest_cmd_vel,
            )
            self.get_logger().warn("Controller initialized.")
            return

        if self.robot_state not in ["STAND", "WALKING"]:
            return

        with self.obs_lock:
            obs = self.latest_observation
        with self.elev_lock:
            elev = self.latest_elevation_vec
        with self.cmd_lock:
            cmd = self.latest_cmd_vel

        if self.robot_state == "STAND":
            cmd = np.zeros((3,))

        self.controller.update(
            self._make_ctrl_input(
                obs,
                elev,
                self.last_action,
            )
        )

        # get action
        if self.robot_state == "WALKING":
            ctrl_out = self.controller.run_control()
            action = ctrl_out["actions"][0]  # First action in sequence
            if not self.first_action_taken:
                self.first_action_taken = True
        else:
            action = np.zeros((12,))

        self.publish_action(action)

        if self.robot_state == "WALKING":
            self.logger.write(obs, elev, action, cmd)

        self.last_action = action

    def publish_action(self, action: np.ndarray):
        joint_pos_cmd: np.ndarray = self.action_scale * action + DEFAULT_JOINT_POS
        action_msg = Float32MultiArray()
        action_msg.data = joint_pos_cmd.tolist()
        self.action_pub.publish(action_msg)

    def robot_state_callback(self, msg: String):
        self.robot_state = msg.data
        with self.log_lock:
            # start logging when the robot starts walking
            if self.robot_state == "WALKING" and self.controller.is_initialized:
                self.first_action_taken = False
                self.logger.open()
                self.loop_timer.enable_print()
            else:
                self.logger.close()
                self.loop_timer.disable_print()

    def observation_callback(self, msg: Float32MultiArray):
        with self.obs_lock:
            self.latest_observation = np.array(msg.data)

    def elevation_callback(self, msg: Float32MultiArray):
        with self.elev_lock:
            vec = np.array(msg.data)
            # to match Isaac Lab height scan (z-down and offset)
            self.latest_elevation_vec = -vec - self.height_scan_offset

    def cmd_vel_callback(self, msg: Twist):
        with self.cmd_lock:
            self.latest_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
            if self.controller is None:
                return
            if self.controller.is_creating:
                return
            if self.controller.is_initialized:
                self.controller.set_fut_cmd(self.latest_cmd_vel)

    def _make_ctrl_input(
        self, proprio_obs: np.ndarray, extero_obs: np.ndarray, prev_action: np.ndarray
    ) -> ControllerInput:
        return {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "prev_action": prev_action,
        }


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
