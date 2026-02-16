#!/usr/bin/env python3

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".01"

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import tf_transformations as tf
from ament_index_python.packages import get_package_share_directory

from tyswy.control.estimators.go2_kalman_filter import Go2KalmanFilter
from tyswy.control.estimators.go2_kalman_filter_velocity import Go2KalmanFilterVelocity
from tyswy.utils.loop_timer import LoopTimer
import numpy as np
import threading
import yaml

from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from unitree_go.msg import LowState


DEFAULT_JOINT_POS_UNITREE = np.array([0.0, 0.67, -1.3] * 4, dtype=np.float32)
DEFAULT_JOINT_POS_ISAAC = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5], dtype=np.float32
)
DEFAULT_JOINT_POS = DEFAULT_JOINT_POS_ISAAC


class ObserverNode(Node):
    def __init__(self):
        super().__init__("observer_node")

        self.g_world = np.array([0, 0, -9.81])

        self.declare_parameter(
            "topics.state_transform_and_filtered", "/state_transform_and_filtered"
        )
        self.declare_parameter("topics.lowstate", "/lowstate")
        self.declare_parameter("topics.imu_body", "/imu_body")
        self.declare_parameter("topics.imu_body_nograv", "/imu_body_nograv")
        self.declare_parameter("topics.observation", "/observation")
        self.declare_parameter("topics.observation_raw", "/observation_raw")
        self.declare_parameter("observer.use_imu_lin_acc", False)
        self.declare_parameter("topics.vel_kf_stds", "/vel_kf_stds")
        self.declare_parameter("topics.vel_kf_imu_bias", "/vel_kf_imu_bias")

        self.declare_parameter("topics.robot_state", "/robot_state")
        self.declare_parameter("topics.elevation_vec", "/elevation_vec")
        self.declare_parameter("topics.joint_pos_cmd", "/joint_pos_cmd")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")

        self.declare_parameter("observer.use_kf", False)
        self.declare_parameter("observer.use_kf_velocity", "auto")  # CHANGE
        self.declare_parameter("observer.action_scale", 0.25)
        self.declare_parameter("observer.height_scan_offset", 0.3)
        self.declare_parameter("observer.phase_period", 0.5)
        self.declare_parameter("observer.rate", 50.0)
        self.declare_parameter("observer.base_lin_vel_source", "odom")
        self.declare_parameter("observer.base_ang_vel_source", "odom")
        self.declare_parameter("observer.proj_grav_source", "odom")

        self.state_topic = (
            self.get_parameter("topics.state_transform_and_filtered")
            .get_parameter_value()
            .string_value
        )
        self.lowstate_topic = (
            self.get_parameter("topics.lowstate").get_parameter_value().string_value
        )
        self.imu_body_topic = (
            self.get_parameter("topics.imu_body").get_parameter_value().string_value
        )
        self.imu_body_nograv_topic = (
            self.get_parameter("topics.imu_body_nograv")
            .get_parameter_value()
            .string_value
        )
        self.observation_topic = (
            self.get_parameter("topics.observation").get_parameter_value().string_value
        )
        self.observation_raw_topic = (
            self.get_parameter("topics.observation_raw")
            .get_parameter_value()
            .string_value
        )
        self.use_imu_lin_acc = (
            self.get_parameter("observer.use_imu_lin_acc")
            .get_parameter_value()
            .bool_value
        )
        self.robot_state_topic = (
            self.get_parameter("topics.robot_state").get_parameter_value().string_value
        )
        self.elevation_topic = (
            self.get_parameter("topics.elevation_vec")
            .get_parameter_value()
            .string_value
        )
        self.joint_pos_cmd_topic = (
            self.get_parameter("topics.joint_pos_cmd")
            .get_parameter_value()
            .string_value
        )
        self.cmd_vel_topic = (
            self.get_parameter("topics.cmd_vel").get_parameter_value().string_value
        )
        self.vel_kf_stds_topic = (
            self.get_parameter("topics.vel_kf_stds").get_parameter_value().string_value
        )
        self.vel_kf_imu_bias_topic = (
            self.get_parameter("topics.vel_kf_imu_bias")
            .get_parameter_value()
            .string_value
        )

        self.use_kf = (
            self.get_parameter("observer.use_kf").get_parameter_value().bool_value
        )
        use_kf_velocity = (
            self.get_parameter("observer.use_kf_velocity")
            .get_parameter_value()
            .string_value
        )
        assert use_kf_velocity in ["auto", "true", "false"]
        if use_kf_velocity == "auto":
            no_mocap = os.environ.get("NO_MOCAP").lower() == "true"
            self.use_kf_velocity = no_mocap
        else:
            self.use_kf_velocity = use_kf_velocity == "true"
        self.action_scale = (
            self.get_parameter("observer.action_scale")
            .get_parameter_value()
            .double_value
        )
        self.height_scan_offset = (
            self.get_parameter("observer.height_scan_offset")
            .get_parameter_value()
            .double_value
        )
        self.phase_period = (
            self.get_parameter("observer.phase_period")
            .get_parameter_value()
            .double_value
        )
        self.rate = (
            self.get_parameter("observer.rate").get_parameter_value().double_value
        )
        self.dt = 1.0 / self.rate
        self.base_lin_vel_source = (
            self.get_parameter("observer.base_lin_vel_source")
            .get_parameter_value()
            .string_value
        )
        self.base_ang_vel_source = (
            self.get_parameter("observer.base_ang_vel_source")
            .get_parameter_value()
            .string_value
        )
        self.proj_grav_source = (
            self.get_parameter("observer.proj_grav_source")
            .get_parameter_value()
            .string_value
        )

        assert self.base_lin_vel_source in ["odom"]
        assert self.base_ang_vel_source in ["odom", "imu"]
        assert self.proj_grav_source in ["odom", "imu"]

        self.latest_base_lin_vel = None
        self.latest_base_lin_acc = None
        self.latest_base_lin_acc_nograv = None
        self.latest_base_ang_vel = None
        self.latest_proj_grav = None
        self.latest_joint_pos = None
        self.latest_joint_vel = None
        self.latest_elevation_vec = None
        self.latest_cmd_vel = None
        self.latest_action = DEFAULT_JOINT_POS
        self.robot_state = None
        self.obs = None
        self.step_count = 0

        self.kf = None
        if self.use_kf:
            config_dir = get_package_share_directory("config")
            yaml_path = os.path.join(config_dir, "config", "kalman_filter.yaml")
            with open(yaml_path, "r") as file:
                kf_config_dict = yaml.safe_load(file)
            self.kf = Go2KalmanFilter(kf_config_dict)

        self.kfv = None
        if self.use_kf_velocity:
            config_dir = get_package_share_directory("config")
            yaml_path = os.path.join(
                config_dir, "config", "kalman_filter_velocity.yaml"
            )
            with open(yaml_path, "r") as file:
                kf_config_dict = yaml.safe_load(file)
            self.kfv = Go2KalmanFilterVelocity(kf_config_dict)

        self.low_state_lock = threading.Lock()
        self.imu_body_lock = threading.Lock()
        self.imu_body_nograv_lock = threading.Lock()
        self.odom_lock = threading.Lock()
        self.elev_lock = threading.Lock()
        self.act_lock = threading.Lock()
        self.cmd_lock = threading.Lock()

        self.low_state_sub_group = MutuallyExclusiveCallbackGroup()
        self.imu_body_sub_group = MutuallyExclusiveCallbackGroup()
        self.imu_body_nograv_sub_group = MutuallyExclusiveCallbackGroup()
        self.odom_sub_group = MutuallyExclusiveCallbackGroup()
        self.observation_group = MutuallyExclusiveCallbackGroup()
        self.robot_state_sub_group = MutuallyExclusiveCallbackGroup()
        self.elevation_sub_group = MutuallyExclusiveCallbackGroup()
        self.joint_pos_cmd_sub_group = MutuallyExclusiveCallbackGroup()
        self.cmd_vel_sub_group = MutuallyExclusiveCallbackGroup()

        self.lowstate_sub = self.create_subscription(
            LowState,
            self.lowstate_topic,
            self.lowstate_callback,
            1,
            callback_group=self.low_state_sub_group,
        )
        self.imu_body_sub = self.create_subscription(
            Imu,
            self.imu_body_topic,
            self.imu_body_callback,
            1,
            callback_group=self.imu_body_sub_group,
        )
        self.imu_body_nograv_sub = self.create_subscription(
            Imu,
            self.imu_body_nograv_topic,
            self.imu_body_nograv_callback,
            1,
            callback_group=self.imu_body_nograv_sub_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.state_topic,
            self.odom_callback,
            1,
            callback_group=self.odom_sub_group,
        )
        self.robot_state_sub = self.create_subscription(
            String,
            self.robot_state_topic,
            self.robot_state_callback,
            1,
            callback_group=self.robot_state_sub_group,
        )
        self.elevation_sub = self.create_subscription(
            Float32MultiArray,
            self.elevation_topic,
            self.elevation_callback,
            1,
            callback_group=self.elevation_sub_group,
        )
        self.joint_pos_cmd_sub = self.create_subscription(
            Float32MultiArray,
            self.joint_pos_cmd_topic,
            self.joint_pos_cmd_callback,
            1,
            callback_group=self.joint_pos_cmd_sub_group,
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            1,
            callback_group=self.cmd_vel_sub_group,
        )

        self.observation_pub = self.create_publisher(
            Float32MultiArray,
            self.observation_topic,
            100,
            callback_group=self.observation_group,
        )
        self.observation_raw_pub = self.create_publisher(
            Float32MultiArray,
            self.observation_raw_topic,
            100,
            callback_group=self.observation_group,
        )
        self.vel_kf_stds_pub = self.create_publisher(
            Float32MultiArray,
            self.vel_kf_stds_topic,
            100,
            callback_group=self.observation_group,
        )
        self.vel_kf_imu_bias_pub = self.create_publisher(
            Float32MultiArray,
            self.vel_kf_imu_bias_topic,
            100,
            callback_group=self.observation_group,
        )

        self.obs_timer = LoopTimer(
            print_interval_sec=1.0,
            node=self,
            name="Observation Timer",
            window_size=50,
            print_info=False,
            warn_rate=0.98 * self.rate,
        )
        self.create_timer(
            1 / self.rate,
            self.publish_observation,
            callback_group=self.observation_group,
        )

    def robot_state_callback(self, msg: String):
        self.robot_state = msg.data
        if self.robot_state == "WALKING" or self.robot_state == "STAND":
            if self.use_kf:
                self.kf.reset(
                    self.obs,
                    self.latest_elevation_vec,
                )
            if self.use_kf_velocity:
                self.kfv.reset(
                    self._get_obs_no_lin_vel(self.obs),
                    self.latest_elevation_vec,
                )

    def elevation_callback(self, msg: Float32MultiArray):
        with self.elev_lock:
            vec = np.array(msg.data)
            # to match Isaac Lab height scan (z-down and offset)
            self.latest_elevation_vec = -vec - self.height_scan_offset

    def joint_pos_cmd_callback(self, msg: Float32MultiArray):
        with self.act_lock:
            joint_pos_cmd = np.array(msg.data)
            self.latest_action = (joint_pos_cmd - DEFAULT_JOINT_POS) / self.action_scale

    def cmd_vel_callback(self, msg: Twist):
        with self.cmd_lock:
            self.latest_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def odom_callback(self, msg: Odometry):
        with self.odom_lock:
            if self.base_lin_vel_source == "odom":
                self.latest_base_lin_vel = np.array(
                    [
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                    ]
                )

            if self.base_ang_vel_source == "odom":
                self.latest_base_ang_vel = np.array(
                    [
                        msg.twist.twist.angular.x,
                        msg.twist.twist.angular.y,
                        msg.twist.twist.angular.z,
                    ]
                )

            if self.proj_grav_source == "odom":
                q = [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
                R = tf.quaternion_matrix(q)[:3, :3]
                proj_grav_unnorm = R.T @ self.g_world
                self.latest_proj_grav = np.array(
                    proj_grav_unnorm / np.linalg.norm(proj_grav_unnorm)
                )

    def lowstate_callback(self, msg: LowState):
        with self.low_state_lock:
            self.latest_joint_pos = (
                np.array([msg.motor_state[i].q for i in range(12)]) - DEFAULT_JOINT_POS
            )
            self.latest_joint_vel = np.array([msg.motor_state[i].dq for i in range(12)])

    def imu_body_callback(self, msg: Imu):
        with self.imu_body_lock:
            self.latest_base_lin_acc = np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            )

            if self.base_ang_vel_source == "imu":
                self.latest_base_ang_vel = np.array(
                    [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ]
                )

            if self.proj_grav_source == "imu":
                q = [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ]
                R = tf.quaternion_matrix(q)[:3, :3]
                proj_grav_unnorm = R.T @ self.g_world
                self.latest_proj_grav = np.array(
                    proj_grav_unnorm / np.linalg.norm(proj_grav_unnorm)
                )

    def imu_body_nograv_callback(self, msg: Imu):
        if self.kfv is None:
            return
        with self.imu_body_nograv_lock:
            if self.robot_state == "WALKING":
                imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                imu = np.array(
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    ]
                )
                self.kfv.propagate(imu, imu_time)

    def publish_observation(self):
        self.obs_timer.tick()

        if (
            self.latest_base_lin_vel is None
            or self.latest_base_lin_acc is None
            or self.latest_base_ang_vel is None
            or self.latest_proj_grav is None
            or self.latest_joint_pos is None
            or self.latest_joint_vel is None
            or self.latest_elevation_vec is None
        ):
            return

        phase = self.step_count * self.dt * 2.0 * np.pi / self.phase_period
        cos_sin_phase = np.array([np.cos(phase), np.sin(phase)])

        with self.low_state_lock, self.imu_body_lock, self.odom_lock:
            if self.use_imu_lin_acc:
                self.obs = np.concatenate(
                    [
                        self.latest_base_lin_vel,
                        self.latest_base_lin_acc,
                        self.latest_base_ang_vel,
                        self.latest_proj_grav,
                        self.latest_joint_pos,
                        self.latest_joint_vel,
                        cos_sin_phase,
                    ]
                )
            else:
                self.obs = np.concatenate(
                    [
                        self.latest_base_lin_vel,
                        self.latest_base_ang_vel,
                        self.latest_proj_grav,
                        self.latest_joint_pos,
                        self.latest_joint_vel,
                        cos_sin_phase,
                    ]
                )

        # publish raw observation
        msg = Float32MultiArray()
        msg.data = self.obs.tolist()
        self.observation_raw_pub.publish(msg)

        # KF estimation
        if self.use_kf:
            if not self.kf.initialized:
                self.get_logger().warn("Initializing Kalman filter...")
                with self.elev_lock:
                    self.kf.initialize(
                        self.obs,
                        self.latest_elevation_vec,
                    )
                self.get_logger().warn("Kalman filter initialized.")

            if self.robot_state == "WALKING":
                self.obs = self.kf.estimate(self.obs)
                self.obs[-2:] = cos_sin_phase  # don't estimate phase

        # KF estimation (velocity only)
        if self.use_kf_velocity:
            if not self.kfv.initialized:
                self.get_logger().warn("Initializing Kalman filter (velocity)...")
                with self.elev_lock:
                    self.kfv.initialize(
                        self._get_obs_no_lin_vel(self.obs),
                        self.latest_elevation_vec,
                    )
                self.get_logger().warn("Kalman filter (velocity) initialized.")
            if self.robot_state == "WALKING":
                with self.elev_lock, self.act_lock, self.cmd_lock:
                    est = self.kfv.estimate(
                        self._get_obs_no_lin_vel(self.obs),
                        self.latest_elevation_vec,
                        self.latest_cmd_vel,
                        self.latest_action,
                    )
                self.obs[:3] = est["v_hat"]

                stds = est["stds"][0:3]
                msg = Float32MultiArray()
                msg.data = stds.tolist()
                self.vel_kf_stds_pub.publish(msg)

                if est["bias"] is not None:
                    msg = Float32MultiArray()
                    msg.data = est["bias"].tolist()
                    self.vel_kf_imu_bias_pub.publish(msg)

        # publish filtered observation
        msg = Float32MultiArray()
        msg.data = self.obs.tolist()
        self.observation_pub.publish(msg)

        # propagate here
        if self.use_kf:
            with self.elev_lock, self.act_lock, self.cmd_lock:
                if self.robot_state == "WALKING":
                    self.kf.propagate(
                        self.obs,
                        self.latest_elevation_vec,
                        self.latest_cmd_vel,
                        self.latest_action,
                    )

        self.step_count += 1

    def _get_obs_no_lin_vel(self, obs: np.ndarray):
        return obs[3:]


def main(args=None):
    rclpy.init(args=args)
    node = ObserverNode()

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
