from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from simdist.rl import go2_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCfg, patterns, ImuCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
import torch
from typing import Any
import math
import functools as ft

USE_FOOT_FORCE_OBS = True
USE_FOOT_HEIGHT_OBS = True
USE_BASE_MASS_OBS = True
USE_MATERIAL_PROPS_OBS = True
USE_JOINT_PROP_OBS = True
HISTORY_LENGTH = 1
WALKING_PERIOD = 0.5
SWING_HEIGHT = 0.09
BIAS = [0.0, 0.5, 0.5, 0.0]
R_GAIT = 0.5
HEIGHT_SCAN_OFFSET = 0.3
FOOT_RADIUS = 0.023
JOINT_STIFFNESS = 25.0
JOINT_STIFF_SCALE_DEV = 0.1
JOINT_DAMPING = 0.5
JOINT_DAMP_SCALE_DEV = 0.1
MAX_JOINT_FRICTION = 0.05
JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
UNITREE_DEFAULT_JOINT_ANGLES = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": 0.0,
    "F[L,R]_thigh_joint": 0.67,
    "R[L,R]_thigh_joint": 0.67,
    ".*_calf_joint": -1.3,
}
ISAAC_DEFAULT_JOINT_ANGLES = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}
FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


def _get_joint_params():
    # use Unitree's joint ordering
    asset_cfg = SceneEntityCfg(
        name="robot", joint_names=JOINT_NAMES, preserve_order=True
    )
    joint_params = {"asset_cfg": asset_cfg}
    return joint_params


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=go2_mdp.terrain_levels_cmd_vel_sim,
        params={"up_thresh": 0.75, "down_thresh": 0.25},
    )
    joint_stiffness_and_damping_difficulty = CurrTerm(
        func=go2_mdp.event_linear_difficulty,
        params={
            "term_name": "joint_stiffness_and_damping",
            "start_step": 0,
            "end_step": 5000 * 24,
        },
    )
    joint_friction_difficulty = CurrTerm(
        func=go2_mdp.event_linear_difficulty,
        params={
            "term_name": "joint_friction",
            "start_step": 0,
            "end_step": 5000 * 24,
        },
    )


@configclass
class CurriculumObsCfg(ObsGroup):
    """Observations for curriculum"""

    cum_cmd_vel_similarity = ObsTerm(func=go2_mdp.cum_cmd_vel_similarity)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class Go2EnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings
        self.sim.dt = 0.005
        self.decimation = 4  # 50 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.scene.num_envs = 4096

        # use custom curriculum
        self.curriculum = CurriculumCfg()
        self.curriculum.joint_stiffness_and_damping_difficulty = None
        self.curriculum.joint_friction_difficulty = None

        # add lots of terrain levels
        self.scene.terrain.terrain_generator.num_rows = 50
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 0

        # modifications to the height scanner to shift grid and make bigger
        self.scene.height_scanner.offset.pos = (0.5, 0.0, 20.0)
        self.scene.height_scanner.pattern_cfg.resolution = 0.1
        self.scene.height_scanner.pattern_cfg.size = [2.0, 1.4]

        # add an IMU sensor
        self.scene.imu_body = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)),
        )

        # add sensors for determining the height of the ground under the feet
        foot_height_scan = ft.partial(
            RayCasterCfg,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.fr_foot_scan = foot_height_scan(
            prim_path="{ENV_REGEX_NS}/Robot/FR_foot"
        )
        self.scene.fl_foot_scan = foot_height_scan(
            prim_path="{ENV_REGEX_NS}/Robot/FL_foot"
        )
        self.scene.rr_foot_scan = foot_height_scan(
            prim_path="{ENV_REGEX_NS}/Robot/RR_foot"
        )
        self.scene.rl_foot_scan = foot_height_scan(
            prim_path="{ENV_REGEX_NS}/Robot/RL_foot"
        )

        # modifications to the observations to use Unitree's joint ordering and modify
        # the height scan offset
        self.observations.policy.joint_pos.params = _get_joint_params()
        self.observations.policy.joint_vel.params = _get_joint_params()
        self.observations.policy.height_scan.params["offset"] = HEIGHT_SCAN_OFFSET

        # add a cosine/sine phase observation
        self.observations.policy.cos_sin_phase = ObsTerm(
            func=go2_mdp.cos_sin_phase, params={"period": WALKING_PERIOD}
        )

        # add observation history
        history_length = HISTORY_LENGTH
        self.observations.policy.base_lin_vel.history_length = history_length
        self.observations.policy.base_ang_vel.history_length = history_length
        self.observations.policy.projected_gravity.history_length = history_length
        self.observations.policy.velocity_commands.history_length = history_length
        self.observations.policy.joint_pos.history_length = history_length
        self.observations.policy.joint_vel.history_length = history_length
        self.observations.policy.actions.history_length = history_length
        self.observations.policy.cos_sin_phase.history_length = history_length

        # adjust noise
        self.observations.policy.base_lin_vel.noise = Gnoise(std=0.05)
        self.observations.policy.base_ang_vel.noise = Gnoise(std=0.1)
        self.observations.policy.projected_gravity.noise = Gnoise(std=0.025)
        self.observations.policy.joint_pos.noise = Gnoise(std=0.01)
        self.observations.policy.joint_vel.noise = Gnoise(std=0.75)
        self.observations.policy.height_scan.noise = Gnoise(std=0.02)

        # add privileged observations
        if USE_FOOT_FORCE_OBS:
            self.observations.policy.feet_body_forces = ObsTerm(
                func=mdp.body_incoming_wrench,
                scale=0.1,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=FOOT_NAMES, preserve_order=True
                    )
                },
            )
            self.observations.policy.feet_body_forces.history_length = history_length

        if USE_FOOT_HEIGHT_OBS:
            self.observations.policy.feet_height = ObsTerm(
                func=go2_mdp.feet_height,
                params={
                    "scan_sensor_names": [
                        "fr_foot_scan",
                        "fl_foot_scan",
                        "rr_foot_scan",
                        "rl_foot_scan",
                    ],
                    "foot_radius": FOOT_RADIUS,
                },
                clip=(-2.0, 2.0),
            )
            self.observations.policy.feet_height.history_length = history_length

        if USE_BASE_MASS_OBS:
            self.observations.policy.base_mass = ObsTerm(
                func=go2_mdp.mass,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names="base", preserve_order=True
                    )
                },
            )

        if USE_MATERIAL_PROPS_OBS:
            self.observations.policy.material_properties = ObsTerm(
                func=go2_mdp.material_properties
            )

        if USE_JOINT_PROP_OBS:
            self.observations.policy.joint_stiffness = ObsTerm(
                func=go2_mdp.joint_stiffness,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=JOINT_NAMES, preserve_order=True
                    ),
                    "scale": 1 / (JOINT_STIFF_SCALE_DEV * JOINT_STIFFNESS),
                    "bias": 1 - (1 + JOINT_STIFF_SCALE_DEV) / JOINT_STIFF_SCALE_DEV,
                },
            )
            self.observations.policy.joint_damping = ObsTerm(
                func=go2_mdp.joint_damping,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=JOINT_NAMES, preserve_order=True
                    ),
                    "scale": 1 / (JOINT_DAMP_SCALE_DEV * JOINT_DAMPING),
                    "bias": 1 - (1 + JOINT_DAMP_SCALE_DEV) / JOINT_DAMP_SCALE_DEV,
                },
            )
            self.observations.policy.joint_friction = ObsTerm(
                func=go2_mdp.joint_friction,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=JOINT_NAMES, preserve_order=True
                    ),
                    "scale": 2 / MAX_JOINT_FRICTION,
                    "bias": -1.0,
                },
            )

        self.observations.curriculum = CurriculumObsCfg()

        # modifications to the action space to use Unitree's joint ordering
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.preserve_order = True

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # use lots of varying commands
        self.commands.base_velocity = go2_mdp.VaryingVelocityCommandsCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )

        # modification to add friction randomization
        self.events.physics_material.func = go2_mdp.randomize_rigid_body_material_same
        self.events.physics_material.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=FOOT_NAMES, preserve_order=True
        )
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.3)
        self.events.physics_material.params["num_buckets"] = 4096
        self.events.physics_material.params["make_consistent"] = True

        # add some randomization to the starting joint angles
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)

        # add actuator randomization
        self.events.joint_stiffness_and_damping = EventTerm(
            func=go2_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
                "stiffness_distribution_params": (
                    1.0 - JOINT_STIFF_SCALE_DEV,
                    1.0 + JOINT_STIFF_SCALE_DEV,
                ),
                "damping_distribution_params": (
                    1.0 - JOINT_DAMP_SCALE_DEV,
                    1.0 + JOINT_DAMP_SCALE_DEV,
                ),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.joint_friction = EventTerm(
            func=go2_mdp.randomize_joint_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
                "friction_distribution_params": (0.0, MAX_JOINT_FRICTION),
                "operation": "add",
                "distribution": "uniform",
            },
        )

        # reward modifications
        self.rewards.feet_air_time = RewTerm(
            func=spot_mdp.rewards.air_time_reward,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=FOOT_NAMES, preserve_order=True
                ),
                "mode_time": 0.3,
                "velocity_threshold": 0.75,
            },
        )
        self.rewards.foot_slip_penalty = RewTerm(
            func=spot_mdp.foot_slip_penalty,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=FOOT_NAMES, preserve_order=True
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=FOOT_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=go2_mdp.foot_clearance_reward,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=FOOT_NAMES, preserve_order=True
                ),
                "scan_sensor_names": [
                    "fr_foot_scan",
                    "fl_foot_scan",
                    "rr_foot_scan",
                    "rl_foot_scan",
                ],
                "foot_radius": FOOT_RADIUS,
                "target_height": SWING_HEIGHT,
                "std": 0.05,
                "tanh_mult": 2.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -4.0
        joint_dev_weights = (1.0, 0.0, 0.0) * 4
        self.rewards.joint_deviation_l1 = RewTerm(
            func=go2_mdp.joint_deviation_l1_weighted,
            params={
                "weights": joint_dev_weights,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                ),
            },
            weight=-0.25,
        )
        self.rewards.gait = RewTerm(
            func=go2_mdp.gait_reward,
            weight=0.05,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=FOOT_NAMES, preserve_order=True
                ),
                "bias": BIAS,
                "contact_ratio": R_GAIT,
                "period": WALKING_PERIOD,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards.feet_height_gait = RewTerm(
            func=go2_mdp.feet_height_gait_reward,
            weight=0.2,
            params={
                "scan_sensor_names": [
                    "fr_foot_scan",
                    "fl_foot_scan",
                    "rr_foot_scan",
                    "rl_foot_scan",
                ],
                "foot_radius": FOOT_RADIUS,
                "bias": BIAS,
                "contact_ratio": R_GAIT,
                "period": WALKING_PERIOD,
                "swing_height": SWING_HEIGHT,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base",
            ".*_thigh",
            "Head.*",
        ]
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "limit_angle": 1.0472,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # modifications to the robot for default joint positions, ordering, and ordering
        go2_cfg = UNITREE_GO2_CFG
        init_state = go2_cfg.init_state
        init_state = init_state.replace(joint_pos=ISAAC_DEFAULT_JOINT_ANGLES)
        dc_motor_cfg = go2_cfg.actuators["base_legs"]
        dc_motor_cfg = dc_motor_cfg.replace(
            joint_names_expr=JOINT_NAMES,
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
        )
        go2_cfg = go2_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=init_state,
            actuators={"base_legs": dc_motor_cfg},
        )
        self.scene.robot = go2_cfg


class Go2PlayEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10

        # spawn the robot randomly instead of according to curriculum
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False

        # disable joint curriculum
        self.curriculum.joint_stiffness_and_damping_difficulty = None
        self.curriculum.joint_friction_difficulty = None

        self.events.push_robot = None


class ManagerBasedRLEnvRecord(ManagerBasedRLEnv):
    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        critic: Any,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.critic = critic
        self.expert_policy_flag_buf = None


@configclass
class HardwareObsCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_lin_acc = ObsTerm(
        func=mdp.imu_lin_acc,
        params={
            "asset_cfg": SceneEntityCfg(name="imu_body"),
        },
    )
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params=_get_joint_params())
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params=_get_joint_params())
    cos_sin_phase = ObsTerm(
        func=go2_mdp.cos_sin_phase, params={"period": WALKING_PERIOD}
    )
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": HEIGHT_SCAN_OFFSET,
        },
        clip=(-2.0, 2.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


class TransformerObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs", self._env.obs_buf["recorded_obs"]


class PreStepCommandsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "commands", self._env.command_manager.get_command("base_velocity")


class RewardRecorder(RecorderTerm):
    def record_post_step(self):
        return "reward", self._env.reward_buf


class ValueRecorder(RecorderTerm):
    def record_post_step(self):
        with torch.no_grad():
            return "value", self._env.critic(self._env.obs_buf["policy"]).squeeze()


class ExpertPolicyFlagRecorder(RecorderTerm):
    def record_pre_step(self):
        with torch.no_grad():
            return "expert_policy_flag", self._env.expert_policy_flag_buf.squeeze()


@configclass
class TransformerObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = TransformerObservationsRecorder


@configclass
class ExpertPolicyFlagRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ExpertPolicyFlagRecorder


@configclass
class PreStepCommandsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepCommandsRecorder


@configclass
class PreStepValueRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ValueRecorder


@configclass
class PostStepRewardRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = RewardRecorder


@configclass
class TransformerRecorderManagerCfg(RecorderManagerBaseCfg):

    record_pre_step_obs = TransformerObservationsRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_expert_policy_flag = ExpertPolicyFlagRecorderCfg()
    record_pre_step_commands = PreStepCommandsRecorderCfg()
    record_post_step_reward = PostStepRewardRecorderCfg()
    record_pre_step_value = PreStepValueRecorderCfg()


class Go2RecordEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.recorded_obs = HardwareObsCfg()
        self.recorders = TransformerRecorderManagerCfg()

        # spawn the robot randomly instead of according to curriculum
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False

        # disable joint curriculum
        self.curriculum.joint_stiffness_and_damping_difficulty = None
        self.curriculum.joint_friction_difficulty = None

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


class Go2SimEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        self.observations.obs = HardwareObsCfg()
        self.observations.policy.enable_corruption = False

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # disable joint curriculum
        self.curriculum.joint_stiffness_and_damping_difficulty = None
        self.curriculum.joint_friction_difficulty = None

        # add a plane terrain option
        self.scene.terrain.terrain_generator.sub_terrains["plane"] = (
            terrain_gen.MeshPlaneTerrainCfg(proportion=0.1)
        )

        # remove randomization
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.05, 0.05)
        self.events.reset_robot_joints = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # uniform velocity command
        # vel and ranges should be set after creating this cfg
        self.commands.base_velocity = go2_mdp.WalkStraightCommandCfg(
            asset_name="robot",
            forward_vel=0.0,
            kx=1.0,
            ky=1.0,
            kyaw=1.0,
            debug_vis=True,
            ranges=go2_mdp.WalkStraightCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
            resampling_time_range=(1000.0, 1000.0),
        )
