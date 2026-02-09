from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.managers import (
    SceneEntityCfg,
    ManagerTermBase,
    EventTermCfg,
    CommandTermCfg,
    CommandTerm,
)
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import isaaclab.envs.mdp.events as events
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.terrains import TerrainImporter
import torch
import functools as ft
from typing import Tuple, List, Sequence, Literal
from dataclasses import MISSING


def cos_sin_phase(env: ManagerBasedRLEnv, period: float = 0.5) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        steps = env.episode_length_buf  # (N,)
        phase = steps * env.step_dt * 2.0 * torch.pi / period  # (N,)
        return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)  # (N, 2)
    else:
        return torch.zeros(env.num_envs, 2)


def cum_cmd_vel_similarity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Observation that holds the cumulative cosine similarity of the commanded
    velocity and the current velocity of the robot."""
    if not hasattr(env, "obs_buf"):
        return torch.zeros(env.num_envs, device=env.device).unsqueeze(-1)
    elif "curriculum" not in env.obs_buf:
        return torch.zeros(env.num_envs, device=env.device).unsqueeze(-1)

    command = env.command_manager.get_command("base_velocity")[..., :2]
    lin_vel = mdp.base_lin_vel(env)[..., :2]
    sim = torch.nn.functional.cosine_similarity(command, lin_vel, dim=-1).unsqueeze(-1)
    sim *= torch.norm(command, dim=-1).unsqueeze(-1) > 0.1
    sim_before = env.obs_buf["curriculum"]["cum_cmd_vel_similarity"]
    sim_before[env.episode_length_buf.unsqueeze(-1) == 0] = 0.0
    return sim_before + sim


def terrain_levels_cmd_vel_sim(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    up_thresh: float = 0.5,
    down_thresh: float = 0.1,
) -> torch.Tensor:
    """Curriculum based on the cosine similarity between the commanded velocity
    and the current velocity of the robot. The cumulative sum of this metric
    for the episode is used to determine terrain difficulty."""
    terrain: TerrainImporter = env.scene.terrain

    if "curriculum" not in env.obs_buf:
        return torch.zeros((), device=env.device)

    sim = (
        env.obs_buf["curriculum"]["cum_cmd_vel_similarity"][env_ids]
        / env.max_episode_length
    ).squeeze()
    move_up = sim > up_thresh
    move_down = sim < down_thresh
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float()).unsqueeze(-1)


def joint_deviation_l1_weighted(
    env: ManagerBasedRLEnv,
    weights: Tuple[float, ...],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one, weighted by
    the provided weights."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    weights = torch.tensor(weights).to(env.device)
    return torch.sum(torch.abs(angle) * weights, dim=1)


def mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Get the difference of the masses of specified body from their default masses."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    default_mass = asset.data.default_mass[:, body_ids]
    masses = asset.root_physx_view.get_masses()[:, body_ids]
    return (masses - default_mass).to(env.device)


def joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """Get the difference of the joint stiffness of specified joints from their
    default stiffness."""
    asset: Articulation = env.scene[asset_cfg.name]
    stiffness = asset.actuators["base_legs"].stiffness[:, asset_cfg.joint_ids]
    stiffness = stiffness * scale + bias
    return stiffness


def joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """Get the difference of the joint stiffness of specified joints from their
    default stiffness."""
    asset: Articulation = env.scene[asset_cfg.name]
    damping = asset.actuators["base_legs"].damping[:, asset_cfg.joint_ids]
    damping = damping * scale + bias
    return damping


def joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """Get the difference of the joint stiffness of specified joints from their
    default stiffness."""
    asset: Articulation = env.scene[asset_cfg.name]
    friction = asset._data.joint_friction[:, asset_cfg.joint_ids]
    friction = friction * scale + bias
    return friction


class DifficultyManagerTerm(ManagerTermBase):
    """Base class manager terms that keep track of a difficulty level."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._difficulty = 1.0

    def set_difficulty(self, difficulty: float):
        """Set the difficulty of the term."""
        self._difficulty = max(0.0, min(1.0, difficulty))

    @property
    def difficulty(self) -> float:
        """Get the difficulty of the term."""
        return self._difficulty


class randomize_actuator_gains(DifficultyManagerTerm):
    """Randomize the actuator gains of the robot. As difficulty increases, the
    gains multiplier are scaled away from 1.0."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        self.stiffness_range = cfg.params.get(
            "stiffness_distribution_params", (1.0, 1.0)
        )
        self.damping_range = cfg.params.get("damping_distribution_params", (1.0, 1.0))

        self._randomize_fn = ft.partial(
            mdp.randomize_actuator_gains,
            operation=cfg.params.get("operation", "scale"),
            distribution=cfg.params.get("distribution", "uniform"),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        stiff_low = 1.0 + (self.stiffness_range[0] - 1.0) * self.difficulty
        stiff_high = 1.0 + (self.stiffness_range[1] - 1.0) * self.difficulty
        damp_low = 1.0 + (self.damping_range[0] - 1.0) * self.difficulty
        damp_high = 1.0 + (self.damping_range[1] - 1.0) * self.difficulty
        self._randomize_fn(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params=(stiff_low, stiff_high),
            damping_distribution_params=(damp_low, damp_high),
        )


class randomize_joint_friction(DifficultyManagerTerm):
    """Randomize the joint friction of the robot. As difficulty increases, the
    friction upper range is increased from low to high."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        self.friction_range = cfg.params.get("friction_distribution_params", (0.0, 0.0))
        self._ranomize_fn = ft.partial(
            mdp.randomize_joint_parameters,
            operation=cfg.params.get("operation", "add"),
            distribution=cfg.params.get("distribution", "uniform"),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        lowest = self.friction_range[0]
        highest = self.friction_range[1]
        high = lowest + (highest - lowest) * self.difficulty
        self._ranomize_fn(
            env,
            env_ids,
            asset_cfg,
            friction_distribution_params=(lowest, high),
        )


def event_linear_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_step: int,
    end_step: int,
):
    """Curriculum that modifies the difficulty of the event term linearly from
    start_step to end_step. The difficulty is set to 0.0 at start_step and 1.0
    at end_step. The event term is expected to have a set_difficulty method."""
    if not hasattr(env, "event_manager"):
        return
    term_cfg = env.event_manager.get_term_cfg(term_name)
    if not hasattr(term_cfg.func, "set_difficulty"):
        raise ValueError(
            f"Event term '{term_name}' does not have a set_difficulty method."
        )
    func: DifficultyManagerTerm = term_cfg.func
    step = env.common_step_counter
    difficulty = max(0.0, min((step - start_step) / (end_step - start_step), 1.0))
    func.set_difficulty(difficulty)
    return difficulty


class randomize_rigid_body_material_same(events.randomize_rigid_body_material):
    """Assign the same material to all of the specified bodies."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(
            0, num_buckets, (len(env_ids), total_num_shapes), device="cpu"
        )
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # assign the same material to each body
        self.assigned_material = material_samples[:, 0:1]

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                # assign the same material to each body
                materials[env_ids, start_idx:end_idx] = self.assigned_material
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)

        # prepare to be used for observations
        self.assigned_material = self.assigned_material.squeeze(1).to(env.device)


def material_properties(env: ManagerBasedRLEnv):
    if not hasattr(env, "event_manager"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    event_func = None
    for event_term in env.event_manager._mode_term_cfgs["startup"]:
        if isinstance(event_term.func, randomize_rigid_body_material_same):
            event_func: randomize_rigid_body_material_same = event_term.func
            break

    if event_func is None:
        raise RuntimeError("No event found for material properties")

    return event_func.assigned_material


def desired_contacts(
    env: ManagerBasedRLEnv,
    bias: List[float],
    contact_ratio: float,
    period: float,
) -> torch.Tensor:
    """Compute the desired contacts (contact = true) for the robot to follow a
    desired gait pattern given by bias and contact ratio."""
    cs_phase = cos_sin_phase(env, period)
    phase = torch.atan2(cs_phase[:, 1], cs_phase[:, 0]).unsqueeze(-1)
    bias = torch.tensor(bias).to(env.device).unsqueeze(0).expand(env.num_envs, -1)
    tf = torch.remainder(phase / (2 * torch.pi) + 0.5 + bias, 1)
    des_contacts = tf < contact_ratio
    return des_contacts


def gait_reward(
    env: ManagerBasedRLEnv,
    bias: List[float],
    contact_ratio: float,
    period: float,
    command_name: str,
    command_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the robot for following a desired gait pattern."""
    # Get contacts
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    # desired contacts
    des_contacts = desired_contacts(env, bias, contact_ratio, period)

    # reward
    reward = torch.logical_not(torch.logical_xor(contacts, des_contacts))
    reward = reward.float().sum(dim=-1)
    # no reward for commands below the threshold
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > command_threshold
    )
    return reward


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_ratio: float,
    period: float,
    stds: float = 4.0,
) -> torch.Tensor:
    """
    This function rewards the agent for taking steps with a air time that is
    close to the desired air time deterined by the contact ratio and period.
    This helps ensure that the robot lifts its feet off the ground and takes steps.

    If the commands are small (i.e. the agent is not supposed to take a step),
    then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = last_air_time * first_contact
    mu = (1 - contact_ratio) * period
    reward = _gaussian(air_time, mu, mu / stds)
    reward = reward.sum(dim=1)

    # no reward for almost zero commands
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    scan_sensor_names: List[str],
    foot_radius: float,
) -> torch.Tensor:
    """Get the height of the feet above the ground."""
    sensors = [env.scene.sensors[name] for name in scan_sensor_names]
    feet_height = torch.stack(
        [
            sensor.data.pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].squeeze()
            for sensor in sensors
        ],
        dim=-1,
    )
    return feet_height - foot_radius


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scan_sensor_names: List[str],
    foot_radius: float,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        feet_height(env, scan_sensor_names, foot_radius) - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_height_gait_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    scan_sensor_names: List[str],
    foot_radius: float,
    bias: List[float],
    contact_ratio: float,
    period: float,
    swing_height: float,
    stds: float = 4.0,
) -> torch.Tensor:
    """Reward the robot for foot height near the desired foot height during the
    swing phase of the gait."""
    # Get contacts
    des_contacts = desired_contacts(env, bias, contact_ratio, period)
    des_swings = torch.logical_not(des_contacts)

    fh = feet_height(env, scan_sensor_names, foot_radius)
    reward = _gaussian(fh, swing_height, swing_height / stds)
    reward *= des_swings  # only reward during swing phase
    reward = reward.sum(dim=1)

    # no reward for commands below the threshold
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > command_threshold
    )
    return reward


def _gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2))


class VaryingVelocityCommands(UniformVelocityCommand, DifficultyManagerTerm):
    cfg: "VaryingVelocityCommandsCfg"

    def __init__(self, cfg: "VaryingVelocityCommandsCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.type_probs = torch.tensor(cfg.type_probs, device=self.device)

        self.env_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.const_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.accel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.accel_std = torch.zeros(self.num_envs, 3, device=self.device)
        self.jerk_std = torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        self.env_type[env_ids] = torch.multinomial(
            self.type_probs, len(env_ids), replacement=True
        )

        r = torch.empty(len(env_ids), device=self.device)
        d = self._difficulty
        self.vel_command_b[env_ids, 0] = d * r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = d * r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = d * r.uniform_(*self.cfg.ranges.ang_vel_z)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)

        self.accel_cmd[env_ids, 0] = d * r.uniform_(
            *self.cfg.const_accel_ranges.lin_vel_x
        )
        self.accel_cmd[env_ids, 1] = d * r.uniform_(
            *self.cfg.const_accel_ranges.lin_vel_y
        )
        self.accel_cmd[env_ids, 2] = d * r.uniform_(
            *self.cfg.const_accel_ranges.ang_vel_z
        )

        self.accel_std[env_ids, 0] = d * r.uniform_(
            *self.cfg.rand_accel_std_ranges.lin_vel_x
        )
        self.accel_std[env_ids, 1] = d * r.uniform_(
            *self.cfg.rand_accel_std_ranges.lin_vel_y
        )
        self.accel_std[env_ids, 2] = d * r.uniform_(
            *self.cfg.rand_accel_std_ranges.ang_vel_z
        )

        self.jerk_std[env_ids, 0] = d * r.uniform_(
            *self.cfg.rand_jerk_std_ranges.lin_vel_x
        )
        self.jerk_std[env_ids, 1] = d * r.uniform_(
            *self.cfg.rand_jerk_std_ranges.lin_vel_y
        )
        self.jerk_std[env_ids, 2] = d * r.uniform_(
            *self.cfg.rand_jerk_std_ranges.ang_vel_z
        )

    def _update_command(self):
        d = self._difficulty

        # constant; heading command
        env_ids = torch.nonzero(self.env_type == 1, as_tuple=False).squeeze()
        heading_error = math_utils.wrap_to_pi(
            self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
        )
        self.vel_command_b[env_ids, 2] = (
            self.cfg.heading_control_stiffness * heading_error
        )

        # standing still
        env_ids = torch.nonzero(self.env_type == 2, as_tuple=False).squeeze()
        self.vel_command_b[env_ids, :] = 0.0

        # constant acceleration
        env_ids = torch.nonzero(self.env_type == 3, as_tuple=False).squeeze()
        self.vel_command_b[env_ids] += self.accel_cmd[env_ids] * self._env.step_dt

        # random acceleration
        env_ids = torch.nonzero(self.env_type == 4, as_tuple=False).squeeze()
        randn = torch.randn_like(self.accel_std[env_ids], device=self.device)
        rand_accels = randn * self.accel_std[env_ids]
        self.vel_command_b[env_ids] += rand_accels * self._env.step_dt

        # random jerk
        env_ids = torch.nonzero(self.env_type == 5, as_tuple=False).squeeze()
        randn = torch.randn_like(self.accel_std[env_ids], device=self.device)
        rand_jerk = randn * self.jerk_std[env_ids]
        self.accel_cmd[env_ids] += rand_jerk * self._env.step_dt
        self.accel_cmd[env_ids, 0] = torch.clamp(
            self.accel_cmd[env_ids, 0],
            min=self.cfg.const_accel_ranges.lin_vel_x[0],
            max=self.cfg.const_accel_ranges.lin_vel_x[1],
        )
        self.accel_cmd[env_ids, 1] = torch.clamp(
            self.accel_cmd[env_ids, 1],
            min=self.cfg.const_accel_ranges.lin_vel_y[0],
            max=self.cfg.const_accel_ranges.lin_vel_y[1],
        )
        self.accel_cmd[env_ids, 2] = torch.clamp(
            self.accel_cmd[env_ids, 2],
            min=self.cfg.const_accel_ranges.ang_vel_z[0],
            max=self.cfg.const_accel_ranges.ang_vel_z[1],
        )
        self.vel_command_b[env_ids] += self.accel_cmd[env_ids] * self._env.step_dt

        # clip all cmds within range
        self.vel_command_b[:, 0] = torch.clamp(
            self.vel_command_b[:, 0],
            min=d * self.cfg.ranges.lin_vel_x[0],
            max=d * self.cfg.ranges.lin_vel_x[1],
        )
        self.vel_command_b[:, 1] = torch.clamp(
            self.vel_command_b[:, 1],
            min=d * self.cfg.ranges.lin_vel_y[0],
            max=d * self.cfg.ranges.lin_vel_y[1],
        )
        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            min=d * self.cfg.ranges.ang_vel_z[0],
            max=d * self.cfg.ranges.ang_vel_z[1],
        )


@configclass
class VaryingVelocityCommandsCfg(UniformVelocityCommandCfg):
    class_type: type = VaryingVelocityCommands

    type_probs: Sequence[float] = [0.3, 0.3, 0.01, 0.1, 0.1, 0.2]
    """Probabilities for each command type; does not need to sum to 1.0.

    The command types are:
        0: constant; angular velocity command
        1: constant; heading command
        2: standing still
        3: constant acceleration
        4: random acceleration
        5: random jerk
    """

    const_accel_ranges: UniformVelocityCommandCfg.Ranges = (
        UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.25, 0.25),
        )
    )

    rand_accel_std_ranges: UniformVelocityCommandCfg.Ranges = (
        UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 2.0),
            lin_vel_y=(0.25, 1.0),
            ang_vel_z=(0.25, 1.0),
        )
    )

    rand_jerk_std_ranges: UniformVelocityCommandCfg.Ranges = (
        UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(2.0, 10.0),
            lin_vel_y=(1.0, 5.0),
            ang_vel_z=(1.0, 5.0),
        )
    )


def cmd_linear_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_step: int,
    end_step: int,
    start_difficulty: float = 0.0,
    end_difficulty: float = 1.0,
):
    """Curriculum that modifies the difficulty of the command term linearly from
    start_step to end_step. The command term is expected to have a set_difficulty
    method."""
    term: DifficultyManagerTerm = env.command_manager.get_term(term_name)
    if not hasattr(term, "set_difficulty"):
        raise ValueError(
            f"Command term '{term_name}' does not have a set_difficulty method."
        )
    step = env.common_step_counter
    difficulty = start_difficulty + (end_difficulty - start_difficulty) * (
        step - start_step
    ) / (end_step - start_step)
    difficulty = max(start_difficulty, min(difficulty, end_difficulty))
    term.set_difficulty(difficulty)
    return difficulty


class WalkStraightCommand(CommandTerm):
    cfg: "WalkStraightCommandCfg"

    def __init__(self, cfg: "WalkStraightCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_heading = torch.zeros(self.num_envs, device=self.device)
        self.x_des = torch.zeros(self.num_envs, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self.init_pos[env_ids] = self.robot.data.root_pos_w[env_ids]
        self.init_heading[env_ids] = self.robot.data.heading_w[env_ids]
        self.x_des[env_ids] = 0.0
        return extras

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        errors = self._compute_errors(range(self.num_envs))
        self.metrics.update(errors)

    def _resample_command(self, env_ids: Sequence[int]):
        # no resampling needed
        pass

    def _update_command(self):
        self.x_des += self.cfg.forward_vel * self._env.step_dt
        errors = self._compute_errors(range(self.num_envs))
        self.vel_command_b[:, 0] = self.cfg.kx * errors["x_error"]
        self.vel_command_b[:, 1] = self.cfg.ky * errors["y_error"]
        self.vel_command_b[:, 2] = self.cfg.kyaw * errors["yaw_error"]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _compute_errors(self, env_ids: Sequence[int]):
        curr_pos = self.robot.data.root_pos_w[env_ids]
        curr_heading = self.robot.data.heading_w[env_ids]
        curr_vx = self.robot.data.root_lin_vel_b[env_ids, 0]

        delta_pos = curr_pos - self.init_pos[env_ids]
        heading_error = math_utils.wrap_to_pi(curr_heading - self.init_heading[env_ids])
        cos_heading = torch.cos(-self.init_heading[env_ids])
        sin_heading = torch.sin(-self.init_heading[env_ids])
        dx = cos_heading * delta_pos[:, 0] - sin_heading * delta_pos[:, 1]
        dy = sin_heading * delta_pos[:, 0] + cos_heading * delta_pos[:, 1]

        x_error = self.x_des[env_ids] - dx
        y_error = -dy
        yaw_error = -heading_error
        vx_error = self.cfg.forward_vel - curr_vx
        return {
            "x_error": x_error,
            "y_error": y_error,
            "yaw_error": yaw_error,
            "vx_error": vx_error,
        }


@configclass
class WalkStraightCommandCfg(CommandTermCfg):
    class_type: type = WalkStraightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    forward_vel: float = MISSING
    kx: float = 1.0
    ky: float = 1.0
    kyaw: float = 1.0

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
