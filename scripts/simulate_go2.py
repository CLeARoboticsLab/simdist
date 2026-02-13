import argparse
import sys
import os

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # Keep only unknown args for hydra
    return args


args = parse_args()

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb
import torch
from isaaclab.envs import ManagerBasedRLEnv

from simdist.utils import paths, model as model_utils, config
from simdist.control.controller_base import ControllerInput
from simdist.control.mppi import MppiController
from simdist.rl.go2 import Go2SimEnvCfg


class Go2Sim:
    def __init__(self, cfg: dict):
        print("Loaded config:")
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        self.total_steps = cfg["sim"]["total_steps"]
        self.reset_steps = cfg["sim"]["reset_steps"]

        # create controller
        ckpt_dir = os.path.join(
            paths.get_model_checkpoints_dir(), self.cfg["model"]["checkpoint"]
        )
        model, model_cfg, _ = model_utils.load_model_from_ckpt(
            ckpt_dir, self.cfg["model"]["step"]
        )
        self.controller = MppiController(model, model_cfg, self.cfg["control"])
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(
            model_cfg["system"]
        )

        # create env cfg
        env_cfg = Go2SimEnvCfg()

        # set command in env
        env_cfg.commands.base_velocity.forward_vel = cfg["task"]["forward_vel"]
        env_cfg.commands.base_velocity.kx = cfg["task"]["kx"]
        env_cfg.commands.base_velocity.ky = cfg["task"]["ky"]
        env_cfg.commands.base_velocity.k_heading = cfg["task"]["k_heading"]

        # set up env
        terr = cfg["sim"]["terrain"]
        diff = cfg["sim"]["terrain_difficulty"]
        env_cfg.scene.terrain.terrain_generator.seed = cfg["sim"]["seed"]
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.size = (15.0, 15.0)
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (diff, diff)
        terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {terr: terrains[terr]}
        env_cfg.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        fric = cfg["sim"]["friction"]
        env_cfg.events.physics_material.params["static_friction_range"] = (fric, fric)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (fric, fric)
        rest = cfg["sim"]["restitution"]
        env_cfg.events.physics_material.params["restitution_range"] = (rest, rest)
        mass = cfg["sim"]["add_mass"]
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (mass, mass)

        # create the env
        self.env = ManagerBasedRLEnv(env_cfg)
        self.cmd_manager = self.env.command_manager
        self.obs_dict, _ = self.env.reset()
        proprio_obs, height_scan = self.get_obs(self.obs_dict)

        # set up the controller
        self.zero_cmd = np.zeros((3,))
        self.zero_action = np.zeros((12,))
        x = self.make_controller_input(proprio_obs, height_scan, self.zero_action)
        self.controller.initialize(x, self.zero_cmd)
        self.controller.reset(x, self.zero_cmd)
        self.controller.set_fut_cmd(self.zero_cmd)

        self.pbar = tqdm(
            desc="Simulation",
            unit="step",
            total=self.total_steps,
        )

        # wandb
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = (
            f"{date_time}"
            if cfg["wandb"]["run_name"] is None
            else cfg["wandb"]["run_name"]
        )
        if self.cfg["wandb"]["log"]:
            wandb.init(
                project="tyswy_go2_sim",
                entity=self.cfg["wandb"]["entity"],
                name=self.run_name,
                config=cfg,
            )

        # state
        self.steps = 0
        self.total_step_count = 0
        self.last_action = self.zero_action.copy()
        self.total_reward = 0
        self.episode_terminated = False
        self.episode_length = 0

    def run(self):
        while simulation_app.is_running():
            was_reset = self.step()

            if self.total_step_count >= self.total_steps:
                print("Total steps reached. Exiting simulation.")
                break

            if self.cfg["sim"]["kill_on_reset"] and was_reset:
                print("Episode terminated. Exiting simulation.")
                break

        self.logging()
        self.env.close()
        simulation_app.close()

    def get_obs(self, obs_dict):
        proprio_obs = np.concatenate(
            [
                obs_dict["obs"][name].squeeze().cpu().numpy()
                for name in self.proprio_obs_names
            ]
        )
        height_scan = obs_dict["obs"]["height_scan"].squeeze().cpu().numpy()
        return proprio_obs, height_scan

    def make_controller_input(
        self, proprio_obs: np.ndarray, extero_obs: np.ndarray, prev_action: np.ndarray
    ) -> ControllerInput:
        return {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "prev_action": prev_action,
        }

    def step(self):
        resetting = self.steps < self.reset_steps

        # send the command to the controller
        if resetting:
            # apply no command just after the robot is reset
            cmd = self.zero_cmd
        else:
            cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy()[0]
        self.controller.set_fut_cmd(cmd)

        # update the controller with latest measurements
        proprio_obs, height_scan = self.get_obs(self.obs_dict)
        self.controller.update(
            self.make_controller_input(proprio_obs, height_scan, self.last_action)
        )

        # get the action from the controller
        ctrl_out = self.controller.run_control()
        action = ctrl_out["actions"][0]  # get first action in sequence

        if resetting:
            # apply zero action just after reset to stabilize the simulation
            action = self.zero_action

        self.last_action = action

        # step the simulation
        action_torch = self.action_to_torch(action)
        self.obs_dict, reward, reset = self.env.step(action_torch)[0:3]
        self.pbar.update(1)
        self.steps += 1
        self.total_step_count += 1
        if not self.episode_terminated:
            self.total_reward += reward[0].item()
            self.episode_length += 1

        # process reset if terminated
        if reset[0]:
            self.reset()

        return reset[0]

    def action_to_torch(self, action):
        return torch.from_numpy(np.asarray(action)).unsqueeze(0).to(self.env.device)

    def reset(self):
        self.episode_terminated = True
        self.steps = 0
        proprio_obs, height_scan = self.get_obs(self.obs_dict)
        x = self.make_controller_input(proprio_obs, height_scan, self.zero_action)
        self.controller.reset(x, self.zero_cmd)

    def logging(self):
        rps = self.total_reward / (self.episode_length - self.reset_steps)
        metrics = {
            "total_reward": self.total_reward,
            "reward_per_step": rps,
            "total_steps": self.total_step_count,
            "episode_length": self.episode_length,
        }
        print(metrics)

        if self.cfg["wandb"]["log"]:
            wandb.log(metrics)
            wandb.finish()


@hydra.main(**paths.get_simulate_go2_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    go2_sim = Go2Sim(dict_cfg)
    go2_sim.run()


if __name__ == "__main__":
    main()
