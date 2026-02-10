"""
Script load in model checkpoints and export the policies and value functions.
"""

import argparse
import re
import os
from isaaclab.app import AppLauncher

from simdist.rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export policies and value functions.")
parser.add_argument("--task", type=str, default="Go2Play", help="Name of the task.")
parser.add_argument(
    "-r", "--rl_run", type=str, required=True, help="Folder name of the run."
)
parser.add_argument(
    "--model_dir", type=str, default=None, help="Path to the model directory."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 1
args_cli.disable_fabric = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
)
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg

from simdist import rl  # noqa: F401
from simdist.utils.torch import export_torch_as_jit
from simdist.utils import paths


def find_model_files(directory):
    """
    Search for files matching the pattern 'model_*.pt' where * is an integer.
    """
    model_files = []
    pattern = re.compile(r"model_(\d+)\.pt$")  # Regex to capture the integer

    for root, _, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                model_path = os.path.join(root, file)
                model_id = int(match.group(1))  # Extract integer
                model_files.append((model_path, model_id))

    return model_files


def main():
    run_name = args_cli.rl_run
    rl_run_dir = paths.get_rl_run_dir(run_name)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # create isaac environment and agent
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )

    # find model checkpoints
    model_checkpoints = find_model_files(rl_run_dir)
    policies_dir = paths.get_rl_policies_dir(run_name)
    critics_dir = paths.get_rl_critics_dir(run_name)

    for resume_path, iteration in model_checkpoints:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)
        export_policy_as_jit(
            ppo_runner.alg.actor_critic,
            ppo_runner.obs_normalizer,
            path=policies_dir,
            filename=f"policy_{iteration}.pt",
        )
        export_torch_as_jit(
            ppo_runner.alg.actor_critic.critic,
            path=critics_dir,
            normalizer=ppo_runner.critic_obs_normalizer,
            filename=f"critic_{iteration}.pt",
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
