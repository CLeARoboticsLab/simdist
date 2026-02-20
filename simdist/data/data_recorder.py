import os
from datetime import datetime
import yaml

import torch
from tqdm import trange

from simdist.rl.go2 import Go2RecordEnvCfg, ManagerBasedRLEnvRecord
from simdist.utils.torch import get_actor_critic_from_iteration
from simdist.utils import paths


class DataRecorder:
    def __init__(self, simulation_app, cfg: dict):
        self.simulation_app = simulation_app
        self.N = cfg["envs"]
        self.steps = cfg["steps"]
        run_name = cfg["rl_run"]
        seed = cfg["seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nm = cfg["action_corruption"]["intervals"]["noise_min"]
        self.nM = cfg["action_corruption"]["intervals"]["noise_max"]
        self.nnm = cfg["action_corruption"]["intervals"]["no_noise_min"]
        self.nnM = cfg["action_corruption"]["intervals"]["no_noise_max"]
        self.nnp = cfg["action_corruption"]["never_noise_prob"]
        assert self.nM > self.nm
        assert self.nnM > self.nnm

        dataset_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_path = paths.get_sim_dataset_dir(dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        with open(os.path.join(dataset_path, "record.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # load actors and critic
        self.expert_policy, self.critic = get_actor_critic_from_iteration(
            run_name, cfg["expert"], self.device
        )
        if self.critic is None:
            raise FileNotFoundError(f"Critic file not found for {run_name}")
        self.non_expert_policies = [
            get_actor_critic_from_iteration(run_name, i, self.device)[0]
            for i in cfg["non_experts"]
        ]
        self.num_non_experts = len(self.non_expert_policies)
        self.expert_prob = cfg["expert_prob"]

        # create environment
        env_cfg = Go2RecordEnvCfg()
        env_cfg.seed = seed
        env_cfg.recorders.dataset_export_dir_path = dataset_path
        env_cfg.recorders.dataset_filename = paths.get_raw_data_filename()
        env_cfg.scene.num_envs = self.N

        self.env = ManagerBasedRLEnvRecord(env_cfg, self.critic)

        total_steps = self.N * self.steps
        print(f"{total_steps/1e6:.2f}M steps will be generated")

        self.init_action_corruption(cfg)

    def init_policies(self, obs: dict):
        self.action_dim = self.expert_policy(obs["policy"][0:1]).shape[-1]
        self.uses_expert = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.non_expert_policy = torch.zeros(
            self.N, dtype=torch.int64, device=self.device
        )
        self.update_policies(torch.ones(self.N, device=self.device))

    def update_policies(self, reset_buf: torch.Tensor):
        reset_indices = reset_buf.nonzero(as_tuple=True)[0]

        # Decide which reset environments use expert
        use_expert = (
            torch.rand(len(reset_indices), device=self.device) < self.expert_prob
        )
        self.uses_expert[reset_indices] = use_expert

        # For non-expert policies, randomly select policy indices
        num_non_expert = (~use_expert).sum()
        non_expert_indices = reset_indices[~use_expert]
        if num_non_expert > 0:
            sampled_policies = torch.randint(
                low=0,
                high=self.num_non_experts,
                size=(num_non_expert,),
                device=self.device,
            )
            self.non_expert_policy[non_expert_indices] = sampled_policies

    def get_action(self, obs: dict):
        obs = obs["policy"]
        expert_indices = self.uses_expert.nonzero(as_tuple=True)[0]
        actions = torch.zeros((self.N, self.action_dim), device=self.device)

        if len(expert_indices) > 0:
            actions[expert_indices] = self.expert_policy(obs[expert_indices])

        for i in range(self.num_non_experts):
            idx = (self.non_expert_policy == i) & (~self.uses_expert)
            indices = idx.nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                actions[indices] = self.non_expert_policies[i](obs[indices])

        return actions

    def init_action_corruption(self, cfg: dict):
        # Initialize noised states and timers

        # some envs will never be noised
        self.never_noised = torch.rand(self.N, device=self.device) < self.nnp

        # start with half of the envs in the noised state
        self.noised = torch.rand(self.N, device=self.device) < 0.5
        self.noised = torch.where(
            self.never_noised,
            torch.zeros_like(self.noised, dtype=torch.bool),
            self.noised,
        )

        # Initialize self.timers for switching between noised and non-noised
        # states
        self.timers = torch.empty(self.N, device=self.device, dtype=torch.long)
        self.timers[self.noised] = torch.randint(
            self.nm, self.nM, size=(self.noised.sum(),), device=self.device
        )
        self.timers[~self.noised] = torch.randint(
            self.nnm, self.nnM, size=((~self.noised).sum(),), device=self.device
        )

        # initialize noise distributions
        min_noise = torch.tensor([a["min_noise"] for a in cfg["system"]["actions"]])
        max_noise = torch.tensor([a["max_noise"] for a in cfg["system"]["actions"]])
        action_dim = len(min_noise)
        uniform_samples = torch.rand((self.N, action_dim))
        noise_std = min_noise + (max_noise - min_noise) * uniform_samples
        self.noise_std = noise_std.to(self.device)

    def corrupt_action(self, action: torch.Tensor):
        # Corrupt actions for envs in noised state
        noise = torch.randn_like(action) * self.noise_std
        action = torch.where(self.noised.unsqueeze(1), action + noise, action)
        return action

    def update_corruption_state(self):
        # Update timers
        self.timers -= 1

        # Identify which environments should switch state
        flip_mask = (self.timers <= 0) & ~self.never_noised
        if flip_mask.any():
            # flip noised state for environments that should switch
            self.noised[flip_mask] = ~self.noised[flip_mask]

            # reset timers for environments that switched
            new_timers = torch.empty_like(self.timers[flip_mask])
            new_timers[self.noised[flip_mask]] = torch.randint(
                self.nm,
                self.nM,
                size=(self.noised[flip_mask].sum(),),
                device=self.device,
            )
            new_timers[~self.noised[flip_mask]] = torch.randint(
                self.nnm,
                self.nnM,
                size=((~self.noised[flip_mask]).sum(),),
                device=self.device,
            )
            self.timers[flip_mask] = new_timers

    def run(self):
        # reset environment
        obs, _ = self.env.reset()

        # initialize policies
        self.init_policies(obs)

        # start data generation
        for i in trange(self.steps, desc="Generating data"):
            with torch.no_grad():
                action = self.get_action(obs)
                action = self.corrupt_action(action)

            # the expert policy is used when the actions are not noised
            self.env.expert_policy_flag_buf = ~self.noised & self.uses_expert
            obs, rew, _, _, _ = self.env.step(action)
            self.update_policies(self.env.reset_buf)
            self.update_corruption_state()

        # export episodes
        print("Exporting episodes")
        self.env.recorder_manager.export_episodes()
        print("Done!")
        self.env.close()
        self.simulation_app.close()
