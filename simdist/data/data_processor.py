import numpy as np
import h5py
import os
import yaml
import json

import torch
from tqdm import trange
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from simdist.utils import config as config_utils
from simdist.utils import paths
from simdist.utils import io as io_utils
from simdist.data import DATA_KEY
from simdist.modeling import types


class DataProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.sys = cfg["system"]
        self.H = cfg["history_length"]
        self.T = cfg["prediction_length"]
        self.beg_padding = cfg["beg_padding"]
        self.end_padding = cfg["end_padding"]

        self.proprio_obs_dim = config_utils.proprio_obs_dim_from_sys_config(self.sys)
        self.extero_obs_dim = config_utils.extero_obs_dim_from_sys_config(self.sys)
        self.action_dim = config_utils.action_dim_from_sys_config(self.sys)
        self.cmd_dim = config_utils.cmd_dim_from_sys_config(self.sys)
        self.proprio_obs_names = config_utils.proprio_obs_names_from_sys_config(
            self.sys
        )
        self.extero_obs_names = config_utils.extero_obs_names_from_sys_config(self.sys)

        self.stats_trackers = {
            "proprio_obs": _RunningStats(self.proprio_obs_dim),
            "extero_obs": _RunningStats(self.extero_obs_dim),
            "actions": _RunningStats(self.action_dim),
            "commands": _RunningStats(self.cmd_dim),
            "rewards": _RunningStats(0),
            "values": _RunningStats(0),
        }
        self.exp_pol_flags_tracker = _RunningStats(0)

    def run(self):
        raw_data_path = paths.get_raw_data_path(self.dataset_name)
        raw_data_handler = HDF5DatasetFileHandler()
        raw_data_handler.open(raw_data_path)

        processed_data_dir = paths.get_processed_data_dir(
            self.dataset_name, self.sys["name"], self.H, self.T
        )
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        with open(os.path.join(processed_data_dir, "system.yaml"), "w") as f:
            yaml.dump(self.sys, f, default_flow_style=False)

        episode_names = list(raw_data_handler.get_episode_names())
        device = "cpu"

        start_idxs_h5 = _H5Appender(
            paths.get_start_idxs_path(processed_data_dir),
            DATA_KEY,
            0,
        )
        proprio_obs_h5 = _H5Appender(
            paths.get_proprio_obs_path(processed_data_dir),
            DATA_KEY,
            self.proprio_obs_dim,
        )
        extero_obs_h5 = _H5Appender(
            paths.get_extero_obs_path(processed_data_dir),
            DATA_KEY,
            self.extero_obs_dim,
        )
        acts_h5 = _H5Appender(
            paths.get_actions_path(processed_data_dir),
            DATA_KEY,
            self.action_dim,
        )
        cmd_h5 = _H5Appender(
            paths.get_commands_path(processed_data_dir),
            DATA_KEY,
            self.cmd_dim,
        )
        rewards_h5 = _H5Appender(
            paths.get_rewards_path(processed_data_dir),
            DATA_KEY,
            0,
        )
        values_h5 = _H5Appender(
            paths.get_values_path(processed_data_dir),
            DATA_KEY,
            0,
        )
        exp_pol_flags_h5 = _H5Appender(
            paths.get_expert_policy_flags_path(processed_data_dir),
            DATA_KEY,
            0,
        )

        all_ep_lens = []
        ep_rewards = []
        start = 0

        for ep_name in trange(len(episode_names), desc="Processing episodes"):
            ep_name = episode_names[ep_name]
            episode_data = raw_data_handler.load_episode(ep_name, device)
            ep_len = episode_data.data["actions"].shape[0]
            all_ep_lens.append(ep_len)

            if ep_len < self.H + self.T + self.beg_padding + self.end_padding:
                continue

            if "obs_concat" in episode_data.data:
                ep_proprio_obs = episode_data.data["obs_concat"]
            elif "obs" in episode_data.data:
                ep_proprio_obs = torch.cat(
                    [episode_data.data["obs"][name] for name in self.proprio_obs_names],
                    dim=-1,
                )
            else:
                raise ValueError(
                    f"Episode {ep_name} does not contain obs_concat or obs data."
                )
            proprio_obs_np = _to_valid_numpy(ep_proprio_obs)
            proprio_obs_h5.append(proprio_obs_np)
            self.stats_trackers["proprio_obs"].update(proprio_obs_np)

            ep_extero_obs = []
            if "obs" in episode_data.data:
                ep_extero_obs.extend(
                    [episode_data.data["obs"][name] for name in self.extero_obs_names]
                )
            else:
                for name in self.extero_obs_names:
                    if name not in episode_data.data:
                        raise ValueError(
                            f"Episode {ep_name} does not contain {name} in obs data."
                        )
                    ep_extero_obs.append(episode_data.data[name])
            ep_extero_obs = torch.cat(ep_extero_obs, dim=-1)
            extero_obs_np = _to_valid_numpy(ep_extero_obs)
            extero_obs_h5.append(extero_obs_np)
            self.stats_trackers["extero_obs"].update(extero_obs_np)

            ep_acts = episode_data.data["actions"]
            acts_np = _to_valid_numpy(ep_acts)
            acts_h5.append(acts_np)
            self.stats_trackers["actions"].update(acts_np)

            ep_commands = episode_data.data["commands"]
            cmd_np = _to_valid_numpy(ep_commands)
            cmd_h5.append(cmd_np)
            self.stats_trackers["commands"].update(cmd_np)

            if "reward" in episode_data.data:
                rewards = episode_data.data["reward"]
            else:
                rewards = torch.zeros(ep_len, device=device)
            rewards_np = _to_valid_numpy(rewards)
            ep_rewards.append(np.sum(rewards_np))
            rewards_h5.append(rewards_np)
            self.stats_trackers["rewards"].update(rewards_np)

            if "value" in episode_data.data:
                values = episode_data.data["value"]
            else:
                values = torch.zeros(ep_len, device=device)
            values_np = _to_valid_numpy(values)
            values_h5.append(values_np)
            self.stats_trackers["values"].update(values_np)

            if "expert_policy_flag" in episode_data.data:
                exp_pol_flags = episode_data.data["expert_policy_flag"]
            else:
                exp_pol_flags = torch.zeros(ep_len, device=device)
            exp_pol_flags_np = _to_valid_numpy(exp_pol_flags)
            exp_pol_flags_h5.append(exp_pol_flags_np)
            self.exp_pol_flags_tracker.update(exp_pol_flags_np)

            # Store valid starting indices where the episode didn't terminate
            # before the end of the trajectory
            ep_start_idxs = torch.arange(
                start + self.beg_padding,
                start + ep_len - (self.H + self.T + self.end_padding),
                device=device,
            )
            start_idxs_h5.append(_to_valid_numpy(ep_start_idxs))
            start += ep_len

        scaler_params: types.ScalerParams = {
            k: tracker.finalize() for k, tracker in self.stats_trackers.items()
        }
        io_utils.save_scaler_params(scaler_params, processed_data_dir)

        exp_ratio = self.exp_pol_flags_tracker.finalize()["mean"]
        rew_per_step = self.stats_trackers["rewards"].finalize()["mean"]
        dataset_metrics = {
            "num_episodes": len(all_ep_lens),
            "avg_episode_length": np.mean(all_ep_lens).item(),
            "num_trajectories": len(start_idxs_h5),
            "avg_reward_per_episode": np.mean(ep_rewards).item(),
            "avg_reward_per_step": rew_per_step,
            "exp_actions_to_total_actions_ratio": exp_ratio,
        }
        with open(os.path.join(processed_data_dir, "dataset_metrics.json"), "w") as f:
            json.dump(dataset_metrics, f, indent=4)

        raw_data_handler.close()
        start_idxs_h5.close()
        proprio_obs_h5.close()
        extero_obs_h5.close()
        acts_h5.close()
        cmd_h5.close()
        rewards_h5.close()
        values_h5.close()
        exp_pol_flags_h5.close()


class _H5Appender:
    def __init__(
        self,
        path: str,
        key: str,
        feat_dim: int,
        dtype: np.dtype = np.float32,
    ):
        self.f = h5py.File(path, "w")
        self.feat_dim = feat_dim
        self.dtype = np.dtype(dtype)

        if self.feat_dim == 0:
            shape = (0,)
            maxshape = (None,)
        else:
            shape = (0, feat_dim)
            maxshape = (None, feat_dim)

        self.dset = self.f.create_dataset(
            key,
            shape=shape,
            maxshape=maxshape,
            dtype=self.dtype,
        )

    def __len__(self):
        return self.dset.shape[0]

    def append(self, batch: np.ndarray):
        if self.feat_dim == 0:
            n = self.dset.shape[0]
            assert batch.ndim == 1
        else:
            n, m = self.dset.shape
            assert batch.shape[1] == m
        self.dset.resize(n + batch.shape[0], axis=0)
        self.dset[n : n + batch.shape[0]] = batch.astype(self.dtype, copy=False)

    def close(self):
        self.f.flush()
        self.f.close()


class _RunningStats:
    def __init__(self, dim: int = 1):
        # Ensure dim is at least 1 so sum is an array, not a scalar
        actual_dim = max(1, dim)
        self.n = 0
        self.sum = np.zeros(actual_dim)
        self.sum_sq = np.zeros(actual_dim)

    def update(self, data: np.ndarray):
        # data shape: (T, dim) or (T,)
        # Force data to be 2D so axis=0 always leaves a 1D array result
        if data.ndim == 1:
            data = data[:, np.newaxis]

        self.n += data.shape[0]
        self.sum += np.sum(data, axis=0)
        self.sum_sq += np.sum(np.square(data), axis=0)

    def finalize(self) -> types.Stats:
        mean = self.sum / self.n
        var = (self.sum_sq / self.n) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))

        # Ensure we return a list, even if it's just one element
        # .tolist() on a 1D array [x] returns [x]
        return {"mean": mean.tolist(), "std": std.tolist()}


def _replace_nans_and_infs(tensor: torch.Tensor):
    """
    Replaces NaNs and Infs in a PyTorch tensor with 0 and prints a warning if
    replacements occur.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with NaNs and Infs replaced by 0.
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    if nan_mask.any() or inf_mask.any():
        num_nans = nan_mask.sum().item()
        num_infs = inf_mask.sum().item()
        print(f"⚠️ Warning: Replacing {num_nans} NaNs and {num_infs} Infs with 0.")

    tensor = torch.where(nan_mask | inf_mask, torch.zeros_like(tensor), tensor)
    return tensor


def _to_valid_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array, replacing NaNs and Infs with 0.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Valid NumPy array.
    """
    tensor = _replace_nans_and_infs(tensor)
    return tensor.detach().cpu().numpy()
