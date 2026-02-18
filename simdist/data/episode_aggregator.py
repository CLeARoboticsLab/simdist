import os

from tqdm import tqdm
import json
import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from simdist.utils import paths
from simdist.data import REAL_DATA_KEYS


class EpisodeAggregator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        dataset_name = cfg["dataset_name"]
        self.raw_dir = paths.get_real_raw_data_dir(dataset_name)
        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.raw_dir}")
        self.episode_paths = self._find_all_episodes()

        self._dataset_file_handler = HDF5DatasetFileHandler()
        raw_data_file_path = paths.get_raw_data_path(dataset_name)
        self.dir = os.path.dirname(raw_data_file_path)
        self._dataset_file_handler.create(raw_data_file_path)
        self._episodes = {}

    def _find_all_episodes(self):
        """Recursively find all .hdf5 files under the dataset directory."""
        hdf5_paths = []
        for dirpath, _, filenames in os.walk(self.raw_dir):
            for filename in filenames:
                if filename.endswith(".hdf5"):
                    hdf5_paths.append(os.path.join(dirpath, filename))
        return sorted(hdf5_paths, key=os.path.getmtime)

    def process_all_episodes(self):
        total_steps = 0
        for path in tqdm(self.episode_paths):
            print(f"Processing {path}")
            last_ep_len = self.process_episode(path)
            total_steps += last_ep_len
        print(f"Total steps: {total_steps}")
        print(f"Total episodes: {len(self.episode_paths)}")
        metrics = {
            "total_episodes": len(self.episode_paths),
            "total_steps": total_steps,
            "control_rate_hz": self.cfg["control_rate"],
            "total_duration_min": total_steps / self.cfg["control_rate"] / 60.0,
        }
        metrics_path = os.path.join(self.dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self._dataset_file_handler.close()
        return last_ep_len

    def process_episode(self, file_path: str):
        ep_data = EpisodeData()
        ep_len = 0
        with h5py.File(file_path, "r") as f:
            for key in REAL_DATA_KEYS.values():
                if key in f:
                    data = f[key][:]
                    ep_len = data.shape[0]
                    first_sample = data[0]
                    if (
                        key == REAL_DATA_KEYS["actions"]
                        or key == REAL_DATA_KEYS["commands"]
                    ):
                        first_sample = np.zeros_like(first_sample)
                    first_sample = first_sample[None, ...].repeat(
                        self.cfg["beg_padding"], axis=0
                    )
                    data = np.concatenate([first_sample, data], axis=0)
                    ep_data.add(key, torch.tensor(data))
                    ep_data.data[key] = ep_data.data[key].squeeze()
                else:
                    print(f"{key}: NOT FOUND")

        self._dataset_file_handler.write_episode(ep_data)
        self._dataset_file_handler.flush()
        return ep_len
