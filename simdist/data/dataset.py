from typing import TypedDict, Any
import os

from torch.utils.data import Dataset
import h5py
import numpy as np

from simdist.modeling import types
from simdist.utils import paths
from simdist.data import DATA_KEY


class DatasetItem(TypedDict):
    model_in: types.ModelInputs
    labels: types.TrainingLabels
    metadata: dict[str, Any]


class DatasetBatch(TypedDict):
    model_in: types.ModelInputs
    labels: types.TrainingLabels


class DatasetBase(Dataset):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._eval_mode = False

    def eval(self) -> None:
        self._eval_mode = True

    def train(self) -> None:
        self._eval_mode = False

    @property
    def _can_add_noise(self) -> bool:
        return not self._eval_mode

    def __len__(self) -> int:
        raise NotImplementedError("This method must be implemented.")

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError("This method must be implemented.")


class WorldModelDatasetBase(DatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        dataset_name = cfg["data"]["dataset_name"]
        sys_name = cfg["system"]["name"]
        self.H = cfg["model"]["dataset"]["history_length"]
        self.T = cfg["model"]["dataset"]["prediction_length"]

        self._files: dict[str, h5py.File] = {}
        self._data: dict[str, h5py.Dataset] = {}

        # Load data
        data_dir = paths.get_data_dir(dataset_name, sys_name, self.H, self.T)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dataset directory '{data_dir}' does not exist."
                "Did you remember to run scripts/process_data.py?"
            )
        self._load_files(data_dir)
        self._load_data()

    def __len__(self) -> int:
        return len(self._data["start_idxs"])

    def __getitem__(self, idx: int) -> DatasetItem:
        t_H = self._data["start_idxs"][idx]
        return self._get_item_by_t_H(t_H)

    def _get_item_by_t_H(self, t_H: int) -> DatasetItem:
        # metadata to be passed to subclasses
        t = t_H + self.H
        t_T = t + self.T
        metadata = {
            "t_H": t_H,
            "t": t,
            "t_T": t_T,
        }

        # inputs
        proprio_obs_hist: np.ndarray = self._data["proprio_obs"][t_H : t + 1]
        extero_obs: np.ndarray = self._data["extero_obs"][t]
        acts_hist: np.ndarray = self._data["actions"][t_H:t]
        fut_acts: np.ndarray = self._data["actions"][t:t_T]
        fut_cmd: np.ndarray = self._data["commands"][t:t_T]
        model_inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": proprio_obs_hist,
            "extero_obs": extero_obs,
            "acts_hist": acts_hist,
            "fut_acts": fut_acts,
            "fut_cmd": fut_cmd,
        }

        # labels
        proprio_obs: np.ndarray = self._data["proprio_obs"][t + 1 : t_T + 1]
        extero_obs: np.ndarray = self._data["extero_obs"][t + 1 : t_T + 1]
        rewards: np.ndarray = self._data["rewards"][t:t_T]
        values: np.ndarray = self._data["values"][t + 1 : t_T + 1]
        actions: np.ndarray = self._data["actions"][t:t_T]
        exp_pol_flags: np.ndarray = self._data["exp_pol_flags"][t:t_T]
        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "rewards": rewards,
            "values": values,
            "actions": actions,
            "exp_pol_flags": exp_pol_flags,
        }

        item: DatasetItem = {
            "model_in": model_inputs,
            "labels": labels,
            "metadata": metadata,
        }
        return item

    def _load_files(self, data_dir: str) -> None:
        self._files["start_idxs"] = h5py.File(paths.get_start_idxs_path(data_dir))
        self._files["proprio_obs"] = h5py.File(paths.get_proprio_obs_path(data_dir))
        self._files["extero_obs"] = h5py.File(paths.get_extero_obs_path(data_dir))
        self._files["actions"] = h5py.File(paths.get_actions_path(data_dir))
        self._files["commands"] = h5py.File(paths.get_commands_path(data_dir))
        self._files["rewards"] = h5py.File(paths.get_rewards_path(data_dir))
        self._files["values"] = h5py.File(paths.get_values_path(data_dir))
        self._files["exp_pol_flags"] = h5py.File(
            paths.get_expert_policy_flags_path(data_dir)
        )

    def _load_data(self) -> None:
        for key, file in self._files.items():
            self._data[key] = file[DATA_KEY]
