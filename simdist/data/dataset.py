from typing import TypedDict, Any
import os

from torch.utils.data import Dataset
import h5py
import numpy as np

from simdist.modeling import types
from simdist.utils import paths, config
from simdist.data import DATA_KEY


_DATASET_REGISTRY: dict[str, "DatasetBase"] = {}


def register_dataset(name: str):
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(cfg: dict) -> "DatasetBase":
    dataset_name = cfg["model"]["dataset"]["type"]
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Registered: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[dataset_name](cfg)


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
        self._data_dir = ""

    def eval(self) -> None:
        self._eval_mode = True

    def train(self) -> None:
        self._eval_mode = False

    def __len__(self) -> int:
        raise NotImplementedError("This method must be implemented.")

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError("This method must be implemented.")

    @property
    def data_dir(self) -> str:
        return self._data_dir


@register_dataset("world_model")
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
        self._data_dir = paths.get_processed_data_dir(
            dataset_name, sys_name, self.H, self.T
        )
        if not os.path.exists(self._data_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self._data_dir}' does not exist."
                "Did you remember to run scripts/process_data.py?"
            )
        self._load_files(self._data_dir)
        self._load_data()

    def __len__(self) -> int:
        return len(self._data["start_idxs"])

    def __getitem__(self, idx: int) -> DatasetItem:
        t_H = self._data["start_idxs"][idx]
        item = self.get_item_by_t_H(t_H)
        if not self._eval_mode:
            item = self.data_augmentations(item)
        return item

    def get_item_by_t_H(self, t_H: int) -> DatasetItem:
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
        fut_cmds: np.ndarray = self._data["commands"][t:t_T]
        model_inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": proprio_obs_hist,
            "extero_obs": extero_obs,
            "acts_hist": acts_hist,
            "fut_acts": fut_acts,
            "fut_cmds": fut_cmds,
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

    def data_augmentations(self, item: DatasetItem) -> DatasetItem:
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

    @staticmethod
    def get_dummy_item(cfg: dict) -> DatasetItem:
        sys_cfg = cfg["system"]
        proprio_obs_dim = config.proprio_obs_dim_from_sys_config(sys_cfg)
        extero_obs_dim = config.extero_obs_dim_from_sys_config(sys_cfg)
        act_dim = config.action_dim_from_sys_config(sys_cfg)
        cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)
        H = cfg["model"]["dataset"]["history_length"]
        T = cfg["model"]["dataset"]["prediction_length"]

        rng = np.random.default_rng()

        inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": rng.standard_normal((H + 1, proprio_obs_dim)),
            "extero_obs": rng.standard_normal((extero_obs_dim,)),
            "acts_hist": rng.standard_normal((H, act_dim)),
            "fut_acts": rng.standard_normal((T, act_dim)),
            "fut_cmds": rng.standard_normal((T + 1, cmd_dim)),
        }

        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": rng.standard_normal((T, proprio_obs_dim)),
            "extero_obs": rng.standard_normal((T, extero_obs_dim)),
            "rewards": rng.standard_normal((T,)),
            "values": rng.standard_normal((T,)),
            "actions": rng.standard_normal((T, act_dim)),
            "exp_pol_flags": np.ones((T,)),
        }

        return {
            "model_in": inputs,
            "labels": labels,
            "metadata": {"dummy": True},
        }


@register_dataset("quadruped_world_model")
class QuadrupedWorldModelDataset(WorldModelDatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sys_cfg = self.cfg["system"]
        aug_cfg = self.cfg["model"]["dataset"]["augmentations"]
        self.add_noise = aug_cfg["add_noise"]

        if not self.add_noise:
            return

        # Determine noise to add to proprioceptive observations
        proprio_ob_dims = [ob["dim"] for ob in sys_cfg["proprio_obs"]["types"]]
        self.proprio_obs_dim = sum(proprio_ob_dims)
        proprio_obs_noises = [ob["noise"] for ob in sys_cfg["proprio_obs"]["types"]]
        self.proprio_obs_noise_stds = []
        for dim, noise in zip(proprio_ob_dims, proprio_obs_noises):
            self.proprio_obs_noise_stds.extend([noise] * dim)
        self.proprio_obs_noise_stds = np.array(self.proprio_obs_noise_stds)

        # Determine noise to add to the height_scan
        self.height_noise = None
        for ob in sys_cfg["extero_obs"]["types"]:
            if ob["name"] == "height_scan":
                self.height_noise = ob["noise"]
                break
        if self.height_noise is None:
            raise ValueError("Height scan noise not found")

    def data_augmentations(self, item: DatasetItem) -> DatasetItem:
        if not self.add_noise:
            return item

        # Apply noise to proprioceptive observations
        item["model_in"]["proprio_obs_hist"] = _apply_noise(
            item["model_in"]["proprio_obs_hist"], self.proprio_obs_noise_stds
        )

        # Apply noise to height scan
        item["model_in"]["extero_obs"]["height_scan"] = _apply_noise(
            item["model_in"]["extero_obs"]["height_scan"], self.height_noise
        )

        return item


def _apply_noise(arr: np.ndarray, std: float | np.ndarray) -> np.ndarray:
    noise = np.random.randn(*arr.shape)
    return arr + noise * std
