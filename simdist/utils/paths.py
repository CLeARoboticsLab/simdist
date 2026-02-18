import os
import re
from pathlib import Path

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
SIMDIST_MODULE_PATH = os.path.dirname(THIS_MODULE_PATH)
SIMDIST_ROOT_PATH = os.path.dirname(SIMDIST_MODULE_PATH)
_PATHS = {
    "RL_CHECKPOINTS": os.path.join(SIMDIST_ROOT_PATH, "checkpoints", "rl"),
    "CONFIG": os.path.join(SIMDIST_ROOT_PATH, "config"),
    "ALL_DATASETS": os.path.join(SIMDIST_ROOT_PATH, "datasets"),
    "SIM_DATASETS": os.path.join(SIMDIST_ROOT_PATH, "datasets", "sim"),
    "REAL_DATASETS": os.path.join(SIMDIST_ROOT_PATH, "datasets", "real"),
    "MODEL_CHECKPOINTS": os.path.join(SIMDIST_ROOT_PATH, "checkpoints", "models"),
}
_FILENAMES = {
    "RAW_DATA_FILE_NAME": "raw_data.hdf5",
    "PROCESSED_DATA_DIR_NAME": "processed_data_{}_H-{}_T-{}",
    "START_IDXS_FILE_NAME": "start_idxs.hdf5",
    "PROPRIO_OBS_FILE_NAME": "proprio_obs.hdf5",
    "EXTERO_OBS_FILE_NAME": "extero_obs.hdf5",
    "ACTIONS_FILE_NAME": "acts.hdf5",
    "COMMANDS_FILE_NAME": "cmds.hdf5",
    "REWARDS_FILE_NAME": "rewards.hdf5",
    "VALUES_FILE_NAME": "values.hdf5",
    "EXPERT_POLICY_FLAG_FILE_NAME": "expert_policy_flag.hdf5",
    "SCALER_PARAMS_FILE_NAME": "scaler_params.json",
    "MODEL_CONFIG_FILE_NAME": "model_config.yaml",
}


def get_rl_checkpoint_dir():
    """Get the directory for RL checkpoints."""
    return _PATHS["RL_CHECKPOINTS"]


def get_rl_run_dir(rl_run: str):
    """Get the directory for a specific RL run."""
    return os.path.join(get_rl_checkpoint_dir(), rl_run)


def get_rl_policies_dir(rl_run: str):
    """Get the directory for RL policies."""
    return os.path.join(get_rl_run_dir(rl_run), "policies")


def get_rl_critics_dir(rl_run: str):
    """Get the directory for RL critics."""
    return os.path.join(get_rl_run_dir(rl_run), "critics")


def get_highest_numbered_file(folder_path, prefix, suffix):
    """
    Returns the file with the highest number in the given folder.

    Args:
        folder_path (str): Path to the folder containing the files.
        prefix (str): The prefix of the file names (e.g., "critic" or another
        identifier).
        suffix (str): The suffix of the file names (e.g., ".pt", ".onnx").

    Returns:
        str or None: The filename with the highest number, or None if no matching files
        are found.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}")

    numbered_files = []
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            numbered_files.append((int(match.group(1)), filename))

    return max(numbered_files, key=lambda x: x[0])[1] if numbered_files else None


def find_folders(root_dir, folder_name) -> list[str]:
    """Find all folders with a specific name under a root directory."""
    path_root = Path(root_dir)
    return [str(p) for p in path_root.rglob(folder_name) if p.is_dir()]


def get_config_dir():
    """Get the directory for configuration files."""
    return _PATHS["CONFIG"]


def get_generate_data_hydra_config():
    return {"config_path": get_config_dir(), "config_name": "generate_data"}


def get_process_data_hydra_config():
    return {"config_path": get_config_dir(), "config_name": "process_data"}


def get_train_model_hydra_config():
    return {"config_path": get_config_dir(), "config_name": "train_model"}


def get_simulate_go2_hydra_config():
    return {"config_path": get_config_dir(), "config_name": "simulate_go2"}


def get_aggregate_realworld_data_hydra_config():
    return {"config_path": get_config_dir(), "config_name": "aggregate_realworld_data"}


def get_sim_dataset_dir(dataset_name: str):
    """Get the directory for a specific simulation dataset."""
    return os.path.join(_PATHS["SIM_DATASETS"], dataset_name)


def get_real_dataset_dir(dataset_name: str):
    """Get the directory for a specific real dataset."""
    return os.path.join(_PATHS["REAL_DATASETS"], dataset_name)


def get_raw_data_filename():
    """Get the filename for the raw simulation data."""
    return _FILENAMES["RAW_DATA_FILE_NAME"]


def get_dataset_dir(dataset_name: str):
    """Get the directory for a specific dataset."""
    dirs = find_folders(_PATHS["ALL_DATASETS"], dataset_name)
    if len(dirs) == 0:
        raise FileNotFoundError(f"Dataset directory not found for: {dataset_name}")
    elif len(dirs) > 1:
        raise FileExistsError(
            f"Multiple dataset directories found for: {dataset_name}"
            f"Found: {dirs}"
            "Please rename or remove the duplicates."
        )
    return dirs[0]


def get_raw_data_path(dataset_name: str):
    """Get the path for the raw data file."""
    return os.path.join(get_dataset_dir(dataset_name), _FILENAMES["RAW_DATA_FILE_NAME"])


def get_real_raw_data_dir(dataset_name: str):
    """Get the directory for the real raw data."""
    return os.path.join(get_real_dataset_dir(dataset_name), "raw")


def get_processed_data_dir(
    dataset_name: str, system_name: str, history_length: int, prediction_length: int
):
    """Get the directory for processed data."""
    return os.path.join(
        get_dataset_dir(dataset_name),
        _FILENAMES["PROCESSED_DATA_DIR_NAME"].format(
            system_name, history_length, prediction_length
        ),
    )


def get_start_idxs_path(processed_data_dir: str):
    """Get the path for the start indices file."""
    return os.path.join(processed_data_dir, _FILENAMES["START_IDXS_FILE_NAME"])


def get_proprio_obs_path(processed_data_dir: str):
    """Get the path for the proprioceptive observations file."""
    return os.path.join(processed_data_dir, _FILENAMES["PROPRIO_OBS_FILE_NAME"])


def get_extero_obs_path(processed_data_dir: str):
    """Get the path for the exteroceptive observations file."""
    return os.path.join(processed_data_dir, _FILENAMES["EXTERO_OBS_FILE_NAME"])


def get_actions_path(processed_data_dir: str):
    """Get the path for the actions file."""
    return os.path.join(processed_data_dir, _FILENAMES["ACTIONS_FILE_NAME"])


def get_commands_path(processed_data_dir: str):
    """Get the path for the commands file."""
    return os.path.join(processed_data_dir, _FILENAMES["COMMANDS_FILE_NAME"])


def get_rewards_path(processed_data_dir: str):
    """Get the path for the rewards file."""
    return os.path.join(processed_data_dir, _FILENAMES["REWARDS_FILE_NAME"])


def get_values_path(processed_data_dir: str):
    """Get the path for the values file."""
    return os.path.join(processed_data_dir, _FILENAMES["VALUES_FILE_NAME"])


def get_expert_policy_flags_path(processed_data_dir: str):
    """Get the path for the expert policy flags file."""
    return os.path.join(processed_data_dir, _FILENAMES["EXPERT_POLICY_FLAG_FILE_NAME"])


def get_scaler_params_filename():
    """Get the filename for the scaler parameters file."""
    return _FILENAMES["SCALER_PARAMS_FILE_NAME"]


def get_model_config_filename():
    """Get the filename for the model config file."""
    return _FILENAMES["MODEL_CONFIG_FILE_NAME"]


def get_model_checkpoints_dir():
    """Get the directory for model checkpoints."""
    return _PATHS["MODEL_CHECKPOINTS"]


def get_control_config_dir():
    return os.path.join(_PATHS["CONFIG"], "control")
