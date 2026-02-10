import os
import re

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
SIMDIST_MODULE_PATH = os.path.dirname(THIS_MODULE_PATH)
SIMDIST_ROOT_PATH = os.path.dirname(SIMDIST_MODULE_PATH)
_PATHS = {
    "RL_CHECKPOINTS": os.path.join(SIMDIST_ROOT_PATH, "checkpoints", "rl"),
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
        prefix (str): The prefix of the file names (e.g., "critic" or another identifier).
        suffix (str): The suffix of the file names (e.g., ".pt", ".onnx").

    Returns:
        str or None: The filename with the highest number, or None if no matching files are found.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}")

    numbered_files = []
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            numbered_files.append((int(match.group(1)), filename))

    return max(numbered_files, key=lambda x: x[0])[1] if numbered_files else None
