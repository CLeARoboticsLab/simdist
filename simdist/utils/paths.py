import os

THIS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
SIMDIST_MODULE_PATH = os.path.dirname(THIS_MODULE_PATH)
SIMDIST_ROOT_PATH = os.path.dirname(SIMDIST_MODULE_PATH)
print(f"[INFO] SimDist root path: {SIMDIST_ROOT_PATH}")

_PATHS = {
    "RL_CHECKPOINTS": os.path.join(SIMDIST_ROOT_PATH, "checkpoints", "rl"),
}


def get_rl_checkpoint_dir():
    """Get the directory for RL checkpoints."""
    return _PATHS["RL_CHECKPOINTS"]
