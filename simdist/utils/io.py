import os
import json

from simdist.utils import paths
from simdist.modeling import types


def save_scaler_params(scaler_params: types.ScalerParams, save_dir: str):
    """
    Save scaler parameters to a JSON file in the specified dataset directory.

    Args:
        scaler_params (types.ScalerParams): Scaler parameters to save.
        save_dir (str): Directory where the scaler parameters will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    scaler_params_path = os.path.join(save_dir, paths.get_scaler_params_filename())
    with open(scaler_params_path, "w") as f:
        json.dump(scaler_params, f, indent=4)
