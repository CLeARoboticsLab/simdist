from __future__ import annotations
from typing import TypedDict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import jax.numpy as jnp


ArrayLike = Union["np.ndarray", "jnp.ndarray"]


class Stats(TypedDict):
    mean: ArrayLike
    std: ArrayLike


class ScalerParams(TypedDict):
    proprio_obs: Stats
    extero_obs: Stats
    actions: Stats
    commands: Stats
    rewards: Stats
    values: Stats


ModelInputs = dict[str, ArrayLike]
ModelOutputs = dict[str, ArrayLike]
TrainingLabels = dict[str, ArrayLike]
ModelData = ModelInputs | ModelOutputs | TrainingLabels

ScalerParamsMapping = dict[str, str]


class WorldModelSchema:
    class Inputs(TypedDict):
        proprio_obs_hist: ArrayLike  # proprio history from t-H to t (..., H+1, o_p_dim)
        extero_obs: ArrayLike  # last extero obs (..., o_e_dim)
        acts_hist: ArrayLike  # action history from t-H to t-1 (..., H, act_dim)
        fut_acts: ArrayLike  # future actions to apply from t to t+T-1 (..., T, act_dim)
        fut_cmds: ArrayLike  # future commands from t to t+T (..., T+1, cmd_dim)

    class Outputs(TypedDict):
        latents: ArrayLike  # future latents from t+1 to t+T (..., T, lat_dim)
        rewards: ArrayLike  # future rewards from t to t+T-1 (..., T)
        values: ArrayLike  # future values from t+1 to t+T (..., T)
        actions: ArrayLike  # future actions from t to t+T-1 (..., T, act_dim)

    class Labels(TypedDict):
        proprio_obs: ArrayLike  # future proprio from t+1 to t+T (..., T, o_p_dim)
        extero_obs: ArrayLike  # future extero from t+1 to t+T (..., T, o_e_dim)
        rewards: ArrayLike  # future rewards from t to t+T-1 (..., T)
        values: ArrayLike  # future values from t+1 to t+T (..., T)
        actions: ArrayLike  # future actions from t to t+T-1 (..., T, act_dim)
        exp_pol_flags: ArrayLike  # true when act from expert policy; t to T-1 (..., T)

    class Encoding(TypedDict):
        history: ArrayLike  # encoded history of observations and actions
        latent: ArrayLike  # encoded latent representation (from last observation)

    scaler_params_mapping: ScalerParamsMapping = {
        "proprio_obs_hist": "proprio_obs",
        "extero_obs": "extero_obs",
        "acts_hist": "actions",
        "fut_acts": "actions",
        "fut_cmds": "commands",
        "proprio_obs": "proprio_obs",
        "extero_obs": "extero_obs",
        "rewards": "rewards",
        "values": "values",
        "actions": "actions",
    }
