"""Shared helper functions for tests."""

import jax.numpy as jnp
from flax import nnx

from simdist.modeling import types, models
from simdist.utils import config, model as model_utils
from simdist.data.dataset import WorldModelDatasetBase

WORLD_MODEL_CONFIGS = [
    "quadruped_world_model",
]


def make_dummy_scaler_params(cfg: dict) -> types.ScalerParams:
    sys_cfg = cfg["system"]
    proprio_obs_dim = config.proprio_obs_dim_from_sys_config(sys_cfg)
    extero_obs_dim = config.extero_obs_dim_from_sys_config(sys_cfg)
    act_dim = config.action_dim_from_sys_config(sys_cfg)
    cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)

    return {
        "proprio_obs": {
            "mean": jnp.zeros(proprio_obs_dim),
            "std": jnp.ones(proprio_obs_dim),
        },
        "extero_obs": {
            "mean": jnp.zeros(extero_obs_dim),
            "std": jnp.ones(extero_obs_dim),
        },
        "actions": {"mean": jnp.zeros(act_dim), "std": jnp.ones(act_dim)},
        "commands": {"mean": jnp.zeros(cmd_dim), "std": jnp.ones(cmd_dim)},
        "rewards": {"mean": jnp.zeros(1), "std": jnp.ones(1)},
        "values": {"mean": jnp.zeros(1), "std": jnp.ones(1)},
    }


def make_dummy_model(cfg: dict) -> models.ModelBase:
    scaler_params = make_dummy_scaler_params(cfg)
    model = models.get_model(cfg, scaler_params, nnx.Rngs(0))
    return model


def make_dummy_world_model_input(
    cfg: dict, batch_size: int
) -> types.WorldModelSchema.Inputs:
    item = WorldModelDatasetBase.get_dummy_item(cfg)
    item.pop("metadata")
    item = model_utils.dataset_batch_to_jax(item)
    item = model_utils.repeat_along_batch_dim(item, batch_size)
    model_in = item["model_in"]
    return model_in
