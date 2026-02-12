"""Shared helper functions for tests."""

import jax.numpy as jnp

from simdist.modeling import types
from simdist.utils import config


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
