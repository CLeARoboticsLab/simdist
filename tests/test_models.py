import time

import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf
import jax.numpy as jnp
import jax
from flax import nnx

from simdist.data.dataset import WorldModelDatasetBase
from simdist.utils import model as model_utils, config
from simdist.modeling import types, models

REL_CONFIG_PATH = "../config"


@pytest.fixture
def cfg(request):
    model_name = request.param
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"model={model_name}"],
        )
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return cfg


@pytest.mark.parametrize(
    "cfg",
    [
        "quadruped_world_model",
    ],
    indirect=True,
)
def test_world_models(cfg: dict):
    B = 512
    num_inferences = 100

    # create model
    scaler_params = _make_dummy_scaler_params(cfg)
    model = models.get_model(cfg, scaler_params, nnx.Rngs(0))
    model_inf_fn = jax.jit(model.inference)

    # create dummy model input
    item = WorldModelDatasetBase.get_dummy_item(cfg)
    item.pop("metadata")
    item = model_utils.dataset_batch_to_jax(item)
    item = model_utils.repeat_along_batch_dim(item, B)
    model_in = item["model_in"]

    # run inference
    _ = model_inf_fn(model_in)  # warm up
    start_time = time.perf_counter()
    for _ in range(num_inferences):
        _ = jax.block_until_ready(model_inf_fn(model_in))
    end_time = time.perf_counter()
    print(f"Average inference rate: {num_inferences / (end_time - start_time):.2f} Hz")


def _make_dummy_scaler_params(cfg: dict) -> types.ScalerParams:
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
