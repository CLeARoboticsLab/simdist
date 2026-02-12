from typing import Any, Tuple
import os

from omegaconf import OmegaConf, DictConfig
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import numpy as np
import orbax.checkpoint as ocp

from simdist.data.dataset import DatasetBatch
from simdist.utils import paths
from simdist.modeling import models


def load_model_from_ckpt(
    ckpt_dir: str,
    step: int | None = None,  # if none, load the latest
) -> Tuple[nnx.Module, DictConfig, int]:
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist. ")

    model_cfg = OmegaConf.load(
        os.path.join(ckpt_dir, paths.get_model_config_filename())
    )
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    dummy_scaler_params = make_dummy_scaler_params(model_cfg)
    model = models.get_model(model_cfg, dummy_scaler_params, nnx.Rngs(0))
    graphdef, model_state = nnx.split(model)

    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as mngr:
        if step is None:
            step = mngr.latest_step()
        restored_pure_dict = mngr.restore(
            step,
            args=ocp.args.StandardRestore(item=model_state.to_pure_dict()),
        )

    model_state.replace_by_pure_dict(restored_pure_dict)
    model = nnx.merge(graphdef, model_state)

    return model, model_cfg, step


def make_dummy_scaler_params(cfg: dict):
    struct = cfg["scaler_params_struct"]
    dummy_scalar_params = {}
    for k, dim in struct.items():
        dummy_scalar_params[k] = {
            "mean": jnp.zeros(dim, dtype=jnp.float32),
            "std": jnp.ones(dim, dtype=jnp.float32),
        }
    return dummy_scalar_params


def dataset_batch_to_jax(batch: DatasetBatch) -> DatasetBatch:
    return _numpy_dict_to_jax(batch)


def repeat_along_batch_dim(x: jnp.ndarray | dict[str, jnp.ndarray], B: int):
    """Repeat the input array or dict of arrays along the batch dimension B."""
    return jax.tree.map(lambda y: jnp.tile(y[None, ...], (B,) + (1,) * y.ndim), x)


def create_param_filter(param_names: list[str]):
    filters = []
    for path in param_names:
        names = path.split("/")
        f = []
        for name in names:
            if name.isdigit():
                name = int(name)
            f.append(nnx.PathContains(name))
        f = tuple(f)
        f = nnx.All(nnx.Param, nnx.All(*f))
        filters.append(f)
    return nnx.Any(*tuple(filters))


def count_params(model: nnx.Module, filter=None):
    full_state = nnx.state(model)
    if filter is None:
        sel_state = full_state
    else:
        sel_state = full_state.filter(filter)

    def _leaf_numel(x: Any) -> int:
        # Handle JAX/NumPy arrays and Python scalars uniformly.
        try:
            return int(np.size(x))
        except Exception:
            return 0

    leaves = jax.tree_util.tree_leaves(sel_state)
    return int(sum(_leaf_numel(x) for x in leaves))


def _numpy_dict_to_jax(numpy_dict: dict[Any, np.ndarray]) -> dict[Any, jnp.ndarray]:
    """Convert a dict of numpy arrays to a dict of JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x), numpy_dict)
