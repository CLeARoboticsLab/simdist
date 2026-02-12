from typing import Any

import jax.numpy as jnp
import jax
import numpy as np

from simdist.data.dataset import DatasetBatch


def dataset_batch_to_jax(batch: DatasetBatch) -> DatasetBatch:
    return _numpy_dict_to_jax(batch)


def repeat_along_batch_dim(x: jnp.ndarray | dict[str, jnp.ndarray], B: int):
    """Repeat the input array or dict of arrays along the batch dimension B."""
    return jax.tree.map(lambda y: jnp.tile(y[None, ...], (B,) + (1,) * y.ndim), x)


def _numpy_dict_to_jax(numpy_dict: dict[Any, np.ndarray]) -> dict[Any, jnp.ndarray]:
    """Convert a dict of numpy arrays to a dict of JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x), numpy_dict)
