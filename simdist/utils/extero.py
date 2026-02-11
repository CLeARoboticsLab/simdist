from functools import partial, reduce

import jax.numpy as jnp
import jax


@partial(jax.jit, static_argnums=(1, 2))
def restore_height_map(flattened_map: jnp.ndarray, h_l: int, h_w: int) -> jnp.ndarray:
    leading_dims = len(flattened_map.shape) - 1
    restored = flattened_map.reshape(flattened_map.shape[:-1] + (h_w, h_l))
    return vmap_over_leading_dims(
        lambda x: jnp.flipud(jnp.fliplr(x)).T, restored, leading_dims
    )


@jax.jit
def flatten_height_map(restored_map: jnp.ndarray) -> jnp.ndarray:
    leading_dims = len(restored_map.shape) - 2
    h_l, h_w = restored_map.shape[-2], restored_map.shape[-1]
    flipped = vmap_over_leading_dims(
        lambda x: jnp.fliplr(jnp.flipud(x.T)), restored_map, leading_dims
    )
    return flipped.reshape(restored_map.shape[:-2] + (h_l * h_w,))


def vmap_over_leading_dims(fn, x, num_leading_dims):
    # Apply vmap iteratively over the specified number of leading dimensions
    return reduce(lambda f, _: jax.vmap(f), range(num_leading_dims), fn)(x)
