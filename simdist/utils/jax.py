"""Utilities for configuring JAX runtime behavior."""

import getpass
import os


_JAX_CACHE_CONFIGURED = False


def configure_jax_compilation_cache(cache_root: str = "/tmp") -> None:
    """Configure JAX persistent compilation cache settings.

    Call this very early in a script, before any other module initializes JAX.
    """

    global _JAX_CACHE_CONFIGURED
    if _JAX_CACHE_CONFIGURED:
        return

    import jax

    user = os.environ.get("USER", getpass.getuser())
    jax.config.update("jax_compilation_cache_dir", f"{cache_root}/{user}/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    _JAX_CACHE_CONFIGURED = True
