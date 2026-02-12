import time

import pytest

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax

import helpers

@pytest.mark.parametrize(
    "cfg",
    helpers.WORLD_MODEL_CONFIGS,
    indirect=True,
)
def test_world_models(cfg: dict):
    B = 512
    num_inferences = 100

    # create model
    model = helpers.make_dummy_model(cfg)
    model_inf_fn = jax.jit(model.inference)

    # create dummy model input
    model_in = helpers.make_dummy_world_model_input(cfg, B)

    # run inference
    _ = model_inf_fn(model_in)  # warm up
    start_time = time.perf_counter()
    for _ in range(num_inferences):
        _ = jax.block_until_ready(model_inf_fn(model_in))
    end_time = time.perf_counter()
    print(f"Average inference rate: {num_inferences / (end_time - start_time):.2f} Hz")
