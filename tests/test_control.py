import pytest

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

from simdist.control.mppi import MppiController
import helpers


@pytest.mark.parametrize(
    "cfg",
    helpers.WORLD_MODEL_CONFIGS,
    indirect=True,
)
def test_mppi_controller(cfg: dict):
    model = helpers.make_dummy_model(cfg)
    model_in = helpers.make_dummy_world_model_input(cfg, 1)
    controller = MppiController(model, cfg)
