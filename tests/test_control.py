import os

import pytest
import yaml
import numpy as np
import time
import hydra
from omegaconf import OmegaConf, DictConfig


from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

from simdist.control.controller_base import ControllerInput
from simdist.control.mppi import MppiController
from simdist.utils import paths
import helpers


NUM_STEPS = 100


@pytest.mark.parametrize(
    "cfg",
    helpers.WORLD_MODEL_CONFIGS,
    indirect=True,
)
def test_mppi_controller(cfg: dict):
    # Create the controller
    model = helpers.make_dummy_model(cfg)
    model_type = cfg["model"]["type"]
    config_dir = paths.get_control_config_dir()
    config_file = os.path.join(config_dir, "mppi.yaml")
    with open(config_file, "r") as f:
        ctrl_cfg = yaml.safe_load(f)
    controller = MppiController(model, cfg, ctrl_cfg)

    # Make dummy inputs
    model_in = helpers.make_dummy_world_model_input(cfg, 1)
    ctrl_in: ControllerInput = {
        "proprio_obs": model_in["proprio_obs_hist"][0, 0],
        "extero_obs": model_in["extero_obs"][0],
        "command": model_in["fut_cmds"][0, 0],
        "prev_action": model_in["acts_hist"][0, 0],
    }
    ctrl_in: ControllerInput = {k: np.array(v) for k, v in ctrl_in.items()}
    cmd = ctrl_in["command"]

    controller.initialize(ctrl_in, cmd)

    step_times = []
    start_time = time.perf_counter()
    for _ in range(NUM_STEPS):
        step_start_time = time.perf_counter()
        controller.set_fut_cmd(cmd)
        controller.update(ctrl_in)
        _ = controller.run_control()
        step_time = time.perf_counter() - step_start_time
        step_times.append(step_time)
    time_per_step = (time.perf_counter() - start_time) / NUM_STEPS
    print(f"Maximum controller frequency for {model_type}: {1 / time_per_step:.2f} Hz")
    print(f"Min step time (ms): {min(step_times) * 1000:.2f}")
    print(f"Max step time (ms): {max(step_times) * 1000:.2f}")


@hydra.main(**paths.get_train_model_hydra_config())
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    test_mppi_controller(cfg)


if __name__ == "__main__":
    main()
