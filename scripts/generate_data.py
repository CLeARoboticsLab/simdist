"""Script to generate data using Isaac Sim."""

print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.paths import get_generate_data_hydra_config
from simdist.data.data_recorder import DataRecorder


@hydra.main(**get_generate_data_hydra_config())
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    data_recorder = DataRecorder(simulation_app, dict_cfg)
    data_recorder.run()


if __name__ == "__main__":
    main()
