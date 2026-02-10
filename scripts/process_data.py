"""Script to merge generated data into a dataset and compute scaler params."""

import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.paths import get_process_data_hydra_config
from simdist.data.data_processor import DataProcessor


@hydra.main(**get_process_data_hydra_config())
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    data_processor = DataProcessor(dict_cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
