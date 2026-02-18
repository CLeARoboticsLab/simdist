"""Script to aggregate real-world data into a single file which can then be
used to create training datasets."""

import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.paths import get_aggregate_realworld_data_hydra_config
from simdist.data.episode_aggregator import EpisodeAggregator


@hydra.main(**get_aggregate_realworld_data_hydra_config())
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    episode_aggregator = EpisodeAggregator(dict_cfg)
    episode_aggregator.process_all_episodes()


if __name__ == "__main__":
    main()
