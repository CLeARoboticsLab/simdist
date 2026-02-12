"""Script to train the world model"""

import hydra
from omegaconf import DictConfig, OmegaConf

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

from simdist.utils.paths import get_train_model_hydra_config
from simdist.modeling import trainer


@hydra.main(**get_train_model_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    trainer.train(dict_cfg)


if __name__ == "__main__":
    main()
