import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

REL_CONFIG_PATH = "../config"


@pytest.fixture
def cfg(request):
    model_name = request.param
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="train_model",
            overrides=[f"model={model_name}"],
        )
        return OmegaConf.to_container(cfg, resolve=True)
