from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp

from simdist.modeling import models
from simdist.utils import config


class ControllerInput(TypedDict):
    proprio_obs: np.ndarray
    extero_obs: np.ndarray
    command: np.ndarray
    prev_action: np.ndarray


class ControllerOutput(TypedDict):
    actions: jnp.ndarray


class ControllerBase:
    def __init__(self, model: models.ModelBase, model_cfg: dict, *args, **kwargs):
        self.model = model
        self.model_cfg = model_cfg
        self.sys_cfg = model_cfg["system"]
        self.H = config.history_length_from_config(model_cfg)
        self.T = config.prediction_length_from_config(model_cfg)

        if "buffer_length" in kwargs:
            self.buffer_length = kwargs["buffer_length"]
        else:
            self.buffer_length = self.H + 1

        self.buf = None
        self.initialized = False
        self.is_initing = False
        self.device = (
            jax.devices("gpu")[0]
            if any(d.platform == "gpu" for d in jax.devices())
            else jax.devices()[0]
        )
