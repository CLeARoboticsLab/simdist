from typing import TypedDict
import threading

import numpy as np
import jax
import jax.numpy as jnp

from simdist.modeling import models
from simdist.utils import config, buffer


class ControllerInput(TypedDict):
    proprio_obs: np.ndarray
    extero_obs: np.ndarray
    command: np.ndarray
    prev_action: np.ndarray


class ControllerOutput(TypedDict):
    actions: jnp.ndarray


class ControllerBase:
    def __init__(
        self,
        model: models.ModelBase,
        model_cfg: dict,
        controller_cfg: dict,
        *args,
        **kwargs,
    ):
        self.model = model
        self.model_cfg = model_cfg
        self.sys_cfg = model_cfg["system"]
        self.ctrl_cfg = controller_cfg
        self.H = config.history_length_from_config(model_cfg)
        self.T = config.prediction_length_from_config(model_cfg)
        self.act_dim = config.action_dim_from_sys_config(self.sys_cfg)
        self.fut_cmds_len = self.T + 1

        if "buffer_length" in kwargs:
            self.buffer_length = kwargs["buffer_length"]
        else:
            self.buffer_length = self.H + 1

        self.buf = None
        self.fut_cmds = None
        self.initialized = False
        self.is_initing = False
        self.device = (
            jax.devices("gpu")[0]
            if any(d.platform == "gpu" for d in jax.devices())
            else jax.devices()[0]
        )
        self.buf_lock = threading.Lock()

    def reset(self, x: ControllerInput, cmd: np.ndarray):
        """Resets the controller by filling the buffer with the given input and
        command"""
        with self.buf_lock:
            self.buf.fill(x)
            self.set_fut_cmd(cmd)

    def initialize(self, x: ControllerInput, cmd: np.ndarray):
        """Initializes the controller by filling buffers with the given
        input and command, and jitting all operations. Must be called before
        run_control()
        """
        self.buf = buffer.MultiRingBuffer(maxlen=self.buffer_length, example_item=x)
        self.reset(x, cmd)

        # initialze the controller
        cls_name = self.__class__.__name__
        print(f"Initializing {cls_name}...", flush=True)
        self.is_initing = True
        _ = self._init_fn()
        self.initialized = True
        self.is_initing = False
        print(f"{cls_name} initialized.", flush=True)

    def set_fut_cmd(self, fut_cmd: np.ndarray):
        """Updates the future commands for the controller. Here we assume that the
        future command remains constant for the entire prediction horizon."""
        self.fut_cmds = np.tile(fut_cmd, (self.fut_cmds_len, 1))

    def update(self, x: ControllerInput):
        """Updates the controller's buffer; should be called every dt."""
        with self.buf_lock:
            self.buf.append(x)

    def run_control(self) -> ControllerOutput:
        """Returns the actions for the given the measurements in the buffer."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_fn(self):
        """Initializes the controller by calling the main control method once.
        Doing this JITs the function so that future calls are fast."""
        self.run_control()

    def _to_dev_f32(self, x):
        return jax.device_put(jnp.asarray(x, jnp.float32), self.device)

    def _to_dev_u32(self, x):
        return jax.device_put(jnp.asarray(x, jnp.uint32), self.device)

    def _to_key_data(self, key):
        return self._to_dev_u32(jax.random.key_data(key))

    def _to_key(self, key_data):
        return jax.random.wrap_key_data(key_data)
