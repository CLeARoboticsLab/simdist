import functools

import flax
from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np

from simdist.control.controller_base import (
    ControllerBase,
    ControllerInput,
    ControllerOutput,
)
from simdist.modeling import models, types
from simdist.utils.model import repeat_along_batch_dim


@flax.struct.dataclass
class MppiState:
    mean: jnp.ndarray
    std: jnp.ndarray
    key_data: jnp.ndarray


class MppiController(ControllerBase):
    def __init__(
        self,
        model: models.WorldModelBase,
        model_cfg: dict,
        controller_cfg: dict,
        *args,
        **kwargs,
    ):
        super().__init__(model, model_cfg, controller_cfg, *args, **kwargs)

        self.mppi_state = None
        self.shift = self.ctrl_cfg["delay"]
        assert self.shift > 0
        self.discounts = self.ctrl_cfg["discount"] ** jnp.arange(self.T)
        self.rngs = nnx.Rngs(self.ctrl_cfg["seed"])
        self.dummy_actions = jnp.zeros((self.T, self.act_dim))

    def reset(self, x: ControllerInput, cmd: np.ndarray):
        super().reset(x, cmd)
        self.mppi_state = MppiState(
            mean=self._to_dev_f32(jnp.zeros((self.T, self.act_dim))),
            std=self._to_dev_f32(
                jnp.ones((self.T, self.act_dim)) * self.ctrl_cfg["init_std"]
            ),
            key_data=self._to_key_data(self.rngs()),
        )

    def run_control(self) -> ControllerOutput:
        model_inputs = self._make_model_inputs(self.dummy_actions)
        base_actions = self._get_base_policy_actions(model_inputs)

        # TODO

    def _make_model_inputs(
        self, fut_acts: jnp.ndarray
    ) -> types.WorldModelSchema.Inputs:
        """
        Create model inputs for the world model. fut_acts should be shape (T, act_dim)
        """
        with self.buf_lock:
            hist: ControllerInput = self.buf.get()
            model_inputs: types.WorldModelSchema.Inputs = {
                "proprio_obs_hist": hist["proprio_obs"][-(self.H + 1) :],
                "extero_obs": hist["extero_obs"][-1],
                "acts_hist": hist["prev_action"][-self.H :],
                "fut_acts": fut_acts,
                "fut_cmds": self.fut_cmds,
            }
            model_inputs = jax.tree.map(jnp.asarray, model_inputs)
        return model_inputs

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _get_base_policy_actions(
        self, model_inputs: types.WorldModelSchema.Inputs
    ) -> jnp.ndarray:
        model_inputs = repeat_along_batch_dim(model_inputs, 1)
        model_outputs = self.model.inference(model_inputs)
        return model_outputs["actions"][0]


# TODO: don't need command hist
