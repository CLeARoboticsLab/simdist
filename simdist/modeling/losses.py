from typing import Tuple, Any

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from simdist.modeling import types, models
from simdist.utils import registry


Losses = dict[str, jnp.ndarray]
ExtraInfo = dict[str, Any]
_LOSS_REGISTRY: registry.Registry["Loss"] = registry.Registry("Loss")


def register_loss(name: str):
    return _LOSS_REGISTRY.register(name)


def get_loss(cfg: dict) -> "Loss":
    loss_name = cfg["loss"]["type"]
    return _LOSS_REGISTRY.create(loss_name, cfg)


class Loss:
    def __init__(self, cfg: dict):
        self.loss_cfg: dict = cfg["loss"]
        self.weights: dict = self.loss_cfg["weights"]

    @property
    def loss_terms(self) -> list[str]:
        return list(self.weights.keys())

    def __call__(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.TrainingLabels,
        deterministic: bool,
        aux_losses: dict = {},
        **kwargs,
    ) -> Tuple[jnp.ndarray, Losses]:
        losses, _ = self.compute_losses(model, x, y, deterministic, **kwargs)
        losses.update(aux_losses)
        assert set(losses.keys()) == set(self.loss_terms), (
            "Losses do not match weights. "
            f"Got {losses.keys()} losses and {self.weights.keys()} weights. "
            "Check the loss config file and add or remove weight terms as needed."
        )
        weighted_losses = {k: self.weights[k] * losses[k] for k in self.loss_terms}
        loss = jnp.sum(weighted_losses.values())
        return loss, weighted_losses

    def compute_losses(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.TrainingLabels,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[Losses, ExtraInfo]:
        """Returns a dict of losses (shape (B,)) with keys matching the weights in
        the config file, and a dict of extra info"""
        raise NotImplementedError("Must implement compute_losses method")


@register_loss("world_model")
class WorldModelLoss(Loss):
    def compute_losses(
        self,
        model: models.WorldModelBase,
        x: types.WorldModelSchema.Inputs,
        y: types.WorldModelSchema.Labels,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[Losses, ExtraInfo]:
        # Compute the predicted outputs
        y_pred = model(x, deterministic=deterministic)

        # Get the scaler parameters to automatically scale losses
        scaler_params = model.get_scaler_params()

        # Encode future obs for latent dynamics loss
        fut_latents = jax.lax.stop_gradient(
            model.encode_latent(y["proprio_obs"], y["extero_obs"], deterministic=True)
        )

        latents_error = y_pred["latents"] - fut_latents
        latent_dynamics_loss = jnp.mean(latents_error**2)

        reward_error = y_pred["rewards"] - y["rewards"]
        reward_error /= scaler_params["rewards"]
        reward_loss = jnp.mean(reward_error**2)

        value_error = y_pred["values"] - y["values"]
        value_error /= scaler_params["values"]
        value_loss = jnp.mean(value_error**2)

        # set loss to zero for all actions after the expert policy isn't used
        act_loss_mask = jnp.cumprod(y["exp_pol_flags"].astype(jnp.int32), axis=-1)
        act_loss_mask = act_loss_mask[:, :, None]

        action_error = y_pred["actions"] - y["actions"]
        action_error /= scaler_params["actions"]
        action_loss = jnp.mean((action_error * act_loss_mask) ** 2)

        losses = {
            "latent_dynamics": latent_dynamics_loss,
            "reward": reward_loss,
            "value": value_loss,
            "action": action_loss,
        }

        return losses, {}
