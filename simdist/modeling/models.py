import flax.nnx as nnx
import jax.numpy as jnp

from simdist.modeling import types, scaler, encoders, modules
from simdist.utils import config


_MODEL_REGISTRY: dict[str, "ModelBase"] = {}


def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(
    cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs
) -> "ModelBase":
    model_name = cfg["model"]["name"]
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Registered: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](cfg, scaler_params, rngs)


class ModelBase(nnx.Module):
    def __init__(
        self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs, **kwargs
    ):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.sys_cfg = cfg["system"]

    def __call__(
        self,
        x: types.ModelInputs,
        deterministic: bool | None = None,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement __call__ method")

    def inference(
        self,
        x: types.ModelInputs,
        **kwargs,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement inference method")


class WorldModelBase(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.proprio_obs_dim = config.proprio_obs_dim_from_sys_config(self.sys_cfg)
        self.extero_obs_dim = config.extero_obs_dim_from_sys_config(self.sys_cfg)
        self.action_dim = config.action_dim_from_sys_config(self.sys_cfg)
        self.cmd_dim = config.cmd_dim_from_sys_config(self.sys_cfg)
        self.latent_dim = self.model_cfg["latent_dim"]
        self.mlp_dropout_rate = self.model_cfg["dropout"]["mlp"]
        self.attention_dropout_rate = self.model_cfg["dropout"]["attention"]
        T = self.model_cfg["dataset"]["prediction_length"]
        emb_cfg = self.model_cfg["embedding"]
        emb_mlp_hsf = emb_cfg["mlp_hidden_size_factor"]
        emb_hidden_size = emb_mlp_hsf * self.latent_dim

        # input processing
        self.scaler = scaler.Scaler(
            scaler_params,
            types.WorldModelSchema.scaler_params_mapping,
        )
        self.encoder = encoders.WorldModelEncoderBase(cfg, rngs)
        self.fut_acts_embed = modules.Embedding(
            seq_len=T,
            input_dim=self.action_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["future_acts_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.fut_cmds_embed = modules.Embedding(
            seq_len=T,
            input_dim=self.action_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["future_cmds_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )

        # dynamics
        attn_cfg = self.model_cfg["dynamics"]["attention"]
        self.dynamics = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )

        # reward head
        attn_cfg = self.model_cfg["reward"]["attention"]
        dec_cfg = self.model_cfg["reward"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.reward_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.action_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["reward_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.reward = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.reward_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        # value head
        attn_cfg = self.model_cfg["value"]["attention"]
        dec_cfg = self.model_cfg["value"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.value_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["value_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.value = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.value_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        # policy head
        attn_cfg = self.model_cfg["policy"]["attention"]
        dec_cfg = self.model_cfg["policy"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.policy = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.policy_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=self.action_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Outputs:

        # pre-processing, encoding, and embedding
        x = self.scaler(x)
        encoding = self.encoder(x, deterministic=deterministic)
        fut_acts_emb = self.fut_acts_embed(x["fut_acts"], deterministic=deterministic)
        fut_cmds_emb = self.fut_cmds_embed(
            x["fut_cmds"][:, :-1], deterministic=deterministic
        )

        # concatenate latent to the end of the history encoding
        latent_enc = jnp.expand_dims(encoding["latent"], axis=1)
        context = jnp.concatenate([encoding["history"], latent_enc], axis=1)

        # dynamics
        latents = self.dynamics(fut_acts_emb, context, deterministic=deterministic)

        # reward head
        # concatenate last latent with latent prediction
        z_t_tm1 = jnp.concatenate((encoding["latent"], latents[:, :-1]), axis=1)
        # concatenate future actions and commands to latents
        rew_in = jnp.concatenate(
            (z_t_tm1, x["fut_acts"], x["fut_cmds"][:, :-1]), axis=-1
        )
        # embedding
        rew_in_emb = self.reward_emb(rew_in, deterministic=deterministic)
        # prediction
        rew_pred = self.reward(rew_in_emb, deterministic=deterministic)
        rewards = self.reward_dec(rew_pred, deterministic=deterministic).squeeze()

        # value head
        # concatenate latent prediction with future commands
        value_in = jnp.concatenate((latents, x["fut_cmds"][:, 1:]), axis=-1)
        # embedding
        value_in_emb = self.value_emb(value_in, deterministic=deterministic)
        # prediction
        value_pred = self.value(value_in_emb, deterministic=deterministic)
        values = self.value_dec(value_pred, deterministic=deterministic).squeeze()

        # policy head
        latent_acts_pred = self.policy(
            fut_cmds_emb, context, deterministic=deterministic
        )
        actions = self.policy_dec(latent_acts_pred, deterministic=deterministic)

        return {
            "latents": latents,
            "rewards": rewards,
            "values": values,
            "actions": actions,
        }

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """For consistency loss"""
        return self.encoder.encode_latent(
            proprio_obs, extero_obs, deterministic=deterministic
        )


@register_model("quadruped_world_model")
class QuadrupedWorldModel(WorldModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.encoder = encoders.QuadrupedEncoder(cfg, scaler_params, rngs)
