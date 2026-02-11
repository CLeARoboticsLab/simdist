import flax.nnx as nnx

from simdist.modeling import types, scaler
from simdist.utils import config


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


class WorldModel(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.proprio_obs_dim = config.proprio_obs_dim_from_sys_config(self.sys_cfg)
        self.extero_obs_dim = config.extero_obs_dim_from_sys_config(self.sys_cfg)
        self.latent_dim = self.model_cfg["latent_dim"]

        # input processing
        self.scaler = scaler.Scaler(
            scaler_params,
            types.WorldModelSchema.scaler_params_mapping,
        )
