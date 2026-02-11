from typing import Union, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import simdist.modeling.types as types


class Scaler(nnx.Module):
    def __init__(
        self,
        scaler_params: types.ScalerParams,
        scaler_params_mapping: types.ScalerParamsMapping,
        subset_mapping: dict[str, jnp.ndarray] | None = None,
    ):
        self.scaler_params = jax.tree.map(nnx.Variable, scaler_params)
        self.scaler_params_mapping = scaler_params_mapping
        self.subset_mapping = subset_mapping
        self.scale_fn = lambda value, mean, std: (value - mean) / (std + 1e-8)
        self.unscale_fn = lambda value, mean, std: value * std + mean

    def __call__(
        self, x: Union[types.ModelInputs, types.ModelOutputs]
    ) -> Union[types.ModelInputs, types.ModelOutputs]:
        raise NotImplementedError(
            "Scaler should not be called directly. Use scale() or unscale() methods."
        )

    def scale(
        self, x: Union[types.ModelInputs, types.ModelOutputs]
    ) -> Union[types.ModelInputs, types.ModelOutputs]:
        """
        Scale the input using the scaler parameters.
        """
        return self._transform(x, self.scale_fn)

    def unscale(
        self, x: Union[types.ModelInputs, types.ModelOutputs]
    ) -> Union[types.ModelInputs, types.ModelOutputs]:
        """
        Unscale the input using the scaler parameters.
        """
        return self._transform(x, self.unscale_fn)

    def _transform(
        self, x: Union[types.ModelInputs, types.ModelOutputs], func: Callable
    ) -> Union[types.ModelInputs, types.ModelOutputs]:
        transformed_x = {}
        for key, value in x.items():
            if key in self.scaler_params_mapping:
                param_key = self.scaler_params_mapping[key]
                mean = self.scaler_params[param_key]["mean"]
                std = self.scaler_params[param_key]["std"]
                if self.subset_mapping and key in self.subset_mapping:
                    mean = mean[self.subset_mapping[key]]
                    std = std[self.subset_mapping[key]]
                transformed_x[key] = func(value, mean, std)
            else:
                transformed_x[key] = value
        return transformed_x
