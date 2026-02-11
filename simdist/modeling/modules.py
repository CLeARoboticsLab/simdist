from typing import List
import math

import flax.nnx as nnx
import jax.numpy as jnp
import jax


class EmptyModule(nnx.Module):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, **kwargs):
        return x


class TransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = None,
        **kwargs,
    ):
        self.mask = mask

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            decode=False,
            deterministic=deterministic,
            rngs=rngs,
        )

        self.dense_proj = MLP(
            input_dim=embed_dim,
            hidden_dims=[mlp_hidden_dim],
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=mlp_dropout_rate,
            deterministic=deterministic,
        )

        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, mask=None, deterministic: bool | None = None):
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        elif self.mask == "causal" or mask == "causal":
            padding_mask = nnx.make_causal_mask(inputs[:, :, 0])
        else:
            padding_mask = None

        attention_output = self.attention(
            inputs_q=inputs,
            inputs_k=inputs,
            inputs_v=inputs,
            mask=padding_mask,
            decode=False,
            deterministic=deterministic,
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input, deterministic=deterministic)
        return self.layernorm_2(proj_input + proj_output)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = None,
        **kwargs,
    ):
        self.layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                deterministic=deterministic,
                mask=mask,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None, deterministic: bool | None = None):
        for layer in self.layers:
            x = layer(x, mask=mask, deterministic=deterministic)
        return x


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = "causal",
        **kwargs,
    ):
        self.mask = mask

        self.attention_1 = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            deterministic=deterministic,
            decode=False,
            rngs=rngs,
        )
        self.attention_2 = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            deterministic=deterministic,
            decode=False,
            rngs=rngs,
        )

        self.dense_proj = MLP(
            input_dim=embed_dim,
            hidden_dims=[mlp_hidden_dim],
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=mlp_dropout_rate,
            deterministic=deterministic,
        )
        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_3 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(
        self,
        inputs,
        encoder_outputs,
        mask=None,
        deterministic: bool | None = None,
    ):
        if self.mask == "causal":
            causal_mask = nnx.make_causal_mask(inputs[:, :, 0])
        else:
            causal_mask = None

        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            inputs_q=inputs,
            inputs_v=inputs,
            inputs_k=inputs,
            mask=causal_mask,
            deterministic=deterministic,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            inputs_q=out_1,
            inputs_v=encoder_outputs,
            inputs_k=encoder_outputs,
            mask=padding_mask,
            deterministic=deterministic,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2, deterministic=deterministic)
        return self.layernorm_3(out_2 + proj_output)


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = "causal",
        **kwargs,
    ):
        self.layers = [
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                deterministic=deterministic,
                mask=mask,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self, x, encoder_outputs, mask=None, deterministic: bool | None = None
    ):
        for layer in self.layers:
            x = layer(x, encoder_outputs, mask=mask, deterministic=deterministic)
        return x


class Embedding(nnx.Module):
    """Projection plus positional encoding"""

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        hidden_dims: List[int],
        embed_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
    ):

        self.enc = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
        )

        self.pos_emb = nnx.Param(
            jax.random.normal(rngs["params"](), (seq_len, embed_dim))
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool | None = None):
        x = self.enc(x, deterministic=deterministic)
        x += self.pos_emb
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
    ):
        self.layers: List[nnx.Linear] = []
        self.dropout_layers: List[nnx.Dropout] = []
        prev_dim = input_dim  # Track input size for each layer
        self.output_dim = output_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                nnx.Linear(in_features=prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            prev_dim = hidden_dim
            if i < len(hidden_dims) - 1:
                self.dropout_layers.append(
                    nnx.Dropout(
                        rate=dropout_rate, deterministic=deterministic, rngs=rngs
                    )
                )

        self.output_layer = nnx.Linear(
            in_features=prev_dim, out_features=output_dim, rngs=rngs
        )

    def __call__(self, x, deterministic: bool | None = None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = nnx.gelu(x)
            if i < len(self.layers) - 1:
                x = self.dropout_layers[i](x, deterministic=deterministic)
        x = self.output_layer(x)
        return x


class CNN(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        strides: List[int],
        latent_dim: int,
        kernel_size: int,
        ht_in: int,
        wd_in: int,
        rngs: nnx.Rngs,
    ):
        self.strides = strides
        self.latent_dim = latent_dim
        self.ht_in = ht_in
        self.wd_in = wd_in

        # Define convolutional layers
        self.layers = []
        prev_channels = in_channels  # Track the number of input channels

        for out_channels, stride in zip(features, strides):
            self.layers.append(
                nnx.Conv(
                    in_features=prev_channels,
                    out_features=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    rngs=rngs,
                )
            )
            prev_channels = out_channels  # Update input channels for next layer

        # Final Conv layer (latent space)
        self.output_layer = nnx.Conv(
            in_features=prev_channels,
            out_features=latent_dim,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides[-1], strides[-1]),
            rngs=rngs,
        )

    @property
    def output_shape(self):
        """Computes the final spatial size after convolutions."""
        ht, wd = self._compute_conv_output_size()
        return ht, wd, self.latent_dim

    def __call__(self, x):
        """Forward pass through CNN."""
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)

        x = self.output_layer(x)  # Final Conv layer (latent representation)
        return x

    def _compute_conv_output_size(self):
        """Computes output width and length after convolutions."""
        ht, wd = self.ht_in, self.wd_in
        for s in self.strides:
            ht = math.ceil(ht / s)
            wd = math.ceil(wd / s)
        return ht, wd


class TransposeCNN(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        strides: List[int],
        output_dim: int,
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        self.strides = strides

        self.layers = []
        prev_channels = in_channels  # Track the number of input channels

        for out_channels, stride in zip(features, strides):
            self.layers.append(
                nnx.ConvTranspose(
                    in_features=prev_channels,
                    out_features=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    rngs=rngs,
                )
            )
            prev_channels = out_channels  # Update input channels for next layer

        # Final layer
        self.output_layer = nnx.ConvTranspose(
            in_features=prev_channels,
            out_features=output_dim,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides[-1], strides[-1]),
            rngs=rngs,
        )

    def __call__(self, x):
        """Forward pass through TCNN."""
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)

        x = self.output_layer(x)
        return x
