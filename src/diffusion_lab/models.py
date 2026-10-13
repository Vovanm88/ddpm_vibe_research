"""Diffusers-compatible denoisers with explicit point / trajectory shapes."""

import math
from dataclasses import dataclass

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn

from .conditioning import CONDITIONERS


@dataclass
class DenoiserOutput(BaseOutput):
    sample: torch.Tensor


def sinusoidal_embedding(timesteps, width):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    phases = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat((phases.sin(), phases.cos()), dim=-1)
    return torch.nn.functional.pad(result, (0, width - result.shape[-1]))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(width, 2 * width)
        self.net = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def forward(self, sample, context):
        scale, shift = self.modulation(context).chunk(2, dim=-1)
        return sample + self.net(self.norm(sample) * (1 + scale) + shift)


class PointDenoiser(ModelMixin, ConfigMixin):
    """All parameters exist at construction; batches never act as attention tokens.

    MLP handles arbitrary fixed sample shapes. Transformer attends over positions
    within each [L,D] trajectory, or one token per point for [D] data.
    """

    @register_to_config
    def __init__(
        self,
        sample_shape=(2,),
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        backbone="mlp",
        conditioner="one_hot",
        num_classes=2,
        condition_dim=0,
    ):
        super().__init__()
        sample_shape = tuple(sample_shape)
        if len(sample_shape) not in (1, 2) or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in sample_shape
        ):
            raise ValueError("sample_shape must be (D,) or (L,D), with positive integers")
        if hidden_dim < 4 or num_layers < 1:
            raise ValueError("hidden_dim >= 4 and num_layers >= 1 are required")
        if backbone not in ("mlp", "transformer"):
            raise ValueError(f"Unknown backbone: {backbone}")
        self.sample_shape = sample_shape
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        if conditioner == "none":
            self.condition_encoder = None
        else:
            if conditioner not in CONDITIONERS:
                raise ValueError(f"Unknown conditioner: {conditioner}")
            self.condition_encoder = CONDITIONERS[conditioner](
                hidden_dim, num_classes, condition_dim
            )
        if backbone == "mlp":
            size = math.prod(sample_shape)
            self.input_projection = nn.Linear(size, hidden_dim)
            self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
            self.output_projection = nn.Linear(hidden_dim, size)
        else:
            if num_heads < 1 or hidden_dim % num_heads:
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.input_projection = nn.Linear(sample_shape[-1], hidden_dim)
            layer = nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                4 * hidden_dim,
                dropout=0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            self.output_projection = nn.Linear(hidden_dim, sample_shape[-1])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sample, timestep, conditions=None, condition_mask=None, return_dict=True):
        if tuple(sample.shape[1:]) != self.sample_shape:
            raise ValueError(f"Expected [B,{self.sample_shape}], received {tuple(sample.shape)}")
        batch = len(sample)
        time = torch.as_tensor(timestep, device=sample.device)
        if time.ndim == 0 or time.numel() == 1:
            time = time.reshape(1).expand(batch)
        if time.shape != (batch,) or not torch.isfinite(time).all() or torch.any(time < 0):
            raise ValueError("timestep must be finite nonnegative scalar or [B]")
        context = self.time_encoder(
            sinusoidal_embedding(time, self.config.hidden_dim).to(sample.dtype)
        )
        encoded = self.null_condition.expand(batch, -1)
        if conditions is not None and self.condition_encoder is not None:
            if len(conditions) != batch:
                raise ValueError("Condition and sample batch sizes must match")
            conditioned = self.condition_encoder(conditions.to(sample.device))
            if condition_mask is None:
                encoded = conditioned
            else:
                mask = torch.as_tensor(condition_mask, device=sample.device, dtype=torch.bool)
                if mask.shape != (batch,):
                    raise ValueError("condition_mask must have shape [B]; True keeps condition")
                encoded = torch.where(mask[:, None], conditioned, encoded)
        context = context + encoded
        if self.config.backbone == "mlp":
            hidden = self.input_projection(sample.flatten(1))
            for block in self.blocks:
                hidden = block(hidden, context)
            predicted = self.output_projection(self.output_norm(hidden)).reshape_as(sample)
        else:
            tokens = sample.reshape(batch, -1, self.sample_shape[-1])
            hidden = self.input_projection(tokens)
            positions = sinusoidal_embedding(
                torch.arange(tokens.shape[1], device=sample.device), self.config.hidden_dim
            )
            hidden = self.transformer(hidden + positions[None].to(sample.dtype) + context[:, None])
            predicted = self.output_projection(self.output_norm(hidden)).reshape_as(sample)
        return DenoiserOutput(sample=predicted) if return_dict else (predicted,)
