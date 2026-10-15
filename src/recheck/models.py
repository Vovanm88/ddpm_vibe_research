"""Small unconditional velocity networks and an independent MNIST evaluator.

All generator timesteps use the experiment's normalized coordinate: zero is
data and one is noise. A fixed sinusoidal frequency scale is internal to the
embedding; there is no conditioning on the integration step or a class label.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn
from torch.nn import functional as F


@dataclass
class VelocityOutput(BaseOutput):
    sample: torch.Tensor


def normalized_time(timestep, sample: torch.Tensor) -> torch.Tensor:
    """Expand one shared time or accept per-example times, without rescaling it."""
    time = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
    if time.numel() == 1:
        time = time.reshape(1).expand(len(sample))
    if time.shape != (len(sample),):
        raise ValueError("timestep must be a scalar or a tensor of shape [batch]")
    if not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
        raise ValueError("normalized timestep must lie in [0, 1]")
    return time


def sinusoidal_time(time: torch.Tensor, width: int, scale: float = 1000.0):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = time.float()[:, None] * scale * frequencies[None]
    embedded = torch.cat((phase.sin(), phase.cos()), dim=1)
    return F.pad(embedded, (0, width - embedded.shape[1])).to(time.dtype)


class FMMLP(ModelMixin, ConfigMixin):
    """Three-hidden-layer toy velocity model; labels never enter its forward."""

    @register_to_config
    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        num_layers=3,
        time_embedding_dim=32,
        time_scale=1000.0,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) < 1 or time_embedding_dim < 2:
            raise ValueError("positive dimensions/layers and time embedding >= 2 required")
        self.sample_shape = (input_dim,)
        layers = []
        width = input_dim + time_embedding_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, sample, timestep, return_dict=True):
        if sample.ndim != 2 or sample.shape[1:] != self.sample_shape:
            raise ValueError(f"sample must have shape [batch, {self.sample_shape[0]}]")
        time = normalized_time(timestep, sample)
        embedded = sinusoidal_time(time, self.config.time_embedding_dim, self.config.time_scale)
        output = self.net(torch.cat((sample, embedded), dim=1))
        return VelocityOutput(sample=output) if return_dict else (output,)


def _groups(channels):
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, sample, embedding):
        hidden = self.conv1(F.silu(self.norm1(sample)))
        hidden = hidden + self.time_projection(F.silu(embedding))[:, :, None, None]
        return self.shortcut(sample) + self.conv2(F.silu(self.norm2(hidden)))


class SmallUNetModel(ModelMixin, ConfigMixin):
    """Residual 28→14→7 U-Net, two residual blocks per resolution and path."""

    @register_to_config
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        time_embedding_dim=128,
        time_sinusoidal_dim=32,
        time_scale=1000.0,
    ):
        super().__init__()
        if min(in_channels, base_channels, time_embedding_dim) < 1 or time_sinusoidal_dim < 2:
            raise ValueError("invalid channel or embedding dimensions")
        self.sample_shape = (in_channels, 28, 28)
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_sinusoidal_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, c, 3, padding=1)

        def block(a, b):
            return TimeResidualBlock(a, b, time_embedding_dim)

        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleList([block(c, c), block(c, c)]),
                nn.ModuleList([block(c * 2, c * 2), block(c * 2, c * 2)]),
                nn.ModuleList([block(c * 4, c * 4), block(c * 4, c * 4)]),
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.Conv2d(c * 4, c * 2, 3, padding=1),
                nn.Conv2d(c * 2, c, 3, padding=1),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleList([block(c * 4, c * 2), block(c * 2, c * 2)]),
                nn.ModuleList([block(c * 2, c), block(c, c)]),
            ]
        )
        self.output_norm = nn.GroupNorm(_groups(c), c)
        self.output_conv = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, sample, timestep, return_dict=True):
        if sample.ndim != 4 or tuple(sample.shape[1:]) != self.sample_shape:
            raise ValueError(f"sample must have shape [batch, {self.sample_shape}]")
        time = normalized_time(timestep, sample)
        embedding = self.time_mlp(
            sinusoidal_time(time, self.config.time_sinusoidal_dim, self.config.time_scale)
        )
        hidden = self.input_conv(sample)
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, embedding)
            if level < 2:
                skips.append(hidden)
                hidden = self.downsamples[level](hidden)
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            hidden = upsample(F.interpolate(hidden, size=skip.shape[-2:], mode="nearest"))
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, embedding)
        output = self.output_conv(F.silu(self.output_norm(hidden)))
        return VelocityOutput(sample=output) if return_dict else (output,)


class EvaluatorCNN(ModelMixin, ConfigMixin):
    """Independent real-data classifier with the fixed post-activation FC128 features."""

    @register_to_config
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        if feature_dim != 128 or num_classes != 10:
            raise ValueError("the registered MNIST evaluator has 128 features and 10 classes")
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)
        self.logits = nn.Linear(feature_dim, num_classes)

    def features(self, pixel01):
        if pixel01.ndim != 4 or tuple(pixel01.shape[1:]) != (1, 28, 28):
            raise ValueError("MNIST evaluator expects [batch, 1, 28, 28]")
        return F.relu(self.fc(self.convs(pixel01).flatten(1)))

    def forward(self, pixel01, return_features=False):
        features = self.features(pixel01)
        logits = self.logits(features)
        return (logits, features) if return_features else logits


def state_dict_sha256(model_or_state) -> str:
    """Hash the actual named tensors, independently of serialization/container metadata."""
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def architecture_manifest(model):
    return {
        "class": type(model).__name__,
        "config": dict(model.config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "flops_estimate": None,
        "flops_note": "not measured; no FLOP counter is installed by the experiment",
    }
