"""Dataset-independent fixed Fourier features for a Cartesian FM velocity MLP.

This is an architecture variant, not an analytical teacher or a correction to
generated points. It contains no ray count, labels, radial target constraint or
output projection onto a curve. Polar harmonics are integer powers of a smooth
normalized (x,y), computed entirely with real operations for CPU/CUDA/MPS.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from torch import nn

from .models import VelocityOutput, normalized_time, sinusoidal_time


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def real_complex_power(real, imaginary, exponent):
    """Compute (real+i*imaginary)**n by real binary exponentiation.

    n is a fixed positive integer. In particular, no atan2/complex tensor occurs
    in the autograd graph, and the harmonics smoothly vanish at the origin.
    """
    _positive_integer(exponent, "exponent")
    result_real, result_imaginary = torch.ones_like(real), torch.zeros_like(imaginary)
    base_real, base_imaginary = real, imaginary
    remaining = int(exponent)
    while remaining:
        if remaining & 1:
            result_real, result_imaginary = (
                result_real * base_real - result_imaginary * base_imaginary,
                result_real * base_imaginary + result_imaginary * base_real,
            )
        remaining >>= 1
        if remaining:
            base_real, base_imaginary = (
                base_real.square() - base_imaginary.square(),
                2 * base_real * base_imaginary,
            )
    return result_real, result_imaginary


class FixedSpatialEncoder(nn.Module):
    """Raw coordinates, fixed Cartesian bands, smooth radius, and polar harmonics.

    Cartesian features have raw-coordinate periods 1/f. Polar harmonics use the
    same frequency list with amplitude (r/sqrt(r²+epsilon²))**f. Thus values and
    derivatives remain finite at zero, and no data-dependent frequency fitting
    or implicit knowledge of an environment enters the encoder.
    """

    def __init__(
        self,
        input_dim=2,
        spatial_frequencies=(1, 2, 4, 8, 16, 32),
        polar_features=True,
        epsilon=0.001,
    ):
        super().__init__()
        _positive_integer(input_dim, "input_dim")
        frequencies = tuple(spatial_frequencies)
        if not frequencies:
            raise ValueError("spatial_frequencies must be a nonempty sequence")
        for frequency in frequencies:
            _positive_integer(frequency, "every spatial frequency")
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("spatial_frequencies must be distinct")
        if not isinstance(polar_features, bool):
            raise ValueError("polar_features must be a bool")
        if polar_features and input_dim != 2:
            raise ValueError(
                "polar features require input_dim=2; disable them for other dimensions"
            )
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.input_dim = int(input_dim)
        self.frequency_powers = tuple(int(frequency) for frequency in frequencies)
        self.polar_features, self.epsilon = polar_features, float(epsilon)
        self.register_buffer(
            "frequencies", torch.tensor(self.frequency_powers, dtype=torch.float32)
        )
        self.output_dim = self.input_dim * (1 + 2 * len(frequencies)) + 2
        if polar_features:
            self.output_dim += 2 * len(frequencies)

    def forward(self, sample):
        if sample.ndim != 2 or sample.shape[1] != self.input_dim:
            raise ValueError(f"spatial encoder expects [batch,{self.input_dim}]")
        # Match dtype/device after model.to(...), including float64 CPU gradcheck.
        frequencies = self.frequencies.to(dtype=sample.dtype, device=sample.device)
        phase = 2 * math.pi * sample[:, :, None] * frequencies[None, None, :]
        radial_squared = sample.square().sum(dim=1, keepdim=True)
        denominator = torch.sqrt(radial_squared + self.epsilon**2)
        smooth_radius = denominator - self.epsilon
        parts = [
            sample,
            phase.sin().flatten(1),
            phase.cos().flatten(1),
            smooth_radius,
            radial_squared,
        ]
        if self.polar_features:
            unit_x = sample[:, :1] / denominator
            unit_y = sample[:, 1:2] / denominator
            for frequency in self.frequency_powers:
                real, imaginary = real_complex_power(unit_x, unit_y, frequency)
                parts.extend((real, imaginary))
        return torch.cat(parts, dim=1)


class FourierFMMLP(ModelMixin, ConfigMixin):
    """Plain SiLU MLP with fixed spatial features and normalized FM time input."""

    @register_to_config
    def __init__(
        self,
        input_dim=2,
        hidden_dim=256,
        num_layers=4,
        time_embedding_dim=32,
        time_scale=1000.0,
        spatial_frequencies=(1, 2, 4, 8, 16, 32),
        polar_features=True,
        epsilon=0.001,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("num_layers", num_layers),
            ("time_embedding_dim", time_embedding_dim),
        ):
            _positive_integer(value, name)
        if time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be >=2")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        self.sample_shape = (int(input_dim),)
        self.spatial_encoder = FixedSpatialEncoder(
            input_dim, spatial_frequencies, polar_features, epsilon
        )
        width = self.spatial_encoder.output_dim + time_embedding_dim
        layers = []
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, sample, timestep, return_dict=True):
        if sample.ndim != 2 or tuple(sample.shape[1:]) != self.sample_shape:
            raise ValueError(f"sample must have shape [batch,{self.sample_shape[0]}]")
        time = normalized_time(timestep, sample)
        temporal = sinusoidal_time(time, self.config.time_embedding_dim, self.config.time_scale)
        encoded = self.spatial_encoder(sample)
        velocity = self.net(torch.cat((encoded, temporal), dim=1))
        return VelocityOutput(sample=velocity) if return_dict else (velocity,)
