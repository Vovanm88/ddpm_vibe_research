"""Native interval-conditioned velocities for Shortcut and MeanFlow experiments.

Time is normalized: data is at zero, noise at one. The model predicts an
average velocity over ``[t - step_size, t]``; sampling subtracts that velocity
times ``step_size``. Both label and continuous conditioners are inherited from
PointDenoiser. Allowing a null condition is an API choice, not evidence that the
null branch has been trained.
"""

import math

import torch
from diffusers.configuration_utils import register_to_config
from torch import nn

from .models import DenoiserOutput, PointDenoiser, sinusoidal_embedding

_BASE_KEYS = (
    "sample_shape",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "backbone",
    "conditioner",
    "num_classes",
    "condition_dim",
)


def _batch_time(value, sample, name):
    result = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if result.ndim == 0 or result.numel() == 1:
        result = result.reshape(1).expand(len(sample))
    if result.shape != (len(sample),) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite scalar or [B]")
    return result


def interval_times(sample, timestep, step_size):
    """Validate normalized scalar / batch times, retaining their derivatives."""
    t = _batch_time(timestep, sample, "timestep")
    d = _batch_time(step_size, sample, "step_size")
    if torch.any((t < 0) | (t > 1) | (d < 0) | (d > t)):
        raise ValueError("Normalized interval must satisfy 0 <= step_size <= timestep <= 1")
    return t, d


def _time_embedding(time, width):
    # Match the existing parent's float32 arithmetic exactly. Double precision
    # remains double so input/time JVPs can also be verified with gradcheck.
    if time.dtype != torch.float64:
        return sinusoidal_embedding(time, width).to(time.dtype)
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    phases = time[:, None] * frequencies[None]
    encoded = torch.cat((phases.sin(), phases.cos()), dim=-1)
    return torch.nn.functional.pad(encoded, (0, width - encoded.shape[-1]))


class IntervalPointDenoiser(PointDenoiser):
    """PointDenoiser plus a zero-initialized duration branch.

    ``timestep`` and ``step_size`` use normalized units, unlike the existing
    PointDenoiser's embedding input. A model initialized from a float32 parent
    equals ``parent(x, 1000*t, conditions)`` at every valid duration before any
    update. No output correction, target labels, geometry, or CFG is added.
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
        time_scale=1000.0,
        step_scale=1000.0,
        allow_null=False,
        base_config=None,
    ):
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if not math.isfinite(step_scale) or step_scale <= 0:
            raise ValueError("step_scale must be finite and positive")
        if not isinstance(allow_null, bool):
            raise ValueError("allow_null must be boolean")
        super().__init__(
            sample_shape=sample_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            backbone=backbone,
            conditioner=conditioner,
            num_classes=num_classes,
            condition_dim=condition_dim,
        )
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        nn.init.zeros_(self.step_encoder[-1].weight)
        nn.init.zeros_(self.step_encoder[-1].bias)
        if base_config is None:
            self.register_to_config(base_config={key: self.config[key] for key in _BASE_KEYS})

    def forward(
        self,
        sample,
        timestep,
        conditions=None,
        step_size=0.0,
        condition_mask=None,
        return_dict=True,
    ):
        if tuple(sample.shape[1:]) != self.sample_shape:
            raise ValueError(f"Expected [B,{self.sample_shape}], received {tuple(sample.shape)}")
        if not sample.is_floating_point() or not torch.isfinite(sample).all():
            raise ValueError("sample must contain finite floating-point values")
        t, d = interval_times(sample, timestep, step_size)
        if conditions is None and not self.config.allow_null:
            raise ValueError("Explicit conditions are required; allow_null is False")
        batch = len(sample)
        context = self.time_encoder(
            _time_embedding(t * self.config.time_scale, self.config.hidden_dim)
        )
        encoded = self.null_condition.expand(batch, -1)
        if conditions is not None:
            if self.condition_encoder is None:
                raise ValueError("conditions supplied to a model without a conditioner")
            if len(conditions) != batch:
                raise ValueError("Condition and sample batch sizes must match")
            conditioned = self.condition_encoder(conditions.to(sample.device))
            if condition_mask is None:
                encoded = conditioned
            else:
                mask = torch.as_tensor(condition_mask, device=sample.device, dtype=torch.bool)
                if mask.shape != (batch,):
                    raise ValueError("condition_mask must have shape [B]")
                if not self.config.allow_null and not mask.all():
                    raise ValueError("Condition dropout requires allow_null=True")
                encoded = torch.where(mask[:, None], conditioned, encoded)
        context = context + encoded
        context = context + self.step_encoder(
            _time_embedding(d * self.config.step_scale, self.config.hidden_dim)
        )
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


def initialize_interval_from_parent(parent, *, seed=0, allow_null=False):
    """Copy a PointDenoiser without changing it or the caller's CPU RNG state.

    The duration branch has small random input weights and an exactly zero
    output layer. Its input initialization is controlled by ``seed``. All
    original weights/buffers are copied; the result is independently trainable
    even when the parent is frozen. ``base_config`` records the parent's config.
    """
    if not isinstance(parent, PointDenoiser) or isinstance(parent, IntervalPointDenoiser):
        raise TypeError("Expected a PointDenoiser parent without an interval branch")
    base_config = {key: parent.config[key] for key in _BASE_KEYS}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = IntervalPointDenoiser(
            **base_config,
            base_config=base_config,
            allow_null=allow_null,
        ).to(device=parent.device, dtype=parent.dtype)
    incompatible = model.load_state_dict(parent.state_dict(), strict=False)
    expected = {f"step_encoder.{suffix}" for suffix in ("0.weight", "0.bias", "2.weight", "2.bias")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected parent state mismatch: {incompatible}")
    model.train(parent.training)
    return model
