"""Small, inspectable mathematical references for interval-velocity methods.

These functions are not trainers. Research notebooks spell out their target,
loss, optimizer, and EMA updates directly. Time runs from data=0 to noise=1;
sampling is ``x_next = x - d*u(x,t,d)``. The common-parent warm starts in this
project are toy adaptations, not reproductions of the papers' from-scratch
image-generation protocols.

Shortcut: Frans et al., https://arxiv.org/abs/2410.12557.
MeanFlow: Geng et al., https://arxiv.org/abs/2505.13447, Eqs. (9)-(11).
"""

import math

import torch
from torch.func import functional_call, jvp

from .interval_model import interval_times


def _expand_batch(values, sample):
    return values.reshape((-1,) + (1,) * (sample.ndim - 1))


def _output_sample(output):
    return output.sample if hasattr(output, "sample") else output


def flow_matching_pair(clean, noise, timestep):
    """Return ``x_t=(1-t)*clean+t*noise`` and ``v=noise-clean``."""
    if clean.shape != noise.shape or clean.device != noise.device or clean.dtype != noise.dtype:
        raise ValueError("clean and noise must have identical shape, device and dtype")
    if (
        not clean.is_floating_point()
        or not torch.isfinite(clean).all()
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("clean and noise must be finite floating-point tensors")
    t, _ = interval_times(clean, timestep, 0.0)
    weight = _expand_batch(t, clean)
    return (1 - weight) * clean + weight * noise, noise - clean


def interval_euler_step(model, sample, timestep, step_size, conditions):
    """A differentiable one-evaluation normalized-time sampling step."""
    t, d = interval_times(sample, timestep, step_size)
    velocity = _output_sample(model(sample, t, conditions=conditions, step_size=d))
    return sample - _expand_batch(d, sample) * velocity


@torch.no_grad()
def shortcut_target(model, sample, timestep, step_size, conditions, *, smallest_horizon=None):
    """Detached two-half-step target, adapted to decreasing sampling time.

    A coarse width D uses h=D/2 and target
    ``[u(x,t,h)+u(x-h*u(x,t,h),t-h,h)]/2``. At the smallest dyadic coarse
    interval, ``smallest_horizon=h`` optionally queries the d=0 FM boundary for
    both velocities, while the physical half-step remains h. The source model
    may be an EMA copy; this function never updates it or clips its outputs.
    """
    t, d = interval_times(sample, timestep, step_size)
    if torch.any(d <= 0):
        raise ValueError("Shortcut bootstrap requires positive step_size; d=0 uses the FM target")
    half = d / 2
    query_width = half
    if smallest_horizon is not None:
        if (
            not isinstance(smallest_horizon, (int, float))
            or not math.isfinite(smallest_horizon)
            or smallest_horizon <= 0
        ):
            raise ValueError("smallest_horizon must be positive")
        query_width = torch.where(half <= smallest_horizon, torch.zeros_like(half), half)
    first = _output_sample(model(sample, t, conditions=conditions, step_size=query_width))
    midpoint = sample - _expand_batch(half, sample) * first
    second = _output_sample(model(midpoint, t - half, conditions=conditions, step_size=query_width))
    return ((first + second) / 2).detach()


def meanflow_target(
    model,
    sample,
    timestep,
    step_size,
    conditional_velocity,
    conditions,
    *,
    return_derivative=False,
):
    """Detached MeanFlow target with the correct total derivative.

    Paper coordinates are (x,r,t), tangent (v,0,1). This model's coordinates
    are (x,t,d=t-r), so the SAME derivative uses tangent (v,1,1). Omitting the
    duration tangent differentiates a different function. Parameters and
    conditions are stopped for the JVP; input/time derivatives remain active.
    The gradient-enabled prediction is evaluated separately by the caller.
    """
    if conditional_velocity.shape != sample.shape:
        raise ValueError("conditional_velocity must have sample's shape")
    if conditional_velocity.device != sample.device or conditional_velocity.dtype != sample.dtype:
        raise ValueError("conditional_velocity must match sample's device and dtype")
    if not torch.isfinite(conditional_velocity).all():
        raise ValueError("conditional_velocity must be finite")
    t, d = interval_times(sample, timestep, step_size)
    stopped_state = {
        key: tensor.detach() for key, tensor in (*model.named_parameters(), *model.named_buffers())
    }
    stopped_conditions = conditions.detach() if conditions is not None else None

    def field(x, time, width):
        return _output_sample(
            functional_call(
                model,
                stopped_state,
                (x, time),
                {"conditions": stopped_conditions, "step_size": width},
                strict=True,
            )
        )

    with torch.no_grad():
        _, derivative = jvp(
            field,
            (sample.detach(), t.detach(), d.detach()),
            (conditional_velocity.detach(), torch.ones_like(t), torch.ones_like(d)),
        )
        target = conditional_velocity.detach() - _expand_batch(d.detach(), sample) * derivative
    if return_derivative:
        return target.detach(), derivative.detach()
    return target.detach()


def adaptive_velocity_loss(prediction, target, *, power=1.0, epsilon=1e-3, state_reduction="sum"):
    """MeanFlow stopped adaptive weight on per-sample squared L2 error.

    The default sum across state dimensions matches the official MeanFlow
    implementation. ``state_reduction='mean'`` is an explicit toy adaptation;
    the same epsilon then changes the effective regularizer by dimension.
    ``power=0`` removes adaptive weighting. Weight derivatives are stopped, as in the
    paper; this is not the derivative of log-MSE when power is one.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if not isinstance(power, (int, float)) or power < 0 or not math.isfinite(power):
        raise ValueError("power must be finite and nonnegative")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if state_reduction not in ("sum", "mean"):
        raise ValueError("state_reduction must be 'sum' or 'mean'")
    error = (prediction - target.detach()).square().flatten(1)
    squared = error.sum(1) if state_reduction == "sum" else error.mean(1)
    weights = (squared.detach() + epsilon).pow(-power)
    return (weights * squared).mean()
