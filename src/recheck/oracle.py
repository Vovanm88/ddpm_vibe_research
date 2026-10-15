"""Analytic marginal FM velocity for the registered Gaussian-mixture toy.

This is a mathematical control for the new small benchmark. It is never used
as an unannounced replacement for a learned MNIST teacher.
"""

import torch

GMM_CENTERS = ((-0.6, 0.0), (0.6, 0.0), (0.0, 0.65))
GMM_SIGMA = 0.15


def _as_like(value, x):
    if isinstance(value, torch.Tensor):
        if x.device.type == "mps":
            return value.to(dtype=x.dtype).to(device=x.device)
        return value.to(device=x.device).to(dtype=x.dtype)
    return torch.as_tensor(value, device=x.device, dtype=x.dtype)


def analytic_gmm_velocity(x, t, *, centers=GMM_CENTERS, sigma=GMM_SIGMA, weights=None):
    """v*(x,t), with t=0 data and t=1 noise, from conditional Gaussians."""
    x = torch.as_tensor(x)
    if x.ndim != 2 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Expected finite floating point x[N,D]")
    means = _as_like(centers, x)
    if means.ndim != 2 or means.shape[1] != x.shape[1] or sigma <= 0:
        raise ValueError(
            "Mixture centers must match the state dimension and sigma must be positive"
        )
    time = _as_like(t, x)
    if time.ndim == 0 or time.numel() == 1:
        time = time.reshape(1).expand(len(x))
    if time.shape != (len(x),) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("Time must be a scalar or [N], in [0,1]")
    if weights is None:
        weights = torch.ones(len(means), device=x.device, dtype=x.dtype) / len(means)
    else:
        weights = _as_like(weights, x)
        if (
            weights.shape != (len(means),)
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("Weights must be finite and strictly positive")
        weights = weights / weights.sum()
    time = time[:, None, None]
    q = (1 - time).square() * sigma**2 + time.square()
    b = time - (1 - time) * sigma**2
    centered = x[:, None, :] - (1 - time) * means[None]
    log_responsibility = weights.log()[None, :] - centered.square().sum(-1) / (2 * q.squeeze(-1))
    responsibility = log_responsibility.softmax(-1)
    conditional_velocity = -means[None] + b / q * centered
    return (responsibility[..., None] * conditional_velocity).sum(1)


def _velocity(field, x, t):
    output = field(x, t)
    value = output.sample if hasattr(output, "sample") else output
    if value.shape != x.shape or not torch.isfinite(value).all():
        raise ValueError("Velocity must be finite and have the same shape as the state")
    return value


@torch.no_grad()
def integrate_rk4(field, initial_noise, *, steps=64, times=None, record_history=True):
    """Signed RK4 on a decreasing time grid; exactly four NFE per step."""
    x = torch.as_tensor(initial_noise).clone()
    if times is None:
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("RK4 needs at least one step")
        times = torch.linspace(1, 0, steps + 1, device=x.device, dtype=x.dtype)
    else:
        times = _as_like(times, x)
    if (
        times.ndim != 1
        or len(times) < 2
        or not torch.isfinite(times).all()
        or not (times[1:] < times[:-1]).all()
    ):
        raise ValueError("RK4 times must be finite and strictly decreasing")
    history = [x.detach().clone()] if record_history else None
    nfe = 0
    for start, end in zip(times[:-1], times[1:]):
        dt = end - start
        k1 = _velocity(field, x, start)
        k2 = _velocity(field, x + dt * k1 / 2, start + dt / 2)
        k3 = _velocity(field, x + dt * k2 / 2, start + dt / 2)
        k4 = _velocity(field, x + dt * k3, end)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        nfe += 4
        if record_history:
            history.append(x.detach().clone())
    return {
        "samples": x,
        "history": torch.stack(history) if history is not None else None,
        "nfe": nfe,
        "solver_steps": len(times) - 1,
        "times": times,
    }


def oracle_numerical_report(*, seed=2525, samples=64, steps=32, sigma=GMM_SIGMA):
    """Endpoint tests and step-doubling evidence, with one fixed noise bank."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((samples, 2), generator=generator, dtype=torch.float64)
    means = torch.as_tensor(GMM_CENTERS, dtype=x.dtype)
    v0, v1 = analytic_gmm_velocity(x, 0, sigma=sigma), analytic_gmm_velocity(x, 1, sigma=sigma)

    def field(state, time):
        return analytic_gmm_velocity(state, time, sigma=sigma)

    results = [
        integrate_rk4(field, x, steps=steps * factor, record_history=False) for factor in (1, 2, 4)
    ]
    coarse_error = float((results[0]["samples"] - results[1]["samples"]).square().mean())
    fine_error = float((results[1]["samples"] - results[2]["samples"]).square().mean())
    return {
        "dtype": str(x.dtype),
        "seed": seed,
        "samples": samples,
        "endpoint_t0_max_abs_error": float((v0 + x).abs().max()),
        "endpoint_t1_max_abs_error": float((v1 - (x - means.mean(0))).abs().max()),
        "rk4_solver_steps": [result["solver_steps"] for result in results],
        "rk4_actual_nfe": [result["nfe"] for result in results],
        "rk4_step_doubling_mse": [coarse_error, fine_error],
        "rk4_fine_to_coarse_error_ratio": fine_error / coarse_error if coarse_error > 0 else None,
        "rk4_converging": fine_error < coarse_error,
        "scope": "analytic_GMM_only",
    }


# Explicit aliases ease use in notebooks without hiding the mathematical field.
gmm_velocity = analytic_gmm_velocity
rk4_integrate = integrate_rk4
