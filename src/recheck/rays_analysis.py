"""Fixed geometry diagnostics for unconditional rotating rays.

The environment uses r=R*sqrt(u) and theta=pi/2+2*pi*k/C+kappa*u,
where kappa=1.6*pi/(.9*C). Removing the known radial twist exposes C
equally spaced arms. This analysis receives coordinates only, never labels.
Residuals are descriptive geometry, not a likelihood/noise-model assumption:
every generated statistic is compared with an independent real reference.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

PHASE_TAIL_THRESHOLDS = (0.05, 0.10, 0.20)
PHASE_HISTOGRAM_EDGES = np.linspace(-0.5, 0.5, 41)
RADIAL_CDF_NODES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def _points(samples):
    points = torch.as_tensor(samples).detach().cpu().to(torch.float64).numpy()
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
        raise ValueError(
            "rotating rays analysis expects nonempty [N,2] coordinates, without labels"
        )
    if not np.isfinite(points).all():
        raise ValueError("rotating rays coordinates must be finite")
    return points


def _geometry(n_rays, radius, outer_radius_fraction, noise):
    if not isinstance(n_rays, Integral) or isinstance(n_rays, bool) or n_rays < 1:
        raise ValueError("n_rays must be a positive integer (it is not the state dimension)")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(outer_radius_fraction) or not 0 <= outer_radius_fraction < 1:
        raise ValueError("outer_radius_fraction must lie in [0,1)")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise metadata must be finite and nonnegative")


def phase_coordinates(samples, n_rays=16, radius=10 / 11, *, outer_radius_fraction=0.25):
    """Return nearest-arm IDs and signed within-sector residuals from coordinates.

    IDs are a geometric nearest assignment. At tiny radius they are unstable,
    so occupancy/residual comparisons use the explicit outer mask r>.25R.
    The residual is periodic: fractions wrap to [-.5,.5). We compare its absolute
    distance distribution, avoiding a false distance across the signed wrap cut.
    """
    _geometry(n_rays, radius, outer_radius_fraction, 0.0)
    points = _points(samples)
    radial = np.hypot(points[:, 0], points[:, 1])
    sector = 2 * math.pi / n_rays
    kappa = 1.6 * math.pi / (0.9 * n_rays)
    with np.errstate(over="ignore", invalid="ignore"):
        unwrapped_phase = np.arctan2(points[:, 1], points[:, 0]) - math.pi / 2
        unwrapped_phase -= (radial / radius) ** 2 * kappa
    if not np.isfinite(unwrapped_phase).all():
        raise ValueError("coordinates are too large for finite rotating-ray phase geometry")
    # Both calculations share the same nearest integer, including sector wraps.
    sector_position = np.remainder(unwrapped_phase, 2 * math.pi) / sector
    nearest_integer = np.floor(sector_position + 0.5)
    residual_fraction = sector_position - nearest_integer
    nearest_arm = np.remainder(nearest_integer.astype(np.int64), n_rays)
    return {
        "radius": radial,
        "normalized_radius": radial / radius,
        "nearest_arm": nearest_arm,
        "signed_phase_residual_sector_fraction": residual_fraction,
        "absolute_phase_distance_sector_fraction": np.abs(residual_fraction),
        "signed_phase_residual_radians": residual_fraction * sector,
        "outer_mask": radial > outer_radius_fraction * radius,
        "sector_radians": sector,
        "kappa_radians": kappa,
    }


def _summary(coordinates, n_rays):
    radial = coordinates["normalized_radius"]
    outer = coordinates["outer_mask"]
    signed = coordinates["signed_phase_residual_sector_fraction"][outer]
    distance = np.abs(signed)
    count = int(outer.sum())
    if count:
        occupancy = np.bincount(coordinates["nearest_arm"][outer], minlength=n_rays) / count
        circular_mean = np.exp(2j * math.pi * signed).mean()
        histogram = np.histogram(signed, bins=PHASE_HISTOGRAM_EDGES)[0]
        density = histogram / (count * np.diff(PHASE_HISTOGRAM_EDGES))
        phase = {
            "valid": True,
            "mean_absolute_sector_fraction": float(distance.mean()),
            "median_absolute_sector_fraction": float(np.median(distance)),
            "p95_absolute_sector_fraction": float(np.quantile(distance, 0.95)),
            "circular_concentration": float(abs(circular_mean)),
            "circular_mean_sector_fraction": (
                float(np.angle(circular_mean) / (2 * math.pi))
                if abs(circular_mean) > 1e-12
                else None
            ),
            "off_curve_tail_fractions": {
                f"distance_gt_{threshold:.2f}_sector": float((distance > threshold).mean())
                for threshold in PHASE_TAIL_THRESHOLDS
            },
            "signed_histogram_density": density.tolist(),
        }
        occupancy = occupancy.tolist()
    else:
        occupancy = None
        phase = {
            "valid": False,
            "reason": "no generated points beyond the fixed outer radius",
            "mean_absolute_sector_fraction": None,
            "median_absolute_sector_fraction": None,
            "p95_absolute_sector_fraction": None,
            "circular_concentration": None,
            "circular_mean_sector_fraction": None,
            "off_curve_tail_fractions": None,
            "signed_histogram_density": None,
        }
    return {
        "sample_count": len(radial),
        "outer_count": count,
        "outer_fraction": count / len(radial),
        "nearest_arm_occupancy_outer": occupancy,
        "phase_outer": phase,
        "radial": {
            "mean_raw": float(coordinates["radius"].mean()),
            "mean_normalized": float(radial.mean()),
            "p95_normalized": float(np.quantile(radial, 0.95)),
            "outside_nominal_radius_fraction": float((radial > 1).mean()),
            "mean_excess_above_nominal_radius_normalized": float(np.maximum(radial - 1, 0).mean()),
            "cdf_at_fixed_normalized_nodes": [
                float((radial <= r).mean()) for r in RADIAL_CDF_NODES
            ],
        },
    }


def analyse_rays(
    samples, reference, n_rays=16, radius=10 / 11, *, outer_radius_fraction=0.25, noise=0.008
):
    """JSON-friendly real-relative radial, phase/gap and outer occupancy diagnostics.

    No labels or fitted generated thresholds enter this calculation. The fixed
    .05/.10/.20-sector tails are geometry descriptors, not exact outlier rates;
    noisy real rays also have nonzero residuals and may lie beyond nominal R.
    Small-radius collapse is reported through outer coverage and undefined phase
    metrics, rather than receiving a misleading zero phase/occupancy error.
    """
    _geometry(n_rays, radius, outer_radius_fraction, noise)
    generated = phase_coordinates(
        samples, n_rays, radius, outer_radius_fraction=outer_radius_fraction
    )
    real = phase_coordinates(reference, n_rays, radius, outer_radius_fraction=outer_radius_fraction)
    if not real["outer_mask"].any():
        raise ValueError("independent real reference has no outer points for phase/arm diagnostics")
    generated_summary, real_summary = _summary(generated, n_rays), _summary(real, n_rays)
    generated_outer, real_outer = generated["outer_mask"], real["outer_mask"]
    phase_valid = bool(generated_outer.any())
    metrics = {
        "outer_coverage_fraction_difference": generated_summary["outer_fraction"]
        - real_summary["outer_fraction"],
        "nearest_arm_occupancy_tv_outer": (
            float(
                0.5
                * np.abs(
                    np.asarray(generated_summary["nearest_arm_occupancy_outer"])
                    - np.asarray(real_summary["nearest_arm_occupancy_outer"])
                ).sum()
            )
            if phase_valid
            else None
        ),
        "absolute_phase_distance_w1_sector_fraction": (
            float(
                wasserstein_distance(
                    generated["absolute_phase_distance_sector_fraction"][generated_outer],
                    real["absolute_phase_distance_sector_fraction"][real_outer],
                )
            )
            if phase_valid
            else None
        ),
        "absolute_phase_distance_w1_radians": (
            float(
                wasserstein_distance(
                    generated["absolute_phase_distance_sector_fraction"][generated_outer],
                    real["absolute_phase_distance_sector_fraction"][real_outer],
                )
                * generated["sector_radians"]
            )
            if phase_valid
            else None
        ),
        "radial_w1_raw": float(wasserstein_distance(generated["radius"], real["radius"])),
        "radial_w1_normalized": float(
            wasserstein_distance(generated["normalized_radius"], real["normalized_radius"])
        ),
        "radial_ks_statistic": float(
            ks_2samp(
                generated["normalized_radius"], real["normalized_radius"], method="asymp"
            ).statistic
        ),
        "outside_nominal_radius_fraction_difference": (
            generated_summary["radial"]["outside_nominal_radius_fraction"]
            - real_summary["radial"]["outside_nominal_radius_fraction"]
        ),
        "off_curve_tail_fraction_differences": (
            {
                key: value - real_summary["phase_outer"]["off_curve_tail_fractions"][key]
                for key, value in generated_summary["phase_outer"][
                    "off_curve_tail_fractions"
                ].items()
            }
            if phase_valid
            else None
        ),
    }
    return {
        "schema_version": 1,
        "scope": "coordinate_only_unconditional_rotating_rays_geometry",
        "geometry": {
            "state_dimension": 2,
            "n_rays": int(n_rays),
            "radius": float(radius),
            "environment_noise_parameter": float(noise),
            "outer_radius_fraction": float(outer_radius_fraction),
            "outer_condition": "r > outer_radius_fraction * radius",
            "kappa_radians": generated["kappa_radians"],
            "sector_radians": generated["sector_radians"],
            "phase_formula": "atan2(y,x)-pi/2-(r/R)^2*1.6*pi/(0.9*C)",
            "phase_tail_thresholds_sector_fraction": list(PHASE_TAIL_THRESHOLDS),
            "phase_histogram_edges_sector_fraction": PHASE_HISTOGRAM_EDGES.tolist(),
            "radial_cdf_nodes_normalized": list(RADIAL_CDF_NODES),
        },
        "generated": generated_summary,
        "reference": real_summary,
        "generated_phase_coverage_valid": phase_valid,
        "metrics": metrics,
        "limitations": [
            "Nearest arm is a geometric assignment, not a latent class label or conditioning.",
            "Noise changes both angle and radius; residuals are compared with noisy real reference, not an assumed Gaussian law.",
            "The outer-radius restriction avoids unstable phase near the origin; radial/coverage metrics still include every point.",
            "Geometry metrics supplement independent quality/fidelity; they do not measure trajectory accuracy or prove an OT mechanism.",
        ],
    }


def plot_rays_diagnostics(
    samples,
    reference,
    n_rays=16,
    radius=10 / 11,
    *,
    outer_radius_fraction=0.25,
    noise=0.008,
    title="Rotating rays geometry",
):
    """Return four panels with fixed axes/bins, suitable for same-NFE comparisons."""
    import matplotlib.pyplot as plt

    result = analyse_rays(
        samples, reference, n_rays, radius, outer_radius_fraction=outer_radius_fraction, noise=noise
    )
    points, real_points = _points(samples), _points(reference)
    generated = phase_coordinates(
        points, n_rays, radius, outer_radius_fraction=outer_radius_fraction
    )
    real = phase_coordinates(
        real_points, n_rays, radius, outer_radius_fraction=outer_radius_fraction
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(
        real_points[:, 0],
        real_points[:, 1],
        s=3,
        alpha=0.15,
        color="#8795a7",
        label="Real reference",
    )
    ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.35, color="#7654b7", label="Generated")
    u = np.linspace(0, 1, 256)
    for arm in range(n_rays):
        theta = math.pi / 2 + 2 * math.pi * arm / n_rays + generated["kappa_radians"] * u
        ax.plot(
            radius * np.sqrt(u) * np.cos(theta),
            radius * np.sqrt(u) * np.sin(theta),
            color="#28364a",
            lw=0.5,
            alpha=0.45,
        )
    limit = 1.25 * radius
    outside_view = int((np.abs(points) > limit).any(axis=1).sum())
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title=f"Fixed view ±1.25R · {outside_view} generated points outside view",
    )
    ax.legend(markerscale=3)

    ax = axes[0, 1]
    for label, values, color in (
        ("Real reference", real, "#8795a7"),
        ("Generated", generated, "#7654b7"),
    ):
        outer = values["outer_mask"]
        if outer.any():
            ax.hist(
                values["signed_phase_residual_sector_fraction"][outer],
                bins=PHASE_HISTOGRAM_EDGES,
                density=True,
                histtype="step",
                lw=1.8,
                label=label,
                color=color,
            )
    ax.axvline(0, color="#28364a", lw=0.7, ls="--")
    ax.set(
        xlim=(-0.5, 0.5),
        ylim=(0, 1 / np.diff(PHASE_HISTOGRAM_EDGES).min()),
        xlabel="Signed phase residual / sector width",
        ylabel="Density",
        title=f"Folded phase · outer r>{outer_radius_fraction:g}R",
    )
    ax.legend()

    ax = axes[1, 0]
    arm_ids = np.arange(n_rays)
    ax.bar(
        arm_ids - 0.2,
        result["reference"]["nearest_arm_occupancy_outer"],
        width=0.4,
        color="#a3afbe",
        label="Real reference",
    )
    if result["generated_phase_coverage_valid"]:
        ax.bar(
            arm_ids + 0.2,
            result["generated"]["nearest_arm_occupancy_outer"],
            width=0.4,
            color="#7654b7",
            label="Generated",
        )
    else:
        ax.text(
            0.5,
            0.8,
            "Generated occupancy undefined: no outer points",
            ha="center",
            transform=ax.transAxes,
        )
    ax.set(
        xticks=arm_ids,
        xlabel="Nearest geometric arm (no labels used)",
        ylabel="Outer occupancy",
        ylim=(0, 1),
        title="Arm occupancy · same fixed scale [0,1]",
    )
    ax.legend()

    ax = axes[1, 1]
    for label, values, color in (
        ("Real reference", real, "#8795a7"),
        ("Generated", generated, "#7654b7"),
    ):
        ordered = np.sort(values["normalized_radius"])
        ax.step(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            where="post",
            label=label,
            color=color,
        )
    ax.axvline(1, color="#28364a", lw=0.7, ls="--")
    ax.set(
        xlim=(0, 1.25),
        ylim=(0, 1),
        xlabel="Radius / R",
        ylabel="Empirical CDF",
        title="Radial distribution · raw distances also retained in metrics",
    )
    ax.legend()
    figure.suptitle(title + " · independent real reference", fontsize=13)
    return figure, axes
