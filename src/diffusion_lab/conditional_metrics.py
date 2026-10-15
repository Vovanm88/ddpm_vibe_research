"""Coordinate-only condition routing checks for the rotating-rays environment.

These are geometric proxies, not a learned classifier or recovery of a latent
ground-truth label from arbitrary coordinates. Near the common origin arm IDs
are ill-conditioned; exact origins receive -1, and outer coverage is always
reported alongside accuracy. For initial Gaussian noise beyond the nominal
radius, the phase formula extends the curves: it is NOT Euclidean nearest
distance to a finite ray. The frozen unconditional recheck protocol is untouched.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


def _settings(n_classes, radius, minimum_radius_fraction=0.25):
    if not isinstance(n_classes, Integral) or isinstance(n_classes, bool) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    if (
        not isinstance(radius, Real)
        or isinstance(radius, bool)
        or not math.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("radius must be finite and positive")
    if (
        not isinstance(minimum_radius_fraction, Real)
        or isinstance(minimum_radius_fraction, bool)
        or not math.isfinite(minimum_radius_fraction)
        or not 0 <= minimum_radius_fraction < 1
    ):
        raise ValueError("minimum_radius_fraction must lie in [0,1)")


def _points(points, name="points"):
    result = torch.as_tensor(points).detach().cpu()
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 1:
        raise ValueError(f"{name} must be nonempty [N,2] coordinates")
    if result.is_complex() or result.dtype == torch.bool or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real coordinates")
    return result.double()


def _geometry(points, n_classes, radius):
    radial = torch.hypot(points[:, 0], points[:, 1])
    phase = torch.atan2(points[:, 1], points[:, 0]) - math.pi / 2
    phase -= (radial / radius).square() * (1.6 * math.pi / (0.9 * n_classes))
    if not torch.isfinite(phase).all():
        raise ValueError("coordinates are too large for finite ray phase geometry")
    sector = 2 * math.pi / n_classes
    position = torch.remainder(phase, 2 * math.pi) / sector
    nearest_integer = torch.floor(position + 0.5)
    ids = torch.remainder(nearest_integer.long(), n_classes)
    ids[radial == 0] = -1
    return ids, radial, phase, (position - nearest_integer) * sector


def nearest_ray_ids(points, n_classes=16, radius=10 / 11):
    """Return detached CPU long[N] nearest unwound-phase arm IDs; origin is -1.

    Clean environment points obey r=R*sqrt(u), theta=pi/2+2*pi*k/C+
    1.6*pi*u/(.9*C), so the inverse phase recovers label k (except at r=0,
    where all curves meet). Noise perturbs this assignment. IDs outside the
    nominal radial support describe a phase continuation, not a finite-curve
    Euclidean nearest neighbour. Half-sector ties round upward, periodically.
    """
    _settings(n_classes, radius)
    return _geometry(_points(points), n_classes, radius)[0]


def _fraction(event, mask):
    count = int(mask.sum())
    return float(event[mask].double().mean()) if count else None


def _phase_summary(error, defined, outer, sector):
    result = {}
    for name, mask in (("all", defined), ("outer", outer)):
        values = error[mask]
        result[f"defined_count_{name}"] = len(values)
        result[f"mean_signed_radians_{name}"] = float(values.mean()) if len(values) else None
        result[f"mean_absolute_radians_{name}"] = (
            float(values.abs().mean()) if len(values) else None
        )
        result[f"mean_absolute_sector_fraction_{name}"] = (
            float(values.abs().mean() / sector) if len(values) else None
        )
        result[f"p95_absolute_sector_fraction_{name}"] = (
            float(torch.quantile(values.abs(), 0.95) / sector) if len(values) else None
        )
    return result


def _counterfactual_groups(initial, targets, ids, correct, outer, n_classes):
    """Summarize exactly repeated noise vectors, without inferring pairing by order."""
    unique, inverse = torch.unique(initial, dim=0, return_inverse=True)
    counts = torch.bincount(
        inverse * n_classes + targets, minlength=len(unique) * n_classes
    ).reshape(len(unique), n_classes)
    complete = (counts == 1).all(1)
    valid = ids >= 0
    outputs = torch.bincount(
        inverse[valid] * n_classes + ids[valid], minlength=len(unique) * n_classes
    ).reshape(len(unique), n_classes)
    correct_counts = torch.bincount(inverse[correct], minlength=len(unique))
    outer_counts = torch.bincount(inverse[outer], minlength=len(unique))
    return {
        "unique_noise_vectors": len(unique),
        "complete_noise_groups": int(complete.sum()),
        "definition": "complete group contains the EXACT same initial vector once for every requested class",
        "mean_distinct_output_arms_per_complete_noise": float(
            (outputs[complete] > 0).sum(1).double().mean()
        )
        if complete.any()
        else None,
        "fraction_complete_groups_all_targets_correct": _fraction(
            correct_counts == n_classes, complete
        ),
        "fraction_complete_groups_all_outputs_outer": _fraction(
            outer_counts == n_classes, complete
        ),
    }


def conditional_routing_metrics(
    samples,
    targets,
    initial_noise=None,
    *,
    n_classes=16,
    radius=10 / 11,
    minimum_radius_fraction=0.25,
):
    """Measure requested-label adherence, separately from nearest-curve quality.

    Input samples are raw [N,2] endpoints; targets are strict int32/int64 [N]
    categorical IDs. Optional initial_noise must have the same shape and order.
    For a direct intervention test repeat one noise bank for ALL requested IDs.
    Requested phase wraps the FULL target-relative angle to [-pi,pi], rather
    than folding every arm into a zero residual. Thus landing on the wrong arm
    does not earn zero condition error. Phase summaries exclude exact origins.

    accuracy_all counts exact-origin assignments as incorrect. accuracy_outer
    is undefined (None) if no points have r>minimum_radius_fraction*R; always
    inspect outer_coverage and correct_outer_fraction_of_all to expose tiny-
    radius collapse. Confusion rows are requested labels; columns are inferred
    geometric IDs. Unassigned origin counts are separate, so normalized rows
    can sum to less than one. Absent requested classes have null metrics/rows.
    """
    _settings(n_classes, radius, minimum_radius_fraction)
    points = _points(samples, "samples")
    targets = torch.as_tensor(targets).detach().cpu()
    if targets.dtype not in (torch.int32, torch.int64) or targets.shape != (len(points),):
        raise ValueError("targets must be strict int32/int64 categorical IDs of shape [N]")
    targets = targets.long()
    if ((targets < 0) | (targets >= n_classes)).any():
        raise ValueError(f"targets must lie in [0,{n_classes})")
    ids, radial, phase, nearest_residual = _geometry(points, n_classes, radius)
    defined = ids >= 0
    outer = radial > minimum_radius_fraction * radius
    correct = defined & (ids == targets)
    all_points = torch.ones(len(points), dtype=torch.bool)
    counts = torch.bincount(targets, minlength=n_classes)
    confusion = torch.bincount(
        targets[defined] * n_classes + ids[defined], minlength=n_classes**2
    ).reshape(n_classes, n_classes)
    unassigned = torch.bincount(targets[~defined], minlength=n_classes)
    normalized = [
        (confusion[k].double() / counts[k]).tolist() if counts[k] else [None] * n_classes
        for k in range(n_classes)
    ]
    sector = 2 * math.pi / n_classes
    requested_error = torch.remainder(phase - targets * sector + math.pi, 2 * math.pi) - math.pi
    per_class = []
    for k in range(n_classes):
        mask = targets == k
        per_class.append(
            {
                "target_id": k,
                "count": int(mask.sum()),
                "correct_count": int((correct & mask).sum()),
                "accuracy_all": _fraction(correct, mask),
                "outer_count": int((outer & mask).sum()),
                "accuracy_outer": _fraction(correct, outer & mask),
                "outer_coverage": _fraction(outer, mask),
                "correct_outer_fraction_of_all": _fraction(correct & outer, mask),
                "unassigned_count": int(unassigned[k]),
            }
        )
    initial_routing = None
    counterfactual = None
    if initial_noise is not None:
        initial = _points(initial_noise, "initial_noise")
        if initial.shape != points.shape:
            raise ValueError("initial_noise must have the same [N,2] shape/order as samples")
        initial_ids, initial_radii, _, _ = _geometry(initial, n_classes, radius)
        initial_defined = initial_ids >= 0
        contrast = initial_defined & (targets != initial_ids)
        changed = defined & initial_defined & (ids != initial_ids)
        matches_initial = defined & initial_defined & (ids == initial_ids)
        initial_routing = {
            "proxy_caveat": "unwound-phase ID; outside nominal R it extends the curves, not Euclidean nearest finite curve",
            "initial_unassigned_count": int((~initial_defined).sum()),
            "initial_outside_nominal_radius_fraction": float(
                (initial_radii > radius).double().mean()
            ),
            "target_matches_initial_fraction": _fraction(
                initial_defined & (targets == initial_ids), all_points
            ),
            "output_matches_initial_fraction": _fraction(matches_initial, all_points),
            "output_changes_arm_fraction": _fraction(changed, all_points),
            "target_differs_from_initial_count": int(contrast.sum()),
            "output_matches_target_when_target_differs_from_initial": _fraction(correct, contrast),
            "output_changes_arm_when_target_differs_from_initial": _fraction(changed, contrast),
            "output_stays_initial_when_target_differs_from_initial": _fraction(
                matches_initial, contrast
            ),
            "correct_outer_fraction_when_target_differs_from_initial": _fraction(
                correct & outer, contrast
            ),
        }
        counterfactual = _counterfactual_groups(initial, targets, ids, correct, outer, n_classes)
    return {
        "scope": "requested_condition_coordinate_routing_proxy",
        "n_samples": len(points),
        "n_classes": int(n_classes),
        "radius": float(radius),
        "minimum_radius_fraction": float(minimum_radius_fraction),
        "accuracy_all": float(correct.double().mean()),
        "accuracy_outer": _fraction(correct, outer),
        "outer_count": int(outer.sum()),
        "outer_coverage": float(outer.double().mean()),
        "correct_outer_fraction_of_all": float((correct & outer).double().mean()),
        "assignment_defined_count": int(defined.sum()),
        "unassigned_count": int((~defined).sum()),
        "per_class_counts": counts.tolist(),
        "unassigned_by_target": unassigned.tolist(),
        "confusion_counts": confusion.tolist(),
        "confusion_row_normalized": normalized,
        "per_class": per_class,
        "requested_phase_error": {
            "definition": "wrap_pi(atan2(y,x)-pi/2-1.6*pi*(r/R)^2/(.9*C)-2*pi*requested_id/C)",
            **_phase_summary(requested_error, defined, outer, sector),
        },
        "nearest_phase_residual": {
            "definition": "within-sector residual after nearest-arm folding; can be near zero on the WRONG requested arm",
            **_phase_summary(nearest_residual, defined, outer, sector),
        },
        "initial_routing": initial_routing,
        "counterfactual_same_noise": counterfactual,
        "limitations": [
            "Coordinate-derived nearest IDs are geometric proxies, not a classifier or latent ground-truth labels for arbitrary points.",
            "All curves meet at zero; near-zero routing is ill-conditioned. Always inspect outer coverage and outer correctness together.",
            "Nearest-arm phase residual alone cannot establish requested-condition adherence.",
        ],
    }
