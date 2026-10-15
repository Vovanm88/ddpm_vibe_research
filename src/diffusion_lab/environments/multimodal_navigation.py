"""Two collision-checked route modes for the EXACT same A/B/wall condition.

The model receives eight continuous numbers: start(2), goal(2), wall low(2),
wall high(2). Top/bottom route IDs are evaluation labels ONLY. Paths preserve
the two obstacle-clearance corners during resampling, rather than cutting
through an obstacle when a uniform arc-length grid misses a critical corner.
This environment is independent of the frozen unconditional recheck protocol.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import numpy as np
import torch

from .base import SampleDataset

CONDITION_LAYOUT = "start(2), goal(2), wall_low(2), wall_high(2)"
BOUNDS = ((-1.0, -1.0), (1.0, 1.0))


def _integer(value, name, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _nonnegative(value, name):
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if np.iscomplexobj(value):
        raise ValueError("navigation geometry requires real coordinates")
    return np.asarray(value, dtype=np.float64)


def _inputs(paths, conditions):
    paths, conditions = _numpy(paths), _numpy(conditions)
    if paths.ndim != 3 or paths.shape[0] < 1 or paths.shape[1] < 2 or paths.shape[2] != 2:
        raise ValueError("paths must be nonempty [B,L,2] with L>=2")
    if conditions.shape != (len(paths), 8):
        raise ValueError(f"conditions must have shape [B,8]: {CONDITION_LAYOUT}")
    if not np.isfinite(paths).all() or not np.isfinite(conditions).all():
        raise ValueError("paths and conditions must be finite")
    if np.any(conditions[:, 4:6] >= conditions[:, 6:8]):
        raise ValueError("wall low/high corners must be strictly ordered")
    return paths, conditions


def _wall_collisions(paths, conditions, margin):
    """Exact closed-segment/per-example AABB slab test, including zero lengths."""
    start = paths[:, :-1]
    delta = paths[:, 1:] - start
    low, high = conditions[:, None, 4:6] - margin, conditions[:, None, 6:8] + margin
    parallel = np.abs(delta) < 1e-12
    safe_delta = np.where(parallel, 1.0, delta)
    first, last = (low - start) / safe_delta, (high - start) / safe_delta
    outside = (start < low) | (start > high)
    entry = np.minimum(first, last)
    exit = np.maximum(first, last)
    entry = np.where(parallel, np.where(outside, np.inf, -np.inf), entry)
    exit = np.where(parallel, np.where(outside, -np.inf, np.inf), exit)
    enter_t = np.maximum(entry.max(axis=-1), 0)
    exit_t = np.minimum(exit.min(axis=-1), 1)
    return (enter_t <= exit_t).any(axis=-1)


def _assessment(paths, conditions, margin, endpoint_tol):
    collision_free = ~_wall_collisions(paths, conditions, margin)
    bounds_valid = ((paths >= -1) & (paths <= 1)).all(axis=(1, 2))
    start_error = np.linalg.norm(paths[:, 0] - conditions[:, :2], axis=1)
    goal_error = np.linalg.norm(paths[:, -1] - conditions[:, 2:4], axis=1)
    endpoint_ok = (start_error <= endpoint_tol) & (goal_error <= endpoint_tol)
    success = collision_free & bounds_valid & endpoint_ok
    # A successful A-left/B-right path crosses the wall's central vertical line.
    # Multiple crossings on BOTH sides represent a loopy/ambiguous route, not a
    # clean demonstration of either reference mode, and remain unassigned.
    middle = (conditions[:, 4] + conditions[:, 6]) / 2
    start, end = paths[:, :-1], paths[:, 1:]
    dx = end[:, :, 0] - start[:, :, 0]
    parallel = np.abs(dx) < 1e-12
    fraction = (middle[:, None] - start[:, :, 0]) / np.where(parallel, 1.0, dx)
    crossing = ~parallel & (fraction >= 0) & (fraction <= 1)
    on_line = parallel & (np.abs(start[:, :, 0] - middle[:, None]) < 1e-12)
    crossing |= on_line
    y = start[:, :, 1] + fraction * (end[:, :, 1] - start[:, :, 1])
    minimum_y = np.where(on_line, np.minimum(start[:, :, 1], end[:, :, 1]), y)
    maximum_y = np.where(on_line, np.maximum(start[:, :, 1], end[:, :, 1]), y)
    above = (minimum_y > conditions[:, None, 7] + margin) | ~crossing
    below = (maximum_y < conditions[:, None, 5] - margin) | ~crossing
    has_crossing = crossing.any(axis=1)
    modes = np.full(len(paths), -1, dtype=np.int64)
    modes[success & has_crossing & above.all(axis=1)] = 0
    modes[success & has_crossing & below.all(axis=1)] = 1
    return {
        "collision_free": collision_free,
        "bounds_valid": bounds_valid,
        "endpoint_ok": endpoint_ok,
        "success": success,
        "route_ids": modes,
        "start_error": start_error,
        "goal_error": goal_error,
        "path_length": np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1),
    }


def classify_navigation_modes(paths, conditions, *, margin=0.035, endpoint_tol=0.08):
    """CPU long[B]: 0 top, 1 bottom; -1 invalid/unsuccessful/ambiguous.

    All crossings of the wall's x-centre must pass the same side of its inflated
    box. Collision, bounds and endpoint checks apply BEFORE assigning a mode.
    A failed path cannot increase measured top/bottom coverage.
    """
    _nonnegative(margin, "margin")
    _nonnegative(endpoint_tol, "endpoint_tol")
    paths, conditions = _inputs(paths, conditions)
    return torch.from_numpy(_assessment(paths, conditions, margin, endpoint_tol)["route_ids"])


def evaluate_navigation(paths, conditions, *, margin=0.035, endpoint_tol=0.08):
    """JSON metrics for full spatial paths; not FM denoising-time histories.

    Segment/AABB tests include inter-waypoint wall crossings and closed-boundary
    contact. The bounding square is convex, so checking all waypoints suffices
    for all connecting straight segments to remain within bounds. Endpoint MSE
    is descriptive: when a model reconstructs A/B exactly, it is a constraint,
    not evidence that endpoints were learned. Mode coverage groups EXACT equal
    eight-number conditions and counts ONLY successful unambiguous routes.
    """
    _nonnegative(margin, "margin")
    _nonnegative(endpoint_tol, "endpoint_tol")
    paths, conditions = _inputs(paths, conditions)
    assessment = _assessment(paths, conditions, margin, endpoint_tol)
    modes, success = assessment["route_ids"], assessment["success"]
    unique, inverse = np.unique(conditions, axis=0, return_inverse=True)
    groups = []
    for index, condition in enumerate(unique):
        mask = inverse == index
        valid = mask & success
        counts = [int((valid & (modes == k)).sum()) for k in (0, 1)]
        assigned = sum(counts)
        groups.append(
            {
                "condition": condition.tolist(),
                "sample_count": int(mask.sum()),
                "successful_count": int(valid.sum()),
                "success_fraction": float(success[mask].mean()),
                "successful_mode_counts": counts,
                "both_modes_successful": all(count > 0 for count in counts),
                "successful_unassigned_mode_count": int((valid & (modes < 0)).sum()),
                "mode_balance_tv_successful_assigned": float(abs(counts[0] / assigned - 0.5))
                if assigned
                else None,
            }
        )
    with_success = [group for group in groups if group["successful_count"] > 0]
    both_count = sum(group["both_modes_successful"] for group in groups)
    balances = [
        group["mode_balance_tv_successful_assigned"]
        for group in groups
        if group["mode_balance_tv_successful_assigned"] is not None
    ]
    endpoint_delta = np.stack(
        (paths[:, 0] - conditions[:, :2], paths[:, -1] - conditions[:, 2:4]), axis=1
    )
    return {
        "scope": "full_spatial_multimodal_navigation_geometry",
        "n_samples": len(paths),
        "path_length": paths.shape[1],
        "n_conditions": len(unique),
        "condition_layout": CONDITION_LAYOUT,
        "margin": float(margin),
        "endpoint_tol": float(endpoint_tol),
        "success_fraction": float(success.mean()),
        "collision_free_fraction": float(assessment["collision_free"].mean()),
        "bounds_fraction": float(assessment["bounds_valid"].mean()),
        "endpoint_success_fraction": float(assessment["endpoint_ok"].mean()),
        "endpoint_mse": float(np.square(endpoint_delta).mean()),
        "mean_start_error": float(assessment["start_error"].mean()),
        "mean_goal_error": float(assessment["goal_error"].mean()),
        "mean_path_length": float(assessment["path_length"].mean()),
        "mean_path_length_successful": float(assessment["path_length"][success].mean())
        if success.any()
        else None,
        "mean_squared_acceleration": float(np.square(np.diff(paths, n=2, axis=1)).mean())
        if paths.shape[1] > 2
        else None,
        "successful_mode_counts": [int((modes == k).sum()) for k in (0, 1)],
        "unassigned_route_count": int((modes < 0).sum()),
        "mode_coverage": {
            "both_modes_fraction_all_conditions": both_count / len(groups),
            "both_modes_fraction_conditions_with_success": both_count / len(with_success)
            if with_success
            else None,
            "conditions_with_success": len(with_success),
            "mean_mode_balance_tv_successful_assigned": float(np.mean(balances))
            if balances
            else None,
            "condition_groups": groups,
        },
        "per_sample": {key: value.tolist() for key, value in assessment.items()},
        "limitations": [
            "Route IDs are evaluation labels; only A/B/wall geometry is a model condition.",
            "Endpoints may be reconstructed constraints rather than learned predictions.",
            "Mode coverage uses successful routes only; both-mode coverage over ALL requested conditions remains the primary denominator.",
            "Successful paths crossing above and below the wall are conservatively unassigned, rather than credited as a clean reference mode.",
        ],
    }


def _resample_corners(controls, path_length):
    """Vectorized four-corner/three-segment allocation with both corners retained."""
    lengths = np.linalg.norm(np.diff(controls, axis=1), axis=2)
    allocation = (path_length - 4) * lengths / lengths.sum(axis=1, keepdims=True)
    intervals = np.floor(allocation).astype(np.int64) + 1
    remainder = path_length - 1 - intervals.sum(axis=1)
    priority = np.argsort(-(allocation - np.floor(allocation)), axis=1, kind="stable")
    for offset in range(2):
        rows = np.flatnonzero(remainder > offset)
        intervals[rows, priority[rows, offset]] += 1
    cumulative = intervals.cumsum(axis=1)
    positions = np.arange(path_length)[None, :]
    segment = (positions > cumulative[:, :1]).astype(np.int64)
    segment += (positions > cumulative[:, 1:2]).astype(np.int64)
    begins = np.concatenate(
        (np.zeros((len(controls), 1), dtype=np.int64), cumulative[:, :-1]), axis=1
    )
    rows = np.arange(len(controls))[:, None]
    fraction = (positions - begins[rows, segment]) / intervals[rows, segment]
    start, end = controls[rows, segment], controls[rows, segment + 1]
    paths = start + fraction[:, :, None] * (end - start)
    paths[:, 0], paths[:, -1] = controls[:, 0], controls[:, -1]
    return paths


def make_multimodal_navigation(
    n_scenes=256,
    samples_per_scene=8,
    path_length=32,
    seed=42,
    *,
    split="train",
    jitter=0.025,
    clearance=0.06,
    margin=0.035,
):
    """Deterministic variable-wall scenes, each repeated in BOTH route modes.

    samples_per_scene must be even and >=2; every exact condition has half top
    and half bottom reference routes. Outward-only corner jitter changes route
    shape without changing A/B/wall conditions or sacrificing clearance. RNG
    streams for train/validation/test use distinct stable hashes; no global RNG
    is touched. No A* planner or route-mode ID is used as a model input.
    """
    _integer(n_scenes, "n_scenes")
    _integer(samples_per_scene, "samples_per_scene", 2)
    _integer(path_length, "path_length", 4)
    _integer(seed, "seed", 0)
    if samples_per_scene % 2:
        raise ValueError("samples_per_scene must be even so every condition has both modes equally")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    for name, value in (("jitter", jitter), ("clearance", clearance), ("margin", margin)):
        _nonnegative(value, name)
    if clearance <= margin:
        raise ValueError("clearance must exceed the closed-wall collision margin")
    if 0.6 + clearance + jitter >= 1:
        raise ValueError("clearance+jitter leaves insufficient room inside [-1,1] bounds")
    resolved_seed = int.from_bytes(
        hashlib.sha256(f"multimodal-navigation-v1:{seed}:{split}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(resolved_seed)
    starts = np.column_stack(
        (rng.uniform(-0.85, -0.7, n_scenes), rng.uniform(-0.15, 0.15, n_scenes))
    )
    goals = np.column_stack((rng.uniform(0.7, 0.85, n_scenes), rng.uniform(-0.15, 0.15, n_scenes)))
    centres = np.column_stack(
        (rng.uniform(-0.12, 0.12, n_scenes), rng.uniform(-0.10, 0.10, n_scenes))
    )
    size = np.column_stack((rng.uniform(0.20, 0.35, n_scenes), rng.uniform(0.6, 1.0, n_scenes)))
    scene_conditions = np.concatenate(
        (starts, goals, centres - size / 2, centres + size / 2), axis=1
    ).astype(np.float32)
    scene_indices = np.repeat(np.arange(n_scenes, dtype=np.int64), samples_per_scene)
    conditions = scene_conditions[scene_indices]
    route_ids = np.tile(np.arange(samples_per_scene) % 2, n_scenes).astype(np.int64)
    count = len(conditions)
    offsets = rng.uniform(0, jitter, (count, 4))
    controls = np.empty((count, 4, 2), dtype=np.float64)
    controls[:, 0], controls[:, -1] = conditions[:, :2], conditions[:, 2:4]
    controls[:, 1, 0] = conditions[:, 4] - clearance - offsets[:, 0]
    controls[:, 2, 0] = conditions[:, 6] + clearance + offsets[:, 1]
    top = route_ids == 0
    for index in (1, 2):
        controls[:, index, 1] = np.where(
            top,
            conditions[:, 7] + clearance + offsets[:, index + 1],
            conditions[:, 5] - clearance - offsets[:, index + 1],
        )
    paths = _resample_corners(controls, path_length).astype(np.float32)
    # Validate the ACTUAL float32 reference segments, not just control points.
    checked = _assessment(paths.astype(np.float64), conditions.astype(np.float64), margin, 1e-6)
    if not checked["success"].all() or not np.array_equal(checked["route_ids"], route_ids):
        raise RuntimeError("reference construction failed exact segment/endpoint/mode validation")
    order = rng.permutation(count)
    paths, conditions, route_ids, scene_indices = (
        paths[order],
        conditions[order],
        route_ids[order],
        scene_indices[order],
    )
    dataset = SampleDataset(
        paths,
        conditions,
        labels=route_ids,
        metadata={
            "name": "multimodal_navigation_2d",
            "dim": 2,
            "seed": int(seed),
            "split": split,
            "resolved_seed": int(resolved_seed),
            "n_scenes": int(n_scenes),
            "samples_per_scene": int(samples_per_scene),
            "path_length": int(path_length),
            "condition_layout": CONDITION_LAYOUT,
            "condition_dim": 8,
            "route_label_meaning": {"0": "top", "1": "bottom"},
            "route_id_is_condition": False,
            "jitter": float(jitter),
            "clearance": float(clearance),
            "margin": float(margin),
            "bounds": BOUNDS,
            "scenes": scene_conditions.tolist(),
            "scene_indices": scene_indices.tolist(),
            "wall_layout": "one axis-aligned wall; low/high corners are condition indices4:8",
            "reference_method": "two safe corner routes, outward-only corner jitter, corner-preserving resampling",
            "reference_exact_segment_validation": True,
        },
    )
    dataset.scene_indices = torch.from_numpy(scene_indices)
    dataset.scene_conditions = torch.from_numpy(scene_conditions)
    return dataset
