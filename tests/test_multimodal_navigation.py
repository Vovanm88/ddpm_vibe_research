"""Independent geometry and no-condition-leak checks for two-mode A/B scenes."""

import json

import numpy as np
import pytest
import torch

from diffusion_lab.environments import WallNavigation
from diffusion_lab.environments.multimodal_navigation import (
    _resample_corners,
    classify_navigation_modes,
    evaluate_navigation,
    make_multimodal_navigation,
)


def test_each_exact_eight_number_condition_contains_both_private_route_labels():
    data = make_multimodal_navigation(n_scenes=7, samples_per_scene=10)
    assert data.samples.shape == (70, 32, 2) and data.sample_shape == (32, 2)
    assert data.condition_kind == "continuous" and data.condition_dim == 8
    assert data.num_classes == 0
    assert set(data[0]) == {"samples", "conditions"}  # No privileged route ID in batches.
    assert data.labels.shape == (70,) and data.labels.dtype == torch.long
    assert data.scene_conditions.shape == (7, 8)
    for scene in range(7):
        selected = data.scene_indices == scene
        assert torch.equal(data.conditions[selected], data.scene_conditions[scene].expand(10, -1))
        assert torch.bincount(data.labels[selected], minlength=2).tolist() == [5, 5]
    assert torch.equal(data.samples[:, 0], data.conditions[:, :2])
    assert torch.equal(data.samples[:, -1], data.conditions[:, 2:4])
    assert torch.equal(classify_navigation_modes(data.samples, data.conditions), data.labels)
    assert not data.metadata["route_id_is_condition"]
    json.dumps(data.metadata, allow_nan=False)


@pytest.mark.parametrize("length", [4, 5, 7, 32, 64])
def test_all_reference_segments_checked_by_independent_existing_wall_navigation(length):
    data = make_multimodal_navigation(
        n_scenes=5, samples_per_scene=12, path_length=length, jitter=0.05
    )
    for scene, condition in enumerate(data.scene_conditions):
        independent = WallNavigation(
            2, walls=condition[4:].numpy().reshape(1, 2, 2), path_length=length, margin=0.035
        )
        assert independent.collision_free(data.samples[data.scene_indices == scene]).all()
    result = evaluate_navigation(data.samples, data.conditions)
    assert (
        result["success_fraction"]
        == result["collision_free_fraction"]
        == result["bounds_fraction"]
        == 1
    )
    assert result["endpoint_mse"] == result["mean_goal_error"] == result["mean_start_error"] == 0
    assert result["mode_coverage"]["both_modes_fraction_all_conditions"] == 1
    assert result["mode_coverage"]["mean_mode_balance_tv_successful_assigned"] == 0
    json.dumps(result, allow_nan=False)


def test_corner_preserving_resampling_avoids_naive_between_waypoint_corner_cut():
    control = np.array([[[-0.8, 0], [-0.14, 0.64], [0.14, 0.64], [0.8, 0]]])
    scene = WallNavigation(2, walls=[[[-0.1, -0.6], [0.1, 0.6]]], margin=0.035)
    reference = _resample_corners(control, 4)
    assert np.array_equal(reference, control)
    assert scene.collision_free(reference).all()
    distance = np.r_[0, np.linalg.norm(np.diff(control[0], axis=0), axis=1).cumsum()]
    grid = np.linspace(0, distance[-1], 4)
    naive = np.stack([np.interp(grid, distance, control[0, :, axis]) for axis in range(2)], axis=1)
    assert not scene.collision_free(naive)


def test_scenes_follow_documented_ranges_and_all_sample_conditions_keep_wall_geometry():
    data = make_multimodal_navigation(n_scenes=128, samples_per_scene=2, seed=19)
    condition = data.scene_conditions.numpy()
    assert ((condition[:, 0] >= -0.85) & (condition[:, 0] <= -0.7)).all()
    assert ((condition[:, 2] >= 0.7) & (condition[:, 2] <= 0.85)).all()
    assert (np.abs(condition[:, [1, 3]]) <= 0.15).all()
    centre = (condition[:, 4:6] + condition[:, 6:8]) / 2
    size = condition[:, 6:8] - condition[:, 4:6]
    assert (np.abs(centre[:, 0]) <= 0.12).all() and (np.abs(centre[:, 1]) <= 0.1).all()
    assert ((size[:, 0] >= 0.20 - 1e-7) & (size[:, 0] <= 0.35 + 1e-7)).all()
    assert ((size[:, 1] >= 0.6 - 1e-7) & (size[:, 1] <= 1.0 + 1e-7)).all()
    assert len(torch.unique(data.scene_conditions, dim=0)) == 128


def test_deterministic_split_streams_are_disjoint_and_do_not_touch_global_rng():
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    train = make_multimodal_navigation(5, 6, seed=113, split="train")
    repeated = make_multimodal_navigation(5, 6, seed=113, split="train")
    validation = make_multimodal_navigation(5, 6, seed=113, split="validation")
    test = make_multimodal_navigation(5, 6, seed=113, split="test")
    assert torch.equal(train.samples, repeated.samples)
    assert torch.equal(train.conditions, repeated.conditions) and torch.equal(
        train.labels, repeated.labels
    )
    assert train.metadata == repeated.metadata
    assert len({d.metadata["resolved_seed"] for d in (train, validation, test)}) == 3
    sets = [{tuple(row.tolist()) for row in d.scene_conditions} for d in (train, validation, test)]
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    assert torch.equal(torch_state, torch.random.get_rng_state())
    current = np.random.get_state()
    assert (
        numpy_state[0] == current[0]
        and np.array_equal(numpy_state[1], current[1])
        and numpy_state[2:] == current[2:]
    )


def test_thin_wall_crossed_between_free_waypoints_is_invalid_and_never_a_mode():
    paths = torch.tensor([[[-0.8, 0], [0.8, 0]]])
    condition = torch.tensor([[-0.8, 0, 0.8, 0, -0.0005, -0.6, 0.0005, 0.6]])
    result = evaluate_navigation(paths, condition, margin=0)
    assert result["endpoint_success_fraction"] == result["bounds_fraction"] == 1
    assert result["collision_free_fraction"] == result["success_fraction"] == 0
    assert result["per_sample"]["route_ids"] == [-1]
    assert result["mode_coverage"]["both_modes_fraction_all_conditions"] == 0
    assert result["mode_coverage"]["both_modes_fraction_conditions_with_success"] is None


def test_closed_contact_parallel_inside_and_corner_crossings_are_collisions():
    condition = np.array([[-0.8, 0, 0.8, 0, -0.1, -0.6, 0.1, 0.6]])
    for paths in (
        np.array([[[-0.8, 0.6], [0.8, 0.6]]]),
        np.array([[[0, -0.9], [0, 0.9]]]),
        np.array([[[-0.2, 0.65], [0.2, 0.55]]]),
        np.array([[[0, 0], [0, 0]]]),
    ):
        result = evaluate_navigation(paths, condition, margin=0)
        assert result["collision_free_fraction"] == 0


def test_wrong_endpoints_or_out_of_bounds_paths_cannot_inflate_route_coverage():
    data = make_multimodal_navigation(1, 2, path_length=4, jitter=0)
    paths = data.samples.clone()
    paths[:, 0, 0] -= 0.2
    result = evaluate_navigation(paths, data.conditions)
    assert result["endpoint_success_fraction"] == result["success_fraction"] == 0
    assert result["successful_mode_counts"] == [0, 0]
    paths = data.samples.clone()
    paths[:, 1, 1] = 1.2
    assert evaluate_navigation(paths, data.conditions)["bounds_fraction"] == 0
    assert classify_navigation_modes(paths, data.conditions).tolist() == [-1, -1]


def test_mode_coverage_grouped_per_exact_condition_not_global_frequency():
    data = make_multimodal_navigation(2, 4, path_length=8)
    collapsed = data.samples.clone()
    for scene in (0, 1):
        mask = data.scene_indices == scene
        # Global top/bottom frequency remains balanced, but EVERY condition collapses.
        source = torch.where(mask & (data.labels == scene))[0][0]
        collapsed[mask] = data.samples[source]
    result = evaluate_navigation(collapsed, data.conditions)
    assert result["success_fraction"] == 1 and result["successful_mode_counts"] == [4, 4]
    assert result["mode_coverage"]["both_modes_fraction_all_conditions"] == 0
    assert result["mode_coverage"]["mean_mode_balance_tv_successful_assigned"] == 0.5


def test_all_conditions_denominator_does_not_hide_failed_layouts():
    data = make_multimodal_navigation(2, 4, path_length=8)
    paths = data.samples.clone()
    mask = data.scene_indices == 1
    fraction = torch.linspace(0, 1, 8)[None, :, None]
    paths[mask] = (1 - fraction) * data.conditions[mask, :2, None].transpose(
        1, 2
    ) + fraction * data.conditions[mask, 2:4, None].transpose(1, 2)
    result = evaluate_navigation(paths, data.conditions)
    assert result["success_fraction"] == 0.5
    assert result["mode_coverage"]["both_modes_fraction_all_conditions"] == 0.5
    assert result["mode_coverage"]["both_modes_fraction_conditions_with_success"] == 1


def test_successful_loopy_path_crossing_above_and_below_remains_unassigned():
    condition = np.array([[-0.8, 0, 0.8, 0, -0.1, -0.4, 0.1, 0.4]])
    path = np.array(
        [
            [
                [-0.8, 0],
                [-0.3, 0.7],
                [0.3, 0.7],
                [0.5, 0.8],
                [0.5, -0.7],
                [-0.3, -0.7],
                [-0.5, -0.8],
                [-0.5, 0.8],
                [0.3, 0.8],
                [0.8, 0],
            ]
        ]
    )
    result = evaluate_navigation(path, condition)
    assert result["success_fraction"] == 1
    assert result["per_sample"]["route_ids"] == [-1]
    assert result["successful_mode_counts"] == [0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_scenes": 0},
        {"samples_per_scene": 1},
        {"samples_per_scene": 3},
        {"path_length": 3},
        {"seed": -1},
        {"split": "unknown"},
        {"jitter": -0.1},
        {"jitter": float("nan")},
        {"clearance": 0.035},
        {"clearance": 0.4},
    ],
)
def test_invalid_factory_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        make_multimodal_navigation(**kwargs)


def test_invalid_evaluation_shapes_and_nonfinite_rejected():
    data = make_multimodal_navigation(1, 2)
    for paths, conditions in (
        (data.samples[:1], data.conditions),
        (data.samples[:, :, 0], data.conditions),
        (data.samples * float("nan"), data.conditions),
        (data.samples, data.conditions * float("inf")),
    ):
        with pytest.raises(ValueError):
            evaluate_navigation(paths, conditions)
    bad = data.conditions.clone()
    bad[:, 6] = bad[:, 4]
    with pytest.raises(ValueError, match="ordered"):
        evaluate_navigation(data.samples, bad)
    for kwargs in ({"margin": -0.1}, {"endpoint_tol": float("nan")}):
        with pytest.raises(ValueError):
            evaluate_navigation(data.samples, data.conditions, **kwargs)
