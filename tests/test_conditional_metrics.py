"""Conditional geometry checks use synthetic labelled points; no learned models."""

import json
import math

import pytest
import torch

from diffusion_lab.conditional_metrics import conditional_routing_metrics, nearest_ray_ids
from diffusion_lab.environments import make_environment


def curve_points(labels, u=0.6, n_classes=16, radius=10 / 11):
    labels = torch.as_tensor(labels, dtype=torch.long)
    u = torch.as_tensor(u, dtype=torch.float64).expand(len(labels))
    angle = math.pi / 2 + labels * (2 * math.pi / n_classes) + u * 1.6 * math.pi / (0.9 * n_classes)
    return radius * u.sqrt()[:, None] * torch.stack((angle.cos(), angle.sin()), 1)


def test_noise_free_generator_recovers_actual_dataset_labels_and_conditions():
    data = make_environment("rotating_rays", 4096, seed=819, n_classes=16, noise=0)
    assert torch.equal(data.labels, data.conditions)
    assert torch.equal(nearest_ray_ids(data.samples), data.labels)
    result = conditional_routing_metrics(data.samples, data.conditions)
    assert result["accuracy_all"] == result["accuracy_outer"] == 1
    assert sum(result["per_class_counts"]) == 4096
    assert all(count == 256 for count in result["per_class_counts"])
    assert result["requested_phase_error"]["mean_absolute_sector_fraction_outer"] < 1e-6
    assert result["initial_routing"] is None
    json.dumps(result, allow_nan=False)


def test_rotation_hits_wrong_labels_despite_exact_nearest_curve_residual():
    targets = torch.arange(16)
    actual = (targets + 1) % 16
    points = curve_points(actual)
    result = conditional_routing_metrics(points, targets)
    assert result["accuracy_all"] == result["accuracy_outer"] == 0
    assert result["requested_phase_error"]["mean_absolute_sector_fraction_outer"] == pytest.approx(
        1, abs=2e-6
    )
    assert result["nearest_phase_residual"]["mean_absolute_sector_fraction_outer"] < 2e-6
    assert all(result["confusion_counts"][k][(k + 1) % 16] == 1 for k in range(16))


def test_zero_origin_is_unassigned_and_outer_accuracy_is_undefined():
    points = torch.zeros(16, 2)
    assert nearest_ray_ids(points).tolist() == [-1] * 16
    result = conditional_routing_metrics(points, torch.arange(16), initial_noise=points)
    assert result["accuracy_all"] == result["correct_outer_fraction_of_all"] == 0
    assert result["accuracy_outer"] is None and result["outer_coverage"] == 0
    assert result["unassigned_count"] == 16
    assert sum(map(sum, result["confusion_counts"])) == 0
    assert result["requested_phase_error"]["mean_absolute_radians_all"] is None
    assert result["initial_routing"]["output_matches_initial_fraction"] == 0
    assert (
        result["initial_routing"]["output_matches_target_when_target_differs_from_initial"] is None
    )
    json.dumps(result, allow_nan=False)


def test_tiny_radius_apparent_success_does_not_hide_zero_outer_coverage():
    targets = torch.arange(16)
    points = curve_points(targets, u=1e-12)
    result = conditional_routing_metrics(points, targets)
    assert result["accuracy_all"] == 1
    assert result["accuracy_outer"] is None and result["outer_coverage"] == 0
    assert result["correct_outer_fraction_of_all"] == 0


def test_same_noise_all_requested_labels_distinguishes_conditioned_and_ignored():
    # Every exact noise vector occurs once for each requested label.
    initial_labels = torch.arange(16)
    bank = curve_points(initial_labels, u=0.7)
    noise = bank.repeat(16, 1)
    targets = torch.arange(16).repeat_interleave(16)
    correct = conditional_routing_metrics(curve_points(targets), targets, noise)
    ignored = conditional_routing_metrics(
        curve_points(initial_labels).repeat(16, 1), targets, noise
    )
    assert correct["accuracy_outer"] == 1
    assert ignored["accuracy_outer"] == pytest.approx(1 / 16)
    assert correct["initial_routing"]["output_matches_target_when_target_differs_from_initial"] == 1
    assert correct["initial_routing"]["output_changes_arm_when_target_differs_from_initial"] == 1
    assert ignored["initial_routing"]["output_matches_target_when_target_differs_from_initial"] == 0
    assert ignored["initial_routing"]["output_stays_initial_when_target_differs_from_initial"] == 1
    assert ignored["confusion_row_normalized"] == [[1 / 16] * 16] * 16
    assert correct["counterfactual_same_noise"]["complete_noise_groups"] == 16
    assert (
        correct["counterfactual_same_noise"]["mean_distinct_output_arms_per_complete_noise"] == 16
    )
    assert ignored["counterfactual_same_noise"]["mean_distinct_output_arms_per_complete_noise"] == 1
    assert correct["counterfactual_same_noise"]["fraction_complete_groups_all_targets_correct"] == 1
    assert ignored["counterfactual_same_noise"]["fraction_complete_groups_all_targets_correct"] == 0


def test_incomplete_classes_have_null_rows_not_zero_accuracy():
    targets = torch.tensor([0, 1])
    result = conditional_routing_metrics(curve_points(targets), targets)
    assert result["per_class"][2]["count"] == 0
    assert result["per_class"][2]["accuracy_all"] is None
    assert result["confusion_row_normalized"][2] == [None] * 16


@pytest.mark.parametrize(
    "points",
    [
        torch.tensor([[float("nan"), 0.0]]),
        torch.tensor([[float("inf"), 0.0]]),
        torch.empty(0, 2),
        torch.ones(2, 3),
        torch.ones(2, 2, dtype=torch.complex64),
        torch.tensor([[1e300, 1e300]], dtype=torch.float64),
    ],
)
def test_nonfinite_or_invalid_coordinates_rejected(points):
    with pytest.raises(ValueError):
        nearest_ray_ids(points)
    with pytest.raises(ValueError):
        conditional_routing_metrics(points, torch.zeros(len(points), dtype=torch.long))


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.tensor([0], dtype=torch.uint8),
        torch.tensor([-1]),
        torch.tensor([16]),
        torch.tensor([[0]]),
        torch.tensor([0, 1]),
    ],
)
def test_targets_are_strict_categorical_ids_with_exact_batch_shape(targets):
    with pytest.raises(ValueError, match="targets"):
        conditional_routing_metrics(curve_points([0]), targets)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_classes": 0},
        {"n_classes": True},
        {"n_classes": 2.5},
        {"radius": 0},
        {"radius": float("inf")},
        {"minimum_radius_fraction": 1},
        {"minimum_radius_fraction": -0.1},
    ],
)
def test_invalid_geometry_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        conditional_routing_metrics(curve_points([0]), torch.tensor([0]), **kwargs)


def test_initial_noise_must_match_shape_and_be_finite():
    for initial in (torch.ones(2, 2), torch.tensor([[float("nan"), 0.0]])):
        with pytest.raises(ValueError, match="initial_noise"):
            conditional_routing_metrics(curve_points([0]), torch.tensor([0]), initial)


def test_generic_geometry_and_scale_invariance_without_fixed_16_label_assumption():
    for count in (1, 2, 3, 8, 32):
        targets = torch.arange(count)
        points = curve_points(targets, n_classes=count)
        assert torch.equal(nearest_ray_ids(points, n_classes=count), targets)
        result = conditional_routing_metrics(points * 3, targets, n_classes=count, radius=30 / 11)
        assert result["accuracy_outer"] == 1
        assert result["requested_phase_error"]["mean_absolute_sector_fraction_outer"] < 4e-6
