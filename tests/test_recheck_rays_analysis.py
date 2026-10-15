"""Known-geometry controls catch arm/gap failures hidden by radial/occupancy metrics."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from diffusion_lab.environments import make_environment
from recheck.rays_analysis import analyse_rays, phase_coordinates, plot_rays_diagnostics


def ideal_points(n_rays=16, radius=10 / 11, points_per_ray=64, phase_offsets=None):
    u = np.tile(np.linspace(0.07, 0.99, points_per_ray), n_rays)
    arm = np.repeat(np.arange(n_rays), points_per_ray)
    theta = math.pi / 2 + arm * 2 * math.pi / n_rays + u * 1.6 * math.pi / (0.9 * n_rays)
    if phase_offsets is not None:
        theta += np.asarray(phase_offsets) * 2 * math.pi / n_rays
    radial = radius * np.sqrt(u)
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta))), arm


@pytest.mark.parametrize("n_rays", [1, 2, 4, 16])
def test_coordinate_assignment_matches_original_environment_without_noise(n_rays):
    dataset = make_environment("rotating_rays", 1024, seed=83, n_classes=n_rays, noise=0)
    geometry = phase_coordinates(dataset.samples, n_rays=n_rays)
    outer = geometry["outer_mask"]
    assert np.array_equal(geometry["nearest_arm"][outer], dataset.conditions.numpy()[outer])
    assert geometry["absolute_phase_distance_sector_fraction"][outer].max() < 2e-6
    assert geometry["kappa_radians"] == pytest.approx(math.radians(320 / n_rays))


def test_exact_ideal_curves_identity_and_permutation_invariants():
    points, arms = ideal_points()
    geometry = phase_coordinates(points)
    assert np.array_equal(geometry["nearest_arm"], arms)
    assert geometry["absolute_phase_distance_sector_fraction"].max() < 2e-14
    result = analyse_rays(points, points[::-1].copy())
    metrics = result["metrics"]
    assert metrics["nearest_arm_occupancy_tv_outer"] == 0
    assert metrics["absolute_phase_distance_w1_sector_fraction"] == pytest.approx(0, abs=1e-15)
    assert metrics["radial_w1_raw"] == 0 and metrics["radial_ks_statistic"] == 0
    assert result["generated_phase_coverage_valid"]
    assert result["generated"]["outer_count"] == len(points)
    assert result["generated"]["phase_outer"]["circular_concentration"] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_gap_filling_control_preserves_radial_and_occupancy_but_fails_phase():
    real, _ = ideal_points(points_per_ray=200)
    # Uniform within-sector angular fill at IDENTICAL radii and arm counts.
    offsets = np.tile(np.linspace(-0.4975, 0.4975, 200), 16)
    filled, _ = ideal_points(points_per_ray=200, phase_offsets=offsets)
    result = analyse_rays(filled, real)
    metrics = result["metrics"]
    assert metrics["nearest_arm_occupancy_tv_outer"] == 0
    assert metrics["radial_w1_raw"] < 1e-15
    assert metrics["absolute_phase_distance_w1_sector_fraction"] > 0.24
    assert metrics["off_curve_tail_fraction_differences"][
        "distance_gt_0.10_sector"
    ] == pytest.approx(0.8)
    assert result["generated"]["phase_outer"]["circular_concentration"] < 0.01


def test_single_arm_mode_collapse_has_correct_occupancy_total_variation():
    real, _ = ideal_points()
    generated = real[:64]
    result = analyse_rays(generated, real)
    assert result["metrics"]["nearest_arm_occupancy_tv_outer"] == pytest.approx(15 / 16)
    assert result["metrics"]["absolute_phase_distance_w1_sector_fraction"] < 1e-14
    assert result["generated"]["nearest_arm_occupancy_outer"][0] == 1
    assert sum(result["generated"]["nearest_arm_occupancy_outer"]) == 1


def test_central_collapse_does_not_receive_zero_phase_or_occupancy_error():
    reference, _ = ideal_points()
    collapsed = np.zeros((128, 2))
    result = analyse_rays(collapsed, reference)
    assert not result["generated_phase_coverage_valid"]
    assert result["generated"]["outer_count"] == 0
    assert result["generated"]["nearest_arm_occupancy_outer"] is None
    assert result["metrics"]["nearest_arm_occupancy_tv_outer"] is None
    assert result["metrics"]["absolute_phase_distance_w1_sector_fraction"] is None
    assert result["metrics"]["outer_coverage_fraction_difference"] == -1
    assert result["metrics"]["radial_w1_raw"] > 0.5
    json.dumps(result, allow_nan=False)


def test_outer_mask_is_strict_and_keeps_radial_coverage_of_all_points():
    radius = 10 / 11
    points = np.array(
        [[0.0, 0.0], [radius * 0.25, 0.0], [radius * 0.250001, 0.0], [radius * 1.3, 0.0]]
    )
    geometry = phase_coordinates(points)
    assert geometry["outer_mask"].tolist() == [False, False, True, True]
    reference, _ = ideal_points()
    result = analyse_rays(points, reference)
    assert result["generated"]["sample_count"] == 4
    assert result["generated"]["outer_fraction"] == 0.5
    assert result["generated"]["radial"]["outside_nominal_radius_fraction"] == 0.25
    assert result["generated"]["radial"]["cdf_at_fixed_normalized_nodes"][0] == 0.25


def test_raw_units_scale_while_normalized_geometry_remains_identical():
    reference, _ = ideal_points()
    generated = reference * 1.1
    base = analyse_rays(generated, reference)
    scaled = analyse_rays(generated * 3, reference * 3, radius=3 * 10 / 11)
    assert scaled["metrics"]["radial_w1_raw"] == pytest.approx(3 * base["metrics"]["radial_w1_raw"])
    assert scaled["metrics"]["radial_w1_normalized"] == pytest.approx(
        base["metrics"]["radial_w1_normalized"]
    )
    assert (
        scaled["metrics"]["nearest_arm_occupancy_tv_outer"]
        == base["metrics"]["nearest_arm_occupancy_tv_outer"]
    )
    assert scaled["metrics"]["absolute_phase_distance_w1_sector_fraction"] == pytest.approx(
        base["metrics"]["absolute_phase_distance_w1_sector_fraction"]
    )


def test_noisy_real_reference_is_compared_as_observed_without_gaussian_calibration():
    real = make_environment("rotating_rays", 8192, seed=17, n_classes=16).samples
    original = real.clone()
    result = analyse_rays(real, real[torch.arange(len(real) - 1, -1, -1)])
    assert torch.equal(real, original)
    assert result["metrics"]["radial_w1_raw"] == 0
    assert result["metrics"]["absolute_phase_distance_w1_sector_fraction"] == 0
    assert result["reference"]["phase_outer"]["mean_absolute_sector_fraction"] > 0
    assert result["reference"]["radial"]["outside_nominal_radius_fraction"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "points",
    [
        torch.zeros(3, 3),
        torch.empty(0, 2),
        torch.tensor([[float("nan"), 0.0]]),
        torch.tensor([[float("inf"), 0.0]]),
    ],
)
def test_invalid_shapes_or_nonfinite_data_fail(points):
    reference, _ = ideal_points()
    with pytest.raises(ValueError):
        analyse_rays(points, reference)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rays": True},
        {"n_rays": 0},
        {"radius": 0},
        {"outer_radius_fraction": 1.0},
        {"noise": -0.1},
    ],
)
def test_invalid_geometry_parameters_fail(kwargs):
    reference, _ = ideal_points()
    with pytest.raises(ValueError):
        analyse_rays(reference, reference, **kwargs)


def test_plots_use_same_limits_and_support_collapsed_generation(tmp_path):
    reference, _ = ideal_points(points_per_ray=16)
    figure, axes = plot_rays_diagnostics(reference, reference)
    normal_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes.flat]
    figure.savefig(tmp_path / "rays_geometry.png")
    assert (tmp_path / "rays_geometry.png").stat().st_size > 1000
    plt.close(figure)
    figure, axes = plot_rays_diagnostics(np.zeros((32, 2)), reference)
    collapsed_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes.flat]
    assert normal_limits == collapsed_limits
    plt.close(figure)
