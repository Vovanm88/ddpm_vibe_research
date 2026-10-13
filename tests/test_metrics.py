"""Distribution and navigation metrics must detect meaningful failure modes."""

import math

import numpy as np
import pytest
import torch

from diffusion_lab.environments import SampleDataset, WallNavigation
from diffusion_lab.metrics import evaluate_samples, sliced_wasserstein


def test_sliced_wasserstein_has_exact_one_dimensional_shift_and_permutation_invariance():
    points = np.array([[-3.0], [-1.0], [2.0], [4.0]])
    assert sliced_wasserstein(points, points[::-1], seed=7) == 0
    assert sliced_wasserstein(points, points + 2.5, seed=7) == pytest.approx(2.5)


def test_per_class_metrics_detect_swapped_classes_with_identical_marginal():
    points = torch.tensor([[-3.0], [-2.0], [2.0], [3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    dataset = SampleDataset(points, labels)
    correct = evaluate_samples(dataset, points, labels)
    assert correct["sliced_wasserstein"] == 0
    assert correct["class_0_sliced_wasserstein"] == 0
    swapped = evaluate_samples(dataset, points, 1 - labels)
    assert swapped["sliced_wasserstein"] == 0
    assert swapped["class_0_sliced_wasserstein"] == pytest.approx(5.0)
    assert swapped["class_1_sliced_wasserstein"] == pytest.approx(5.0)


def test_navigation_metrics_detect_segment_crossing_between_free_waypoints():
    environment = WallNavigation(dim=2, walls=[[[-0.2, -0.4], [0.2, 0.4]]], path_length=4, margin=0)
    safe = np.array([[-0.8, 0.0], [-0.4, 0.8], [0.4, 0.8], [0.8, 0.0]])
    crossing = np.array([[-0.8, 0.0], [-0.4, 0.0], [0.4, 0.0], [0.8, 0.0]])
    conditions = np.repeat(environment.encode_condition(safe[0], safe[-1])[None], 2, axis=0)
    dataset = SampleDataset(np.repeat(safe[None], 2, axis=0), conditions)
    dataset.environment = environment
    generated = np.stack((safe, crossing))
    metrics = evaluate_samples(dataset, generated, conditions)
    assert metrics["collision_free_fraction"] == 0.5
    assert metrics["endpoint_error"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["mean_path_length"] == pytest.approx((2 * math.sqrt(0.8) + 0.8 + 1.6) / 2)
    generated[:, -1, 0] += 0.1
    moved = evaluate_samples(dataset, generated, conditions)
    assert moved["endpoint_error"] == pytest.approx(0.05, abs=1e-7)


def test_nonfinite_samples_return_finite_diagnostic_instead_of_invalid_distance():
    dataset = SampleDataset(torch.ones(2, 2))
    generated = np.ones((2, 2))
    generated[0, 0] = np.nan
    assert evaluate_samples(dataset, generated) == {"finite_fraction": 0.75}
    with pytest.raises(ValueError, match="finite"):
        sliced_wasserstein(dataset.samples, generated)
