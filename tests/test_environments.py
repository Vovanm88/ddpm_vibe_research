"""Behavioral checks for research data and obstacle-aware reference routes."""

import json

import numpy as np
import pytest
import torch

from diffusion_lab.environments import (
    SampleDataset,
    WallNavigation,
    collision_loss,
    endpoint_loss,
    make_environment,
    walls_from_conditions,
)


@pytest.mark.parametrize(
    "name, shape",
    [
        ("gaussian_mixture_1d", (1,)),
        ("n_moons", (2,)),
        ("rotating_rays", (2,)),
        ("rings", (2,)),
        ("spirals", (2,)),
        ("blobs_3d", (3,)),
        ("helix_3d", (3,)),
    ],
)
def test_point_environment_shapes_and_reproducibility(name, shape):
    first = make_environment(name, n_samples=37, seed=123)
    same = make_environment(name, n_samples=37, seed=123)
    different = make_environment(name, n_samples=37, seed=124)
    assert first.samples.shape == (37, *shape)
    assert first.sample_shape == shape
    assert first.samples.dtype == torch.float32
    assert first.conditions.dtype == torch.long
    assert first.condition_kind == "categorical"
    assert torch.equal(first.samples, same.samples)
    assert not torch.equal(first.samples, different.samples)
    assert set(first[0]) == {"samples", "conditions"}
    json.dumps(first.metadata)


def test_default_is_notebook_rotating_rays_with_total_sample_count():
    dataset = make_environment(n_samples=41, n_classes=8, noise=0, seed=10)
    assert len(dataset) == 41
    assert dataset.metadata["name"] == "rotating_rays"
    assert dataset.samples.norm(dim=-1).max() <= 10 / 11


def test_moons_two_class_geometry():
    dataset = make_environment("moons", n_samples=64, noise=0, seed=10)
    upper = dataset.samples[dataset.labels == 0]
    lower = dataset.samples[dataset.labels == 1]
    assert torch.all(upper[:, 1] >= -0.25)
    assert torch.all(lower[:, 1] <= 0.25)
    assert dataset.num_classes == 2


def test_local_rng_does_not_modify_global_numpy_state():
    np.random.seed(5)
    expected = np.random.random(8)
    np.random.seed(5)
    make_environment("rays", n_samples=20, seed=81)
    assert np.array_equal(np.random.random(8), expected)


def test_dataset_none_and_continuous_condition_contracts():
    plain = SampleDataset(np.zeros((4, 3)))
    assert plain.condition_kind == "none"
    assert set(plain[0]) == {"samples"}
    continuous = SampleDataset(np.zeros((4, 3)), np.zeros((4, 5)))
    assert continuous.condition_kind == "continuous"
    assert continuous.condition_dim == 5
    assert continuous.labels is None
    with pytest.raises(ValueError, match="integer"):
        SampleDataset(np.zeros((4, 3)), [0.5, 1, 1, 1])


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_navigation_reference_paths_and_condition_layout(dim):
    dataset = make_environment(f"navigation_{dim}d", n_samples=5, seed=21, path_length=24)
    same = make_environment(f"navigation_{dim}d", n_samples=5, seed=21, path_length=24)
    assert dataset.samples.shape == (5, 24, dim)
    assert dataset.sample_shape == (24, dim)
    assert dataset.condition_kind == "continuous"
    assert torch.equal(dataset.samples, same.samples)
    assert dataset.environment.collision_free(dataset.samples).all()
    assert torch.allclose(dataset.samples[:, 0], dataset.conditions[:, :dim])
    assert torch.allclose(dataset.samples[:, -1], dataset.conditions[:, dim : 2 * dim])
    assert endpoint_loss(dataset.samples, dataset.conditions) == 0
    walls = walls_from_conditions(dataset.conditions, dim)
    assert walls.shape == (5, len(dataset.environment.walls), 2, dim)
    assert collision_loss(dataset.samples, walls, margin=dataset.environment.margin) < 1e-10
    json.dumps(dataset.metadata)


def test_wall_crossing_between_free_waypoints_is_detected():
    scene = WallNavigation(2, walls=[[[-0.05, -0.6], [0.05, 0.6]]], margin=0.01)
    crossing = np.array([[-0.8, 0.0], [0.8, 0.0]])
    assert not scene.collision_free(crossing)
    route = scene.plan(crossing[0], crossing[-1])
    assert scene.collision_free(route)
    assert np.max(np.abs(route[:, 1])) > 0.61
    assert np.linalg.norm(np.diff(route, axis=0), axis=-1).sum() > 1.6


def test_diagonal_corner_cut_and_parallel_segment_collision():
    scene = WallNavigation(2, walls=[[[0.0, 0.0], [0.5, 0.5]]], margin=0)
    assert not scene.collision_free([[-0.2, 0.4], [0.4, -0.2]])
    assert not scene.collision_free([[0.25, -0.5], [0.25, 0.8]])
    assert scene.collision_free([[-0.2, -0.2], [-0.2, 0.8]])


def test_thin_wall_segment_loss_has_finite_gradients():
    paths = torch.tensor([[[-0.8, 0.1], [0.8, 0.15]]], requires_grad=True)
    thin_wall = torch.tensor([[[-0.012, -0.6], [0.012, 0.6]]])
    # Two interpolation samples cannot see this wall: slab overlap must.
    value = collision_loss(paths, thin_wall, margin=0, interpolation_steps=2)
    assert value > 0
    value.backward()
    assert torch.isfinite(paths.grad).all()
    assert paths.grad.abs().sum() > 0
    safe = torch.tensor([[[-0.8, 0.8], [0.8, 0.8]]], requires_grad=True)
    safe_loss = collision_loss(safe, thin_wall, margin=0)
    assert safe_loss == 0
    safe_loss.backward()
    assert torch.isfinite(safe.grad).all()


def test_empty_walls_loss_and_stationary_route():
    scene = WallNavigation(1)
    route = scene.plan([0.2], [0.2])
    assert np.allclose(route, 0.2)
    paths = torch.as_tensor(route[None]).requires_grad_(True)
    value = collision_loss(paths, scene.walls)
    value.backward()
    assert value == 0
    assert torch.isfinite(paths.grad).all()


def test_disconnected_1d_obstacle_has_no_reference_route():
    scene = WallNavigation(1, walls=[[[-0.1], [0.1]]], margin=0)
    with pytest.raises(ValueError, match="no collision-free route"):
        scene.plan([-0.8], [0.8])


def test_analytical_field_is_negative_gradient_of_potential():
    scene = WallNavigation(2)
    points = torch.tensor([[-0.7, 0.2], [0.6, -0.5]], requires_grad=True)
    goal = torch.tensor([0.8, 0.1])
    potential = scene.potential(points, goal)
    gradient = torch.autograd.grad(potential.sum(), points)[0]
    assert torch.allclose(scene.vector_field(points, goal), -gradient)
    assert np.isfinite(scene.vector_field(points.detach().numpy(), goal)).all()
