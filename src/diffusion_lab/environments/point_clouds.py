"""Seeded synthetic distributions in one, two, and three dimensions."""

from __future__ import annotations

import numpy as np

from .base import SampleDataset


def _parameters(n_samples: int, n_classes: int, noise: float, seed: int):
    if n_samples < 1 or n_classes < 1 or noise < 0:
        raise ValueError("n_samples and n_classes must be positive; noise must be nonnegative")
    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples, dtype=np.int64) % n_classes
    rng.shuffle(labels)
    return rng, labels


def gaussian_mixture(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    dim: int = 1,
    n_classes: int = 3,
    noise: float = 0.12,
    centers=None,
) -> SampleDataset:
    """Gaussian components; explicit centers also permit higher-dimensional tasks."""
    if dim < 1:
        raise ValueError("dim must be positive")
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != dim:
            raise ValueError("centers must have shape [n_classes, dim]")
        n_classes = len(centers)
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if centers is None:
        if dim == 1:
            centers = np.linspace(-1, 1, n_classes)[:, None]
        else:
            # A deterministic spread with no global RNG side effects.
            angles = np.arange(n_classes) * 2 * np.pi / n_classes
            centers = np.zeros((n_classes, dim))
            centers[:, 0] = np.cos(angles)
            centers[:, 1] = np.sin(angles)
            if dim > 2:
                centers[:, 2:] = rng.uniform(-0.8, 0.8, (n_classes, dim - 2))
    samples = centers[labels] + rng.normal(0, noise, (n_samples, dim))
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": f"gaussian_mixture_{dim}d",
            "dim": dim,
            "seed": seed,
            "centers": centers.tolist(),
            "noise": noise,
        },
    )


def n_moons(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    n_classes: int = 2,
    noise: float = 0.05,
    radius: float = 1.0,
) -> SampleDataset:
    """Two standard interleaving moons, or N rotated semicircular components."""
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if radius <= 0:
        raise ValueError("radius must be positive")
    theta = rng.uniform(0, np.pi, n_samples)
    samples = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    if n_classes == 2:
        second = labels == 1
        samples[second, 0] = 1 - samples[second, 0]
        samples[second, 1] = 0.5 - samples[second, 1]
        samples -= np.array([0.5, 0.25])
    elif n_classes > 2:
        angle = labels * 2 * np.pi / n_classes
        local_x, local_y = samples[:, 0].copy(), samples[:, 1].copy()
        samples[:, 0] = local_x * np.cos(angle) - local_y * np.sin(angle) + 1.5 * np.cos(angle)
        samples[:, 1] = local_x * np.sin(angle) + local_y * np.cos(angle) + 1.5 * np.sin(angle)
    samples = radius * samples + rng.normal(0, noise, samples.shape)
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": "n_moons",
            "dim": 2,
            "seed": seed,
            "n_classes": n_classes,
            "noise": noise,
            "radius": radius,
        },
    )


def rotating_rays(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    n_classes: int = 8,
    noise: float = 0.008,
    radius: float = 10 / 11,
) -> SampleDataset:
    """The notebook's rotating rays, with N interpreted as the total sample count.

    Its sqrt radial parameter and angular sweep are preserved. This distribution
    is intentionally named rays rather than moons.
    """
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if radius <= 0:
        raise ValueError("radius must be positive")
    parameter = rng.uniform(0, 1, n_samples)
    r = radius * np.sqrt(parameter)
    angle = parameter * 1.6 * np.pi / (n_classes * 0.9) + 2 * np.pi * labels / n_classes + np.pi / 2
    samples = r[:, None] * np.stack((np.cos(angle), np.sin(angle)), axis=1)
    samples += rng.normal(size=samples.shape) * noise * (r / radius)[:, None]
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": "rotating_rays",
            "dim": 2,
            "seed": seed,
            "n_classes": n_classes,
            "noise": noise,
            "radius": radius,
        },
    )


def rings(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    n_classes: int = 3,
    noise: float = 0.035,
    min_radius: float = 0.3,
    max_radius: float = 1.0,
) -> SampleDataset:
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if min_radius <= 0 or max_radius < min_radius:
        raise ValueError("require 0 < min_radius <= max_radius")
    radii = np.linspace(min_radius, max_radius, n_classes)[labels]
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    samples = radii[:, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    samples += rng.normal(0, noise, samples.shape)
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": "rings",
            "dim": 2,
            "seed": seed,
            "n_classes": n_classes,
            "noise": noise,
            "radii": np.linspace(min_radius, max_radius, n_classes).tolist(),
        },
    )


def spirals(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    n_classes: int = 3,
    noise: float = 0.025,
    turns: float = 1.5,
) -> SampleDataset:
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if turns <= 0:
        raise ValueError("turns must be positive")
    parameter = rng.uniform(0.05, 1, n_samples)
    angle = turns * 2 * np.pi * parameter + 2 * np.pi * labels / n_classes
    samples = parameter[:, None] * np.stack((np.cos(angle), np.sin(angle)), axis=1)
    samples += rng.normal(0, noise, samples.shape)
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": "spirals",
            "dim": 2,
            "seed": seed,
            "n_classes": n_classes,
            "noise": noise,
            "turns": turns,
        },
    )


def helix(
    n_samples: int = 2048,
    seed: int = 42,
    *,
    n_classes: int = 2,
    noise: float = 0.025,
    turns: float = 2.0,
    radius: float = 0.7,
) -> SampleDataset:
    rng, labels = _parameters(n_samples, n_classes, noise, seed)
    if turns <= 0 or radius <= 0:
        raise ValueError("turns and radius must be positive")
    parameter = rng.uniform(-1, 1, n_samples)
    angle = turns * np.pi * parameter + 2 * np.pi * labels / n_classes
    samples = np.stack((radius * np.cos(angle), radius * np.sin(angle), parameter), axis=1)
    samples += rng.normal(0, noise, samples.shape)
    return SampleDataset(
        samples,
        labels,
        num_classes=n_classes,
        metadata={
            "name": "helix_3d",
            "dim": 3,
            "seed": seed,
            "n_classes": n_classes,
            "noise": noise,
            "turns": turns,
            "radius": radius,
        },
    )
