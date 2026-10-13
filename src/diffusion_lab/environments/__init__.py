"""Environment registry with a uniform dataset contract."""

from __future__ import annotations

from .base import SampleDataset
from .navigation import WallNavigation, collision_loss, endpoint_loss, walls_from_conditions
from .point_clouds import gaussian_mixture, helix, n_moons, rings, rotating_rays, spirals

ENVIRONMENT_NAMES = (
    "gaussian_mixture_1d",
    "n_moons",
    "rotating_rays",
    "rings",
    "spirals",
    "blobs_3d",
    "helix_3d",
    "navigation_1d",
    "navigation_2d",
    "navigation_3d",
)


def make_environment(
    name: str = "rotating_rays", n_samples: int = 2048, seed: int = 42, **kwargs
) -> SampleDataset:
    """Build a seeded dataset; N always means the total number of examples."""
    normalized = name.lower().replace("-", "_")
    aliases = {
        "moons": "n_moons",
        "nmoons": "n_moons",
        "rays": "rotating_rays",
        "gaussian_mixture": "gaussian_mixture_1d",
        "mixture_1d": "gaussian_mixture_1d",
        "blobs": "blobs_3d",
        "helix": "helix_3d",
        "walls": "navigation_2d",
        "navigation": "navigation_2d",
        "wall_navigation": "navigation_2d",
        "wall_navigation_1d": "navigation_1d",
        "wall_navigation_2d": "navigation_2d",
        "wall_navigation_3d": "navigation_3d",
        "corridor_1d": "navigation_1d",
    }
    normalized = aliases.get(normalized, normalized)
    functions = {
        "n_moons": n_moons,
        "rotating_rays": rotating_rays,
        "rings": rings,
        "spirals": spirals,
        "helix_3d": helix,
    }
    if normalized in functions:
        return functions[normalized](n_samples, seed, **kwargs)
    if normalized in ("gaussian_mixture_1d", "blobs_3d"):
        return gaussian_mixture(
            n_samples, seed, dim=1 if normalized.endswith("1d") else 3, **kwargs
        )
    if normalized.startswith("navigation_") and normalized[-2:] in ("1d", "2d", "3d"):
        return WallNavigation(dim=int(normalized[-2]), **kwargs).sample_dataset(n_samples, seed)
    raise ValueError(f"unknown environment {name!r}; choose one of {', '.join(ENVIRONMENT_NAMES)}")


__all__ = [
    "SampleDataset",
    "WallNavigation",
    "collision_loss",
    "endpoint_loss",
    "walls_from_conditions",
    "make_environment",
    "ENVIRONMENT_NAMES",
    "gaussian_mixture",
    "n_moons",
    "rotating_rays",
    "rings",
    "spirals",
    "helix",
]
