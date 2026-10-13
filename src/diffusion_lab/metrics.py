"""Seeded distribution comparisons and exact geometric path evaluation."""

import numpy as np
from scipy.stats import wasserstein_distance


def sliced_wasserstein(reference, generated, n_projections=128, seed=42):
    reference, generated = np.asarray(reference), np.asarray(generated)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError("Expected reference[N,D] and generated[M,D]")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("Distribution metrics require finite samples")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, reference.shape[-1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(
        np.mean(
            [
                wasserstein_distance(reference @ direction, generated @ direction)
                for direction in directions
            ]
        )
    )


def evaluate_samples(dataset, generated, conditions=None, seed=42):
    reference = dataset.samples.detach().cpu().numpy()
    if hasattr(generated, "detach"):
        generated = generated.detach().cpu().numpy()
    generated = np.asarray(generated)
    if not np.isfinite(generated).all():
        return {"finite_fraction": float(np.isfinite(generated).mean())}
    if generated.ndim == 2:
        result = {
            "sliced_wasserstein": sliced_wasserstein(reference, generated, seed=seed),
            "mean_error": float(np.linalg.norm(reference.mean(0) - generated.mean(0))),
            "std_error": float(np.linalg.norm(reference.std(0) - generated.std(0))),
        }
        if dataset.condition_kind == "categorical" and conditions is not None:
            labels = np.asarray(
                conditions.detach().cpu() if hasattr(conditions, "detach") else conditions
            )
            for class_id in np.unique(labels):
                ref = reference[dataset.conditions.cpu().numpy() == class_id]
                if len(ref):
                    result[f"class_{class_id}_sliced_wasserstein"] = sliced_wasserstein(
                        ref, generated[labels == class_id], seed=seed
                    )
        return result
    environment = dataset.environment
    condition_values = np.asarray(
        conditions.detach().cpu() if hasattr(conditions, "detach") else conditions
    )
    dim = generated.shape[-1]
    return {
        "collision_free_fraction": float(environment.collision_free(generated).mean()),
        "endpoint_error": float(
            (
                np.linalg.norm(generated[:, 0] - condition_values[:, :dim], axis=-1)
                + np.linalg.norm(generated[:, -1] - condition_values[:, dim : 2 * dim], axis=-1)
            ).mean()
            / 2
        ),
        "mean_path_length": float(
            np.linalg.norm(np.diff(generated, axis=1), axis=-1).sum(1).mean()
        ),
        "mean_squared_acceleration": float(np.square(np.diff(generated, n=2, axis=1)).mean()),
    }
