"""Export an environment atlas and analytic field diagnostics, without training.

Run: python examples/field_gallery.py --output outputs/gallery
Optional interactive explorer requires diffusion-lab[interactive].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from diffusion_lab.environments import WallNavigation, make_environment
from diffusion_lab.visualization import (
    Grid,
    draw_walls,
    interactive_field,
    plot_dataset,
    plot_density,
    plot_potential,
    plot_trajectories,
    plot_vector_field,
    recover_potential,
    save_figure,
)


def _export(figure, output, name):
    files = save_figure(figure, output / name)
    plt.close(figure)
    return [str(path) for path in files]


def environment_atlas(output, seed):
    specs = [
        ("gaussian_mixture_1d", "1D · Gaussian mixture", {}),
        ("n_moons", "2D · N moons", {"n_classes": 4}),
        ("rotating_rays", "2D · Notebook rotating rays", {}),
        ("rings", "2D · Concentric rings", {}),
        ("spirals", "2D · Spirals", {}),
        ("helix_3d", "3D · Double helix", {}),
    ]
    figure = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    figure.suptitle("Diffusion Lab · environment atlas", fontsize=21, color="#0f172a")
    for index, (name, title, kwargs) in enumerate(specs):
        dataset = make_environment(name, n_samples=900, seed=seed, **kwargs)
        points = dataset.samples.numpy()
        axis = figure.add_subplot(
            2, 3, index + 1, projection="3d" if points.shape[1] == 3 else None
        )
        if points.shape[1] == 1:
            plot_density(points, ax=axis, title=title)
            axis.fill_between(
                axis.lines[0].get_xdata(), axis.lines[0].get_ydata(), color="#2563eb", alpha=0.1
            )
        else:
            plot_dataset(points, dataset.labels, ax=axis, size=8, alpha=0.65, title=title)
    return _export(figure, output, "environments")


def diagnostic_fields(output):
    grid = Grid.from_bounds([(-2, 2), (-1.5, 1.5)], (49, 37))
    x, y = grid.points.T
    fields = {"quadratic": np.c_[-x, -2 * y], "rotational": np.c_[-y, x]}
    artifacts, diagnostics = {}, {}
    for name, vectors in fields.items():
        result = recover_potential(grid, vectors)
        title = "Conservative F = (−x, −2y)" if name == "quadratic" else "Rotational F = (−y, x)"
        figure, _ = plot_vector_field(grid, vectors, stride=2, title=title, streamlines=True)
        artifacts[f"{name}_field"] = _export(figure, output, f"{name}_field")
        figure, _ = plot_potential(
            grid,
            result,
            title="Quadratic U recovered"
            if name == "quadratic"
            else "Rotational field: potential projection",
        )
        artifacts[f"{name}_potential"] = _export(figure, output, f"{name}_potential")
        diagnostics[name] = _diagnostics(result)
    return artifacts, diagnostics


def _diagnostics(result):
    return {
        "relative_residual": result.relative_residual,
        "residual_rms": result.residual_rms,
        "curl_rms": result.curl_rms,
        "relative_curl": result.relative_curl,
        "is_conservative": result.is_conservative,
        "component_count": result.component_count,
        "solver_converged": result.solver_converged,
        "message": result.message,
    }


def mixture_score(points, sigma, condition=None):
    """Exact score of a normalized equal-weight isotropic Gaussian mixture."""
    centers = np.array([[-0.9, -0.15], [0.55, 0.7], [0.55, -0.7]])
    if condition is not None:
        centers = centers[[condition]]
    variance = 0.13**2 + sigma**2
    offsets = points[:, None, :] - centers
    log_components = -0.5 * np.sum(offsets**2, axis=-1) / variance
    log_normalizer = logsumexp(log_components, axis=1, keepdims=True)
    responsibilities = np.exp(log_components - log_normalizer)
    score = -np.sum(responsibilities[..., None] * offsets, axis=1) / variance
    log_density = log_normalizer[:, 0] - np.log(len(centers)) - np.log(2 * np.pi * variance)
    return score, -log_density


def mixture_explorer(output):
    grid = Grid.from_bounds([(-1.6, 1.6), (-1.3, 1.3)], (49, 39))
    states = {}
    for condition in (None, 0, 1, 2):
        for sigma in (0.1, 0.3, 0.6):
            label = f"condition={'all' if condition is None else condition} · noise σ={sigma:g}"
            states[label] = grid.reshape(mixture_score(grid.points, sigma, condition)[0])
    field, analytic = mixture_score(grid.points, sigma=0.3)
    result = recover_potential(grid, field)
    figure, _ = plot_vector_field(
        grid, field, stride=2, title="Exact Gaussian-mixture score · σ=0.3"
    )
    artifacts = {"mixture_score_field": _export(figure, output, "mixture_score_field")}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].pcolormesh(*grid.axes, grid.reshape(analytic), cmap="viridis", shading="auto")
    figure.colorbar(image, ax=axes[0], label="−log p(x) (normalized analytic mixture)")
    axes[0].set(xlabel="x", ylabel="y", title="Analytic U = −log p(x)")
    axes[0].set_aspect("equal")
    plot_potential(grid, result, ax=axes[1], title="Recovered U · independent additive gauge")
    artifacts["mixture_potential"] = _export(figure, output, "mixture_potential")
    try:
        html = output / "mixture_score_explorer.html"
        interactive_field(
            grid, states, stride=2, path=html, title="Analytic score · condition and noise"
        )
        artifacts["mixture_score_explorer"] = [str(html)]
    except ImportError:
        print(
            "Plotly unavailable: install diffusion-lab[interactive] for the offline HTML explorer."
        )
    return artifacts, _diagnostics(result)


def navigation_gallery(output, seed):
    scene = WallNavigation(dim=2, path_length=32)
    dataset = scene.sample_dataset(16, seed=seed)
    figure, _ = plot_trajectories(
        dataset.samples, walls=scene.walls, title="A → B · reference routes around walls"
    )
    artifacts = {"wall_routes": _export(figure, output, "wall_routes")}
    grid = Grid.from_bounds(scene.bounds.T, (61, 49))
    goal = np.array([0.85, -0.75])
    mask = scene.collision_free(grid.points[:, None, :]).reshape(grid.shape)
    vectors = grid.reshape(scene.vector_field(grid.points, goal, repulsion=3.0))
    analytic = grid.reshape(scene.potential(grid.points, goal, repulsion=3.0))
    analytic = np.where(mask, analytic, np.nan)
    result = recover_potential(grid, vectors, mask=mask)
    figure, axis = plot_vector_field(
        grid,
        vectors,
        mask=mask,
        stride=2,
        title="Local attraction + wall repulsion · F = −∇U",
        streamlines=True,
    )
    draw_walls(axis, scene.walls)
    axis.scatter(*goal, marker="*", s=150, color="#e11d48", edgecolor="white", zorder=7)
    artifacts["wall_field"] = _export(figure, output, "wall_field")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].pcolormesh(*grid.axes, analytic, shading="auto", cmap="viridis")
    figure.colorbar(image, ax=axes[0], label="Analytic local potential U")
    axes[0].set(xlabel="x", ylabel="y", title="Analytic U · local minima are possible")
    axes[0].set_aspect("equal")
    plot_potential(grid, result, ax=axes[1], title="Masked grid reconstruction")
    for axis in axes:
        draw_walls(axis, scene.walls)
    artifacts["wall_potential"] = _export(figure, output, "wall_potential")
    return artifacts, _diagnostics(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gallery"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"environments": environment_atlas(output, args.seed)}
    field_artifacts, diagnostics = diagnostic_fields(output)
    artifacts.update(field_artifacts)
    mixture_artifacts, diagnostics["mixture_score"] = mixture_explorer(output)
    artifacts.update(mixture_artifacts)
    wall_artifacts, diagnostics["wall_navigation"] = navigation_gallery(output, args.seed)
    artifacts.update(wall_artifacts)
    summary = {
        "seed": args.seed,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
        "note": "No model training. Navigation fields can have local minima; A* supplies reference routes.",
    }
    (output / "gallery.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(f"Gallery exported to {output}")
    return summary


if __name__ == "__main__":
    main()
