"""Small runnable A-to-B trajectory experiment.

Run from the repository root: python examples/navigation_research.py --dim 2
The defaults exercise the research interfaces on CPU; three epochs do not train
a competent navigation policy. Increase samples/epochs for an actual experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Permit direct execution from a source checkout, as well as editable installs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion_lab import (  # noqa: E402
    DiffusionTrainer,
    PointDenoiser,
    ResearchDiffusionPipeline,
    TrainerConfig,
    make_scheduler,
)
from diffusion_lab.environments import SampleDataset, WallNavigation  # noqa: E402
from diffusion_lab.visualization import (  # noqa: E402
    Grid,
    draw_walls,
    plot_potential,
    plot_trajectories,
    plot_vector_field,
    recover_potential,
    save_figure,
)


@dataclass
class NavigationExperiment:
    scene: WallNavigation
    dataset: SampleDataset
    pipeline: ResearchDiffusionPipeline
    trainer: DiffusionTrainer
    output: object
    conditions: torch.Tensor
    metrics: dict[str, float]


def run_navigation(
    *,
    dim: int = 2,
    n_samples: int = 64,
    epochs: int = 3,
    seed: int = 7,
    device: str = "cpu",
    path_length: int = 32,
    batch_size: int = 16,
) -> NavigationExperiment:
    """Train and sample using continuous scene conditions and endpoint inpainting."""
    if dim not in (2, 3):
        raise ValueError("this tutorial supports 2D/3D navigation")
    torch.manual_seed(seed)
    scene = WallNavigation(dim=dim, path_length=path_length)
    dataset = scene.sample_dataset(n_samples=n_samples, seed=seed)
    model = PointDenoiser(
        sample_shape=dataset.sample_shape,
        backbone="transformer",
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        conditioner="mlp",
        condition_dim=dataset.condition_dim,
    )
    scheduler = make_scheduler("ddim", num_train_timesteps=100, beta_schedule="linear")
    pipeline = ResearchDiffusionPipeline(model=model, scheduler=scheduler)
    trainer = DiffusionTrainer(
        pipeline,
        TrainerConfig(
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            ema_decay=0.95,
            regularizers={"endpoint": 1.0, "collision": 0.5, "smoothness": 0.1, "bounds": 0.5},
        ),
        device=device,
    )
    trainer.fit(dataset)

    count = min(16, n_samples)
    conditions = dataset.conditions[:count]
    known_values = torch.zeros_like(dataset.samples[:count])
    known_values[:, 0] = conditions[:, :dim]
    known_values[:, -1] = conditions[:, dim : 2 * dim]
    known_mask = torch.zeros_like(known_values, dtype=torch.bool)
    known_mask[:, (0, -1)] = True
    # A seeded CPU generator works for CPU and is copied by Diffusers for MPS.
    generator_device = device if torch.device(device).type == "cuda" else "cpu"
    with trainer.ema_weights():
        output = pipeline(
            batch_size=count,
            conditions=conditions,
            num_inference_steps=20,
            generator=torch.Generator(device=generator_device).manual_seed(seed + 100),
            known_values=known_values,
            known_mask=known_mask,
            record_history=True,
        )
    paths = output.samples.detach().cpu()
    endpoint_error = torch.maximum(
        (paths[:, 0] - conditions[:, :dim]).abs().max(),
        (paths[:, -1] - conditions[:, dim : 2 * dim]).abs().max(),
    )
    metrics = {
        "reference_collision_free_fraction": float(scene.collision_free(dataset.samples).mean()),
        "generated_collision_free_fraction": float(scene.collision_free(paths).mean()),
        "maximum_endpoint_error": float(endpoint_error),
        "final_training_loss": float(trainer.history[-1]["loss"]),
    }
    print(json.dumps(metrics, indent=2))
    print("Generated feasibility includes every path segment, expanded walls, and domain bounds.")
    print(
        "Endpoint inpainting preserves A/B exactly; learned collision avoidance needs more training."
    )
    return NavigationExperiment(scene, dataset, pipeline, trainer, output, conditions, metrics)


def navigation_field(experiment: NavigationExperiment, resolution: int = 40):
    """Compare an analytic local field with its masked numerical potential fit."""
    scene = experiment.scene
    counts = resolution if scene.dim == 2 else (resolution // 2, resolution // 2, 9)
    grid = Grid.from_bounds(scene.bounds.T.tolist(), counts)
    goal = experiment.conditions[0, scene.dim : 2 * scene.dim].numpy()
    # Single-point paths give the public geometry API a point-accessibility mask.
    mask = scene.collision_free(grid.points[:, None, :]).reshape(grid.shape)
    vectors = grid.reshape(scene.vector_field(grid.points, goal))
    analytic = grid.reshape(scene.potential(grid.points, goal))
    analytic = np.where(mask, analytic, np.nan)
    recovered = recover_potential(grid, vectors, mask=mask)
    diagnostics = {
        "relative_residual": recovered.relative_residual,
        "curl_rms": recovered.curl_rms,
        "relative_curl": recovered.relative_curl,
        "component_count": recovered.component_count,
        "is_conservative_on_grid": recovered.is_conservative,
        "solver_converged": recovered.solver_converged,
    }
    print("Potential reconstruction diagnostics:", json.dumps(diagnostics, indent=2))
    print(recovered.message)
    print("This attractive/repulsive potential can have local minima; the reference routes use A*.")
    return grid, vectors, mask, analytic, recovered


def _analytic_potential_plot(grid: Grid, potential: np.ndarray, scene: WallNavigation):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if scene.dim == 3:
        index = len(grid.axes[2]) // 2
        values = potential[index]
        title = f"Analytical navigation potential — z={grid.axes[2][index]:.2f}"
        z = grid.axes[2][index]
        walls = scene.walls[(scene.walls[:, 0, 2] <= z) & (scene.walls[:, 1, 2] >= z), :, :2]
    else:
        values, title, walls = potential, "Analytical navigation potential", scene.walls
    image = axis.pcolormesh(*grid.axes[:2], values, shading="auto", cmap="viridis")
    figure.colorbar(image, ax=axis, label="U(x); local minima are possible")
    axis.contour(*grid.axes[:2], values, levels=12, colors="white", alpha=0.35, linewidths=0.5)
    draw_walls(axis, walls)
    axis.set(xlabel="x", ylabel="y", title=title)
    axis.set_aspect("equal")
    return figure, axis


def render_navigation(experiment: NavigationExperiment, output_dir="outputs/navigation-example"):
    """Export routes, vector arrows, analytic U and recovered U as PNG and SVG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = experiment.scene
    reference, _ = plot_trajectories(
        experiment.dataset.samples[:16],
        walls=scene.walls,
        title=f"Reference A* trajectories — {scene.dim}D",
    )
    generated, _ = plot_trajectories(
        experiment.output.samples,
        walls=scene.walls,
        title=f"Learned trajectories — collision-free {experiment.metrics['generated_collision_free_fraction']:.0%}",
    )
    grid, vectors, mask, analytic, recovered = navigation_field(experiment)
    arrows, axis = plot_vector_field(
        grid,
        vectors,
        mask=mask,
        stride=2,
        title="Analytical local navigation field F = −∇U",
        streamlines=scene.dim == 2,
    )
    draw_walls(axis, scene.walls)
    actual, _ = _analytic_potential_plot(grid, analytic, scene)
    fit, fit_axis = plot_potential(grid, recovered)
    # The scalar 3D visualization is a z slice, so overlay intersecting 2D boxes.
    if scene.dim == 2:
        fit_walls = scene.walls
    else:
        z = grid.axes[2][len(grid.axes[2]) // 2]
        intersects = (scene.walls[:, 0, 2] <= z) & (scene.walls[:, 1, 2] >= z)
        fit_walls = scene.walls[intersects, :, :2]
    draw_walls(fit_axis, fit_walls)
    artifacts = {}
    for name, figure in [
        ("reference_routes", reference),
        ("generated_routes", generated),
        ("navigation_field", arrows),
        ("analytical_potential", actual),
        ("recovered_potential", fit),
    ]:
        artifacts[name] = save_figure(figure, output_dir / name)
        plt.close(figure)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                **experiment.metrics,
                "potential_relative_residual": recovered.relative_residual,
                "potential_relative_curl": recovered.relative_curl,
                "potential_is_conservative_on_grid": recovered.is_conservative,
                "seed": experiment.trainer.config.seed,
                "epochs": experiment.trainer.config.epochs,
            },
            indent=2,
        )
        + "\n"
    )
    # Optional point-cloud view of all trajectory waypoints during denoising:
    # from diffusion_lab.visualization import save_denoising_gif
    # history = experiment.output.history
    # save_denoising_gif(history.reshape(len(history), -1, scene.dim),
    #                    output_dir / "denoising_waypoints.gif", walls=scene.walls)
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/navigation-example")
    args = parser.parse_args(argv)
    # Keep the small tutorial responsive on many-core machines.
    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    experiment = run_navigation(
        dim=args.dim, n_samples=args.samples, epochs=args.epochs, seed=args.seed, device=args.device
    )
    render_navigation(experiment, args.output)
    print(f"Saved research figures and metrics to {args.output.resolve()}")
    return experiment


if __name__ == "__main__":
    main()
