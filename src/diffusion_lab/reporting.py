"""Export figures, fields, potential diagnostics and distribution/path metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .experiment import endpoint_constraints, write_json
from .metrics import evaluate_samples
from .visualization import (
    Grid,
    evaluate_field,
    interactive_field,
    plot_dataset,
    plot_density,
    plot_potential,
    plot_trajectories,
    plot_vector_field,
    recover_potential,
    save_denoising_gif,
    save_figure,
)


def export_report(
    pipeline,
    dataset,
    output,
    *,
    num_samples=512,
    steps=100,
    guidance_scale=1.5,
    seed=42,
    history=None,
    make_gif=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    write_json(
        output / "sampling_config.json",
        {
            "seed": seed,
            "num_samples": num_samples,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "scheduler_class": type(pipeline.scheduler).__name__,
            "scheduler_config": dict(pipeline.scheduler.config),
            "device": str(pipeline.model.device),
            "pinned_endpoints": len(pipeline.model.sample_shape) == 2,
        },
    )
    count = num_samples
    indices = torch.arange(count) % len(dataset)
    conditions = (
        dataset.conditions[indices].to(pipeline.model.device)
        if dataset.conditions is not None
        else None
    )
    generated = pipeline(
        batch_size=count,
        conditions=conditions,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        record_history=make_gif,
        **endpoint_constraints(pipeline.model, conditions),
    )
    samples = generated.samples.cpu().numpy()
    reference = dataset.samples.cpu().numpy()
    np.save(output / "samples.npy", samples)
    if conditions is not None:
        np.save(output / "conditions.npy", conditions.cpu().numpy())
    metrics = evaluate_samples(dataset, samples, conditions, seed=seed)
    if history:
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        ax.plot(
            [row["epoch"] for row in history],
            [row["loss"] for row in history],
            color="#2563eb",
            lw=2,
            label="Total objective",
        )
        ax.plot(
            [row["epoch"] for row in history],
            [row["regression"] for row in history],
            color="#f97316",
            lw=1.5,
            label="Regression",
        )
        ax.set(xlabel="Epoch", ylabel="Training loss", title="Optimization history")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
        save_figure(fig, output / "learning_curve")
        plt.close(fig)
    if reference.ndim == 3:
        walls = dataset.metadata.get("walls", [])
        for name, paths in (
            ("reference_routes", reference[:64]),
            ("generated_routes", samples[:64]),
        ):
            fig, _ = plot_trajectories(paths, walls=walls, title=name.replace("_", " ").title())
            save_figure(fig, output / name)
            plt.close(fig)
        if make_gif:
            save_denoising_gif(generated.history[:, :24], output / "denoising.gif", walls=walls)
        write_json(output / "metrics.json", metrics)
        return metrics
    labels = (
        conditions.cpu().numpy()
        if dataset.condition_kind == "categorical" and conditions is not None
        else None
    )
    for name, points, point_labels in (
        ("reference", reference, dataset.labels),
        ("generated", samples, labels),
    ):
        fig, _ = plot_dataset(points, point_labels, title=f"{name.title()} samples")
        save_figure(fig, output / name)
        plt.close(fig)
    bounds = [
        (float(reference[:, axis].min() - 0.25), float(reference[:, axis].max() + 0.25))
        for axis in range(reference.shape[-1])
    ]
    grid = Grid.from_bounds(
        bounds,
        resolution=100 if reference.shape[-1] == 1 else 36 if reference.shape[-1] == 2 else 16,
    )
    for name, points in (("reference_density", reference), ("generated_density", samples)):
        fig, _ = plot_density(points, grid, title=f"{name.replace('_', ' ').title()} (KDE)")
        save_figure(fig, output / name)
        plt.close(fig)
    time_count = pipeline.scheduler.config.num_train_timesteps
    flow = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    field_kind = "denoising" if flow else "score"
    field_condition = (
        int(dataset.conditions[0]) if dataset.condition_kind == "categorical" else None
    )
    fields = {}
    selected = None
    for fraction in (0.1, 0.35, 0.65):
        time = int((time_count - 1) * fraction)

        def field(points, time=time):
            return (
                pipeline.vector_field(
                    points,
                    time,
                    conditions=field_condition,
                    kind=field_kind,
                    guidance_scale=guidance_scale,
                )
                .cpu()
                .numpy()
            )

        vectors = evaluate_field(field, grid)
        fields[f"{field_kind} · t={time} · condition={field_condition}"] = vectors
        if fraction == 0.35:
            selected = vectors
            fig, _ = plot_vector_field(
                grid,
                vectors,
                samples=reference[:500],
                normalize=True,
                stride=2,
                streamlines=grid.ndim == 2,
                title=f"{field_kind.title()} · t={time} · condition={field_condition}",
            )
            save_figure(fig, output / "vector_field")
            plt.close(fig)
    potential = recover_potential(grid, selected)
    fig, _ = plot_potential(grid, potential, title="Potential projection · F = −∇U")
    save_figure(fig, output / "potential")
    plt.close(fig)
    np.savez_compressed(
        output / "field_data.npz",
        vectors=selected,
        potential=potential.potential,
        **{f"axis_{axis}": values for axis, values in enumerate(grid.axes)},
    )
    write_json(
        output / "potential_diagnostics.json",
        {
            "relative_residual": potential.relative_residual,
            "curl_rms": potential.curl_rms,
            "is_conservative": potential.is_conservative,
            "component_count": potential.component_count,
            "message": potential.message,
            "field_kind": field_kind,
            "condition": field_condition,
            "timestep": int((time_count - 1) * 0.35),
            "interpretation": "Grid least-squares projection, not a guaranteed exact energy or density",
        },
    )
    try:
        interactive_field(
            grid, fields, samples=reference[:500], stride=2, path=output / "fields.html"
        )
    except ImportError:
        pass
    if make_gif:
        save_denoising_gif(
            generated.history[:, :150],
            output / "denoising.gif",
            labels=labels[:150] if labels is not None else None,
        )
    write_json(output / "metrics.json", metrics)
    return metrics
