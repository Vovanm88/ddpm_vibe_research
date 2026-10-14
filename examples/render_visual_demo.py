"""Render the rotating-rays training showcase from saved run artifacts.

Run from a checkout (or after installing diffusion-lab):
    python examples/render_visual_demo.py --run outputs/visual_demo_rays

No training happens here. Distribution comparisons are computed from saved
snapshots, while score fields use the exported Diffusers pipeline. All scientific
figures are exported to PNG and editable SVG; the explorer works offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import colors
from matplotlib.lines import Line2D

from diffusion_lab.environments import make_environment
from diffusion_lab.experiment import switch_scheduler
from diffusion_lab.metrics import sliced_wasserstein
from diffusion_lab.pipeline import ResearchDiffusionPipeline
from diffusion_lab.visualization import Grid, evaluate_field, recover_potential
from diffusion_lab.visualization.plots import PALETTE, estimate_density, save_figure

INK = "#14263d"
MUTED = "#5c6e80"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#f97316"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "axes.edgecolor": "#cbd5e1",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def axes_style(ax, limits=None):
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=0.13, color="#7890a5", linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if limits is not None:
        ax.set(xlim=limits, ylim=limits, aspect="equal")


def save(fig, output, name):
    paths = save_figure(fig, output / name, dpi=180)
    plt.close(fig)
    return [str(path) for path in paths]


def load_reference(run, config):
    path = run / "reference.npz"
    if path.exists():
        with np.load(path) as values:
            return values["samples"], values["conditions"]
    environment = dict(config["environment"])
    environment["seed"] = int(config.get("seed", 42)) + 1000
    environment["n_samples"] = 4096
    dataset = make_environment(**environment)
    return dataset.samples.numpy(), dataset.conditions.numpy()


def load_snapshots(run, reference, reference_conditions, seed):
    records = []
    for path in sorted((run / "snapshots").glob("epoch_*.npz")):
        with np.load(path) as data:
            samples, conditions = data["samples"], data["conditions"]
        epoch = int(path.stem.split("_")[-1])
        global_swd = sliced_wasserstein(reference, samples, seed=seed)
        per_class = {
            str(int(c)): sliced_wasserstein(
                reference[reference_conditions == c], samples[conditions == c], seed=seed
            )
            for c in np.unique(reference_conditions)
        }
        records.append(
            dict(
                epoch=epoch,
                samples=samples,
                conditions=conditions,
                sliced_wasserstein=global_swd,
                class_mean_sliced_wasserstein=float(np.mean(list(per_class.values()))),
                per_class=per_class,
            )
        )
    if len(records) < 2:
        raise ValueError("Save at least the before-training and final snapshots first")
    return records


def scatter(ax, samples, conditions, limits, title):
    for class_id in np.unique(conditions):
        points = samples[conditions == class_id]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=PALETTE[int(class_id) % len(PALETTE)],
            s=5,
            alpha=0.43,
            linewidths=0,
            rasterized=True,
        )
    axes_style(ax, limits)
    ax.set_title(title, loc="left", pad=12)


def comparison(output, reference, labels, records):
    initial, final = records[0], records[-1]
    combined = np.concatenate([reference, initial["samples"], final["samples"]])
    extent = max(1.05, np.ceil(np.max(np.abs(combined)) * 10) / 10)
    limits = (-extent, extent)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), layout="constrained")
    scatter(axes[0], reference, labels, limits, "Target distribution")
    initial_title = (
        "Before training · epoch 0"
        if initial["epoch"] == 0
        else f"Earlier snapshot · epoch {initial['epoch']}"
    )
    scatter(axes[1], initial["samples"], initial["conditions"], limits, initial_title)
    scatter(
        axes[2],
        final["samples"],
        final["conditions"],
        limits,
        f"After training · epoch {final['epoch']}",
    )
    for ax, row in zip(axes[1:], [initial, final]):
        ax.text(
            0.03,
            0.03,
            f"SWD = {row['sliced_wasserstein']:.4f}",
            transform=ax.transAxes,
            color=INK,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=5),
        )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=PALETTE[int(c)],
            markeredgecolor="none",
            label=f"Condition {int(c)}",
            markersize=7,
        )
        for c in np.unique(labels)
    ]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="outside lower center")
    fig.suptitle("Rotating rays · conditional diffusion", fontsize=17, fontweight="bold")
    save(fig, output, "comparison")


def learning(output, run, records):
    history = json.loads((run / "history.json").read_text())
    epochs = np.asarray([row["epoch"] for row in history])
    loss = np.asarray([row["loss"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), layout="constrained")
    ax = axes[0]
    ax.plot(epochs, loss, color=BLUE, alpha=0.28, lw=1, label="Epoch mean")
    width = min(7, len(loss))
    if width >= 2:
        average = np.convolve(loss, np.ones(width) / width, mode="valid")
        ax.plot(epochs[width - 1 :], average, color=BLUE, lw=2.3, label=f"{width}-epoch average")
    ax.set(title="Training objective", xlabel="Epoch", ylabel="v-prediction MSE")
    ax.legend(frameon=False)
    ax = axes[1]
    snapshot_epochs = [row["epoch"] for row in records]
    ax.plot(
        snapshot_epochs,
        [row["sliced_wasserstein"] for row in records],
        "o-",
        color=GREEN,
        lw=2.2,
        ms=5,
        label="Global SWD",
    )
    ax.plot(
        snapshot_epochs,
        [row["class_mean_sliced_wasserstein"] for row in records],
        "s--",
        color=ORANGE,
        lw=1.8,
        ms=4,
        label="Mean of condition SWDs",
    )
    ax.set(
        title="Sample quality · lower is better",
        xlabel="Epoch",
        ylabel="Sliced Wasserstein distance",
    )
    ax.legend(frameon=False)
    for ax in axes:
        ax.set_facecolor("#f8fafc")
        ax.grid(alpha=0.15)
        ax.set_xlim(left=0)
    fig.suptitle(
        "Loss and generation quality are separate measurements", fontsize=15, fontweight="bold"
    )
    save(fig, output, "learning")


def density(output, reference, records):
    grid = Grid.from_bounds([(-1.15, 1.15), (-1.15, 1.15)], 85)
    data = [reference, records[-1]["samples"]]
    densities = [estimate_density(samples, grid, bandwidth=0.12) for samples in data]
    maximum = max(float(density.max()) for density in densities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained")
    for ax, values, title in zip(axes, densities, ["Target density", "Generated density"]):
        image = ax.pcolormesh(
            *grid.axes, values, shading="auto", cmap="magma", vmin=0, vmax=maximum, rasterized=True
        )
        ax.contour(*grid.axes, values, levels=7, colors="white", alpha=0.22, linewidths=0.4)
        axes_style(ax, (-1.15, 1.15))
        ax.set_title(title, loc="left")
    fig.colorbar(image, ax=axes, label="Gaussian KDE density", shrink=0.8)
    fig.suptitle("KDE factor = 0.12 · shared grid and color scale", fontsize=16, fontweight="bold")
    save(fig, output, "density")


def field_arrays(pipeline, grid, classes, times, guidance):
    arrays = {}
    for condition in classes:
        for time in times:
            arrays[(int(condition), int(time))] = evaluate_field(
                lambda points, c=int(condition), t=int(time): pipeline.vector_field(
                    points, t, conditions=c, kind="score", guidance_scale=guidance
                ),
                grid,
            )
    return arrays


def quiver(ax, grid, values, norm, *, background=None, color=True):
    magnitude = np.linalg.norm(values, axis=-1)
    selected = (slice(None, None, 3), slice(None, None, 3))
    coords = np.stack(grid.mesh, axis=-1)[selected]
    vectors = values[selected]
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    directions = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)
    display = directions * 0.065
    if background is not None:
        ax.scatter(
            background[:, 0],
            background[:, 1],
            s=4,
            c="#f97316",
            alpha=0.36,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    kwargs = dict(
        angles="xy", scale_units="xy", scale=1, width=0.003, headwidth=3.4, headlength=4.2, zorder=4
    )
    if color:
        artist = ax.quiver(
            *coords.reshape(-1, 2).T,
            *display.reshape(-1, 2).T,
            magnitude[selected].ravel(),
            cmap="viridis",
            norm=norm,
            **kwargs,
        )
    else:
        artist = ax.quiver(
            *coords.reshape(-1, 2).T,
            *display.reshape(-1, 2).T,
            color="#334155",
            alpha=0.62,
            **kwargs,
        )
    axes_style(ax, (float(grid.axes[0][0]), float(grid.axes[0][-1])))
    return artist


def score_fields(output, pipeline, grid, arrays, times, reference, labels):
    selected = [arrays[(0, int(time))] for time in times]
    magnitude = np.concatenate([np.linalg.norm(values, axis=-1).ravel() for values in selected])
    positive = magnitude[magnitude > 0]
    norm = colors.LogNorm(vmin=max(1e-3, float(positive.min())), vmax=float(positive.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained")
    for ax, values, time in zip(axes, selected, times):
        artist = quiver(ax, grid, values, norm, background=reference[labels == 0])
        alpha = float(pipeline.scheduler.alphas_cumprod[int(time)])
        ax.set_title(f"t = {time} · noise std = {np.sqrt(1 - alpha):.3f}", loc="left")
    fig.colorbar(artist, ax=axes, label="Original |score| · logarithmic scale", shrink=0.74)
    fig.suptitle(
        "Score field · condition 0 · arrows show direction", fontsize=16, fontweight="bold"
    )
    save(fig, output, "score_fields")


def potential(output, grid, vectors, time, reference, labels):
    result = recover_potential(grid, vectors)
    curl = np.gradient(vectors[..., 1], grid.axes[0], axis=1) - np.gradient(
        vectors[..., 0], grid.axes[1], axis=0
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained")
    quiver(axes[0], grid, vectors, None, background=reference[labels == 0], color=False)
    axes[0].set_title(f"Learned score · t = {time}", loc="left")
    # Potential has an arbitrary additive gauge. Centering changes no gradient.
    values = result.potential - np.median(result.potential)
    image = axes[1].pcolormesh(*grid.axes, values, cmap="viridis", shading="auto", rasterized=True)
    axes[1].contour(*grid.axes, values, levels=10, colors="white", linewidths=0.5, alpha=0.45)
    axes[1].set_title("Least-squares potential U", loc="left")
    axes_style(axes[1], (-1.15, 1.15))
    fig.colorbar(image, ax=axes[1], shrink=0.72, label="U · arbitrary additive constant")
    max_curl = max(float(np.max(np.abs(curl))), 1e-12)
    image = axes[2].pcolormesh(
        *grid.axes,
        curl,
        cmap="RdBu_r",
        shading="auto",
        vmin=-max_curl,
        vmax=max_curl,
        rasterized=True,
    )
    axes_style(axes[2], (-1.15, 1.15))
    axes[2].set_title("Curl diagnoses rotation", loc="left")
    fig.colorbar(image, ax=axes[2], shrink=0.72, label="∂Fᵧ/∂x − ∂Fₓ/∂y")
    fig.suptitle(
        "Project F onto −∇U · a learned score need not be conservative",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        f"Projection residual = {result.relative_residual:.2%}   ·   "
        f"relative curl = {result.relative_curl:.3f}   ·   "
        "grid diagnostics, not a guaranteed exact energy",
        fontsize=10,
        color=MUTED,
    )
    save(fig, output, "potential_diagnostics")
    diagnostics = {
        "field": "score",
        "condition": 0,
        "timestep": int(time),
        "relative_residual": result.relative_residual,
        "curl_rms": result.curl_rms,
        "relative_curl": result.relative_curl,
        "is_conservative_on_grid": result.is_conservative,
        "message": result.message,
    }
    (output / "potential_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(
        output / "field_data.npz",
        score=vectors,
        potential=values,
        curl=curl,
        x=grid.axes[0],
        y=grid.axes[1],
    )


def explorer(output, pipeline, reference, labels, guidance):
    from interactive_visual_demo import build_explorer

    return build_explorer(
        pipeline,
        reference,
        labels,
        output / "explorer.html",
        guidance_scale=guidance,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/visual_demo_rays"))
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "showcase"
    output.mkdir(parents=True, exist_ok=True)
    style()
    config = yaml.safe_load((run / "config.yaml").read_text())
    torch.set_num_threads(int(config.get("num_threads", 4)))
    reference, labels = load_reference(run, config)
    projection_seed = int(config.get("seed", 42))
    records = load_snapshots(run, reference, labels, projection_seed)
    if reference.shape[1:] != (2,):
        raise ValueError("This showcase renderer is specifically for 2D point distributions")
    comparison(output, reference, labels, records)
    learning(output, run, records)
    density(output, reference, records)
    pipeline = ResearchDiffusionPipeline.from_pretrained(
        run / "pipeline", low_cpu_mem_usage=False
    ).to("cpu")
    switch_scheduler(pipeline, "ddim")
    pipeline.set_progress_bar_config(disable=True)
    grid = Grid.from_bounds([(-1.15, 1.15), (-1.15, 1.15)], 49)
    maximum = pipeline.scheduler.config.num_train_timesteps - 1
    times = [int(maximum * fraction) for fraction in (0.1, 0.35, 0.65)]
    classes = np.unique(labels)
    guidance = float(config.get("sampling", {}).get("guidance_scale", 1.0))
    arrays = field_arrays(pipeline, grid, classes, times, guidance)
    score_fields(output, pipeline, grid, arrays, times, reference, labels)
    potential(output, grid, arrays[(0, times[1])], times[1], reference, labels)
    explorer_metadata = explorer(output, pipeline, reference, labels, guidance)
    metrics = [
        {key: value for key, value in record.items() if key not in {"samples", "conditions"}}
        for record in records
    ]
    (output / "snapshot_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "showcase.json").write_text(
        json.dumps(
            {
                "projection_seed": projection_seed,
                "swd_projections": 128,
                "reference_samples": len(reference),
                "generated_samples_per_snapshot": len(records[-1]["samples"]),
                "initial_swd": records[0]["sliced_wasserstein"],
                "final_swd": records[-1]["sliced_wasserstein"],
                "epochs": records[-1]["epoch"],
                "guidance_scale": guidance,
                "field_timesteps": times,
                "conditions": [int(c) for c in classes],
                "figures": [
                    "comparison",
                    "learning",
                    "density",
                    "score_fields",
                    "potential_diagnostics",
                ],
                "offline_explorer": "explorer.html",
                "interactive_controls": explorer_metadata,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "snapshot_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
