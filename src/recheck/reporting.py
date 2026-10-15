"""Rebuild the rays pilot report and actual-state animations from saved evidence.

No model training, generation, metric selection, or example search happens here.
All tables derive from results.json. Animations use the first registered seed
and exact solver states, synchronized by elapsed fraction without interpolation.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .evaluate import paired_statistics
from .runtime import atomic_json, file_hash, tensor_hash

METHOD_ORDER = ("Original N32", "Original N16", "M0", "M1", "E0", "E1")
METHOD_COLORS = {
    "Original N32": "#263449",
    "Original N16": "#8b95a7",
    "M0": "#3176d1",
    "M1": "#26a5a5",
    "E0": "#e28b28",
    "E1": "#c54f77",
}
RAINBOW = colors.LinearSegmentedColormap.from_list(
    "start_red_finish_violet", ["#ef2d2d", "#ff9d18", "#f5dc35", "#37b866", "#257de4", "#7645b5"]
)
QUALITY_METRICS = ("mmd2_real", "sliced_wasserstein")
MNIST_METRICS = ("mmd2_mnist_features_real", "digit_frequency_tv_real")


def _metric_label(metric):
    return {
        "mmd2_real": "MMD² to real",
        "sliced_wasserstein": "Sliced W1 to real",
        "mmd2_mnist_features_real": "MMD² · MNIST features to real",
        "digit_frequency_tv_real": "Predicted digit frequency TV to real",
    }.get(metric, metric)


def _metric_names(rows):
    return (
        MNIST_METRICS
        if any("mmd2_mnist_features_real" in row.get("quality", {}) for row in rows)
        else QUALITY_METRICS
    )


def _style():
    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e9f0",
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfdff",
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure, report_dir, name):
    report_dir = Path(report_dir)
    paths = [report_dir / f"{name}.{extension}" for extension in ("png", "svg")]
    for path in paths:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [str(path) for path in paths]


def _histories(histories):
    if not histories:
        raise ValueError("At least one saved trajectory is required")
    result, initial = {}, None
    for name, value in histories.items():
        value = torch.as_tensor(value).detach().cpu()
        if (
            value.ndim != 3
            or value.shape[-1] != 2
            or min(value.shape[:2]) < 1
            or len(value) < 2
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Saved trajectories must be finite [T+1,N,2], with at least two nodes")
        if initial is None:
            initial = value[0]
        elif not torch.equal(initial, value[0]):
            raise ValueError(
                "Trajectory comparison requires exactly the same initial noise and sample ordering"
            )
        result[str(name)] = value
    return result


def _bounds(histories, real=None):
    points = [value.flatten(0, 1).numpy() for value in histories.values()]
    if real is not None:
        real = torch.as_tensor(real).detach().cpu().numpy()
        if real.ndim == 2 and real.shape[1] == 2 and np.isfinite(real).all():
            points.append(real)
    points = np.concatenate(points)
    lower, upper = points.min(0), points.max(0)
    margin = np.maximum((upper - lower) * 0.07, 0.08)
    return lower - margin, upper + margin


def actual_frame_indices(histories, *, max_frames=33):
    """Choose existing nodes at each shared elapsed fraction, never synthetic states."""
    histories = _histories(histories)
    if not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError("At least two animation frames are required")
    count = min(max_frames, max(len(value) for value in histories.values()))
    fractions = np.linspace(0, 1, count)
    result = {}
    for name, history in histories.items():
        # Holding the preceding real node avoids displaying a future solver state.
        indices = np.floor(fractions * (len(history) - 1) + 1e-10).astype(int)
        result[name] = indices.tolist()
    return fractions.tolist(), result


def _trajectory_figure(histories, *, particles, real, bounds, columns=3):
    count = len(histories)
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    single = count == 1
    size = (6.2, 6.7) if single else (4.6 * columns, 4.4 * rows + 1.0)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    figure.subplots_adjust(
        top=0.83 if single else 0.88, bottom=0.28 if single else 0.15, hspace=0.35, wspace=0.24
    )
    artists = []
    for ax, (name, history) in zip(axes.flat, histories.items()):
        path = history[:, :particles].numpy()
        if real is not None:
            points = torch.as_tensor(real).detach().cpu().numpy()[:1500]
            ax.scatter(
                points[:, 0], points[:, 1], s=3, c="#a7b3c2", alpha=0.12, rasterized=True, zorder=0
            )
        trail = LineCollection(
            [], cmap=RAINBOW, norm=colors.Normalize(0, 1), linewidths=1.8, alpha=0.86, zorder=3
        )
        ax.add_collection(trail)
        ax.scatter(
            path[0, :, 0],
            path[0, :, 1],
            s=27,
            c=[RAINBOW(0.0)],
            marker="o",
            edgecolors="#263449",
            linewidths=0.6,
            zorder=5,
        )
        ax.scatter(
            path[-1, :, 0],
            path[-1, :, 1],
            s=60,
            c=[RAINBOW(1.0)],
            marker="*",
            edgecolors="white",
            linewidths=0.45,
            zorder=6,
        )
        head = ax.scatter(
            path[0, :, 0],
            path[0, :, 1],
            s=23,
            c=[RAINBOW(0.0)],
            edgecolors="white",
            linewidths=0.5,
            zorder=7,
        )
        ax.set(
            xlim=(bounds[0][0], bounds[1][0]),
            ylim=(bounds[0][1], bounds[1][1]),
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        ax.set_aspect("equal", adjustable="box")
        artists.append((ax, name, path, trail, head))
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            color=RAINBOW(0.0),
            marker="o",
            linestyle="",
            markersize=6,
            label="Start · common Gaussian noise",
        ),
        Line2D(
            [],
            [],
            color=RAINBOW(1.0),
            marker="*",
            linestyle="",
            markersize=10,
            label="Finish · generated sample",
        ),
        Line2D(
            [],
            [],
            color="#a7b3c2",
            marker=".",
            linestyle="",
            markersize=6,
            label="Independent real reference",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09 if single else 0.05),
        ncols=1 if single else 3,
        fontsize=9,
    )
    colorbar_axes = figure.add_axes([0.23, 0.055 if single else 0.025, 0.54, 0.012])
    colorbar = figure.colorbar(
        ScalarMappable(norm=colors.Normalize(0, 1), cmap=RAINBOW),
        cax=colorbar_axes,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1], labels=["Start · red", "solver progress", "Finish · violet"])
    colorbar.ax.tick_params(labelsize=8, length=0)
    return figure, artists


def _set_trajectory_frame(artists, indices):
    for ax, name, path, trail, head in artists:
        index = int(indices[name])
        segments = np.stack((path[:index], path[1 : index + 1]), axis=2).reshape(-1, 2, 2)
        interval_progress = (np.arange(index) + 0.5) / (len(path) - 1)
        # The first dimension is original time, the second is particle index.
        trail.set_segments(segments)
        trail.set_array(np.repeat(interval_progress, path.shape[1]))
        head.set_offsets(path[index])
        head.set_facecolor(RAINBOW(index / (len(path) - 1)))
        ax.set_title(f"{name}\nactual state {index}/{len(path) - 1}")


def save_comparison_gif(
    histories,
    path,
    *,
    max_frames=33,
    fps=10,
    particles=16,
    real=None,
    bounds=None,
    title="Rotating rays · identical noise · actual Euler states",
    dpi=90,
):
    """Render synchronized actual trajectories with red start and violet finish."""
    histories = _histories(histories)
    if fps <= 0 or particles < 1:
        raise ValueError("GIF fps and particle count must be positive")
    particles = min(particles, next(iter(histories.values())).shape[1])
    fractions, indices = actual_frame_indices(histories, max_frames=max_frames)
    bounds = _bounds(histories, real) if bounds is None else bounds
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _style():
        figure, artists = _trajectory_figure(
            histories, particles=particles, real=real, bounds=bounds
        )

        def frame(index):
            _set_trajectory_frame(
                artists, {name: values[index] for name, values in indices.items()}
            )
            figure.suptitle(
                f"{title}\nshared solver progress {fractions[index]:.0%}", fontsize=12, y=0.98
            )
            return []

        movie = animation.FuncAnimation(
            figure,
            frame,
            frames=len(fractions),
            interval=1000 / fps,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        try:
            movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        finally:
            plt.close(figure)
    return {
        "file": str(path),
        "frames": len(fractions),
        "fractions": fractions,
        "actual_node_indices": indices,
        "particles": particles,
        "fps": fps,
        "initial_noise_sha256": tensor_hash(next(iter(histories.values()))[0]),
        "common_bounds": [np.asarray(bound).tolist() for bound in bounds],
        "interpolation": False,
        "colors": "red start → orange → yellow → green → blue → violet finish",
    }


def _ordered_histories(histories):
    ordered = {}
    for method in METHOD_ORDER:
        matches = [name for name in histories if name == method or name.startswith(method + " ")]
        for name in matches:
            ordered[name] = histories[name]
    ordered.update({name: value for name, value in histories.items() if name not in ordered})
    return ordered


def _write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flat_row(row):
    result = {key: row.get(key) for key in ("run_id", "method", "student_seed", "nfe")}
    for category in ("quality", "fidelity", "cost"):
        for key, value in row.get(category, {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"{category}.{key}"] = value
    return result


def compute_contrasts(rows):
    """Only pair matched training seeds; frozen baselines are never replicated."""
    indexed = {}
    for row in rows:
        method, seed = row["method"], row.get("student_seed")
        if seed is not None:
            key = method, int(seed)
            if key in indexed:
                raise ValueError(f"Duplicate training result for {method}, seed {seed}")
            indexed[key] = row
    result = []
    for metric in _metric_names(rows):
        for contrast, first, second in (
            ("E0_minus_M0", "E0", "M0"),
            ("M1_minus_M0", "M1", "M0"),
            ("E1_minus_M1", "E1", "M1"),
        ):
            seeds = sorted(
                seed for method, seed in indexed if method == first and (second, seed) in indexed
            )
            seeds = [
                seed
                for seed in seeds
                if metric in indexed[first, seed].get("quality", {})
                and metric in indexed[second, seed].get("quality", {})
            ]
            if seeds:
                first_values = [indexed[first, seed]["quality"][metric] for seed in seeds]
                second_values = [indexed[second, seed]["quality"][metric] for seed in seeds]
                stats = paired_statistics(first_values, second_values)
                result.append(
                    {
                        "contrast": contrast,
                        "metric": metric,
                        "student_seeds": seeds,
                        "first": first,
                        "second": second,
                        **stats,
                    }
                )
        seeds = sorted(
            seed
            for method, seed in indexed
            if method == "M0" and all((other, seed) in indexed for other in ("M1", "E0", "E1"))
        )
        seeds = [
            seed
            for seed in seeds
            if all(
                metric in indexed[method, seed].get("quality", {})
                for method in ("M0", "M1", "E0", "E1")
            )
        ]
        if seeds:
            effects = [
                (indexed["E1", seed]["quality"][metric] - indexed["M1", seed]["quality"][metric])
                - (indexed["E0", seed]["quality"][metric] - indexed["M0", seed]["quality"][metric])
                for seed in seeds
            ]
            result.append(
                {
                    "contrast": "OT_x_tube_interaction",
                    "metric": metric,
                    "student_seeds": seeds,
                    **paired_statistics(effects),
                }
            )
    return result


def _quality_plots(rows, report_dir):
    outputs = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for x_name, x_label, filename in (
        ("nfe", "Actual model evaluations (NFE)", "quality_vs_nfe"),
        ("training_seconds", "Additional training wall time (seconds)", "quality_vs_training_time"),
    ):
        with _style():
            figure, axes = plt.subplots(1, 2, figsize=(11.3, 4.5))
            for ax, metric in zip(axes, _metric_names(rows)):
                for method in list(METHOD_ORDER) + [
                    method for method in grouped if method not in METHOD_ORDER
                ]:
                    points = grouped.get(method, [])
                    x, y = [], []
                    for row in points:
                        value = row.get("quality", {}).get(metric)
                        position = (
                            row.get("nfe") if x_name == "nfe" else row.get("cost", {}).get(x_name)
                        )
                        if x_name == "training_seconds" and row.get("student_seed") is None:
                            position = 0.0  # Shared teacher is already paid for every method.
                        if value is not None and position is not None:
                            x.append(position)
                            y.append(value)
                    if y:
                        color = METHOD_COLORS.get(method, "#555555")
                        ax.scatter(x, y, s=27, alpha=0.55, color=color)
                        ax.errorbar(
                            np.mean(x),
                            np.mean(y),
                            yerr=np.std(y, ddof=1) if len(y) > 1 else 0,
                            marker="D",
                            capsize=3,
                            color=color,
                            label=f"{method} · n={len(y)}",
                            linestyle="",
                        )
                ax.set(
                    xlabel=x_label,
                    ylabel=_metric_label(metric),
                    title="Independent quality · lower is better",
                )
                if "mmd2" in metric:
                    ax.axhline(0, color="#8d96a5", lw=0.7, zorder=0)
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
            figure.suptitle(
                "Frozen baseline shown once; student markers are independent training seeds",
                fontsize=11,
            )
            figure.tight_layout()
            outputs += _save_figure(figure, report_dir, filename)
    return outputs


def _paired_plot(contrasts, report_dir):
    with _style():
        figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
        metrics = (
            MNIST_METRICS
            if any(row["metric"] == "mmd2_mnist_features_real" for row in contrasts)
            else QUALITY_METRICS
        )
        for ax, metric in zip(axes, metrics):
            selected = [row for row in contrasts if row["metric"] == metric]
            for index, row in enumerate(selected):
                offsets = np.linspace(-0.09, 0.09, len(row["values"]))
                ax.scatter(index + offsets, row["values"], s=32, color="#3176d1", alpha=0.8)
                ci = row["paired_t_ci"]
                if ci is not None:
                    ax.errorbar(
                        index,
                        row["mean"],
                        yerr=[[row["mean"] - ci[0]], [ci[1] - row["mean"]]],
                        fmt="D",
                        color="#263449",
                        capsize=4,
                        markersize=5,
                    )
            ax.axhline(0, color="#888888", lw=0.9)
            ax.set_xticks(
                range(len(selected)),
                [
                    row["contrast"]
                    .replace("_minus_", " − ")
                    .replace("OT_x_tube_interaction", "OT × tube")
                    for row in selected
                ],
                rotation=18,
            )
            ax.set(
                ylabel="Absolute paired difference · " + _metric_label(metric),
                title="Dots: training seeds · bars: approximate 95% paired t CI",
            )
        figure.suptitle(
            "Three seeds are a pilot; a CI containing zero does not establish equivalence",
            fontsize=11,
        )
        figure.tight_layout()
        return _save_figure(figure, report_dir, "paired_seed_differences")


def _diagnostic_plots(diagnostics, report_dir):
    if isinstance(diagnostics, dict):
        diagnostics = diagnostics.get("runs", diagnostics.get("entries", [diagnostics]))
    records = []
    for entry in diagnostics:
        value = entry.get("diagnostics", entry)
        label = entry.get("method", entry.get("run_id", "student"))
        if entry.get("student_seed") is not None:
            label += f" · {entry['student_seed']}"
        records.append((label, value))
    if not records:
        return []
    outputs = []
    sources = ("interpolant", "teacher_rollout", "student_rollout")
    with _style():
        figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
        for ax, source in zip(axes, sources):
            for label, entry in records:
                rows = [
                    row
                    for row in entry.get("local_error", [])
                    if row["source"] == source and row["rho_eval"] == 0
                ]
                if rows:
                    ax.plot(
                        [row["t"] for row in rows],
                        [row["local_mse"] for row in rows],
                        label=label,
                        linewidth=1.6,
                    )
            ax.set(
                xlabel="t · noise 1 → data 0",
                ylabel="Raw local endpoint MSE",
                title=source.replace("_", " "),
            )
            ax.invert_xaxis()
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        figure.suptitle(
            "Identical starting state for each teacher/student local comparison", fontsize=11
        )
        figure.tight_layout()
        outputs += _save_figure(figure, report_dir, "local_error_vs_t")
    with _style():
        figure, ax = plt.subplots(figsize=(8.4, 4.5))
        for label, entry in records:
            rows = entry.get("rollout_error", [])
            if rows:
                ax.plot(
                    [row["t"] for row in rows], [row["rollout_mse"] for row in rows], label=label
                )
        ax.set(
            xlabel="t · noise 1 → data 0",
            ylabel="Raw paired rollout MSE",
            title="Common initial noise · common grid nodes",
        )
        ax.invert_xaxis()
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        figure.tight_layout()
        outputs += _save_figure(figure, report_dir, "rollout_error_vs_t")
    with _style():
        figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
        for ax, key in zip(axes, ("local_mse", "relative_local_mse")):
            for label, entry in records:
                groups = defaultdict(list)
                for row in entry.get("local_error", []):
                    if row["source"] == "interpolant":
                        groups[row["rho_eval"]].append(row[key])
                if groups:
                    rhos = sorted(groups)
                    ax.plot(
                        rhos,
                        [np.mean(groups[rho]) for rho in rhos],
                        marker="o",
                        markersize=4,
                        label=label,
                    )
            ax.set(
                xlabel="Evaluation tube ρ · δ=ρH",
                ylabel="Raw local MSE"
                if key == "local_mse"
                else "Local MSE / teacher displacement energy",
                title="Teacher target retargeted for every perturbation",
            )
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        figure.tight_layout()
        outputs += _save_figure(figure, report_dir, "robustness_vs_rho")
    return outputs


def _sample_grid(histories, real, report_dir):
    count = len(histories) + (real is not None)
    columns, rows = min(4, count), math.ceil(count / min(4, count))
    all_points = [history[-1].numpy() for history in histories.values()]
    if real is not None:
        all_points.append(torch.as_tensor(real).cpu().numpy())
    points = np.concatenate(all_points)
    lower, upper = points.min(0), points.max(0)
    margin = np.maximum((upper - lower) * 0.08, 0.08)
    with _style():
        figure, axes = plt.subplots(
            rows, columns, figsize=(3.3 * columns, 3.35 * rows), squeeze=False
        )
        panels = []
        if real is not None:
            panels.append(("Independent real reference", torch.as_tensor(real).cpu().numpy()))
        panels += [(name, history[-1].numpy()) for name, history in histories.items()]
        for ax, (name, point) in zip(axes.flat, panels):
            ax.scatter(point[:, 0], point[:, 1], s=6, alpha=0.42, color="#3176d1", rasterized=True)
            ax.set(
                xlim=(lower[0] - margin[0], upper[0] + margin[0]),
                ylim=(lower[1] - margin[1], upper[1] + margin[1]),
                xlabel="$x_1$",
                ylabel="$x_2$",
                title=f"{name}\nN={len(point)} · fixed order",
            )
            ax.set_aspect("equal", adjustable="box")
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        figure.suptitle(
            "Fixed samples · first registered student seed, no example selection", fontsize=12
        )
        figure.tight_layout()
        return _save_figure(figure, report_dir, "fixed_sample_grids")


def _training_plots(run_root, report_dir, seed):
    """The first registered seed illustrates optimization; losses are not quality."""
    teacher_path = run_root / "teacher" / "history.json"
    student_histories = {}
    for method in ("M0", "M1", "E0", "E1"):
        path = run_root / "students" / f"seed{seed}_{method}" / "history.json"
        if path.is_file():
            student_histories[method] = json.loads(path.read_text())
    if not teacher_path.is_file() and not student_histories:
        return []
    outputs = []
    with _style():
        figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
        if teacher_path.is_file():
            history = json.loads(teacher_path.read_text())
            axes[0].plot(
                [row["update"] for row in history],
                [row["fm_loss"] for row in history],
                color="#263449",
                linewidth=0.8,
                alpha=0.7,
            )
        axes[0].set(
            title="Shared teacher · raw FM loss",
            xlabel="Optimizer update",
            ylabel="Coordinate-mean FM MSE",
        )
        for method, history in student_histories.items():
            for ax, candidate_keys in zip(
                axes[1:], (("pair_loss", "pair"), ("dist_loss", "distribution_loss", "dist"))
            ):
                key = next(
                    (
                        candidate
                        for candidate in candidate_keys
                        if history and candidate in history[0]
                    ),
                    None,
                )
                if key is not None:
                    axes_values = [
                        (row["update"], row[key]) for row in history if row.get(key) is not None
                    ]
                    if axes_values:
                        x, y = zip(*axes_values)
                        ax.plot(
                            x,
                            y,
                            color=METHOD_COLORS[method],
                            alpha=0.65,
                            linewidth=1.0,
                            label=method,
                        )
        axes[1].set(
            title=f"Student paired MSE · seed {seed}",
            xlabel="Optimizer update",
            ylabel="Raw local endpoint MSE",
        )
        axes[2].set(
            title=f"Student distribution term · seed {seed}",
            xlabel="Optimizer update",
            ylabel="Raw distribution loss",
        )
        axes[2].legend(fontsize=8)
        figure.suptitle(
            "Optimization diagnostics · first registered seed · loss is not independent quality",
            fontsize=11,
        )
        figure.tight_layout()
        outputs += _save_figure(figure, report_dir, "training_loss_curves")
    return outputs


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _report_text(rows, contrasts, metadata, config, animations):
    seeds = config.get("distillation", {}).get("seeds", [])
    count = len({row["student_seed"] for row in rows if row.get("student_seed") is not None})
    lines = [
        "# Контролируемый пилот на вращающихся лучах",
        "",
        "Это независимая реализация протокола [P] из REPLICATION_SPEC.md. Числа ниже получены на вращающихся лучах; это не результаты MNIST и не буквальное воспроизведение диплома.",
        "",
        "**Основной MNIST-контраст E0 − M0: not_tested.** Полная MNIST-серия не выполнена на CPU. Результаты этого пилота проверяют реализацию и показывают поведение на малом стенде; они не устанавливают перенос на MNIST.",
        "",
        f"Зафиксированы один общий teacher и {count} student seed: `{seeds}`. Выбор анимаций: первый зарегистрированный seed `{seeds[0] if seeds else 'unknown'}`, без выбора по качеству. Все методы и NFE используют общий исходный шум; real reference независима от генераций.",
        "",
        f"Protocol: `{metadata.get('protocol', metadata.get('protocol_id', config.get('protocol_id', 'unknown')))}`. Teacher SHA: `{metadata.get('teacher_sha', metadata.get('teacher_sha256', 'recorded in run manifest'))}`. Noise SHA: `{metadata.get('noise_sha', metadata.get('noise_sha256', metadata.get('initial_noise_sha256', 'recorded in animations_manifest.json')))}`.",
        "",
        "## Качество относительно реальных данных",
        "",
        f"MMD² — несмещённая двухвыборочная оценка с фиксированным train-only RBF; отрицательные значения допустимы и сохранены. SWD здесь — среднее projected W1 существующего проекта, {config.get('evaluation', {}).get('swd_projections', 1024)} фиксированных независимых направления; это другая величина, чем squared SW2 training loss.",
        "",
        "| Метод | Student seed | NFE | MMD² → real | SWD → real |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = row.get("quality", {})
        lines.append(
            f"| {row['method']} | {row.get('student_seed') if row.get('student_seed') is not None else 'frozen baseline'} | {row['nfe']} | {_number(quality.get('mmd2_real'))} | {_number(quality.get('sliced_wasserstein'))} |"
        )
    lines += [
        "",
        "![Quality versus NFE](quality_vs_nfe.png)",
        "",
        "## Парные различия между независимыми обучениями",
        "",
        "Отрицательное различие благоприятно для первого метода. Интервал — приближённый paired t CI по seed; три seed являются пилотом. Интервал, включающий ноль, не доказывает эквивалентность. Относительные проценты от MMD не вычисляются.",
        "",
        "| Контраст | Метрика | Парные значения по seed | Mean | Sample std | 95% CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for contrast in contrasts:
        ci = contrast["paired_t_ci"]
        interval = "—" if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]"
        lines.append(
            f"| {contrast['contrast']} | {contrast['metric']} | {', '.join(_number(value) for value in contrast['values'])} | {_number(contrast['mean'])} | {_number(contrast['sample_std'])} | {interval} |"
        )
    lines += [
        "",
        "![Paired seed differences](paired_seed_differences.png)",
        "",
        "## Fidelity относительно общего teacher",
        "",
        "Fidelity измеряет воспроизведение teacher при общем шуме; улучшение fidelity само по себе не означает улучшение качества относительно данных. Raw MSE считается без clamp.",
        "",
        "| Метод | Student seed | NFE | Paired endpoint MSE |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row.get('student_seed') if row.get('student_seed') is not None else 'frozen'} | {row['nfe']} | {_number(row.get('fidelity', {}).get('paired_endpoint_mse'))} |"
        )
    lines += [
        "",
        "## Стоимость",
        "",
        "Время обучения teacher является общей стоимостью. В таблице training wall time ученика — дополнительная стоимость; frozen baseline не считается отдельным повторным обучением. Timings относятся к указанному batch и этому оборудованию. NFE — число последовательных evaluations для одного примера; total model calls дополнительно учитывает разбиение всего набора на batches.",
        "",
        "| Метод | Seed | NFE / sample | Total model calls | Training seconds | Median generation seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cost = row.get("cost", {})
        lines.append(
            f"| {row['method']} | {row.get('student_seed') if row.get('student_seed') is not None else 'frozen'} | {row['nfe']} | {_number(cost.get('total_model_calls', row.get('total_model_calls')))} | {_number(cost.get('training_seconds'))} | {_number(cost.get('median_sampling_seconds', cost.get('median_seconds')))} |"
        )
    lines += [
        "",
        "![Quality versus training time](quality_vs_training_time.png)",
        "",
        "## Реальные траектории",
        "",
        "Начало красное, конец фиолетовый. Кружки — общий Gaussian noise, звёзды — финальные точки. Кадры выбирают сохранённые состояния Euler; между узлами нет интерполяции. Все панели имеют одинаковые оси. Фиксированный первый seed не является выбором лучшего запуска.",
        "",
        "![Synchronized trajectories](trajectories_comparison.gif)",
        "",
        "![Fixed samples](fixed_sample_grids.png)",
        "",
        "## Диагностика и ограничения",
        "",
        "Local error сравнивает teacher/student из одного и того же anchor. Рассматриваются interpolant, реальный teacher rollout и method-dependent student rollout. Для robustness teacher target пересчитывается после каждого возмущения; tube имеет δ=ρH на координату. Линейный интерполянт пары данных и шума не является индивидуальной ODE-траекторией маргинального поля.",
        "",
        "Один teacher ограничивает выводы этим teacher. Три student seed недостаточны для универсального ранжирования. Ученики оцениваются на зарегистрированной сетке; frozen original на той же короткой сетке нужен для сравнения практической полезности. Снижение training loss, local error или совпадение с teacher не заменяет независимую оценку качества.",
        "",
        "Основные H1–H4 для unconditional pixel MNIST остаются `not_tested`; сравнения лучей являются отдельным диагностическим пилотом. H3 (все пространства endpoint/displacement/joint) не проверяется матрицей MVP.",
        "",
        "## Данные и воспроизведение отчёта",
        "",
        "Таблицы автоматически построены из `evaluation/results.json`; парные различия сохранены в `paired_comparisons.csv`, отдельные строки — в `results_per_seed.csv`, происхождение кадров — в `animations_manifest.json`. Используются только сохранённые метрики и траектории; отчёт не обучает и не выбирает примеры.",
        "",
        f"Команда: `python -m recheck.cli report --runs {metadata.get('run_root', '<run_root>')} --output <report_dir>/REPORT.md`.",
        "",
    ]
    return "\n".join(lines)


def _load_real(run_root):
    path = run_root / "datasets.pt"
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("test", "validation", "real", "samples"):
        if isinstance(data, dict) and key in data:
            value = data[key]
            value = value.get("samples") if isinstance(value, dict) else value
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == 2:
                return value
    return None


def _mnist_preview(histories, report_dir, config):
    """Fixed image mosaics and actual-state GIF, using identical visual clamp."""
    histories = _ordered_histories(histories)
    initial = None
    for value in histories.values():
        value = torch.as_tensor(value)
        if (
            value.ndim != 5
            or value.shape[2:] != (1, 28, 28)
            or len(value) < 2
            or not torch.isfinite(value).all()
        ):
            raise ValueError("MNIST previews require finite histories [T+1,N,1,28,28]")
        if initial is not None and not torch.equal(initial, value[0]):
            raise ValueError("MNIST preview histories must share exactly the same initial noise")
        initial = value[0]
    count = len(histories)
    if not count:
        return [], []
    columns, rows = min(3, count), math.ceil(count / min(3, count))
    particles = min(config.get("evaluation", {}).get("gif_particles", 16), len(initial))
    tile_columns, tile_rows = (
        math.ceil(particles**0.5),
        math.ceil(particles / math.ceil(particles**0.5)),
    )
    frames = min(33, max(len(value) for value in histories.values()))
    fractions = np.linspace(0, 1, frames)
    indices = {
        name: np.floor(fractions * (len(value) - 1) + 1e-10).astype(int).tolist()
        for name, value in histories.items()
    }

    def mosaic(value):
        pixel = (
            ((torch.as_tensor(value[:particles]).detach().cpu().float() + 1) / 2)
            .clamp(0, 1)
            .numpy()[:, 0]
        )
        tile = np.ones((tile_rows * 28, tile_columns * 28)) * 0.5
        for index, image in enumerate(pixel):
            y, x = divmod(index, tile_columns)
            tile[y * 28 : (y + 1) * 28, x * 28 : (x + 1) * 28] = image
        return tile

    with _style():
        figure, axes = plt.subplots(
            rows, columns, figsize=(3.65 * columns, 3.9 * rows), squeeze=False
        )
        artists = []
        for ax, (name, history) in zip(axes.flat, histories.items()):
            artist = ax.imshow(
                mosaic(history[0]), cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
            ax.set_axis_off()
            artists.append((ax, name, artist))
        for ax in list(axes.flat)[count:]:
            ax.set_visible(False)
        figure.subplots_adjust(top=0.88, bottom=0.08, wspace=0.12, hspace=0.27)
        figure.text(
            0.5,
            0.025,
            "Fixed first samples · identical noise · visualization only: (raw+1)/2, clamp[0,1]",
            ha="center",
            fontsize=9,
        )

        def frame(index):
            for ax, name, artist in artists:
                node = indices[name][index]
                artist.set_data(mosaic(histories[name][node]))
                ax.set_title(f"{name}\nactual state {node}/{len(histories[name]) - 1}", fontsize=10)
            figure.suptitle(
                f"Unconditional pixel MNIST · common noise\nshared solver progress {fractions[index]:.0%}",
                fontsize=12,
                y=0.98,
            )
            return []

        path = report_dir / "mnist_denoising_comparison.gif"
        fps = config.get("evaluation", {}).get("gif_fps", 10)
        movie = animation.FuncAnimation(
            figure,
            frame,
            frames=frames,
            interval=1000 / fps,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        try:
            movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
            frame(frames - 1)
            files = _save_figure(figure, report_dir, "fixed_sample_grids")
        finally:
            plt.close(figure)
    return [
        {
            "file": str(path),
            "frames": frames,
            "fractions": fractions.tolist(),
            "actual_node_indices": indices,
            "particles": particles,
            "fps": fps,
            "initial_noise_sha256": tensor_hash(initial),
            "interpolation": False,
            "visualization_preprocessing": "(raw_fm+1)/2 then clamp[0,1], fixed for all methods and frames",
        }
    ], files


def _contrast_status(contrast):
    if contrast is None or contrast["n_student_seeds"] < 3 or contrast["paired_t_ci"] is None:
        return "inconclusive"
    lower, upper = contrast["paired_t_ci"]
    if upper < 0:
        return "supported_on_small_benchmark"
    if lower > 0:
        return "no_advantage_in_tested_setting"
    return "inconclusive"


def _mnist_claims(rows, contrasts, schedule, source, report_dir):
    selected = {
        row["contrast"]: row for row in contrasts if row["metric"] == "mmd2_mnist_features_real"
    }
    result = []
    for claim, comparison in (
        ("H1", "E0_minus_M0"),
        ("H2", "M1_minus_M0"),
        ("H3", "representations_extended_matrix"),
        ("H4", "student_vs_original_same_grid"),
    ):
        contrast = selected.get(comparison)
        baseline = None
        if claim == "H4":
            m0 = [row for row in rows if row["method"] == "M0"]
            baseline = next(
                (
                    row
                    for row in rows
                    if row["method"].startswith("Original") and m0 and row["nfe"] == m0[0]["nfe"]
                ),
                None,
            )
            if baseline is not None and m0:
                values = [
                    row["quality"]["mmd2_mnist_features_real"]
                    - baseline["quality"]["mmd2_mnist_features_real"]
                    for row in m0
                ]
                contrast = {
                    **paired_statistics(values),
                    "student_seeds": [row["student_seed"] for row in m0],
                }
        methods = (
            {"E0", "M0"}
            if claim == "H1"
            else {"M1", "M0"}
            if claim == "H2"
            else {"M0"}
            if claim == "H4"
            else set()
        )
        evidence_ids = [row["run_id"] for row in rows if row["method"] in methods]
        if baseline is not None:
            evidence_ids.append(baseline["run_id"])
        result.append(
            {
                "claim_id": claim,
                "status": _contrast_status(contrast) if contrast is not None else "not_tested",
                "comparison": comparison,
                "metric": "mmd2_mnist_features_real",
                "schedule": schedule,
                "teacher_count": 1,
                "student_seed_count": contrast["n_student_seeds"] if contrast else 0,
                "effect_mean": contrast["mean"] if contrast else None,
                "paired_t_ci": contrast["paired_t_ci"] if contrast else None,
                "run_ids": evidence_ids,
                "evidence_files": [str(source), str(report_dir / "paired_comparisons.csv")],
                "scope": "unconditional_pixel_MNIST",
                "limitations": [
                    "Conditional on one frozen teacher and shared real reference/noise; three training seeds are a pilot. Quality, fidelity and cost must be read separately. H3 requires the separate extended representation matrix."
                ],
            }
        )
    return result


def _mnist_report_text(rows, contrasts, metadata, config, claims):
    seeds = sorted({row["student_seed"] for row in rows if row.get("student_seed") is not None})
    lines = [
        "# Контролируемая дистилляция на unconditional pixel MNIST",
        "",
        "Ниже приведены фактически сохранённые результаты маленького MNIST-стенда [P]. Это независимая реализация математического протокола, не буквальное воспроизведение кода диплома и не проверка результатов Z-Image / TripoSG.",
        "",
        f"Общий teacher: один. Student seed в результатах: `{seeds}`. Устройство: `{config.get('device', 'recorded in manifest')}`. Protocol: `{metadata.get('protocol', metadata.get('protocol_id', config.get('protocol_id', 'unknown')))}`.",
        "",
        "Основной заранее выбранный контраст — E0 − M0 по MMD² в признаках независимого MNIST-CNN относительно real test на целевой сетке. Другие сравнения являются дополнительными; они не заменяют главный контраст задним числом.",
        "",
        "## Качество относительно реального test",
        "",
        "RBF и стандартизация признаков зафиксированы на train; estimator MMD² является U-statistic и может быть отрицательным. Для оценщика ко всем генерациям применяется один preprocessing: `(raw+1)/2`, clamp `[0,1]`. Доля выхода за диапазон измеряется до clamp. Confidence и частоты цифр не доказывают визуальное качество и разнообразие.",
        "",
        "| Метод | Student seed | NFE | MMD² features → real | Digit TV → real | Confidence | Out of range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        q = row.get("quality", {})
        lines.append(
            f"| {row['method']} | {row.get('student_seed') if row.get('student_seed') is not None else 'frozen'} | {row['nfe']} | {_number(q.get('mmd2_mnist_features_real'))} | {_number(q.get('digit_frequency_tv_real'))} | {_number(q.get('mean_confidence'))} | {_number(q.get('out_of_range_fraction'))} |"
        )
    lines += [
        "",
        "![Independent quality](quality_vs_nfe.png)",
        "",
        "## Парные эффекты по seed",
        "",
        "Разности и approximate 95% paired t CI считаются по независимым обучением, не по изображениям. При трёх seed вывод пилотный и условен относительно одного teacher. CI, включающий ноль, не доказывает эквивалентность. У MMD нет относительного процента улучшения.",
        "",
        "| Контраст | Метрика | Парные значения | Mean | Sample std | 95% CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in contrasts:
        ci = row["paired_t_ci"]
        interval = "—" if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]"
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {', '.join(_number(value) for value in row['values'])} | {_number(row['mean'])} | {_number(row['sample_std'])} | {interval} |"
        )
    lines += [
        "",
        "![Paired seed differences](paired_seed_differences.png)",
        "",
        "## Fidelity и стоимость",
        "",
        "Raw endpoint MSE использует FM-координаты без clamp. SSIM имеет явно записанный data_range. Совпадение с teacher не является независимой оценкой истинного качества. Frozen baseline на короткой сетке сравнивается с учениками на той же сетке; baseline не превращается в несколько независимых обучений. NFE — на пример; total calls дополнительно учитывает batches всего evaluation-набора.",
        "",
        "| Метод | Seed | NFE | Total calls | Paired MSE | SSIM | data_range | Training seconds | Median sampling seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f, c = row.get("fidelity", {}), row.get("cost", {})
        lines.append(
            f"| {row['method']} | {row.get('student_seed') if row.get('student_seed') is not None else 'frozen'} | {row['nfe']} | {_number(c.get('total_model_calls', row.get('total_model_calls')))} | {_number(f.get('paired_endpoint_mse'))} | {_number(f.get('pixel_ssim'))} | {_number(f.get('ssim_data_range'))} | {_number(c.get('training_seconds'))} | {_number(c.get('median_sampling_seconds', c.get('median_seconds')))} |"
        )
    lines += [
        "",
        "![Quality versus training cost](quality_vs_training_time.png)",
        "",
        "## Зафиксированные примеры и траектории",
        "",
        "Примеры взяты в исходном порядке для первого зарегистрированного student seed; лучшая картинка или seed не выбираются. Анимация содержит сохранённые состояния решателя без интерполяции. Clamp используется только для одинаковой визуализации; raw MSE от него не меняется.",
        "",
        "![Fixed samples](fixed_sample_grids.png)",
        "",
        "![Actual denoising states](mnist_denoising_comparison.gif)",
        "",
        "## Статусы гипотез",
        "",
        "Статусы ниже основаны на абсолютном парном эффекте главной метрики, а не на loss. Устойчивое улучшение на маленьком стенде не устанавливает перенос на большие модели. H3 остаётся вне основной матрицы MVP.",
        "",
    ]
    for claim in claims:
        lines.append(
            f"- {claim['claim_id']}: `{claim['status']}`; `{claim['comparison']}`; effect mean `{_number(claim['effect_mean'])}`."
        )
    lines += [
        "",
        "## Воспроизведение",
        "",
        "Таблицы автоматически построены из `evaluation/results.json`; отдельные строки находятся в `results_per_seed.csv`, контрасты в `paired_comparisons.csv`, provenance кадров в `animations_manifest.json`. Измерения датасетов и noise общие для сравниваемых методов; для MMD к real выборки независимы.",
        "",
        f"Команда: `python -m recheck.cli report --runs {metadata.get('run_root', '<run_root>')} --output <report_dir>/REPORT.md`.",
        "",
    ]
    return "\n".join(lines)


def build_report(run_root, report_dir, config):
    """Regenerate all report artifacts from the root runner's saved contracts."""
    run_root, report_dir = Path(run_root), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    source = run_root / "evaluation" / "results.json"
    payload = json.loads(source.read_text())
    rows, metadata = payload["rows"], payload.get("metadata", {})
    is_mnist = config.get("data", {}).get("name") == "mnist"
    if is_mnist and any("mmd2_mnist_features_real" not in row.get("quality", {}) for row in rows):
        raise ValueError(
            "MNIST report requires actual MNIST feature-quality results, not rays metrics"
        )
    metadata = {**metadata, "run_root": str(run_root)}
    if not rows:
        raise ValueError("Cannot build a research report without actual result rows")
    contrasts = compute_contrasts(rows)
    _write_csv(report_dir / "results_per_seed.csv", [_flat_row(row) for row in rows])
    _write_csv(
        report_dir / "paired_comparisons.csv",
        [
            {
                "contrast": row["contrast"],
                "metric": row["metric"],
                "student_seeds": json.dumps(row["student_seeds"]),
                "paired_values": json.dumps(row["values"]),
                "n_student_seeds": row["n_student_seeds"],
                "mean": row["mean"],
                "sample_std": row["sample_std"],
                "ci_low": row["paired_t_ci"][0] if row["paired_t_ci"] else None,
                "ci_high": row["paired_t_ci"][1] if row["paired_t_ci"] else None,
            }
            for row in contrasts
        ],
    )
    first_seed = config.get("distillation", {}).get("seeds", [101])[0]
    files = _quality_plots(rows, report_dir) + _paired_plot(contrasts, report_dir)
    files += _training_plots(run_root, report_dir, first_seed)
    diagnostics_path = run_root / "evaluation" / "diagnostics.json"
    if diagnostics_path.is_file():
        files += _diagnostic_plots(json.loads(diagnostics_path.read_text()), report_dir)
    trajectory_path = run_root / "evaluation" / "trajectories.pt"
    animations = []
    if trajectory_path.is_file() and is_mnist:
        animations, previews = _mnist_preview(
            torch.load(trajectory_path, map_location="cpu", weights_only=True), report_dir, config
        )
        files += previews
    elif trajectory_path.is_file():
        histories = _ordered_histories(
            _histories(torch.load(trajectory_path, map_location="cpu", weights_only=True))
        )
        real = _load_real(run_root)
        bounds = _bounds(histories, real)
        settings = config.get("evaluation", {})
        particles, fps = settings.get("gif_particles", 16), settings.get("gif_fps", 10)
        animations.append(
            save_comparison_gif(
                histories,
                report_dir / "trajectories_comparison.gif",
                particles=particles,
                fps=fps,
                real=real,
                bounds=bounds,
            )
        )
        for name, history in histories.items():
            slug = "".join(
                character.lower() if character.isalnum() else "_" for character in name
            ).strip("_")
            animations.append(
                save_comparison_gif(
                    {name: history},
                    report_dir / f"trajectory_{slug}.gif",
                    particles=particles,
                    fps=fps,
                    real=real,
                    bounds=bounds,
                    title=f"{name} · exact saved Euler states",
                )
            )
        with _style():
            figure, artists = _trajectory_figure(
                histories,
                particles=min(particles, next(iter(histories.values())).shape[1]),
                real=real,
                bounds=bounds,
            )
            _set_trajectory_frame(
                artists, {name: len(history) - 1 for name, history in histories.items()}
            )
            figure.suptitle(
                "Actual trajectories · first registered seed · identical initial noise",
                y=0.98,
                fontsize=12,
            )
            files += _save_figure(figure, report_dir, "trajectories_comparison")
        files += _sample_grid(histories, real, report_dir)
    atomic_json(
        report_dir / "animations_manifest.json",
        {
            "selection": "first preregistered student seed, never selected by quality",
            "student_seed": config.get("distillation", {}).get("seeds", [None])[0],
            "animations": animations,
        },
    )
    distillation = config.get("distillation", {})
    schedule = f"teacher{distillation.get('teacher_steps', 32)}_student{distillation.get('student_steps', 16)}_euler"
    claims = [
        {
            "claim_id": claim,
            "status": "not_tested",
            "comparison": comparison,
            "metric": "mmd2_mnist_features_real",
            "schedule": schedule,
            "teacher_count": 0,
            "student_seed_count": 0,
            "effect_mean": None,
            "run_ids": [],
            "evidence_files": [],
            "scope": "unconditional_pixel_MNIST",
            "limitations": [
                "CPU rotating-rays pilot is a separate diagnostic benchmark; full MNIST series has not been executed."
            ],
        }
        for claim, comparison in (
            ("H1", "E0_minus_M0"),
            ("H2", "M1_minus_M0"),
            ("H3", "representations_extended_matrix"),
            ("H4", "student_vs_original_same_grid"),
        )
    ]
    if is_mnist:
        claims = _mnist_claims(rows, contrasts, schedule, source, report_dir)
    for contrast in [] if is_mnist else contrasts:
        methods = (
            {contrast["first"], contrast["second"]}
            if "first" in contrast
            else {"M0", "M1", "E0", "E1"}
        )
        claims.append(
            {
                "claim_id": "rays_pilot_" + contrast["contrast"],
                "status": "inconclusive",
                "comparison": contrast["contrast"],
                "metric": contrast["metric"],
                "schedule": schedule,
                "teacher_count": 1,
                "student_seed_count": contrast["n_student_seeds"],
                "effect_mean": contrast["mean"],
                "paired_t_ci": contrast["paired_t_ci"],
                "run_ids": [
                    row["run_id"]
                    for row in rows
                    if row.get("student_seed") in contrast["student_seeds"]
                    and row["method"] in methods
                ],
                "evidence_files": [str(source), str(report_dir / "paired_comparisons.csv")],
                "scope": "rotating_rays_CPU_diagnostic_pilot",
                "limitations": [
                    "Exploratory diagnostic pilot conditional on one teacher, not the preregistered MNIST contrast; no universal ranking."
                ],
            }
        )
    atomic_json(report_dir / "claims_verdict.json", claims)
    report_text = (
        _mnist_report_text(rows, contrasts, metadata, config, claims)
        if is_mnist
        else _report_text(rows, contrasts, metadata, config, animations)
    )
    if diagnostics_path.is_file():
        report_text += "\n## Кривые локальной ошибки, rollout и robustness\n\nStudent rollout anchors зависят от метода. Для каждого local сравнения вход у teacher/student общий; при каждом rho target пересчитывается. Производные, если измерены, следует читать вместе с проверкой при ζ/2.\n\n![Local error](local_error_vs_t.png)\n\n![Rollout error](rollout_error_vs_t.png)\n\n![Robustness](robustness_vs_rho.png)\n"
    if (report_dir / "training_loss_curves.png").is_file():
        report_text += "\n## Наблюдаемая оптимизация\n\n![Training losses](training_loss_curves.png)\n\nLoss служит диагностикой оптимизации; вывод о качестве основан на независимых quality metrics.\n"
    (report_dir / "REPORT.md").write_text(report_text)
    summary = {
        "report": str(report_dir / "REPORT.md"),
        "source_results": str(source),
        "source_results_sha256": file_hash(source),
        "rows": len(rows),
        "student_seed_count": len(
            {row["student_seed"] for row in rows if row.get("student_seed") is not None}
        ),
        "contrasts": contrasts,
        "animations": animations,
        "figures": files,
        "mnist_status": "measured" if is_mnist else "not_tested",
        "teacher_count": 1,
    }
    atomic_json(report_dir / "report_manifest.json", summary)
    return summary
