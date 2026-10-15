"""Cached, paired condition-routing views for rotating-ray trajectories.

Every panel uses the same noise index and exact stored solver nodes. These
functions never query a model or resample. Time is red-to-violet; target curves
use a separate charcoal/teal visual language, not a categorical rainbow.
"""

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .visualization.fields import as_numpy

_PROGRESS_CMAP = colors.LinearSegmentedColormap.from_list(
    "condition_routing_progress",
    ["#ef2d2d", "#ff9d18", "#f5dc35", "#37b866", "#257de4", "#7645b5"],
)
_TARGET_COLOR = "#087e8b"
_IDEAL_COLOR = "#202a37"


def _integer(value, name, low, high):
    if not isinstance(value, Integral) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(value)


def _prepare(histories_by_label, real, real_labels, n_rays, radius):
    if not isinstance(n_rays, Integral) or isinstance(n_rays, bool) or n_rays < 1:
        raise ValueError("n_rays must be a positive integer")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not isinstance(histories_by_label, Mapping) or not histories_by_label:
        raise ValueError("histories_by_label must be a nonempty mapping")
    histories, initial, shape = {}, None, None
    for label, value in histories_by_label.items():
        key = None if label is None else _integer(label, "condition label", 0, n_rays - 1)
        points = np.asarray(as_numpy(value), dtype=float)
        if (
            points.ndim != 3
            or points.shape[-1] != 2
            or points.shape[0] < 2
            or points.shape[1] < 1
            or not np.isfinite(points).all()
        ):
            raise ValueError("Each history must be finite [T+1,B,2], T>=1 and B>=1")
        if shape is not None and points.shape != shape:
            raise ValueError("All conditions must have the same T and batch shape")
        if initial is not None and not np.array_equal(points[0], initial):
            raise ValueError("All conditions must share exactly the same initial noise")
        shape, initial = points.shape, points[0]
        histories[key] = points
    real = np.asarray(as_numpy(real), dtype=float)
    real_labels = np.asarray(as_numpy(real_labels))
    if real.ndim != 2 or real.shape[1] != 2 or len(real) < 1 or not np.isfinite(real).all():
        raise ValueError("real must be nonempty finite [N,2] coordinates")
    if (
        real_labels.shape != (len(real),)
        or not np.isfinite(real_labels).all()
        or not np.equal(real_labels, np.floor(real_labels)).all()
        or np.any((real_labels < 0) | (real_labels >= n_rays))
    ):
        raise ValueError("real_labels must contain one valid integer ray ID per real point")
    # Copying metadata does not change stored trajectories or notebook RNG state.
    return histories, real, real_labels.astype(np.int64), int(n_rays), radius


def _choose_labels(histories, labels, n_rays):
    if labels is None:
        available = sorted(label for label in histories if label is not None)
        if n_rays == 16 and all(label in histories for label in (0, 4, 8, 12)):
            chosen = [0, 4, 8, 12]
        elif available:
            indices = np.linspace(0, len(available) - 1, min(4, len(available)), dtype=int)
            chosen = [available[index] for index in indices]
        else:
            chosen = [None]
    else:
        chosen = list(labels)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("labels must be a nonempty sequence without duplicates")
    for label in chosen:
        if label is not None:
            _integer(label, "condition label", 0, n_rays - 1)
        if label not in histories:
            raise ValueError(f"Requested condition {label!r} has no cached history")
    return chosen


def _ideal_curves(n_rays, radius):
    parameter = np.linspace(0, 1, 256)
    radial = radius * np.sqrt(parameter)
    twist = parameter * 1.6 * np.pi / (0.9 * n_rays)
    return [
        np.column_stack(
            (
                radial * np.cos(np.pi / 2 + 2 * np.pi * label / n_rays + twist),
                radial * np.sin(np.pi / 2 + 2 * np.pi * label / n_rays + twist),
            )
        )
        for label in range(n_rays)
    ]


def _bounds(histories, real, curves, noise_index):
    points = np.concatenate(
        [real, *curves, *(value[:, noise_index] for value in histories.values())]
    )
    lower, upper = points.min(0), points.max(0)
    margin = np.maximum((upper - lower) * 0.08, 0.06)
    return lower - margin, upper + margin


def _final_proxy(point, n_rays, radius):
    # Shared coordinate-only diagnostic, imported lazily to keep widgets optional.
    from .conditional_metrics import nearest_ray_ids

    label = int(nearest_ray_ids(np.asarray(point)[None], n_classes=n_rays, radius=radius)[0])
    radial = float(np.linalg.norm(point))
    text = "undefined at origin" if label < 0 else str(label)
    if label >= 0 and radial <= 0.25 * radius:
        text += " (near origin)"
    elif radial > radius:
        text += " (outside R)"
    return text


def _build(histories, real, real_labels, chosen, noise_index, n_rays, radius):
    curves = _ideal_curves(n_rays, radius)
    lower, upper = _bounds(histories, real, curves, noise_index)
    columns = min(2, len(chosen)) if len(chosen) <= 4 else min(4, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.4 * columns, 4.7 * rows + 1.2), squeeze=False
    )
    figure.subplots_adjust(top=0.87, bottom=0.20, hspace=0.38, wspace=0.25)
    artists = []
    total = next(iter(histories.values())).shape[0] - 1
    for ax, label in zip(axes.flat, chosen):
        target = real_labels == label if label is not None else np.zeros(len(real), dtype=bool)
        ax.scatter(
            real[~target, 0],
            real[~target, 1],
            s=4,
            color="#aab6c4",
            alpha=0.18,
            rasterized=True,
            zorder=0,
        )
        for curve in curves:
            ax.plot(curve[:, 0], curve[:, 1], lw=0.6, color="#b5beca", alpha=0.42, zorder=1)
        if label is not None:
            ax.scatter(
                real[target, 0],
                real[target, 1],
                s=9,
                color=_TARGET_COLOR,
                alpha=0.45,
                rasterized=True,
                zorder=2,
            )
            ax.plot(
                curves[label][:, 0],
                curves[label][:, 1],
                color=_IDEAL_COLOR,
                lw=2.2,
                ls="--",
                zorder=3,
            )
        path = histories[label][:, noise_index]
        trail = LineCollection(
            [], cmap=_PROGRESS_CMAP, norm=colors.Normalize(0, 1), lw=2.6, zorder=4
        )
        ax.add_collection(trail, autolim=False)
        ax.scatter(*path[0], s=57, c=[_PROGRESS_CMAP(0.0)], edgecolors="#202a37", lw=0.7, zorder=7)
        ax.scatter(
            *path[-1],
            marker="*",
            s=155,
            c=[_PROGRESS_CMAP(1.0)],
            edgecolors="white",
            lw=0.7,
            zorder=9,
        )
        head = ax.scatter(
            *path[0], s=45, c=[_PROGRESS_CMAP(0.0)], edgecolors="white", lw=0.8, zorder=8
        )
        proxy = _final_proxy(path[-1], n_rays, radius)
        requested = "null / unconditional" if label is None else f"ray {label}"
        ax.set(
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            aspect="equal",
            xlabel="x",
            ylabel="y",
        )
        ax.grid(color="#e2e7ee", alpha=0.6)
        artists.append((ax, requested, proxy, path, trail, head))
    for ax in list(axes.flat)[len(chosen) :]:
        ax.set_visible(False)
    legend = [
        Line2D([], [], marker="o", ls="", color=_PROGRESS_CMAP(0.0), label="Same initial noise"),
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            color=_PROGRESS_CMAP(1.0),
            markersize=11,
            label="Cached final endpoint",
        ),
        Line2D([], [], color=_IDEAL_COLOR, ls="--", lw=2, label="Requested ideal ray"),
        Line2D(
            [], [], marker="o", ls="", color=_TARGET_COLOR, label="Real points of requested ray"
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        ncols=2,
        frameon=False,
        fontsize=9,
    )
    color_ax = figure.add_axes((0.23, 0.065, 0.54, 0.017))
    colorbar = figure.colorbar(
        ScalarMappable(norm=colors.Normalize(0, 1), cmap=_PROGRESS_CMAP),
        cax=color_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks(
        [0, 0.5, 1], labels=["Start · red", "Actual solver progress", "Finish · violet"]
    )
    colorbar.ax.tick_params(length=0, labelsize=8)
    figure.text(
        0.5,
        0.012,
        "Nearest proxy is geometric, not a true class label.\nUnstable near the origin; extended outside R.",
        ha="center",
        fontsize=8,
    )
    return figure, axes, artists, total


def _frame(figure, artists, index, total, noise_index):
    for ax, requested, proxy, path, trail, head in artists:
        segments = np.stack((path[:index], path[1 : index + 1]), axis=1)
        trail.set_segments(segments)
        trail.set_array((np.arange(index) + 0.5) / total)
        head.set_offsets(path[index][None])
        head.set_facecolor(_PROGRESS_CMAP(index / total))
        ax.set_title(
            f"Requested: {requested}\nFinal nearest proxy: {proxy} · actual node {index}/{total}",
            fontsize=10,
        )
    initial = artists[0][3][0]
    figure.suptitle(
        f"Shared noise [{noise_index}] = ({initial[0]:.3f}, {initial[1]:.3f})\n"
        "Cached trajectories · red start → violet finish",
        fontsize=12,
        y=0.98,
    )


def plot_condition_routing(
    histories_by_label,
    real,
    real_labels,
    *,
    n_rays=16,
    labels=None,
    noise_index=0,
    frame_index=None,
    radius=10 / 11,
):
    """Plot one shared noise particle routed under several requested ray labels.

    Histories map integer labels (optionally ``None`` for null) to [T+1,B,2].
    Shapes, finite coordinates and exact initial-noise equality are validated
    across ALL entries. Default panels are [0,4,8,12] for a full16-ray cache;
    otherwise up to4 evenly spaced available labels. ``labels=[c]`` gives a
    single panel, and ``frame_index`` selects an actual node (default final).
    Both cached endpoints remain visible. Limits include real/ideal geometry and
    this noise particle under every condition at every node, so panels/frames
    share bounds. Changing the selected noise index can change bounds. Axes are
    always a2-D NumPy array.
    """
    histories, real, real_labels, n_rays, radius = _prepare(
        histories_by_label, real, real_labels, n_rays, radius
    )
    chosen = _choose_labels(histories, labels, n_rays)
    shape = next(iter(histories.values())).shape
    noise_index = _integer(noise_index, "noise_index", 0, shape[1] - 1)
    index = (
        shape[0] - 1
        if frame_index is None
        else _integer(frame_index, "frame_index", 0, shape[0] - 1)
    )
    figure, axes, artists, total = _build(
        histories, real, real_labels, chosen, noise_index, n_rays, radius
    )
    _frame(figure, artists, index, total, noise_index)
    return figure, axes


def save_condition_routing_gif(
    histories_by_label,
    real,
    real_labels,
    path,
    *,
    n_rays=16,
    labels=None,
    noise_index=0,
    fps=10,
    radius=10 / 11,
):
    """Save every actual solver node, with fixed bounds and no interpolation."""
    histories, real, real_labels, n_rays, radius = _prepare(
        histories_by_label, real, real_labels, n_rays, radius
    )
    chosen = _choose_labels(histories, labels, n_rays)
    shape = next(iter(histories.values())).shape
    noise_index = _integer(noise_index, "noise_index", 0, shape[1] - 1)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    figure, _, artists, total = _build(
        histories, real, real_labels, chosen, noise_index, n_rays, radius
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def frame(index):
        _frame(figure, artists, index, total, noise_index)
        return []

    movie = animation.FuncAnimation(
        figure,
        frame,
        frames=range(total + 1),
        interval=1000 / fps,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    try:
        movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    finally:
        plt.close(figure)
    return path


def condition_routing_widget(
    histories_by_label,
    real,
    real_labels,
    *,
    n_rays=16,
    radius=10 / 11,
    noise_index=0,
):
    """Cached target/noise/frame controls; no sampling, model calls or RNG use.

    Available targets come from cached keys. A null option appears only when a
    ``None`` history exists (dropdown value -1, state condition None). Frame
    Slider and Play move by one exact solver node.
    The returned VBox exposes ``controls``, ``state``, ``output`` and ``refresh``.
    """
    histories, real, real_labels, n_rays, radius = _prepare(
        histories_by_label, real, real_labels, n_rays, radius
    )
    shape = next(iter(histories.values())).shape
    noise_index = _integer(noise_index, "noise_index", 0, shape[1] - 1)
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            'Routing widgets require: pip install "diffusion-lab[interactive]"'
        ) from error
    labels = sorted(label for label in histories if label is not None)
    options = [(f"Ray {label}", label) for label in labels]
    if None in histories:
        # A real None value means "no selection" in ipywidgets; use a UI key.
        options.append(("Null / unconditional", -1))
    controls = {
        "target": widgets.Dropdown(options=options, description="Target:"),
        "noise_index": widgets.IntSlider(
            value=noise_index,
            min=0,
            max=shape[1] - 1,
            step=1,
            description="Noise:",
            continuous_update=False,
        ),
        "play": widgets.Play(value=0, min=0, max=shape[0] - 1, step=1, interval=100),
        "frame": widgets.IntSlider(
            value=0, min=0, max=shape[0] - 1, step=1, description="Node:", continuous_update=False
        ),
    }
    output, status, state = widgets.Output(), widgets.HTML(), {}
    widget = widgets.VBox(
        [
            widgets.HBox([controls["target"], controls["noise_index"]]),
            widgets.HBox([controls["play"], controls["frame"]]),
            status,
            output,
        ]
    )
    widget.history_link = widgets.link((controls["play"], "value"), (controls["frame"], "value"))

    def refresh(*_):
        figure = None
        try:
            label = controls["target"].value
            label = None if label == -1 else label
            particle = controls["noise_index"].value
            index = controls["frame"].value
            figure, axes = plot_condition_routing(
                histories,
                real,
                real_labels,
                n_rays=n_rays,
                labels=[label],
                noise_index=particle,
                frame_index=index,
                radius=radius,
            )
            state.update(
                condition=label,
                noise_index=particle,
                frame_index=index,
                history=histories[label],
                histories_by_label=histories,
                figure=figure,
                axes=axes,
                error=None,
            )
            status.value = f"Cached target={label!s}; same noise[{particle}]; actual node {index}/{shape[0] - 1}. No resampling."
            with output:
                output.clear_output(wait=True)
                display(figure)
            return figure
        except Exception as error:
            state["error"] = error
            status.value = f"{type(error).__name__}: {error}"
            return None
        finally:
            if figure is not None:
                plt.close(figure)

    for name in ("target", "noise_index", "frame"):
        controls[name].observe(refresh, names="value")
    widget.controls, widget.state, widget.output, widget.refresh = controls, state, output, refresh
    refresh()
    return widget
