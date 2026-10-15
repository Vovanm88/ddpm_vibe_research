"""Two distinct views: FM denoising of whole routes and motion along final routes.

Spatial paths have shape [B,L,2]; an FM history has shape [T+1,B,L,2].
No function samples a model. Whole-route FM frames always use actual stored
nodes. Physical playback follows the final piecewise-linear spatial path;
optional substeps interpolate only its spatial segments, never FM states.
"""

from collections.abc import Mapping
from fractions import Fraction
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .visualization.fields import as_numpy
from .visualization.plots import draw_walls

_RAINBOW = colors.LinearSegmentedColormap.from_list(
    "navigation_spatial_progress",
    ["#ef2d2d", "#ff9d18", "#f5dc35", "#37b866", "#257de4", "#7645b5"],
)
_MODE_COLORS = {0: "#2563eb", 1: "#d97706", -1: "#64748b"}
_DEFAULT_BOUNDS = ((-1.1, 1.1), (-1.1, 1.1))


def _paths(value):
    paths = np.asarray(as_numpy(value), dtype=float)
    if paths.ndim == 2:
        paths = paths[None]
    if (
        paths.ndim != 3
        or paths.shape[-1] != 2
        or paths.shape[0] < 1
        or paths.shape[1] < 2
        or not np.isfinite(paths).all()
    ):
        raise ValueError("Spatial paths must be finite [B,L,2], B>=1 and L>=2")
    return paths


def _scene(walls, start, goal, bounds):
    walls = np.asarray(as_numpy(walls), dtype=float)
    if walls.size == 0:
        walls = np.empty((0, 2, 2))
    elif walls.shape == (2, 2):
        walls = walls[None]
    if (
        walls.ndim != 3
        or walls.shape[1:] != (2, 2)
        or not np.isfinite(walls).all()
        or np.any(walls[:, 0] >= walls[:, 1])
    ):
        raise ValueError("walls must be ordered finite [W,2,2] low/high boxes")
    start, goal = np.asarray(as_numpy(start), dtype=float), np.asarray(as_numpy(goal), dtype=float)
    if start.shape != (2,) or goal.shape != (2,) or not np.isfinite([start, goal]).all():
        raise ValueError("One displayed scene requires finite start and goal vectors [2]")
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("bounds must contain finite (min,max) pairs for x and y")
    return walls, start, goal, bounds


def _mode_ids(paths, modes, walls, start, goal):
    if modes is None and len(walls) == 1:
        from .environments.multimodal_navigation import classify_navigation_modes

        condition = np.concatenate((start, goal, walls.reshape(-1)))
        conditions = np.repeat(condition[None], len(paths), axis=0)
        modes = classify_navigation_modes(paths, conditions)
    if modes is None:
        modes = np.full(len(paths), -1)
    modes = np.asarray(as_numpy(modes))
    if modes.shape != (len(paths),) or not np.isin(modes, (-1, 0, 1)).all():
        raise ValueError("modes must contain one ID per path:0 top,1 bottom,−1 unassigned/invalid")
    return modes.astype(int)


def _path_colors(count, color_by, modes=None):
    if color_by == "mode":
        return [_MODE_COLORS[int(mode)] for mode in modes]
    if color_by == "path":
        return [plt.get_cmap("tab20")(index % 20) for index in range(count)]
    if color_by != "progress":
        raise ValueError("color_by must be 'mode', 'path', or 'progress'")
    return None


def _outside(paths, bounds):
    outside = ((paths < bounds[:, 0]) | (paths > bounds[:, 1])).any(axis=-1)
    return int(outside.sum()), int(outside.any(axis=-1).sum())


def _base(ax, walls, start, goal, bounds):
    draw_walls(ax, walls, alpha=0.58)
    ax.scatter(*start, s=70, c=[_RAINBOW(0.0)], marker="o", edgecolors="white", lw=0.8, zorder=9)
    ax.scatter(*goal, s=150, c=[_RAINBOW(1.0)], marker="*", edgecolors="white", lw=0.8, zorder=9)
    ax.annotate("A", start, xytext=(7, 7), textcoords="offset points", weight="bold", zorder=10)
    ax.annotate("B", goal, xytext=(7, 7), textcoords="offset points", weight="bold", zorder=10)
    ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal", xlabel="x", ylabel="y")
    ax.grid(alpha=0.25, color="#cbd5e1")


def _legend(ax, color_by):
    handles = [
        Line2D([], [], marker="o", ls="", color=_RAINBOW(0.0), label="A · requested start"),
        Line2D([], [], marker="*", ls="", color=_RAINBOW(1.0), label="B · requested goal"),
    ]
    if color_by == "mode":
        handles += [
            Line2D([], [], color=color, label=name)
            for color, name in [
                (_MODE_COLORS[0], "Top route · evaluation only"),
                (_MODE_COLORS[1], "Bottom route · evaluation only"),
                (_MODE_COLORS[-1], "Invalid / unassigned"),
            ]
        ]
    elif color_by == "path":
        handles.append(
            Line2D([], [], color="#2563eb", label="Paired noise/path ID ·20-colour cycle")
        )
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        ncols=2,
    )


def plot_navigation_paths(
    paths,
    *,
    walls,
    start,
    goal,
    modes=None,
    color_by="mode",
    ax=None,
    title="Final decoded spatial routes A → B",
    bounds=_DEFAULT_BOUNDS,
):
    """Plot final [B,L,2] routes; blue/orange encode evaluated top/bottom modes.

    ``color_by='progress'`` instead colours spatial waypoint progress red A to
    violet B. This spatial coordinate is distinct from the FM denoising clock.
    Raw paths are never clipped; the fixed view reports out-of-view coordinates.
    A/B are requested conditions, not a claim of learned endpoint accuracy.
    """
    paths = _paths(paths)
    walls, start, goal, bounds = _scene(walls, start, goal, bounds)
    modes = _mode_ids(paths, modes, walls, start, goal) if color_by == "mode" else None
    palette = _path_colors(len(paths), color_by, modes)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.3, 6.6))
        figure.subplots_adjust(bottom=0.23)
    else:
        figure = ax.figure
    _base(ax, walls, start, goal, bounds)
    if color_by == "progress":
        segments = np.stack((paths[:, :-1], paths[:, 1:]), axis=2).reshape(-1, 2, 2)
        collection = LineCollection(
            segments, cmap=_RAINBOW, norm=colors.Normalize(0, 1), lw=1.9, alpha=0.8, zorder=5
        )
        collection.set_array(
            np.tile((np.arange(paths.shape[1] - 1) + 0.5) / (paths.shape[1] - 1), len(paths))
        )
        ax.add_collection(collection, autolim=False)
        bar = figure.colorbar(
            ScalarMappable(norm=colors.Normalize(0, 1), cmap=_RAINBOW), ax=ax, shrink=0.8
        )
        bar.set_label("Spatial waypoint progress A → B; not FM time")
    else:
        for path, color in zip(paths, palette):
            ax.plot(path[:, 0], path[:, 1], c=color, alpha=0.62, lw=1.4, zorder=5)
    points, routes = _outside(paths, bounds)
    ax.set_title(
        f"{title}\n{len(paths)} paths · {points} waypoints / {routes} paths outside view (raw unchanged)",
        fontsize=11,
    )
    _legend(ax, color_by)
    return figure, ax


def save_navigation_particle_gif(
    paths,
    path,
    *,
    walls,
    start,
    goal,
    modes=None,
    fps=10,
    substeps=1,
    color_by="progress",
    bounds=_DEFAULT_BOUNDS,
):
    """Move particles along FINAL spatial routes, never through FM states.

    ``substeps=1`` visits every actual saved waypoint. Larger integer values
    linearly interpolate each existing straight spatial segment with equal time
    per segment; this defines display motion, not an additional learned dynamics.
    No smoothed route or collision-free guarantee is introduced by playback.
    """
    paths = _paths(paths)
    walls, start, goal, bounds = _scene(walls, start, goal, bounds)
    if not isinstance(substeps, Integral) or isinstance(substeps, bool) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    modes = _mode_ids(paths, modes, walls, start, goal) if color_by == "mode" else None
    palette = _path_colors(len(paths), color_by, modes)
    figure, ax = plt.subplots(figsize=(7.4, 7.1))
    figure.subplots_adjust(bottom=0.23, top=0.86)
    _base(ax, walls, start, goal, bounds)
    for route in paths:
        ax.plot(route[:, 0], route[:, 1], color="#94a3b8", alpha=0.2, lw=0.8, zorder=2)
    trail = LineCollection(
        [], cmap=_RAINBOW, norm=colors.Normalize(0, 1), lw=2, alpha=0.8, zorder=6
    )
    ax.add_collection(trail, autolim=False)
    head = ax.scatter(paths[:, 0, 0], paths[:, 0, 1], s=30, edgecolors="white", lw=0.7, zorder=8)
    _legend(ax, color_by)
    if color_by == "progress":
        cax = figure.add_axes((0.23, 0.05, 0.54, 0.018))
        bar = figure.colorbar(
            ScalarMappable(norm=colors.Normalize(0, 1), cmap=_RAINBOW),
            cax=cax,
            orientation="horizontal",
        )
        bar.set_ticks([0, 1], labels=["Red · A", "Violet · B"])
        bar.ax.tick_params(labelsize=8, length=0)
    total = (paths.shape[1] - 1) * substeps
    outside_points, outside_paths = _outside(paths, bounds)

    def frame(number):
        position = number / substeps
        node = min(int(position), paths.shape[1] - 1)
        fraction = position - node
        visible = paths[:, : node + 1]
        current = paths[:, node].copy()
        if fraction and node < paths.shape[1] - 1:
            current += fraction * (paths[:, node + 1] - current)
            visible = np.concatenate((visible, current[:, None]), axis=1)
        segments = np.stack((visible[:, :-1], visible[:, 1:]), axis=2).reshape(-1, 2, 2)
        trail.set_segments(segments)
        if color_by == "progress":
            trail.set_array(
                np.tile((np.arange(visible.shape[1] - 1) + 0.5) / (paths.shape[1] - 1), len(paths))
            )
            head.set_facecolor(_RAINBOW(number / total))
        else:
            trail.set_color(
                np.repeat(
                    np.asarray([colors.to_rgba(c) for c in palette]), visible.shape[1] - 1, axis=0
                )
            )
            head.set_facecolor(palette)
        head.set_offsets(current)
        interpolation = (
            "actual saved waypoint"
            if substeps == 1
            else f"linear spatial segment substep ·{substeps} per segment"
        )
        ax.set_title(
            f"PHYSICAL PLAYBACK of final decoded polyline\nWaypoint position {position:g}/{paths.shape[1] - 1} · {interpolation}",
            fontsize=10,
        )
        figure.suptitle(
            "Particle motion along A → B · not FM denoising time\n"
            f"{outside_points} waypoints / {outside_paths} paths outside fixed view (raw unchanged)",
            fontsize=11,
        )
        return []

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=90)
    finally:
        plt.close(figure)
    return output


def _histories(values):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("histories_by_method must be a nonempty mapping")
    result, shape, initial = {}, None, None
    for name, value in values.items():
        array = np.asarray(as_numpy(value), dtype=float)
        if array.ndim != 4 or array.shape[0] < 2:
            raise ValueError(
                "FM histories must be [T+1,B,L,2]; a spatial path is not an FM history"
            )
        _paths(array[0])
        if not np.isfinite(array).all():
            raise ValueError("FM histories must be finite")
        if shape is not None and array.shape[1:] != shape:
            raise ValueError("FM methods must share batch/waypoint shape")
        if initial is not None and not np.array_equal(array[0], initial):
            raise ValueError(
                "FM methods must share exactly the same initial full paths and condition"
            )
        shape, initial = array.shape[1:], array[0]
        result[str(name)] = array
    return result


def _indices(histories, path_indices):
    batch = next(iter(histories.values())).shape[1]
    if path_indices is None:
        return np.arange(min(32, batch))
    result = np.asarray(path_indices)
    if (
        result.ndim != 1
        or len(result) < 1
        or not np.issubdtype(result.dtype, np.integer)
        or np.any((result < 0) | (result >= batch))
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError("path_indices must be distinct valid integer indices")
    return result


def _denoising_figure(histories, walls, start, goal, bounds, path_indices):
    columns = min(3, len(histories))
    rows = (len(histories) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.7 * columns, 4.3 * rows + 0.9), squeeze=False
    )
    figure.subplots_adjust(top=0.77 if rows == 1 else 0.87, bottom=0.20, hspace=0.44, wspace=0.25)
    palette = [plt.get_cmap("tab20")(int(index) % 20) for index in path_indices]
    artists = []
    for ax, (name, history) in zip(axes.flat, histories.items()):
        _base(ax, walls, start, goal, bounds)
        collection = LineCollection(
            history[0, path_indices], colors=palette, lw=1.2, alpha=0.67, zorder=5
        )
        ax.add_collection(collection, autolim=False)
        artists.append((ax, name, history, collection))
    for ax in list(axes.flat)[len(histories) :]:
        ax.set_visible(False)
    figure.text(
        0.5,
        0.015,
        "Each frame is an ENTIRE path state; particles are not moving.\n"
        "Line colour = paired path ID (20-colour cycle).\n"
        "Fixed scene view; raw coordinates are not clamped.",
        ha="center",
        fontsize=8,
    )
    return figure, axes, artists


def _denoising_frame(figure, artists, progress, indices, bounds):
    for ax, name, history, collection in artists:
        total = len(history) - 1
        index = min(int(np.floor(progress * total + 1e-10)), total)
        paths = history[index, indices]
        collection.set_segments(paths)
        points, count = _outside(paths, bounds)
        ax.set_title(
            f"{name} · actual FM node {index}/{total}\n{points} waypoints / {count} paths outside view (raw unchanged)",
            fontsize=9,
        )
    figure.suptitle(
        "FM DENOISING · whole-route state\n"
        f"Solver progress {progress:.0%} · cached actual nodes\n"
        "Shared noise/condition; no FM interpolation",
        fontsize=11,
        y=0.98,
    )


def plot_navigation_denoising(
    histories_by_method,
    *,
    walls,
    start,
    goal,
    path_indices=None,
    progress=1.0,
    bounds=_DEFAULT_BOUNDS,
):
    """Plot an actual FM whole-route state per method, with paired noise IDs.

    Different NFE histories use floor(progress*T), so a displayed state is never
    interpolated. The fixed scene view counts points outside it without clamping.
    """
    histories = _histories(histories_by_method)
    walls, start, goal, bounds = _scene(walls, start, goal, bounds)
    indices = _indices(histories, path_indices)
    if not np.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("progress must be finite and within [0,1]")
    figure, axes, artists = _denoising_figure(histories, walls, start, goal, bounds, indices)
    _denoising_frame(figure, artists, progress, indices, bounds)
    return figure, axes


def save_navigation_denoising_gif(
    histories_by_method,
    path,
    *,
    walls,
    start,
    goal,
    path_indices=None,
    fps=10,
    max_frames=33,
    bounds=_DEFAULT_BOUNDS,
):
    """Animate actual Euler states of the ENTIRE decoded path, not its particle."""
    histories = _histories(histories_by_method)
    walls, start, goal, bounds = _scene(walls, start, goal, bounds)
    indices = _indices(histories, path_indices)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(max_frames, Integral) or isinstance(max_frames, bool) or max_frames < 2:
        raise ValueError("max_frames must be an integer>=2")
    count = min(max_frames, max(len(value) for value in histories.values()))
    fractions = np.linspace(0, 1, count)
    figure, _, artists = _denoising_figure(histories, walls, start, goal, bounds, indices)

    def frame(index):
        _denoising_frame(figure, artists, float(fractions[index]), indices, bounds)
        return []

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        frame,
        frames=range(count),
        interval=1000 / fps,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    try:
        movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=90)
    finally:
        plt.close(figure)
    return output


def summarize_navigation_benchmark(
    histories_by_method,
    conditions,
    *,
    parent="Teacher N32",
    margin=0.035,
    endpoint_tol=0.08,
):
    """Assess cached routes with exact segment geometry and paired interior MSE.

    Quality/mode counts come from the shared geometry evaluator. Constructed A/B
    endpoints do not demonstrate learned endpoint accuracy. MSE to the parent is
    fidelity, not collision avoidance. Common FM nodes are matched as rational
    fractions with no temporal or spatial interpolation. No seed-level CI is
    inferred from a batch of paths.
    """
    from .environments.multimodal_navigation import evaluate_navigation

    histories = _histories(histories_by_method)
    if parent not in histories:
        raise ValueError("The named paired parent history is missing")
    reference = histories[parent]
    parent_nodes = {Fraction(index, len(reference) - 1): index for index in range(len(reference))}
    rows = []
    for name, history in histories.items():
        full = float(np.square(history[-1] - reference[-1]).mean())
        interior = (
            float(np.square(history[-1, :, 1:-1] - reference[-1, :, 1:-1]).mean())
            if history.shape[2] > 2
            else None
        )
        common = []
        for index in range(len(history)):
            fraction = Fraction(index, len(history) - 1)
            if fraction in parent_nodes:
                target = reference[parent_nodes[fraction]]
                mse = (
                    float(np.square(history[index, :, 1:-1] - target[:, 1:-1]).mean())
                    if history.shape[2] > 2
                    else None
                )
                common.append(
                    {
                        "solver_progress": float(fraction),
                        "student_node": index,
                        "parent_node": parent_nodes[fraction],
                        "interior_mse": mse,
                    }
                )
        rows.append(
            {
                "method": name,
                "nfe": len(history) - 1,
                "quality": evaluate_navigation(
                    history[-1], conditions, margin=margin, endpoint_tol=endpoint_tol
                ),
                "fidelity": {
                    "paired_parent_fullpath_mse": full,
                    "paired_parent_interior_mse": interior,
                    "common_actual_fm_nodes": common,
                },
            }
        )
    return {
        "scope": "cached_navigation_benchmark",
        "paired_initial_fullpath_assertion": True,
        "parent": parent,
        "rows": rows,
        "limitations": [
            "Exact A/B construction is not learned endpoint accuracy.",
            "Quality checks every spatial segment, bounds and route-mode occupancy; fidelity to the parent is separate.",
            "Route IDs are evaluation-only and are not condition inputs.",
            "A path batch does not replace independent teacher/student training seeds; this helper creates no training-seed confidence interval.",
            "Frozen original on each student's exact low-NFE grid is required for practical comparisons.",
        ],
    }
