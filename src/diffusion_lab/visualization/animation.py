"""Rainbow trails in original coordinates, with an actual moving particle.

A path's colour is tied to its original waypoint index for every frame. A
Diffusers denoising history [steps,N,D] becomes paths via transpose(1,0,2);
those are noise-to-sample trajectories, distinct from physical A-to-B routes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .fields import as_numpy
from .plots import _axes, draw_walls


def _paths(paths):
    value = np.asarray(as_numpy(paths), dtype=float)
    if value.ndim == 2:
        value = value[None]
    if (
        value.ndim != 3
        or value.shape[0] < 1
        or value.shape[1] < 2
        or value.shape[-1] not in (1, 2, 3)
        or not np.isfinite(value).all()
    ):
        raise ValueError("paths must be finite [N,L,D] or [L,D], with N>=1, L>=2, D=1,2,3")
    return value


def _progress(progress):
    value = 1.0 if progress is None else float(progress)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("progress must be finite and in [0,1]")
    return value


def _coordinates(paths):
    if paths.shape[-1] > 1:
        return paths
    indices = np.broadcast_to(np.arange(paths.shape[1]), paths.shape[:2])
    return np.stack((indices, paths[..., 0]), axis=-1)


def _wall_bounds(walls, dimension):
    for wall in walls:
        if hasattr(wall, "lower") and hasattr(wall, "upper"):
            bounds = np.stack((wall.lower, wall.upper)).astype(float)
        else:
            bounds = np.asarray(wall, dtype=float)
            if bounds.shape == (4,) and dimension == 2:
                bounds = bounds.reshape(2, 2)
        if (
            bounds.shape != (2, dimension)
            or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])
        ):
            raise ValueError("wall bounds must be finite (lower,upper) matching path dimension")
        yield bounds


def _trail(coordinates, progress):
    length = coordinates.shape[1]
    position = progress * (length - 1)
    index = min(int(np.floor(position)), length - 1)
    fraction = position - index
    visible = coordinates[:, : index + 1]
    head = coordinates[:, index].copy()
    if fraction > 1e-12 and index < length - 1:
        head += fraction * (coordinates[:, index + 1] - head)
        visible = np.concatenate((visible, head[:, None]), axis=1)
    segments = np.stack((visible[:, :-1], visible[:, 1:]), axis=2)
    segments = segments.reshape(-1, 2, coordinates.shape[-1])
    # Midpoints of ORIGINAL intervals, even when the current interval is partial.
    # Neither colour nor normalisation changes when the trail becomes longer.
    time = (np.arange(visible.shape[1] - 1) + 0.5) / (length - 1)
    return segments, np.tile(time, len(coordinates)), head


def _build(paths, *, cmap, walls, ax, title, progress, start_label, end_label):
    dimension = paths.shape[-1]
    coordinates = _coordinates(paths)
    walls = tuple(walls)
    bounds = list(_wall_bounds(walls, dimension))
    figure, ax = _axes(dimension, ax, figsize=(7.0, 5.7))
    palette = plt.get_cmap(cmap)
    normalization = colors.Normalize(vmin=0, vmax=1)
    segments, times, head = _trail(coordinates, progress)
    collection_class = Line3DCollection if dimension == 3 else LineCollection
    trail = collection_class(
        segments, cmap=palette, norm=normalization, linewidths=2.1, alpha=0.86, zorder=3
    )
    trail.set_array(times)
    # Limits are set explicitly below. Autoscaling an empty 3D trail at t=0
    # otherwise asks Matplotlib to concatenate an empty list of segments.
    ax.add_collection(trail, autolim=False)
    size = 40 if len(paths) <= 12 else 25
    ax.scatter(
        *coordinates[:, 0].T,
        marker="o",
        s=size + 12,
        color=palette(0.0),
        edgecolors="#0f172a",
        linewidths=0.8,
        label=start_label,
        zorder=8,
    )
    ax.scatter(
        *coordinates[:, -1].T,
        marker="*",
        s=3 * size,
        color=palette(1.0),
        edgecolors="white",
        linewidths=0.7,
        label=end_label,
        zorder=9,
    )
    particle = ax.scatter(
        *head.T,
        marker="o",
        s=size + 8,
        color=palette(progress),
        edgecolors="white",
        linewidths=1.0,
        label="Current particle",
        zorder=7,
    )
    if walls:
        draw_walls(ax, walls, alpha=0.82, one_dimensional_path=dimension == 1)
    points = coordinates.reshape(-1, coordinates.shape[-1])
    if bounds:
        wall_points = np.concatenate(bounds)
        if dimension == 1:
            # In 1D, physical x is the vertical axis; time stays on the horizontal axis.
            wall_points = np.column_stack((np.zeros(len(wall_points)), wall_points[:, 0]))
        points = np.concatenate((points, wall_points))
    lower, upper = points.min(0), points.max(0)
    margin = np.maximum((upper - lower) * 0.09, 0.06)
    for coordinate, setter in enumerate(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim if dimension == 3 else None)
    ):
        if coordinate < len(lower):
            setter(lower[coordinate] - margin[coordinate], upper[coordinate] + margin[coordinate])
    if dimension == 1:
        ax.set(xlabel="Original waypoint / time index", ylabel="Particle position x")
    elif dimension == 3:
        ax.set_box_aspect(np.maximum(upper - lower, 0.1))
        ax.view_init(elev=24, azim=-58)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    legend.set_zorder(12)
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=palette),
        ax=ax,
        fraction=0.042,
        pad=0.035,
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    colorbar.ax.set_yticklabels(["0 · Start", "0.25", "0.50", "0.75", "1 · Finish"])
    colorbar.set_label("Original normalized time", fontsize=9)
    ax.set_title(f"{title}\nProgress {progress:.0%}", fontsize=12, pad=10)
    return figure, ax, coordinates, trail, particle, palette


def plot_rainbow_trajectories(
    paths,
    *,
    cmap="turbo_r",
    progress=None,
    walls=(),
    ax=None,
    title="Particle trajectories · Start → Finish",
    start_label="Start A",
    end_label="Finish B",
):
    """Plot [N,L,D] or a single [L,D] path in 1D / 2D / 3D.

    ``progress`` is a fraction in [0,1], default 1. Both original endpoints always
    remain visible. The head is interpolated within the current ORIGINAL segment;
    path coordinates are never scaled or mutated. In 1D the plot is time versus x.
    """
    figure, ax, *_ = _build(
        _paths(paths),
        cmap=cmap,
        walls=walls,
        ax=ax,
        title=title,
        progress=_progress(progress),
        start_label=start_label,
        end_label=end_label,
    )
    return figure, ax


def save_trajectory_gif(
    paths,
    path,
    *,
    fps=12,
    max_frames=80,
    cmap="turbo_r",
    walls=(),
    title="Particle trajectories · Start → Finish",
    dpi=90,
    start_label="Start A",
    end_label="Finish B",
):
    """Save a GIF of advancing heads and accumulated rainbow trails.

    Frames select actual original waypoints, including BOTH endpoints. Limits,
    colourbar, and interval colours stay fixed over the entire animation.
    """
    paths = _paths(paths)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, (int, np.integer))
        or max_frames < 2
    ):
        raise ValueError("max_frames must be an integer >=2")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.unique(
        np.rint(np.linspace(0, paths.shape[1] - 1, min(max_frames, paths.shape[1]))).astype(int)
    )
    figure, ax, coordinates, trail, particle, palette = _build(
        paths,
        cmap=cmap,
        walls=walls,
        ax=None,
        title=title,
        progress=0.0,
        start_label=start_label,
        end_label=end_label,
    )

    def frame(index):
        progress = index / (paths.shape[1] - 1)
        segments, times, head = _trail(coordinates, progress)
        trail.set_segments(segments)
        trail.set_array(times)
        particle.set_offsets(head[:, :2])
        if paths.shape[-1] == 3:
            particle.set_3d_properties(head[:, 2], zdir="z")
        particle.set_facecolor(palette(progress))
        ax.set_title(
            f"{title}\nProgress {progress:.0%} · waypoint {index}/{paths.shape[1] - 1}",
            fontsize=12,
            pad=10,
        )
        return trail, particle

    movie = animation.FuncAnimation(
        figure,
        frame,
        frames=indices,
        blit=False,
        interval=1000 / fps,
        repeat=False,
        cache_frame_data=False,
    )
    try:
        movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    return path
