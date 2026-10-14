"""Matplotlib research figures with physical coordinates and honest magnitudes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde

from .fields import Grid, PotentialResult, as_numpy, evaluate_field

PALETTE = (
    "#2563eb",
    "#f97316",
    "#059669",
    "#a855f7",
    "#e11d48",
    "#0891b2",
    "#ca8a04",
    "#db2777",
    "#4f46e5",
    "#475569",
)


def _points(points):
    points = np.asarray(as_numpy(points), dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or not 1 <= points.shape[1] <= 3:
        raise ValueError("Expected points with shape (N,D), D=1,2,3")
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("Points must be nonempty and finite")
    return points


def _axes(ndim, ax=None, figsize=(8, 6)):
    if ax is None:
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d" if ndim == 3 else None)
    else:
        figure = ax.figure
        if ndim == 3 and not hasattr(ax, "zaxis"):
            raise ValueError("3D plots require a 3D matplotlib axis")
    ax.set_xlabel("x")
    ax.set_ylabel("y" if ndim > 1 else "")
    if ndim == 3:
        ax.set_zlabel("z")
    elif ndim == 2:
        ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=0.18)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    return figure, ax


def plot_dataset(points, labels=None, title="Dataset", ax=None, size=12, alpha=0.65):
    """Draw 1D/2D/3D samples; class values need not be consecutive integers."""
    points = _points(points)
    figure, ax = _axes(points.shape[1], ax)
    if labels is None:
        groups = [(None, np.ones(len(points), dtype=bool))]
    else:
        labels = as_numpy(labels)
        if labels.shape != (len(points),):
            raise ValueError("labels must contain one value per point")
        groups = [(label, labels == label) for label in np.unique(labels)]
    for i, (label, mask) in enumerate(groups):
        coords = [points[mask, k] for k in range(points.shape[1])]
        if points.shape[1] == 1:
            coords.append(np.zeros(np.count_nonzero(mask)))
        ax.scatter(
            *coords,
            color=PALETTE[i % len(PALETTE)],
            s=size,
            alpha=alpha,
            edgecolors="none",
            label=str(label) if label is not None else None,
        )
    if points.shape[1] == 1:
        ax.set_yticks([])
    if labels is not None:
        ax.legend(
            title="Condition",
            frameon=False,
            markerscale=1.7,
            ncols=2 if len(groups) > 4 else 1,
            fontsize=9,
        )
    if points.shape[1] == 3:
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
    ax.set_title(title)
    return figure, ax


def plot_vector_field(
    grid: Grid,
    vectors=None,
    *,
    field=None,
    samples=None,
    mask=None,
    normalize=True,
    stride=1,
    title="Vector field",
    ax=None,
    cmap="viridis",
    streamlines=False,
):
    """Draw directions with original magnitude in color and optional streamlines.

    normalize=True makes arrows a readable fixed coordinate length; color always
    uses unnormalized |F|. normalize=False preserves relative lengths with one
    global display scale, recorded in the title. Fields are never rescaled in data.
    """
    if (vectors is None) == (field is None):
        raise ValueError("Provide exactly one of vectors or field")
    if int(stride) != stride or stride < 1:
        raise ValueError("stride must be a positive integer")
    vectors = evaluate_field(field, grid) if field is not None else grid.reshape(vectors)
    if vectors.shape != grid.shape + (grid.ndim,):
        raise ValueError("Field shape must match grid and coordinate dimension")
    valid = np.ones(grid.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != grid.shape:
        raise ValueError("mask must match grid shape")
    if not np.all(np.isfinite(vectors[valid])):
        raise ValueError("Accessible field values must be finite")
    figure, ax = _axes(grid.ndim, ax)
    if samples is not None:
        plot_dataset(samples, ax=ax, size=5, alpha=0.16, title=title)
    selected = tuple(slice(None, None, int(stride)) for _ in range(grid.ndim))
    coords = np.stack(grid.mesh, axis=-1)[selected][valid[selected]]
    arrows = np.asarray(vectors)[selected][valid[selected]]
    if not len(arrows):
        raise ValueError("No accessible field arrows remain after subsampling")
    magnitude = np.linalg.norm(arrows, axis=-1)
    spacing = min(np.min(np.diff(axis)) for axis in grid.axes) * stride
    length = 0.75 * spacing
    display = np.zeros_like(arrows, dtype=float)
    if normalize:
        np.divide(arrows * length, magnitude[:, None], out=display, where=magnitude[:, None] > 0)
        subtitle = "direction arrows; color = |F|"
    else:
        scale = length / max(float(np.quantile(magnitude, 0.95)), np.finfo(float).eps)
        display = arrows * scale
        subtitle = f"arrow displacement = F × {scale:.3g}; color = |F|"
    norm = colors.Normalize(vmin=0, vmax=max(float(magnitude.max()), 1e-12))
    if grid.ndim == 3:
        ax.quiver(
            *coords.T,
            *display.T,
            color=plt.get_cmap(cmap)(norm(magnitude)),
            length=1,
            normalize=False,
            linewidth=0.8,
            arrow_length_ratio=0.3,
        )
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.set_box_aspect([axis[-1] - axis[0] for axis in grid.axes])
    else:
        x = coords[:, 0]
        y = coords[:, 1] if grid.ndim == 2 else np.zeros(len(coords))
        u = display[:, 0]
        v = display[:, 1] if grid.ndim == 2 else np.zeros(len(coords))
        scalar = ax.quiver(
            x,
            y,
            u,
            v,
            magnitude,
            cmap=cmap,
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0035,
            headwidth=3.2,
        )
        if grid.ndim == 1:
            ax.set_ylim(-length * 2, length * 2)
            ax.set_yticks([])
        if streamlines:
            if grid.ndim != 2:
                raise ValueError("streamlines require a 2D grid")
            if any(not np.allclose(np.diff(a), np.diff(a)[0]) for a in grid.axes):
                raise ValueError("matplotlib streamlines require uniformly spaced axes")
            u_grid = np.ma.array(vectors[..., 0], mask=~valid)
            v_grid = np.ma.array(vectors[..., 1], mask=~valid)
            ax.streamplot(
                *grid.axes,
                u_grid,
                v_grid,
                color="#334155",
                density=0.7,
                linewidth=0.55,
                arrowsize=0.65,
            )
    figure.colorbar(scalar, ax=ax, label="Original vector magnitude |F|", shrink=0.8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    for coordinate, axis in enumerate(grid.axes):
        getattr(ax, ("set_xlim", "set_ylim", "set_zlim")[coordinate])(axis[0], axis[-1])
    return figure, ax


def estimate_density(points, grid: Grid, bandwidth=None):
    """Evaluate a normalized Gaussian KDE, in units of coordinate-volume⁻¹.

    bandwidth follows scipy.stats.gaussian_kde (Scott/Silverman/scalar/callable).
    Singular or insufficient samples raise a useful error instead of silent jitter.
    """
    points = _points(points)
    if points.shape[1] != grid.ndim:
        raise ValueError("Sample dimension must match the density grid")
    try:
        kde = gaussian_kde(points.T, bw_method=bandwidth)
        return grid.reshape(kde(grid.points.T))
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(
            "KDE needs sufficient samples with full-rank covariance; "
            "use a lower-dimensional projection for degenerate data"
        ) from exc


def _scalar_plot(grid, values, title, color_label, ax=None, slice_index=None, cmap="magma"):
    values = grid.reshape(values)
    if values.shape != grid.shape:
        raise ValueError("Scalar field must match the grid")
    figure, ax = _axes(min(grid.ndim, 2), ax)
    if grid.ndim == 1:
        ax.plot(grid.axes[0], values, color=PALETTE[0], linewidth=2)
        ax.set_ylabel(color_label)
    else:
        if grid.ndim == 3:
            index = len(grid.axes[2]) // 2 if slice_index is None else slice_index
            if not 0 <= index < len(grid.axes[2]):
                raise ValueError("slice_index is outside the z axis")
            values = values[index]
            title += f" — z={grid.axes[2][index]:.3g} slice"
        image = ax.pcolormesh(grid.axes[0], grid.axes[1], values, shading="auto", cmap=cmap)
        figure.colorbar(image, ax=ax, label=color_label, shrink=0.85)
        finite = values[np.isfinite(values)]
        if len(finite) and np.ptp(finite) > 1e-12:
            ax.contour(
                grid.axes[0],
                grid.axes[1],
                values,
                levels=8,
                colors="white",
                alpha=0.32,
                linewidths=0.45,
            )
    ax.set_title(title)
    return figure, ax


def plot_density(
    points,
    grid=None,
    *,
    labels=None,
    bandwidth=None,
    ax=None,
    title="Gaussian KDE density",
    slice_index=None,
):
    """Plot genuine KDE in 1D/2D, or a slice of a full 3D density (not a marginal)."""
    points = _points(points)
    if labels is not None:
        labels = as_numpy(labels)
        if labels.shape != (len(points),):
            raise ValueError("labels must contain one value per point")
        if ax is not None:
            raise ValueError("Class-specific densities create multiple axes")
        unique = np.unique(labels)
        figure, axes = plt.subplots(
            1, len(unique), figsize=(5 * len(unique), 4), constrained_layout=True, squeeze=False
        )
        if grid is None:
            grid = _density_grid(points)
        for axis, label in zip(axes[0], unique):
            plot_density(
                points[labels == label],
                grid,
                bandwidth=bandwidth,
                ax=axis,
                title=f"{title}: condition {label}",
                slice_index=slice_index,
            )
        return figure, axes[0]
    grid = _density_grid(points) if grid is None else grid
    values = estimate_density(points, grid, bandwidth)
    return _scalar_plot(
        grid,
        values,
        title,
        f"Density p(x) [coordinate⁻{grid.ndim}]",
        ax=ax,
        slice_index=slice_index,
    )


def _density_grid(points):
    margin = np.maximum(np.ptp(points, axis=0) * 0.15, 0.1)
    bounds = list(zip(points.min(axis=0) - margin, points.max(axis=0) + margin))
    return Grid.from_bounds(bounds, 60 if points.shape[1] < 3 else (50, 50, 25))


def plot_potential(
    grid,
    result: PotentialResult,
    *,
    ax=None,
    slice_index=None,
    title="Recovered potential U (F = −∇U)",
    scale="linear",
    linthresh=None,
):
    """Draw U with linear or symmetric-log colors without changing its gauge.

    ``symlog`` retains zero and negative potentials: only the color mapping is
    logarithmic outside ``±linthresh``. The automatic threshold is 2% of the
    largest finite |U| (1 for an all-zero field). In 1D the curve uses a symlog
    ordinate instead. The underlying potential, signs and additive gauge stay
    unchanged. For 3D the color scale spans the full volume, including when a
    particular z slice is displayed.
    """
    if scale not in ("linear", "symlog"):
        raise ValueError("Potential scale must be 'linear' or 'symlog'")
    if linthresh is not None:
        linthresh = float(linthresh)
        if not np.isfinite(linthresh) or linthresh <= 0:
            raise ValueError("linthresh must be finite and positive")
    values = np.asarray(grid.reshape(result.potential))
    finite = values[np.isfinite(values)]
    if scale == "symlog" and linthresh is None:
        magnitude = float(np.max(np.abs(finite))) if len(finite) else 0.0
        linthresh = max(magnitude * 0.02, np.finfo(float).tiny) if magnitude else 1.0
    subtitle = (
        f"residual={result.relative_residual:.2%}; curl RMS={result.curl_rms:.3g}; "
        f"{'conservative on grid' if result.is_conservative else 'projection only'}"
    )
    if scale == "symlog":
        subtitle += f"; symlog colors, linthresh={linthresh:.3g}"
    collection_count = len(ax.collections) if ax is not None else 0
    figure, ax = _scalar_plot(
        grid,
        values,
        f"{title}\n{subtitle}",
        "Potential U (symlog)" if scale == "symlog" else "Potential U",
        ax=ax,
        slice_index=slice_index,
        cmap="viridis",
    )
    if scale == "symlog":
        if grid.ndim == 1:
            ax.set_yscale("symlog", linthresh=linthresh)
        else:
            # The scalar plot appends its QuadMesh before contour collections;
            # account for any artists already present on caller-supplied axes.
            image = ax.collections[collection_count]
            image.set_norm(
                colors.SymLogNorm(
                    linthresh=linthresh,
                    vmin=float(finite.min()) if len(finite) else -1.0,
                    vmax=float(finite.max()) if len(finite) else 1.0,
                )
            )
    return figure, ax


def draw_walls(ax, walls, alpha=0.9, *, one_dimensional_path=False):
    """Overlay axis-aligned wall bounds [W,2,D] or 2D (xmin,ymin,xmax,ymax)."""
    for wall in walls:
        if hasattr(wall, "lower") and hasattr(wall, "upper"):
            lower, upper = np.asarray(wall.lower), np.asarray(wall.upper)
        else:
            bounds = np.asarray(wall, dtype=float)
            if bounds.shape == (4,):
                lower, upper = bounds[:2], bounds[2:]
            elif bounds.ndim == 2 and bounds.shape[0] == 2:
                lower, upper = bounds[0], bounds[1]
            else:
                raise ValueError("Walls must be (xmin,ymin,xmax,ymax) or (lower,upper)")
        if not 1 <= len(lower) <= 3 or np.any(upper <= lower):
            raise ValueError("Wall bounds must have positive extent in 1–3 dimensions")
        if len(lower) == 1:
            span = ax.axhspan if one_dimensional_path else ax.axvspan
            span(lower[0], upper[0], color="#334155", alpha=alpha, zorder=4)
        elif len(lower) == 2:
            ax.add_patch(
                Rectangle(
                    lower,
                    *(upper - lower),
                    facecolor="#334155",
                    edgecolor="#0f172a",
                    linewidth=1,
                    alpha=alpha,
                    zorder=4,
                )
            )
        else:
            if not hasattr(ax, "zaxis"):
                raise ValueError("3D walls require a 3D axis")
            faces = []
            for dimension in range(3):
                other = [i for i in range(3) if i != dimension]
                for side in (lower[dimension], upper[dimension]):
                    face = []
                    for a, b in ((0, 0), (0, 1), (1, 1), (1, 0)):
                        vertex = lower.copy()
                        vertex[dimension] = side
                        vertex[other[0]] = (lower, upper)[a][other[0]]
                        vertex[other[1]] = (lower, upper)[b][other[1]]
                        face.append(vertex)
                    faces.append(face)
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolor="#334155",
                    edgecolor="#0f172a",
                    linewidth=0.5,
                    alpha=min(alpha, 0.25),
                )
            )
    return ax


def plot_trajectories(
    trajectories,
    *,
    walls=(),
    starts=None,
    goals=None,
    title="Particle trajectories",
    ax=None,
    alpha=0.55,
):
    """Plot (N,T,D) paths with start/goal markers and optional box walls."""
    paths = as_numpy(trajectories)
    if paths.ndim == 2:
        paths = paths[None]
    if paths.ndim != 3 or paths.shape[-1] not in (1, 2, 3) or not np.all(np.isfinite(paths)):
        raise ValueError("Expected finite trajectories with shape (N,T,D), D=1,2,3")
    dim = paths.shape[-1]
    figure, ax = _axes(dim, ax)
    for i, path in enumerate(paths):
        coordinates = [path[:, k] for k in range(dim)]
        if dim == 1:
            coordinates = [np.arange(len(path)), path[:, 0]]
        ax.plot(*coordinates, color=PALETTE[i % len(PALETTE)], alpha=alpha, linewidth=1.6)
    if dim == 1:
        ax.set_xlabel("Path step")
        ax.set_ylabel("x")
    else:
        starts = paths[:, 0] if starts is None else _points(starts)
        goals = paths[:, -1] if goals is None else _points(goals)
        ax.scatter(
            *starts.T,
            marker="o",
            s=42,
            color="#059669",
            edgecolors="white",
            label="Start A",
            zorder=6,
        )
        ax.scatter(
            *goals.T,
            marker="*",
            s=90,
            color="#e11d48",
            edgecolors="white",
            label="Goal B",
            zorder=6,
        )
        ax.legend(frameon=False)
    if len(walls):
        draw_walls(ax, walls, one_dimensional_path=dim == 1)
    ax.set_title(title)
    return figure, ax


def save_figure(figure, path, formats=("png", "svg"), dpi=180):
    """Export a figure in raster and editable vector formats; return paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("") if path.suffix else path
    outputs = []
    for fmt in formats:
        if fmt not in ("png", "svg", "pdf"):
            raise ValueError("Supported figure formats: png, svg, pdf")
        output = stem.with_suffix(f".{fmt}")
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(output)
    return outputs


def save_denoising_gif(history, path, *, labels=None, fps=8, walls=(), max_frames=80):
    """Animate points [steps,N,D] or paths [steps,N,T,D], using Pillow."""
    history = as_numpy(history)
    if (
        history.ndim not in (3, 4)
        or history.shape[-1] not in (1, 2, 3)
        or not np.all(np.isfinite(history))
    ):
        raise ValueError("history must have finite shape (steps,N,D) or (steps,N,T,D), D=1,2,3")
    if fps <= 0 or max_frames < 1:
        raise ValueError("fps and max_frames must be positive")
    if any(n == 0 for n in history.shape):
        raise ValueError("history cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.unique(np.linspace(0, len(history) - 1, min(max_frames, len(history))).astype(int))
    dim = history.shape[-1]
    figure, ax = _axes(dim)
    sample_axes = tuple(range(history.ndim - 1))
    bounds_min, bounds_max = history.min(axis=sample_axes), history.max(axis=sample_axes)
    margin = np.maximum((bounds_max - bounds_min) * 0.05, 0.1)

    def frame(index):
        ax.clear()
        if history.ndim == 4:
            plot_trajectories(
                history[index], walls=walls, ax=ax, title=f"Trajectory denoising — state {index}"
            )
        else:
            plot_dataset(history[index], labels, ax=ax, title=f"Denoising — state {index}")
            if len(walls):
                draw_walls(ax, walls)
        if dim == 1 and history.ndim == 4:
            ax.set_xlim(0, history.shape[2] - 1)
            ax.set_ylim(bounds_min[0] - margin[0], bounds_max[0] + margin[0])
        else:
            for k in range(dim):
                getattr(ax, ("set_xlim", "set_ylim", "set_zlim")[k])(
                    bounds_min[k] - margin[k], bounds_max[k] + margin[k]
                )
        return ()

    movie = animation.FuncAnimation(figure, frame, frames=indices, interval=1000 / fps)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)
    return path
