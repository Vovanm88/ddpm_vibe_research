"""Portable Plotly field explorers; browser sliders need no notebook widgets."""

from collections.abc import Mapping
from itertools import product
from pathlib import Path

import numpy as np

from .fields import Grid, as_numpy


def interactive_field(
    grid: Grid,
    fields,
    *,
    labels=None,
    samples=None,
    mask=None,
    walls=(),
    stride=1,
    title="Diffusion field explorer",
    path=None,
):
    """Create a 1D/2D/3D Plotly field explorer with a state slider.

    ``fields`` can be a sequence or an ordered mapping such as
    ``{'class=0, t=900': field_0, 'class=0, t=500': field_1, ...}``.
    Directions have fixed display length and the shared color scale shows true
    magnitude, so timesteps/conditions are directly comparable. If path is set,
    write self-contained HTML with bundled Plotly JavaScript for offline viewing.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Install diffusion-lab[interactive] to use interactive Plotly plots"
        ) from exc
    if isinstance(fields, Mapping):
        if labels is not None:
            raise ValueError("Mapping field keys already provide the state labels")
        labels, fields = list(map(str, fields.keys())), list(fields.values())
    else:
        fields = list(fields)
        labels = [str(i) for i in range(len(fields))] if labels is None else list(map(str, labels))
    if not fields or len(fields) != len(labels):
        raise ValueError("Provide nonempty fields and one label per state")
    if int(stride) != stride or stride < 1:
        raise ValueError("stride must be a positive integer")
    valid = np.ones(grid.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != grid.shape:
        raise ValueError("mask must match grid shape")
    selected = tuple(slice(None, None, int(stride)) for _ in range(grid.ndim))
    keep = valid[selected]
    points = np.stack(grid.mesh, axis=-1)[selected][keep]
    if not len(points):
        raise ValueError("No accessible field points remain")
    arrays = []
    for field in fields:
        vectors = grid.reshape(field)
        if vectors.shape != grid.shape + (grid.ndim,):
            raise ValueError("Field shape must match grid")
        values = vectors[selected][keep]
        if not np.all(np.isfinite(values)):
            raise ValueError("Accessible vectors must be finite")
        arrays.append(values)
    cmax = max(max(float(np.linalg.norm(v, axis=-1).max()) for v in arrays), 1e-12)
    length = 0.72 * stride * min(np.min(np.diff(axis)) for axis in grid.axes)

    def traces(vectors):
        magnitude = np.linalg.norm(vectors, axis=-1)
        directions = np.zeros_like(vectors, dtype=float)
        np.divide(vectors, magnitude[:, None], out=directions, where=magnitude[:, None] > 0)
        tips = points + directions * length
        markers = dict(
            size=5,
            color=magnitude,
            colorscale="Viridis",
            cmin=0,
            cmax=cmax,
            colorbar=dict(title="Original |F|"),
            showscale=True,
        )
        hover = "Magnitude %{customdata:.4g}<extra></extra>"
        if grid.ndim == 3:
            dots = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=markers,
                customdata=magnitude,
                hovertemplate=hover,
                showlegend=False,
            )
            arrows = go.Cone(
                x=tips[:, 0],
                y=tips[:, 1],
                z=tips[:, 2],
                u=directions[:, 0],
                v=directions[:, 1],
                w=directions[:, 2],
                anchor="tip",
                sizemode="absolute",
                sizeref=length * 0.55,
                colorscale=[[0, "#475569"], [1, "#475569"]],
                showscale=False,
                hoverinfo="skip",
            )
            segments = np.stack([points, tips, np.full_like(points, np.nan)], axis=1).reshape(-1, 3)
            lines = go.Scatter3d(
                x=segments[:, 0],
                y=segments[:, 1],
                z=segments[:, 2],
                mode="lines",
                line=dict(color="#475569", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
            return [dots, arrows, lines]
        p = np.c_[points, np.zeros(len(points))] if grid.ndim == 1 else points
        q = np.c_[tips, np.zeros(len(points))] if grid.ndim == 1 else tips
        d = np.c_[directions, np.zeros(len(points))] if grid.ndim == 1 else directions
        perpendicular = np.c_[-d[:, 1], d[:, 0]]
        head_base = q - d * length * 0.28
        left = head_base + perpendicular * length * 0.13
        right = head_base - perpendicular * length * 0.13
        empty = np.full_like(p, np.nan)
        segments = np.stack([p, q, empty, left, q, right, empty], axis=1).reshape(-1, 2)
        dots = go.Scatter(
            x=p[:, 0],
            y=p[:, 1],
            mode="markers",
            marker=markers,
            customdata=magnitude,
            hovertemplate=hover,
            showlegend=False,
        )
        arrows = go.Scatter(
            x=segments[:, 0],
            y=segments[:, 1],
            mode="lines",
            line=dict(color="#475569", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
        return [dots, arrows]

    initial = traces(arrays[0])
    frame_count = len(initial)
    frames = [
        go.Frame(name=str(i), data=traces(v), traces=list(range(frame_count)))
        for i, v in enumerate(arrays)
    ]
    figure = go.Figure(data=initial, frames=frames)
    if samples is not None:
        samples = as_numpy(samples)
        if samples.ndim != 2 or samples.shape[1] != grid.ndim:
            raise ValueError("samples must have shape (N,D) matching grid")
        marker = dict(size=3, color="#94a3b8", opacity=0.25)
        if grid.ndim == 3:
            figure.add_trace(
                go.Scatter3d(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    z=samples[:, 2],
                    mode="markers",
                    marker=marker,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=samples[:, 0],
                    y=samples[:, 1] if grid.ndim == 2 else np.zeros(len(samples)),
                    mode="markers",
                    marker=marker,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    steps = [
        dict(
            label=label,
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
        )
        for i, label in enumerate(labels)
    ]
    figure.update_layout(
        template="plotly_white",
        title=f"{title} — direction arrows, true |F| colors",
        height=750,
        margin=dict(b=130),
        sliders=[dict(active=0, currentvalue=dict(prefix="State: "), steps=steps)],
    )
    if grid.ndim == 3:
        figure.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
                xaxis=dict(range=[grid.axes[0][0], grid.axes[0][-1]]),
                yaxis=dict(range=[grid.axes[1][0], grid.axes[1][-1]]),
                zaxis=dict(range=[grid.axes[2][0], grid.axes[2][-1]]),
            )
        )
    else:
        figure.update_xaxes(title="x", range=[grid.axes[0][0], grid.axes[0][-1]])
        figure.update_yaxes(
            title="y" if grid.ndim == 2 else "",
            scaleanchor="x",
            scaleratio=1,
            range=[grid.axes[1][0], grid.axes[1][-1]] if grid.ndim == 2 else [-length, length],
        )
    for wall in walls:
        if hasattr(wall, "lower") and hasattr(wall, "upper"):
            bounds = np.asarray([wall.lower, wall.upper], dtype=float)
        else:
            bounds = np.asarray(wall, dtype=float)
        if bounds.shape == (4,) and grid.ndim == 2:
            bounds = bounds.reshape(2, 2)
        if bounds.shape != (2, grid.ndim) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Wall shape must be (2,D) with positive extent")
        if grid.ndim < 3:
            y0, y1 = bounds[:, 1] if grid.ndim == 2 else (-length, length)
            figure.add_shape(
                type="rect",
                x0=bounds[0, 0],
                y0=y0,
                x1=bounds[1, 0],
                y1=y1,
                fillcolor="#334155",
                line=dict(color="#0f172a"),
                layer="above",
            )
        else:
            corners = np.asarray(
                [np.where(bits, bounds[1], bounds[0]) for bits in product((0, 1), repeat=3)]
            )
            # Twelve outward-independent triangles cover the six box faces.
            triangles = np.asarray(
                [
                    (0, 1, 3),
                    (0, 3, 2),
                    (4, 6, 7),
                    (4, 7, 5),
                    (0, 4, 5),
                    (0, 5, 1),
                    (2, 3, 7),
                    (2, 7, 6),
                    (0, 2, 6),
                    (0, 6, 4),
                    (1, 5, 7),
                    (1, 7, 3),
                ]
            )
            figure.add_trace(
                go.Mesh3d(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    z=corners[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color="#334155",
                    opacity=0.25,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path, include_plotlyjs=True, full_html=True)
    return figure
