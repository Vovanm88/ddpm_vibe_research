"""Coordinate-safe field evaluation and discrete potential reconstruction.

Coordinates are always (x, y, z); arrays are indexed [z, y, x]. A recovered
potential uses F = -grad(U), with U=0 at one vertex of each connected component.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr


def as_numpy(values):
    """Convert NumPy or detached CPU/GPU torch data without importing torch."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


@dataclass(frozen=True)
class Grid:
    """A rectilinear 1–3D grid; axes are ascending in (x, y, z) order."""

    axes: tuple[np.ndarray, ...]

    def __post_init__(self):
        axes = tuple(np.asarray(axis, dtype=float).copy() for axis in self.axes)
        if not 1 <= len(axes) <= 3:
            raise ValueError("Grid requires one, two, or three coordinate axes")
        for axis in axes:
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
                raise ValueError("Each axis must contain at least two finite values")
            if not np.all(np.diff(axis) > 0):
                raise ValueError("Grid axes must be strictly increasing")
            axis.setflags(write=False)
        object.__setattr__(self, "axes", axes)

    @classmethod
    def from_bounds(cls, bounds: Sequence[Sequence[float]], resolution=40):
        """Build axes for [(xmin,xmax), ...]; resolution can differ per axis."""
        counts = (resolution,) * len(bounds) if np.isscalar(resolution) else tuple(resolution)
        if len(counts) != len(bounds):
            raise ValueError("resolution must specify one size per coordinate")
        if any(int(n) != n or n < 2 for n in counts):
            raise ValueError("Every grid resolution must be an integer >= 2")
        return cls(tuple(np.linspace(lo, hi, int(n)) for (lo, hi), n in zip(bounds, counts)))

    @property
    def ndim(self):
        return len(self.axes)

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axes[::-1])

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def mesh(self):
        """Coordinate arrays (x,y,z) with shape [z,y,x]."""
        return tuple(np.meshgrid(*self.axes[::-1], indexing="ij")[::-1])

    @property
    def points(self):
        return np.stack([component.ravel() for component in self.mesh], axis=-1)

    def reshape(self, values):
        values = as_numpy(values)
        if values.shape[: self.ndim] == self.shape:
            return values
        if values.ndim < 1 or values.shape[0] != self.size:
            raise ValueError(f"Expected {self.size} points, got shape {values.shape}")
        return values.reshape(self.shape + values.shape[1:])


def evaluate_field(field: Callable, grid: Grid, batch_size=4096):
    """Evaluate a NumPy-input callable in batches, accepting torch outputs.

    Adapt a torch pipeline explicitly, e.g. ``lambda p:
    pipeline.vector_field(torch.as_tensor(p, device=device).float(), t)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    points = grid.points
    batches = []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        vectors = as_numpy(field(batch))
        if vectors.shape != batch.shape or not np.all(np.isfinite(vectors)):
            raise ValueError(f"Field must return finite vectors with shape {batch.shape}")
        batches.append(vectors)
    return grid.reshape(np.concatenate(batches))


@dataclass(frozen=True)
class PotentialResult:
    potential: np.ndarray
    relative_residual: float
    residual_rms: float
    curl_rms: float
    relative_curl: float
    is_conservative: bool
    component_count: int
    solver_converged: bool
    message: str


def _curl_rms(grid, vectors, mask):
    # Circulation/area on every valid elementary face. This avoids differentiating
    # through a masked wall or NaN and respects anisotropic coordinate spacing.
    values = []
    for a, b in combinations(range(grid.ndim), 2):
        da, db = grid.ndim - 1 - a, grid.ndim - 1 - b
        s00 = [slice(None)] * grid.ndim
        s10, s01, s11 = (s00.copy() for _ in range(3))
        s00[da], s00[db] = slice(None, -1), slice(None, -1)
        s10[da], s10[db] = slice(1, None), slice(None, -1)
        s01[da], s01[db] = slice(None, -1), slice(1, None)
        s11[da], s11[db] = slice(1, None), slice(1, None)
        s00, s10, s01, s11 = map(tuple, (s00, s10, s01, s11))
        valid = mask[s00] & mask[s10] & mask[s01] & mask[s11]
        ha_shape, hb_shape = [1] * grid.ndim, [1] * grid.ndim
        ha_shape[da], hb_shape[db] = len(grid.axes[a]) - 1, len(grid.axes[b]) - 1
        ha = np.diff(grid.axes[a]).reshape(ha_shape)
        hb = np.diff(grid.axes[b]).reshape(hb_shape)
        circulation = (
            0.5 * ha * (vectors[s00][..., a] + vectors[s10][..., a])
            + 0.5 * hb * (vectors[s10][..., b] + vectors[s11][..., b])
            - 0.5 * ha * (vectors[s01][..., a] + vectors[s11][..., a])
            - 0.5 * hb * (vectors[s00][..., b] + vectors[s01][..., b])
        ) / (ha * hb)
        values.append(circulation[valid])
    if not values:
        return 0.0  # A one-dimensional field has no curl.
    # Each plane contributes one curl component. Sum its mean-square rather
    # than pooling face counts (which would shrink a 3D curl by sqrt(3)).
    return (
        float(np.sqrt(sum(np.mean(component**2) for component in values)))
        if all(len(component) for component in values)
        else float("nan")
    )


def recover_potential(
    grid: Grid,
    vectors,
    mask=None,
    residual_tolerance=0.05,
    curl_tolerance=0.05,
    solver_tolerance=1e-10,
):
    """Project a sampled field onto F=-grad(U) using sparse edge constraints.

    ``mask=True`` means an accessible grid vertex. Each connected component has
    an independent additive gauge, and masked vertices return NaN. In 1D this
    reduces to trapezoidal integration. In 2D/3D least squares fits finite
    differences to the mean endpoint field component on each accessible edge.

    ``is_conservative`` is a numerical diagnostic on this grid, not proof of a
    global potential: holes can support circulation even with small local curl,
    coarse sampling can miss loops, and interpolation may cross obstacles.
    Relative residual checks all graph edges (including loops around holes).
    Relative curl is curl RMS × mean axis span / field RMS.
    """
    vectors = np.asarray(grid.reshape(vectors), dtype=float)
    if vectors.shape != grid.shape + (grid.ndim,):
        raise ValueError(f"Expected field shape {grid.shape + (grid.ndim,)}")
    mask = np.ones(grid.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != grid.shape or not np.any(mask):
        raise ValueError("mask must match the grid and contain an accessible vertex")
    if not np.all(np.isfinite(vectors[mask])):
        raise ValueError("Field values at accessible vertices must be finite")
    if min(residual_tolerance, curl_tolerance, solver_tolerance) < 0:
        raise ValueError("Tolerances must be nonnegative")

    ids = np.full(grid.shape, -1, dtype=int)
    ids[mask] = np.arange(np.count_nonzero(mask))
    sources, targets, inverse_spacings, rhs_parts = [], [], [], []
    for coordinate, axis in enumerate(grid.axes):
        dimension = grid.ndim - 1 - coordinate
        left, right = [slice(None)] * grid.ndim, [slice(None)] * grid.ndim
        left[dimension], right[dimension] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = mask[left] & mask[right]
        hshape = [1] * grid.ndim
        hshape[dimension] = len(axis) - 1
        h = np.broadcast_to(np.diff(axis).reshape(hshape), valid.shape)
        sources.append(ids[left][valid])
        targets.append(ids[right][valid])
        inverse_spacings.append(1.0 / h[valid])
        rhs_parts.append(
            -0.5 * (vectors[left][..., coordinate][valid] + vectors[right][..., coordinate][valid])
        )
    source, target = np.concatenate(sources), np.concatenate(targets)
    inverse_spacing, rhs = np.concatenate(inverse_spacings), np.concatenate(rhs_parts)
    n_vertices, n_edges = np.count_nonzero(mask), len(source)
    adjacency = sparse.coo_matrix(
        (np.ones(2 * n_edges), (np.r_[source, target], np.r_[target, source])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    count, components = connected_components(adjacency, directed=False)
    anchors = np.unique(components, return_index=True)[1]
    free = np.ones(n_vertices, dtype=bool)
    free[anchors] = False
    rows = np.arange(n_edges)
    gradient = sparse.coo_matrix(
        (np.r_[-inverse_spacing, inverse_spacing], (np.r_[rows, rows], np.r_[source, target])),
        shape=(n_edges, n_vertices),
    ).tocsr()
    vertex_potential = np.zeros(n_vertices)
    if grid.ndim == 1:
        # A one-dimensional accessible component is a path with no loops.
        for label in range(count):
            members = np.flatnonzero(components == label)
            position_indices = np.flatnonzero(mask)[members]
            x = grid.axes[0][position_indices]
            f = vectors[position_indices, 0]
            vertex_potential[members[1:]] = np.cumsum(-0.5 * (f[:-1] + f[1:]) * np.diff(x))
        converged = True
    elif np.any(free) and n_edges:
        solution = lsqr(
            gradient[:, free],
            rhs,
            atol=solver_tolerance,
            btol=solver_tolerance,
            iter_lim=max(1000, 10 * n_vertices),
        )
        vertex_potential[free] = solution[0]
        converged = solution[1] in (0, 1, 2, 4, 5)
    else:
        converged = True
    error = gradient @ vertex_potential - rhs
    rhs_norm = np.linalg.norm(rhs)
    relative = float(np.linalg.norm(error) / rhs_norm) if rhs_norm else float(np.linalg.norm(error))
    residual_rms = float(np.sqrt(np.mean(error**2))) if n_edges else 0.0
    curl_rms = _curl_rms(grid, vectors, mask)
    field_rms = float(np.sqrt(np.mean(np.sum(vectors[mask] ** 2, axis=-1))))
    length = np.mean([axis[-1] - axis[0] for axis in grid.axes])
    relative_curl = float(curl_rms * length / field_rms) if field_rms else float(curl_rms)
    conservative = bool(
        converged
        and relative <= residual_tolerance
        and np.isfinite(relative_curl)
        and relative_curl <= curl_tolerance
    )
    potential = np.full(grid.shape, np.nan)
    potential[mask] = vertex_potential
    message = (
        "Compatible with a conservative field at this grid resolution. "
        if conservative
        else "Least-squares potential projection; the sampled field is not certified conservative. "
    ) + (
        f"F=-grad(U); {count} independently anchored connected component(s). "
        "Local curl and graph residual are numerical diagnostics; topology and grid "
        "resolution limit conclusions about a global potential."
    )
    return PotentialResult(
        potential,
        relative,
        residual_rms,
        curl_rms,
        relative_curl,
        conservative,
        count,
        converged,
        message,
    )
