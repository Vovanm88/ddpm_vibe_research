"""Obstacle-aware trajectory datasets, geometric validation, and training losses."""

from __future__ import annotations

import heapq
from itertools import product
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import SampleDataset


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _segment_box_intersections(start, end, walls, margin: float = 0.0):
    """Exact closed-segment/AABB slab test; includes crossings between waypoints."""
    start, end, walls = np.asarray(start), np.asarray(end), np.asarray(walls)
    if len(walls) == 0:
        return np.zeros(start.shape[:-1], dtype=bool)
    delta = (end - start)[..., None, :]
    origin = start[..., None, :]
    low, high = walls[:, 0] - margin, walls[:, 1] + margin
    parallel = np.abs(delta) < 1e-12
    safe_delta = np.where(parallel, 1.0, delta)
    first, last = (low - origin) / safe_delta, (high - origin) / safe_delta
    entry, exit = np.minimum(first, last), np.maximum(first, last)
    outside = (origin < low) | (origin > high)
    entry = np.where(parallel, np.where(outside, np.inf, -np.inf), entry)
    exit = np.where(parallel, np.where(outside, -np.inf, np.inf), exit)
    enter_t = np.maximum(entry.max(axis=-1), 0)
    exit_t = np.minimum(exit.min(axis=-1), 1)
    return (enter_t <= exit_t).any(axis=-1)


def collision_loss(
    paths: torch.Tensor,
    walls: torch.Tensor | np.ndarray,
    *,
    margin: float = 0.035,
    interpolation_steps: int = 8,
) -> torch.Tensor:
    """Differentiable obstacle penalty for points AND connecting segments.

    ``walls`` is [W,2,D] or per-example [B,W,2,D], with low/high corners.
    Exact slab overlap penalizes even a thin wall crossed between sampled points;
    interpolated signed penetration supplies gradients inside obstacle regions.
    Zero loss does not certify feasibility: use ``collision_free`` for evaluation.
    """
    if paths.ndim == 2:
        paths = paths.unsqueeze(0)
    if paths.ndim != 3 or paths.shape[1] < 2:
        raise ValueError("paths must have shape [B,L,D] with L >= 2")
    if margin < 0 or interpolation_steps < 2:
        raise ValueError("margin must be nonnegative; interpolation_steps must be >= 2")
    boxes = torch.as_tensor(walls, dtype=paths.dtype, device=paths.device)
    if boxes.ndim == 3:
        boxes = boxes.unsqueeze(0)
    if boxes.ndim != 4 or boxes.shape[-2:] != (2, paths.shape[-1]):
        raise ValueError("walls must have shape [W,2,D] or [B,W,2,D]")
    if boxes.shape[0] not in (1, paths.shape[0]):
        raise ValueError("batched walls must match paths batch size")
    if boxes.shape[1] == 0:
        return paths.sum() * 0
    low, high = boxes[:, None, :, 0, :] - margin, boxes[:, None, :, 1, :] + margin
    start, delta = paths[:, :-1, None, :], (paths[:, 1:] - paths[:, :-1])[:, :, None, :]
    parallel = delta.abs() < 1e-8
    safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
    first, last = (low - start) / safe_delta, (high - start) / safe_delta
    entry, exit = torch.minimum(first, last), torch.maximum(first, last)
    outside = (start < low) | (start > high)
    entry = torch.where(parallel, torch.where(outside, 2.0, -1.0), entry)
    exit = torch.where(parallel, torch.where(outside, -1.0, 2.0), exit)
    overlap = F.relu(exit.amin(dim=-1).clamp(max=1) - entry.amax(dim=-1).clamp(min=0))
    overlap_length = overlap * delta.norm(dim=-1)
    fractions = torch.linspace(0, 1, interpolation_steps, device=paths.device, dtype=paths.dtype)
    points = start[:, :, None] + fractions[None, None, :, None, None] * delta[:, :, None]
    depth = torch.minimum(points - low[:, :, None], high[:, :, None] - points).amin(dim=-1)
    return overlap_length.square().mean() + F.relu(depth).square().mean()


def endpoint_loss(paths: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
    """MSE against the start/goal prefix of a navigation condition vector."""
    if paths.ndim != 3 or conditions.ndim != 2 or len(paths) != len(conditions):
        raise ValueError("expected paths [B,L,D] and conditions [B,K]")
    dim = paths.shape[-1]
    if conditions.shape[1] < 2 * dim:
        raise ValueError("navigation conditions must begin with start(D), goal(D)")
    return 0.5 * (
        F.mse_loss(paths[:, 0], conditions[:, :dim])
        + F.mse_loss(paths[:, -1], conditions[:, dim : 2 * dim])
    )


def walls_from_conditions(conditions: torch.Tensor, dim: int) -> torch.Tensor:
    """Decode the shared ``start, goal, low/high wall corners`` contract."""
    if conditions.ndim != 2 or dim < 1 or conditions.shape[1] < 2 * dim:
        raise ValueError("expected conditions [B,K] beginning with start and goal")
    features = conditions[:, 2 * dim :]
    if features.shape[1] % (2 * dim):
        raise ValueError("wall condition features must be divisible by 2 * dim")
    return features.reshape(len(conditions), -1, 2, dim)


class WallNavigation:
    """A bounded 1D/2D/3D particle scene with axis-aligned solid walls.

    Reference trajectories use A* on a collision-checked grid, line-of-sight
    simplification, and corner-preserving resampling. Analytical local potential
    fields are available for visualization; they can have local minima and are
    not used as a substitute for the reference planner.
    """

    def __init__(
        self,
        dim: int = 2,
        *,
        walls=None,
        bounds=None,
        path_length: int = 32,
        grid_resolution: int | None = None,
        margin: float = 0.035,
    ) -> None:
        if dim not in (1, 2, 3):
            raise ValueError("navigation supports dim=1, 2, or 3")
        if path_length < 2 or margin < 0:
            raise ValueError("path_length must be >= 2 and margin nonnegative")
        self.dim, self.path_length, self.margin = dim, path_length, margin
        self.grid_resolution = grid_resolution or (25 if dim < 3 else 15)
        if self.grid_resolution < 3:
            raise ValueError("grid_resolution must be >= 3")
        self.bounds = _as_numpy(
            bounds if bounds is not None else [np.full(dim, -1.0), np.full(dim, 1.0)]
        )
        if (
            self.bounds.shape != (2, dim)
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[0] >= self.bounds[1])
        ):
            raise ValueError("bounds must contain ordered finite low/high corners [2,D]")
        if walls is None:
            walls = (
                []
                if dim == 1
                else [
                    [
                        [-0.3, -1.0] + ([-1.0] if dim == 3 else []),
                        [-0.15, 0.35] + ([1.0] if dim == 3 else []),
                    ],
                    [
                        [0.2, -0.35] + ([-1.0] if dim == 3 else []),
                        [0.35, 1.0] + ([1.0] if dim == 3 else []),
                    ],
                ]
            )
        self.walls = _as_numpy(walls).reshape(-1, 2, dim)
        if not np.isfinite(self.walls).all() or np.any(self.walls[:, 0] >= self.walls[:, 1]):
            raise ValueError("walls must contain ordered finite low/high corners [W,2,D]")
        self._nodes = None
        self._adjacency = None
        self._route_cache: dict[tuple[int, int], list[int]] = {}

    @property
    def condition_dim(self) -> int:
        return 2 * self.dim + self.walls.size

    def _points_free(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        inside_bounds = ((points >= self.bounds[0]) & (points <= self.bounds[1])).all(axis=-1)
        if len(self.walls) == 0:
            return inside_bounds
        inside_wall = (
            (
                (points[..., None, :] >= self.walls[:, 0] - self.margin)
                & (points[..., None, :] <= self.walls[:, 1] + self.margin)
            )
            .all(axis=-1)
            .any(axis=-1)
        )
        return inside_bounds & ~inside_wall

    def collision_free(self, paths) -> np.ndarray:
        """Return one boolean per path, checking every segment, walls and bounds."""
        paths = _as_numpy(paths)
        if paths.ndim < 2 or paths.shape[-1] != self.dim or paths.shape[-2] < 1:
            raise ValueError("paths must have shape [...,L,D]")
        valid = np.isfinite(paths).all(axis=(-2, -1)) & self._points_free(paths).all(axis=-1)
        crossing = _segment_box_intersections(
            paths[..., :-1, :], paths[..., 1:, :], self.walls, self.margin
        )
        return valid & ~crossing.any(axis=-1)

    def _build_grid(self) -> None:
        if self._nodes is not None:
            return
        axes = [np.linspace(low, high, self.grid_resolution) for low, high in zip(*self.bounds)]
        all_nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.dim)
        grid_shape = (self.grid_resolution,) * self.dim
        free_indices = np.flatnonzero(self._points_free(all_nodes))
        self._nodes = all_nodes[free_indices]
        if not len(self._nodes):
            raise ValueError(
                "scene has no free grid nodes; increase grid_resolution or change walls"
            )
        node_lookup = np.full(len(all_nodes), -1, dtype=int)
        node_lookup[free_indices] = np.arange(len(free_indices))
        coordinates = np.stack(np.unravel_index(free_indices, grid_shape), axis=1)
        source_nodes, target_nodes = [], []
        for axis, sign in product(range(self.dim), (-1, 1)):
            adjacent = coordinates.copy()
            adjacent[:, axis] += sign
            valid = ((adjacent >= 0) & (adjacent < self.grid_resolution)).all(axis=1)
            source = np.flatnonzero(valid)
            target = node_lookup[np.ravel_multi_index(adjacent[valid].T, grid_shape)]
            free_target = target >= 0
            source, target = source[free_target], target[free_target]
            safe = ~_segment_box_intersections(
                self._nodes[source], self._nodes[target], self.walls, self.margin
            )
            source_nodes.extend(source[safe].tolist())
            target_nodes.extend(target[safe].tolist())
        self._adjacency = [[] for _ in range(len(self._nodes))]
        for source, target in zip(source_nodes, target_nodes):
            self._adjacency[source].append(
                (target, float(np.linalg.norm(self._nodes[source] - self._nodes[target])))
            )

    def _nearest_reachable_node(self, point: np.ndarray) -> int:
        distance = np.linalg.norm(self._nodes - point, axis=1)
        for node in np.argsort(distance):
            if not _segment_box_intersections(point, self._nodes[node], self.walls, self.margin):
                return int(node)
        raise ValueError("endpoint cannot reach the planning grid; increase grid_resolution")

    def _astar(self, start: int, goal: int) -> list[int]:
        key = (start, goal)
        if key in self._route_cache:
            return self._route_cache[key]
        distances = {start: 0.0}
        parents: dict[int, int] = {}
        queue = [(float(np.linalg.norm(self._nodes[start] - self._nodes[goal])), 0.0, start)]
        while queue:
            _, distance, node = heapq.heappop(queue)
            if distance > distances.get(node, np.inf):
                continue
            if node == goal:
                route = [goal]
                while route[-1] != start:
                    route.append(parents[route[-1]])
                route.reverse()
                self._route_cache[key] = route
                return route
            for neighbor, edge_length in self._adjacency[node]:
                candidate = distance + edge_length
                if candidate < distances.get(neighbor, np.inf):
                    distances[neighbor], parents[neighbor] = candidate, node
                    heuristic = float(np.linalg.norm(self._nodes[neighbor] - self._nodes[goal]))
                    heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        raise ValueError("no collision-free route exists on this planning grid")

    def _simplify(self, points: np.ndarray) -> np.ndarray:
        kept, current = [points[0]], 0
        while current < len(points) - 1:
            candidate = len(points) - 1
            while candidate > current + 1 and _segment_box_intersections(
                points[current], points[candidate], self.walls, self.margin
            ):
                candidate -= 1
            kept.append(points[candidate])
            current = candidate
        result = np.asarray(kept)
        if len(result) > 1:
            result = result[np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-12]]
        return result

    def _resample(self, points: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if len(lengths) == 0 or lengths.sum() <= 1e-12:
            return np.repeat(points[:1], self.path_length, axis=0)
        if len(lengths) > self.path_length - 1:
            raise ValueError("path_length is too short to preserve all collision-free corners")
        extra = self.path_length - 1 - len(lengths)
        allocation = extra * lengths / lengths.sum()
        intervals = np.floor(allocation).astype(int) + 1
        remainder = self.path_length - 1 - intervals.sum()
        intervals[np.argsort(-(allocation - np.floor(allocation)))[:remainder]] += 1
        result = [points[:1]]
        for start, end, count in zip(points[:-1], points[1:], intervals):
            fraction = np.linspace(0, 1, count + 1)[1:, None]
            result.append(start + fraction * (end - start))
        return np.concatenate(result)

    def plan(self, start, goal) -> np.ndarray:
        """Return [path_length,D], preserving exact endpoints and obstacle corners."""
        start, goal = _as_numpy(start), _as_numpy(goal)
        if start.shape != (self.dim,) or goal.shape != (self.dim,):
            raise ValueError("start and goal must have shape [D]")
        if not self._points_free(np.stack((start, goal))).all():
            raise ValueError("start and goal must lie in free space inside bounds")
        if not _segment_box_intersections(start, goal, self.walls, self.margin):
            points = np.stack((start, goal))
        else:
            self._build_grid()
            start_node, goal_node = (
                self._nearest_reachable_node(start),
                self._nearest_reachable_node(goal),
            )
            route = self._astar(start_node, goal_node)
            points = self._simplify(np.concatenate((start[None], self._nodes[route], goal[None])))
        path = self._resample(points)
        if not self.collision_free(path):
            raise RuntimeError("reference planner produced an invalid route")
        return path.astype(np.float32)

    def encode_condition(self, start, goal) -> np.ndarray:
        return np.concatenate((_as_numpy(start), _as_numpy(goal), self.walls.ravel())).astype(
            np.float32
        )

    def sample_dataset(self, n_samples: int = 512, seed: int = 42) -> SampleDataset:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        paths, conditions = [], []
        span = self.bounds[1] - self.bounds[0]
        for _ in range(n_samples):
            for attempt in range(128):
                start = rng.uniform(self.bounds[0] + 0.02 * span, self.bounds[1] - 0.02 * span)
                goal = rng.uniform(self.bounds[0] + 0.02 * span, self.bounds[1] - 0.02 * span)
                if self.dim > 1:
                    # Boundary strips make the default alternating walls matter.
                    start[0] = rng.uniform(
                        self.bounds[0, 0] + 0.02 * span[0], self.bounds[0, 0] + 0.15 * span[0]
                    )
                    goal[0] = rng.uniform(
                        self.bounds[1, 0] - 0.15 * span[0], self.bounds[1, 0] - 0.02 * span[0]
                    )
                try:
                    path = self.plan(start, goal)
                except ValueError:
                    continue
                paths.append(path)
                conditions.append(self.encode_condition(start, goal))
                break
            else:
                raise ValueError(
                    "could not sample reachable start/goal pairs; check scene connectivity"
                )
        dataset = SampleDataset(
            np.stack(paths),
            np.stack(conditions),
            metadata={
                "name": f"navigation_{self.dim}d",
                "dim": self.dim,
                "seed": seed,
                "walls": self.walls.tolist(),
                "bounds": self.bounds.tolist(),
                "path_length": self.path_length,
                "margin": self.margin,
                "grid_resolution": self.grid_resolution,
                "condition_layout": "start(D), goal(D), walls(W,2,D) flattened in C order",
                "planner": "A* with exact segment-AABB tests and corner-preserving resampling",
            },
        )
        dataset.environment = self
        return dataset

    def potential(
        self,
        points,
        goal,
        *,
        repulsion: float = 5.0,
        influence: float = 0.2,
        softness: float = 0.05,
    ):
        """Attractive quadratic plus smooth signed-distance wall repulsion."""
        is_tensor = isinstance(points, torch.Tensor)
        tensor = points if is_tensor else torch.as_tensor(np.asarray(points), dtype=torch.float32)
        target = torch.as_tensor(goal, dtype=tensor.dtype, device=tensor.device)
        if tensor.shape[-1] != self.dim or target.shape != (self.dim,) or softness <= 0:
            raise ValueError("points must end in D, goal must be [D], and softness > 0")
        value = 0.5 * (tensor - target).square().sum(dim=-1)
        if len(self.walls):
            boxes = torch.as_tensor(self.walls, dtype=tensor.dtype, device=tensor.device)
            centers, half_size = boxes.mean(dim=1), (boxes[:, 1] - boxes[:, 0]) * 0.5 + self.margin
            q = (tensor[..., None, :] - centers).abs() - half_size
            signed_distance = F.relu(q).norm(dim=-1) + q.amax(dim=-1).clamp(max=0)
            barrier = softness * F.softplus((influence - signed_distance) / softness)
            value = value + repulsion * barrier.square().sum(dim=-1)
        return value if is_tensor else value.detach().cpu().numpy()

    def vector_field(self, points, goal, **kwargs):
        """Return -grad(potential); vectors have the same [...,D] shape as points."""
        is_tensor = isinstance(points, torch.Tensor)
        tensor = (
            points.detach().clone()
            if is_tensor
            else torch.as_tensor(np.asarray(points), dtype=torch.float32)
        )
        tensor.requires_grad_(True)
        with torch.enable_grad():
            values = self.potential(tensor, goal, **kwargs)
            gradient = torch.autograd.grad(values.sum(), tensor)[0]
        vectors = -gradient
        return vectors if is_tensor else vectors.detach().cpu().numpy()
