import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import trapezoid

from diffusion_lab.visualization import (
    Grid,
    estimate_density,
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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_nonconsecutive_labels_and_physical_density_coordinates():
    rng = np.random.default_rng(4)
    points = np.r_[rng.normal([2, 4], 0.2, size=(120, 2)), rng.normal([3, 6], 0.2, size=(120, 2))]
    labels = np.r_[np.full(120, 4), np.full(120, 9)]
    _, ax = plot_dataset(points, labels)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["4", "9"]
    grid = Grid.from_bounds([(1, 4), (3, 7)], (41, 63))
    values = estimate_density(points[:120], grid)
    y_index, x_index = np.unravel_index(np.argmax(values), values.shape)
    assert abs(grid.axes[0][x_index] - 2) < 0.2
    assert abs(grid.axes[1][y_index] - 4) < 0.2
    _, axes = plot_density(points, grid, labels=labels)
    assert len(axes) == 2
    assert "condition 4" in axes[0].get_title()


def test_one_dimensional_kde_is_normalized():
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(900, 1))
    grid = Grid.from_bounds([(-8, 8)], 1000)
    values = estimate_density(samples, grid)
    assert trapezoid(values, grid.axes[0]) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_figures_export_and_interactive_states(tmp_path, dimension):
    grid = Grid.from_bounds([(-1, 1)] * dimension, (8, 6, 4)[:dimension])
    field = -grid.points
    figure, ax = plot_vector_field(grid, field, stride=2)
    outputs = save_figure(figure, tmp_path / f"field_{dimension}")
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)
    assert "Original vector magnitude" in figure.axes[-1].get_ylabel()
    result = recover_potential(grid, field)
    plot_potential(grid, result)
    html = tmp_path / f"field_{dimension}.html"
    interactive = interactive_field(
        grid, {"condition=4 t=90": field, "condition=9 t=10": field * 2}, path=html
    )
    assert len(interactive.frames) == 2
    assert len(interactive.layout.sliders[0].steps) == 2
    assert html.is_file() and html.stat().st_size > 1000
    plot_dataset(grid.points)


def test_arrow_colors_keep_original_magnitudes():
    grid = Grid.from_bounds([(0, 2), (0, 1)], (5, 3))
    field = np.c_[np.arange(grid.size), np.zeros(grid.size)]
    _, ax = plot_vector_field(grid, field, normalize=True)
    quiver = ax.collections[0]
    np.testing.assert_array_equal(quiver.get_array(), np.arange(grid.size))
    np.testing.assert_allclose(np.hypot(quiver.U[1:], quiver.V[1:]), 0.375)
    assert quiver.U[0] == 0


def test_three_dimensional_density_is_a_slice():
    rng = np.random.default_rng(14)
    points = rng.normal(size=(200, 3))
    grid = Grid.from_bounds([(-2, 2)] * 3, (12, 17, 5))
    _, ax = plot_density(points, grid, slice_index=1)
    assert "z=-1 slice" in ax.get_title()


def test_wall_overlay_and_gif_export(tmp_path):
    paths = np.array([[[0, 0], [0.2, 0.3], [0.8, 0.3], [1, 0]]])
    walls = np.array([[[0.3, -0.1], [0.7, 0.15]]])
    _, ax = plot_trajectories(paths, walls=walls)
    assert len(ax.patches) == 1
    gif = save_denoising_gif(
        np.stack([paths[:, 0], paths[:, 1], paths[:, 2]]),
        tmp_path / "denoising.gif",
        walls=walls,
        fps=3,
    )
    assert gif.read_bytes().startswith(b"GIF")
    path_gif = save_denoising_gif(
        np.stack([paths, paths + 0.02]), tmp_path / "path_denoising.gif", walls=walls, fps=3
    )
    assert path_gif.read_bytes().startswith(b"GIF")
    grid = Grid.from_bounds([(-1, 1), (-1, 1)], (7, 9))
    explorer = interactive_field(grid, [-grid.points], walls=walls)
    assert len(explorer.layout.shapes) == 1


def test_box_wall_overlay_in_three_dimensions():
    paths = np.array([[[0, 0, 0], [0.2, 0.3, 0.5], [0.8, 0.3, 0.7], [1, 0, 1]]])
    walls = np.array([[[0.3, -0.1, 0.2], [0.7, 0.15, 0.7]]])
    _, ax = plot_trajectories(paths, walls=walls)
    assert len(ax.collections) == 3  # start, goal, box faces
    grid = Grid.from_bounds([(-1, 1)] * 3, 4)
    explorer = interactive_field(grid, [-grid.points], walls=walls)
    assert explorer.data[-1].type == "mesh3d"
