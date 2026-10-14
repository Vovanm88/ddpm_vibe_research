"""Original time colours, coordinates and endpoints survive actual GIF rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import LineCollection
from PIL import Image

from diffusion_lab.visualization import animation as trajectory_animation
from diffusion_lab.visualization.animation import plot_rainbow_trajectories, save_trajectory_gif


def test_torch_history_paths_and_explicit_noise_sample_labels():
    history = torch.tensor([[[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]])
    paths = history.permute(1, 0, 2)
    original = paths.clone()
    figure, axis = plot_rainbow_trajectories(paths, start_label="Noise x_T", end_label="Sample x_0")
    labels = {text.get_text() for text in axis.get_legend().get_texts()}
    assert labels == {"Noise x_T", "Sample x_0", "Current particle"}
    torch.testing.assert_close(paths, original)
    plt.close(figure)


def test_progress_keeps_original_segment_colors_and_both_endpoints():
    paths = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]]])
    original = paths.copy()
    full_figure, full_axis = plot_rainbow_trajectories(paths)
    partial_figure, partial_axis = plot_rainbow_trajectories(paths, progress=0.5)
    full = next(item for item in full_axis.collections if isinstance(item, LineCollection))
    partial = next(item for item in partial_axis.collections if isinstance(item, LineCollection))
    np.testing.assert_allclose(partial.get_array(), full.get_array()[:2])
    assert full.get_clim() == partial.get_clim() == (0, 1)
    assert full_axis.get_xlim() == partial_axis.get_xlim()
    assert full_axis.get_ylim() == partial_axis.get_ylim()
    np.testing.assert_allclose(
        np.asarray(partial.get_segments()), [[[0, 0], [1, 1]], [[1, 1], [2, 0]]]
    )
    labeled = {item.get_label(): item for item in partial_axis.collections}
    np.testing.assert_allclose(labeled["Start A"].get_offsets(), paths[:, 0])
    np.testing.assert_allclose(labeled["Finish B"].get_offsets(), paths[:, -1])
    np.testing.assert_allclose(labeled["Current particle"].get_offsets(), paths[:, 2])
    assert partial_axis.get_legend() is not None
    assert full_figure.axes[1].get_ylim() == partial_figure.axes[1].get_ylim() == (0, 1)
    np.testing.assert_array_equal(paths, original)
    plt.close(full_figure)
    plt.close(partial_figure)


def test_fractional_head_uses_actual_segment_not_reindexed_trail():
    paths = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 8.0]])
    figure, axis = plot_rainbow_trajectories(paths, progress=0.75)
    head = next(item for item in axis.collections if item.get_label() == "Current particle")
    np.testing.assert_allclose(head.get_offsets(), [[4, 4]])
    trail = next(item for item in axis.collections if isinstance(item, LineCollection))
    np.testing.assert_allclose(trail.get_array(), [0.25, 0.75])
    plt.close(figure)


def test_1d_time_axis_and_horizontal_walls_include_physical_bounds():
    path = np.array([[0.0], [0.5], [1.0]])
    figure, axis = plot_rainbow_trajectories(path, walls=[[[-2.0], [-1.5]]])
    markers = {item.get_label(): item for item in axis.collections}
    np.testing.assert_allclose(markers["Start A"].get_offsets(), [[0, 0]])
    np.testing.assert_allclose(markers["Finish B"].get_offsets(), [[2, 1]])
    assert axis.get_ylim()[0] < -2
    assert axis.get_ylim()[1] > 1
    assert "time index" in axis.get_xlabel()
    assert len(axis.patches) == 1
    plt.close(figure)


def test_3d_endpoints_and_limits_include_walls_without_changing_coordinates():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    original = path.copy()
    figure, axis = plot_rainbow_trajectories(path, progress=0.5, walls=[[[3, 4, 5], [4, 5, 6]]])
    assert hasattr(axis, "zaxis")
    markers = {item.get_label(): item for item in axis.collections}
    np.testing.assert_allclose(
        np.stack(markers["Current particle"]._offsets3d, axis=-1), [[1, 2, 3]]
    )
    np.testing.assert_allclose(np.stack(markers["Finish B"]._offsets3d, axis=-1), [[2, 0, 1]])
    assert axis.get_xlim()[1] > 4 and axis.get_ylim()[1] > 5 and axis.get_zlim()[1] > 6
    np.testing.assert_array_equal(path, original)
    plt.close(figure)


@pytest.mark.parametrize(
    "paths",
    [
        np.empty((0, 3, 2)),
        np.zeros((2, 1, 2)),
        np.zeros((2, 3, 4)),
        np.ones((3,)),
        np.array([[0.0, 1.0], [np.nan, 0.0]]),
    ],
)
def test_invalid_paths_are_rejected(paths):
    with pytest.raises(ValueError, match="paths must"):
        plot_rainbow_trajectories(paths)


@pytest.mark.parametrize("progress", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_progress_is_rejected(progress):
    with pytest.raises(ValueError, match="progress"):
        plot_rainbow_trajectories([[0, 0], [1, 1]], progress=progress)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": -1},
        {"fps": np.inf},
        {"max_frames": 1},
        {"max_frames": 2.5},
        {"max_frames": True},
        {"dpi": 0},
    ],
)
def test_invalid_gif_settings_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        save_trajectory_gif([[0, 0], [1, 1]], tmp_path / "invalid.gif", **kwargs)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_gif_has_actual_changing_frames_and_exact_first_last_heads(
    tmp_path, monkeypatch, dimension
):
    path = np.stack([np.linspace(0, 1, 5) ** (i + 1) for i in range(dimension)], axis=-1)
    original = path.copy()
    observed_heads = []
    original_trail = trajectory_animation._trail

    def observe(coordinates, progress):
        result = original_trail(coordinates, progress)
        observed_heads.append(result[2].copy())
        return result

    monkeypatch.setattr(trajectory_animation, "_trail", observe)
    output = save_trajectory_gif(
        path, tmp_path / f"trajectory_{dimension}d.gif", fps=8, max_frames=4, dpi=45
    )
    assert output.exists()
    with Image.open(output) as gif:
        assert gif.n_frames == 4
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"))
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("RGB"))
        assert first.shape == last.shape
        assert np.count_nonzero(np.any(first != last, axis=-1)) > 200
    expected_first, expected_last = path[:1], path[-1:]
    if dimension == 1:
        expected_first = np.column_stack(([0], expected_first[:, 0]))
        expected_last = np.column_stack(([4], expected_last[:, 0]))
    np.testing.assert_allclose(observed_heads[0], expected_first)
    np.testing.assert_allclose(observed_heads[-1], expected_last)
    np.testing.assert_array_equal(path, original)
