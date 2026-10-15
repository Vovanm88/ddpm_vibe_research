"""Separate spatial route playback from paired whole-path FM denoising."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import LineCollection
from PIL import Image

from diffusion_lab.navigation_visualization import (
    _RAINBOW,
    plot_navigation_denoising,
    plot_navigation_paths,
    save_navigation_denoising_gif,
    save_navigation_particle_gif,
    summarize_navigation_benchmark,
)

START = np.array([-0.9, 0.0])
GOAL = np.array([0.9, 0.0])
WALLS = np.array([[[-0.15, -0.3], [0.15, 0.3]]])
SCENE = {"start": START, "goal": GOAL, "walls": WALLS}


def _routes():
    top = np.stack((START, [-0.3, 0.6], [0.3, 0.6], GOAL))
    bottom = np.stack((START, [-0.3, -0.6], [0.3, -0.6], GOAL))
    return np.stack((top, bottom))


def _histories():
    paths = _routes()
    initial = paths.copy()
    initial[:, 1:3] = [[-2.0, 1.5], [2.0, -1.5]]
    straight = paths.copy()
    straight[:, 1:3, 1] = 0  # Free waypoints; connecting segment crosses the wall.
    teacher = np.stack((initial, straight, paths * 1.2, paths, paths))
    teacher[:, :, 0] = START
    teacher[:, :, -1] = GOAL
    student = np.stack((initial, straight, paths))
    return {"Teacher N4": teacher, "Student N2": student}


def test_path_and_fm_history_shapes_cannot_be_silently_confused():
    with pytest.raises(ValueError, match="Spatial paths"):
        plot_navigation_paths(_histories()["Teacher N4"], modes=[0, 1], **SCENE)
    with pytest.raises(ValueError, match="spatial path is not an FM history"):
        plot_navigation_denoising({"wrong": _routes()}, **SCENE)


def test_paired_initial_fullpath_required_before_gif_creation(tmp_path):
    histories = _histories()
    histories["Student N2"][0, 0, 1, 0] += 0.001
    with pytest.raises(ValueError, match="same initial full paths"):
        save_navigation_denoising_gif(histories, tmp_path / "bad.gif", **SCENE)
    assert not (tmp_path / "bad.gif").exists()


def test_fm_frame_is_actual_whole_path_and_crossing_segments_remain_visible():
    histories = _histories()
    original = {key: value.copy() for key, value in histories.items()}
    rng = torch.get_rng_state().clone()
    figure, axes = plot_navigation_denoising(histories, progress=0.25, **SCENE)
    try:
        for ax, (name, history), index in zip(axes.flat, histories.items(), (1, 0)):
            lines = next(item for item in ax.collections if isinstance(item, LineCollection))
            np.testing.assert_array_equal(np.asarray(lines.get_segments()), history[index])
            assert f"actual FM node {index}/" in ax.get_title()
            assert "outside view (raw unchanged)" in ax.get_title()
            assert lines.get_zorder() > max(patch.get_zorder() for patch in ax.patches)
            assert ax.get_xlim() == pytest.approx((-1.1, 1.1))
        assert "FM DENOISING · whole-route state" in figure._suptitle.get_text()
        assert any("particles are not moving" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(np.array_equal(histories[key], value) for key, value in original.items())


def test_static_route_modes_and_red_A_violet_B_are_separate_encodings():
    figure, ax = plot_navigation_paths(_routes(), modes=[0, 1], **SCENE)
    try:
        assert ax.lines[0].get_color() != ax.lines[1].get_color()
        np.testing.assert_allclose(ax.collections[0].get_facecolors()[0], _RAINBOW(0.0))
        np.testing.assert_allclose(ax.collections[1].get_facecolors()[0], _RAINBOW(1.0))
        legend = " ".join(text.get_text() for text in ax.get_legend().get_texts())
        assert "Top route" in legend and "Bottom route" in legend
        assert "evaluation only" in legend
    finally:
        plt.close(figure)


@pytest.mark.parametrize("substeps,expected_frames", [(1, 4), (2, 7)])
def test_physical_gif_visits_saved_waypoints_or_explicit_linear_substeps(
    tmp_path, substeps, expected_frames
):
    path = save_navigation_particle_gif(
        _routes(), tmp_path / f"physical{substeps}.gif", substeps=substeps, fps=5, **SCENE
    )
    with Image.open(path) as image:
        assert image.n_frames == expected_frames
        first = np.asarray(image.convert("RGB")).copy()
        image.seek(image.n_frames - 1)
        final = np.asarray(image.convert("RGB")).copy()
        assert np.mean(first != final) > 0.005


def test_fm_gif_uses_actual_states_and_may_have_different_nfe(tmp_path):
    path = save_navigation_denoising_gif(
        _histories(), tmp_path / "fm.gif", fps=5, max_frames=5, **SCENE
    )
    with Image.open(path) as image:
        assert image.n_frames == 5
        first = np.asarray(image.convert("RGB")).copy()
        image.seek(4)
        final = np.asarray(image.convert("RGB")).copy()
        assert np.mean(first != final) > 0.01


def test_cached_assessment_detects_wall_crossing_between_free_waypoints_and_raw_fidelity():
    histories = _histories()
    histories["Student N2"][-1, :, 1:3, 1] = 0
    conditions = np.repeat(np.concatenate((START, GOAL, WALLS.reshape(-1)))[None], 2, axis=0)
    result = summarize_navigation_benchmark(histories, conditions, parent="Teacher N4")
    teacher, student = result["rows"]
    assert teacher["quality"]["success_fraction"] == 1
    assert student["quality"]["success_fraction"] == 0
    assert student["quality"]["collision_free_fraction"] == 0
    assert (
        student["fidelity"]["paired_parent_interior_mse"]
        > student["fidelity"]["paired_parent_fullpath_mse"]
        > 0
    )
    assert [node["parent_node"] for node in student["fidelity"]["common_actual_fm_nodes"]] == [
        0,
        2,
        4,
    ]
    assert any("not learned endpoint accuracy" in text for text in result["limitations"])
