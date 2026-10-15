"""Condition routing must expose actual paired trajectories, including failures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import LineCollection
from PIL import Image

from diffusion_lab.conditional_visualization import (
    _PROGRESS_CMAP,
    _ideal_curves,
    condition_routing_widget,
    plot_condition_routing,
    save_condition_routing_gif,
)


def _data(n_rays=16):
    curves = _ideal_curves(n_rays, 10 / 11)
    real = np.concatenate([curve[::25] for curve in curves])
    real_labels = np.repeat(np.arange(n_rays), len(curves[0][::25]))
    initial = np.array([[1.4, -1.1], [-1.2, 1.3]])
    histories = {}
    for label in range(n_rays):
        final = np.stack((curves[label][-1], curves[label][128]))
        # Intentionally nonlinear middle node; fractional interpolation is wrong.
        middle = 0.2 * initial + 0.8 * final + np.array([0.05, -0.04])
        histories[label] = np.stack((initial, middle, final))
    histories[None] = np.stack((initial, initial * 0.5, np.zeros_like(initial)))
    return histories, real, real_labels


def test_same_noise_required_for_every_cached_condition_before_any_output(tmp_path):
    histories, real, labels = _data()
    histories[None][0, 1, 0] += 1e-6  # Even an unshown null condition must be checked.
    with pytest.raises(ValueError, match="exactly the same initial noise"):
        save_condition_routing_gif(histories, real, labels, tmp_path / "bad.gif")
    assert not (tmp_path / "bad.gif").exists()
    histories, real, labels = _data()
    histories[5] = histories[5][:-1]
    with pytest.raises(ValueError, match="same T and batch shape"):
        plot_condition_routing(histories, real, labels)


def test_actual_node_shared_bounds_target_curve_and_time_colors_preserve_inputs():
    histories, real, labels = _data()
    saved = {key: value.copy() for key, value in histories.items()}
    rng = torch.get_rng_state().clone()
    figure, axes = plot_condition_routing(histories, real, labels, noise_index=1, frame_index=1)
    try:
        assert axes.shape == (2, 2)
        limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes.flat]
        assert all(value == limits[0] for value in limits)
        for ax, label in zip(axes.flat, (0, 4, 8, 12)):
            assert f"Requested: ray {label}" in ax.get_title()
            assert f"Final nearest proxy: {label}" in ax.get_title()
            assert "actual node 1/2" in ax.get_title()
            trail = next(item for item in ax.collections if isinstance(item, LineCollection))
            np.testing.assert_array_equal(trail.get_segments()[0], histories[label][:2, 1])
            np.testing.assert_array_equal(
                ax.collections[-1].get_offsets(), histories[label][1:2, 1]
            )
            np.testing.assert_allclose(ax.collections[-3].get_facecolors()[0], _PROGRESS_CMAP(0.0))
            np.testing.assert_allclose(ax.collections[-2].get_facecolors()[0], _PROGRESS_CMAP(1.0))
            target = next(line for line in ax.lines if line.get_linestyle() == "--")
            np.testing.assert_allclose(target.get_xydata(), _ideal_curves(16, 10 / 11)[label])
    finally:
        plt.close(figure)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(np.array_equal(histories[key], value) for key, value in saved.items())


def test_ignored_condition_remains_visible_in_endpoint_proxy():
    histories, real, labels = _data(n_rays=4)
    histories[1] = histories[0].copy()  # A model that ignores the condition.
    figure, axes = plot_condition_routing(histories, real, labels, n_rays=4, labels=[1, None])
    try:
        assert "Requested: ray 1" in axes[0, 0].get_title()
        assert "Final nearest proxy: 0" in axes[0, 0].get_title()
        assert "null / unconditional" in axes[0, 1].get_title()
        assert "undefined at origin" in axes[0, 1].get_title()
    finally:
        plt.close(figure)


def test_gif_uses_each_actual_node_and_changes_frames(tmp_path):
    histories, real, labels = _data(n_rays=4)
    path = save_condition_routing_gif(
        histories, real, labels, tmp_path / "routing.gif", n_rays=4, labels=[0, 2], fps=4
    )
    with Image.open(path) as image:
        assert image.n_frames == 3
        image.seek(0)
        first = np.asarray(image.convert("RGB")).copy()
        image.seek(2)
        last = np.asarray(image.convert("RGB")).copy()
        assert np.mean(first != last) > 0.005


def test_cached_widget_switches_null_noise_and_exact_frame_without_resampling(monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    histories, real, labels = _data(n_rays=4)
    saved = {key: value.copy() for key, value in histories.items()}
    rng = torch.get_rng_state().clone()
    widget = condition_routing_widget(histories, real, labels, n_rays=4)
    try:
        assert widget.controls["play"].step == widget.controls["frame"].step == 1
        assert ("Null / unconditional", -1) in widget.controls["target"].options
        widget.controls["target"].value = -1
        widget.controls["noise_index"].value = 1
        widget.controls["frame"].value = 2
        assert widget.state["error"] is None
        assert widget.state["condition"] is None
        assert widget.state["frame_index"] == 2
        assert widget.state["noise_index"] == 1
        np.testing.assert_array_equal(widget.state["history"], histories[None])
        assert widget.controls["play"].value == 2
        assert torch.equal(rng, torch.get_rng_state())
        assert all(np.array_equal(histories[key], value) for key, value in saved.items())
    finally:
        widget.history_link.unlink()
        widget.close()


@pytest.mark.parametrize(
    "argument,value", [("noise_index", 2), ("frame_index", 0.5), ("radius", 0)]
)
def test_invalid_plot_settings_rejected(argument, value):
    histories, real, labels = _data(n_rays=4)
    with pytest.raises(ValueError):
        plot_condition_routing(histories, real, labels, n_rays=4, **{argument: value})
