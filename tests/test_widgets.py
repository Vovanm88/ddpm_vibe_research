"""Exercise widget callbacks with actual tiny PyTorch / Diffusers pipelines."""

import matplotlib

matplotlib.use("Agg")

import IPython.display
import numpy as np
import pytest
import torch

pytest.importorskip("ipywidgets")

from diffusion_lab.environments import SampleDataset
from diffusion_lab.models import PointDenoiser
from diffusion_lab.pipeline import ResearchDiffusionPipeline
from diffusion_lab.schedulers import make_scheduler
from diffusion_lab.visualization import notebook_field_explorer, notebook_sampling_explorer


@pytest.fixture(autouse=True)
def headless_display(monkeypatch):
    monkeypatch.setattr(IPython.display, "display", lambda *_: None)


def pipeline(
    shape=(2,), kind="ddim", conditioner="one_hot", condition_dim=0, num_train_timesteps=12
):
    return ResearchDiffusionPipeline(
        PointDenoiser(
            sample_shape=shape,
            hidden_dim=16,
            num_layers=1,
            num_classes=3,
            conditioner=conditioner,
            condition_dim=condition_dim,
        ),
        make_scheduler(kind, num_train_timesteps=num_train_timesteps),
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_field_controls_query_actual_model_coordinates_and_preserve_model_state(dimension):
    pipe = pipeline((dimension,))
    pipe.model.train()
    points = torch.randn(12, dimension)
    data = SampleDataset(points, torch.arange(12) % 3)
    explorer = notebook_field_explorer(pipe, data, resolution=4, bounds=[(-2, 3)] * dimension)
    assert explorer.state["error"] is None
    assert explorer.state["grid"].shape == (4,) * dimension
    assert explorer.state["vectors"].shape == (4,) * dimension + (dimension,)
    assert explorer.controls["timestep"].max == 11
    assert explorer.controls["timestep"].step == 1
    assert not explorer.controls["resolution"].continuous_update
    explorer.controls["condition"].value = 2
    explorer.controls["timestep"].value = 7
    explorer.controls["guidance_scale"].value = 2.0
    assert explorer.state["condition"] == 1
    assert explorer.state["timestep"] == 7
    grid = explorer.state["grid"]
    expected = pipe.vector_field(grid.points, 7, conditions=1, kind="score", guidance_scale=2.0)
    np.testing.assert_allclose(explorer.state["vectors"].reshape(-1, dimension), expected.numpy())
    axis = explorer.state["figure"].axes[0]
    assert axis.get_xlabel() == "x"
    if dimension > 1:
        assert axis.get_ylabel() == "y"
    if dimension == 3:
        assert axis.get_zlabel() == "z"
    assert pipe.model.training
    explorer.close()


def test_field_cfg_can_be_disabled_and_unconditional_is_distinct_from_class_zero(monkeypatch):
    pipe = pipeline(num_train_timesteps=1000)
    calls = []
    original = pipe.vector_field

    def record(*args, **kwargs):
        calls.append((kwargs["conditions"], kwargs["guidance_scale"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipe, "vector_field", record)
    before = torch.random.get_rng_state().clone()
    explorer = notebook_field_explorer(pipe, resolution=4, timestep=349, guidance_scale=2.3)
    controls = explorer.controls
    assert controls["condition"].options[0] == ("Unconditional", 0)
    assert calls[-1] == (None, 1.0)
    assert explorer.state["condition"] is None
    assert not controls["cfg_enabled"].disabled and controls["guidance_scale"].disabled
    assert controls["cfg_enabled"].value
    assert controls["guidance_scale"].value == 2.3
    controls["cfg_enabled"].value = False
    assert calls[-1] == (None, 1.0)
    assert not controls["cfg_enabled"].value
    controls["cfg_enabled"].value = True
    assert (controls["timestep"].min, controls["timestep"].max, controls["timestep"].step) == (
        0,
        999,
        1,
    )
    for timestep in (0, 1, 348, 349, 350, 998, 999):
        controls["timestep"].value = timestep
        assert explorer.state["error"] is None
        assert explorer.state["timestep"] == timestep

    controls["condition"].value = 1
    assert calls[-1] == (0, 2.3)
    assert not controls["cfg_enabled"].disabled and not controls["guidance_scale"].disabled
    controls["cfg_enabled"].value = False
    assert calls[-1] == (0, 1.0)
    assert explorer.state["guidance_scale"] == 1.0
    assert not explorer.state["cfg_enabled"]
    assert controls["guidance_scale"].disabled
    assert controls["guidance_scale"].value == 2.3
    assert "CFG=off (scale=1)" in explorer.state["figure"].axes[0].get_title()
    expected = original(explorer.state["grid"].points, 999, conditions=0, guidance_scale=1.0)
    np.testing.assert_allclose(explorer.state["vectors"].reshape(-1, 2), expected.numpy())
    controls["condition"].value = 0
    assert calls[-1] == (None, 1.0)
    assert not controls["cfg_enabled"].disabled and controls["guidance_scale"].disabled
    assert not controls["cfg_enabled"].value
    controls["condition"].value = 1
    assert calls[-1] == (0, 1.0)
    assert not controls["cfg_enabled"].disabled and controls["guidance_scale"].disabled
    controls["cfg_enabled"].value = True
    assert calls[-1] == (0, 2.3)
    assert not controls["guidance_scale"].disabled
    assert pipe.model.training
    assert torch.equal(before, torch.random.get_rng_state())
    explorer.close()


def test_sampling_cfg_toggle_uses_plain_conditioned_prediction_and_preserves_selected_scale():
    class RecordingPipeline(ResearchDiffusionPipeline):
        def __call__(self, *args, **kwargs):
            self.last_call = (kwargs["conditions"], kwargs["guidance_scale"])
            return super().__call__(*args, **kwargs)

    base = pipeline()
    pipe = RecordingPipeline(base.model, base.scheduler)
    before = torch.random.get_rng_state().clone()
    explorer = notebook_sampling_explorer(pipe, num_samples=3, steps=2, guidance_scale=2.3)
    controls = explorer.controls
    assert controls["condition"].options[0] == ("Unconditional", 0)
    assert not controls["cfg_enabled"].disabled and controls["guidance_scale"].disabled
    assert controls["cfg_enabled"].value and controls["guidance_scale"].value == 2.3
    controls["cfg_enabled"].value = False
    assert explorer.generate() is not None
    assert pipe.last_call == (None, 1.0)
    assert explorer.state["condition"] is None
    controls["cfg_enabled"].value = True
    controls["condition"].value = 1
    assert not controls["cfg_enabled"].disabled and not controls["guidance_scale"].disabled
    controls["cfg_enabled"].value = False
    assert explorer.generate() is not None
    assert pipe.last_call == (0, 1.0)
    assert explorer.state["guidance_scale"] == 1.0
    assert not explorer.state["cfg_enabled"]
    assert "CFG=off (scale=1)" in explorer.children[-2].value
    assert "CFG=2.3" not in explorer.children[-2].value
    assert controls["guidance_scale"].disabled and controls["guidance_scale"].value == 2.3
    controls["condition"].value = 0
    assert not controls["cfg_enabled"].disabled and not controls["cfg_enabled"].value
    controls["condition"].value = 1
    assert not controls["cfg_enabled"].disabled and controls["guidance_scale"].disabled
    controls["cfg_enabled"].value = True
    assert not controls["guidance_scale"].disabled
    assert explorer.generate() is not None
    assert pipe.last_call == (0, 2.3)
    assert explorer.state["cfg_enabled"] and explorer.state["guidance_scale"] == 2.3
    assert pipe.model.training
    assert torch.equal(before, torch.random.get_rng_state())
    explorer.close()


def test_flow_field_exposes_sigma_velocity_and_continuous_condition_options():
    pipe = pipeline(kind="flow", conditioner="mlp", condition_dim=2)
    explorer = notebook_field_explorer(
        pipe,
        condition_options={"Angle zero": [0.0, 1.0]},
        resolution=4,
        timestep=12,
    )
    assert explorer.controls["kind"].options == ("velocity", "denoising")
    assert "σ=1.000" in explorer.state["figure"].axes[0].get_title()
    explorer.controls["condition"].value = 1
    torch.testing.assert_close(explorer.state["condition"], torch.tensor([0.0, 1.0]))
    assert explorer.state["error"] is None
    explorer.close()


def test_continuous_conditions_use_actual_representative_dataset_rows():
    pipe = pipeline(conditioner="mlp", condition_dim=2)
    data = SampleDataset(
        torch.zeros(3, 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        labels=torch.tensor([4, 9, 4]),
    )
    explorer = notebook_field_explorer(pipe, data, resolution=4)
    assert [name for name, _ in explorer.controls["condition"].options] == [
        "Unconditional",
        "Label 4",
        "Label 9",
    ]
    explorer.controls["condition"].value = 2
    torch.testing.assert_close(explorer.state["condition"], data.conditions[1])
    explorer.close()


def test_widget_validation_and_render_failure_are_clear(monkeypatch):
    with pytest.raises(ValueError, match="point model"):
        notebook_field_explorer(pipeline((4, 2)), resolution=4)
    with pytest.raises(ValueError, match="shape"):
        notebook_field_explorer(pipeline(), points=np.zeros((3, 3)), resolution=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        notebook_field_explorer(pipeline(), bounds=[(1, 0), (0, 1)], resolution=4)
    with pytest.raises(ValueError, match="shape"):
        notebook_field_explorer(
            pipeline(conditioner="mlp", condition_dim=2),
            condition_options={"bad": [1]},
            resolution=4,
        )
    pipe = pipeline()
    monkeypatch.setattr(
        pipe,
        "vector_field",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Model query failed")),
    )
    explorer = notebook_field_explorer(pipe, resolution=4)
    assert str(explorer.state["error"]) == "Model query failed"
    assert "Model query failed" in explorer.children[-2].value
    explorer.close()


def test_potential_uses_raw_field_when_arrow_normalization_changes(monkeypatch):
    pipe = pipeline()
    monkeypatch.setattr(
        pipe, "vector_field", lambda points, *args, **kwargs: -torch.as_tensor(points)
    )
    explorer = notebook_field_explorer(pipe, resolution=6)
    explorer.controls["potential"].value = True
    first = explorer.state["potential"]
    assert first.is_conservative
    assert first.relative_residual < 1e-8
    explorer.controls["normalize"].value = False
    np.testing.assert_allclose(explorer.state["potential"].potential, first.potential)
    assert "Potential:" in explorer.children[-2].value
    explorer.close()


def test_potential_log_control_changes_colors_and_retains_preference(monkeypatch):
    from matplotlib.colors import SymLogNorm

    pipe = pipeline()
    monkeypatch.setattr(
        pipe, "vector_field", lambda points, *args, **kwargs: -torch.as_tensor(points)
    )
    explorer = notebook_field_explorer(pipe, resolution=6)
    controls = explorer.controls
    assert controls["potential_log"].disabled
    controls["potential"].value = True
    assert not controls["potential_log"].disabled
    original = explorer.state["potential"].potential.copy()
    controls["potential_log"].value = True
    assert explorer.state["error"] is None
    assert explorer.state["potential_scale"] == "symlog"
    np.testing.assert_array_equal(explorer.state["potential"].potential, original)
    meshes = [
        collection for axis in explorer.state["figure"].axes for collection in axis.collections
    ]
    assert any(isinstance(collection.norm, SymLogNorm) for collection in meshes)
    assert "Potential colors: symlog" in explorer.children[-2].value
    controls["potential"].value = False
    assert controls["potential_log"].disabled and controls["potential_log"].value
    controls["potential"].value = True
    assert not controls["potential_log"].disabled and controls["potential_log"].value
    assert explorer.state["potential_scale"] == "symlog"
    controls["potential_log"].value = False
    assert explorer.state["potential_scale"] == "linear"
    np.testing.assert_array_equal(explorer.state["potential"].potential, original)
    explorer.close()


@pytest.mark.parametrize("kind", ["ddpm", "ddim", "flow"])
def test_sampling_is_button_triggered_deterministic_and_history_slider_only_redraws(kind):
    pipe = pipeline(kind=kind)
    pipe.model.train()
    explorer = notebook_sampling_explorer(pipe, num_samples=4, steps=3, seed=125)
    explorer.controls["guidance_scale"].value = 2.0
    explorer.controls["condition"].value = 2
    assert "history" not in explorer.state
    before = torch.random.get_rng_state().clone()
    first = explorer.generate()
    assert first is not None
    assert torch.equal(before, torch.random.get_rng_state())
    assert explorer.state["history"].shape == (4, 4, 2)
    assert explorer.controls["frame"].max == 3
    assert explorer.controls["frame"].value == 3
    torch.testing.assert_close(explorer.state["history"][-1], first.samples.cpu())
    history = explorer.state["history"]
    explorer.controls["frame"].value = 0
    assert explorer.state["history"] is history
    assert "next t=" in explorer.state["figure"].axes[0].get_title()
    second = explorer.generate()
    torch.testing.assert_close(first.samples, second.samples)
    assert pipe.model.training
    explorer.close()


def test_navigation_conditions_pin_endpoints_and_draw_encoded_walls():
    pipe = pipeline((4, 2), conditioner="mlp", condition_dim=8)
    condition = torch.tensor([-0.8, -0.4, 0.8, 0.4, -0.2, -0.3, 0.2, 0.3])
    data = SampleDataset(
        torch.zeros(2, 4, 2),
        condition.repeat(2, 1),
        metadata={"condition_layout": "start(D), goal(D), walls(W,2,D) flattened in C order"},
    )
    explorer = notebook_sampling_explorer(pipe, data, num_samples=3, steps=3)
    explorer.controls["condition"].value = 1
    result = explorer.generate()
    assert result is not None
    torch.testing.assert_close(result.samples[:, 0], condition[:2].expand(3, 2))
    torch.testing.assert_close(result.samples[:, -1], condition[2:4].expand(3, 2))
    assert len(explorer.state["figure"].axes[0].patches) == 1
    explorer.close()


def test_general_path_condition_is_not_decoded_as_navigation():
    pipe = pipeline((4, 2), conditioner="mlp", condition_dim=4)
    data = SampleDataset(torch.zeros(2, 4, 2), torch.ones(2, 4))
    explorer = notebook_sampling_explorer(pipe, data, num_samples=2, steps=2)
    explorer.controls["condition"].value = 1
    result = explorer.generate()
    assert result is not None
    assert len(explorer.state["walls"]) == 0
    assert not torch.equal(result.samples[:, 0], torch.ones(2, 2))
    explorer.close()
