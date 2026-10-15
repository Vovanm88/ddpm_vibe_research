"""Execute the actual visible wall notebook loops with continuous conditions."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from torch import nn

from diffusion_lab import PointDenoiser
from recheck.runtime import model_hash

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/14_wall_navigation_distillation.ipynb"


def source(tag):
    return next(c.source for c in nbformat.read(NOTEBOOK, 4).cells
                if tag in c.metadata.get("tags", []))


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_source_is_clean_compilable_and_training_calls_have_continuous_conditions():
    book = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(book)
    assert NOTEBOOK.stat().st_size < 100_000
    for index, cell in enumerate(book.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None
        tree = ast.parse(cell.source)
        compile(tree, f"wall-notebook:cell{index}", "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"teacher_model", "student", "model"}:
                    assert any(k.arg == "conditions" for k in node.keywords)
    for tag in ("teacher_training", "student_training"):
        assert "loss.backward()" in source(tag) and ".step()" in source(tag)
        assert "lerp_(" in source(tag)
        assert "train.labels" not in source(tag)
    assert 'calibration[str(scene_id)][j]["epsilon"]' in source("student_training")
    assert "train_scene_indices[scene_id]" in source("student_training")
    assert 'MIN_VERSION = "0.4.5"' in source("install")
    assert "base64" not in source("install")
    assert "diffusion_lab.transport" in source("imports")
    assert "iterations=OT_ITERATIONS_PER_STAGE" in source("student_training")
    assert "tolerance=OT_TOLERANCE" in source("student_training")
    assert "epsilon_scaling=OT_EPSILON_SCALING" in source("student_training")


class ContinuousSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, sample, timestep, conditions=None):
        assert conditions is not None
        self.calls.append((sample.clone(), float(timestep), conditions.clone()))
        return SimpleNamespace(sample=conditions[:, 2, None, None].expand_as(sample))


def test_sampler_preserves_condition_and_constructs_AB_at_every_actual_frame():
    namespace = {"torch": torch, "math": math, "PATH_LENGTH": 32,
                 "TRAIN_TIMESTEPS": 1000, "tensor_hash": lambda _: "fixed"}
    exec(source("transition_and_sampling"), namespace)
    sampler = namespace["sample_navigation"]
    noise = torch.randn(8, 30, 2)
    condition = torch.tensor([-.8, .1, .8, -.1, -.2, -.4, .2, .4]).repeat(8, 1)
    model = ContinuousSpy()
    output = sampler(model, noise, condition, steps=4)
    assert output["history"].shape == (5, 8, 32, 2)
    torch.testing.assert_close(output["history"][:, :, 0], condition[None, :, :2].expand(5, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(output["history"][:, :, -1], condition[None, :, 2:4].expand(5, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(output["history"][0, :, 1:-1], noise, rtol=0, atol=0)
    torch.testing.assert_close(output["samples"][:, 1:-1], noise - .8)
    assert [call[1] for call in model.calls] == [1000, 750, 500, 250]
    assert all(torch.equal(call[2], condition) for call in model.calls)
    assert output["endpoint_clamped_contract"] and output["nfe"] == 4
    changed = condition.clone()
    changed[:, 2] = .7
    assert not torch.equal(sampler(model, noise, changed, 4)["samples"], output["samples"])
    with pytest.raises(ValueError, match="no trained null"):
        sampler(model, noise, None, 4)


def test_microstep_target_keeps_the_whole_geometry_condition():
    namespace = {"torch": torch, "TRAIN_TIMESTEPS": 1000}
    exec(source("transition_and_sampling"), namespace)
    model = ContinuousSpy()
    anchor = torch.randn(8, 30, 2)
    condition = torch.tensor([-.8, 0., .8, 0., -.2, -.4, .2, .4]).repeat(8, 1)
    target = namespace["teacher_microsteps"](model, anchor, .5, .25, condition, 8)
    assert len(model.calls) == 8 and all(torch.equal(c[2], condition) for c in model.calls)
    torch.testing.assert_close(target, anchor - .25 * .8)
    assert not target.requires_grad


def test_interactive_scene_and_noise_selection_uses_only_matching_cached_geometry():
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    conditions = torch.tensor([
        [-.8, .1, .8, -.1, -.2, -.4, .2, .4],
        [-.7, -.2, .9, .2, -.1, -.5, .1, .5],
    ]).repeat_interleave(2, dim=0)
    generations = {
        "Teacher N2": {"history": torch.randn(3, 4, 32, 2)},
        "E0 N1": {"history": torch.randn(2, 4, 32, 2)},
    }
    original = {name: result["history"].clone() for name, result in generations.items()}
    calls = []

    def cached_plot(histories, **kwargs):
        calls.append((histories, kwargs))
        return plt.subplots()

    namespace = {"torch": torch, "widgets": widgets, "plt": plt,
                 "SHOW_WIDGETS": True, "EVAL_SCENES": 2, "SAMPLES_PER_SCENE": 2,
                 "generations": generations, "evaluation_conditions": conditions,
                 "plot_navigation_denoising": cached_plot, "display": lambda *_: None,
                 "clear_output": lambda **_: None}
    exec(source("interactive"), namespace)
    namespace["scene_selector"].value = 1
    namespace["noise_selector"].value = 1
    namespace["selector"].value = "E0 N1"
    namespace["frame"].value = 1
    histories, kwargs = calls[-1]
    torch.testing.assert_close(histories["E0 N1"], original["E0 N1"][:, 2:4])
    torch.testing.assert_close(kwargs["start"], conditions[2, :2])
    torch.testing.assert_close(kwargs["goal"], conditions[2, 2:4])
    torch.testing.assert_close(kwargs["walls"], conditions[2, 4:].reshape(1, 2, 2))
    assert kwargs["path_indices"] == [1]
    assert kwargs["progress"] == 1
    assert namespace["play"].step == namespace["frame"].step == 1
    assert namespace["frame"].max == namespace["play"].max == 1
    for name in original:
        torch.testing.assert_close(generations[name]["history"], original[name], rtol=0, atol=0)
    plt.close("all")


def test_actual_two_update_teacher_and_MSE_Sinkhorn_students_keep_scene_and_parent(tmp_path):
    namespace = {}
    exec(source("imports"), namespace)
    exec(source("parameters"), namespace)
    namespace.update(OUTPUT_DIR=tmp_path / "wall", TRAIN_SCENES=2, VALIDATION_SCENES=2,
        PATHS_PER_SCENE=4, HIDDEN_DIM=16, NUM_LAYERS=1, TRAIN_TEACHER=True,
        TEACHER_UPDATES=2, TEACHER_BATCH_SIZE=8, TEACHER_WARMUP=1, TEACHER_EMA=.9,
        UPDATES=2, BATCH_SIZE=8, CALIBRATION_SAMPLES=8, WARMUP=1, CHECKPOINT_EVERY=1,
        EVAL_SCENES=2, SAMPLES_PER_SCENE=4, SAVE_GIFS=False, SHOW_WIDGETS=False, PROGRESS=False)
    for tag in ("native_parent_guard", "setup", "teacher_training", "transition_and_sampling", "calibration"):
        exec(source(tag), namespace)
    teacher = namespace["teacher"]
    digest = model_hash(teacher)
    observed = []

    def capture(module, args, kwargs):
        observed.append(kwargs["conditions"].detach().cpu().clone())

    hook = teacher.register_forward_pre_hook(capture, with_kwargs=True)
    exec(source("student_training"), namespace)
    hook.remove()
    assert set(namespace["trained"]) == {"M0", "E0"}
    assert all(c.shape == (8, 8) and c.is_floating_point() for c in observed)
    assert all(torch.equal(c, c[:1].expand_as(c)) for c in observed)
    assert model_hash(teacher) == digest
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert namespace["pairing_traces"]["M0"] == namespace["pairing_traces"]["E0"]
    assert len(namespace["pairing_traces"]["M0"]) == 2
    for name, model in namespace["trained"].items():
        assert model_hash(model) != digest
        cp = torch.load(namespace["OUTPUT_DIR"] / name / "training.pt", weights_only=True)
        assert cp["updates_completed"] == 2 and cp["status"] == "completed"
        assert cp["contract"]["state_dimension"] == 60
        assert cp["contract"]["endpoint_clamped_contract"]
        assert not cp["contract"]["route_id_input"]
        assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in cp["history"])
    exec(source("evaluation"), namespace)
    exec(source("reload"), namespace)
    assert all(r["endpoint_mse"] == 0 for r in namespace["metrics"].values())
    before = {name: model_hash(m) for name, m in namespace["trained"].items()}
    namespace["RESUME"] = True
    exec(source("student_training"), namespace)
    assert before == {name: model_hash(m) for name, m in namespace["trained"].items()}


@pytest.mark.parametrize("failure", ["short", "endpoint_contract", "full_path_state", "wrong_conditioner"])
def test_native_guard_rejects_incompatible_navigation_parent(tmp_path, failure):
    namespace = {"Path": Path, "json": json, "PointDenoiser": PointDenoiser, "model_hash": model_hash}
    exec(source("parameters"), namespace)
    exec(source("native_parent_guard"), namespace)
    expected = namespace["expected_teacher_profile"]()
    manifest = {"status": "completed", "updates_completed": 20000,
                "config": expected, "endpoint_clamped_contract": True}
    native = {"_class_name": "PointDenoiser", **expected["model"]}
    if failure == "short":
        manifest["updates_completed"] = 19999
    elif failure == "endpoint_contract":
        manifest["endpoint_clamped_contract"] = False
    elif failure == "full_path_state":
        native["sample_shape"] = [32, 2]
    else:
        native["conditioner"] = "none"
    directory = tmp_path / "teacher"
    (directory / "ema_model").mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "ema_model/config.json").write_text(json.dumps(native))
    with pytest.raises(FileNotFoundError, match="matching continuous wall teacher"):
        namespace["load_navigation_parent"]([directory / "ema_model"], expected, "cpu")
