"""Execute NB13's actual visible Torch loops and conditioning paths on CPU."""

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
NOTEBOOK = ROOT / "notebooks/13_conditional_rays_distillation.ipynb"


def source(tag):
    notebook = nbformat.read(NOTEBOOK, 4)
    return next(c.source for c in notebook.cells if tag in c.metadata.get("tags", []))


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_namespace(tmp_path):
    namespace = {}
    exec(source("imports"), namespace)
    exec(source("parameters"), namespace)
    namespace.update(
        OUTPUT_DIR=tmp_path,
        N_TRAIN=128,
        N_VALIDATION=64,
        HIDDEN_DIM=16,
        NUM_LAYERS=1,
        TRAIN_TEACHER=True,
        TEACHER_UPDATES=2,
        TEACHER_BATCH_SIZE=16,
        TEACHER_WARMUP=1,
        TEACHER_EMA=.9,
        UPDATES=2,
        BATCH_SIZE=16,
        CALIBRATION_SAMPLES=16,
        WARMUP=1,
        CHECKPOINT_EVERY=1,
        SAMPLES_PER_TARGET=4,
        SAVE_GIFS=False,
        SHOW_WIDGETS=False,
        PROGRESS=False,
    )
    for tag in ("native_parent_guard", "setup", "teacher_training", "transition_and_sampling", "calibration"):
        exec(source(tag), namespace)
    return namespace


def test_clean_notebook_keeps_model_condition_and_all_training_operations_visible():
    book = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(book)
    assert NOTEBOOK.stat().st_size < 100_000
    for index, cell in enumerate(book.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None
        tree = ast.parse(cell.source)
        compile(tree, f"conditional-notebook:cell{index}", "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"teacher_model", "student", "model"}:
                    assert any(keyword.arg == "conditions" for keyword in node.keywords)
            assert not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and len(node.value) > 20_000)
    for tag in ("teacher_training", "student_training"):
        text = source(tag)
        assert "loss.backward()" in text
        assert ".step()" in text and "lerp_(" in text
    assert "conditions=labels, condition_mask=keep" in source("teacher_training")
    assert "teacher_microsteps(teacher, a_used, t, H, labels, MICROSTEPS)" in source("student_training")
    assert "class_indices[condition_id]" in source("student_training")
    assert "calibration[calibration_key][j]" in source("student_training")
    assert 'Checkbox(value=False, description="Use CFG")' in source("interactive")
    assert 'MIN_VERSION = "0.4.3"' in source("install")
    assert "base64" not in source("install")


class ConditionSpy(nn.Module):
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
        self.calls.append((sample.clone(), float(timestep),
                           None if conditions is None else conditions.clone()))
        value = -torch.ones(len(sample)) if conditions is None else conditions.float() + 1
        return SimpleNamespace(sample=value[:, None].expand_as(sample))


def test_sampling_keeps_requested_condition_null_cfg_off_and_actual_nfe():
    namespace = {"torch": torch, "math": math, "TRAIN_TIMESTEPS": 1000, "N_CLASSES": 16,
                 "tensor_hash": lambda _: "fixed"}
    exec(source("transition_and_sampling"), namespace)
    sampler = namespace["sample_conditioned"]
    noise = torch.randn(8, 2)
    model = ConditionSpy()
    first = sampler(model, noise, 0, steps=2)
    assert len(model.calls) == first["nfe"] == 2
    assert all(call[2].eq(0).all() for call in model.calls)
    assert [call[1] for call in model.calls] == [1000, 500]
    torch.testing.assert_close(first["samples"], noise - 1)
    model.calls.clear()
    second = sampler(model, noise, 1, steps=2)
    torch.testing.assert_close(second["samples"], noise - 2)
    assert not torch.equal(first["samples"], second["samples"])
    model.calls.clear()
    guided = sampler(model, noise, 0, steps=2, guidance_scale=2.)
    assert len(model.calls) == guided["nfe"] == 4 and guided["steps"] == 2
    assert [call[2] is None for call in model.calls] == [False, True, False, True]
    torch.testing.assert_close(guided["samples"], noise - 3)
    model.calls.clear()
    null = sampler(model, noise, None, steps=2, guidance_scale=5.)
    assert len(model.calls) == null["nfe"] == 2
    assert all(call[2] is None for call in model.calls)
    torch.testing.assert_close(null["samples"], noise + 1)
    with pytest.raises(ValueError, match="integer"):
        sampler(model, noise, torch.zeros(8), steps=2)


def test_tube_retarget_preserves_the_same_condition_at_every_teacher_microstep():
    namespace = {"torch": torch, "math": math, "TRAIN_TIMESTEPS": 1000}
    exec(source("transition_and_sampling"), namespace)
    model = ConditionSpy()
    anchor = torch.randn(8, 2)
    perturbed = anchor + .1 * .0625 * torch.randn_like(anchor)
    labels = torch.full((8,), 7)
    target = namespace["teacher_microsteps"](model, perturbed, .5, .0625, labels, 2)
    assert torch.equal(model.calls[0][0], perturbed)
    assert all(torch.equal(call[2], labels) for call in model.calls)
    assert [call[1] for call in model.calls] == [500, 468.75]
    torch.testing.assert_close(target, perturbed - .0625 * 8)
    assert not target.requires_grad


def test_actual_two_update_teacher_and_four_students_route_labels_and_reload(tmp_path):
    namespace = tiny_namespace(tmp_path / "actual")
    teacher = namespace["teacher"]
    parent_hash = model_hash(teacher)
    observed = []

    def capture(module, args, kwargs):
        labels = kwargs["conditions"]
        observed.append(None if labels is None else labels.detach().cpu().clone())

    hook = teacher.register_forward_pre_hook(capture, with_kwargs=True)
    exec(source("student_training"), namespace)
    hook.remove()
    assert set(namespace["trained"]) == {"M0", "M1", "E0", "E1"}
    assert any(labels is None for labels in observed)
    assert any(labels is not None for labels in observed)
    assert all(labels is None or labels.unique().numel() == 1 for labels in observed)
    assert all(model_hash(model) != parent_hash for model in namespace["trained"].values())
    assert model_hash(teacher) == parent_hash
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    traces = list(namespace["pairing_traces"].values())
    assert len(traces[0]) == 2 and all(trace == traces[0] for trace in traces)
    assert len(namespace["calibration"]) == 17
    for name, history in namespace["training_logs"].items():
        assert len(history) == 2 and history[0]["condition"] is None
        assert history[1]["condition"] is not None
        state = torch.load(namespace["OUTPUT_DIR"] / name / "training.pt", weights_only=True)
        assert state["status"] == "completed" and state["updates_completed"] == 2
        assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in history)
    exec(source("evaluation"), namespace)
    exec(source("reload"), namespace)
    # These are different requested-label model predictions for identical noise,
    # not nearest-arm labels assigned after sampling.
    original = namespace["generations"]["Teacher N32"]
    assert not torch.equal(original[0]["samples"], original[1]["samples"])
    assert torch.equal(original[0]["history"][0], original[1]["history"][0])
    before = {name: model_hash(model) for name, model in namespace["trained"].items()}
    namespace["RESUME"] = True
    exec(source("student_training"), namespace)
    assert before == {name: model_hash(model) for name, model in namespace["trained"].items()}


@pytest.mark.parametrize("failure", ["short", "unconditional", "wrong_classes", "weights"])
def test_native_guard_rejects_missing_condition_or_parent_provenance(tmp_path, failure):
    namespace = {"torch": torch, "Path": Path, "json": json,
                 "PointDenoiser": PointDenoiser, "model_hash": model_hash}
    exec(source("parameters"), namespace)
    exec(source("native_parent_guard"), namespace)
    profile = namespace["expected_teacher_profile"]()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        teacher = PointDenoiser(**profile["model"])
    directory = tmp_path / "teacher"
    teacher.save_pretrained(directory / "ema_model", safe_serialization=True)
    manifest = {"status": "completed", "updates_completed": 20000, "config": profile,
                "teacher_sha256": model_hash(teacher), "data_sha256": "a" * 64}
    native = json.loads((directory / "ema_model/config.json").read_text())
    if failure == "short":
        manifest["updates_completed"] = 19999
    elif failure == "unconditional":
        native["conditioner"] = "none"
    elif failure == "wrong_classes":
        native["num_classes"] = 4
    else:
        manifest["teacher_sha256"] = "b" * 64
    (directory / "ema_model/config.json").write_text(json.dumps(native))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="No unconditional teacher is substituted"):
        namespace["load_conditional_parent"]([directory / "ema_model"], profile, "cpu")
