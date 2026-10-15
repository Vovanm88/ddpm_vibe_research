"""Execute the notebook's actual targets and visible student loop, without network."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from torch import nn

from recheck.data import make_toy_dataset
from recheck.models import FMMLP
from recheck.runtime import model_hash

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/10_wasserstein_tube_rays.ipynb"


def cell_source(tag):
    notebook = json.loads(NOTEBOOK.read_text())
    return next(
        "".join(c["source"]) for c in notebook["cells"] if tag in c["metadata"].get("tags", [])
    )


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def notebook_namespace(output_dir):
    namespace = {}
    exec(cell_source("imports"), namespace)
    exec(cell_source("parameters"), namespace)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        teacher = FMMLP(hidden_dim=8, num_layers=2).eval().requires_grad_(False)
    namespace.update(
        {
            "teacher": teacher,
            "teacher_hash": model_hash(teacher),
            "device": torch.device("cpu"),
            "OUTPUT_DIR": output_dir,
            "train": make_toy_dataset("rotating_rays", 64, 41, n_classes=4),
            "validation": make_toy_dataset(
                "rotating_rays", 64, 41, split="validation", n_classes=4
            ),
            "UPDATES": 3,
            "BATCH_SIZE": 16,
            "CALIBRATION_SAMPLES": 16,
            "CHECKPOINT_EVERY": 1,
            "WARMUP": 1,
            "MICROSTEPS": 2,
            "H": 1 / 16,
            "EVAL_SAMPLES": 64,
            "MAKE_GIFS": False,
            "SHOW_WIDGETS": False,
        }
    )
    output_dir.mkdir(parents=True)
    exec(cell_source("transition"), namespace)
    exec(cell_source("calibration"), namespace)
    return namespace


def test_notebook_source_is_clean_small_compilable_and_training_is_visible():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert NOTEBOOK.stat().st_size < 100_000
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and not cell.outputs
        tree = ast.parse(cell.source, filename=f"rays-notebook:cell{index}")
        compile(tree, f"rays-notebook:cell{index}", "exec")
        assert not any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 20_000
            for node in ast.walk(tree)
        )
    training = cell_source("training")
    for visible_operation in (
        "teacher_microsteps(teacher, a_used",
        "student(a_used, t)",
        "loss.backward()",
        "optimizer.step()",
        "lerp_(",
    ):
        assert visible_operation in training
    assert "base64" not in cell_source("install")


def test_actual_notebook_target_has_correct_sign_and_retargets_tube():
    namespace = {}
    exec("import torch\nimport math", namespace)
    exec(cell_source("transition"), namespace)
    transition = namespace["teacher_microsteps"]

    class Constant(nn.Module):
        def forward(self, state, timestep):
            return SimpleNamespace(sample=torch.ones_like(state) * 2)

    anchor = torch.randn(12, 2, dtype=torch.float64)
    torch.testing.assert_close(
        transition(Constant(), anchor, 1.0, 0.25, 2), anchor - 0.5, atol=1e-12, rtol=1e-12
    )

    class Nonlinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.first_inputs = []

        def forward(self, state, timestep):
            self.first_inputs.append(state.clone())
            return SimpleNamespace(sample=state.square() + timestep)

    field = Nonlinear()
    perturbed = anchor + 0.1 * 0.0625 * torch.ones_like(anchor)
    clean_target = transition(field, anchor, 1.0, 0.0625, 2)
    perturbed_target = transition(field, perturbed, 1.0, 0.0625, 2)
    assert torch.equal(field.first_inputs[2], perturbed)
    assert not torch.equal(clean_target, perturbed_target)
    assert not perturbed_target.requires_grad
    # Joint always divides by sqrt(original D), despite its doubled feature width.
    joint = namespace["transition_features"](anchor, anchor + 1, "joint", 4.0)
    torch.testing.assert_close(joint[:, :2], 2 * anchor / 2**0.5)
    torch.testing.assert_close(joint[:, 2:], torch.ones_like(anchor) / 2**0.5)


@pytest.mark.parametrize("family,representation", [("sinkhorn", "endpoint"), ("swd", "joint")])
def test_actual_visible_training_updates_all_students_preserves_parent_and_rng(
    tmp_path, family, representation
):
    namespace = notebook_namespace(tmp_path / family)
    namespace.update(DISTRIBUTION_FAMILY=family, REPRESENTATION=representation)
    original_teacher = model_hash(namespace["teacher"])
    exec(cell_source("training"), namespace)
    assert len(namespace["trained"]) == 4
    assert all(model_hash(model) != original_teacher for model in namespace["trained"].values())
    assert model_hash(namespace["teacher"]) == original_teacher
    assert all(p.grad is None for p in namespace["teacher"].parameters())
    traces = list(namespace["pairing_traces"].values())
    assert all(trace == traces[0] for trace in traces) and len(traces[0]) == 3
    for name, model in namespace["trained"].items():
        state = torch.load(namespace["OUTPUT_DIR"] / name / "training.pt", weights_only=True)
        assert state["status"] == "completed" and state["updates_completed"] == 3
        assert len(state["history"]) == 3
        assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in state["history"])
        assert state["contract"]["teacher_sha256"] == original_teacher
        assert state["contract"]["representation"] == representation
    before = {name: model_hash(model) for name, model in namespace["trained"].items()}
    namespace["RESUME"] = True
    exec(cell_source("training"), namespace)
    assert before == {name: model_hash(model) for name, model in namespace["trained"].items()}


def test_notebook_lambda_zero_matches_mse_under_identical_rng(tmp_path):
    namespace = notebook_namespace(tmp_path / "lambda-zero")
    namespace["LAMBDA_W"] = 0.0
    exec(cell_source("training"), namespace)
    trained = namespace["trained"]
    assert model_hash(trained["M0"]) == model_hash(trained["E0"])
    assert model_hash(trained["M1"]) == model_hash(trained["E1"])
