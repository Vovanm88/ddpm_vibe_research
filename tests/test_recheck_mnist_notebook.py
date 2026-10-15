"""MNIST notebook contracts execute synthetic 4x4 images only, with no data/GPU access."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from torch import nn

from recheck.config import resolve_config
from recheck.runtime import model_hash
from recheck.schedules import make_schedule

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/11_mnist_wasserstein_tube.ipynb"


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


class TinyImageVelocity(nn.Module):
    """Nonconstant image field lets two Euler steps differ from one without MNIST."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.tensor(0.3))
        self.sample_shape = (1, 4, 4)

    def forward(self, sample, timestep):
        time = torch.as_tensor(timestep, dtype=sample.dtype, device=sample.device)
        while time.ndim < sample.ndim:
            time = time.unsqueeze(-1)
        return SimpleNamespace(sample=self.scale * sample + self.bias * (7 * time).sin())


def tiny_namespace(tmp_path):
    namespace = {}
    exec(cell_source("imports"), namespace)
    exec(cell_source("parameters"), namespace)
    exec(cell_source("transition"), namespace)
    teacher = TinyImageVelocity().eval().requires_grad_(False)
    config = resolve_config(
        {
            "device": "cpu",
            "output_dir": str(tmp_path / "registered-ledger"),
            "data": {"name": "mnist"},
            "model": {"kind": "unet"},
        }
    )
    schedule = make_schedule(32, 16)
    generator = torch.Generator().manual_seed(17)
    train = torch.rand(32, 1, 4, 4, generator=generator) * 2 - 1
    calibration = {
        "sha256": "synthetic-test-calibration",
        "intervals": [
            {
                "representations": {
                    rep: {"epsilon": 0.1} for rep in ("endpoint", "displacement", "joint")
                }
            }
            for _ in range(16)
        ],
    }
    namespace.update(
        {
            "RUN_TRAINING": True,
            "teacher": teacher,
            "teacher_hash": model_hash(teacher),
            "config": config,
            "device": torch.device("cpu"),
            "schedule": schedule,
            "train_samples": train,
            "calibration": calibration,
            "EXPLORATORY_UPDATES": 2,
            "BATCH_SIZE": 8,
            "CHECKPOINT_EVERY": 1,
            "WARMUP": 1,
            "OUTPUT_DIR": tmp_path / "exploration",
            "require_rays_gate": lambda config: None,
        }
    )
    return namespace


def test_notebook_is_clean_compilable_and_training_defaults_off():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert NOTEBOOK.stat().st_size < 100_000
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and cell.outputs == []
        tree = ast.parse(cell.source, filename=f"mnist-notebook:cell{index}")
        compile(tree, f"mnist-notebook:cell{index}", "exec")
        assert not any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 20_000
            for node in ast.walk(tree)
        )
    namespace = {}
    exec(cell_source("parameters"), {"Path": Path}, namespace)
    assert namespace["RUN_TRAINING"] is False and namespace["ALLOW_CPU_EXPLORATION"] is False
    training = cell_source("training")
    assert "teacher_microsteps(teacher, a_used" in training
    assert "student(a_used, t).sample" in training
    assert "loss.backward()" in training and "optimizer.step()" in training
    assert "phi_student" in training and "production_sinkhorn" in training
    assert "mnist_mps_registered.yaml" in cell_source("configuration")
    assert "require_rays_gate(config)" in cell_source("teacher_setup")


def test_disabled_cells_do_not_require_data_teacher_device_or_budget():
    namespace = {"RUN_TRAINING": False}
    exec(cell_source("training_data"), namespace)
    exec(cell_source("training"), namespace)
    assert namespace["train_samples"] is None and namespace["calibration"] is None
    assert namespace["trained"] == {} and namespace["training_logs"] == {}


def test_actual_notebook_teacher_image_target_sign_no_grad_and_tube_retargeting():
    namespace = {"torch": torch, "math": __import__("math")}
    exec(cell_source("transition"), namespace)
    transition = namespace["teacher_microsteps"]

    class Constant(nn.Module):
        def forward(self, state, timestep):
            return SimpleNamespace(sample=torch.ones_like(state) * 3)

    anchor = torch.randn(3, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    target = transition(Constant(), anchor, 1.0, 0.125, 2)
    torch.testing.assert_close(target, anchor.detach() - 0.375, atol=1e-12, rtol=1e-12)
    assert target.shape == anchor.shape and not target.requires_grad
    field = TinyImageVelocity().double().requires_grad_(False)
    clean_target = transition(field, anchor, 1.0, 1 / 16, 2)
    perturbed = anchor.detach() + 0.1 / 16
    perturbed_target = transition(field, perturbed, 1.0, 1 / 16, 2)
    assert not torch.equal(perturbed_target, clean_target)
    assert not perturbed_target.requires_grad
    assert anchor.grad is None and all(p.grad is None for p in field.parameters())


def test_actual_full_image_features_use_one_image_per_support_and_sqrt_original_d():
    namespace = {"torch": torch, "math": __import__("math")}
    exec(cell_source("transition"), namespace)
    features = namespace["image_transition_features"]
    anchor = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    endpoint = anchor + 1
    phi_endpoint = features(anchor, endpoint, "endpoint")
    phi_joint = features(anchor, endpoint, "joint", 4.0)
    assert phi_endpoint.shape == (3, 784)
    assert phi_joint.shape == (3, 1568)
    torch.testing.assert_close(phi_endpoint, endpoint.flatten(1) / 28.0)
    torch.testing.assert_close(phi_joint[:, :784], 2 * anchor.flatten(1) / 28.0)
    torch.testing.assert_close(phi_joint[:, 784:], torch.ones(3, 784, dtype=torch.float64) / 28.0)


def test_actual_visible_image_training_updates_all_four_preserves_teacher_rng_and_resume(tmp_path):
    namespace = tiny_namespace(tmp_path)
    parent_hash = model_hash(namespace["teacher"])
    exec(cell_source("training"), namespace)
    assert set(namespace["trained"]) == {"M0", "M1", "E0", "E1"}
    assert model_hash(namespace["teacher"]) == parent_hash
    assert all(p.grad is None for p in namespace["teacher"].parameters())
    traces = list(namespace["pairing_traces"].values())
    assert all(trace == traces[0] for trace in traces) and len(traces[0]) == 2
    trained_hashes = {name: model_hash(model) for name, model in namespace["trained"].items()}
    assert all(digest != parent_hash for digest in trained_hashes.values())
    for name in namespace["trained"]:
        directory = namespace["OUTPUT_DIR"] / name
        checkpoint = torch.load(directory / "training.pt", weights_only=True)
        assert checkpoint["status"] == "completed" and checkpoint["updates_completed"] == 2
        assert checkpoint["contract"]["scope"] == "notebook_exploratory_one_paired_seed"
        assert checkpoint["contract"]["batch_size"] == 8
        assert checkpoint["contract"]["teacher_sha256"] == parent_hash
        assert (directory / "ema_state_dict.pt").is_file()
        assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in checkpoint["history"])
    namespace["RESUME"] = True
    exec(cell_source("training"), namespace)
    assert trained_hashes == {
        name: model_hash(model) for name, model in namespace["trained"].items()
    }


def test_actual_image_training_lambda_zero_is_identical_to_mse(tmp_path):
    namespace = tiny_namespace(tmp_path)
    namespace["LAMBDA_W"] = 0.0
    exec(cell_source("training"), namespace)
    models = namespace["trained"]
    assert model_hash(models["M0"]) == model_hash(models["E0"])
    assert model_hash(models["M1"]) == model_hash(models["E1"])
