"""Execute the visible notebook target and training loop, without copying them."""

import ast
import hashlib
import json
import time
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from diffusion_lab.environments import make_environment
from diffusion_lab.models import DenoiserOutput, PointDenoiser
from diffusion_lab.schedulers import make_scheduler, prediction_to_x0

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "08_model_and_guidance_distillation.ipynb"
)


def _cell_defining(function_name):
    """Locate source by its actual AST definition rather than a cell index."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        module = ast.parse(source)
        if any(
            isinstance(node, ast.FunctionDef) and node.name == function_name for node in module.body
        ):
            return module
    raise AssertionError(f"Notebook must visibly define {function_name}")


def _define_notebook_function(function_name, namespace):
    module = _cell_defining(function_name)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    # Retain decorators and defaults, including the actual torch.no_grad().
    definition = ast.Module(body=[function], type_ignores=[])
    exec(compile(definition, str(NOTEBOOK), "exec"), namespace)
    return namespace[function_name]


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)
    torch.random.set_rng_state(previous_rng)


class OracleTeacher(nn.Module):
    """Known predictions: unconditional=(2,-3), class delta=(4,1)*(c+1)."""

    def __init__(self):
        super().__init__()
        # A trainable anchor makes a missing no_grad decorator observable.
        self.anchor = nn.Parameter(torch.tensor([2.0, -3.0]))
        self.calls = []

    def forward(self, noisy, timesteps, conditions=None):
        self.calls.append((conditions, torch.is_grad_enabled(), self.training))
        result = self.anchor.expand(len(noisy), -1)
        if conditions is not None:
            result = result + (conditions + 1)[:, None] * torch.tensor([4.0, 1.0])
        return DenoiserOutput(sample=result)


@pytest.mark.parametrize(
    "guidance, expected",
    [
        (0.0, [[2.0, -3.0], [2.0, -3.0], [2.0, -3.0], [2.0, -3.0]]),
        (1.0, [[6.0, -2.0], [2.0, -3.0], [14.0, 0.0], [2.0, -3.0]]),
        (1.5, [[8.0, -1.5], [2.0, -3.0], [20.0, 1.5], [2.0, -3.0]]),
    ],
)
def test_visible_teacher_target_guidance_and_null_mask_are_exact_and_no_grad(guidance, expected):
    teacher = OracleTeacher().train()
    namespace = {"torch": torch, "teacher": teacher, "GUIDANCE": 1.5}
    target_function = _define_notebook_function("teacher_target", namespace)
    conditions = torch.tensor([0, 1, 2, 0])
    target = target_function(
        torch.zeros(4, 2, requires_grad=True),
        torch.tensor([0, 1, 500, 999]),
        conditions,
        torch.tensor([True, False, True, False]),
        guidance=guidance,
    )
    torch.testing.assert_close(target, torch.tensor(expected), rtol=0, atol=0)
    assert not target.requires_grad and target.grad_fn is None
    assert teacher.anchor.grad is None
    assert len(teacher.calls) == 2
    assert teacher.calls[0][0] is None
    assert torch.equal(teacher.calls[1][0], conditions)
    assert all(not grad_enabled and not training for _, grad_enabled, training in teacher.calls)


def test_visible_training_cell_updates_small_student_and_keeps_frozen_parent_bitwise():
    torch.manual_seed(719)
    teacher = (
        PointDenoiser(
            sample_shape=(2,),
            hidden_dim=16,
            num_layers=1,
            num_classes=4,
            conditioner="embedding",
        )
        .eval()
        .requires_grad_(False)
    )
    student = PointDenoiser(
        sample_shape=teacher.sample_shape,
        hidden_dim=8,
        num_layers=1,
        num_classes=4,
        conditioner="embedding",
    )
    teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
    student_before = {name: value.clone() for name, value in student.state_dict().items()}
    data = make_environment("rotating_rays", n_samples=16, n_classes=4, seed=720)
    loader = DataLoader(
        data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(721)
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    namespace = {
        "torch": torch,
        "nn": nn,
        "time": time,
        "hashlib": hashlib,
        "teacher": teacher,
        "student": student,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "loader": loader,
        "noise_generator": torch.Generator().manual_seed(722),
        "device": torch.device("cpu"),
        "train_scheduler": make_scheduler(
            "ddim", num_train_timesteps=12, prediction_type="v_prediction"
        ),
        "prediction_to_x0": prediction_to_x0,
        "T": 12,
        "EPOCHS": 2,
        "GUIDANCE": 1.5,
        "NULL_PROBABILITY": 0.15,
        "X0_LOSS_WEIGHT": 0.1,
    }
    state_hash = _define_notebook_function("model_state_sha256", namespace)
    namespace["teacher_state_before"] = state_hash(teacher)
    # Execute the complete real cell, including target, sampling, losses,
    # backward, optimizer/LR steps and its frozen-teacher integrity check.
    exec(compile(_cell_defining("teacher_target"), str(NOTEBOOK), "exec"), namespace)

    assert any(
        not torch.equal(value, student_before[name]) for name, value in student.state_dict().items()
    )
    assert all(
        torch.equal(value, teacher_before[name]) for name, value in teacher.state_dict().items()
    )
    assert namespace["teacher_state_before"] == namespace["teacher_state_after"]
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert len(namespace["history"]) == 2
    assert [row["epoch"] for row in namespace["history"]] == [1, 2]
    for row in namespace["history"]:
        assert all(
            torch.isfinite(torch.tensor(row[key])) for key in ("loss", "v_mse", "x0_mse", "lr")
        )
        assert row["loss"] >= 0 and 0 <= row["null_fraction"] <= 1
    assert optimizer.state[student.input_projection.weight]["step"] == 4
    assert namespace["training_seconds"] > 0
