"""Numerical contracts of the actual Euler / paired-reflow code in notebook 09.

Extract function definitions instead of importing the notebook: no installation
cell, dataset generation or long training run is executed during these tests.
"""

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diffusion_lab import PointDenoiser

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "09_rectified_flow_reflow.ipynb"


def _extract_functions(namespace):
    book = json.loads(NOTEBOOK.read_text())
    required = {"sample_euler", "train_reflow"}
    found = set()
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in required:
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, f"{NOTEBOOK.name}::{node.name}", "exec"), namespace)
                found.add(node.name)
    assert found == required, "Notebook helpers must remain available for regression checks"
    return namespace


@pytest.fixture
def notebook_functions():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(102)
        namespace = _extract_functions(
            {
                "torch": torch,
                "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
                "TRAIN_TIMESTEPS": 1000,
                "GRAD_CLIP": 1.0,
                "loss_fn": nn.MSELoss(),
            }
        )
        try:
            yield namespace
        finally:
            torch.set_num_threads(previous_threads)


class ConstantVelocity(nn.Module):
    """An exactly solvable, conditioned translation ODE with a separate null field."""

    sample_shape = (2,)

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "velocities",
            torch.tensor([[0.5, -0.25], [-1.0, 0.75], [0.125, -0.375], [-0.5, -0.2], [0.3, 0.45]]),
        )
        self.register_buffer("null_velocity", torch.tensor([0.4, -0.6]))
        self.calls = []

    @property
    def device(self):
        return self.velocities.device

    @property
    def dtype(self):
        return self.velocities.dtype

    def forward(self, state, timestep, conditions=None, condition_mask=None):
        null = self.null_velocity.expand(len(state), -1)
        velocity = null if conditions is None else self.velocities[conditions]
        if condition_mask is not None:
            velocity = torch.where(condition_mask[:, None], velocity, null)
        self.calls.append(
            {
                "labels": conditions.detach().clone() if conditions is not None else None,
                "mask": condition_mask.detach().clone() if condition_mask is not None else None,
                "training": self.training,
            }
        )
        return SimpleNamespace(sample=velocity)


@pytest.mark.parametrize("steps", [1, 4, 16])
def test_euler_noise_to_data_sign_endpoints_and_fixed_mask_history(notebook_functions, steps):
    model = ConstantVelocity().train()
    initial_noise = torch.tensor([[2.0, 1.0], [0.0, -1.0], [-2.0, 0.5], [1.25, 2.0], [-0.1, -0.8]])
    original_noise = initial_noise.clone()
    labels = torch.arange(len(initial_noise))
    keep = torch.tensor([True, False, True, False, True])
    velocity = torch.where(keep[:, None], model.velocities[labels], model.null_velocity)

    result = notebook_functions["sample_euler"](
        model, initial_noise, labels, keep, steps=steps, batch_size=3, record_history=True
    )

    # The exact ODE is x_sigma=z+(sigma-1)*v, so x_0=z-v at EVERY step budget.
    expected_history = initial_noise[None] + (result["sigmas"][:, None, None] - 1) * velocity[None]
    torch.testing.assert_close(result["samples"], initial_noise - velocity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["history"], expected_history, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["history"][0], original_noise, atol=0, rtol=0)
    torch.testing.assert_close(result["history"][-1], result["samples"], atol=0, rtol=0)
    torch.testing.assert_close(initial_noise, original_noise, atol=0, rtol=0)
    assert result["history"].shape == (steps + 1, len(initial_noise), 2)
    assert result["sigmas"][0] == 1 and result["sigmas"][-1] == 0
    assert torch.all(result["sigmas"].diff() < 0)
    assert model.training, "Sampler should restore the caller's model mode"
    for call in model.calls:
        torch.testing.assert_close(call["mask"], keep[call["labels"]], atol=0, rtol=0)
        assert not call["training"]
    # Batching may change, but every original condition must be used on every ODE step.
    visits = torch.bincount(
        torch.cat([call["labels"] for call in model.calls]), minlength=len(labels)
    )
    torch.testing.assert_close(visits, torch.full_like(visits, steps))


def test_euler_null_condition_is_the_actual_null_field(notebook_functions):
    model = ConstantVelocity().eval()
    initial_noise = torch.tensor([[0.1, 0.2], [1.0, -1.0], [-0.5, 0.75]])
    result = notebook_functions["sample_euler"](
        model, initial_noise, conditions=None, steps=4, batch_size=2, record_history=True
    )
    torch.testing.assert_close(
        result["samples"], initial_noise - model.null_velocity, atol=1e-6, rtol=1e-6
    )
    assert all(call["labels"] is None and call["mask"] is None for call in model.calls)
    assert not model.training


def test_paired_reflow_updates_student_and_preserves_teacher_coupling_after_shuffle(
    notebook_functions,
):
    n_pairs = 8
    teacher = (
        PointDenoiser(
            sample_shape=(2,),
            hidden_dim=16,
            num_layers=1,
            conditioner="embedding",
            num_classes=n_pairs,
        )
        .eval()
        .requires_grad_(False)
    )
    noise = torch.randn(n_pairs, 2, generator=torch.Generator().manual_seed(47))
    labels = torch.arange(n_pairs)  # Unique IDs let the test identify paired rows.
    keep = torch.tensor([True, False, True, True, False, True, False, True])
    endpoint = notebook_functions["sample_euler"](
        teacher, noise, labels, keep, steps=4, batch_size=4, record_history=False
    )["samples"]
    teacher_before = {
        name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()
    }
    noise_before, endpoint_before = noise.clone(), endpoint.clone()

    student = copy.deepcopy(teacher).requires_grad_(True)
    student_before = {
        name: tensor.detach().clone() for name, tensor in student.state_dict().items()
    }
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
    loader = DataLoader(
        TensorDataset(noise, endpoint, labels, keep),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(9),
    )
    observed_labels = []
    latest_labels = None

    def check_training_input(module, args, kwargs):
        nonlocal latest_labels
        bridge, timestep = args
        row_ids = kwargs["conditions"].detach()
        latest_labels = row_ids
        observed_labels.append(row_ids.cpu().clone())
        # Stored masks must follow the same row as BOTH original paired endpoints.
        torch.testing.assert_close(kwargs["condition_mask"], keep[row_ids], atol=0, rtol=0)
        sigma = timestep[:, None] / notebook_functions["TRAIN_TIMESTEPS"]
        expected_bridge = (1 - sigma) * endpoint[row_ids] + sigma * noise[row_ids]
        torch.testing.assert_close(bridge, expected_bridge, atol=1e-6, rtol=1e-6)

    class CoupledMSE(nn.Module):
        def forward(self, prediction, target):
            # This detects reversed targets or fresh noise replacing the paired noise.
            expected_velocity = noise[latest_labels] - endpoint[latest_labels]
            torch.testing.assert_close(target, expected_velocity, atol=0, rtol=0)
            return torch.nn.functional.mse_loss(prediction, target)

    hook = student.register_forward_pre_hook(check_training_input, with_kwargs=True)
    notebook_functions["loss_fn"] = CoupledMSE()
    try:
        history = notebook_functions["train_reflow"](
            student, loader, optimizer, epochs=3, generator=torch.Generator().manual_seed(10)
        )
    finally:
        hook.remove()

    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert [row["optimizer_steps"] for row in history] == [2, 4, 6]
    assert all(math.isfinite(row["loss"]) and row["loss"] >= 0 for row in history)
    assert any(
        not torch.equal(tensor, student_before[name])
        for name, tensor in student.state_dict().items()
    )
    assert all(torch.isfinite(parameter).all() for parameter in student.parameters())
    for name, tensor in teacher.state_dict().items():
        torch.testing.assert_close(tensor, teacher_before[name], atol=0, rtol=0)
    assert not teacher.training
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters()
    )
    torch.testing.assert_close(noise, noise_before, atol=0, rtol=0)
    torch.testing.assert_close(endpoint, endpoint_before, atol=0, rtol=0)
    first_epoch_ids = torch.cat(observed_labels[: len(loader)])
    assert not torch.equal(first_epoch_ids, labels), "Exercise actual shuffled paired rows"
    for epoch in range(3):
        ids = torch.cat(observed_labels[epoch * len(loader) : (epoch + 1) * len(loader)])
        torch.testing.assert_close(ids.sort().values, labels, atol=0, rtol=0)
