"""Exercise the notebook's actual DDIM/distillation helpers against an oracle."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler


@pytest.fixture
def notebook_helpers():
    """Extract real function ASTs without running installs or notebook training."""
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks/07_progressive_distillation.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    needed = {
        "batch_times",
        "coefficients",
        "sampling_grid",
        "stage_triplets",
        "prediction_to_clean",
        "ddim_from_clean",
        "ddim_step",
        "distillation_target",
        "inference",
    }
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        tree = ast.parse("".join(source) if isinstance(source, list) else source)
        definitions.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed
        )
    assert {node.name for node in definitions} == needed
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
        timestep_spacing="trailing",
        set_alpha_to_one=True,
    )
    scope = {
        "torch": torch,
        "np": np,
        "DDIMScheduler": DDIMScheduler,
        "DEVICE": "cpu",
        "base_scheduler": scheduler,
        "alpha_bars": scheduler.alphas_cumprod.clone(),
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(notebook_path), "exec"), scope)
    return SimpleNamespace(**scope)


class KnownCleanTeacher(torch.nn.Module):
    """Perfect v-prediction on a known forward trajectory, using Diffusers APIs.

    The expected clean samples/noise are fixed externally. In particular, no
    distillation inverse formula is copied into this oracle or the assertions.
    """

    def __init__(self, scheduler, clean, noise):
        super().__init__()
        self.scheduler = scheduler
        self.register_buffer("clean", clean)
        self.register_buffer("noise", noise)
        self.frozen_weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def forward(self, noisy, timestep, conditions=None, condition_mask=None):
        times = torch.as_tensor(timestep, dtype=torch.long).expand(len(noisy))
        expected = self.scheduler.add_noise(self.clean, self.noise, times)
        # A wrong intermediate timestep or DDIM transition breaks this check.
        torch.testing.assert_close(noisy, expected, atol=2e-5, rtol=2e-5)
        self.calls.append(
            {
                "times": times.clone(),
                "conditions": conditions,
                "mask": condition_mask,
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        velocity = self.scheduler.get_velocity(self.clean, self.noise, times)
        return SimpleNamespace(sample=velocity + self.frozen_weight)


@pytest.mark.parametrize("teacher_steps,student_steps", [(100, 50), (50, 25)])
def test_known_clean_targets_match_official_two_to_one_endpoints(
    notebook_helpers, teacher_steps, student_steps
):
    helpers = notebook_helpers
    t, s, r = helpers.stage_triplets(teacher_steps, student_steps)
    scheduler = helpers.base_scheduler
    generator = torch.Generator().manual_seed(918)
    conditions = torch.arange(student_steps) % 4
    keep = torch.arange(student_steps) % 3 != 0
    class_clean = torch.tensor(
        [[-0.8, 0.4], [0.3, -0.7], [0.75, -0.2], [-0.15, 0.85]], dtype=torch.float64
    )
    null_clean = torch.tensor([0.05, -0.1], dtype=torch.float64)
    clean = torch.where(keep[:, None], class_clean[conditions], null_clean)
    noise = torch.randn(student_steps, 2, generator=generator, dtype=clean.dtype)
    noisy = scheduler.add_noise(clean, noise, t).requires_grad_()
    teacher = KnownCleanTeacher(scheduler, clean, noise).train()
    before = {name: value.clone() for name, value in teacher.state_dict().items()}

    target_v, target_clean, teacher_endpoint = helpers.distillation_target(
        teacher, noisy, t, s, r, conditions, keep
    )

    torch.testing.assert_close(target_clean, clean, atol=2e-5, rtol=2e-5)
    expected_v = scheduler.get_velocity(clean, noise, t)
    torch.testing.assert_close(target_v, expected_v, atol=2e-5, rtol=2e-5)
    assert len(teacher.calls) == 2
    for call, expected_times in zip(teacher.calls, (t, s), strict=True):
        torch.testing.assert_close(call["times"], expected_times)
        assert call["conditions"] is conditions and call["mask"] is keep
        assert not call["grad_enabled"] and not call["training"]
    assert all(not value.requires_grad for value in (target_v, target_clean, teacher_endpoint))
    assert noisy.grad is None
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters()
    )
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    # Independent reference: use the official scheduler for both teacher steps
    # and the single student step, rather than reimplementing the DDIM formula.
    official_teacher = DDIMScheduler.from_config(scheduler.config)
    official_teacher.set_timesteps(teacher_steps)
    official_student = DDIMScheduler.from_config(scheduler.config)
    official_student.set_timesteps(student_steps)
    for index in range(student_steps):
        z_t = noisy[index : index + 1].detach()
        clean_row, noise_row = clean[index : index + 1], noise[index : index + 1]
        v_t = scheduler.get_velocity(clean_row, noise_row, t[index : index + 1])
        z_s = official_teacher.step(v_t, int(t[index]), z_t, eta=0).prev_sample
        v_s = scheduler.get_velocity(clean_row, noise_row, s[index : index + 1])
        expected_endpoint = official_teacher.step(v_s, int(s[index]), z_s, eta=0).prev_sample
        student_endpoint = official_student.step(
            target_v[index : index + 1], int(t[index]), z_t, eta=0
        ).prev_sample
        torch.testing.assert_close(
            teacher_endpoint[index : index + 1], expected_endpoint, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(student_endpoint, expected_endpoint, atol=2e-5, rtol=2e-5)
    assert int(r[-1]) == -1
    torch.testing.assert_close(teacher_endpoint[-1], clean[-1], atol=2e-5, rtol=2e-5)
    terminal_alpha, terminal_sigma = helpers.coefficients(-1, noisy[:1])
    torch.testing.assert_close(terminal_alpha, torch.ones_like(terminal_alpha), rtol=0, atol=0)
    torch.testing.assert_close(terminal_sigma, torch.zeros_like(terminal_sigma), rtol=0, atol=0)


def test_nondivisor_ddim_grid_is_rejected_before_distillation(notebook_helpers):
    with pytest.raises(AssertionError, match="divisors"):
        notebook_helpers.sampling_grid(64)
