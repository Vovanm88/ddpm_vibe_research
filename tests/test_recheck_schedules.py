from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from recheck.losses import pair_loss
from recheck.schedules import call_velocity, make_schedule, student_endpoint, teacher_target


class ConstantVelocity(nn.Module):
    def __init__(self, value=(0.25, -0.75), diffusers_output=False):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64))
        self.diffusers_output = diffusers_output
        self.calls = 0
        self.times = []

    def forward(self, x, t):
        self.calls += 1
        self.times.append(t.detach().clone())
        value = self.value.expand_as(x)
        return SimpleNamespace(sample=value) if self.diffusers_output else value


@pytest.mark.parametrize(
    "teacher,student,groups", [(32, 16, None), (32, 8, None), (24, 16, [2, 1] * 8)]
)
def test_schedule_covers_exact_teacher_grid(teacher, student, groups):
    schedule = make_schedule(teacher, student, groups)
    assert schedule.times.dtype == torch.float64
    assert schedule.times.shape == (student + 1,)
    assert schedule.times[0] == 1 and schedule.times[-1] == 0
    assert (schedule.times[:-1] > schedule.times[1:]).all()
    torch.testing.assert_close(schedule.widths.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        schedule.widths * teacher, torch.tensor(schedule.groups, dtype=torch.float64)
    )
    assert sum(schedule.groups) == teacher
    assert schedule.to_dict()["student_steps"] == student


@pytest.mark.parametrize(
    "arguments",
    [
        dict(teacher_steps=24, student_steps=16),
        dict(teacher_steps=32, student_steps=16, groups=[2] * 15),
        dict(teacher_steps=4, student_steps=2, groups=[1, 1]),
        dict(teacher_steps=4, student_steps=2, groups=[0, 4]),
        dict(teacher_steps=4.0),
        dict(student_steps=True),
    ],
)
def test_schedule_rejects_silent_nfe_changes(arguments):
    with pytest.raises(ValueError):
        make_schedule(**arguments)


@pytest.mark.parametrize("output", [False, True])
def test_signed_euler_and_teacher_actual_nfe(output):
    model = ConstantVelocity(diffusers_output=output)
    anchor = torch.tensor([[1.0, 2.0], [-2.0, 0.5]], dtype=torch.float64)
    endpoint = teacher_target(model, anchor, 0.75, 0.25, 4)
    torch.testing.assert_close(endpoint, anchor - 0.25 * model.value)
    assert not endpoint.requires_grad
    assert model.calls == 4
    torch.testing.assert_close(
        torch.stack(model.times)[:, 0],
        torch.tensor([0.75, 0.6875, 0.625, 0.5625], dtype=torch.float64),
    )
    differentiable = student_endpoint(model, anchor, 0.75, 0.25)
    differentiable.square().mean().backward()
    assert model.value.grad is not None and torch.isfinite(model.value.grad).all()


def test_teacher_copy_one_microstep_and_detachment():
    teacher = ConstantVelocity().requires_grad_(False).eval()
    student = deepcopy(teacher).requires_grad_(True)
    anchor = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(call_velocity(teacher, anchor, 1), call_velocity(student, anchor, 1))
    target = teacher_target(teacher, anchor, 1, 1 / 16, 1)
    endpoint = student_endpoint(student, anchor, 1, 1 / 16)
    torch.testing.assert_close(pair_loss(endpoint, target), torch.tensor(0.0, dtype=torch.float64))
    assert not target.requires_grad
    assert teacher.value.grad is None


def test_tube_retargets_from_perturbed_anchor():
    class LinearVelocity(nn.Module):
        def forward(self, x, t):
            return x

    teacher = LinearVelocity()
    anchor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    noise = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
    width, rho = 1 / 16, 0.1
    shifted = anchor + rho * width * noise
    clean_target = teacher_target(teacher, anchor, 0.5, width, 2)
    tube_target = teacher_target(teacher, shifted, 0.5, width, 2)
    torch.testing.assert_close(
        tube_target - clean_target, (1 - width / 2) ** 2 * (shifted - anchor)
    )
    assert not torch.equal(clean_target, tube_target)
    torch.testing.assert_close(
        teacher_target(teacher, anchor + 0 * noise, 0.5, width, 2), clean_target
    )


@pytest.mark.parametrize("time,width", [(-0.1, 0.1), (1.1, 0.1), (0.1, 0.2), (0.5, 0), (0.5, -0.2)])
def test_invalid_transition_is_rejected(time, width):
    with pytest.raises(ValueError):
        teacher_target(ConstantVelocity(), torch.zeros(2, 2, dtype=torch.float64), time, width, 2)
