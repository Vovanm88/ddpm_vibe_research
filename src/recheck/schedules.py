"""Signed Euler transitions on an explicitly grouped teacher grid.

Time is normalized: data live at 0, noise at 1, and generation subtracts
positive step widths. A model consumes a batch of normalized times.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TimeSchedule:
    times: Tensor
    widths: Tensor
    groups: tuple[int, ...]
    teacher_steps: int

    @property
    def student_steps(self) -> int:
        return len(self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "widths": self.widths.tolist(),
            "groups": list(self.groups),
            "teacher_steps": self.teacher_steps,
            "student_steps": self.student_steps,
            "time_convention": "data=0, noise=1; x_next=x-width*v(x,t)",
            "solver": "euler",
        }


def make_schedule(
    teacher_steps: int = 32,
    student_steps: int = 16,
    groups: Sequence[int] | None = None,
) -> TimeSchedule:
    """Cover exactly the teacher grid, without rounding a noninteger ratio.

    The optional 24 -> 16 protocol must explicitly supply ``[2, 1] * 8``.
    """
    if not isinstance(teacher_steps, Integral) or isinstance(teacher_steps, bool):
        raise ValueError("teacher_steps must be a positive integer")
    if not isinstance(student_steps, Integral) or isinstance(student_steps, bool):
        raise ValueError("student_steps must be a positive integer")
    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("step counts must be positive")
    if groups is None:
        if teacher_steps % student_steps:
            raise ValueError("Noninteger ratio requires explicit teacher-step groups")
        groups = (teacher_steps // student_steps,) * student_steps
    if len(groups) != student_steps or any(
        not isinstance(m, Integral) or isinstance(m, bool) or m <= 0 for m in groups
    ):
        raise ValueError("groups must contain one positive integer per student step")
    groups = tuple(int(m) for m in groups)
    if sum(groups) != teacher_steps:
        raise ValueError("groups must cover exactly teacher_steps teacher evaluations")
    cumulative = torch.tensor((0, *groups), dtype=torch.float64).cumsum(0)
    times = 1.0 - cumulative / teacher_steps
    widths = times[:-1] - times[1:]
    return TimeSchedule(times, widths, groups, int(teacher_steps))


def _batch_times(t: float | Tensor, x: Tensor) -> Tensor:
    times = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if times.numel() == 1:
        times = times.reshape(()).expand(x.shape[0])
    elif times.shape != (x.shape[0],):
        raise ValueError("time must be scalar or have shape [batch]")
    if not torch.isfinite(times).all() or (times < 0).any() or (times > 1).any():
        raise ValueError("normalized time must be finite and lie in [0, 1]")
    return times


def call_velocity(model: Any, x: Tensor, t: float | Tensor) -> Tensor:
    """Call a plain Torch or Diffusers-style model with normalized batch times."""
    output = model(x, _batch_times(t, x))
    velocity = output if isinstance(output, Tensor) else output.sample
    if velocity.shape != x.shape:
        raise ValueError("velocity and state shapes must agree")
    return velocity


def student_endpoint(
    model: Any, anchor: Tensor, t: float | Tensor, width: float | Tensor
) -> Tensor:
    """One differentiable Euler step, without division of the loss by H²."""
    width = torch.as_tensor(width, dtype=anchor.dtype, device=anchor.device)
    if width.numel() != 1 or not torch.isfinite(width) or width <= 0:
        raise ValueError("width must be a positive finite shared scalar")
    times = _batch_times(t, anchor)
    if (times + 8 * torch.finfo(anchor.dtype).eps < width).any():
        raise ValueError("the transition must not pass below data time 0")
    return anchor - width * call_velocity(model, anchor, times)


@torch.no_grad()
def teacher_target(
    model: Any,
    anchor: Tensor,
    t: float | Tensor,
    width: float | Tensor,
    substeps: int,
) -> Tensor:
    """Recompute the teacher endpoint from the actual (possibly tube) anchor.

    Does not mutate model train/eval mode: the caller freezes/evals its teacher.
    Each microstep consumes one model evaluation, including a group of size 1.
    """
    if not isinstance(substeps, Integral) or isinstance(substeps, bool) or substeps <= 0:
        raise ValueError("substeps must be a positive integer")
    width = torch.as_tensor(width, dtype=anchor.dtype, device=anchor.device)
    if width.numel() != 1 or not torch.isfinite(width) or width <= 0:
        raise ValueError("width must be a positive finite shared scalar")
    times = _batch_times(t, anchor)
    if (times + 8 * torch.finfo(anchor.dtype).eps < width).any():
        raise ValueError("the transition must not pass below data time 0")
    state = anchor.detach()
    micro_width = width / substeps
    for step in range(substeps):
        state = state - micro_width * call_velocity(model, state, times - step * micro_width)
    return state.detach()
