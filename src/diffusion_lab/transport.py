"""Strict full-KL Sinkhorn with epsilon-continuation of physical dual potentials.

This explicit backend solves the SAME final squared-cost, uniform-product KL
problem as the frozen recheck reference. Intermediate epsilons are warm starts,
not a replacement final epsilon, truncated kernel, softened tolerance, or a
fallback loss. Every stage, including the exact requested final epsilon, uses
the requested marginal tolerance. Failure raises by default.

Epsilon scaling follows the standard stabilization idea discussed by Schmitzer,
https://arxiv.org/abs/1610.06519 . No sparse kernel truncation is used here.
The converged-plan envelope supplies first-order derivatives; student self-cost
retains BOTH occurrences of the student, and the teacher is detached.
"""

from __future__ import annotations

import math

import torch

BACKEND = "epsilon_continuation_log_newton_v1"


def _validate(student, teacher, epsilon, iterations, tolerance, epsilon_scaling):
    if (
        student.ndim != 2
        or teacher.ndim != 2
        or student.shape[1] != teacher.shape[1]
        or not len(student)
        or not len(teacher)
    ):
        raise ValueError("transport expects nonempty [support,features] tensors")
    if (
        student.device != teacher.device
        or not student.is_floating_point()
        or not teacher.is_floating_point()
    ):
        raise ValueError("transport supports must be floating point on the same device")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("nonfinite transport features")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer per continuation stage")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(epsilon_scaling) or not 0 < epsilon_scaling < 1:
        raise ValueError("epsilon_scaling must lie strictly between zero and one")


def _epsilon_schedule(costs, epsilon, scaling):
    current = max(float(costs.max()), float(epsilon))
    levels = [current]
    while current * scaling > epsilon:
        current *= scaling
        levels.append(current)
        if len(levels) >= 128:
            raise RuntimeError(
                "epsilon continuation requires more than 128 stages; choose a faster scaling factor"
            )
    if levels[-1] != epsilon:
        levels.append(float(epsilon))
    return levels


@torch.no_grad()
def _solve_plan(costs, epsilon, iterations, tolerance, scaling, require_convergence):
    n, m = costs.shape
    if not torch.isfinite(costs).all():
        raise ValueError("nonfinite squared transport costs")
    levels = _epsilon_schedule(costs, epsilon, scaling)
    loga, logb = -math.log(n), -math.log(m)
    u, v = costs.new_zeros(n), costs.new_zeros(m)
    previous = levels[0]
    stages = []
    for index, stage_epsilon in enumerate(levels):
        # Preserve physical potentials f=epsilon*logu and g=epsilon*logv.
        # A zero/gauge-incompatible restart at the final tiny epsilon loses
        # the cross-component mass transfer that the connected stages establish.
        u *= previous / stage_epsilon
        v *= previous / stage_epsilon
        previous = stage_epsilon
        shift = u.mean()
        u -= shift
        v += shift
        kernel = -costs / stage_epsilon
        if not torch.isfinite(kernel).all():
            raise FloatingPointError("cost/epsilon exceeds finite float64 range")
        corrections, max_nullity = 0, 0
        for iteration in range(1, iterations + 1):
            u = loga - torch.logsumexp(kernel + v[None, :], dim=1)
            v = logb - torch.logsumexp(kernel + u[:, None], dim=0)
            if iteration % 10 != 0 and iteration != iterations:
                continue
            logplan = kernel + u[:, None] + v[None, :]
            plan = logplan.exp()
            row_mass = plan.sum(1)
            row_error = float((row_mass - 1 / n).abs().max())
            column_error = float((plan.sum(0) - 1 / m).abs().max())
            if max(row_error, column_error) <= tolerance:
                break
            if iteration < iterations and 1 < n <= 512:
                # Reduced dual Hessian after exact column normalization.
                # Rank-revealing solve removes machine-precision gauge modes;
                # continuation prevents required mass-transfer directions from
                # disappearing before their physical dual offsets are found.
                jacobian = torch.diag(row_mass) - m * (plan @ plan.T)
                eigenvalues, eigenvectors = torch.linalg.eigh(jacobian)
                threshold = torch.finfo(costs.dtype).eps * n * eigenvalues[-1].abs()
                active = eigenvalues > threshold
                max_nullity = max(max_nullity, n - int(active.sum()))
                basis = eigenvectors[:, active]
                direction = basis @ ((basis.T @ (row_mass - 1 / n)) / eigenvalues[active])
                if torch.isfinite(direction).all():
                    for backtrack in range(30):
                        trial_u = u - (0.5**backtrack) * direction
                        trial_v = logb - torch.logsumexp(kernel + trial_u[:, None], dim=0)
                        trial = (kernel + trial_u[:, None] + trial_v[None, :]).exp()
                        if float((trial.sum(1) - 1 / n).abs().max()) < row_error:
                            u, v = trial_u, trial_v
                            corrections += 1
                            break
        converged = max(row_error, column_error) <= tolerance
        stages.append(
            {
                "epsilon": stage_epsilon,
                "iterations": iteration,
                "converged": converged,
                "row_marginal_error": row_error,
                "column_marginal_error": column_error,
                "newton_corrections": corrections,
                "max_numerical_dual_nullity": max_nullity,
            }
        )
        if not converged and require_convergence:
            raise RuntimeError(
                f"{BACKEND} did not converge at stage {index + 1}/{len(levels)} "
                f"epsilon={stage_epsilon:.17g} after {iterations} iterations: "
                f"row={row_error:.3g}, column={column_error:.3g}, tolerance={tolerance:g}; "
                "no fallback loss or marginal relaxation was applied"
            )
    return plan, logplan, stages


def _transport(x, y, epsilon, iterations, tolerance, scaling, require_convergence):
    costs = (x[:, None, :] - y[None, :, :]).square().sum(-1)
    plan, logplan, stages = _solve_plan(
        costs.detach(), epsilon, iterations, tolerance, scaling, require_convergence
    )
    with torch.no_grad():
        kl = (plan * (logplan + math.log(len(x)) + math.log(len(y)))).sum()
        entropy = -(plan * logplan).sum()
    objective = (plan * costs).sum() + epsilon * kl
    if not torch.isfinite(objective):
        raise FloatingPointError("nonfinite full-KL regularized transport objective")
    final = stages[-1]
    return objective, {
        "iterations": sum(stage["iterations"] for stage in stages),
        "final_iterations": final["iterations"],
        "converged": final["converged"],
        "all_stages_converged": all(stage["converged"] for stage in stages),
        "row_marginal_error": final["row_marginal_error"],
        "column_marginal_error": final["column_marginal_error"],
        "epsilon_schedule": [stage["epsilon"] for stage in stages],
        "stages": stages,
        "newton_corrections": sum(stage["newton_corrections"] for stage in stages),
        "max_numerical_dual_nullity": max(stage["max_numerical_dual_nullity"] for stage in stages),
        "diagonal_mass": float(plan.diagonal().sum()),
        "plan_entropy": float(entropy),
        "regularized_objective": float(objective.detach()),
        "transport_cost": float((plan * costs.detach()).sum()),
        "kl_to_product": float(kl),
    }


def sinkhorn_divergence(
    student,
    teacher,
    epsilon,
    *,
    iterations=2000,
    tolerance=1e-10,
    return_diagnostics=False,
    require_convergence=True,
    epsilon_scaling=0.5,
):
    """Debiased squared-cost OT at EXACT epsilon, with strict continuation.

    iterations is the budget PER continuation stage, not a cap on total work.
    Every stage uses the caller's tolerance. Returned value is OT(x,y) minus
    half each full-KL self-objective; no detached/clamped bias correction is
    substituted. The teacher is detached. On MPS, separate differentiable CPU
    transfer and float64 cast avoid the PyTorch2.14 combined-copy zero bug.
    require_convergence=False is an explicit diagnostic opt-out, never used as
    an automatic fallback. Registered training must keep its default True.
    """
    _validate(student, teacher, epsilon, iterations, tolerance, epsilon_scaling)
    bridge = student.device.type == "mps"
    compute_device = torch.device("cpu") if bridge else student.device
    x = student.to(device=compute_device).to(dtype=torch.float64)
    y = teacher.detach().to(device=compute_device).to(dtype=torch.float64)
    cross, cross_diag = _transport(
        x, y, epsilon, iterations, tolerance, epsilon_scaling, require_convergence
    )
    self_x, self_x_diag = _transport(
        x, x, epsilon, iterations, tolerance, epsilon_scaling, require_convergence
    )
    self_y, self_y_diag = _transport(
        y, y, epsilon, iterations, tolerance, epsilon_scaling, require_convergence
    )
    result = cross - 0.5 * self_x - 0.5 * self_y
    if bridge:
        dtype = torch.float32 if student.dtype in (torch.float16, torch.bfloat16) else student.dtype
        result = result.to(dtype=dtype).to(device=student.device)
    if not return_diagnostics:
        return result
    return result, {
        "backend": BACKEND,
        "dtype": "float64",
        "compute_device": str(compute_device),
        "student_device": str(student.device),
        "output_dtype": str(result.dtype),
        "differentiable_cpu_bridge": bridge,
        "cost": "squared_euclidean",
        "epsilon": float(epsilon),
        "tolerance": float(tolerance),
        "epsilon_scaling": float(epsilon_scaling),
        "iterations_per_stage": iterations,
        "regularization": "KL(plan || uniform_product)",
        "cross": cross_diag,
        "student_self": self_x_diag,
        "teacher_self": self_y_diag,
    }
