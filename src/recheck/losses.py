"""Full-state transport losses with an explicit, tested scaling contract.

One support point is one complete flattened state, never an individual pixel.
The float64 solver uses the KL-to-product-measure regularized objective. Its
converged plan is treated through the envelope theorem; gradients in a student
self-cost still flow through BOTH occurrences of the student features.
"""

import math
from typing import Any

import torch
from torch import Tensor


def pair_loss(student_endpoint: Tensor, teacher_endpoint: Tensor) -> Tensor:
    if student_endpoint.shape != teacher_endpoint.shape or student_endpoint.ndim < 2:
        raise ValueError("pair loss expects matching full-state [batch, ...] tensors")
    return (student_endpoint - teacher_endpoint.detach()).square().mean()


def features(anchor: Tensor, endpoint: Tensor, representation: str, gamma: float = 1.0) -> Tensor:
    """Normalize by sqrt(original state D), also for a joint vector of size 2D."""
    if anchor.shape != endpoint.shape or endpoint.ndim < 2:
        raise ValueError("anchor and endpoint must have matching [batch, ...] shapes")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    flat_anchor, flat_endpoint = anchor.flatten(1), endpoint.flatten(1)
    scale = math.sqrt(flat_endpoint.shape[1])
    if representation == "endpoint":
        return flat_endpoint / scale
    displacement = flat_endpoint - flat_anchor
    if representation == "displacement":
        return displacement / scale
    if representation == "joint":
        return torch.cat((math.sqrt(gamma) * flat_anchor, displacement), dim=1) / scale
    raise ValueError(f"Unknown representation: {representation!r}")


def _validate_points(x: Tensor, y: Tensor) -> None:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        raise ValueError("transport expects nonempty [support, features] tensors")
    if x.device != y.device:
        raise ValueError("transport supports must use the same device")
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise ValueError("transport features must use floating point")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite transport features")


def _cost(x: Tensor, y: Tensor) -> Tensor:
    # Direct differences avoid cancellation on identity/self diagonals.
    return (x[:, None, :] - y[None, :, :]).square().sum(-1)


def _regularized_transport(
    x: Tensor,
    y: Tensor,
    epsilon: float,
    iterations: int,
    tolerance: float,
    require_convergence: bool,
) -> tuple[Tensor, dict[str, Any]]:
    costs = _cost(x, y)
    n, m = costs.shape
    loga, logb = -math.log(n), -math.log(m)
    with torch.no_grad():
        kernel = -costs.detach() / epsilon
        logu = torch.zeros(n, dtype=costs.dtype, device=costs.device)
        logv = torch.zeros(m, dtype=costs.dtype, device=costs.device)
        newton_steps = 0
        max_dual_nullity = 0
        for iteration in range(1, iterations + 1):
            logu = loga - torch.logsumexp(kernel + logv[None, :], dim=1)
            logv = logb - torch.logsumexp(kernel + logu[:, None], dim=0)
            if iteration % 10 == 0 or iteration == iterations:
                logplan = kernel + logu[:, None] + logv[None, :]
                plan = logplan.exp()
                row_error = (plan.sum(1) - 1 / n).abs().max()
                column_error = (plan.sum(0) - 1 / m).abs().max()
                if max(float(row_error), float(column_error)) <= tolerance:
                    break
                # Pure alternating scaling can take millions of iterations for
                # almost disconnected small-epsilon supports. The dual has one
                # gauge per disconnected component, not necessarily one global
                # gauge. A rank-revealing symmetric pseudoinverse handles these
                # null modes; fixing only a single coordinate can instead yield
                # spurious Newton directions of size 1e14. The rank threshold
                # is float64 machine precision, never the marginal tolerance.
                # Costs/epsilon/required marginal accuracy remain unchanged.
                if iteration < iterations and n > 1 and n <= 512:
                    row_mass = plan.sum(1)
                    jacobian = torch.diag(row_mass) - m * (plan @ plan.T)
                    eigenvalues, eigenvectors = torch.linalg.eigh(jacobian)
                    rank_threshold = torch.finfo(costs.dtype).eps * n * eigenvalues[-1].abs()
                    active = eigenvalues > rank_threshold
                    max_dual_nullity = max(max_dual_nullity, n - int(active.sum()))
                    basis = eigenvectors[:, active]
                    error = row_mass - 1 / n
                    direction = basis @ ((basis.T @ error) / eigenvalues[active])
                    if torch.isfinite(direction).all():
                        previous_error = float(row_error)
                        for backtrack in range(30):
                            trial_u = logu.clone()
                            trial_u -= (0.5**backtrack) * direction
                            trial_v = logb - torch.logsumexp(kernel + trial_u[:, None], dim=0)
                            trial_plan = (kernel + trial_u[:, None] + trial_v[None, :]).exp()
                            trial_error = float((trial_plan.sum(1) - 1 / n).abs().max())
                            if trial_error < previous_error:
                                logu, logv = trial_u, trial_v
                                newton_steps += 1
                                break
        converged = max(float(row_error), float(column_error)) <= tolerance
        if not converged and require_convergence:
            raise RuntimeError(
                f"Sinkhorn did not converge in {iterations} iterations: "
                f"row={float(row_error):.3g}, column={float(column_error):.3g}, "
                f"tolerance={tolerance:g}. Increase iterations or investigate epsilon."
            )
        kl_term = (plan * (logplan - loga - logb)).sum()
        entropy = -(plan * logplan).sum()
    objective = (plan * costs).sum() + epsilon * kl_term
    if not torch.isfinite(objective):
        raise RuntimeError("nonfinite regularized OT objective")
    diagnostics = {
        "iterations": iteration,
        "newton_corrections": newton_steps,
        "max_numerical_dual_nullity": max_dual_nullity,
        "converged": converged,
        "row_marginal_error": float(row_error),
        "column_marginal_error": float(column_error),
        "diagonal_mass": float(plan.diagonal().sum()),
        "plan_entropy": float(entropy),
        "regularized_objective": float(objective.detach()),
        "transport_cost": float((plan * costs.detach()).sum()),
        "kl_to_product": float(kl_term),
    }
    return objective, diagnostics


def sinkhorn_divergence(
    student: Tensor,
    teacher: Tensor,
    epsilon: float,
    *,
    iterations: int = 2000,
    tolerance: float = 1e-10,
    return_diagnostics: bool = False,
    require_convergence: bool = True,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Canonical squared-cost Sinkhorn reference, computed in float64.

    Teacher features are always detached. The converged-plan envelope gradient
    is checked against finite differences and the independent GeomLoss solver.
    A failed solver raises by default; no clamp/nan replacement masks failure.
    """
    _validate_points(student, teacher)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if (
        not isinstance(iterations, int)
        or iterations < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("iterations and tolerance must be positive")
    # Metal does not implement float64. Copying student features to CPU is a
    # differentiable operation; it preserves the network gradient while the
    # reference solver keeps its registered numerical precision. Copy only the
    # final scalar back to Metal, with at least float32 loss precision.
    use_cpu_bridge = student.device.type == "mps"
    compute_device = torch.device("cpu") if use_cpu_bridge else student.device
    # Do not combine Metal transfer and dtype conversion in one ``to`` call:
    # torch 2.14.1 can silently return zeros for MPS32 -> CPU64. Separate copies
    # preserve both the actual feature values and their gradients.
    x = student.to(device=compute_device).to(dtype=torch.float64)
    y = teacher.detach().to(device=compute_device).to(dtype=torch.float64)
    cross, cross_diag = _regularized_transport(
        x, y, epsilon, iterations, tolerance, require_convergence
    )
    self_x, self_x_diag = _regularized_transport(
        x, x, epsilon, iterations, tolerance, require_convergence
    )
    self_y, self_y_diag = _regularized_transport(
        y, y, epsilon, iterations, tolerance, require_convergence
    )
    result = cross - 0.5 * self_x - 0.5 * self_y
    if use_cpu_bridge:
        output_dtype = (
            torch.float32 if student.dtype in (torch.float16, torch.bfloat16) else student.dtype
        )
        result = result.to(dtype=output_dtype).to(device=student.device)
    if return_diagnostics:
        return result, {
            "dtype": "float64",
            "compute_device": str(compute_device),
            "student_device": str(student.device),
            "output_dtype": str(result.dtype),
            "differentiable_cpu_bridge": use_cpu_bridge,
            "cost": "squared_euclidean",
            "epsilon": epsilon,
            "regularization": "KL(plan || uniform_product)",
            "cross": cross_diag,
            "student_self": self_x_diag,
            "teacher_self": self_y_diag,
        }
    return result


def geomloss_sinkhorn(
    student: Tensor,
    teacher: Tensor,
    epsilon: float,
    *,
    scaling: float = 0.99,
) -> Tensor:
    """Independent tensorized library comparison; float32 minimum under AMP.

    GeomLoss p=2 uses 0.5*||x-y||²: sqrt(2) scaling restores our full cost.
    blur=sqrt(epsilon), debias=True supplies both regularized self-terms.

    Legacy SamplesLoss stops after its annealing schedule rather than checking
    marginals. Even scaling=.99 can leave material gradient error at small
    epsilon. Use the converged ``sinkhorn_divergence`` for registered training;
    this independent implementation validates cost/epsilon/gradient scaling.
    """
    _validate_points(student, teacher)
    if not math.isfinite(epsilon) or epsilon <= 0 or not 0 < scaling < 1:
        raise ValueError("epsilon must be positive; scaling must lie in (0, 1)")
    from geomloss import SamplesLoss

    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    x, y = student.to(dtype), teacher.detach().to(dtype)
    solver = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=math.sqrt(epsilon),
        debias=True,
        backend="tensorized",
        scaling=scaling,
    )
    with torch.autocast(device_type=x.device.type, enabled=False):
        result = solver(math.sqrt(2) * x, math.sqrt(2) * y)
    if not torch.isfinite(result):
        raise RuntimeError("nonfinite GeomLoss Sinkhorn divergence")
    return result


def random_directions(
    dimensions: int,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Use an explicitly supplied SWD RNG, independent of training data RNGs."""
    if dimensions <= 0 or count <= 0:
        raise ValueError("dimensions and count must be positive")
    # Training RNG streams are intentionally CPU generators, including on GPU.
    # Draw on the generator's own device before transferring the fixed bank.
    directions = torch.randn(
        count, dimensions, generator=generator, device=generator.device, dtype=dtype
    ).to(device=device)
    return directions / directions.norm(dim=1, keepdim=True)


def sliced_wasserstein_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    directions: Tensor | None = None,
    generator: torch.Generator | None = None,
    num_directions: int = 128,
) -> Tensor:
    """P times mean squared sorted projections; no square root is taken."""
    _validate_points(student, teacher)
    if student.shape != teacher.shape:
        raise ValueError("sorted-projection SWD requires equal support and feature sizes")
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    x, y = student.to(dtype), teacher.detach().to(dtype)
    if directions is None:
        if generator is None:
            raise ValueError("supply an independent SWD generator or fixed direction bank")
        directions = random_directions(
            x.shape[1], num_directions, generator=generator, device=x.device, dtype=dtype
        )
    directions = directions.to(device=x.device, dtype=dtype)
    if directions.ndim != 2 or directions.shape[1] != x.shape[1] or not len(directions):
        raise ValueError("directions must have shape [num_directions, feature_dimension]")
    if not torch.allclose(
        directions.norm(dim=1), torch.ones(len(directions), device=x.device, dtype=dtype), atol=1e-5
    ):
        raise ValueError("SWD directions must be unit vectors")
    with torch.autocast(device_type=x.device.type, enabled=False):
        projected_x = (x @ directions.T).sort(dim=0).values
        projected_y = (y @ directions.T).sort(dim=0).values
        return x.shape[1] * (projected_x - projected_y).square().mean()


def exact_assignment_cost(student: Tensor, teacher: Tensor) -> Tensor:
    """Exact empirical W₂² for equal uniform supports via linear assignment."""
    _validate_points(student, teacher)
    if student.shape != teacher.shape:
        raise ValueError("assignment reference requires matching uniform support sizes")
    from scipy.optimize import linear_sum_assignment

    compute_device = "cpu" if student.device.type == "mps" else student.device
    costs = _cost(
        student.to(device=compute_device).to(dtype=torch.float64),
        teacher.detach().to(device=compute_device).to(dtype=torch.float64),
    )
    rows, columns = linear_sum_assignment(costs.detach().cpu().numpy())
    value = costs[rows, columns].mean()
    if student.device.type == "mps":
        value = value.to(dtype=torch.float32).to(device=student.device)
    return value


# The same converged solver is used for diagnostics and registered training.
# No silent switch to the faster, approximate legacy GeomLoss annealing loss.
production_sinkhorn = sinkhorn_divergence


def verify_loss_contracts() -> dict[str, Any]:
    """Return actual float64 numerical errors for a machine-readable report.

    This is executable verification, not a prefilled list of successful tests.
    Unit tests additionally exercise rejected inputs and failure conditions.
    """
    from .schedules import make_schedule, student_endpoint, teacher_target

    results: list[dict[str, Any]] = []

    def check(name: str, error: float, tolerance: float, **details: Any) -> None:
        results.append(
            {
                "name": name,
                "dtype": "float64",
                "error": float(error),
                "tolerance": tolerance,
                "passed": math.isfinite(error) and error <= tolerance,
                **details,
            }
        )

    generator = torch.Generator().manual_seed(987)
    x = (torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.4).requires_grad_()
    y = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.4
    directions = random_directions(
        3, 256, generator=torch.Generator().manual_seed(998), dtype=torch.float64
    )
    permutation = torch.tensor([4, 2, 0, 3, 1])
    epsilon = 0.3
    reference, diagnostics = sinkhorn_divergence(
        x, y, epsilon, tolerance=1e-13, return_diagnostics=True
    )
    reference_gradient = torch.autograd.grad(reference, x)[0]

    for label, groups, nt, ns in (
        ("32_to_16", None, 32, 16),
        ("32_to_8", None, 32, 8),
        ("24_to_16", [2, 1] * 8, 24, 16),
    ):
        schedule = make_schedule(nt, ns, groups)
        error = max(
            abs(float(schedule.widths.sum()) - 1),
            abs(sum(schedule.groups) - nt),
            abs(float(schedule.times[0]) - 1),
            abs(float(schedule.times[-1])),
        )
        check(
            "grid_coverage_" + label,
            error,
            1e-14,
            teacher_nfe=nt,
            student_nfe=ns,
            groups=list(schedule.groups),
        )

    class ConstantVelocity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64))

        def forward(self, state: Tensor, times: Tensor) -> Tensor:
            return self.value.expand_as(state)

    constant = ConstantVelocity()
    target = teacher_target(constant, x, 0.75, 0.25, 4)
    expected = x.detach() - 0.25 * constant.value.detach()
    check("signed_time_constant_field", float((target - expected).abs().max()), 1e-12)
    single_teacher = teacher_target(constant, x, 0.75, 0.25, 1)
    single_student = student_endpoint(constant, x, 0.75, 0.25)
    check(
        "teacher_copy_one_microstep",
        float((single_teacher - single_student.detach()).abs().max()),
        1e-12,
    )
    check("teacher_target_detached", float(target.requires_grad), 0)

    endpoint_pair = pair_loss(x, y)
    displacement_pair = pair_loss(x - 0.3, y - 0.3)
    check(
        "endpoint_displacement_value",
        float((endpoint_pair - displacement_pair).abs().detach()),
        1e-12,
    )
    endpoint_gradient = torch.autograd.grad(endpoint_pair, x)[0]
    displacement_gradient = torch.autograd.grad(displacement_pair, x)[0]
    check(
        "endpoint_displacement_gradient",
        float((endpoint_gradient - displacement_gradient).abs().max()),
        1e-12,
    )
    check("sinkhorn_identity", abs(float(sinkhorn_divergence(x, x, epsilon).detach())), 1e-10)
    check(
        "sinkhorn_permutation",
        abs(float(sinkhorn_divergence(x, x[permutation], epsilon).detach())),
        1e-10,
    )
    check(
        "swd_identity", float(sliced_wasserstein_loss(x, x, directions=directions).detach()), 1e-12
    )
    check(
        "swd_permutation",
        float(sliced_wasserstein_loss(x, x[permutation], directions=directions).detach()),
        1e-12,
    )
    check("paired_permutation_detected", float(pair_loss(x, x[permutation]).detach() <= 0.01), 0)

    one_cost = (x[:1] - y[:1]).square().sum()
    check(
        "sinkhorn_single_point",
        float((sinkhorn_divergence(x[:1], y[:1], epsilon) - one_cost).abs().detach()),
        1e-12,
    )
    finite_bank = 3 * ((x[:1] - y[:1]) @ directions.T).square().mean()
    check(
        "swd_single_point_finite_bank",
        float(
            (sliced_wasserstein_loss(x[:1], y[:1], directions=directions) - finite_bank)
            .abs()
            .detach()
        ),
        1e-12,
    )
    translation = torch.tensor([0.25, -0.1, 0.4], dtype=torch.float64)
    translated_cost = float(translation.square().sum())
    check(
        "exact_ot_translation",
        abs(float(exact_assignment_cost(x, x + translation).detach()) - translated_cost),
        1e-12,
    )
    check(
        "sinkhorn_translation",
        abs(float(sinkhorn_divergence(x, x + translation, epsilon).detach()) - translated_cost),
        1e-9,
    )
    check(
        "swd_translation_orthogonal_bank",
        abs(
            float(
                sliced_wasserstein_loss(
                    x, x + translation, directions=torch.eye(3, dtype=torch.float64)
                ).detach()
            )
            - translated_cost
        ),
        1e-12,
    )
    assignment = float(exact_assignment_cost(x, y).detach())
    paired = float((x - y).square().sum(1).mean().detach())
    check(
        "exact_ot_paired_upper_bound",
        max(0, assignment - paired),
        1e-12,
        assignment_cost=assignment,
        paired_ground_cost=paired,
    )

    step = 1e-5
    finite_gradient = torch.empty_like(x)
    for index in range(x.numel()):
        offset = torch.zeros_like(x)
        offset.flatten()[index] = step
        plus = sinkhorn_divergence(x.detach() + offset, y, epsilon, tolerance=1e-13)
        minus = sinkhorn_divergence(x.detach() - offset, y, epsilon, tolerance=1e-13)
        finite_gradient.flatten()[index] = (plus - minus) / (2 * step)
    check(
        "both_student_self_arguments_finite_difference",
        float((finite_gradient - reference_gradient).abs().max()),
        1e-7,
        finite_difference_step=step,
    )
    check(
        "student_gradient_finite_nonzero",
        float(not (torch.isfinite(reference_gradient).all() and reference_gradient.norm() > 0)),
        0,
        gradient_norm=float(reference_gradient.norm()),
    )
    teacher_with_grad = y.detach().requires_grad_()
    _, teacher_gradient = torch.autograd.grad(
        sinkhorn_divergence(x, teacher_with_grad, epsilon),
        (x, teacher_with_grad),
        allow_unused=True,
    )
    check("teacher_features_detached", float(teacher_gradient is not None), 0)
    max_marginal = max(
        diagnostics[name][key]
        for name in ("cross", "student_self", "teacher_self")
        for key in ("row_marginal_error", "column_marginal_error")
    )
    check("solver_marginals", max_marginal, 1e-12, components=diagnostics)
    doubled = sinkhorn_divergence(x, y, epsilon, iterations=4000, tolerance=1e-14)
    doubled_gradient = torch.autograd.grad(doubled, x)[0]
    check("solver_doubled_iterations_value", float((reference - doubled).abs().detach()), 1e-10)
    check(
        "solver_doubled_iterations_gradient",
        float((reference_gradient - doubled_gradient).abs().max()),
        1e-9,
    )

    # Preserve the disconnected-support regression in the numeric JSON report,
    # including marginal and gradient errors rather than only a pytest status.
    disconnected_rng = torch.Generator().manual_seed(1)
    disconnected_x = 0.2 * torch.randn(32, 2, generator=disconnected_rng, dtype=torch.float64)
    disconnected_shift = 0.001 * torch.randn(32, 2, generator=disconnected_rng, dtype=torch.float64)
    disconnected_x[-1] = torch.tensor([0.1, -1.8], dtype=torch.float64)
    disconnected_y = disconnected_x + disconnected_shift
    disconnected_x.requires_grad_()
    disconnected, disconnected_diag = sinkhorn_divergence(
        disconnected_x, disconnected_y, 0.02, tolerance=1e-12, return_diagnostics=True
    )
    disconnected_gradient = torch.autograd.grad(disconnected, disconnected_x)[0]
    disconnected_twice = sinkhorn_divergence(
        disconnected_x, disconnected_y, 0.02, iterations=4000, tolerance=1e-13
    )
    disconnected_twice_gradient = torch.autograd.grad(disconnected_twice, disconnected_x)[0]
    disconnected_marginal = max(
        disconnected_diag[name][key]
        for name in ("cross", "student_self", "teacher_self")
        for key in ("row_marginal_error", "column_marginal_error")
    )
    check(
        "disconnected_support_marginals",
        disconnected_marginal,
        1e-12,
        dual_nullity=disconnected_diag["cross"]["max_numerical_dual_nullity"],
    )
    check(
        "disconnected_support_doubled_value",
        float((disconnected - disconnected_twice).abs().detach()),
        1e-12,
    )
    check(
        "disconnected_support_doubled_gradient",
        float((disconnected_gradient - disconnected_twice_gradient).abs().max()),
        1e-10,
    )
    disconnected_direction = torch.randn(
        disconnected_x.shape, generator=torch.Generator().manual_seed(55), dtype=torch.float64
    )
    disconnected_direction /= disconnected_direction.norm()
    disconnected_plus = sinkhorn_divergence(
        disconnected_x.detach() + step * disconnected_direction,
        disconnected_y,
        0.02,
        tolerance=1e-13,
    )
    disconnected_minus = sinkhorn_divergence(
        disconnected_x.detach() - step * disconnected_direction,
        disconnected_y,
        0.02,
        tolerance=1e-13,
    )
    disconnected_finite_difference = (disconnected_plus - disconnected_minus) / (2 * step)
    check(
        "disconnected_support_directional_gradient",
        float(
            (
                (disconnected_gradient * disconnected_direction).sum()
                - disconnected_finite_difference
            ).abs()
        ),
        1e-8,
    )

    anchor = torch.tensor([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    endpoint = anchor + displacement
    reassigned = anchor + displacement[[1, 2, 0]]
    displacement_loss = sinkhorn_divergence(
        features(anchor, endpoint, "displacement"),
        features(anchor, reassigned, "displacement"),
        0.5,
    )
    joint_loss = sinkhorn_divergence(
        features(anchor, endpoint, "joint"), features(anchor, reassigned, "joint"), 0.5
    )
    check("displacement_permutation_blindness", abs(float(displacement_loss)), 1e-10)
    check(
        "joint_association_detected",
        float(joint_loss <= 0.1),
        0,
        joint_divergence=float(joint_loss),
    )

    library_diagnostics: list[dict[str, Any]] = []
    for scaling in (0.95, 0.99, 0.9999):
        library = geomloss_sinkhorn(x, y, epsilon, scaling=scaling)
        library_gradient = torch.autograd.grad(library, x)[0]
        entry = {
            "scaling": scaling,
            "value_error": float((library - reference).abs().detach()),
            "gradient_max_error": float((library_gradient - reference_gradient).abs().max()),
            "gradient_relative_l2_error": float(
                (library_gradient - reference_gradient).norm() / reference_gradient.norm()
            ),
        }
        library_diagnostics.append(entry)
    accurate = library_diagnostics[-1]
    check("geomloss_cost_epsilon_value_parity", accurate["value_error"], 2e-7, scaling=0.9999)
    check("geomloss_gradient_parity", accurate["gradient_max_error"], 3e-5, scaling=0.9999)
    global_rng = torch.random.get_rng_state().clone()
    sliced_wasserstein_loss(x, y, generator=torch.Generator().manual_seed(123))
    check(
        "swd_independent_rng", float(not torch.equal(global_rng, torch.random.get_rng_state())), 0
    )
    return {
        "passed": all(item["passed"] for item in results),
        "dtype": "float64",
        "production_solver": "converged_logdomain_sinkhorn_newton_envelope",
        "tests": results,
        "geomloss_annealing_diagnostics": library_diagnostics,
        "library_policy": "Independent comparison only; registered training checks marginals with converged reference solver",
    }
