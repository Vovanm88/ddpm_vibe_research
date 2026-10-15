"""Local/rollout, tube-retargeting, derivative, and gradient diagnostics.

All diagnostic randomness comes from an explicit independent CPU generator.
Velocity callbacks use physical t in [0,1] and return tensors or ``.sample``.
The module reports numerical evidence; reduced local error alone is not a
claim about robustness of long trajectories.
"""

import torch


def _as_like(value, device, dtype):
    if isinstance(value, torch.Tensor):
        if torch.device(device).type == "mps":
            return value.to(dtype=dtype).to(device=device)
        return value.to(device=device).to(dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _velocity(model, x, time):
    time = _as_like(time, x.device, x.dtype)
    output = model(x, time)
    value = output.sample if hasattr(output, "sample") else output
    if value.shape != x.shape or not torch.isfinite(value).all():
        raise ValueError("Velocity must be finite and have the same shape as the state")
    return value


def student_macro_step(student, anchor, time, length):
    if float(length) <= 0:
        raise ValueError("Macro-step length must be positive")
    return anchor - length * _velocity(student, anchor, time)


@torch.no_grad()
def teacher_microsteps(teacher, anchor, time, length, substeps=2):
    """Recompute the target from the exact supplied (possibly perturbed) anchor."""
    if not isinstance(substeps, int) or substeps < 1 or float(length) <= 0:
        raise ValueError("Positive length and integer teacher substeps are required")
    state = anchor.detach().clone()
    micro_length = length / substeps
    for index in range(substeps):
        state = state - micro_length * _velocity(teacher, state, time - index * micro_length)
    return state


def _grid(times, groups, device, dtype):
    times = _as_like(times, device, dtype)
    if (
        times.ndim != 1
        or len(times) < 2
        or not torch.isfinite(times).all()
        or not (times[:-1] > times[1:]).all()
    ):
        raise ValueError("Diagnostic times must be a finite strictly decreasing grid")
    if not torch.isclose(times[0], times.new_tensor(1)) or not torch.isclose(
        times[-1], times.new_tensor(0)
    ):
        raise ValueError("Diagnostic grid must cover noise t=1 through data t=0")
    if isinstance(groups, int):
        groups = [groups] * (len(times) - 1)
    groups = list(groups)
    if len(groups) != len(times) - 1 or any(not isinstance(g, int) or g < 1 for g in groups):
        raise ValueError(
            "Exactly one positive integer teacher group is needed per student interval"
        )
    return times, groups


@torch.no_grad()
def paired_rollouts(teacher, student, initial_noise, student_times, teacher_steps_per_student=2):
    """Teacher microsteps and independent student rollout on common nodes."""
    initial_noise = torch.as_tensor(initial_noise)
    times, groups = _grid(
        student_times, teacher_steps_per_student, initial_noise.device, initial_noise.dtype
    )
    teacher_state, student_state = initial_noise.clone(), initial_noise.clone()
    teacher_history, student_history = [teacher_state.clone()], [student_state.clone()]
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        teacher_state = teacher_microsteps(
            teacher, teacher_state, start, start - end, groups[index]
        )
        student_state = student_macro_step(student, student_state, start, start - end)
        teacher_history.append(teacher_state.clone())
        student_history.append(student_state.clone())
    teacher_history, student_history = torch.stack(teacher_history), torch.stack(student_history)
    return {
        "teacher_history": teacher_history,
        "student_history": student_history,
        "times": times,
        "rollout_mse_per_node": (teacher_history - student_history).square().flatten(1).mean(1),
        "teacher_nfe": sum(groups),
        "student_nfe": len(groups),
        "student_anchor_warning": "student_rollout states depend on the evaluated method",
    }


@torch.no_grad()
def local_rollout_diagnostics(
    teacher,
    student,
    real,
    initial_noise,
    student_times,
    teacher_steps_per_student=2,
    *,
    seed=3737,
    rhos=(0.0, 0.05, 0.10, 0.20),
    denominator_floor=1e-12,
):
    """Three anchor sources; every robustness target is freshly retargeted.

    One fixed eta bank is reused for all rho values and all evaluated methods
    sharing seed/data/noise/grid. Eta is coordinate-wise Gaussian, not normalized.
    """
    real, noise = torch.as_tensor(real), torch.as_tensor(initial_noise)
    if real.shape != noise.shape or real.device != noise.device or not torch.isfinite(real).all():
        raise ValueError(
            "Real examples and initial noise must be finite, paired, and have equal shapes"
        )
    if denominator_floor <= 0 or not rhos or min(rhos) < 0:
        raise ValueError("Positive denominator floor and nonnegative tube widths are required")
    times, groups = _grid(student_times, teacher_steps_per_student, noise.device, noise.dtype)
    rollout = paired_rollouts(teacher, student, noise, times, groups)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    eta = torch.randn((len(groups), *noise.shape), generator=generator, dtype=noise.dtype).to(
        noise.device
    )
    rows = []
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        length = start - end
        anchors = {
            "interpolant": (1 - start) * real + start * noise,
            "teacher_rollout": rollout["teacher_history"][index],
            "student_rollout": rollout["student_history"][index],
        }
        for source, clean_anchor in anchors.items():
            for rho in rhos:
                anchor = clean_anchor + float(rho) * length * eta[index]
                # Do not reuse a clean endpoint when the starting state changes.
                target = teacher_microsteps(teacher, anchor, start, length, groups[index])
                prediction = student_macro_step(student, anchor, start, length)
                mse = float((prediction - target).square().mean())
                energy = float((target - anchor).square().mean())
                rows.append(
                    {
                        "interval": index,
                        "t": float(start),
                        "H": float(length),
                        "source": source,
                        "rho_eval": float(rho),
                        "local_mse": mse,
                        "teacher_displacement_energy": energy,
                        "relative_local_mse": mse / max(energy, denominator_floor),
                        "denominator_floor": denominator_floor,
                    }
                )
    return {
        "local_error": rows,
        "rollout_error": [
            {"t": float(time), "rollout_mse": float(error)}
            for time, error in zip(times, rollout["rollout_mse_per_node"])
        ],
        "final_rollout_mse": float(rollout["rollout_mse_per_node"][-1]),
        "rhos": list(rhos),
        "seed": seed,
        "n_states": len(noise),
        "noise_definition": "independent coordinate Gaussian eta; delta=rho_eval*H",
        "teacher_targets_recomputed_from_perturbed_anchor": True,
        "student_rollout_states_are_method_dependent": True,
        "teacher_rollout_nfe": rollout["teacher_nfe"],
        "student_rollout_nfe": rollout["student_nfe"],
    }


@torch.no_grad()
def local_map_jacobian_diagnostics(
    teacher_map,
    student_map,
    anchors,
    *,
    zeta=1e-3,
    n_directions=4,
    seed=3838,
    directions=None,
    denominator_floor=1e-12,
):
    """Central Jv at zeta and zeta/2, using shared fixed unit directions.

    Callbacks map full local states to endpoints. The direction is unit length
    over one full image/state. This derivative convention is distinct from tube
    perturbations, whose Gaussian noise has unit variance per coordinate.
    """
    anchors = torch.as_tensor(anchors)
    if zeta <= 0 or n_directions < 1 or denominator_floor <= 0:
        raise ValueError("Finite difference scale, directions, and floor must be positive")
    if directions is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        directions = torch.randn(
            (n_directions, *anchors.shape[1:]), generator=generator, dtype=anchors.dtype
        )
        directions = directions.to(anchors.device)
        directions /= directions.flatten(1).norm(dim=1).reshape(-1, *([1] * (anchors.ndim - 1)))
    else:
        directions = _as_like(directions, anchors.device, anchors.dtype)
        if directions.shape[1:] != anchors.shape[1:]:
            raise ValueError("Directions must have shape [K,*state_shape]")
        norms = directions.flatten(1).norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
            raise ValueError("Finite difference directions must be unit length")
        n_directions = len(directions)
    derivatives = {}
    rows = []
    for scale in (zeta, zeta / 2):
        td, sd = [], []
        for index, direction in enumerate(directions):
            plus, minus = anchors + scale * direction, anchors - scale * direction
            jt = (teacher_map(plus) - teacher_map(minus)) / (2 * scale)
            js = (student_map(plus) - student_map(minus)) / (2 * scale)
            if not torch.isfinite(jt).all() or not torch.isfinite(js).all():
                raise ValueError("Nonfinite finite difference derivative")
            td.append(jt)
            sd.append(js)
            error, energy = float((js - jt).square().mean()), float(jt.square().mean())
            rows.append(
                {
                    "zeta": scale,
                    "direction": index,
                    "jacobian_action_mse": error,
                    "teacher_jacobian_action_energy": energy,
                    "relative_jacobian_action_mse": error / max(energy, denominator_floor),
                }
            )
        derivatives[scale] = torch.stack(td), torch.stack(sd)
    teacher_coarse, student_coarse = derivatives[zeta]
    teacher_fine, student_fine = derivatives[zeta / 2]
    return {
        "rows": rows,
        "seed": seed,
        "zeta": zeta,
        "n_directions": n_directions,
        "teacher_zeta_halving_mse": float((teacher_coarse - teacher_fine).square().mean()),
        "student_zeta_halving_mse": float((student_coarse - student_fine).square().mean()),
        "map_evaluations_per_model": 4 * n_directions,
        "n_states": len(anchors),
        "direction_norm": "unit L2 per complete state",
    }


def gradient_diagnostics(pair_loss, distribution_loss, parameters, *, lambda_w=1.0):
    """Non-mutating autograd probe; caller performs the sole optimizer update."""
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    pair = torch.autograd.grad(pair_loss, parameters, retain_graph=True, allow_unused=True)
    if distribution_loss.requires_grad:
        dist = torch.autograd.grad(
            distribution_loss, parameters, retain_graph=True, allow_unused=True
        )
    else:
        dist = (None,) * len(parameters)
    reference = pair_loss.detach()
    pair2 = sum(
        (value.detach().square().sum() for value in pair if value is not None),
        reference.new_zeros(()),
    )
    dist2 = sum(
        (value.detach().square().sum() for value in dist if value is not None),
        reference.new_zeros(()),
    )
    dot = sum(
        (
            a.detach().mul(b.detach()).sum()
            for a, b in zip(pair, dist)
            if a is not None and b is not None
        ),
        reference.new_zeros(()),
    )
    pnorm, dnorm = float(pair2.sqrt()), float(dist2.sqrt())
    return {
        "pair_grad_norm": pnorm,
        "dist_grad_norm": dnorm,
        "grad_cosine": float(dot) / (pnorm * dnorm) if pnorm > 0 and dnorm > 0 else None,
        "weighted_dist_to_pair_grad_ratio": abs(lambda_w) * dnorm / pnorm if pnorm > 0 else None,
        "distribution_requires_grad": distribution_loss.requires_grad,
    }


def diagnostics_numerical_report():
    """Persistable signed-step, retargeting, and derivative numerical checks."""
    generator = torch.Generator(device="cpu").manual_seed(9292)
    x = torch.randn((8, 2), generator=generator, dtype=torch.float64)

    def constant(state, time):
        return torch.ones_like(state) * 0.75

    def linear(state, time):
        return 0.5 * state

    length, rho = 0.25, 0.1
    eta = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    clean_target = teacher_microsteps(linear, x, 1, length, 2)
    zero_target = teacher_microsteps(linear, x + 0 * eta, 1, length, 2)
    perturbed_target = teacher_microsteps(linear, x + rho * length * eta, 1, length, 2)
    jacobian = local_map_jacobian_diagnostics(
        lambda a: a, lambda a: 2 * a, x, directions=torch.eye(2, dtype=x.dtype)
    )
    return {
        "dtype": "torch.float64",
        "signed_constant_step_max_abs_error": float(
            (student_macro_step(constant, x, 1, length) - (x - length * 0.75)).abs().max()
        ),
        "zero_tube_max_abs_error": float((clean_target - zero_target).abs().max()),
        "perturbed_teacher_target_change_mse": float(
            (perturbed_target - clean_target).square().mean()
        ),
        "linear_jacobian_action_max_abs_error": max(
            abs(row["jacobian_action_mse"] - 0.5) for row in jacobian["rows"]
        ),
        "linear_jacobian_zeta_halving_mse": jacobian["teacher_zeta_halving_mse"],
    }
