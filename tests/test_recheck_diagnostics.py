"""Signed integration, analytic oracle, and off-trajectory numerical tests."""

import math

import pytest
import torch

from recheck.diagnostics import (
    gradient_diagnostics,
    local_map_jacobian_diagnostics,
    local_rollout_diagnostics,
    paired_rollouts,
    student_macro_step,
    teacher_microsteps,
)
from recheck.oracle import analytic_gmm_velocity, integrate_rk4, oracle_numerical_report


def _states():
    return torch.randn((7, 2), generator=torch.Generator().manual_seed(18), dtype=torch.float64)


def test_signed_euler_and_microsteps_constant_field():
    x = _states()

    def velocity(state, time):
        return torch.ones_like(state) * 1.5

    expected = x - 0.25 * 1.5
    assert torch.allclose(student_macro_step(velocity, x, 1, 0.25), expected, atol=1e-15)
    assert torch.allclose(teacher_microsteps(velocity, x, 1, 0.25, 4), expected, atol=1e-15)


def test_copy_has_zero_one_substep_local_loss_but_not_automatically_multi_step():
    x = _states()

    def velocity(state, time):
        return state

    copy = student_macro_step(velocity, x, 1, 0.25)
    assert torch.equal(copy, teacher_microsteps(velocity, x, 1, 0.25, 1))
    assert (copy - teacher_microsteps(velocity, x, 1, 0.25, 2)).square().mean() > 0


def test_paired_rollout_counts_actual_microsteps_and_common_nodes():
    def velocity(state, time):
        return state * 0 + 0.3

    result = paired_rollouts(velocity, velocity, _states(), [1, 0.6, 0.2, 0], [2, 1, 2])
    assert result["teacher_nfe"] == 5
    assert result["student_nfe"] == 3
    assert result["teacher_history"].shape == (4, 7, 2)
    assert result["rollout_mse_per_node"].max() < 1e-30


def test_tube_retargets_teacher_from_changed_input_and_uses_private_rng():
    x = _states()
    teacher_inputs = []

    def teacher(state, time):
        teacher_inputs.append(state.clone())
        return state.square() * 0.1

    def student(state, time):
        return state.square() * 0.1

    state = torch.random.get_rng_state().clone()
    result = local_rollout_diagnostics(teacher, student, x + 0.1, x, [1, 0.5, 0], 2, seed=1919)
    assert torch.equal(state, torch.random.get_rng_state())
    assert result["teacher_targets_recomputed_from_perturbed_anchor"]
    assert len(result["local_error"]) == 2 * 3 * 4
    assert len(result["rollout_error"]) == 3
    assert any(not torch.equal(value, x) for value in teacher_inputs)
    # For rho=0, hand-evaluate the exact same initial local map.
    row = next(
        row
        for row in result["local_error"]
        if row["source"] == "interpolant" and row["rho_eval"] == 0 and row["interval"] == 0
    )
    expected = (
        (student_macro_step(student, x, 1, 0.5) - teacher_microsteps(teacher, x, 1, 0.5, 2))
        .square()
        .mean()
    )
    assert row["local_mse"] == pytest.approx(float(expected), abs=1e-16)
    again = local_rollout_diagnostics(teacher, student, x + 0.1, x, [1, 0.5, 0], 2, seed=1919)
    assert result == again


def test_local_derivative_linear_maps_matches_analytic_action_and_half_scale():
    anchors = _states()
    matrix = torch.tensor([[1.0, 2.0], [-0.5, 0.75]], dtype=torch.float64)

    def teacher(state):
        return state @ matrix.T

    def student(state):
        return 2 * (state @ matrix.T)

    direction = torch.tensor([[1.0, 0], [0, 1]], dtype=torch.float64)
    result = local_map_jacobian_diagnostics(
        teacher, student, anchors, directions=direction, zeta=1e-3
    )
    assert result["n_directions"] == 2
    assert result["map_evaluations_per_model"] == 8
    assert result["teacher_zeta_halving_mse"] < 1e-23
    assert result["student_zeta_halving_mse"] < 1e-22
    assert result["rows"][0]["jacobian_action_mse"] == pytest.approx(
        float(matrix[:, 0].square().mean()), abs=1e-12
    )
    assert all(row["relative_jacobian_action_mse"] == pytest.approx(1) for row in result["rows"])


def test_derivative_probes_use_unit_directions_without_advancing_global_rng():
    anchors = _states()
    state = torch.random.get_rng_state().clone()
    result = local_map_jacobian_diagnostics(lambda x: x.sin(), lambda x: x.sin(), anchors)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(row["jacobian_action_mse"] == 0 for row in result["rows"])
    assert result["teacher_zeta_halving_mse"] > 0


def test_gradient_probe_does_not_accumulate_grads_or_update_parameters():
    p = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    pair, distribution = (p - 1).square(), (p + 1).square()
    report = gradient_diagnostics(pair, distribution, [p], lambda_w=2)
    assert p.grad is None
    assert p.item() == 0.25
    assert report["pair_grad_norm"] == 1.5
    assert report["dist_grad_norm"] == 2.5
    assert report["grad_cosine"] == -1
    assert report["weighted_dist_to_pair_grad_ratio"] == pytest.approx(10 / 3)
    (pair + distribution).backward()
    assert p.grad.item() == 1


def test_analytic_oracle_endpoints_and_single_gaussian_conditional_mean():
    x = _states()
    mean = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    assert torch.allclose(analytic_gmm_velocity(x, 0), -x, atol=1e-14)
    mixture_mean = torch.tensor([0, 0.65 / 3], dtype=x.dtype)
    assert torch.allclose(analytic_gmm_velocity(x, 1), x - mixture_mean, atol=1e-14)
    t, sigma = 0.37, 0.15
    q, covariance = (1 - t) ** 2 * sigma**2 + t**2, t - (1 - t) * sigma**2
    expected = -mean + covariance / q * (x - (1 - t) * mean)
    assert torch.allclose(
        analytic_gmm_velocity(x, t, centers=mean, sigma=sigma), expected, atol=1e-14
    )


def test_analytic_oracle_accepts_per_example_time_and_nonuniform_weights():
    x = _states()
    t = torch.linspace(0, 1, len(x), dtype=x.dtype)
    batch = analytic_gmm_velocity(x, t)
    pointwise = torch.cat([analytic_gmm_velocity(point[None], time) for point, time in zip(x, t)])
    assert torch.allclose(batch, pointwise)
    result = analytic_gmm_velocity(x, 1, centers=[[-1, 0], [1, 0]], weights=[1, 3])
    assert torch.allclose(result, x - x.new_tensor([0.5, 0]), atol=1e-14)
    with pytest.raises(ValueError, match="Time"):
        analytic_gmm_velocity(x, 1.1)


def test_rk4_constant_field_correct_sign_and_actual_four_nfe_per_step():
    calls = []

    def field(state, time):
        calls.append(float(time))
        return torch.ones_like(state) * 0.4

    x = _states()
    result = integrate_rk4(field, x, times=[1, 0.8, 0.3, 0])
    assert result["nfe"] == len(calls) == 12
    assert result["solver_steps"] == 3
    assert result["history"].shape == (4, 7, 2)
    assert torch.allclose(result["samples"], x - 0.4, atol=1e-15)


def test_rk4_fourth_order_convergence_against_exact_exponential_solution():
    x = _states()
    exact = x * math.exp(-1)
    coarse = integrate_rk4(lambda state, time: state, x, steps=4)["samples"]
    fine = integrate_rk4(lambda state, time: state, x, steps=8)["samples"]
    error_coarse = float((coarse - exact).norm())
    error_fine = float((fine - exact).norm())
    assert error_fine < error_coarse / 14


def test_oracle_report_has_actual_errors_dtype_nfe_and_step_doubling():
    report = oracle_numerical_report(samples=16, steps=16)
    assert report["dtype"] == "torch.float64"
    assert report["endpoint_t0_max_abs_error"] < 1e-14
    assert report["endpoint_t1_max_abs_error"] < 1e-14
    assert report["rk4_actual_nfe"] == [64, 128, 256]
    assert report["rk4_converging"]
    assert report["rk4_step_doubling_mse"][1] < report["rk4_step_doubling_mse"][0]
