"""Same-objective/gradient tests and the actual frozen wall update17 regression."""

import json
from pathlib import Path

import pytest
import torch

from diffusion_lab.transport import BACKEND, sinkhorn_divergence
from recheck.losses import sinkhorn_divergence as reference

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "replication/reports/wall_navigation/numerics/failed_e0_n2_update17.pt"
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_values_gradients_and_full_kl_match_frozen_reference_on_moderate_problem():
    rng = torch.Generator().manual_seed(712)
    x = (torch.randn(7, 3, generator=rng, dtype=torch.float64) * 0.3).requires_grad_()
    y = (torch.randn(5, 3, generator=rng, dtype=torch.float64) * 0.3).requires_grad_()
    value, diagnostics = sinkhorn_divergence(x, y, 0.2, tolerance=1e-13, return_diagnostics=True)
    expected, old = reference(x, y, 0.2, tolerance=1e-13, return_diagnostics=True)
    gradient, teacher_gradient = torch.autograd.grad(value, (x, y), allow_unused=True)
    expected_gradient = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(value, expected, atol=2e-12, rtol=1e-10)
    torch.testing.assert_close(gradient, expected_gradient, atol=2e-11, rtol=1e-9)
    assert teacher_gradient is None
    for name in ("cross", "student_self", "teacher_self"):
        diag = diagnostics[name]
        assert diag["all_stages_converged"]
        assert diag["epsilon_schedule"][-1] == 0.2
        assert max(diag["row_marginal_error"], diag["column_marginal_error"]) <= 1e-13
        assert diag["regularized_objective"] == pytest.approx(
            diag["transport_cost"] + 0.2 * diag["kl_to_product"], abs=1e-14
        )
        assert diag["regularized_objective"] == pytest.approx(
            old[name]["regularized_objective"], abs=2e-12
        )
    json.dumps(diagnostics, allow_nan=False)


def test_finite_difference_gradcheck_includes_both_student_self_occurrences():
    rng = torch.Generator().manual_seed(817)
    x = torch.randn(4, 2, generator=rng, dtype=torch.float64, requires_grad=True) * 0.2
    y = torch.randn(4, 2, generator=rng, dtype=torch.float64) * 0.2
    assert torch.autograd.gradcheck(
        lambda points: sinkhorn_divergence(points, y, 0.13, tolerance=1e-13),
        (x,),
        eps=1e-6,
        atol=2e-7,
        rtol=1e-4,
    )


def test_identity_permutation_and_translation_contract_without_clamps():
    x = torch.tensor(
        [[-0.3, 0.2], [0.4, -0.1], [0.1, 0.5]], dtype=torch.float64, requires_grad=True
    )
    zero = sinkhorn_divergence(x, x.detach(), 0.1, tolerance=1e-13)
    grad = torch.autograd.grad(zero, x)[0]
    assert abs(float(zero.detach())) < 1e-12 and float(grad.abs().max()) < 1e-10
    permutation = torch.tensor([2, 0, 1])
    assert (
        abs(float(sinkhorn_divergence(x, x.detach()[permutation], 0.1, tolerance=1e-13).detach()))
        < 1e-12
    )
    offset = x.detach().new_tensor([0.25, -0.4])
    shifted = sinkhorn_divergence(x, x.detach() + offset, 0.1, tolerance=1e-13)
    assert float(shifted.detach()) == pytest.approx(float(offset.square().sum()), abs=1e-12)


def test_actual_update17_cold_failure_fixed_at_identical_epsilon_and_tolerance():
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    assert payload["update"] == 17 and payload["scene_id"] == 70 and payload["interval"] == 1
    assert payload["x"].shape == payload["y"].shape == (128, 60)
    epsilon = payload["epsilon"]
    with pytest.raises(RuntimeError, match="did not converge"):
        reference(payload["x"], payload["y"], epsilon, iterations=2000, tolerance=1e-10)
    x = payload["x"].clone().requires_grad_()
    y = payload["y"].clone().requires_grad_()
    loss, diag = sinkhorn_divergence(
        x, y, epsilon, iterations=2000, tolerance=1e-10, return_diagnostics=True
    )
    loss.backward()
    assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.norm() > 0 and y.grad is None
    assert diag["backend"] == BACKEND and diag["epsilon"] == epsilon and diag["tolerance"] == 1e-10
    assert float(loss.detach()) == pytest.approx(0.0013963618331270018, abs=3e-12)
    for term in ("cross", "student_self", "teacher_self"):
        assert diag[term]["all_stages_converged"]
        assert diag[term]["epsilon_schedule"][-1] == epsilon
        assert max(diag[term]["row_marginal_error"], diag[term]["column_marginal_error"]) <= 1e-10
        assert all(
            max(stage["row_marginal_error"], stage["column_marginal_error"]) <= 1e-10
            for stage in diag[term]["stages"]
        )


def test_actual_batch_envelope_gradient_matches_independent_directional_difference():
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    x = payload["x"].clone().requires_grad_()
    y = payload["y"]
    epsilon = payload["epsilon"]
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(719), dtype=x.dtype)
    direction /= direction.norm()
    loss = sinkhorn_divergence(x, y, epsilon, tolerance=1e-12)
    derivative = float((torch.autograd.grad(loss, x)[0] * direction).sum())
    delta = 3e-6
    with torch.no_grad():
        plus = sinkhorn_divergence(x + delta * direction, y, epsilon, tolerance=1e-12)
        minus = sinkhorn_divergence(x - delta * direction, y, epsilon, tolerance=1e-12)
    finite_difference = float((plus - minus) / (2 * delta))
    assert derivative == pytest.approx(finite_difference, rel=3e-4, abs=2e-7)


def test_two_cluster_unequal_component_masses_remain_strict_without_kernel_truncation():
    x = torch.tensor([[0.0], [0.001], [0.002], [2.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0.0], [0.001], [2.0], [2.001]], dtype=torch.float64)
    epsilon = 0.0005
    loss, diag = sinkhorn_divergence(x, y, epsilon, return_diagnostics=True)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert diag["cross"]["converged"]
    assert max(diag["cross"]["row_marginal_error"], diag["cross"]["column_marginal_error"]) <= 1e-10
    assert diag["cross"]["epsilon_schedule"][-1] == epsilon


def test_float32_network_inputs_receive_nonzero_gradients_and_teacher_is_detached():
    rng = torch.Generator().manual_seed(611)
    x = torch.randn(8, 4, generator=rng).requires_grad_()
    y = torch.randn(8, 4, generator=rng).requires_grad_()
    loss, diag = sinkhorn_divergence(x, y, 0.1, return_diagnostics=True)
    loss.backward()
    assert loss.dtype == torch.float64 and x.grad.dtype == torch.float32
    assert x.grad.norm() > 0 and torch.isfinite(x.grad).all() and y.grad is None
    assert diag["compute_device"] == "cpu" and not diag["differentiable_cpu_bridge"]


def test_failure_is_explicit_and_never_switches_loss_epsilon_or_tolerance():
    x = torch.tensor([[0.0], [0.2], [1.0]], dtype=torch.float64)
    y = torch.tensor([[0.1], [0.3], [1.4]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="no fallback"):
        sinkhorn_divergence(x, y, 0.0001, iterations=1, tolerance=1e-13)
    _, diag = sinkhorn_divergence(
        x,
        y,
        0.0001,
        iterations=1,
        tolerance=1e-13,
        require_convergence=False,
        return_diagnostics=True,
    )
    assert diag["epsilon"] == 0.0001 and diag["tolerance"] == 1e-13
    assert not diag["cross"]["all_stages_converged"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0},
        {"epsilon": float("nan")},
        {"iterations": 0},
        {"iterations": True},
        {"tolerance": 0},
        {"epsilon_scaling": 0},
        {"epsilon_scaling": 1},
    ],
)
def test_invalid_solver_parameters_rejected(kwargs):
    supplied = {"epsilon": 0.1} | kwargs
    with pytest.raises(ValueError):
        sinkhorn_divergence(torch.zeros(2, 2), torch.ones(2, 2), **supplied)
