import math

import pytest
import torch

from recheck.calibration import calibrate_interval_features
from recheck.losses import (
    exact_assignment_cost,
    features,
    geomloss_sinkhorn,
    pair_loss,
    random_directions,
    sinkhorn_divergence,
    sliced_wasserstein_loss,
)


@pytest.fixture
def supports():
    generator = torch.Generator().manual_seed(987)
    x = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.4
    y = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.4
    return x, y


def test_full_image_feature_normalization_and_joint_denominator():
    anchor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    displacement = torch.tensor([[[[0.5, 1.0], [-1.0, 2.0]]]], dtype=torch.float64)
    endpoint = anchor + displacement
    torch.testing.assert_close(features(anchor, endpoint, "endpoint"), endpoint.flatten(1) / 2)
    torch.testing.assert_close(
        features(anchor, endpoint, "displacement"), displacement.flatten(1) / 2
    )
    joint = features(anchor, endpoint, "joint", gamma=4)
    assert joint.shape == (1, 8)
    torch.testing.assert_close(
        joint, torch.cat((2 * anchor.flatten(1), displacement.flatten(1)), dim=1) / 2
    )


def test_endpoint_and_displacement_pair_values_and_gradients(supports):
    anchor, target = supports
    endpoint = (target + 0.3).requires_grad_()
    loss_endpoint = pair_loss(endpoint, target)
    loss_displacement = pair_loss(endpoint - anchor, target - anchor)
    torch.testing.assert_close(loss_endpoint, loss_displacement, atol=1e-12, rtol=1e-12)
    grad_endpoint = torch.autograd.grad(loss_endpoint, endpoint)[0]
    grad_displacement = torch.autograd.grad(loss_displacement, endpoint)[0]
    torch.testing.assert_close(grad_endpoint, grad_displacement, atol=1e-12, rtol=1e-12)


def test_identity_permutation_and_paired_mismatch(supports):
    x, _ = supports
    permuted = x[torch.tensor([4, 2, 0, 3, 1])]
    directions = random_directions(
        3, 128, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    torch.testing.assert_close(
        sinkhorn_divergence(x, x, 0.1), torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        sinkhorn_divergence(x, permuted, 0.1),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )
    assert sliced_wasserstein_loss(x, x, directions=directions) == 0
    assert sliced_wasserstein_loss(x, permuted, directions=directions) < 1e-28
    assert pair_loss(x, permuted) > 0.01


def test_endpoint_permutation_is_distinct_from_joint_association_adversary():
    anchor = torch.tensor([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    displacement = torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    endpoint = anchor + displacement
    shuffled_endpoint = endpoint[[1, 2, 0]]
    assert (
        sinkhorn_divergence(
            features(anchor, endpoint, "endpoint"),
            features(anchor, shuffled_endpoint, "endpoint"),
            0.5,
        ).abs()
        < 1e-9
    )
    assert pair_loss(endpoint, shuffled_endpoint) > 1
    reassigned_endpoint = anchor + displacement[[1, 2, 0]]
    assert (
        sinkhorn_divergence(
            features(anchor, endpoint, "displacement"),
            features(anchor, reassigned_endpoint, "displacement"),
            0.5,
        ).abs()
        < 1e-9
    )
    joint_loss = sinkhorn_divergence(
        features(anchor, endpoint, "joint"), features(anchor, reassigned_endpoint, "joint"), 0.5
    )
    assert joint_loss > 0.1


def test_single_point_and_finite_projection_bank():
    x = torch.tensor([[1.0, -2.0, 0.4]], dtype=torch.float64)
    y = torch.tensor([[-0.5, 0.5, 0.7]], dtype=torch.float64)
    expected = (x - y).square().sum()
    torch.testing.assert_close(sinkhorn_divergence(x, y, 0.2), expected)
    directions = random_directions(
        3, 64, generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    swd = sliced_wasserstein_loss(x, y, directions=directions)
    torch.testing.assert_close(swd, 3 * ((x - y) @ directions.T).square().mean())
    # A finite random bank need not equal the exact squared cost.
    assert not torch.isclose(swd, expected)


def test_translation_exact_sinkhorn_and_swd_scaling(supports):
    x, _ = supports
    translation = torch.tensor([0.25, -0.1, 0.4], dtype=torch.float64)
    expected = translation.square().sum()
    torch.testing.assert_close(
        exact_assignment_cost(x, x + translation), expected, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        sinkhorn_divergence(x, x + translation, 0.3), expected, atol=1e-9, rtol=1e-8
    )
    # Orthonormal directions integrate quadratic projections exactly.
    torch.testing.assert_close(
        sliced_wasserstein_loss(x, x + translation, directions=torch.eye(3, dtype=torch.float64)),
        expected,
    )
    bank = random_directions(
        3, 100_000, generator=torch.Generator().manual_seed(6), dtype=torch.float64
    )
    approximation = sliced_wasserstein_loss(x, x + translation, directions=bank)
    torch.testing.assert_close(approximation, expected, atol=0.002, rtol=0.01)


def test_exact_ot_upper_bound_by_paired_coupling(supports):
    x, y = supports
    assert exact_assignment_cost(x, y) <= (x - y).square().sum(1).mean() + 1e-12


def test_sinkhorn_self_term_gradcheck_and_teacher_detached(supports):
    x, y = supports
    x = x[:3].requires_grad_()
    y = y[:3].requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value: sinkhorn_divergence(value, y, 0.3, tolerance=1e-13),
        (x,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )
    loss = sinkhorn_divergence(x, y, 0.3)
    loss.backward()
    assert x.grad is not None and x.grad.norm() > 0 and torch.isfinite(x.grad).all()
    assert y.grad is None


def test_solver_marginals_and_iteration_doubling(supports):
    x, y = supports
    x = x.requires_grad_()
    first, diagnostics = sinkhorn_divergence(
        x, y, 0.3, iterations=500, tolerance=1e-12, return_diagnostics=True
    )
    first_gradient = torch.autograd.grad(first, x)[0]
    second = sinkhorn_divergence(x, y, 0.3, iterations=1000, tolerance=1e-13)
    second_gradient = torch.autograd.grad(second, x)[0]
    assert diagnostics["dtype"] == "float64"
    for name in ("cross", "student_self", "teacher_self"):
        assert diagnostics[name]["converged"]
        assert diagnostics[name]["row_marginal_error"] < 1e-12
        assert diagnostics[name]["column_marginal_error"] < 1e-12
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(first_gradient, second_gradient, atol=1e-9, rtol=1e-9)


def test_unconverged_solver_raises_instead_of_hiding_failure(supports):
    x, y = supports
    with pytest.raises(RuntimeError, match="did not converge"):
        sinkhorn_divergence(x, y, 0.001, iterations=1, tolerance=1e-14)


def test_disconnected_support_dual_gauges_converge_without_relaxing_marginals():
    # Regression distilled from a real failure during the synthetic rays pilot.
    # A dense cloud and a nearly isolated outlier create TWO dual gauge modes.
    # The former one-coordinate Newton solve produced ~1e14 directions and
    # stalled above 1e-10 even after 2,000 iterations. No real data are used.
    rng = torch.Generator().manual_seed(1)
    x = 0.2 * torch.randn(32, 2, generator=rng, dtype=torch.float64)
    displacement = 0.001 * torch.randn(32, 2, generator=rng, dtype=torch.float64)
    x[-1] = torch.tensor([0.1, -1.8], dtype=torch.float64)
    y = x + displacement
    x = x.requires_grad_()
    first, diagnostics = sinkhorn_divergence(
        x, y, 0.02, iterations=2000, tolerance=1e-12, return_diagnostics=True
    )
    gradient = torch.autograd.grad(first, x)[0]
    second = sinkhorn_divergence(x, y, 0.02, iterations=4000, tolerance=1e-13)
    second_gradient = torch.autograd.grad(second, x)[0]
    assert diagnostics["cross"]["max_numerical_dual_nullity"] >= 2
    for name in ("cross", "student_self", "teacher_self"):
        assert diagnostics[name]["converged"]
        assert diagnostics[name]["row_marginal_error"] <= 1e-12
        assert diagnostics[name]["column_marginal_error"] <= 1e-12
    torch.testing.assert_close(first, second, rtol=1e-7, atol=1e-12)
    torch.testing.assert_close(gradient, second_gradient, rtol=1e-7, atol=1e-10)
    direction = torch.randn(
        x.shape, generator=torch.Generator().manual_seed(55), dtype=torch.float64
    )
    direction /= direction.norm()
    step = 1e-5
    plus = sinkhorn_divergence(x.detach() + step * direction, y, 0.02, tolerance=1e-13)
    minus = sinkhorn_divergence(x.detach() - step * direction, y, 0.02, tolerance=1e-13)
    finite_difference = (plus - minus) / (2 * step)
    torch.testing.assert_close(
        (gradient * direction).sum(), finite_difference, rtol=1e-5, atol=1e-8
    )


def test_geomloss_value_and_gradient_parity(supports):
    pytest.importorskip("geomloss")
    x, y = supports
    x = x.requires_grad_()
    reference = sinkhorn_divergence(x, y, 0.3, tolerance=1e-13)
    library = geomloss_sinkhorn(x, y, 0.3, scaling=0.9999)
    reference_gradient = torch.autograd.grad(reference, x)[0]
    library_gradient = torch.autograd.grad(library, x)[0]
    torch.testing.assert_close(library, reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(library_gradient, reference_gradient, atol=2e-5, rtol=2e-4)


def test_swd_uses_independent_rng_and_both_supports_share_directions(supports):
    x, y = supports
    global_state = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(77)
    loss = sliced_wasserstein_loss(x, y, generator=generator)
    assert torch.equal(global_state, torch.random.get_rng_state())
    directions = random_directions(
        3, 128, generator=torch.Generator().manual_seed(77), dtype=torch.float64
    )
    torch.testing.assert_close(loss, sliced_wasserstein_loss(x, y, directions=directions))
    with pytest.raises(ValueError, match="independent"):
        sliced_wasserstein_loss(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_swd_cpu_generator_can_supply_cuda_model():
    cpu_generator = torch.Generator(device="cpu").manual_seed(91)
    cuda_bank = random_directions(3, 128, generator=cpu_generator, device="cuda")
    expected = random_directions(3, 128, generator=torch.Generator().manual_seed(91))
    torch.testing.assert_close(cuda_bank.cpu(), expected)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal unavailable in this process"
)
def test_sinkhorn_metal_cpu_float64_bridge_preserves_student_gradient(supports):
    cpu_x, cpu_y = supports
    x = cpu_x.float().to("mps").requires_grad_()
    y = cpu_y.float().to("mps").requires_grad_()
    result, diagnostics = sinkhorn_divergence(x, y, 0.3, return_diagnostics=True)
    result.backward()
    cpu_reference_x = x.detach().cpu().requires_grad_()
    cpu_reference = sinkhorn_divergence(cpu_reference_x, y.detach().cpu(), 0.3)
    cpu_reference.backward()
    assert result.device.type == "mps" and result.dtype == torch.float32
    assert diagnostics["compute_device"] == "cpu"
    assert diagnostics["dtype"] == "float64"
    assert diagnostics["differentiable_cpu_bridge"]
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert y.grad is None
    torch.testing.assert_close(result.cpu().double(), cpu_reference.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x.grad.cpu(), cpu_reference_x.grad, rtol=1e-5, atol=1e-6)
    bank = random_directions(3, 16, generator=torch.Generator().manual_seed(91), device="mps")
    assert bank.device.type == "mps"
    swd = sliced_wasserstein_loss(x, y, directions=bank)
    swd_gradient = torch.autograd.grad(swd, x)[0]
    assert torch.isfinite(swd_gradient).all() and swd_gradient.norm() > 0
    metal_calibration = calibrate_interval_features(y)
    cpu_calibration = calibrate_interval_features(y.detach().cpu())
    assert metal_calibration == cpu_calibration
    assert not metal_calibration["degenerate"]
    metal_exact = exact_assignment_cost(x, y)
    cpu_exact = exact_assignment_cost(x.detach().cpu(), y.detach().cpu())
    torch.testing.assert_close(metal_exact.cpu().double(), cpu_exact, rtol=1e-6, atol=1e-7)


def test_calibration_fixed_teacher_offdiagonal_median_and_floor():
    teacher = torch.tensor([[0.0], [2.0], [5.0]], dtype=torch.float64, requires_grad=True)
    calibration = calibrate_interval_features(teacher, alpha=0.05)
    assert calibration["median_off_diagonal_cost"] == 9
    assert math.isclose(calibration["epsilon"], 0.45)
    assert calibration["support_size"] == 3
    assert calibration["shared_across_tube_conditions"]
    with pytest.warns(RuntimeWarning, match="Degenerate"):
        degenerate = calibrate_interval_features(torch.zeros(4, 2))
    assert degenerate["degenerate"]
    assert degenerate["epsilon"] == 0.05 * 1e-8
