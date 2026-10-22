from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.func import jvp

from diffusion_lab.interval_model import initialize_interval_from_parent
from diffusion_lab.models import PointDenoiser
from diffusion_lab.research_methods import (
    adaptive_velocity_loss,
    flow_matching_pair,
    interval_euler_step,
    meanflow_target,
    shortcut_target,
)


class AnalyticField(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.calls = []

    def forward(self, sample, timestep, conditions, step_size=0):
        self.calls.append(
            (
                sample.detach().clone(),
                timestep.detach().clone(),
                step_size.detach().clone(),
                conditions.detach().clone(),
            )
        )
        shape = (-1,) + (1,) * (sample.ndim - 1)
        output = (
            self.gain * sample + timestep.reshape(shape).square() + 3 * step_size.reshape(shape)
        )
        return SimpleNamespace(sample=output + conditions.reshape(shape))


def analytic_inputs():
    x = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    t = torch.tensor([0.8, 0.7], dtype=torch.float64)
    d = torch.tensor([0.4, 0.2], dtype=torch.float64)
    c = torch.tensor([0.1, -0.3], dtype=torch.float64)
    v = torch.tensor([[1.1, -0.9], [0.2, 0.5]], dtype=torch.float64)
    return x, t, d, c, v


def test_meanflow_total_derivative_matches_analytic_and_canonical_finite_difference():
    model = AnalyticField()
    x, t, d, c, v = analytic_inputs()
    target, derivative = meanflow_target(model, x, t, d, v, c, return_derivative=True)
    expected_derivative = 0.7 * v + 2 * t[:, None] + 3
    assert torch.allclose(derivative, expected_derivative, atol=1e-13, rtol=0)
    assert torch.allclose(target, v - d[:, None] * expected_derivative, atol=1e-13, rtol=0)
    # Hold endpoint r=t-d fixed while moving x along v and increasing t.
    epsilon = 1e-5
    plus = model(x + epsilon * v, t + epsilon, c, step_size=d + epsilon).sample
    minus = model(x - epsilon * v, t - epsilon, c, step_size=d - epsilon).sample
    assert torch.allclose(derivative, (plus - minus) / (2 * epsilon), atol=1e-10, rtol=0)
    _, wrong = jvp(
        lambda a, b, h: model(a, b, c, step_size=h).sample,
        (x, t, d),
        (v, torch.ones_like(t), torch.zeros_like(d)),
    )
    assert torch.allclose(derivative - wrong, torch.full_like(derivative, 3.0))


def test_meanflow_target_stops_all_reverse_gradients_but_prediction_trains():
    model = AnalyticField()
    x, t, d, c, v = [value.requires_grad_() for value in analytic_inputs()]
    target = meanflow_target(model, x, t, d, v, c)
    assert not target.requires_grad and target.grad_fn is None
    assert all(value.grad is None for value in (x, t, d, c, v, model.gain))
    prediction = model(x, t, c, step_size=d).sample
    (prediction - target).square().mean().backward()
    assert model.gain.grad is not None and model.gain.grad.abs() > 0
    assert v.grad is None


def test_meanflow_zero_width_is_exact_flow_matching_boundary():
    x, t, d, c, v = analytic_inputs()
    target = meanflow_target(AnalyticField(), x, t, d * 0, v, c)
    assert torch.equal(target, v)


def test_shortcut_decreasing_time_two_actual_halfsteps_conditions_and_detach():
    model = AnalyticField()
    x, t, d, c, _ = analytic_inputs()
    target = shortcut_target(model, x.requires_grad_(), t, d, c)
    first_state, first_t, first_d, first_c = model.calls[0]
    second_state, second_t, second_d, second_c = model.calls[1]
    first = 0.7 * x.detach() + t[:, None].square() + 3 * d[:, None] / 2 + c[:, None]
    midpoint = x.detach() - d[:, None] / 2 * first
    second = 0.7 * midpoint + (t - d / 2)[:, None].square() + 3 * d[:, None] / 2 + c[:, None]
    assert torch.equal(first_state, x.detach()) and torch.equal(first_t, t)
    assert torch.equal(first_d, d / 2) and torch.equal(second_d, d / 2)
    assert torch.equal(second_state, midpoint) and torch.equal(second_t, t - d / 2)
    assert torch.equal(first_c, c) and torch.equal(second_c, c)
    assert torch.allclose(target, (first + second) / 2, atol=1e-15, rtol=0)
    assert not target.requires_grad and model.gain.grad is None and x.grad is None


def test_shortcut_smallest_halfstep_queries_fm_zero_but_moves_physical_state():
    model = AnalyticField()
    x, t, _, c, _ = analytic_inputs()
    d = torch.full_like(t, 1 / 16)
    shortcut_target(model, x, t, d, c, smallest_horizon=1 / 32)
    assert torch.equal(model.calls[0][2], torch.zeros_like(t))
    assert torch.equal(model.calls[1][2], torch.zeros_like(t))
    assert torch.equal(model.calls[1][1], t - 1 / 32)
    assert not torch.equal(model.calls[1][0], x)
    with pytest.raises(ValueError, match="positive step_size"):
        shortcut_target(model, x, t, 0.0, c)


def test_flow_matching_pair_and_sampling_share_sign_and_shapes():
    clean = torch.tensor([[[1.0, 2.0]], [[-1.0, 3.0]]], dtype=torch.float64)
    noise = clean + 2
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    state, velocity = flow_matching_pair(clean, noise, t)
    assert torch.allclose(state, clean + 2 * t[:, None, None])
    assert torch.equal(velocity, torch.full_like(clean, 2))

    class Constant(nn.Module):
        def forward(self, x, time, conditions, step_size):
            assert torch.equal(conditions, torch.tensor([2, 3]))
            return SimpleNamespace(sample=torch.full_like(x, 2))

    endpoint = interval_euler_step(Constant(), state, t, t, torch.tensor([2, 3]))
    assert torch.allclose(endpoint, clean)


def test_adaptive_loss_stops_weight_and_target_derivatives():
    prediction = torch.tensor([[1.0, 2.0], [-0.2, 0.7]], requires_grad=True)
    target = torch.tensor([[0.1, 0.4], [0.3, -0.5]], requires_grad=True)
    loss = adaptive_velocity_loss(prediction, target, power=1, epsilon=0.001)
    squared = (prediction.detach() - target.detach()).square().sum(1)
    expected_gradient = (prediction.detach() - target.detach()) / (squared[:, None] + 0.001)
    loss.backward()
    assert torch.allclose(prediction.grad, expected_gradient)
    assert target.grad is None
    assert torch.equal(
        adaptive_velocity_loss(prediction, target, power=0),
        (prediction - target).square().sum(1).mean(),
    )
    assert torch.equal(
        adaptive_velocity_loss(prediction, target, power=0, state_reduction="mean"),
        (prediction - target).square().mean(),
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_real_interval_model_meanflow_target_jvp_and_two_training_updates(continuous):
    torch.manual_seed(31)
    shape = (6, 2) if continuous else (2,)
    parent = PointDenoiser(
        sample_shape=shape,
        hidden_dim=16,
        num_layers=1,
        conditioner="mlp" if continuous else "embedding",
        num_classes=16,
        condition_dim=8 if continuous else 0,
    ).requires_grad_(False)
    model = initialize_interval_from_parent(parent)
    original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    parent_original = {key: tensor.clone() for key, tensor in parent.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    c = torch.randn(4, 8) if continuous else torch.tensor([0, 5, 9, 15])
    for _ in range(2):
        clean, noise = torch.randn(4, *shape), torch.randn(4, *shape)
        t = torch.tensor([0.3, 0.5, 0.8, 1.0])
        d = torch.tensor([0.0, 0.0, 0.2, 0.5])
        x, v = flow_matching_pair(clean, noise, t)
        target = meanflow_target(model, x, t, d, v, c)
        assert target.shape == x.shape and torch.isfinite(target).all()
        prediction = model(x, t, c, step_size=d).sample
        loss = adaptive_velocity_loss(prediction, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss) and model.output_projection.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    assert not torch.equal(original["output_projection.weight"], model.output_projection.weight)
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in parent_original.items())
    assert all(p.grad is None for p in parent.parameters())


def test_double_precision_model_jvp_matches_finite_difference_after_duration_learning():
    torch.manual_seed(53)
    model = initialize_interval_from_parent(
        PointDenoiser(hidden_dim=8, num_layers=1, conditioner="embedding", num_classes=16)
    ).double()
    with torch.no_grad():
        model.step_encoder[-1].weight.normal_(std=0.03)
    x, t, d, c, v = analytic_inputs()
    c = torch.tensor([1, 13])
    _, derivative = meanflow_target(model, x, t, d, v, c, return_derivative=True)
    eps = 1e-7
    plus = model(x + eps * v, t + eps, c, step_size=d + eps).sample
    minus = model(x - eps * v, t - eps, c, step_size=d - eps).sample
    assert torch.allclose(derivative, (plus - minus) / (2 * eps), atol=2e-7, rtol=2e-6)
