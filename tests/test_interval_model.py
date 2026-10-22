import copy
import json

import pytest
import torch

from diffusion_lab.interval_model import IntervalPointDenoiser, initialize_interval_from_parent
from diffusion_lab.models import PointDenoiser


def parent_model(*, continuous=False, backbone="mlp"):
    return PointDenoiser(
        sample_shape=(6, 2) if continuous else (2,),
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        backbone=backbone,
        conditioner="mlp" if continuous else "embedding",
        num_classes=16,
        condition_dim=8 if continuous else 0,
    )


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
def test_parent_exactly_preserved_at_every_initial_interval(continuous, backbone):
    torch.manual_seed(17)
    parent = parent_model(continuous=continuous, backbone=backbone).eval().requires_grad_(False)
    state = {key: value.clone() for key, value in parent.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model = initialize_interval_from_parent(parent, seed=21)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.device == parent.device and model.dtype == parent.dtype
    assert model.training is False
    assert all(parameter.requires_grad for parameter in model.parameters())
    x = torch.randn(4, *parent.sample_shape)
    t = torch.tensor([0.0, 0.25, 0.75, 1.0])
    condition = torch.randn(4, 8) if continuous else torch.tensor([0, 5, 9, 15])
    # Match inference context: a frozen Transformer uses a distinct fast path.
    with torch.no_grad():
        expected = parent(x, t * 1000, condition).sample
        for d in (torch.zeros_like(t), t / 2, t):
            actual = model(x, t, conditions=condition, step_size=d).sample
            assert torch.equal(actual, expected)
    assert model.config.base_config["condition_dim"] == (8 if continuous else 0)
    assert all(torch.equal(value, state[key]) for key, value in parent.state_dict().items())


def test_native_export_reload_preserves_duration_condition_and_config(tmp_path):
    torch.manual_seed(31)
    model = initialize_interval_from_parent(parent_model())
    # Once trained, duration must be a genuine independent input.
    with torch.no_grad():
        model.step_encoder[-1].weight.normal_(std=0.1)
    x, c = torch.randn(3, 2), torch.tensor([0, 7, 15])
    first = model(x, 0.8, c, step_size=0.1).sample
    second = model(x, 0.8, c, step_size=0.7).sample
    assert not torch.allclose(first, second)
    assert not torch.allclose(first, model(x, 0.8, c.flip(0), step_size=0.1).sample)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["_class_name"] == "IntervalPointDenoiser"
    assert config["allow_null"] is False
    assert config["time_scale"] == config["step_scale"] == 1000.0
    restored = IntervalPointDenoiser.from_pretrained(tmp_path)
    assert restored.sample_shape == (2,)
    assert torch.equal(first, restored(x, 0.8, c, step_size=0.1).sample)
    assert set(restored.state_dict()) == set(model.state_dict())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "t,d", [(1.1, 0.1), (-0.1, 0), (0.5, -0.1), (0.5, 0.6), (float("nan"), 0), (1, float("inf"))]
)
def test_invalid_normalized_intervals_rejected(t, d):
    model = initialize_interval_from_parent(parent_model())
    with pytest.raises(ValueError, match="finite|Normalized interval"):
        model(torch.randn(2, 2), t, torch.tensor([0, 1]), step_size=d)


def test_explicit_conditions_and_valid_condition_ids_required():
    model = initialize_interval_from_parent(parent_model())
    x = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="Explicit conditions"):
        model(x, 1.0, step_size=0.5)
    with pytest.raises(ValueError, match="dropout"):
        model(x, 1.0, torch.tensor([0, 1]), condition_mask=torch.tensor([True, False]))
    with pytest.raises(ValueError, match="integer IDs"):
        model(x, 1.0, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="Class IDs"):
        model(x, 1.0, torch.tensor([0, 16]))
    explicit_null = initialize_interval_from_parent(parent_model(), allow_null=True)
    assert explicit_null(x, 1.0, step_size=0.5).sample.shape == x.shape


def test_wall_state_condition_and_finite_backward():
    parent = PointDenoiser(
        sample_shape=(30, 2), hidden_dim=16, num_layers=1, conditioner="mlp", condition_dim=8
    )
    model = initialize_interval_from_parent(parent)
    x = torch.randn(3, 30, 2, requires_grad=True)
    c = torch.randn(3, 8)
    output = model(x, torch.tensor([0.4, 0.7, 1.0]), c, step_size=0.2)
    output.sample.square().mean().backward()
    assert output.sample.shape == (3, 30, 2)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.step_encoder[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in parent.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.equal(
        output.sample.detach(),
        model(x.detach(), torch.tensor([0.4, 0.7, 1.0]), c, step_size=0.2, return_dict=False)[0],
    )


def test_parent_clones_independent_and_deterministic():
    parent = parent_model()
    a = initialize_interval_from_parent(parent, seed=91)
    b = initialize_interval_from_parent(parent, seed=91)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    with torch.no_grad():
        a.output_projection.bias.add_(1)
    assert not torch.equal(a.output_projection.bias, b.output_projection.bias)
    assert torch.equal(b.output_projection.bias, parent.output_projection.bias)
    with pytest.raises(TypeError):
        initialize_interval_from_parent(copy.deepcopy(a))
