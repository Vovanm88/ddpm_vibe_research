"""Architecture-only Fourier checks; no model training runs, datasets or GPU work."""

import copy
import inspect
import math

import pytest
import torch

from recheck.fourier_model import FixedSpatialEncoder, FourierFMMLP, real_complex_power
from recheck.runtime import model_hash
from recheck.schedules import student_endpoint, teacher_target


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_default_architecture_interface_time_and_parameter_count():
    model = FourierFMMLP()
    sample = torch.randn(4, 2)
    assert model.sample_shape == (2,) and model.spatial_encoder.output_dim == 40
    assert model.config.hidden_dim == 256 and model.config.num_layers == 4
    assert model.config.time_embedding_dim == 32 and model.config.time_scale == 1000
    assert sum(p.numel() for p in model.parameters()) == 216578
    assert model(sample, 0.5).sample.shape == sample.shape
    with torch.no_grad():
        assert torch.equal(model(sample, 0.5).sample, model(sample, torch.full((4,), 0.5)).sample)
        assert torch.equal(model(sample, 0.5).sample, model(sample, 0.5, return_dict=False)[0])
        assert not torch.equal(model(sample, 0).sample, model(sample, 1).sample)
    for time in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="normalized"):
            model(sample, time)
    with pytest.raises(TypeError):
        model(sample, 0.5, conditions=torch.zeros(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_origin_and_near_origin_have_finite_input_and_parameter_gradients(dtype):
    model = FourierFMMLP(hidden_dim=16, num_layers=2).to(dtype=dtype)
    sample = torch.tensor(
        [[0.0, 0.0], [1e-12, -1e-12], [0.2, -0.1]], dtype=dtype, requires_grad=True
    )
    output = model(sample, torch.tensor([0.0, 0.5, 1.0], dtype=dtype)).sample
    output.square().mean().backward()
    assert output.dtype == dtype and torch.isfinite(output).all()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0


def test_real_binary_complex_power_matches_independent_complex_reference():
    real = torch.tensor([0.0, 0.3, -0.6], dtype=torch.float64, requires_grad=True)
    imaginary = torch.tensor([0.0, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
    for exponent in (1, 2, 3, 5, 16, 32):
        actual_real, actual_imaginary = real_complex_power(real, imaginary, exponent)
        expected = torch.complex(real, imaginary).pow(exponent)
        torch.testing.assert_close(actual_real, expected.real, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(actual_imaginary, expected.imag, atol=1e-14, rtol=1e-12)
    assert torch.autograd.gradcheck(
        lambda x, y: real_complex_power(x, y, 5), (real, imaginary), eps=1e-6, atol=1e-8, rtol=1e-5
    )


def test_encoder_known_features_and_smooth_polar_amplitude_at_origin():
    encoder = FixedSpatialEncoder(spatial_frequencies=(1, 3, 5), epsilon=0.001).double()
    points = torch.tensor([[0.0, 0.0], [0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    encoded = encoder(points)
    torch.testing.assert_close(encoded[:, :2], points)
    # raw xy (2), Cartesian sin/cos (12), radius/r² (2), polar harmonic pairs (6).
    assert encoded.shape == (2, 22)
    torch.testing.assert_close(encoded[:, 14], torch.sqrt(points.square().sum(1) + 1e-6) - 0.001)
    torch.testing.assert_close(encoded[:, 15], points.square().sum(1))
    assert torch.equal(encoded[0, 16:], torch.zeros(6, dtype=torch.float64))
    theta = math.atan2(0.4, 0.3)
    attenuation = 0.5 / math.sqrt(0.25 + 1e-6)
    for index, frequency in enumerate((1, 3, 5)):
        expected = (
            torch.tensor(
                [math.cos(frequency * theta), math.sin(frequency * theta)], dtype=torch.float64
            )
            * attenuation**frequency
        )
        torch.testing.assert_close(encoded[1, 16 + 2 * index : 18 + 2 * index], expected)
    assert torch.autograd.gradcheck(encoder, (points,), eps=1e-7, atol=1e-4, rtol=1e-5)


def test_fixed_encoding_is_dataset_independent_and_frequency_buffer_is_hashed():
    model = FourierFMMLP(hidden_dim=8, num_layers=1, spatial_frequencies=(1, 3, 7))
    points = torch.tensor([[0.3, 0.4], [-0.1, 0.8]])
    encoded = model.spatial_encoder(points)
    torch.testing.assert_close(encoded[:1], model.spatial_encoder(points[:1]))
    torch.testing.assert_close(encoded, model.spatial_encoder(points.flip(0)).flip(0))
    assert "spatial_encoder.frequencies" in model.state_dict()
    changed = copy.deepcopy(model)
    changed.spatial_encoder.frequencies[0] = 2
    assert model_hash(changed) != model_hash(model)
    source = inspect.getsource(__import__("recheck.fourier_model", fromlist=["FourierFMMLP"]))
    assert "atan2(" not in source and "torch.complex" not in source
    assert "make_environment" not in source and "rotating_rays" not in source


def test_native_diffusers_save_load_preserves_fixed_features_weights_and_dtype(tmp_path):
    model = FourierFMMLP(hidden_dim=16, num_layers=2, spatial_frequencies=(1, 3, 7)).eval()
    sample = torch.tensor([[0.0, 0.0], [0.4, -0.2]])
    model.save_pretrained(tmp_path)
    restored = FourierFMMLP.from_pretrained(tmp_path, low_cpu_mem_usage=False).eval()
    assert tuple(restored.config.spatial_frequencies) == (1, 3, 7)
    assert model_hash(restored) == model_hash(model)
    with torch.no_grad():
        assert torch.equal(restored(sample, 0.25).sample, model(sample, 0.25).sample)


def test_two_step_distillation_backpropagates_only_student_without_snapping():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        parent = FourierFMMLP(hidden_dim=16, num_layers=2).eval().requires_grad_(False)
    student = copy.deepcopy(parent).train().requires_grad_(True)
    original = model_hash(parent)
    assert model_hash(student) == original
    points = torch.tensor([[0.0, 0.0], [0.2, 0.1], [-0.4, 0.3], [1.2, -1.3]], requires_grad=True)
    target = teacher_target(parent, points, 1.0, 1 / 16, 2)
    predicted = student_endpoint(student, points, 1.0, 1 / 16)
    loss = (predicted - target).square().mean()
    loss.backward()
    assert not target.requires_grad and torch.isfinite(loss) and float(loss.detach()) > 0
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert all(p.grad is None for p in parent.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student.parameters())
    assert model_hash(parent) == original
    # General Cartesian velocity does not clip/project arbitrary input endpoints.
    assert predicted.shape == (4, 2) and torch.isfinite(predicted).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_frequencies": []},
        {"spatial_frequencies": [1, 1]},
        {"spatial_frequencies": [1, 0]},
        {"spatial_frequencies": [1, 1.5]},
        {"spatial_frequencies": [True]},
        {"epsilon": 0},
        {"epsilon": float("nan")},
        {"input_dim": 3},
        {"num_layers": 0},
    ],
)
def test_invalid_architecture_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        FourierFMMLP(**kwargs)


def test_cartesian_only_encoding_supports_dimensions_beyond_two():
    model = FourierFMMLP(input_dim=3, hidden_dim=8, num_layers=1, polar_features=False)
    sample = torch.zeros(2, 3, requires_grad=True)
    output = model(sample, 0.5).sample
    assert output.shape == sample.shape
    output.square().mean().backward()
    assert torch.isfinite(sample.grad).all()
