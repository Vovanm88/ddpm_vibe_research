"""Numerical and serialization invariants, including regressions from the notebook."""

import pytest
import torch

from diffusion_lab.models import DenoiserOutput, PointDenoiser
from diffusion_lab.pipeline import ResearchDiffusionPipeline
from diffusion_lab.schedulers import make_scheduler, prediction_to_x0


@pytest.mark.parametrize("sample_shape", [(1,), (2,), (3,), (7, 2), (5, 3)])
@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
def test_forward_preserves_shape_and_optimizer_sees_every_parameter(sample_shape, backbone):
    model = PointDenoiser(
        sample_shape=sample_shape,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        backbone=backbone,
        num_classes=3,
    )
    parameter_ids = {id(p) for p in model.parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample = torch.randn(4, *sample_shape)
    prediction = model(sample, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 0])).sample
    assert prediction.shape == sample.shape
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert parameter_ids == {id(p) for p in model.parameters()}
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized_ids == parameter_ids
    assert model.input_projection.weight.grad is not None
    assert model.output_projection.weight.grad is not None


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
def test_batch_members_do_not_attend_to_other_samples(backbone):
    model = PointDenoiser(
        sample_shape=(4, 2), hidden_dim=16, num_layers=1, backbone=backbone, conditioner="none"
    ).eval()
    first = torch.randn(1, 4, 2)
    together = torch.cat((first, torch.randn(3, 4, 2)), dim=0)
    alone = model(first, 4).sample
    mixed = model(together, 4).sample[:1]
    torch.testing.assert_close(alone, mixed, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("conditioner", ["one_hot", "embedding", "mlp"])
def test_condition_mask_is_an_explicit_unconditional_token(conditioner):
    model = PointDenoiser(
        hidden_dim=16, num_layers=1, conditioner=conditioner, num_classes=3, condition_dim=4
    ).eval()
    sample = torch.randn(4, 2)
    conditions = torch.randn(4, 4) if conditioner == "mlp" else torch.tensor([0, 1, 2, 0])
    unconditional = model(sample, 7).sample
    masked = model(sample, 7, conditions, condition_mask=torch.zeros(4, dtype=torch.bool)).sample
    torch.testing.assert_close(unconditional, masked)
    conditional = model(sample, 7, conditions).sample
    kept = model(sample, 7, conditions, condition_mask=torch.ones(4, dtype=torch.bool)).sample
    torch.testing.assert_close(conditional, kept)


@pytest.mark.parametrize(
    "conditions",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0], [1]]),
    ],
)
def test_invalid_class_conditions_fail_clearly(conditions):
    model = PointDenoiser(hidden_dim=16, num_layers=1, num_classes=3)
    with pytest.raises(ValueError):
        model(torch.randn(2, 2), 0, conditions)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
@pytest.mark.parametrize("sample_shape", [(1,), (3,), (5, 2)])
def test_known_forward_process_recovers_clean_samples(prediction_type, sample_shape):
    scheduler = make_scheduler("ddim", num_train_timesteps=32, prediction_type=prediction_type)
    clean = torch.randn(4, *sample_shape)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 3, 15, 30])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    if prediction_type == "epsilon":
        prediction = noise
    elif prediction_type == "sample":
        prediction = clean
    else:
        prediction = scheduler.get_velocity(clean, noise, timesteps)
    recovered = prediction_to_x0(scheduler, noisy, prediction, timesteps)
    torch.testing.assert_close(recovered, clean, atol=2e-5, rtol=2e-5)


def test_cosine_schedule_is_monotone_and_final_ddim_step_is_clean():
    scheduler = make_scheduler("ddim", num_train_timesteps=32)
    assert torch.all(scheduler.betas > 0)
    assert torch.all(scheduler.betas < 1)
    assert torch.all(scheduler.alphas_cumprod[1:] < scheduler.alphas_cumprod[:-1])
    scheduler.set_timesteps(32)
    clean = torch.randn(5, 2)
    noise = torch.randn_like(clean)
    noisy = scheduler.add_noise(clean, noise, torch.zeros(5, dtype=torch.long))
    recovered = scheduler.step(noise, 0, noisy, eta=0).prev_sample
    assert torch.isfinite(recovered).all()
    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "timesteps", [torch.tensor([-1, 0]), torch.tensor([0, 32]), torch.tensor([0.0, 1.0])]
)
def test_out_of_range_or_noninteger_timestep_reconstruction_is_rejected(timesteps):
    scheduler = make_scheduler("ddim", num_train_timesteps=32)
    sample = torch.randn(2, 2)
    with pytest.raises(ValueError):
        prediction_to_x0(scheduler, sample, torch.zeros_like(sample), timesteps)


def test_wrong_sample_shape_and_transformer_head_dimension_are_rejected():
    model = PointDenoiser(sample_shape=(3,), hidden_dim=16, num_layers=1)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 2), 0)
    with pytest.raises(ValueError, match="divisible"):
        PointDenoiser(backbone="transformer", hidden_dim=17, num_heads=4)


def _small_pipeline(sample_shape=(2,), scheduler_kind="ddim"):
    model = PointDenoiser(sample_shape=sample_shape, hidden_dim=16, num_layers=1, num_classes=3)
    return ResearchDiffusionPipeline(model, make_scheduler(scheduler_kind, num_train_timesteps=24))


@pytest.mark.parametrize("sample_shape", [(1,), (2,), (3,), (8, 2)])
@pytest.mark.parametrize("scheduler_kind", ["ddpm", "ddim", "flow"])
def test_sampling_supports_shapes_and_history_uses_real_timesteps(sample_shape, scheduler_kind):
    pipeline = _small_pipeline(sample_shape, scheduler_kind)
    result = pipeline(
        batch_size=3,
        conditions=1,
        num_inference_steps=6,
        generator=torch.Generator().manual_seed(123),
        record_history=True,
    )
    assert result.samples.shape == (3, *sample_shape)
    assert result.history.shape == (7, 3, *sample_shape)
    assert result.timesteps.shape == (6,)
    assert torch.isfinite(result.samples).all()
    assert torch.all(result.timesteps[:-1] > result.timesteps[1:])
    torch.testing.assert_close(result.history[-1], result.samples.cpu())
    torch.testing.assert_close(result.timesteps, pipeline.scheduler.timesteps.cpu())


@pytest.mark.parametrize("training_mode", [False, True])
def test_sampling_and_field_queries_restore_original_model_mode(training_mode):
    pipeline = _small_pipeline()
    pipeline.model.train(training_mode)
    pipeline(batch_size=2, num_inference_steps=3)
    assert pipeline.model.training is training_mode
    pipeline.vector_field(torch.randn(3, 2), 5)
    assert pipeline.model.training is training_mode
    with pytest.raises(ValueError):
        pipeline(batch_size=2, conditions=torch.tensor([0, 99]), num_inference_steps=3)
    assert pipeline.model.training is training_mode


def test_cfg_zero_matches_unconditional_and_one_matches_conditional():
    pipeline = _small_pipeline()
    points = torch.randn(5, 2)
    conditions = torch.tensor([0, 1, 2, 0, 1])
    unconditional = pipeline.vector_field(points, 5, kind="epsilon")
    zero = pipeline.vector_field(points, 5, conditions, kind="epsilon", guidance_scale=0)
    torch.testing.assert_close(zero, unconditional, atol=1e-5, rtol=1e-5)
    pipeline.model.eval()
    conditional = pipeline.model(points, 5, conditions).sample
    one = pipeline.vector_field(points, 5, conditions, kind="epsilon", guidance_scale=1)
    torch.testing.assert_close(one, conditional, atol=1e-5, rtol=1e-5)
    guided = pipeline.vector_field(points, 5, conditions, kind="epsilon", guidance_scale=2)
    torch.testing.assert_close(
        guided, unconditional + 2 * (conditional - unconditional), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_field_conventions_are_consistent_across_prediction_types(prediction_type):
    pipeline = _small_pipeline()
    pipeline.scheduler = make_scheduler(
        "ddim", num_train_timesteps=24, prediction_type=prediction_type
    )
    points = torch.randn(5, 2)
    epsilon = pipeline.vector_field(points, 5, kind="epsilon")
    score = pipeline.vector_field(points, 5, kind="score")
    denoising = pipeline.vector_field(points, 5, kind="denoising")
    a = pipeline.scheduler.alphas_cumprod[5]
    torch.testing.assert_close(score, -epsilon / (1 - a).sqrt())
    clean = (points - (1 - a).sqrt() * epsilon) / a.sqrt()
    torch.testing.assert_close(denoising, clean - points, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scheduler_kind", ["ddpm", "ddim", "flow"])
def test_endpoint_constraints_are_exact_after_sampling(scheduler_kind):
    pipeline = _small_pipeline((6, 2), scheduler_kind)
    known = torch.zeros(6, 2)
    known[0], known[-1] = torch.tensor([-1.0, 0.5]), torch.tensor([1.0, -0.5])
    mask = torch.zeros(6, 2, dtype=torch.bool)
    mask[0] = mask[-1] = True
    output = pipeline(batch_size=3, num_inference_steps=4, known_values=known, known_mask=mask)
    torch.testing.assert_close(output.samples[:, 0], known[0].expand(3, -1))
    torch.testing.assert_close(output.samples[:, -1], known[-1].expand(3, -1))


def test_pipeline_save_load_reproduces_seeded_ddim_samples(tmp_path):
    pipeline = _small_pipeline((5, 3))
    first = pipeline(
        batch_size=2,
        conditions=1,
        num_inference_steps=6,
        generator=torch.Generator().manual_seed(7),
    ).samples
    pipeline.save_pretrained(tmp_path)
    restored = ResearchDiffusionPipeline.from_pretrained(tmp_path, local_files_only=True)
    second = restored(
        batch_size=2,
        conditions=1,
        num_inference_steps=6,
        generator=torch.Generator().manual_seed(7),
    ).samples
    assert tuple(restored.model.config.sample_shape) == (5, 3)
    assert restored.scheduler.config.prediction_type == "epsilon"
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("steps", [0, 25])
def test_invalid_sampling_step_counts_are_rejected(steps):
    with pytest.raises(ValueError):
        _small_pipeline()(num_inference_steps=steps)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
@pytest.mark.parametrize("scheduler_kind", ["ddpm", "ddim"])
def test_sparse_sampling_with_oracle_reaches_exact_clean_state(
    monkeypatch, prediction_type, scheduler_kind
):
    pipeline = _small_pipeline((3,), scheduler_kind)
    pipeline.scheduler = make_scheduler(
        scheduler_kind, num_train_timesteps=24, prediction_type=prediction_type
    )

    def oracle(sample, timestep, conditions=None):
        clean = torch.full_like(sample, 0.25)
        a = pipeline.scheduler.alphas_cumprod[timestep].to(sample)
        noise = (sample - a.sqrt() * clean) / (1 - a).sqrt()
        prediction = (
            noise
            if prediction_type == "epsilon"
            else (
                clean if prediction_type == "sample" else a.sqrt() * noise - (1 - a).sqrt() * clean
            )
        )
        return DenoiserOutput(sample=prediction)

    monkeypatch.setattr(pipeline.model, "forward", oracle)
    result = pipeline(
        batch_size=4, num_inference_steps=6, generator=torch.Generator().manual_seed(10)
    )
    torch.testing.assert_close(result.samples, torch.full((4, 3), 0.25), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("eta", [float("nan"), float("inf"), -1.0, 1.1])
def test_invalid_eta_is_rejected(eta):
    with pytest.raises(ValueError):
        _small_pipeline()(batch_size=2, num_inference_steps=3, eta=eta)
