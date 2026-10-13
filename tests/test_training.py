"""Training targets, EMA, and reproducible epoch-boundary resume."""

import copy
import math

import pytest
import torch

from diffusion_lab.environments.base import SampleDataset
from diffusion_lab.losses import LOSSES
from diffusion_lab.models import PointDenoiser
from diffusion_lab.objectives import DiffusionObjective, RectifiedFlowObjective
from diffusion_lab.pipeline import ResearchDiffusionPipeline
from diffusion_lab.schedulers import make_scheduler
from diffusion_lab.training import DiffusionTrainer, TrainerConfig


def _pipeline(sample_shape=(2,), kind="ddim", prediction_type="epsilon"):
    model = PointDenoiser(sample_shape=sample_shape, hidden_dim=16, num_layers=1, num_classes=3)
    return ResearchDiffusionPipeline(
        model, make_scheduler(kind, num_train_timesteps=24, prediction_type=prediction_type)
    )


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_diffusion_objective_perfect_targets_recover_clean_and_min_snr(prediction_type):
    scheduler = make_scheduler("ddim", num_train_timesteps=24, prediction_type=prediction_type)
    objective = DiffusionObjective(scheduler, snr_gamma=5.0)
    clean = torch.randn(16, 6, 3)
    pair = objective.prepare(clean, torch.Generator().manual_seed(12))
    restored = objective.clean_prediction(pair, pair.target)
    torch.testing.assert_close(restored, clean, atol=2e-5, rtol=2e-5)
    a = scheduler.alphas_cumprod[pair.timesteps]
    snr = a / (1 - a)
    clipped = snr.clamp(max=5.0)
    expected = (
        clipped / snr
        if prediction_type == "epsilon"
        else (clipped / (snr + 1) if prediction_type == "v_prediction" else clipped)
    )
    torch.testing.assert_close(pair.weights, expected)
    assert torch.isfinite(pair.weights).all()
    assert torch.all(pair.weights > 0)


def test_rectified_flow_objective_matches_linear_bridge():
    objective = RectifiedFlowObjective(make_scheduler("flow", num_train_timesteps=24))
    clean = torch.randn(9, 5, 2)
    pair = objective.prepare(clean, torch.Generator().manual_seed(17))
    sigma = pair.timesteps.reshape(-1, 1, 1) / 24
    implied_noise = clean + pair.target
    torch.testing.assert_close(pair.noisy, (1 - sigma) * clean + sigma * implied_noise)
    torch.testing.assert_close(objective.clean_prediction(pair, pair.target), clean)
    assert torch.all((pair.timesteps >= 0) & (pair.timesteps < 24))
    torch.testing.assert_close(pair.weights, torch.ones(9))


@pytest.mark.parametrize(
    "kind,objective,prediction_type",
    [
        ("ddim", "diffusion", "epsilon"),
        ("ddim", "diffusion", "sample"),
        ("ddim", "diffusion", "v_prediction"),
        ("flow", "rectified_flow", "epsilon"),
    ],
)
@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_train_step_changes_parameters_and_ema_tracks_update(
    kind, objective, prediction_type, loss
):
    pipeline = _pipeline((4, 3), kind, prediction_type)
    trainer = DiffusionTrainer(
        pipeline, TrainerConfig(epochs=1, objective=objective, ema_decay=0.5, loss=loss)
    )
    old = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    metrics = trainer.train_step(
        {"samples": torch.randn(5, 4, 3), "conditions": torch.tensor([0, 1, 2, 0, 1])}
    )
    assert trainer.global_step == 1
    assert all(math.isfinite(metrics[name]) for name in ("loss", "regression", "grad_norm"))
    assert any(not torch.equal(old[name], p) for name, p in trainer.model.named_parameters())
    current = dict(trainer.model.named_parameters())
    for name, p in trainer.ema.named_parameters():
        torch.testing.assert_close(p, 0.5 * old[name] + 0.5 * current[name])
    assert all(not p.requires_grad for p in trainer.ema.parameters())


def test_condition_dropout_one_really_removes_the_condition():
    pipeline = _pipeline()
    trainer = DiffusionTrainer(
        pipeline, TrainerConfig(epochs=1, condition_dropout=1, ema_decay=None, weight_decay=0)
    )
    trainer.train_step({"samples": torch.randn(5, 2), "conditions": torch.tensor([0, 1, 2, 0, 1])})
    assert trainer.model.null_condition.grad is not None
    assert torch.linalg.vector_norm(trainer.model.null_condition.grad) > 0
    for p in trainer.model.condition_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_ema_context_restores_live_model_even_when_callback_fails():
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1))
    live = trainer.model
    with pytest.raises(RuntimeError, match="intentional"):
        with trainer.ema_weights():
            assert trainer.pipeline.model is trainer.ema
            raise RuntimeError("intentional")
    assert trainer.pipeline.model is live


def test_epoch_boundary_resume_matches_uninterrupted_training_exactly(tmp_path):
    torch.manual_seed(1)
    original = _pipeline()
    dataset = SampleDataset(
        torch.randn(11, 2), torch.arange(11) % 3, metadata={"name": "resume-regression"}
    )
    config = dict(batch_size=4, seed=12, learning_rate=1e-3, condition_dropout=0.3, ema_decay=0.8)
    baseline = DiffusionTrainer(copy.deepcopy(original), TrainerConfig(epochs=2, **config))
    baseline.fit(dataset)
    split = DiffusionTrainer(copy.deepcopy(original), TrainerConfig(epochs=1, **config))
    split.fit(dataset)
    checkpoint = tmp_path / "checkpoint.pt"
    split.save_checkpoint(checkpoint)
    resumed = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=2, **config))
    resumed.load_checkpoint(checkpoint)
    assert resumed.completed_epochs == 1
    assert resumed.global_step == 3
    resumed.fit(dataset)
    assert resumed.completed_epochs == 2
    assert resumed.global_step == baseline.global_step == 6
    assert resumed.history == baseline.history
    assert resumed.metadata == dataset.metadata
    for name, p in baseline.model.state_dict().items():
        torch.testing.assert_close(p, resumed.model.state_dict()[name], atol=0, rtol=0)
    for name, p in baseline.ema.state_dict().items():
        torch.testing.assert_close(p, resumed.ema.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(resumed.generator.get_state(), baseline.generator.get_state())
    torch.testing.assert_close(
        resumed.loader_generator.get_state(), baseline.loader_generator.get_state()
    )


def test_resume_rejects_silent_training_recipe_change(tmp_path):
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1))
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    changed = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=2, learning_rate=0.01))
    with pytest.raises(ValueError, match="only epochs"):
        changed.load_checkpoint(checkpoint)


@pytest.mark.parametrize("scheduler_change", ["prediction_type", "num_train_timesteps"])
def test_resume_rejects_incompatible_diffusion_scheduler(tmp_path, scheduler_change):
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1))
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    changed_pipeline = _pipeline(
        prediction_type="sample" if scheduler_change == "prediction_type" else "epsilon"
    )
    if scheduler_change == "num_train_timesteps":
        changed_pipeline.scheduler = make_scheduler("ddim", num_train_timesteps=30)
    changed = DiffusionTrainer(changed_pipeline, TrainerConfig(epochs=2))
    with pytest.raises(ValueError):
        changed.load_checkpoint(checkpoint)


def test_resumed_fit_rejects_changed_dataset(tmp_path):
    dataset = SampleDataset(torch.randn(4, 2), torch.tensor([0, 1, 2, 0]))
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1, batch_size=4))
    trainer.fit(dataset)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    resumed = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=2, batch_size=4))
    resumed.load_checkpoint(checkpoint)
    changed = SampleDataset(dataset.samples + 0.01, dataset.conditions)
    with pytest.raises(ValueError, match="Dataset|dataset"):
        resumed.fit(changed)
    assert resumed.global_step == 1


def test_custom_loss_contract_rejects_scalar_reduction(monkeypatch):
    monkeypatch.setitem(
        LOSSES, "bad_scalar", lambda prediction, target: (prediction - target).square().mean()
    )
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1, loss="bad_scalar"))
    with pytest.raises(ValueError, match="one value per example"):
        trainer.train_step({"samples": torch.randn(3, 2)})


def test_nonfinite_training_data_is_rejected_before_optimizer_update():
    trainer = DiffusionTrainer(_pipeline(), TrainerConfig(epochs=1))
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    with pytest.raises(FloatingPointError):
        trainer.train_step({"samples": torch.full((3, 2), float("nan"))})
    assert trainer.global_step == 0
    for name, p in trainer.model.named_parameters():
        torch.testing.assert_close(p, before[name])
