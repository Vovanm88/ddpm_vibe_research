"""Notebook workflows: repeated training, seeded sampling, and shape inference."""

import copy

import pytest
import torch

from diffusion_lab import ResearchExperiment, create_experiment
from diffusion_lab.environments import SampleDataset, WallNavigation


def _experiment(environment="rotating_rays", scheduler=None, training=None, **kwargs):
    return create_experiment(
        environment,
        n_samples=16,
        model={"hidden_dim": 16, "num_layers": 1},
        scheduler=scheduler or {"kind": "ddim", "num_train_timesteps": 16},
        training={"epochs": 1, "batch_size": 8, **(training or {})},
        device="cpu",
        seed=17,
        **kwargs,
    )


def test_notebook_can_train_more_then_generate_repeatably():
    experiment = _experiment()
    live_model = experiment.pipeline.model
    initial_parameters = {
        name: value.detach().clone() for name, value in live_model.named_parameters()
    }
    experiment.fit(epochs=1)
    experiment.fit(epochs=2)
    assert experiment.trainer.completed_epochs == 3
    assert len(experiment.trainer.history) == 3
    assert experiment.trainer.global_step == 6
    assert any(
        not torch.equal(initial_parameters[name], value)
        for name, value in live_model.named_parameters()
    )
    first = experiment.sample(7, condition=1, steps=4, seed=22, record_history=True)
    second = experiment.sample(7, condition=1, steps=4, seed=22)
    assert first.samples.shape == (7, 2)
    assert first.history.shape == (5, 7, 2)
    assert torch.isfinite(first.samples).all()
    torch.testing.assert_close(first.samples, second.samples, atol=0, rtol=0)
    assert experiment.pipeline.model is live_model


@pytest.mark.parametrize("environment,shape", [("gaussian_mixture_1d", (1,)), ("helix_3d", (3,))])
def test_notebook_infers_point_dimension(environment, shape):
    experiment = _experiment(environment)
    assert experiment.dataset.sample_shape == experiment.pipeline.model.sample_shape == shape
    experiment.fit(epochs=1)
    output = experiment.sample(4, steps=4, seed=2)
    assert output.samples.shape == (4, *shape)
    assert torch.isfinite(output.samples).all()


def test_rectified_flow_default_matches_selected_scheduler():
    experiment = _experiment(scheduler={"kind": "flow", "num_train_timesteps": 16})
    assert experiment.trainer.config.objective == "rectified_flow"
    experiment.fit(epochs=1)
    output = experiment.sample(4, steps=4, seed=22)
    assert output.samples.shape == (4, 2)
    assert torch.isfinite(output.samples).all()
    experiment.fit(epochs=1)
    assert experiment.trainer.completed_epochs == 2


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_sampling_preserves_training_scheduler_state_and_live_parameters(kind):
    experiment = _experiment(scheduler={"kind": kind, "num_train_timesteps": 16})
    experiment.fit(epochs=1)
    scheduler = experiment.pipeline.scheduler
    runtime_state = copy.deepcopy(scheduler.__dict__)
    live_model = experiment.pipeline.model
    parameters = {name: p.detach().clone() for name, p in live_model.named_parameters()}
    experiment.sample(4, steps=4, seed=22)
    assert experiment.pipeline.scheduler is scheduler
    assert experiment.trainer.objective.scheduler is scheduler
    assert scheduler.__dict__.keys() == runtime_state.keys()
    for name, previous in runtime_state.items():
        current = scheduler.__dict__[name]
        if isinstance(previous, torch.Tensor):
            torch.testing.assert_close(current, previous, atol=0, rtol=0)
        else:
            assert current == previous
    assert experiment.pipeline.model is live_model
    for name, parameter in live_model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], atol=0, rtol=0)


def test_navigation_single_condition_broadcasts_and_pins_endpoints():
    experiment = _experiment("navigation_1d")
    condition = experiment.dataset.conditions[0]
    output = experiment.sample(3, condition=condition, steps=4, seed=22)
    assert output.samples.shape == (3, *experiment.dataset.sample_shape)
    torch.testing.assert_close(output.conditions, condition.expand(3, -1))
    torch.testing.assert_close(output.samples[:, 0], condition[:1].expand(3, 1))
    torch.testing.assert_close(output.samples[:, -1], condition[1:2].expand(3, 1))


def test_default_sampling_reports_seeded_dataset_conditions_and_can_be_unconditional():
    experiment = _experiment()
    first = experiment.sample(32, steps=4, seed=22)
    second = experiment.sample(32, steps=4, seed=22)
    assert first.conditions.shape == (32,)
    assert first.conditions.dtype == torch.long
    assert len(first.conditions.unique()) > 1
    assert set(first.conditions.tolist()) <= set(experiment.dataset.conditions.tolist())
    torch.testing.assert_close(first.conditions, second.conditions, atol=0, rtol=0)
    torch.testing.assert_close(first.samples, second.samples, atol=0, rtol=0)
    unconditional = experiment.sample(4, steps=4, seed=22, unconditional=True)
    assert unconditional.conditions is None
    assert unconditional.samples.shape == (4, 2)


def test_default_navigation_sampling_pins_each_returned_condition():
    experiment = _experiment("navigation_1d")
    output = experiment.sample(5, steps=4, seed=22)
    assert output.conditions.shape == (5, 2)
    torch.testing.assert_close(output.samples[:, 0], output.conditions[:, :1])
    torch.testing.assert_close(output.samples[:, -1], output.conditions[:, 1:2])


def test_custom_trajectory_dataset_does_not_infer_navigation_from_shape():
    dataset = SampleDataset(
        torch.zeros(16, 4, 2),
        torch.tensor([[2.0, 3.0, 4.0, 5.0]]).expand(16, -1),
        metadata={"name": "custom-trajectory"},
    )
    experiment = _experiment(dataset=dataset)
    assert experiment.pipeline.model.sample_shape == (4, 2)
    assert experiment.pipeline.model.config.conditioner == "mlp"
    output = experiment.sample(3, condition=dataset.conditions[0], steps=4, seed=22)
    assert output.samples.shape == (3, 4, 2)
    assert not torch.equal(output.samples[:, 0], output.conditions[:, :2])
    assert not torch.equal(output.samples[:, -1], output.conditions[:, 2:4])


def test_failed_sampling_restores_live_model_and_scheduler():
    experiment = _experiment()
    model, scheduler = experiment.pipeline.model, experiment.pipeline.scheduler
    with pytest.raises(ValueError):
        experiment.sample(3, condition=1, steps=17, seed=22)
    assert experiment.pipeline.model is model
    assert experiment.pipeline.scheduler is scheduler
    experiment.fit(epochs=1)
    assert experiment.trainer.completed_epochs == 1


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_notebook_save_after_sampling_resumes_training_exactly(kind, tmp_path):
    scheduler = {"kind": kind, "num_train_timesteps": 16}
    uninterrupted = _experiment(scheduler=scheduler)
    uninterrupted.fit(epochs=2)

    split = _experiment(scheduler=scheduler)
    split.fit(epochs=1)
    before_save = split.sample(4, condition=1, steps=4, seed=22)
    split.save(tmp_path / "experiment")
    resumed = ResearchExperiment.load(tmp_path / "experiment", device="cpu")
    after_load = resumed.sample(4, condition=1, steps=4, seed=22)
    torch.testing.assert_close(before_save.samples, after_load.samples, atol=0, rtol=0)
    resumed.fit(epochs=1)
    assert resumed.trainer.completed_epochs == uninterrupted.trainer.completed_epochs == 2
    assert resumed.trainer.history == uninterrupted.trainer.history
    for name, parameter in uninterrupted.trainer.model.state_dict().items():
        torch.testing.assert_close(
            parameter, resumed.trainer.model.state_dict()[name], atol=0, rtol=0
        )


def test_saved_custom_navigation_scene_restores_geometry_and_conditions(tmp_path):
    scene = WallNavigation(
        dim=2,
        walls=[[[-0.1, -1.0], [0.1, 0.15]]],
        path_length=8,
        grid_resolution=15,
        margin=0.01,
    )
    dataset = scene.sample_dataset(n_samples=6, seed=17)
    experiment = _experiment(dataset=dataset)
    experiment.fit(epochs=1)
    experiment.save(tmp_path / "navigation")
    resumed = ResearchExperiment.load(tmp_path / "navigation", device="cpu")
    assert resumed.is_navigation
    assert resumed.dataset.metadata == dataset.metadata
    assert resumed.dataset.environment.path_length == scene.path_length
    assert resumed.dataset.environment.margin == scene.margin
    torch.testing.assert_close(resumed.dataset.samples, dataset.samples, atol=0, rtol=0)
    torch.testing.assert_close(resumed.dataset.conditions, dataset.conditions, atol=0, rtol=0)
    condition = dataset.conditions[0]
    original = experiment.sample(3, condition=condition, steps=4, seed=22)
    restored = resumed.sample(3, condition=condition, steps=4, seed=22)
    torch.testing.assert_close(original.samples, restored.samples, atol=0, rtol=0)
    torch.testing.assert_close(restored.samples[:, 0], restored.conditions[:, :2])
    torch.testing.assert_close(restored.samples[:, -1], restored.conditions[:, 2:4])
    assert resumed.evaluate(restored)["endpoint_error"] == 0


def test_unconditional_navigation_has_geometric_metrics_without_target_endpoint():
    experiment = _experiment("navigation_1d")
    output = experiment.sample(3, steps=4, seed=22, unconditional=True)
    metrics = experiment.evaluate(output)
    assert 0 <= metrics["collision_free_fraction"] <= 1
    assert metrics["mean_path_length"] >= 0
    assert "endpoint_error" not in metrics


def test_wrapper_does_not_mutate_callers_configuration():
    environment = {"name": "rotating_rays", "n_classes": 3}
    model = {"hidden_dim": 16, "num_layers": 1}
    scheduler = {"kind": "flow", "num_train_timesteps": 16}
    training = {"epochs": 1, "batch_size": 8}
    originals = copy.deepcopy((environment, model, scheduler, training))
    create_experiment(
        environment,
        n_samples=16,
        model=model,
        scheduler=scheduler,
        training=training,
        device="cpu",
    )
    assert (environment, model, scheduler, training) == originals
