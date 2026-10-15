"""Evaluator smoke/resume tests use synthetic real-data stand-ins, never a download."""

import json

import pytest
import torch

from recheck import mnist as mnist_module
from recheck.data import prepare_mnist
from recheck.mnist import (
    extract_features,
    finalize_evaluator_test,
    fit_feature_statistics,
    train_evaluator,
)
from recheck.models import EvaluatorCNN, state_dict_sha256


class FakeMNIST:
    calls = []

    def __init__(self, root, train=True, download=False):
        type(self).calls.append((train, download))
        generator = torch.Generator().manual_seed(951 if train else 952)
        self.data = torch.randint(256, (100, 28, 28), generator=generator, dtype=torch.uint8)
        self.targets = torch.arange(100) % 10


@pytest.fixture(autouse=True)
def cpu_threads_and_calls():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    FakeMNIST.calls.clear()
    yield
    torch.set_num_threads(previous)


def setup_fake(tmp_path):
    prepare_mnist(
        tmp_path / "data",
        tmp_path / "split",
        n_validation=20,
        expected_train_size=100,
        expected_test_size=100,
        dataset_factory=FakeMNIST,
    )
    return dict(
        data_root=tmp_path / "data",
        split_manifest=tmp_path / "split" / "mnist_split_manifest.json",
        dataset_factory=FakeMNIST,
    )


def tiny_config(**extra):
    return dict(
        epochs=2,
        batch_size=20,
        accuracy_threshold=0.0,
        bandwidth_samples=20,
        feature_batch_size=40,
        checkpoint_every=2,
        **extra,
    )


def test_evaluator_dryrun_is_inert_and_rejects_unknown_options(tmp_path):
    kwargs = dict(
        output_dir=tmp_path, data_root=tmp_path, split_manifest=tmp_path / "does_not_exist.json"
    )
    plan = train_evaluator({}, **kwargs, dry_run=True)
    assert plan["status"] == "dry_run" and not FakeMNIST.calls
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="unknown evaluator"):
        train_evaluator({"epohs": 1}, **kwargs, dry_run=True)


def test_extraction_uses_frozen_fc_features_shared_preprocessing_and_restores_mode():
    evaluator = EvaluatorCNN().train()
    raw = torch.rand(20, 1, 28, 28) * 2 - 1
    raw[0, 0, 0, 0] = 2.0
    before = state_dict_sha256(evaluator)
    features, logits, diagnostic = extract_features(evaluator, raw, batch_size=7)
    assert features.shape == (20, 128) and logits.shape == (20, 10)
    assert not features.requires_grad and not logits.requires_grad
    assert evaluator.training and state_dict_sha256(evaluator) == before
    assert diagnostic["out_of_range_fraction"] == pytest.approx(1 / raw.numel())
    mean, std = features.double().mean(0), features.double().std(0, correction=0).clamp_min(1e-6)
    normalized, _, diagnostic = extract_features(evaluator, raw, 7, mean, std)
    assert torch.allclose(normalized, (features - mean) / std)
    assert diagnostic["features_standardized"]


def test_feature_reference_is_real_train_only_and_keeps_original_indices():
    evaluator = EvaluatorCNN().eval()
    raw = torch.rand(30, 1, 28, 28) * 2 - 1
    original_indices = torch.arange(30) * 2 + 1000
    statistics = fit_feature_statistics(
        evaluator, raw, original_indices, seed=23, bandwidth_samples=20, batch_size=11
    )
    assert statistics["mean"].shape == (128,) and statistics["std"].shape == (128,)
    assert (statistics["std"] >= 1e-6).all()
    assert statistics["bandwidth_squared"] > 0
    assert len(statistics["bandwidth_indices"]) == 20
    assert set(statistics["bandwidth_indices"].tolist()) <= set(original_indices.tolist())
    again = fit_feature_statistics(
        evaluator, raw, original_indices, seed=23, bandwidth_samples=20, batch_size=11
    )
    assert torch.equal(statistics["bandwidth_indices"], again["bandwidth_indices"])
    assert statistics["bandwidth_squared"] == again["bandwidth_squared"]


def test_evaluator_train_selects_real_validation_and_opens_test_only_once(tmp_path):
    kwargs = setup_fake(tmp_path)
    output = tmp_path / "evaluator"
    result = train_evaluator(tiny_config(), output_dir=output, **kwargs)
    assert result["status"] == "completed" and result["evaluator_gate_passed"]
    assert result["test_accuracy"] is None and result["test_accessed"] is False
    assert all(train and not download for train, download in FakeMNIST.calls)
    history = json.loads((output / "history.json").read_text())
    assert result["best_validation_accuracy"] == max(x["validation_accuracy"] for x in history)
    reference = torch.load(output / "feature_reference.pt", weights_only=True)
    split = torch.load(
        kwargs["split_manifest"].parent / "mnist_split_indices.pt", weights_only=True
    )
    assert set(reference["bandwidth_indices"].tolist()) <= set(split["train"].tolist())
    measured = finalize_evaluator_test(output, **kwargs)
    assert 0 <= measured["accuracy"] <= 1 and measured["n_examples"] == 100
    assert FakeMNIST.calls[-1] == (False, False)
    with pytest.raises(ValueError, match="already measured"):
        finalize_evaluator_test(output, **kwargs)
    with pytest.raises(ValueError, match="already accessed test"):
        train_evaluator(tiny_config(), output_dir=output, resume=True, **kwargs)


def test_failed_validation_gate_prevents_test_and_feature_reference(tmp_path):
    kwargs = setup_fake(tmp_path)
    output = tmp_path / "evaluator"
    config = tiny_config()
    config["accuracy_threshold"] = 1.0
    result = train_evaluator(config, output_dir=output, **kwargs)
    assert result["status"] == "failed_gate" and not result["evaluator_gate_passed"]
    assert not (output / "feature_reference.pt").exists()
    with pytest.raises(ValueError, match="validation-qualified"):
        finalize_evaluator_test(output, **kwargs)
    assert all(train for train, _ in FakeMNIST.calls)


def test_interrupted_evaluator_exact_resume_preserves_optimizer_rng_and_best_weights(
    tmp_path, monkeypatch
):
    kwargs = setup_fake(tmp_path)
    config = tiny_config()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full_result = train_evaluator(config, output_dir=full, **kwargs)
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupted_step(optimizer, *args, **kw):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return original_step(optimizer, *args, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        train_evaluator(config, output_dir=resumed, **kwargs)
    interrupted = torch.load(resumed / "evaluator_checkpoint.pt", weights_only=True)
    assert interrupted["status"] == "interrupted" and interrupted["update"] == 2
    assert interrupted["offset"] == 40 and interrupted["order"] is not None
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    resumed_result = train_evaluator(config, output_dir=resumed, resume=True, **kwargs)
    assert full_result["evaluator_sha256"] == resumed_result["evaluator_sha256"]
    full_state = torch.load(full / "evaluator_checkpoint.pt", weights_only=True)
    resumed_state = torch.load(resumed / "evaluator_checkpoint.pt", weights_only=True)
    assert full_state["history"] == resumed_state["history"]
    assert full_state["update"] == resumed_state["update"] == 8
    assert state_dict_sha256(full_state["model"]) == state_dict_sha256(resumed_state["model"])
    assert torch.equal(full_state["order_rng"], resumed_state["order_rng"])
    for full_param, resumed_param in zip(
        full_state["optimizer"]["state"].values(), resumed_state["optimizer"]["state"].values()
    ):
        assert torch.equal(full_param["exp_avg"], resumed_param["exp_avg"])
        assert torch.equal(full_param["exp_avg_sq"], resumed_param["exp_avg_sq"])
    with pytest.raises(ValueError, match="resume contract"):
        train_evaluator(config | {"lr": 2e-3}, output_dir=resumed, resume=True, **kwargs)


def test_budget_callback_at_update_100_has_committed_state_and_exact_resume(tmp_path):
    kwargs = setup_fake(tmp_path)
    config = tiny_config() | {"batch_size": 1, "checkpoint_every": 1000}
    full, interrupted = tmp_path / "full-100", tmp_path / "interrupted-100"
    expected = train_evaluator(config, output_dir=full, **kwargs)
    observed = []

    def stop_at_100():
        state = torch.load(interrupted / "evaluator_checkpoint.pt", weights_only=True)
        observed.append(state["update"])
        if state["update"] >= 100:
            assert state["update"] == 100 and state["offset"] == 20 and state["epoch"] == 1
            raise RuntimeError("registered budget exhausted")

    with pytest.raises(RuntimeError, match="budget exhausted"):
        train_evaluator(config, output_dir=interrupted, progress_callback=stop_at_100, **kwargs)
    saved = torch.load(interrupted / "evaluator_checkpoint.pt", weights_only=True)
    assert saved["status"] == "interrupted" and observed[-1] == 100
    assert json.loads((interrupted / "manifest.json").read_text())["status"] == "interrupted"
    resumed = train_evaluator(config, output_dir=interrupted, resume=True, **kwargs)
    assert expected["evaluator_sha256"] == resumed["evaluator_sha256"]
    reference = torch.load(full / "evaluator_checkpoint.pt", weights_only=True)
    saved = torch.load(interrupted / "evaluator_checkpoint.pt", weights_only=True)
    assert state_dict_sha256(reference["model"]) == state_dict_sha256(saved["model"])
    assert reference["history"] == saved["history"]
    assert torch.equal(reference["order_rng"], saved["order_rng"])
    for a, b in zip(reference["optimizer"]["state"].values(), saved["optimizer"]["state"].values()):
        assert torch.equal(a["exp_avg"], b["exp_avg"])
        assert torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])


def test_budget_stop_during_features_keeps_last_optimizer_model_not_selected_best(
    tmp_path, monkeypatch
):
    kwargs = setup_fake(tmp_path)
    config = tiny_config()
    full, interrupted = tmp_path / "full-features", tmp_path / "interrupted-features"
    validation_calls = 0

    def early_best(model, raw_images, labels, batch_size=512):
        nonlocal validation_calls
        validation_calls += 1
        return {
            "accuracy": 0.4 if validation_calls % 2 else 0.3,
            "cross_entropy": 1.0,
            "n_examples": len(labels),
        }

    monkeypatch.setattr(mnist_module, "classifier_accuracy", early_best)
    expected = train_evaluator(config, output_dir=full, **kwargs)
    reference = torch.load(full / "evaluator_checkpoint.pt", weights_only=True)
    assert expected["best_epoch"] == 1
    assert state_dict_sha256(reference["model"]) != state_dict_sha256(reference["best_state"])
    in_features = False
    original_fit = mnist_module.fit_feature_statistics

    def fitting(*args, **kw):
        nonlocal in_features
        in_features = True
        try:
            return original_fit(*args, **kw)
        finally:
            in_features = False

    def stop_features():
        if in_features:
            state = torch.load(interrupted / "evaluator_checkpoint.pt", weights_only=True)
            assert state["epoch"] == 2 and state["update"] == 8
            assert state_dict_sha256(state["model"]) == state_dict_sha256(reference["model"])
            raise RuntimeError("budget reached during train feature extraction")

    monkeypatch.setattr(mnist_module, "fit_feature_statistics", fitting)
    with pytest.raises(RuntimeError, match="budget reached during"):
        train_evaluator(config, output_dir=interrupted, progress_callback=stop_features, **kwargs)
    saved = torch.load(interrupted / "evaluator_checkpoint.pt", weights_only=True)
    assert saved["status"] == "interrupted"
    assert state_dict_sha256(saved["model"]) == state_dict_sha256(reference["model"])
    actual = train_evaluator(config, output_dir=interrupted, resume=True, **kwargs)
    assert actual["evaluator_sha256"] == expected["evaluator_sha256"]
    expected_stats = torch.load(full / "feature_reference.pt", weights_only=True)
    actual_stats = torch.load(interrupted / "feature_reference.pt", weights_only=True)
    assert torch.equal(expected_stats["mean"], actual_stats["mean"])
    assert torch.equal(expected_stats["std"], actual_stats["std"])
    assert expected_stats["bandwidth_squared"] == actual_stats["bandwidth_squared"]
