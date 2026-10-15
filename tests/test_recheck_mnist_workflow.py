"""Workflow/gate tests with synthetic fixtures; no MNIST download or GPU work."""

import functools
import json
from pathlib import Path

import pytest
import torch

from recheck import mnist_workflow as workflow
from recheck.config import resolve_config
from recheck.data import load_mnist_split, prepare_mnist
from recheck.evaluate import fit_feature_reference
from recheck.models import EvaluatorCNN, state_dict_sha256
from recheck.runtime import atomic_json, file_hash


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FakeMNIST:
    calls = []

    def __init__(self, root, train=True, download=False):
        type(self).calls.append((train, download))
        generator = torch.Generator().manual_seed(681 if train else 682)
        self.data = torch.randint(256, (100, 28, 28), generator=generator, dtype=torch.uint8)
        self.targets = torch.arange(100) % 10


def config_for(tmp_path):
    return resolve_config(
        {
            "output_dir": str(tmp_path / "run"),
            "device": "cpu",
            "data": {
                "name": "mnist",
                "n_train": 80,
                "n_validation": 20,
                "n_test": 100,
                "root": str(tmp_path / "data"),
            },
            "model": {"kind": "unet"},
            "evaluator": {"bandwidth_samples": 20},
            "evaluation": {"batch_size": 20, "mmd_block_size": 20},
            "resources": {"rays_gate": str(tmp_path / "missing_gate.json")},
        }
    )


def frozen_fixture(config, monkeypatch):
    """Synthetic model/artifacts test access control, never represent measured 99% accuracy."""
    FakeMNIST.calls.clear()
    monkeypatch.setattr(workflow, "require_rays_gate", lambda config: None)
    root = Path(config["output_dir"])
    split_dir, evaluator_dir = root / "data", root / "evaluator"
    prepare_mnist(
        config["data"]["root"],
        split_dir,
        n_validation=20,
        expected_train_size=100,
        expected_test_size=100,
        dataset_factory=FakeMNIST,
    )
    split_path = split_dir / "mnist_split_manifest.json"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        evaluator = EvaluatorCNN().eval().requires_grad_(False)
    evaluator.save_pretrained(evaluator_dir / "evaluator")
    load_fake = functools.partial(load_mnist_split, dataset_factory=FakeMNIST)
    train, _ = load_fake(config["data"]["root"], "train", split_path)
    train_features, _, _ = workflow.extract_mnist_features(evaluator, train, batch_size=20)
    split_indices = torch.load(split_dir / "mnist_split_indices.pt", weights_only=True)
    reference = fit_feature_reference(
        train_features, seed=43, bandwidth_samples=20, indices=split_indices["train"]
    )
    torch.save(
        {
            "mean": reference.mean,
            "std": reference.std,
            "bandwidth_squared": reference.ell2,
            "bandwidth_indices": reference.indices,
            "bandwidth_degenerate": False,
        },
        evaluator_dir / "feature_reference.pt",
    )
    manifest = {
        "status": "completed",
        "evaluator_gate_passed": True,
        "best_validation_accuracy": 0.995,
        "test_accuracy": None,
        "test_accessed": False,
        "split_manifest_sha256": file_hash(split_path),
        "evaluator_sha256": state_dict_sha256(evaluator),
        "feature_reference_sha256": file_hash(evaluator_dir / "feature_reference.pt"),
    }
    atomic_json(evaluator_dir / "manifest.json", manifest)
    monkeypatch.setattr(workflow, "load_mnist_split", load_fake)
    FakeMNIST.calls.clear()
    return evaluator_dir, split_path, load_fake


def test_missing_rays_gate_prevents_every_dataset_factory_or_trainer(tmp_path, monkeypatch):
    config = config_for(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("MNIST factory/trainer ran before rays gate")

    monkeypatch.setattr(workflow, "prepare_mnist", forbidden)
    monkeypatch.setattr(workflow, "train_evaluator", forbidden)
    monkeypatch.setattr(workflow, "load_mnist_split", forbidden)
    for action in (
        workflow.prepare_stage,
        workflow.evaluator_stage,
        workflow.build_quality_context,
    ):
        with pytest.raises(RuntimeError, match="MNIST is gated"):
            action(config)
    assert not Path(config["output_dir"]).exists()


def test_dryruns_are_inert_without_gate_data_or_accelerator(tmp_path, monkeypatch):
    config = config_for(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run touched gate/data/device/budget")

    monkeypatch.setattr(workflow, "require_rays_gate", forbidden)
    monkeypatch.setattr(workflow, "prepare_mnist", forbidden)
    monkeypatch.setattr(workflow, "choose_device", forbidden)
    monkeypatch.setattr(workflow, "ResourceBudget", forbidden)
    prepared = workflow.prepare_stage(config, dry_run=True)
    evaluated = workflow.evaluator_stage(config, dry_run=True)
    assert prepared["status"] == evaluated["status"] == "dry_run"
    assert prepared["manifest_path"].endswith("data/mnist_split_manifest.json")
    json.dumps(prepared)
    json.dumps(evaluated)
    assert not Path(config["output_dir"]).exists()


def test_prepare_stage_gate_precedes_factory_and_persists_actual_manifest(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    events = []
    monkeypatch.setattr(workflow, "require_rays_gate", lambda config: events.append("gate"))

    def prepare_fake(*args, **kwargs):
        events.append("factory")
        assert kwargs["download"] is True
        return prepare_mnist(*args, **kwargs, dataset_factory=FakeMNIST)

    monkeypatch.setattr(workflow, "prepare_mnist", prepare_fake)
    result = workflow.prepare_stage(config)
    assert events == ["gate", "factory"]
    assert result["status"] == "completed"
    assert Path(result["manifest_path"]).is_file() and Path(result["indices_path"]).is_file()
    assert result["manifest"]["train_size"] == 80 and not result["test_accessed"]
    json.dumps(result)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluator_wrapper_uses_one_root_budget_and_always_checks_closes(
    tmp_path, monkeypatch, fail
):
    config = config_for(tmp_path)
    events = []
    monkeypatch.setattr(workflow, "require_rays_gate", lambda config: events.append("gate"))

    class Budget:
        def __init__(self, actual_config, device):
            assert actual_config["output_dir"] == config["output_dir"]
            assert str(device) == "cpu"
            events.append("budget")

        def check(self):
            events.append("check")

        def close(self):
            events.append("close")

    def train_fake(actual_config, **kwargs):
        events.append("train")
        assert callable(kwargs["progress_callback"])
        assert actual_config["device"] == "cpu" and kwargs["resume"] is True
        assert str(kwargs["output_dir"]).endswith("run/evaluator")
        assert str(kwargs["split_manifest"]).endswith("data/mnist_split_manifest.json")
        if fail:
            raise KeyboardInterrupt()
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "ResourceBudget", Budget)
    monkeypatch.setattr(workflow, "train_evaluator", train_fake)
    if fail:
        with pytest.raises(KeyboardInterrupt):
            workflow.evaluator_stage(config, resume=True)
    else:
        result = workflow.evaluator_stage(config, resume=True)
        assert result["resolved_device"] == "cpu"
        json.dumps(result)
    assert events == ["gate", "budget", "train", "check", "close"]


def test_validation_context_uses_train_only_reference_and_never_test(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    evaluator_dir, _, _ = frozen_fixture(config, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("validation opened classifier test")

    monkeypatch.setattr(workflow, "finalize_evaluator_test", forbidden)
    context = workflow.build_quality_context(config, split="validation")
    assert context["train_features"].shape == (80, 128)
    assert context["real_features"].shape == (20, 128)
    assert context["real_logits"].shape == (20, 10)
    assert context["real_samples"].shape == (20, 1, 28, 28)
    assert all(train for train, _ in FakeMNIST.calls)
    assert not context["evaluator"].training
    assert all(not p.requires_grad for p in context["evaluator"].parameters())
    reference = torch.load(evaluator_dir / "feature_reference.pt", weights_only=True)
    assert torch.equal(context["reference"].mean, reference["mean"])
    assert torch.equal(context["reference"].std, reference["std"])
    json.dumps(context["metadata"])
    raw = context["real_samples"][:10].clone()
    raw[0, 0, 0, 0] = 2.0
    quality = workflow.quality(config, raw, context)
    assert torch.isfinite(torch.tensor(quality["mmd2_mnist_features_real"]))
    assert quality["out_of_range_fraction"] == pytest.approx(1 / raw.numel())
    assert quality["max_range_violation"] == pytest.approx(0.5)
    assert quality["n_real"] == 20 and quality["n_generated"] == 10


def test_test_context_records_accuracy_once_then_reuses_it(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    evaluator_dir, _, _ = frozen_fixture(config, monkeypatch)
    calls = 0

    def record_once(directory, **kwargs):
        nonlocal calls
        calls += 1
        manifest_path = Path(directory) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(test_accuracy=0.987, test_accessed=True)
        atomic_json(manifest_path, manifest)
        return {"accuracy": 0.987}

    monkeypatch.setattr(workflow, "finalize_evaluator_test", record_once)
    first = workflow.build_quality_context(config, split="test")
    again = workflow.build_quality_context(config, split="test")
    assert calls == 1 and first["test_accuracy"] == again["test_accuracy"] == 0.987
    assert first["metadata"]["classifier_test_measurement_created_here"]
    assert not again["metadata"]["classifier_test_measurement_created_here"]
    assert first["real_features"].shape == (100, 128)
    assert first["metadata"]["reference_split"] == "test"
    assert json.loads((evaluator_dir / "manifest.json").read_text())["test_accessed"]


@pytest.mark.parametrize("change", ["threshold", "weights", "reference", "test_record"])
def test_context_refuses_unqualified_or_changed_artifacts_before_data(
    tmp_path, monkeypatch, change
):
    config = config_for(tmp_path)
    evaluator_dir, _, _ = frozen_fixture(config, monkeypatch)
    path = evaluator_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "threshold":
        manifest["best_validation_accuracy"] = 0.98
    elif change == "weights":
        manifest["evaluator_sha256"] = "invalid"
    elif change == "reference":
        manifest["feature_reference_sha256"] = "invalid"
    elif change == "test_record":
        manifest["test_accessed"] = True
    atomic_json(path, manifest)
    FakeMNIST.calls.clear()
    with pytest.raises((ValueError, RuntimeError)):
        workflow.build_quality_context(config, split="test")
    assert FakeMNIST.calls == []


def test_context_rejects_bandwidth_calibration_on_validation_indices(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    evaluator_dir, split_path, _ = frozen_fixture(config, monkeypatch)
    reference_path = evaluator_dir / "feature_reference.pt"
    reference = torch.load(reference_path, weights_only=True)
    indices = torch.load(split_path.parent / "mnist_split_indices.pt", weights_only=True)
    reference["bandwidth_indices"][0] = indices["validation"][0]
    torch.save(reference, reference_path)
    manifest_path = evaluator_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_reference_sha256"] = file_hash(reference_path)
    atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="registered train indices"):
        workflow.build_quality_context(config)
    assert FakeMNIST.calls == []
