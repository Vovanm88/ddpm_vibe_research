"""Bounded-memory final evaluation and validation-only teacher gates."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from recheck import runtime, workflow
from recheck.config import config_hash, method_name, resolve_config


class TinyVelocity(torch.nn.Module):
    sample_shape = (1, 28, 28)

    def __init__(self, value=0.02):
        super().__init__()
        self.velocity = torch.nn.Parameter(torch.tensor(value))

    def forward(self, sample, timestep):
        return SimpleNamespace(sample=sample * 0 + self.velocity)


class TinyEvaluator(torch.nn.Module):
    def forward(self, pixel, return_features=False):
        mean = pixel.flatten(1).mean(1)
        features = torch.stack((mean, mean.square()), 1)
        logits = torch.stack([mean * index for index in range(10)], 1)
        return (logits, features) if return_features else logits


class Budget:
    instances = []

    def __init__(self, *args):
        self.closed = False
        self.instances.append(self)

    def check(self):
        pass

    def close(self):
        self.closed = True


def _setup(monkeypatch, tmp_path):
    config = resolve_config(
        {
            "output_dir": str(tmp_path / "run"),
            "data": {"name": "mnist", "n_train": 4, "n_validation": 4, "n_test": 4},
            "model": {"kind": "unet"},
            "student": {"updates": 2},
            "distillation": {"teacher_steps": 4, "student_steps": 2, "seeds": [101]},
            "evaluation": {
                "samples": 4,
                "validation_samples": 4,
                "teacher_nfe": [2, 4, 128],
                "batch_size": 2,
                "gif_particles": 2,
                "diagnostic_samples": 2,
                "timing_repeats": 1,
            },
        }
    )
    root = tmp_path / "run"
    (root / "teacher").mkdir(parents=True)
    teacher = TinyVelocity().eval().requires_grad_(False)
    digest = runtime.model_hash(teacher)
    teacher_manifest = {"teacher_sha256": digest, "status": "completed", "training_seconds": 1}
    monkeypatch.setattr(workflow, "_teacher", lambda config: (teacher, teacher_manifest))
    monkeypatch.setattr(workflow, "require_rays_gate", lambda config: None)
    monkeypatch.setattr(runtime, "ResourceBudget", Budget)
    models = {}
    for loss in config["distillation"]["losses"]:
        for rho in config["distillation"]["tube_rhos"]:
            name = method_name(loss, rho)
            path = root / "students" / f"seed101_{name}"
            path.mkdir(parents=True)
            models[name] = TinyVelocity(0.015)
            manifest = {
                "status": "completed",
                "updates_completed": 2,
                "teacher_sha256": digest,
                "student_init_sha256": digest,
                "student_ema_sha256": runtime.model_hash(models[name]),
                "config_sha256": config_hash(config),
                "run_id": f"seed101_{name}",
                "training_seconds": 0.2,
            }
            (path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        workflow,
        "load_model",
        lambda path, device: copy.deepcopy(models[path.parent.name.split("_", 1)[1]]),
    )
    real = torch.linspace(-1, 1, 4 * 28 * 28).reshape(4, 1, 28, 28)
    context = {
        "real_samples": real,
        "evaluator": TinyEvaluator(),
        "metadata": {"reference_split": "test"},
    }
    contexts = []
    monkeypatch.setattr(
        workflow, "_context", lambda config, split: contexts.append(split) or context
    )
    monkeypatch.setattr(
        workflow,
        "_quality",
        lambda config, samples, context: {
            "mmd2_mnist_features_real": 0.001,
            "digit_frequency_tv_real": 0.0,
        },
    )
    monkeypatch.setattr(
        workflow,
        "timed_generation",
        lambda *args, **kwargs: {
            "median_seconds": 0.001,
            "p95_seconds": 0.001,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
    )
    monkeypatch.setattr(
        workflow,
        "local_rollout_diagnostics",
        lambda *args, **kwargs: {"local_error": [], "rollout_error": []},
    )
    monkeypatch.setattr(
        workflow, "local_map_jacobian_diagnostics", lambda *args, **kwargs: {"rows": []}
    )
    return config, root, teacher, context, contexts


def test_fidelity_chunks_ssim_and_preserves_exact_json_infinity(monkeypatch):
    x = torch.linspace(-2, 2, 5 * 28 * 28).reshape(5, 1, 28, 28)
    original = workflow.paired_fidelity
    sizes = []

    def bounded(first, second, **kwargs):
        sizes.append(len(first))
        return original(first, second, **kwargs)

    monkeypatch.setattr(workflow, "paired_fidelity", bounded)
    generated, teacher = {"samples": x, "history": None}, {"samples": x.clone(), "history": None}
    logits = torch.zeros((5, 10))
    result = workflow._fidelity(
        generated,
        teacher,
        [1, 0],
        [1, 0],
        batch_size=2,
        student_logits=logits,
        teacher_logits=logits,
    )
    assert sizes == [2, 2, 1]
    assert result["paired_endpoint_mse"] == 0
    assert result["pixel_ssim"] == pytest.approx(1)
    assert result["psnr_db"] == "+inf"
    assert result["class_agreement"] == 1
    json.dumps(result, allow_nan=False)


def test_fidelity_streams_common_nodes_and_rejects_preview_as_full_bank():
    samples = torch.ones((5, 2))
    full = torch.stack([samples * index for index in range(5)])
    short = full[::2].clone()
    short[1] += 0.2
    result = workflow._fidelity(
        {"samples": samples, "history": short},
        {"samples": samples, "history": full},
        [1, 0.5, 0],
        [1, 0.75, 0.5, 0.25, 0],
        batch_size=2,
    )
    assert result["rollout_mse_per_node"] == pytest.approx([0, 0.04, 0])
    preview = workflow._fidelity(
        {"samples": samples, "history": short[:, :2]},
        {"samples": samples, "history": full},
        [1, 0.5, 0],
        [1, 0.75, 0.5, 0.25, 0],
    )
    assert "rollout_mse_per_node" not in preview
    with pytest.raises(FloatingPointError):
        bad = {"samples": samples * float("nan"), "history": None}
        workflow._fidelity(bad, bad, [1, 0], [1, 0])


def test_mnist_final_evaluation_does_not_record_n128_history_and_keeps_only_preview(
    monkeypatch, tmp_path
):
    config, root, _, _, contexts = _setup(monkeypatch, tmp_path)
    sample = workflow.sample_euler
    calls = []

    def spy(model, initial, times=None, **kwargs):
        calls.append(
            {
                "steps": len(times) - 1,
                "n": len(initial),
                "history": kwargs.get("record_history", True),
            }
        )
        return sample(model, initial, times=times, **kwargs)

    monkeypatch.setattr(workflow, "sample_euler", spy)
    result = workflow.evaluate_suite(config)
    assert result["status"] == "completed" and result["rows"] == 7
    assert contexts == ["test"]
    assert all(not call["history"] for call in calls if call["steps"] == 128)
    assert [call for call in calls if call["n"] == 4] == [{"steps": 4, "n": 4, "history": True}]
    saved = json.loads((root / "evaluation/results.json").read_text())
    assert saved["metadata"]["status"] == "completed"
    assert saved["metadata"]["evaluation_sha256"]
    assert saved["metadata"]["quality_source_sha256"]
    assert all(row["fidelity"]["class_agreement"] == 1 for row in saved["rows"])
    main = next(row for row in saved["rows"] if row["method"] == "Original N4")
    assert main["fidelity"]["psnr_db"] == "+inf"
    short = next(row for row in saved["rows"] if row["method"] == "M0")
    assert len(short["fidelity"]["rollout_mse_per_node"]) == 3
    assert "all paired" in short["fidelity"]["rollout_scope"]
    previews = torch.load(root / "evaluation/trajectories.pt", weights_only=True)
    assert set(previews) == {"Original N4", "Original N2", "M0", "M1", "E0", "E1"}
    assert all(value.shape[1] == 2 for value in previews.values())
    datasets = torch.load(root / "datasets.pt", weights_only=True)
    assert datasets["preview_only"] and "train" not in datasets
    assert Budget.instances[-1].closed


def test_unfinished_or_wrong_budget_student_keeps_final_test_closed(monkeypatch, tmp_path):
    config, root, _, _, contexts = _setup(monkeypatch, tmp_path)
    path = root / "students/seed101_E1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["updates_completed"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="final test remains closed"):
        workflow.evaluate_suite(config)
    assert contexts == []


def test_export_hash_failure_keeps_final_test_closed(monkeypatch, tmp_path):
    config, root, _, _, contexts = _setup(monkeypatch, tmp_path)
    path = root / "students/seed101_E1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["student_ema_sha256"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="final test remains closed"):
        workflow.evaluate_suite(config)
    assert contexts == []


def test_dry_run_does_not_access_datasets_models_or_test(monkeypatch, tmp_path):
    config, _, _, _, contexts = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(workflow, "_teacher", lambda config: pytest.fail("dry run loaded a model"))
    assert workflow.evaluate_suite(config, dry_run=True)["split"] == "test"
    assert workflow.teacher_gate(config, dry_run=True)["split"] == "validation"
    assert contexts == []


def test_teacher_gate_is_validation_only_reports_numerical_stabilization(monkeypatch, tmp_path):
    config, root, _, context, contexts = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        workflow,
        "_quality",
        lambda config, samples, context: {
            "mmd2_mnist_features_real": float(samples.mean()),
            "predicted_digit_frequencies": [0.1] * 10,
            "mean_confidence": 0.9,
        },
    )
    rng_state = torch.random.get_rng_state().clone()
    result = workflow.teacher_gate(config)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert result["passed"]
    assert contexts == ["validation"]
    assert result["test_accessed"] is False
    assert result["stabilization"]["converging"]
    assert result["recognizability_and_coverage"]["predicted_digit_count"] == 10
    assert (root / "teacher/validation_gate_previews.pt").is_file()
    cfm = result["validation_cfm"]
    data_generator = torch.Generator().manual_seed(cfm["rng_seeds"]["data"])
    indices = torch.randperm(4, generator=data_generator)
    data = context["real_samples"][indices]
    noise = torch.randn(
        data.shape, generator=torch.Generator().manual_seed(cfm["rng_seeds"]["noise"])
    )
    target = noise - data
    prediction = torch.tensor(0.02).double()
    expected = float((target.double() - prediction).square().mean())
    assert cfm["raw_cfm_mse"] == pytest.approx(expected, abs=1e-14)
    assert cfm["samples"] == 4 and cfm["teacher_forward_calls"] == 1
    assert set(cfm["rng_seeds"].values()).isdisjoint(
        {config["evaluation"]["final_seed"], config["evaluation"]["validation_seed"]}
    )


def test_nonfinite_quality_is_rejected_before_json(monkeypatch):
    from recheck import mnist_workflow

    config = resolve_config({"data": {"name": "mnist"}, "model": {"kind": "unet"}})
    monkeypatch.setattr(
        mnist_workflow, "quality", lambda *args: {"mmd2_mnist_features_real": float("nan")}
    )
    with pytest.raises(FloatingPointError):
        workflow._quality(config, torch.zeros((2, 1, 28, 28)), {})


def test_ephemeral_classifier_measurement_event_is_not_part_of_evaluation_digest(
    monkeypatch, tmp_path
):
    config, root, _, context, _ = _setup(monkeypatch, tmp_path)

    def sample(model, noise, times, **kwargs):
        history = (
            torch.stack([noise - index * 0.01 for index in range(len(times))])
            if kwargs.get("record_history", True)
            else None
        )
        return {
            "samples": noise - (len(times) - 1) * 0.01,
            "history": history,
            "total_model_calls": len(times) - 1,
            "initial_noise_sha256": runtime.tensor_hash(noise),
        }

    monkeypatch.setattr(workflow, "sample_euler", sample)
    monkeypatch.setattr(workflow, "_fidelity", lambda *args, **kwargs: {"paired_endpoint_mse": 0.0})
    context["metadata"]["classifier_test_measurement_created_here"] = True
    first = workflow.evaluate_suite(config)
    context["metadata"]["classifier_test_measurement_created_here"] = False
    second = workflow.evaluate_suite(config)
    assert first["evaluation_sha256"] == second["evaluation_sha256"]
    metadata = json.loads((root / "evaluation/results.json").read_text())["metadata"]
    assert metadata["quality_context"]["classifier_test_measurement_created_here"] is False
