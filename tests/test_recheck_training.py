import copy

import pytest
import torch

from recheck.config import resolve_config
from recheck.distill import calibrate, make_anchor, train_student
from recheck.fm import build_model, train_teacher
from recheck.runtime import model_hash
from recheck.schedules import make_schedule


@pytest.fixture
def tiny_samples():
    return torch.randn(48, 2, generator=torch.Generator().manual_seed(7001)) * 0.5


@pytest.fixture
def tiny_parent(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(1703)
        teacher = build_model(tiny_config(tmp_path / "parent"))
    return teacher.eval().requires_grad_(False)


def tiny_config(directory, **overrides):
    supplied = {
        "protocol_id": "test_recheck_deterministic_training",
        "output_dir": str(directory),
        "threads": 1,
        "device": "cpu",
        "model": {"hidden_dim": 16, "num_layers": 1, "time_embedding_dim": 8},
        "teacher": {
            "updates": 4,
            "batch_size": 8,
            "warmup": 0,
            "ema_decay": 0.9,
            "checkpoint_every": 2,
        },
        "student": {
            "updates": 6,
            "batch_size": 8,
            "warmup": 0,
            "ema_decay": 0.9,
            "checkpoint_every": 2,
        },
        "distillation": {
            "calibration_size": 16,
            "ot_backend": "reference",
            "gradient_diagnostics_every": 2,
            "plan_diagnostics_every": 2,
        },
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            supplied.setdefault(section, {}).update(values)
        else:
            supplied[section] = values
    return resolve_config(supplied)


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def test_teacher_continuous_and_explicit_resume_are_exact(tmp_path, tiny_samples):
    config_continuous = tiny_config(tmp_path / "continuous")
    continuous = train_teacher(config_continuous, samples=tiny_samples)
    config_resume = tiny_config(tmp_path / "resumed")
    interrupted = train_teacher(config_resume, samples=tiny_samples, stop_after=2)
    assert interrupted["manifest"]["status"] == "interrupted"
    assert interrupted["manifest"]["updates_completed"] == 2
    resumed = train_teacher(config_resume, samples=tiny_samples, resume=True)
    assert continuous["manifest"]["status"] == resumed["manifest"]["status"] == "completed"
    assert model_hash(continuous["model"]) == model_hash(resumed["model"])
    state_continuous = torch.load(
        tmp_path / "continuous" / "teacher" / "training.pt", weights_only=True
    )
    state_resumed = torch.load(tmp_path / "resumed" / "teacher" / "training.pt", weights_only=True)
    for key in (
        "model",
        "ema",
        "optimizer",
        "streams",
        "torch_rng",
        "cuda_rng",
        "updates_completed",
    ):
        assert_nested_equal(state_continuous[key], state_resumed[key])
    for original, restored in zip(state_continuous["history"], state_resumed["history"]):
        assert {k: v for k, v in original.items() if k != "training_seconds"} == {
            k: v for k, v in restored.items() if k != "training_seconds"
        }


def test_teacher_existing_run_requires_explicit_resume(tmp_path, tiny_samples):
    config = tiny_config(tmp_path)
    train_teacher(config, samples=tiny_samples, stop_after=2)
    with pytest.raises(FileExistsError, match="resume"):
        train_teacher(config, samples=tiny_samples)


def test_teacher_resume_rejects_changed_data_or_training_contract(tmp_path, tiny_samples):
    config = tiny_config(tmp_path)
    train_teacher(config, samples=tiny_samples, stop_after=2)
    modified = tiny_samples.clone()
    modified[0, 0] += 1
    with pytest.raises(ValueError, match="contract"):
        train_teacher(config, samples=modified, resume=True)
    altered = copy.deepcopy(config)
    altered["teacher"]["lr"] *= 2
    with pytest.raises(ValueError, match="contract"):
        train_teacher(altered, samples=tiny_samples, resume=True)


def _load_training(result):
    from pathlib import Path

    return torch.load(Path(result["directory"]) / "training.pt", weights_only=True)


def test_student_lambda_zero_is_exactly_mse_with_identical_rng(tmp_path, tiny_parent, tiny_samples):
    config = tiny_config(tmp_path, distillation={"lambda_w": 0.0, "plan_diagnostics_every": 1})
    calibration = calibrate(config, tiny_parent, tiny_samples)
    parent_hash = model_hash(tiny_parent)
    mse = train_student(config, tiny_parent, tiny_samples, calibration, loss="mse", rho=0, seed=101)
    distribution_disabled = train_student(
        config, tiny_parent, tiny_samples, calibration, loss="sinkhorn_endpoint", rho=0, seed=101
    )
    assert model_hash(mse["model"]) == model_hash(distribution_disabled["model"])
    assert model_hash(mse["model"]) != parent_hash
    state_mse, state_disabled = _load_training(mse), _load_training(distribution_disabled)
    for key in ("model", "ema", "optimizer", "streams", "updates_completed"):
        assert_nested_equal(state_mse[key], state_disabled[key])
    assert [row["common_rng_trace"] for row in state_mse["history"]] == [
        row["common_rng_trace"] for row in state_disabled["history"]
    ]
    assert [row["pair_loss"] for row in state_mse["history"]] == [
        row["pair_loss"] for row in state_disabled["history"]
    ]
    assert model_hash(tiny_parent) == parent_hash
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in tiny_parent.parameters()
    )


def test_student_continuous_and_resume_are_exact(tmp_path, tiny_parent, tiny_samples):
    uninterrupted_config = tiny_config(tmp_path / "continuous")
    calibration = calibrate(uninterrupted_config, tiny_parent, tiny_samples)
    uninterrupted = train_student(
        uninterrupted_config,
        tiny_parent,
        tiny_samples,
        calibration,
        loss="sinkhorn_endpoint",
        rho=0.1,
    )
    resumed_config = tiny_config(tmp_path / "resumed")
    interrupted = train_student(
        resumed_config,
        tiny_parent,
        tiny_samples,
        calibration,
        loss="sinkhorn_endpoint",
        rho=0.1,
        stop_after=2,
    )
    assert interrupted["manifest"]["status"] == "interrupted"
    restored = train_student(
        resumed_config,
        tiny_parent,
        tiny_samples,
        calibration,
        loss="sinkhorn_endpoint",
        rho=0.1,
        resume=True,
    )
    assert restored["manifest"]["status"] == "completed"
    assert model_hash(restored["model"]) == model_hash(uninterrupted["model"])
    state_full, state_resumed = _load_training(uninterrupted), _load_training(restored)
    for key in (
        "model",
        "ema",
        "optimizer",
        "streams",
        "torch_rng",
        "cuda_rng",
        "updates_completed",
    ):
        assert_nested_equal(state_full[key], state_resumed[key])
    for original, resumed in zip(state_full["history"], state_resumed["history"]):
        assert {
            k: v for k, v in original.items() if k not in {"training_seconds", "ot_seconds"}
        } == {k: v for k, v in resumed.items() if k not in {"training_seconds", "ot_seconds"}}


def test_four_mvp_methods_use_identical_training_streams_and_frozen_parent(
    tmp_path, tiny_parent, tiny_samples
):
    config = tiny_config(
        tmp_path, student={"updates": 3}, distillation={"plan_diagnostics_every": 1}
    )
    calibration = calibrate(config, tiny_parent, tiny_samples)
    parent_hash = model_hash(tiny_parent)
    traces = []
    methods = []
    for loss, rho in (
        ("mse", 0),
        ("mse", 0.1),
        ("sinkhorn_endpoint", 0),
        ("sinkhorn_endpoint", 0.1),
    ):
        result = train_student(config, tiny_parent, tiny_samples, calibration, loss=loss, rho=rho)
        state = _load_training(result)
        methods.append(result["manifest"]["role"])
        assert result["manifest"]["status"] == "completed"
        assert result["manifest"]["updates_completed"] == 3
        assert result["manifest"]["student_init_sha256"] == parent_hash
        assert result["manifest"]["teacher_sha256_after"] == parent_hash
        assert result["manifest"]["micro_batch"] == result["manifest"]["ot_support_size"] == 8
        assert all(torch.isfinite(torch.tensor(row["loss"])) for row in state["history"])
        assert all(row["parameter_update_norm"] > 0 for row in state["history"])
        assert state["history"][0]["pair_grad_norm"] > 0
        if loss.startswith("sinkhorn"):
            assert state["history"][0]["dist_grad_norm"] > 0
            assert state["history"][0]["ot_plan"]["cross"]["converged"]
        traces.append([row["common_rng_trace"] for row in state["history"]])
    assert len(set(methods)) == 4
    assert all(trace == traces[0] for trace in traces[1:])
    assert model_hash(tiny_parent) == parent_hash
    assert all(parameter.grad is None for parameter in tiny_parent.parameters())


@pytest.mark.parametrize(
    "loss",
    ["sinkhorn_displacement", "sinkhorn_joint", "swd_endpoint", "swd_displacement", "swd_joint"],
)
def test_extended_distribution_losses_keep_student_graph(tmp_path, tiny_parent, tiny_samples, loss):
    config = tiny_config(
        tmp_path, student={"updates": 2}, distillation={"swd_train_projections": 16}
    )
    calibration = calibrate(config, tiny_parent, tiny_samples)
    result = train_student(config, tiny_parent, tiny_samples, calibration, loss=loss, rho=0.1)
    state = _load_training(result)
    assert result["manifest"]["status"] == "completed"
    assert state["history"][0]["dist_grad_norm"] > 0
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in state["history"])
    assert model_hash(result["model"]) != model_hash(tiny_parent)


def test_tube_zero_and_retargeted_tube_use_the_same_teacher_student_anchor(
    tmp_path, tiny_parent, tiny_samples, monkeypatch
):
    import recheck.distill as implementation
    from recheck.runtime import RNGStreams

    config = tiny_config(tmp_path)
    calibration = calibrate(config, tiny_parent, tiny_samples)
    real_target, real_student = implementation.teacher_target, implementation.student_endpoint
    captures = {"teacher": [], "student": []}

    def target_spy(model, anchor, t, width, substeps):
        target = real_target(model, anchor, t, width, substeps)
        captures["teacher"].append((anchor.clone(), target.clone(), t, width))
        return target

    def student_spy(model, anchor, t, width):
        captures["student"].append(anchor.clone())
        return real_student(model, anchor, t, width)

    monkeypatch.setattr(implementation, "teacher_target", target_spy)
    monkeypatch.setattr(implementation, "student_endpoint", student_spy)
    train_student(config, tiny_parent, tiny_samples, calibration, loss="mse", rho=0, stop_after=1)
    clean_anchor, clean_target, t, width = captures["teacher"][0]
    assert torch.equal(clean_anchor, captures["student"][0])
    captures["teacher"].clear()
    captures["student"].clear()
    train_student(config, tiny_parent, tiny_samples, calibration, loss="mse", rho=0.1, stop_after=1)
    used_anchor, tube_target, _, _ = captures["teacher"][0]
    assert torch.equal(used_anchor, captures["student"][0])
    streams = RNGStreams(101)
    interval = int(torch.randint(16, (), generator=streams["interval"]))
    indices = torch.randint(len(tiny_samples), (8,), generator=streams["data"])
    noise = torch.randn(8, 2, generator=streams["noise"])
    eta = torch.randn(8, 2, generator=streams["tube"])
    assert t == float(make_schedule().times[interval])
    torch.testing.assert_close(
        clean_anchor, (1 - t) * tiny_samples[indices] + t * noise, rtol=0, atol=0
    )
    torch.testing.assert_close(used_anchor - clean_anchor, 0.1 * width * eta, atol=1e-7, rtol=1e-5)
    assert not torch.equal(clean_target, tube_target)
    assert not tube_target.requires_grad


def test_teacher_rollout_anchors_and_irregular_grouping_keep_actual_nfe():
    class LinearField(torch.nn.Module):
        def forward(self, x, t):
            return x

    schedule = make_schedule(24, 16, groups=[2, 1] * 8)
    noise = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    data = torch.zeros_like(noise)
    interval = 7
    teacher = LinearField()
    anchor, calls = make_anchor(
        teacher, data, noise, float(schedule.times[interval]), interval, schedule, "teacher_rollout"
    )
    expected_calls = sum(schedule.groups[:interval])
    assert calls == expected_calls
    torch.testing.assert_close(anchor, noise * (1 - 1 / 24) ** expected_calls)
    interpolant, calls = make_anchor(
        teacher, data, noise, float(schedule.times[interval]), interval, schedule, "interpolant"
    )
    assert calls == 0
    torch.testing.assert_close(interpolant, float(schedule.times[interval]) * noise)
    assert not torch.allclose(anchor, interpolant)


@pytest.mark.parametrize(
    "changed_settings", [{"gamma": 2.0}, {"epsilon_alpha": 0.1}, {"calibration_size": 32}]
)
def test_student_rejects_calibration_from_different_feature_scaling(
    tmp_path, tiny_parent, tiny_samples, changed_settings
):
    config = tiny_config(tmp_path)
    calibration = calibrate(config, tiny_parent, tiny_samples)
    changed = copy.deepcopy(config)
    changed["distillation"].update(changed_settings)
    with pytest.raises(ValueError, match="Calibration"):
        train_student(changed, tiny_parent, tiny_samples, calibration)


def test_student_rejects_tampered_fixed_teacher_calibration(tmp_path, tiny_parent, tiny_samples):
    config = tiny_config(tmp_path)
    calibration = calibrate(config, tiny_parent, tiny_samples)
    calibration["intervals"][0]["representations"]["endpoint"]["epsilon"] *= 2
    with pytest.raises(ValueError, match="SHA256"):
        train_student(config, tiny_parent, tiny_samples, calibration)


def _tracking_budget():
    instances = []

    class TrackingBudget:
        def __init__(self, config, device):
            assert not any(instance.active for instance in instances), "Previous failed setup leaked the accelerator lock"
            self.active = True
            self.check_calls = 0
            instances.append(self)  # Keep strong references: garbage collection cannot hide a leak.

        def check(self):
            assert self.active
            self.check_calls += 1

        def close(self):
            self.active = False

    return TrackingBudget, instances


def test_student_calibration_setup_failure_closes_budget_before_next_attempt(tmp_path, tiny_parent, tiny_samples, monkeypatch):
    import recheck.distill as implementation

    config = tiny_config(tmp_path)
    calibration = calibrate(config, tiny_parent, tiny_samples)
    invalid = copy.deepcopy(calibration)
    invalid["teacher_sha256"] = "wrong frozen parent"
    budget, instances = _tracking_budget()
    monkeypatch.setattr(implementation, "ResourceBudget", budget)
    with pytest.raises(ValueError, match="Calibration"):
        train_student(config, tiny_parent, tiny_samples, invalid)
    assert len(instances) == 1 and not instances[0].active
    assert instances[0].check_calls >= 1
    result = train_student(config, tiny_parent, tiny_samples, calibration)
    assert result["manifest"]["status"] == "completed"
    assert len(instances) == 2 and not any(instance.active for instance in instances)


def test_teacher_restore_contract_failure_closes_budget_without_waiting_for_gc(tmp_path, tiny_samples, monkeypatch):
    import recheck.fm as implementation

    config = tiny_config(tmp_path)
    train_teacher(config, samples=tiny_samples, stop_after=2)
    invalid = copy.deepcopy(config)
    invalid["teacher"]["lr"] *= 2
    budget, instances = _tracking_budget()
    monkeypatch.setattr(implementation, "ResourceBudget", budget)
    with pytest.raises(ValueError, match="contract"):
        train_teacher(invalid, samples=tiny_samples, resume=True)
    assert len(instances) == 1 and not instances[0].active
    assert instances[0].check_calls >= 1
    result = train_teacher(config, samples=tiny_samples, resume=True)
    assert result["manifest"]["status"] == "completed"
    assert not any(instance.active for instance in instances)


def test_student_restore_contract_failure_closes_budget_without_waiting_for_gc(tmp_path, tiny_parent, tiny_samples, monkeypatch):
    import recheck.distill as implementation

    config = tiny_config(tmp_path)
    calibration = calibrate(config, tiny_parent, tiny_samples)
    train_student(config, tiny_parent, tiny_samples, calibration, stop_after=2)
    invalid = copy.deepcopy(config)
    invalid["student"]["lr"] *= 2
    budget, instances = _tracking_budget()
    monkeypatch.setattr(implementation, "ResourceBudget", budget)
    with pytest.raises(ValueError, match="contract"):
        train_student(invalid, tiny_parent, tiny_samples, calibration, resume=True)
    assert len(instances) == 1 and not instances[0].active
    assert instances[0].check_calls >= 1
    result = train_student(config, tiny_parent, tiny_samples, calibration, resume=True)
    assert result["manifest"]["status"] == "completed"
    assert not any(instance.active for instance in instances)
