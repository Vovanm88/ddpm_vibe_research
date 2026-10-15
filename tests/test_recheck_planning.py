import copy
import json
from pathlib import Path

import pytest
import torch

from recheck.config import config_hash, resolve_config
from recheck.planning import merge_benchmark_ledger, propose_equal_budget
from recheck.runtime import atomic_json, model_hash


def _mnist_plan():
    return resolve_config(
        {
            "data": {"name": "mnist"},
            "model": {"kind": "unet"},
            "teacher": {"updates": 30000},
            "student": {"updates": 10000},
        }
    )


def test_common_reduction_uses_only_timing_and_keeps_teacher_fixed():
    config = _mnist_plan()
    original = copy.deepcopy(config)
    timings = {"teacher": 0.2, "mse": 0.02, "sinkhorn_endpoint": 0.2}
    result = propose_equal_budget(config, timings)
    assert config == original
    assert result["status"] == "reduced_budget_proposal"
    assert result["proposed_common_student_updates"] == 6250
    assert result["student_run_count"] == 12
    assert result["teacher_registered_updates"] == result["teacher_updates_remaining"] == 30000
    assert result["teacher_budget_changed"] is False
    assert result["proposed_total_with_spent_seconds"] <= 6 * 3600
    assert 1.3 * (30000 * 0.2 + 6500 * 6 * (0.02 + 0.2)) + 2700 > 6 * 3600
    assert not result["automatic_config_change"]
    assert result["registration_required_before_reduced_runs"]


def test_completed_teacher_and_prior_spending_are_counted_once():
    config = _mnist_plan()
    timings = {"teacher": 0.2, "mse": 0.02, "sinkhorn_endpoint": 0.2}
    before = propose_equal_budget(config, timings, teacher_completed_updates=30000)
    assert before["status"] == "full_budget_fits"
    assert before["teacher_updates_remaining"] == 0
    assert before["proposed_common_student_updates"] == 10000
    after = propose_equal_budget(
        config, timings, spent_seconds=6000, teacher_completed_updates=30000
    )
    assert after["proposed_common_student_updates"] == 7500
    assert after["already_spent_seconds"] == 6000
    assert after["teacher_updates_remaining"] == 0
    assert after["proposed_total_with_spent_seconds"] <= after["budget_seconds"]


def test_no_feasible_budget_does_not_suggest_skipping_teacher_or_zero_updates():
    result = propose_equal_budget(
        _mnist_plan(), {"teacher": 0.2, "mse": 0.02, "sinkhorn_endpoint": 0.2}, spent_seconds=19000
    )
    assert result["status"] == "no_feasible_budget"
    assert result["proposed_common_student_updates"] is None
    assert result["proposed_total_with_spent_seconds"] is None
    assert result["teacher_updates_remaining"] == 30000


@pytest.mark.parametrize(
    "timings,spent",
    [
        ({"teacher": float("nan"), "mse": 0.1, "sinkhorn_endpoint": 0.2}, 0),
        ({"teacher": 0.1, "mse": float("inf"), "sinkhorn_endpoint": 0.2}, 0),
        ({"teacher": 0.1, "mse": 0.1, "sinkhorn_endpoint": 0}, 0),
        ({"teacher": 0.1, "mse": 0.1, "sinkhorn_endpoint": 0.2}, -1),
        ({"teacher": 0.1, "mse": 0.1, "sinkhorn_endpoint": 0.2}, float("nan")),
    ],
)
def test_invalid_timing_or_spending_never_produces_a_budget(timings, spent):
    with pytest.raises(ValueError):
        propose_equal_budget(_mnist_plan(), timings, spent_seconds=spent)


def test_unmeasured_loss_is_not_assigned_an_invented_timing():
    config = _mnist_plan()
    config["distillation"]["losses"] = ["mse", "swd_endpoint"]
    with pytest.raises(ValueError, match="Missing measured timing"):
        propose_equal_budget(config, {"teacher": 0.1, "mse": 0.1, "sinkhorn_endpoint": 0.2})


def test_child_ledger_receipt_is_idempotent_and_accepts_only_positive_deltas(tmp_path):
    parent, source = tmp_path / "parent", tmp_path / "probe" / "resource_ledger.json"
    atomic_json(parent / "resource_ledger.json", {"gpu_training_seconds": 100})
    atomic_json(source, {"gpu_training_seconds": 20, "cpu_training_wall_seconds": 5})
    first = merge_benchmark_ledger(parent, source)
    assert first == {"credited_gpu_seconds": 20, "credited_cpu_seconds": 5}
    assert merge_benchmark_ledger(parent, source) == {
        "credited_gpu_seconds": 0,
        "credited_cpu_seconds": 0,
    }
    atomic_json(source, {"gpu_training_seconds": 27, "cpu_training_wall_seconds": 6})
    assert merge_benchmark_ledger(parent, source) == {
        "credited_gpu_seconds": 7,
        "credited_cpu_seconds": 1,
    }
    ledger = json.loads((parent / "resource_ledger.json").read_text())
    assert ledger["gpu_training_seconds"] == 127
    assert ledger["cpu_training_wall_seconds"] == 6
    receipt = ledger["benchmark_receipts"][str(source.resolve())]
    assert receipt["source_path"] == str(source.resolve()) and len(receipt["source_sha256"]) == 64
    atomic_json(source, {"gpu_training_seconds": 10, "cpu_training_wall_seconds": 6})
    with pytest.raises(RuntimeError, match="decreased"):
        merge_benchmark_ledger(parent, source)
    assert json.loads((parent / "resource_ledger.json").read_text())["gpu_training_seconds"] == 127


def test_ledger_cannot_credit_itself_or_nonfinite_source_counters(tmp_path):
    parent = tmp_path / "resource_ledger.json"
    atomic_json(parent, {"gpu_training_seconds": 100})
    with pytest.raises(ValueError, match="itself"):
        merge_benchmark_ledger(tmp_path, parent)
    source = tmp_path / "probe" / "resource_ledger.json"
    source.parent.mkdir()
    source.write_text('{"gpu_training_seconds": NaN}')
    with pytest.raises(ValueError, match="finite"):
        merge_benchmark_ledger(tmp_path, source)
    assert json.loads(parent.read_text())["gpu_training_seconds"] == 100


@pytest.fixture
def fake_probe(tmp_path, monkeypatch):
    import recheck.runtime as runtime
    import recheck.workflow as workflow

    class FrozenParent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    parent = FrozenParent()
    config = resolve_config(
        {
            "output_dir": str(tmp_path / "runs"),
            "threads": 1,
            "teacher": {"updates": 30000},
            "student": {"updates": 10000},
        }
    )
    root = Path(config["output_dir"])
    (root / "teacher/ema_model").mkdir(parents=True)
    atomic_json(root / "teacher/ema_model/config.json", {"parent": "untouched"})
    atomic_json(root / "teacher/manifest.json", {"status": "completed", "updates_completed": 30000})
    atomic_json(root / "resource_ledger.json", {"gpu_training_seconds": 100.0})
    clock, active, calls, data_splits = [0.0], [False], [], []
    state = {"interrupt": None, "bad_timing": False}
    costs = {"teacher": 0.1, "mse": 0.2, "sinkhorn_endpoint": 0.3}
    monkeypatch.setattr(workflow, "choose_device", lambda name: torch.device("mps"))
    monkeypatch.setattr(workflow, "_teacher", lambda config: (parent, {}))
    monkeypatch.setattr(workflow, "require_rays_gate", lambda config: None)
    monkeypatch.setattr(workflow, "algorithm_hash", lambda: "fixed_algorithm")
    monkeypatch.setattr(
        workflow, "build_model", lambda config: pytest.fail("fixed parent must not be rebuilt")
    )
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])

    def samples(config, split="train"):
        data_splits.append(split)
        assert split == "train"
        return torch.ones(16, 2)

    monkeypatch.setattr(workflow, "benchmark_data", samples)

    def calibration(config, teacher, samples):
        assert not active[0]
        assert teacher is parent
        clock[0] += 2
        return {"fixed": "train_only"}

    monkeypatch.setattr(workflow, "calibrate", calibration)

    def training(config, *, samples=None, resume=False, kind="teacher", seed=None, **kwargs):
        assert not active[0], "Benchmark training cases must never overlap"
        active[0] = True
        try:
            if kind == "teacher":
                directory = Path(config["output_dir"]) / "teacher"
                assert config["teacher"]["seed"] != 41
            else:
                from recheck.config import method_name

                directory = (
                    Path(config["output_dir"]) / "students" / f"seed{seed}_{method_name(kind, 0)}"
                )
                assert seed not in {101, 202, 303, config["evaluation"]["final_seed"]}
            calls.append((kind, resume, config_hash(config)))
            directory.mkdir(parents=True, exist_ok=True)
            previous = (
                json.loads((directory / "manifest.json").read_text()).get("updates_completed", 0)
                if resume
                else 0
            )
            count = 35 if state["interrupt"] == kind else 120 - previous
            clock[0] += count * costs[kind]
            ledger_path = Path(config["output_dir"]) / "resource_ledger.json"
            ledger = json.loads(ledger_path.read_text())
            # Interrupted work has only an earlier committed timing check;
            # benchmark finally must still account for the unsaved tail.
            recorded = 20 if state["interrupt"] == kind else count
            ledger["gpu_training_seconds"] = (
                ledger.get("gpu_training_seconds", 0) + recorded * costs[kind]
            )
            atomic_json(ledger_path, ledger)
            completed = previous + recorded
            (directory / "training.pt").write_bytes(b"own checkpoint")
            atomic_json(
                directory / "manifest.json",
                {
                    "status": "interrupted" if state["interrupt"] == kind else "completed",
                    "updates_completed": completed,
                },
            )
            history = [
                {"update": i + 1, "training_seconds": (i + 1) * costs[kind]}
                for i in range(completed)
            ]
            if state["bad_timing"]:
                history[-1]["training_seconds"] = float("nan")
            (directory / "history.json").write_text(json.dumps(history))
            if state["interrupt"] == kind:
                raise KeyboardInterrupt
            return {"directory": str(directory)}
        finally:
            active[0] = False

    monkeypatch.setattr(
        workflow,
        "train_teacher",
        lambda config, **kwargs: training(config, kind="teacher", **kwargs),
    )
    monkeypatch.setattr(
        workflow,
        "train_student",
        lambda config, teacher, samples, calibration, *, loss, seed, **kwargs: training(
            config, samples=samples, kind=loss, seed=seed, **kwargs
        ),
    )
    return config, workflow, parent, clock, calls, costs, state, data_splits


def test_probe_accounts_all_cases_and_overhead_without_touching_final_parent(fake_probe):
    config, workflow, parent, _, calls, costs, _, splits = fake_probe
    digest = model_hash(parent)
    parent_config = (Path(config["output_dir"]) / "teacher/ema_model/config.json").read_bytes()
    result = workflow.benchmark_stage(config)
    assert [row[0] for row in calls] == ["teacher", "mse", "sinkhorn_endpoint"]
    assert result["seconds_per_update"] == pytest.approx(costs)
    assert result["budget_proposal"]["teacher_updates_remaining"] == 0
    ledger = json.loads((Path(config["output_dir"]) / "resource_ledger.json").read_text())
    assert ledger["gpu_training_seconds"] == pytest.approx(100 + 2 + 120 * sum(costs.values()))
    assert len(ledger["benchmark_receipts"]) == 1
    assert model_hash(parent) == digest
    assert (
        Path(config["output_dir"]) / "teacher/ema_model/config.json"
    ).read_bytes() == parent_config
    assert splits == ["train"]


def test_same_probe_cache_is_byte_identical_and_different_config_has_another_cache(fake_probe):
    config, workflow, _, _, calls, _, _, _ = fake_probe
    first = workflow.benchmark_stage(config)
    leaf = Path(first["child_resource_ledger"]).parent / "benchmark.json"
    payload = leaf.read_bytes()
    assert workflow.benchmark_stage(config) == first
    assert leaf.read_bytes() == payload and len(calls) == 3
    changed = copy.deepcopy(config)
    changed["student"]["lr"] *= 2
    second = workflow.benchmark_stage(changed)
    assert first["child_resource_ledger"] != second["child_resource_ledger"]
    assert len(calls) == 6 and leaf.read_bytes() == payload


def test_interrupted_probe_tail_is_accounted_then_resume_does_not_recount_old_cases(fake_probe):
    config, workflow, _, _, calls, costs, state, _ = fake_probe
    state["interrupt"] = "mse"
    with pytest.raises(KeyboardInterrupt):
        workflow.benchmark_stage(config)
    ledger_path = Path(config["output_dir"]) / "resource_ledger.json"
    partial = json.loads(ledger_path.read_text())["gpu_training_seconds"]
    assert partial == pytest.approx(100 + 2 + 120 * costs["teacher"] + 35 * costs["mse"])
    assert not list((Path(config["output_dir"]) / "benchmarks").rglob("benchmark.json"))
    state["interrupt"] = None
    workflow.benchmark_stage(config)
    assert [(kind, resumed) for kind, resumed, _ in calls] == [
        ("teacher", False),
        ("mse", False),
        ("mse", True),
        ("sinkhorn_endpoint", False),
    ]
    assert calls[1][2] == calls[2][2], "Resume must use the exact registered child config"
    total = json.loads(ledger_path.read_text())["gpu_training_seconds"]
    assert total == pytest.approx(
        100 + 4 + 120 * costs["teacher"] + 135 * costs["mse"] + 120 * costs["sinkhorn_endpoint"]
    )


def test_nonfinite_measured_timing_stops_probe_and_still_accounts_spending(fake_probe):
    config, workflow, _, _, _, costs, state, _ = fake_probe
    state["bad_timing"] = True
    with pytest.raises(ValueError, match="Nonfinite"):
        workflow.benchmark_stage(config)
    ledger = json.loads((Path(config["output_dir"]) / "resource_ledger.json").read_text())
    assert ledger["gpu_training_seconds"] == pytest.approx(100 + 2 + 120 * costs["teacher"])
    assert not list((Path(config["output_dir"]) / "benchmarks").rglob("benchmark.json"))


def test_probe_dry_run_never_constructs_model_or_changes_artifacts(tmp_path, monkeypatch):
    import recheck.workflow as workflow

    config = resolve_config({"output_dir": str(tmp_path / "unused")})
    monkeypatch.setattr(workflow, "choose_device", lambda config: pytest.fail("dry-run is inert"))
    assert workflow.benchmark_stage(config, dry_run=True)["forecast_hours"] is None
    assert not (tmp_path / "unused").exists()
