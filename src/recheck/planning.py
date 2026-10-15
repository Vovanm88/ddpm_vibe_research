"""Timing-only budget proposals and idempotent accounting of timing probes.

No model quality, held-out data, or final evaluation noise enters a proposal.
The caller must register a derived configuration before running any reduction.
"""

import hashlib
import json
import math
from pathlib import Path


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def propose_equal_budget(config, timings, spent_seconds=0, teacher_completed_updates=0):
    """Propose one common student budget, keeping the teacher budget fixed.

    ``timings`` contains measured seconds/update for teacher, mse and each loss
    family used in the configuration. Endpoint timings are not silently reused
    for unmeasured joint/SWD objectives. Teacher progress is supplied explicitly
    so this function is pure and never inspects or modifies run files.
    """
    overhead, reserve, quantum, minimum = 1.3, 0.75 * 3600, 250, 250
    spent = _nonnegative(spent_seconds, "spent_seconds")
    budget = _nonnegative(config["resources"]["max_gpu_hours"], "max_gpu_hours") * 3600
    if (
        isinstance(teacher_completed_updates, bool)
        or not isinstance(teacher_completed_updates, int)
        or teacher_completed_updates < 0
    ):
        raise ValueError("teacher_completed_updates must be a nonnegative integer")
    teacher_updates = config["teacher"]["updates"]
    student_updates = config["student"]["updates"]
    teacher_remaining = max(0, teacher_updates - teacher_completed_updates)
    teacher_time = _nonnegative(timings["teacher"], "timings.teacher")
    if teacher_time == 0:
        raise ValueError("measured teacher timing must be positive")
    runs_per_loss = len(config["distillation"]["seeds"]) * len(config["distillation"]["tube_rhos"])
    cohort_per_update = 0.0
    used_timings = {"teacher": teacher_time}
    for loss in config["distillation"]["losses"]:
        if loss not in timings:
            raise ValueError(
                f"Missing measured timing for {loss}; benchmark this loss before budgeting it"
            )
        timing = _nonnegative(timings[loss], f"timings.{loss}")
        if timing == 0:
            raise ValueError(f"measured timing for {loss} must be positive")
        used_timings[loss] = timing
        cohort_per_update += runs_per_loss * timing
    if cohort_per_update == 0:
        raise ValueError("budget proposal requires at least one student run")
    teacher_seconds = teacher_remaining * teacher_time
    full_training = teacher_seconds + student_updates * cohort_per_update
    full_future = overhead * full_training + reserve
    remaining = max(0.0, budget - spent)
    fits = spent + full_future <= budget
    if fits:
        status, proposed = "full_budget_fits", student_updates
    else:
        allowance = ((remaining - reserve) / overhead - teacher_seconds) / cohort_per_update
        candidate = max(0, math.floor(allowance / quantum) * quantum)
        proposed = min(student_updates, candidate) if candidate >= minimum else None
        status = "reduced_budget_proposal" if proposed is not None else "no_feasible_budget"
    proposed_training = None if proposed is None else teacher_seconds + proposed * cohort_per_update
    proposed_future = None if proposed_training is None else overhead * proposed_training + reserve
    return {
        "status": status,
        "fits_registered_budget": fits,
        "budget_seconds": budget,
        "already_spent_seconds": spent,
        "remaining_budget_seconds": remaining,
        "teacher_registered_updates": teacher_updates,
        "teacher_updates_completed": teacher_completed_updates,
        "teacher_updates_remaining": teacher_remaining,
        "teacher_budget_changed": False,
        "student_run_count": runs_per_loss * len(config["distillation"]["losses"]),
        "requested_student_updates": student_updates,
        "proposed_common_student_updates": proposed,
        "reduction_quantum": quantum,
        "minimum_reduced_updates": minimum,
        "seconds_per_update": used_timings,
        "cohort_seconds_per_common_update": cohort_per_update,
        "full_training_forecast_seconds": full_training,
        "full_conservative_remaining_seconds": full_future,
        "full_total_with_spent_seconds": spent + full_future,
        "proposed_training_forecast_seconds": proposed_training,
        "proposed_conservative_remaining_seconds": proposed_future,
        "proposed_total_with_spent_seconds": None
        if proposed_future is None
        else spent + proposed_future,
        "automatic_config_change": False,
        "registration_required_before_reduced_runs": proposed is not None and not fits,
        "assumptions": {
            "training_overhead_multiplier": overhead,
            "evaluation_classifier_reserve_seconds": reserve,
            "reserve_is_an_estimate_not_a_guarantee": True,
            "teacher_budget_is_fixed": True,
            "student_updates_are_identical_for_all_methods_and_seeds": True,
            "no_quality_metrics_or_test_data_used": True,
        },
    }


def read_resource_ledger(root):
    path = Path(root) / "resource_ledger.json"
    ledger = json.loads(path.read_text()) if path.is_file() else {}
    for key in ("gpu_training_seconds", "cpu_training_wall_seconds"):
        _nonnegative(ledger.get(key, 0), key)
    return ledger


def merge_benchmark_ledger(parent_root, source_path):
    """Credit only the new cumulative seconds from a child ledger.

    Receipts record the absolute source path, actual file SHA and credited
    counters. A repeated receipt contributes zero, while an interrupted probe
    can later extend the same source monotonically without double counting.
    """
    from .runtime import atomic_json

    source_path = Path(source_path).resolve()
    if source_path == (Path(parent_root) / "resource_ledger.json").resolve():
        raise ValueError("A parent ledger cannot be credited to itself")
    if not source_path.is_file():
        return {"credited_gpu_seconds": 0.0, "credited_cpu_seconds": 0.0}
    raw = source_path.read_bytes()
    source = json.loads(raw)
    parent = read_resource_ledger(parent_root)
    receipts = parent.setdefault("benchmark_receipts", {})
    source_id = str(source_path)
    previous = receipts.get(source_id, {})
    delta = {}
    counters = {}
    for key in ("gpu_training_seconds", "cpu_training_wall_seconds"):
        current = _nonnegative(source.get(key, 0), f"source.{key}")
        credited = _nonnegative(previous.get(key, 0), f"receipt.{key}")
        if current < credited:
            raise RuntimeError(
                "Benchmark ledger counters decreased; retain the original receipt and investigate"
            )
        delta[key] = current - credited
        counters[key] = current
        parent[key] = float(parent.get(key, 0)) + delta[key]
    receipt = {
        "source_path": source_id,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        **counters,
    }
    if receipt != previous:
        receipts[source_id] = receipt
        atomic_json(Path(parent_root) / "resource_ledger.json", parent)
    return {
        "credited_gpu_seconds": delta["gpu_training_seconds"],
        "credited_cpu_seconds": delta["cpu_training_wall_seconds"],
    }


def check_registered_resources(config):
    """Read-only global check; does not acquire a second accelerator lock."""
    root = Path(config["output_dir"])
    ledger = read_resource_ledger(root)
    spent = ledger.get("gpu_training_seconds", 0.0)
    if spent >= config["resources"]["max_gpu_hours"] * 3600:
        raise RuntimeError(
            "Cumulative registered accelerator budget reached, including benchmark receipts"
        )
    size = (
        sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if root.exists()
        else 0
    )
    if size > config["resources"]["max_artifact_gib"] * 1024**3:
        raise RuntimeError(
            "Registered artifact disk budget exceeded, including benchmark artifacts"
        )
    return ledger
