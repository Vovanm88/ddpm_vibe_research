"""Synthetic checks: no training, native exports, or registered final data."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "examples/evaluate_method_zoo.py"
SPEC = importlib.util.spec_from_file_location("method_zoo_evaluation", SOURCE)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def _ray_reference(n_classes=2):
    targets = torch.arange(n_classes).repeat_interleave(16)
    radii = torch.linspace(0.3, 0.9, 16).repeat(n_classes)
    angle = math.pi / 2 + targets * 2 * math.pi / n_classes
    angle += (radii / (10 / 11)).square() * 1.6 * math.pi / (0.9 * n_classes)
    return torch.stack((radii * angle.cos(), radii * angle.sin()), dim=1), targets


def _primary_rows(task="rays", nfe=1):
    rows = []
    for index, method in enumerate(evaluation.METHODS):
        for seed in evaluation.SEEDS:
            flat = {key: 0.1 + index * 0.01 for key in evaluation.METRICS[task]}
            if task == "rays":
                flat["routing_accuracy_outer"] = 1 - index * 0.01
                flat["outer_mass_gap"] = index * 0.01
            else:
                flat["success_fraction"] = 1 - index * 0.01
                flat["both_modes_all_conditions"] = 1.0
            rows.append(
                {
                    "task": task,
                    "nfe": nfe,
                    "method": method,
                    "seed": seed,
                    "league": "primary",
                    "status": "completed",
                    "quality": {"flat": flat},
                }
            )
    return rows


def test_condition_swaps_are_visible_even_with_identical_marginal_samples():
    points = torch.tensor([[0.0, 0.0], [0.0, 0.1], [1.0, 1.0], [1.0, 1.1]])
    targets = torch.tensor([0, 0, 1, 1])
    result = evaluation.conditional_swd(points.flip(0), targets, points, targets, torch.eye(2))
    assert result["marginal"] == pytest.approx(0)
    assert result["mean"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Every requested condition"):
        evaluation.conditional_swd(points[:2], targets[:2], points, targets, torch.eye(2))


def test_origin_collapse_cannot_win_by_undefined_outer_accuracy_or_phase():
    reference, targets = _ray_reference()
    before = torch.get_rng_state().clone()
    quality = evaluation.rays_quality(
        torch.zeros_like(reference), targets, reference, targets, torch.eye(2), n_classes=2
    )
    assert quality["flat"]["routing_accuracy_outer"] is None
    assert quality["flat"]["outer_mass_gap"] == 1
    assert quality["flat"]["correct_outer_fraction_all"] == 0
    assert quality["flat"]["phase_distance_w1_sector"] is None
    assert torch.equal(torch.get_rng_state(), before)
    rows = _primary_rows()
    for row in rows:
        if row["method"] == "endpoint":
            row["quality"] = quality
    summary = evaluation.aggregate_rankings(rows)
    collapsed = next(
        row
        for row in summary["summaries"]
        if row["task"] == "rays" and row["nfe"] == 1 and row["method"] == "endpoint"
    )
    assert collapsed["pareto_layer"] is None
    assert collapsed["primary_point_rank"] is None


def test_wall_between_waypoint_collision_and_valid_only_lengths():
    condition = torch.tensor([[-0.9, 0.0, 0.9, 0.0, -0.2, -0.3, 0.2, 0.3]])
    top = torch.tensor([[[-0.9, 0.0], [-0.3, 0.6], [0.3, 0.6], [0.9, 0.0]]])
    bottom = top.clone()
    bottom[:, 1:3, 1] *= -1
    reference = torch.cat((top, bottom))
    straight = top.clone()
    straight[:, 1:3, 1] = 0
    quality = evaluation.wall_quality(
        straight, condition, reference, condition.repeat(2, 1), torch.eye(4)
    )
    assert quality["flat"]["success_fraction"] == 0
    assert quality["flat"]["collision_free_fraction"] == 0
    assert quality["flat"]["both_modes_all_conditions"] == 0
    assert quality["flat"]["valid_path_length"] is None
    assert quality["per_sample"]["route_ids"] == [-1]
    assert "per_sample" not in quality["geometry"]


class _Network(torch.nn.Module):
    def forward(self, sample, timestep=None, *, conditions=None):
        return sample


def test_count_actual_nfe_accounts_for_batches_and_hooks_are_removed():
    model = _Network()
    noise, conditions = torch.zeros((5, 2)), torch.zeros(5, dtype=torch.long)

    def sampler(model, initial, context, *, nfe, batch_size, **kwargs):
        for offset in range(0, len(initial), batch_size):
            for _ in range(nfe):
                model(
                    initial[offset : offset + batch_size],
                    conditions=context[offset : offset + batch_size],
                )
        return {"samples": initial.clone(), "nfe": nfe}

    result = evaluation.verified_sample(sampler, model, noise, conditions, nfe=2, batch_size=2)
    assert result["actual_network_cost"] == {
        "nfe_per_sample": 2.0,
        "total_model_calls": 6,
        "total_sample_evaluations": 10,
    }
    assert not model._forward_pre_hooks


def test_cfg_doubled_batch_is_two_network_evaluations_not_one():
    model = _Network()
    noise, conditions = torch.zeros((4, 2)), torch.zeros(4, dtype=torch.long)

    def dishonest_sampler(model, initial, context, **kwargs):
        model(torch.cat((initial, initial)), conditions=context.repeat(2))
        return {"samples": initial, "nfe": 1}

    with pytest.raises(RuntimeError, match="Actual network sample-evaluations 8"):
        evaluation.verified_sample(dishonest_sampler, model, noise, conditions, nfe=1)
    assert not model._forward_pre_hooks


def test_nonfinite_outputs_cannot_be_silently_removed_from_metrics():
    reference, targets = _ray_reference()
    corrupted = reference.clone()
    corrupted[0] = float("nan")
    with pytest.raises(ValueError, match="never silently dropped"):
        evaluation.rays_quality(corrupted, targets, reference, targets, torch.eye(2), n_classes=2)


def test_consistency_events_show_real_nonmonotone_times_and_actual_states():
    inputs = torch.tensor([[[2.0, 1.0]], [[0.5, 0.5]]])
    outputs = torch.tensor([[[0.2, 0.1]], [[0.3, 0.2]]])
    result = {
        "nfe": 2,
        "samples": outputs[-1],
        "actual_inputs": inputs,
        "postcall_outputs": outputs,
        "input_times": torch.tensor([1.0, 0.5]),
        "output_times": torch.zeros(2),
    }
    events = evaluation.sampling_events(result, method="consistency")
    assert events["times"].tolist() == [1.0, 0.0, 0.5, 0.0]
    assert events["network_calls_used"] == [0, 1, 1, 2]
    assert torch.equal(
        events["states"], torch.stack((inputs[0], outputs[0], inputs[1], outputs[1]))
    )
    assert "nonmonotonic" in events["time_caveat"]


def test_pareto_fronts_do_not_hide_tradeoffs_or_invent_a_scalar_winner():
    rows = [
        {"method": name, "metrics": {"quality": quality, "cost": cost}}
        for name, quality, cost in [
            ("A", 1.0, 3.0),
            ("B", 3.0, 1.0),
            ("C", 2.0, 4.0),
            ("tieA", 1.0, 3.0),
        ]
    ]
    rows.append({"method": "undefined", "metrics": {"quality": None, "cost": 0.0}})
    layers = evaluation.pareto_layers(rows, (("quality", "min"), ("cost", "min")))
    assert layers == {"undefined": None, "A": 1, "B": 1, "tieA": 1, "C": 2}


def test_statistics_pair_training_seeds_and_ignore_historical_league():
    rows = _primary_rows()
    for row in rows:
        row["quality"]["flat"]["conditional_swd"] = (
            evaluation.SEEDS.index(row["seed"]) + 1.0 + (row["method"] != "cfm")
        )
    # An old excellent score cannot enter the new league or be paired to seed161.
    rows.append(
        {
            "task": "rays",
            "nfe": 1,
            "method": "endpoint",
            "seed": 101,
            "league": "historical_reference",
            "status": "completed",
            "quality": {"flat": {"conditional_swd": 0.0}},
        }
    )
    result = evaluation.aggregate_rankings(rows)
    contrast = next(
        row
        for row in result["paired_vs_warm_cfm"]
        if row["task"] == "rays"
        and row["nfe"] == 1
        and row["method"] == "endpoint"
        and row["metric"] == "conditional_swd"
    )
    assert contrast["n_student_seeds"] == 3
    assert contrast["values"] == [1.0, 1.0, 1.0]
    assert contrast["paired_t_ci"] == [1.0, 1.0]
    assert contrast["training_seeds"] == [161, 262, 363]
    assert result["universal_winner"] is None
    assert "Not familywise-adjusted" in result["uncertainty"]


def test_failed_or_missing_seed_cannot_be_dropped_to_improve_rank_or_ci():
    rows = _primary_rows()
    rows = [row for row in rows if not (row["method"] == "endpoint" and row["seed"] == 363)]
    result = evaluation.aggregate_rankings(rows)
    row = next(
        row
        for row in result["summaries"]
        if row["task"] == "rays" and row["nfe"] == 1 and row["method"] == "endpoint"
    )
    assert row["status"] == "incomplete"
    assert row["primary_point_rank"] is None and row["pareto_layer"] is None
    assert row["seed_statistics"]["conditional_swd"]["paired_t_ci"] is None
    assert row["seed_statistics"]["conditional_swd"]["n_student_seeds"] == 2
    assert not any(
        row["method"] == "endpoint" and row["task"] == "rays" and row["nfe"] == 1
        for row in result["paired_vs_warm_cfm"]
    )


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _gate_fixture(tmp_path, monkeypatch):
    # The original recipe config is a small committed document, not private
    # training artifacts or a constructed dataset/model.
    config = json.loads((SOURCE.parents[1] / "replication/configs/method_zoo_v1.json").read_text())
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation, "SOURCE_PATHS", ("notebooks/15_conditional_method_zoo.ipynb", "model.py")
    )
    _write(tmp_path / "config.json", config)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"tags": ["zoo-api"]},
                "source": ["def numeric(): return 1\n"],
            }
        ]
    }
    _write(tmp_path / "notebooks/15_conditional_method_zoo.ipynb", notebook)
    (tmp_path / "model.py").write_text("frozen model source\n")
    registration = {
        "methods": list(evaluation.METHODS),
        "tasks": list(evaluation.TASKS),
        "training_seeds": list(evaluation.SEEDS),
        "evaluation_network_nfe": list(evaluation.BUDGETS),
        "new_method_runs": 48,
        "parents": {task: {"sha256": "a" * 64} for task in evaluation.TASKS},
    }
    _write(tmp_path / "registration.json", registration)
    monkeypatch.setattr(
        evaluation, "REGISTRATION_SHA256", evaluation._file_hash(tmp_path / "registration.json")
    )
    registration_ref = evaluation._file_ref("registration.json")
    source_refs = [evaluation._file_ref(path) for path in evaluation.SOURCE_PATHS]
    digest = evaluation.hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    freeze = {
        "registration": registration_ref,
        "source_refs": source_refs,
        "config": {
            "path": "config.json",
            "file_sha256": evaluation._file_hash("config.json"),
            "resolved_sha256": digest,
        },
    }
    _write(tmp_path / "source_freeze.json", freeze)
    freeze_ref = evaluation._file_ref("source_freeze.json")
    _write(
        tmp_path / "selection.json",
        {
            "policy": "fixed_final_EMA_no_search",
            "registration": registration_ref,
            "source_freeze": freeze_ref,
            "checkpoint_selection_on_validation": False,
            "hyperparameter_search": False,
        },
    )
    _write(
        tmp_path / "fixed.json",
        {
            "policy": "fixed_final_EMA_no_search",
            "registration_sha256": evaluation.REGISTRATION_SHA256,
            "chosen_exports": [],
        },
    )
    runs = [
        {"task": task, "method": method, "seed": seed}
        for task, method, seed in sorted(evaluation._expected_runs())
    ]
    gate = {
        "registration": registration_ref,
        "training_config": evaluation._file_ref("config.json"),
        "training_source_freeze": freeze_ref,
        "source_refs": source_refs,
        "validation_selection": evaluation._file_ref("selection.json"),
        "fixed_exports": evaluation._file_ref("fixed.json"),
        "evaluation": evaluation.EVALUATION,
        "run_root": "runs",
        "runs": runs,
    }
    return gate


def test_exact_cartesian_final_gate_rejects_duplicate_even_with48_records(tmp_path, monkeypatch):
    gate = _gate_fixture(tmp_path, monkeypatch)
    gate["runs"][-1] = gate["runs"][0].copy()
    _write(tmp_path / "gate.json", gate)
    touched = []
    api = type("API", (), {"load_zoo_model": lambda *a, **k: touched.append(True)})()
    with pytest.raises(RuntimeError, match="ALL48 distinct Cartesian"):
        evaluation.verify_final_gate(tmp_path / "gate.json", api=api)
    assert not touched


def test_failed_first_child_keeps_all_native_model_loading_closed(tmp_path, monkeypatch):
    gate = _gate_fixture(tmp_path, monkeypatch)
    row = gate["runs"][0]
    directory = tmp_path / "runs" / row["task"] / f"seed{row['seed']}" / row["method"]
    _write(
        directory / "manifest.json",
        {"task": row["task"], "method": row["method"], "seed": row["seed"], "status": "failed"},
    )
    row.update(
        directory=evaluation._relative(directory),
        manifest_sha256=evaluation._file_hash(directory / "manifest.json"),
    )
    _write(tmp_path / "gate.json", gate)
    touched = []
    api = type("API", (), {"load_zoo_model": lambda *a, **k: touched.append(True)})()
    with pytest.raises(RuntimeError, match="incomplete or mismatched status"):
        evaluation.verify_final_gate(tmp_path / "gate.json", api=api)
    assert not touched


def _ui_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "notebooks/15_conditional_method_zoo.ipynb"
    old = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"tags": ["zoo-api"]},
                "source": ["def numeric(): return 1\n"],
            },
            {
                "cell_type": "code",
                "metadata": {"tags": ["cached_gif_export"]},
                "source": ["def frame():\n        axis.clear()\n"],
            },
        ]
    }
    _write(path, old)
    original_ref = evaluation._file_ref(path)
    _write(tmp_path / "snapshot.ipynb", old)
    new = json.loads(json.dumps(old))
    new["cells"][1]["source"] = [
        "def frame():\n        figure.clear()\n        axis = figure.add_subplot(111)\n"
    ]
    _write(path, new)
    amendment = {
        "kind": "ui_only_gif_colorbar_lifecycle_fix",
        "training_notebook": evaluation._file_ref("snapshot.ipynb"),
        "current_notebook": evaluation._file_ref(path),
        "changed_cell_indices": [1],
        "changed_cell_tags": ["cached_gif_export"],
        "zoo_api_source_sha256_before": "a" * 64,
        "zoo_api_source_sha256_after": "a" * 64,
    }
    _write(tmp_path / "amendment.json", amendment)
    return {"source_refs": [original_ref]}, amendment, new


def test_ui_only_notebook_amendment_preserves_original_training_sha(tmp_path, monkeypatch):
    freeze, amendment, _ = _ui_fixture(tmp_path, monkeypatch)
    refs, sha = evaluation._verify_training_sources(freeze, evaluation._file_ref("amendment.json"))
    assert sha == freeze["source_refs"][0]["sha256"]
    assert (
        refs["notebooks/15_conditional_method_zoo.ipynb"]["sha256"]
        == amendment["current_notebook"]["sha256"]
    )
    assert refs["notebooks/15_conditional_method_zoo.ipynb"]["sha256"] != sha


def test_ui_amendment_cannot_hide_any_changed_training_cell(tmp_path, monkeypatch):
    freeze, amendment, new = _ui_fixture(tmp_path, monkeypatch)
    new["cells"][0]["source"] = ["def numeric(): return 2\n"]
    path = tmp_path / "notebooks/15_conditional_method_zoo.ipynb"
    _write(path, new)
    amendment["current_notebook"] = evaluation._file_ref(path)
    _write(tmp_path / "amendment.json", amendment)
    with pytest.raises(RuntimeError, match="numeric or unapproved"):
        evaluation._verify_training_sources(freeze, evaluation._file_ref("amendment.json"))


def test_ui_amendment_cannot_change_more_than_the_exact_callback_lines(tmp_path, monkeypatch):
    freeze, amendment, new = _ui_fixture(tmp_path, monkeypatch)
    new["cells"][1]["source"].append("        samples.clamp_(-1,1)\n")
    path = tmp_path / "notebooks/15_conditional_method_zoo.ipynb"
    _write(path, new)
    amendment["current_notebook"] = evaluation._file_ref(path)
    _write(tmp_path / "amendment.json", amendment)
    with pytest.raises(RuntimeError, match="beyond the exact"):
        evaluation._verify_training_sources(freeze, evaluation._file_ref("amendment.json"))


def test_cached_preview_gif_uses_actual_cm_clocks_without_model_or_resampling(tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image

    first = torch.tensor([[1.5, -0.5], [1.1, -0.4]])
    final = torch.tensor([[0.3, 0.7], [0.2, 0.8]])
    euler = torch.stack((first, (first + final) / 2, final))
    cm = torch.stack((first, final * 0.8, first * 0.4, final))
    payload = {
        "task": "rays",
        "conditions": torch.zeros(2, dtype=torch.long),
        "condition_ids": torch.zeros(2, dtype=torch.long),
        "reference_samples": final,
        "reference_conditions": torch.zeros(2, dtype=torch.long),
        "runs": [
            {
                "method": "cfm",
                "nfe": 2,
                "seed": 161,
                "league": "primary",
                "events": {
                    "states": euler,
                    "times": torch.tensor([1.0, 0.5, 0.0]),
                    "kinds": ["input", "euler", "euler"],
                    "network_calls_used": [0, 1, 2],
                },
            },
            {
                "method": "consistency",
                "nfe": 2,
                "seed": 161,
                "league": "primary",
                "events": {
                    "states": cm,
                    "times": torch.tensor([1.0, 0.0, 0.5, 0.0]),
                    "kinds": ["input", "denoise", "renoise", "denoise"],
                    "network_calls_used": [0, 1, 1, 2],
                },
            },
        ],
    }
    before = torch.get_rng_state().clone()
    fig, axes = evaluation.plot_method_zoo_preview(payload, nfe=2, frame_index=2)
    assert "renoise; t=0.500" in axes.flat[1].get_title()
    assert "network calls 1/2" in axes.flat[1].get_title()
    assert any("not by a shared physical time" in text.get_text() for text in fig.texts)
    plt.close(fig)
    paths = evaluation.render_method_zoo_previews(payload, tmp_path)
    path = next(path for path in paths if path.suffix == ".gif")
    with Image.open(path) as image:
        assert image.n_frames == 4
    assert torch.equal(torch.get_rng_state(), before)
