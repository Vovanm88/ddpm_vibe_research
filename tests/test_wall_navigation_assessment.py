"""Synthetic guards for the separate wall pilot; never create real Test scenes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "examples/evaluate_wall_navigation_students.py"
SPEC = importlib.util.spec_from_file_location("wall_final_assessment", SOURCE)
assessment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assessment)


def test_default_plan_cannot_open_data_or_load_weights(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("An inert plan must not open data or model files")

    monkeypatch.setattr(assessment, "make_multimodal_navigation", forbidden)
    monkeypatch.setattr(assessment, "preflight_manifests", forbidden)
    monkeypatch.setattr(assessment, "load_frozen_models", forbidden)
    monkeypatch.setattr("sys.argv", [str(SOURCE)])
    assessment.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "prepared_not_executed"
    assert plan["test"]["seed"] == 20061
    assert plan["test"]["split"] == "test"
    assert plan["noise_seed"] == 614001
    assert len(plan["children"]) == 4


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _cohort_fixture(root):
    """Tiny metadata fixtures; no actual dataset, optimizer or native model."""
    registration = {
        "teacher_sha256": assessment.EXPECTED_TEACHER,
        "teacher_updates": 20000,
        "methods": ["M0", "E0"],
        "student_nfe": [4, 2],
        "teacher_target_nfe": 32,
        "seed": 161,
        "updates_each": 1000,
        "batch_size": 128,
        "warmup": 100,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "lambda_w": 1.0,
        "state_dimension": 60,
        "endpoint_clamped_contract": True,
        "route_id_input": False,
        "whole_ot_support_single_continuous_condition": True,
        "train_data_sha256": "a" * 64,
        "conditions_sha256": "b" * 64,
        "test_preregistered_not_yet_accessed": {
            "n_scenes": 32,
            "samples_per_scene": 128,
            "path_length": 32,
            "seed": 20061,
            "split": "test",
            "noise_seed": 614001,
        },
    }
    _write(root / "child_study_registration.json", registration)
    for name in ("amendment.json", "backend.py", "notebook.ipynb"):
        (root / name).write_text("frozen synthetic provenance\n")

    def reference(path):
        return {"path": str(path), "sha256": assessment._file_hash(path)}

    backend = reference(root / "backend.py") | {
        "name": "exact_epsilon_continuation",
        "tolerance": 1e-10,
        "dtype": "float64",
    }
    cohort = {
        "teacher_sha256": assessment.EXPECTED_TEACHER,
        "original_registration": reference(root / "child_study_registration.json"),
        "amendment": reference(root / "amendment.json"),
        "solver_backend": backend,
        "notebook": reference(root / "notebook.ipynb"),
        "children": [
            {
                "name": f"{method} N{nfe}",
                "student_nfe": nfe,
                "method": method,
                "directory": f"students_nfe{nfe}/{method}",
            }
            for nfe, method in assessment.CHILDREN
        ],
        "test": registration["test_preregistered_not_yet_accessed"],
        "expected_trace_contract": {
            "updates": 1000,
            "seed": 161,
            "batch_size": 128,
            "paired_keys": ["scene", "interval", "data_indices", "noise"],
            "equal_within_nfe": True,
        },
    }
    _write(root / "final_cohort_registration.json", cohort)
    _write(
        root / "teacher/manifest.json",
        {
            "status": "completed",
            "updates_completed": 20000,
            "teacher_sha256": assessment.EXPECTED_TEACHER,
            "route_ids_used_by_model": False,
        },
    )
    for nfe, method in assessment.CHILDREN:
        _write(
            root / f"students_nfe{nfe}/{method}/manifest.json",
            {
                "status": "completed",
                "updates_completed": 1000,
                "updates": 1000,
                "teacher_sha256": assessment.EXPECTED_TEACHER,
                "seed": 161,
                "student_steps": nfe,
                "teacher_steps": 32,
                "batch_size": 128,
                "warmup": 100,
                "lr": 0.0001,
                "weight_decay": 0.0001,
                "ema_decay": 0.99,
                "grad_clip": 1.0,
                "lambda_w": 1.0,
                "gamma": 1.0,
                "state_dimension": 60,
                "endpoint_clamped_contract": True,
                "whole_support_single_condition": True,
                "route_id_input": False,
                "family": None if method == "M0" else "sinkhorn",
                "representation": "endpoint",
                "train_sha256": "a" * 64,
                "conditions_sha256": "b" * 64,
                "calibration_sha256": "e" * 64,
                "ema_sha256": assessment._state_hash({"weight": torch.ones(1)}),
                "solver_backend": backend,
                "solver_amendment_sha256": cohort["amendment"]["sha256"],
            },
        )
    model_directories = [root / "teacher"] + [
        root / child["directory"] for child in cohort["children"]
    ]
    for directory in model_directories:
        _write(
            directory / "ema_model/config.json",
            {
                "sample_shape": [30, 2],
                "condition_dim": 8,
                "conditioner": "mlp",
                "num_classes": 0,
            },
        )
        (directory / "ema_model/diffusion_pytorch_model.safetensors").write_bytes(
            b"dummy native export"
        )
        if directory.name in ("M0", "E0"):
            torch.save(
                {
                    "status": "completed",
                    "updates_completed": 1000,
                    "pairing_trace": [f"{index:064x}" for index in range(1000)],
                    "contract": json.loads((directory / "manifest.json").read_text()),
                    "ema": {"weight": torch.ones(1)},
                },
                directory / "training.pt",
            )


def test_failed_child_manifest_keeps_factory_and_model_loading_closed(monkeypatch, tmp_path):
    _cohort_fixture(tmp_path)
    failed = tmp_path / "students_nfe2/E0/manifest.json"
    manifest = json.loads(failed.read_text())
    manifest.update(status="failed", updates_completed=1)
    _write(failed, manifest)
    calls = []
    monkeypatch.setattr(assessment, "load_frozen_models", lambda *a: calls.append("weights"))
    monkeypatch.setattr(
        assessment, "make_multimodal_navigation", lambda *a, **k: calls.append("data")
    )
    with pytest.raises(RuntimeError, match="Closed Test: E0 N2 status"):
        assessment.run_assessment(tmp_path, tmp_path / "output", tmp_path / "cache")
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_mismatched_training_provenance_closes_test_before_native_exports(tmp_path):
    _cohort_fixture(tmp_path)
    path = tmp_path / "students_nfe4/E0/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["train_sha256"] = "d" * 64
    _write(path, manifest)
    with pytest.raises(RuntimeError, match="train_sha256"):
        assessment.preflight_manifests(tmp_path)


def test_explicit_cohort_paths_cannot_escape_or_select_unregistered_models(tmp_path):
    _cohort_fixture(tmp_path)
    path = tmp_path / "final_cohort_registration.json"
    cohort = json.loads(path.read_text())
    cohort["children"][-1]["directory"] = "../another_experiment/E0"
    _write(path, cohort)
    with pytest.raises(RuntimeError, match="child paths must be distinct and inside"):
        assessment.preflight_manifests(tmp_path)


def test_changed_solver_source_prevents_test_even_if_child_manifests_complete(tmp_path):
    _cohort_fixture(tmp_path)
    (tmp_path / "backend.py").write_text("silently changed backend")
    with pytest.raises(RuntimeError, match="solver_backend source SHA mismatch"):
        assessment.preflight_manifests(tmp_path)


def test_real_saved_pairing_trace_mismatch_rejects_completed_cohort(tmp_path):
    _cohort_fixture(tmp_path)
    checkpoint = tmp_path / "students_nfe2/E0/training.pt"
    saved = torch.load(checkpoint, weights_only=True)
    saved["pairing_trace"][17] = "f" * 64
    torch.save(saved, checkpoint)
    with pytest.raises(RuntimeError, match="unequal paired scene/interval/data/noise"):
        assessment.preflight_manifests(tmp_path)


def test_complete_explicit_cohort_passes_data_free_provenance_preflight(tmp_path):
    _cohort_fixture(tmp_path)
    preflight = assessment.preflight_manifests(tmp_path)
    assert preflight["all_four_training_traces_verified"]
    assert len(preflight["models"]) == 5
    assert (
        preflight["models"]["M0 N2"]["paired_trace_sha256"]
        == preflight["models"]["E0 N2"]["paired_trace_sha256"]
    )


def test_actual_tensor_hash_failure_precedes_any_data_factory(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    preflight = {
        "models": {
            "E0 N2": {
                "directory": str(tmp_path),
                "expected_model_sha256": "0" * 64,
            }
        }
    }
    monkeypatch.setattr(assessment, "preflight_manifests", lambda root: preflight)
    monkeypatch.setattr(
        assessment, "PointDenoiser", SimpleNamespace(from_pretrained=lambda *a, **k: model)
    )
    touched = []
    monkeypatch.setattr(
        assessment, "make_multimodal_navigation", lambda *a, **k: touched.append(True)
    )
    with pytest.raises(RuntimeError, match="native tensor SHA differs"):
        assessment.run_assessment(tmp_path, tmp_path / "output", tmp_path / "cache")
    assert not touched


def test_shared_noise_pairs_ids_across_interleaved_conditions_without_global_rng():
    scene_indices = torch.tensor([1, 0, 1, 0])
    before = torch.get_rng_state().clone()
    noise = assessment.shared_scene_noise(scene_indices, samples_per_scene=2, seed=42)
    assert torch.equal(noise[scene_indices == 0], noise[scene_indices == 1])
    assert not torch.equal(noise[0], noise[2])
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(ValueError, match="Each registered condition"):
        assessment.shared_scene_noise(torch.tensor([0, 0, 1]), samples_per_scene=2)


def test_raw_euler_history_retains_constructed_endpoints_and_actual_nodes():
    class ConstantVelocity:
        def __call__(self, sample, timestep, *, conditions):
            return SimpleNamespace(sample=torch.ones_like(sample))

    noise = torch.zeros((2, 30, 2))
    conditions = torch.tensor([[-0.9, 0.1, 0.9, -0.1, -0.2, -0.3, 0.2, 0.3]]).repeat(2, 1)
    history = assessment.sample_navigation(
        ConstantVelocity(), noise, conditions, steps=2, batch_size=1
    )
    assert history.shape == (3, 2, 32, 2)
    assert torch.equal(history[:, :, 0], conditions[None, :, :2].expand(3, -1, -1))
    assert torch.equal(history[:, :, -1], conditions[None, :, 2:4].expand(3, -1, -1))
    assert torch.equal(history[1, :, 1:-1], torch.full_like(noise, -0.5))
    assert torch.equal(history[-1, :, 1:-1], -torch.ones_like(noise))
    assert torch.equal(noise, torch.zeros_like(noise))


def test_conditional_distribution_does_not_hide_condition_swaps_in_marginal():
    real = torch.zeros((4, 32, 2))
    real[2:, 1:-1] = 1
    generated = real.flip(0)
    scenes = torch.tensor([0, 0, 1, 1])
    projections = torch.ones((1, 60), dtype=torch.float64) / 60**0.5
    result = assessment.distribution_metrics(real, generated, scenes, projections)
    assert result["marginal_sliced_w1_interior_real"] == pytest.approx(0)
    assert result["mean_scene_sliced_w1_interior_real"] == pytest.approx(60**0.5)


def test_compact_endpoint_export_clones_history_views(tmp_path):
    histories = {
        "Original N32": torch.arange(33 * 4 * 32 * 2, dtype=torch.float32).reshape(33, 4, 32, 2)
    }
    noise = torch.zeros((4, 30, 2))
    conditions = torch.zeros((4, 8))
    real = torch.zeros((4, 32, 2))
    scenes = torch.tensor([0, 0, 1, 1])
    preview, _ = assessment.save_public_payload(
        tmp_path,
        histories,
        noise=noise,
        conditions=conditions,
        real_paths=real,
        scene_indices=scenes,
    )
    endpoints = torch.load(tmp_path / "final_endpoints.pt", weights_only=True)
    endpoint = endpoints["endpoints_by_method"]["Original N32"]
    assert torch.equal(endpoint, histories["Original N32"][-1])
    assert endpoint.untyped_storage().nbytes() == endpoint.numel() * endpoint.element_size()
    assert (tmp_path / "final_endpoints.pt").stat().st_size < 16000
    assert preview["Original N32"].shape == (33, 2, 32, 2)
    assert preview["Original N32"].untyped_storage().nbytes() == preview["Original N32"].numel() * 4


def test_descriptive_contrasts_do_not_invent_seed_confidence():
    rows = []
    for nfe in (4, 2):
        for method, success in (("Original", 0.3), ("M0", 0.8), ("E0", 0.7)):
            rows.append(
                {
                    "method": f"{method} N{nfe}",
                    "quality": {
                        "success_fraction": success,
                        "collision_free_fraction": success,
                        "mode_coverage": {"both_modes_fraction_all_conditions": 1.0},
                    },
                    "distribution": {"mean_scene_sliced_w1_interior_real": 1 - success},
                    "fidelity_to_teacher32": {"paired_parent_interior_mse": 1 - success},
                }
            )
    contrasts = assessment.descriptive_contrasts(rows)
    assert len(contrasts) == 6
    assert contrasts[0]["success_fraction_difference"] == pytest.approx(-0.1)
    assert contrasts[1]["success_fraction_difference"] == pytest.approx(0.5)
    assert all(
        row["training_seeds"] == 1 and row["confidence_interval"] is None for row in contrasts
    )
