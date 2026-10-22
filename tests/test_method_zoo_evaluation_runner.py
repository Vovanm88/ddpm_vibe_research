"""Synthetic driver contracts; never open registered data or run trained models."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_lab.models import PointDenoiser
from recheck.runtime import model_hash, tensor_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "zoo_evaluation_driver", ROOT / "examples/run_method_zoo_evaluation.py"
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
METHODS = (
    "cfm",
    "ot_cfm",
    "endpoint",
    "progressive",
    "reflow",
    "consistency",
    "shortcut",
    "meanflow",
)


def fake_gate(tmp_path):
    runs, manifests = [], {}
    for task in ("rays", "wall"):
        for method in METHODS:
            for seed in (161, 262, 363):
                directory = tmp_path / task / method / str(seed)
                directory.mkdir(parents=True)
                manifest = {
                    "training_cost": {"student_forward_calls": 2000},
                    "training_seconds": 1.0,
                }
                (directory / "manifest.json").write_text(json.dumps(manifest))
                manifests[(task, method, seed)] = manifest
                exports = {}
                for nfe in (1, 2, 4):
                    exports[str(nfe)] = {
                        "path": str(directory / f"native{nfe}"),
                        "sha256": f"{task}_{method}_{seed}_{nfe}",
                        "cumulative_updates_at_export": {1: 2000, 2: 1600, 4: 1200}[nfe]
                        if method == "progressive"
                        else 2000,
                    }
                runs.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "directory": str(directory),
                        "inference_exports": exports,
                    }
                )
    historical = [
        {
            "task": task,
            "method": method,
            "seed": seed,
            "nfe": nfe,
            "path": str(tmp_path / "historical"),
            "sha256": "historical",
        }
        for task in ("rays", "wall")
        for method in ("M0", "E0")
        for seed in ((101, 202, 303) if task == "rays" else (161,))
        for nfe in (2, 4)
    ]
    gate = {
        "runs": runs,
        "historical": historical,
        "parents": {
            task: {"path": str(tmp_path / task / "parent"), "sha256": "parent"}
            for task in ("rays", "wall")
        },
    }
    return gate, manifests


def tiny_banks():
    banks = {}
    for task in ("rays", "wall"):
        conditions = torch.tensor([0, 0, 1, 1])
        ids = conditions.clone()
        if task == "wall":
            scene = torch.tensor(
                [
                    [-0.8, 0.0, 0.8, 0.0, -0.1, -0.4, 0.1, 0.4],
                    [-0.8, 0.1, 0.8, 0.1, -0.1, -0.4, 0.1, 0.4],
                ]
            )
            conditions = scene.repeat_interleave(2, dim=0)
            noise = torch.zeros(4, 30, 2)
            reference = torch.cat(
                (
                    conditions[:, :2, None].transpose(1, 2),
                    noise,
                    conditions[:, 2:4, None].transpose(1, 2),
                ),
                dim=1,
            )
        else:
            noise = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).repeat(2, 1)
            reference = noise.clone()
        bank = {
            "noise": noise,
            "conditions": conditions,
            "condition_ids": ids,
            "reference": reference,
            "reference_conditions": conditions.clone(),
            "extra_noise": torch.ones(3, *noise.shape),
            "projections": torch.eye(2 if task == "rays" else 60),
            "reference_seed": 0,
            "noise_seed": 1,
            "extra_noise_seed": 2,
            "projection_seed": 3,
        }
        bank["hashes"] = {
            key: tensor_hash(value) for key, value in bank.items() if torch.is_tensor(value)
        }
        banks[task] = bank
    return banks


def fake_quality(samples, conditions, *args, **kwargs):
    categories = conditions[:, None] if conditions.ndim == 1 else conditions
    unique = torch.unique(categories, dim=0)
    result = {
        "flat": {"conditional_swd": 0.0},
        "distribution": {
            "mean": 0.0,
            "per_condition": [0.0] * len(unique),
            "conditions": unique.tolist(),
        },
    }
    if conditions.ndim == 1:
        result["routing"] = {
            "per_class": [{"target_id": int(x), "count": 2} for x in unique.flatten()]
        }
    else:
        result["per_sample"] = {
            "success": [False] * len(samples),
            "route_ids": [-1] * len(samples),
            "collision_free": [False] * len(samples),
        }
        result["geometry"] = {
            "mode_coverage": {
                "condition_groups": [
                    {
                        "condition": condition.tolist(),
                        "successful_mode_counts": [0, 0],
                        "success_fraction": 0.0,
                    }
                    for condition in unique
                ]
            }
        }
    return result


def fake_measure(job, bank, *args, **kwargs):
    samples = bank["noise"].clone() if job["task"] == "rays" else bank["reference"].clone()
    if job["method"] == "consistency":
        times = torch.tensor(
            [v for index in range(job["nfe"]) for v in (1 - index / job["nfe"], 0.0)]
        )
    else:
        times = torch.linspace(1, 0, job["nfe"] + 1)
    events = {
        "states": samples.unsqueeze(0).expand(len(times), *samples.shape).clone(),
        "times": times,
        "kinds": ["actual_unit_fixture"] * len(times),
        "network_calls_used": list(range(len(times))),
    }
    return (
        samples,
        events,
        {
            "actual_network_cost": {"nfe_per_sample": job["nfe"]},
            "actual_device": "cpu",
            "parameter_count": 1,
            "sampling_seconds": 0.0,
        },
    )


def test_closed_flag_never_loads_gate_or_data(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "evaluation_core", lambda: pytest.fail("Core must not be loaded"))
    monkeypatch.setattr(driver, "build_final_banks", lambda *args: pytest.fail("Final bank opened"))
    with pytest.raises(PermissionError, match="remains closed"):
        driver.run_evaluation(tmp_path / "gate.json")


def test_incomplete_gate_fails_before_final_factory_or_output(monkeypatch, tmp_path):
    def closed(*args, **kwargs):
        raise RuntimeError("ALL48 incomplete")

    monkeypatch.setattr(
        driver, "evaluation_core", lambda: SimpleNamespace(verify_final_gate=closed)
    )
    monkeypatch.setattr(driver, "training_api", lambda: SimpleNamespace())
    monkeypatch.setattr(driver, "build_final_banks", lambda *args: pytest.fail("Final bank opened"))
    with pytest.raises(RuntimeError, match="ALL48"):
        driver.run_evaluation(
            tmp_path / "gate.json", tmp_path / "public", tmp_path / "private", allow_final=True
        )
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "private").exists()


def test_jobs_are_exact_168_and_use_progressive_gate_mapping(tmp_path):
    gate, _ = fake_gate(tmp_path)
    jobs = driver.final_jobs(gate)
    assert len(jobs) == 168
    assert sum(j["league"] == "primary" for j in jobs) == 144
    assert sum(j["league"] == "historical_reference" for j in jobs) == 16
    assert jobs[0]["task"] == "rays" and jobs[0]["nfe"] == 32
    progressive = next(j for j in jobs if j["method"] == "progressive" and j["nfe"] == 4)
    assert progressive["export_cumulative_updates"] == 1200
    assert progressive["export"]["path"].endswith("native4")
    gate["runs"][-1] = gate["runs"][0]
    with pytest.raises(ValueError, match="exactly144"):
        driver.final_jobs(gate)


def test_registered_banks_use_spied_factories_and_paired_context_noise(monkeypatch):
    calls = []

    def rays(name, **kwargs):
        calls.append(("rays", kwargs))
        return SimpleNamespace(samples=torch.zeros(8192, 2), conditions=torch.arange(8192) % 16)

    def wall(**kwargs):
        calls.append(("wall", kwargs))
        scene = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
        return SimpleNamespace(
            samples=torch.zeros(4096, 32, 2),
            conditions=scene.repeat_interleave(128, dim=0),
            scene_conditions=scene,
        )

    monkeypatch.setattr(driver, "make_environment", rays)
    monkeypatch.setattr(driver, "make_multimodal_navigation", wall)
    registration = {
        "final": {
            "rays": {
                "reference_samples": 8192,
                "reference_seed": 30047,
                "generated_per_condition": 512,
                "noise_seed": 945001,
                "total_generated": 8192,
            },
            "wall": {
                "n_scenes": 32,
                "paths_per_scene": 128,
                "scene_seed": 30061,
                "noise_seed": 615001,
                "total_generated": 4096,
            },
        }
    }
    core = SimpleNamespace(
        EVALUATION={
            "rays": {"extra_noise_seed": 945002, "projection_count": 2, "projection_seed": 1717},
            "wall": {"extra_noise_seed": 615002, "projection_count": 2, "projection_seed": 861003},
        }
    )
    banks = driver.build_final_banks(registration, core)
    assert calls[0][1]["seed"] == 30047
    assert calls[1][1]["seed"] == 30061 and calls[1][1]["split"] == "test"
    assert torch.equal(banks["rays"]["noise"][:512], banks["rays"]["noise"][512:1024])
    assert torch.equal(banks["wall"]["noise"][:128], banks["wall"]["noise"][128:256])
    assert torch.equal(
        banks["rays"]["extra_noise"][:, :512], banks["rays"]["extra_noise"][:, 512:1024]
    )
    assert banks["wall"]["conditions"].shape == (4096, 8)
    expected = torch.randn(512, 2, generator=torch.Generator().manual_seed(945001))
    assert torch.equal(banks["rays"]["noise"][:512], expected)


def test_outcomes_preserve_undefined_origin_and_do_not_publish_xy():
    bank = tiny_banks()["rays"]
    samples = torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.7]])
    arrays = driver.outcome_arrays("rays", samples, bank, {})
    assert arrays["nearest_geometric_id"][0] == -1
    assert not arrays["assignment_defined"][0]
    assert np.isnan(arrays["requested_phase_error_radians"][0])
    assert "samples" not in arrays and "endpoints" not in arrays
    assert all(array.ndim == 1 for array in arrays.values())


def test_preview_is_only_first16_of_condition_zero():
    bank = tiny_banks()["rays"]
    bank["conditions"] = torch.arange(4).repeat_interleave(20)
    bank["condition_ids"] = bank["conditions"].clone()
    payload, selected = driver._preview_payload("rays", bank)
    assert selected.tolist() == list(range(16))
    assert payload["conditions"].tolist() == [0] * 16


def test_per_condition_csv_uses_actual_target_id_key():
    bank = tiny_banks()["rays"]
    quality = fake_quality(bank["noise"], bank["conditions"])
    job = {"task": "rays", "method": "cfm", "seed": 161, "nfe": 1, "league": "primary"}
    rows = driver.per_condition_rows(job, quality)
    assert [row["target_id"] for row in rows] == [0, 1]


def test_real_native_sampler_parent32_reference_flag_is_consumed_by_core(tmp_path):
    """Actual native model + notebook sampler, arbitrary synthetic noise only."""
    torch.set_num_threads(1)
    torch.manual_seed(119)
    model = PointDenoiser(hidden_dim=8, num_layers=1, conditioner="embedding", num_classes=16)
    path = tmp_path / "ema_model"
    model.save_pretrained(path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "task": "rays",
                "method": "teacher",
                "ema_sha256": model_hash(model),
            }
        )
    )
    export = {
        "path": str(path),
        "sha256": model_hash(model),
        "file_sha256": driver.file_hash(path / "diffusion_pytorch_model.safetensors"),
        "config_file_sha256": driver.file_hash(path / "config.json"),
    }
    job = {"task": "rays", "method": "Original", "nfe": 32, "export": export}
    samples, events, measure = driver._measure_job(
        job,
        tiny_banks()["rays"],
        driver.training_api(),
        driver.evaluation_core(),
        device="cpu",
        batch_size=2,
    )
    assert samples.shape == (4, 2) and torch.isfinite(samples).all()
    assert events["states"].shape == (33, 4, 2)
    assert measure["actual_network_cost"]["nfe_per_sample"] == 32
    assert measure["actual_network_cost"]["total_sample_evaluations"] == 128
    assert measure["actual_network_cost"]["total_model_calls"] == 64


def test_export_file_mutation_fails_before_loader(tmp_path):
    path = tmp_path / "native"
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"unit")
    export = {
        "path": str(path),
        "sha256": "unused",
        "config_file_sha256": "wrong",
        "file_sha256": driver.file_hash(path / "diffusion_pytorch_model.safetensors"),
    }
    api = SimpleNamespace(
        load_zoo_model=lambda *args, **kwargs: pytest.fail("Mutated export loaded")
    )
    with pytest.raises(RuntimeError, match="file changed"):
        driver._measure_job(
            {"export": export}, {}, api, SimpleNamespace(), device="cpu", batch_size=2
        )


def test_full_synthetic_driver_order_outputs_and_nonmonotone_events(monkeypatch, tmp_path):
    gate, manifests = fake_gate(tmp_path / "runs")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}")
    order, ranked = [], []

    def verify(*args, **kwargs):
        order.append("verified_all48")
        assert kwargs["verify_native"] is True
        return {
            "gate": gate,
            "registration": {},
            "manifests": manifests,
            "all48_metadata_fixed": True,
            "all144_exports_fixed": True,
            "actual_native_hashes_verified": True,
        }

    core = SimpleNamespace(
        __file__=driver.__file__,
        verify_final_gate=verify,
        rays_quality=fake_quality,
        wall_quality=fake_quality,
        conditional_swd=lambda *args: {"mean": 0.0},
        save_ranking_artifacts=lambda rows, output: ranked.append(len(rows)),
    )
    monkeypatch.setattr(driver, "evaluation_core", lambda: core)
    monkeypatch.setattr(driver, "training_api", lambda: SimpleNamespace())

    def banks(*args):
        order.append("final_factories")
        return tiny_banks()

    monkeypatch.setattr(driver, "build_final_banks", banks)
    monkeypatch.setattr(driver, "_measure_job", fake_measure)
    result = driver.run_evaluation(
        gate_path,
        tmp_path / "public",
        tmp_path / "private",
        allow_final=True,
        threads=1,
        render=False,
    )
    assert order == ["verified_all48", "final_factories"]
    assert result["metadata"]["status"] == "completed" and len(result["rows"]) == 168
    assert ranked == [168]
    assert len(list((tmp_path / "public/outcomes").glob("*.npz"))) == 168
    assert all("per_sample" not in row["quality"] for row in result["rows"])
    payload = torch.load(tmp_path / "public/previews/rays_preview.pt", weights_only=True)
    assert len(payload["runs"]) == 28  # eight methods x three budgets, four parents
    assert all(row["seed"] in (None, 161) for row in payload["runs"])
    cm = next(row for row in payload["runs"] if row["method"] == "consistency" and row["nfe"] == 4)
    assert cm["events"]["times"].tolist() == [1.0, 0.0, 0.75, 0.0, 0.5, 0.0, 0.25, 0.0]
    assert (tmp_path / "public/per_condition.csv").exists()
    first = next((tmp_path / "public/outcomes").glob("*.npz"))
    with np.load(first, allow_pickle=False) as arrays:
        assert "condition_ids" in arrays and all(array.ndim == 1 for array in arrays.values())
