"""Synthetic/fake-file gate checks; never open experiment test or run a model."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from recheck.config import config_hash, load_config
from recheck.distill import experiment_schedule
from recheck.runtime import tensor_hash

SCRIPT = Path(__file__).resolve().parents[1] / "examples/evaluate_ray_geometry.py"
SPEC = importlib.util.spec_from_file_location("ray_geometry_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def edit_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    write_json(path, value)


class FakeExport:
    sample_shape = (2,)

    def __init__(self, path):
        self.digest = (Path(path) / "diffusion_pytorch_model.safetensors").read_text()

    def eval(self):
        return self

    def requires_grad_(self, value):
        assert value is False
        return self


@pytest.fixture
def fixed_suites(tmp_path, monkeypatch):
    configs = [
        load_config(study.REPO_ROOT / "replication/configs" / name)
        for name in ("many_rays_fourier_cpu.yaml", "many_rays_fourier_stress_32_8.yaml")
    ]
    for index, config in enumerate(configs):
        config["output_dir"] = str(tmp_path / f"suite{index}")
    parent = "a" * 64
    reference = torch.randn((8192, 2), generator=torch.Generator().manual_seed(191)) * 0.4
    reference_hash = tensor_hash(reference)
    noise_hash = tensor_hash(
        torch.randn((4096, 2), generator=torch.Generator().manual_seed(730001))
    )
    events = []

    def native(path, digest, config):
        m = config["model"]
        write_json(
            path / "config.json",
            {
                "_class_name": "FourierFMMLP",
                "input_dim": 2,
                "hidden_dim": m["hidden_dim"],
                "num_layers": m["num_layers"],
                "time_embedding_dim": m["time_embedding_dim"],
                "time_scale": 1000.0,
                "spatial_frequencies": m["spatial_frequencies"],
                "polar_features": m["polar_features"],
                "epsilon": m["spatial_epsilon"],
            },
        )
        (path / "diffusion_pytorch_model.safetensors").write_text(digest)

    for cfg in configs:
        root = Path(cfg["output_dir"])
        write_json(
            root / "teacher/manifest.json",
            {
                "status": "completed",
                "updates_completed": 100000,
                "teacher_sha256": parent,
                "config": configs[0],
                "config_sha256": config_hash(configs[0]),
            },
        )
        native(root / "teacher/ema_model", parent, cfg)
        hashes, rows = {}, []
        for seed in (101, 202, 303):
            for method, loss, rho in (
                ("M0", "mse", 0.0),
                ("M1", "mse", 0.1),
                ("E0", "sinkhorn_endpoint", 0.0),
                ("E1", "sinkhorn_endpoint", 0.1),
            ):
                name = f"seed{seed}_{method}"
                digest = hashlib.sha256(f"{root}:{name}".encode()).hexdigest()
                hashes[name] = digest
                write_json(
                    root / "students" / name / "manifest.json",
                    {
                        "status": "completed",
                        "updates_completed": cfg["student"]["updates"],
                        "config": cfg,
                        "config_sha256": config_hash(cfg),
                        "teacher_sha256": parent,
                        "student_init_sha256": parent,
                        "teacher_sha256_after": parent,
                        "seed": seed,
                        "loss_family": loss,
                        "rho": rho,
                        "schedule": experiment_schedule(cfg).to_dict(),
                        "student_ema_sha256": digest,
                    },
                )
                native(root / "students" / name / "ema_model", digest, cfg)
                rows.append(
                    {
                        "method": method,
                        "student_seed": seed,
                        "nfe": cfg["distillation"]["student_steps"],
                        "model_sha256": digest,
                    }
                )
        rows += [
            {"method": f"Original N{nfe}", "student_seed": None, "nfe": nfe, "model_sha256": parent}
            for nfe in study.BASELINE_NFE
        ]
        for row in rows:
            row.update(
                initial_noise_sha256=noise_hash,
                evaluation_sha256="e" * 64,
                cost={"generated_samples": 4096},
            )
        write_json(
            root / "evaluation/results.json",
            {
                "rows": rows,
                "metadata": {
                    "status": "completed",
                    "evaluation_split": "test",
                    "config": cfg,
                    "config_sha256": config_hash(cfg),
                    "teacher_sha256": parent,
                    "initial_noise_sha256": noise_hash,
                    "student_checkpoint_hashes": hashes,
                    "evaluation_sha256": "e" * 64,
                    "algorithm_sha256": "f" * 64,
                    "quality_context": {
                        "reference_split": "test",
                        "real_samples_sha256": reference_hash,
                    },
                },
            },
        )

    def load(path, device):
        assert device == "cpu"
        events.append(("load_export", str(path)))
        return FakeExport(path)

    monkeypatch.setattr(study, "load_model", load)
    monkeypatch.setattr(study, "model_hash", lambda model: model.digest)
    return configs, reference, events


def test_preflight_accepts_both_full_suites_and_copied_main_teacher_without_data(
    fixed_suites, monkeypatch
):
    configs, _, events = fixed_suites
    monkeypatch.setattr(
        study, "benchmark_data", lambda *a, **k: pytest.fail("Test opened during preflight")
    )
    monkeypatch.setattr(
        study, "sample_euler", lambda *a, **k: pytest.fail("Sampling during preflight")
    )
    records, evidence = study.preflight(configs)
    assert len(records) == 2 and len(evidence) == 80
    assert len(events) == 26
    assert records[1]["teacher"]["config_sha256"] != config_hash(configs[1])


@pytest.mark.parametrize(
    "corruption",
    [
        "teacher_running",
        "teacher_budget",
        "teacher_config_sha",
        "student_running",
        "student_budget",
        "student_missing_init",
        "student_missing_config_sha",
        "student_parent_after",
        "student_schedule",
        "student_seed",
        "student_loss",
        "student_config_body",
        "stale_native_weights",
        "wrong_native_frequency",
        "missing_student_directory",
        "extra_student_directory",
        "core_running",
        "core_wrong_split",
        "core_duplicate_rows",
        "core_missing_row",
        "core_stale_checkpoint_map",
        "core_missing_checkpoint",
        "core_wrong_noise",
        "core_row_wrong_noise",
        "core_row_wrong_nfe",
        "core_row_wrong_model",
        "core_row_wrong_eval_sha",
        "core_truncated_generation",
        "core_config_body",
        "core_wrong_reference",
    ],
)
def test_either_root_corruption_blocks_before_any_test_sampling_or_writes(
    fixed_suites, tmp_path, monkeypatch, corruption
):
    configs, _, _ = fixed_suites
    root = Path(configs[1]["output_dir"])
    teacher = root / "teacher/manifest.json"
    student = root / "students/seed101_E0/manifest.json"
    core = root / "evaluation/results.json"
    if corruption.startswith("teacher_"):
        change = {
            "teacher_running": ("status", "running"),
            "teacher_budget": ("updates_completed", 99999),
            "teacher_config_sha": ("config_sha256", "b" * 64),
        }[corruption]
        edit_json(teacher, lambda value: value.update({change[0]: change[1]}))
    elif corruption.startswith("student_"):

        def mutate(value):
            if corruption == "student_missing_init":
                value.pop("student_init_sha256")
            elif corruption == "student_missing_config_sha":
                value.pop("config_sha256")
            elif corruption == "student_schedule":
                value["schedule"]["times"][1] += 0.01
            elif corruption == "student_config_body":
                value["config"]["student"]["updates"] = 2499
            else:
                key, wrong = {
                    "student_running": ("status", "running"),
                    "student_budget": ("updates_completed", 2499),
                    "student_parent_after": ("teacher_sha256_after", "b" * 64),
                    "student_seed": ("seed", 303),
                    "student_loss": ("loss_family", "mse"),
                }[corruption]
                value[key] = wrong

        edit_json(student, mutate)
    elif corruption == "stale_native_weights":
        (student.parent / "ema_model/diffusion_pytorch_model.safetensors").write_text("b" * 64)
    elif corruption == "wrong_native_frequency":
        edit_json(
            student.parent / "ema_model/config.json",
            lambda value: value.update(spatial_frequencies=[1, 3, 5]),
        )
    elif corruption == "missing_student_directory":
        (root / "students/seed101_E0").rename(root / "students/wrong_name")
    elif corruption == "extra_student_directory":
        (root / "students/seed999_E0").mkdir()
    else:

        def mutate(value):
            meta = value["metadata"]
            if corruption == "core_duplicate_rows":
                value["rows"][-1] = value["rows"][-2]
            elif corruption == "core_missing_row":
                value["rows"].pop()
            elif corruption == "core_stale_checkpoint_map":
                meta["student_checkpoint_hashes"]["seed101_E0"] = "b" * 64
            elif corruption == "core_missing_checkpoint":
                meta["student_checkpoint_hashes"].pop("seed101_E0")
            elif corruption == "core_config_body":
                meta["config"]["student"]["updates"] = 2499
            elif corruption == "core_wrong_reference":
                meta["quality_context"]["real_samples_sha256"] = "b" * 64
            elif corruption == "core_truncated_generation":
                value["rows"][0]["cost"]["generated_samples"] = 4095
            elif corruption.startswith("core_row_"):
                key, wrong = {
                    "core_row_wrong_noise": ("initial_noise_sha256", "b" * 64),
                    "core_row_wrong_nfe": ("nfe", 16),
                    "core_row_wrong_model": ("model_sha256", "b" * 64),
                    "core_row_wrong_eval_sha": ("evaluation_sha256", "b" * 64),
                }[corruption]
                value["rows"][0][key] = wrong
            else:
                key, wrong = {
                    "core_running": ("status", "running"),
                    "core_wrong_split": ("evaluation_split", "validation"),
                    "core_wrong_noise": ("initial_noise_sha256", "b" * 64),
                }[corruption]
                meta[key] = wrong

        edit_json(core, mutate)
    monkeypatch.setattr(
        study, "benchmark_data", lambda *a, **k: pytest.fail("Closed test was accessed")
    )
    monkeypatch.setattr(
        study, "sample_euler", lambda *a, **k: pytest.fail("Sampling before both gates")
    )
    output = tmp_path / "should_not_exist"
    with pytest.raises(RuntimeError, match="remains closed"):
        study.run_study(configs, output)
    assert not output.exists()


def test_shared_parent_hash_must_match_even_if_each_root_is_internally_consistent(fixed_suites):
    configs, _, _ = fixed_suites
    root = Path(configs[1]["output_dir"])
    for file in root.rglob("*.json"):
        file.write_text(file.read_text().replace("a" * 64, "b" * 64))
    (root / "teacher/ema_model/diffusion_pytorch_model.safetensors").write_text("b" * 64)
    with pytest.raises(RuntimeError, match="do not share one frozen teacher"):
        study.preflight(configs)


def synthetic_rows():
    rows = []
    for nfe in (16, 8):
        grid = f"32_to_{nfe}"
        for seed, shift in ((101, 0.01), (202, 0.02), (303, 0.03)):
            for method, effect in (("M0", 0.2), ("M1", 0.17), ("E0", 0.1), ("E1", 0.06)):
                rows.append(
                    {
                        "grid": grid,
                        "method": method,
                        "student_seed": seed,
                        "nfe": nfe,
                        "geometry": {
                            "metrics": {metric: effect + shift for metric in study.METRICS}
                        },
                    }
                )
    return rows


def test_paired_statistics_count_training_seeds_and_preserve_pairing():
    results = study.paired_geometry(synthetic_rows())
    assert len(results) == 30
    contrast = next(
        row for row in results if row["grid"] == "32_to_16" and row["contrast"] == "E0-M0"
    )
    stats = contrast["statistics"]
    assert stats["n_student_seeds"] == 3
    assert stats["values"] == pytest.approx([-0.1, -0.1, -0.1])
    assert stats["mean"] == pytest.approx(-0.1)
    assert stats["sample_std"] == pytest.approx(0, abs=1e-16)
    assert stats["paired_t_ci"] == pytest.approx([-0.1, -0.1])
    interaction = next(row for row in results if row["contrast"] == "tube_interaction")
    assert interaction["statistics"]["mean"] == pytest.approx(-0.01)


def test_undefined_phase_is_explicit_not_silently_dropped_from_three_seeds():
    rows = synthetic_rows()
    rows[0]["geometry"]["metrics"][study.METRICS[0]] = None
    results = study.paired_geometry(rows)
    first = next(
        row
        for row in results
        if row["grid"] == "32_to_16"
        and row["metric"] == study.METRICS[0]
        and row["contrast"] == "E0-M0"
    )
    assert first["statistics"] is None and first["student_seeds"] == [101, 202, 303]
    incomplete = synthetic_rows()[1:]
    with pytest.raises(RuntimeError, match="Incomplete paired"):
        study.paired_geometry(incomplete)


def test_synthetic_end_to_end_writes_raw_hashes_after_both_preflights(
    fixed_suites, tmp_path, monkeypatch
):
    configs, reference, events = fixed_suites
    seen_noise = []

    def data(config, split):
        events.append(("data", split))
        assert split == "test"
        assert len([e for e in events if e[0] == "load_export"]) == 26
        return reference.clone()

    def sample(model, noise, *, times, batch_size, record_history):
        assert batch_size == 128 and record_history is False
        seen_noise.append(tensor_hash(noise))
        return {
            "samples": noise * 0.3,
            "total_model_calls": (len(times) - 1) * 32,
            "initial_noise_sha256": tensor_hash(noise),
        }

    monkeypatch.setattr(study, "benchmark_data", data)
    monkeypatch.setattr(study, "sample_euler", sample)
    monkeypatch.setattr(study, "render_plots", lambda *a, **k: [])
    output = tmp_path / "synthetic_report"
    metadata = study.run_study(configs, output)
    assert metadata["status"] == "completed" and metadata["teacher_updates"] == 100000
    assert len(seen_noise) == 30 and len(set(seen_noise)) == 1
    assert [e for e in events if e[0] == "data"] == [("data", "test")]
    report = json.loads((output / "geometry_study.json").read_text())
    assert len(report["rows"]) == 36 and len(report["paired_contrasts"]) == 30
    assert len(list((output / "geometry").glob("*.json"))) == 36
    assert len(list((output / "endpoints").glob("*.pt"))) == 30
    assert len(metadata["input_files_sha256"]) == 80
    assert metadata["geometry"]["state_dimension"] == 2
    assert metadata["reference"]["sha256"] == tensor_hash(reference)
    saved = torch.load(next((output / "endpoints").glob("*.pt")), weights_only=True)
    assert torch.equal(
        saved, torch.randn((4096, 2), generator=torch.Generator().manual_seed(730001)) * 0.3
    )
    with pytest.raises(FileExistsError):
        study.run_study(configs, output)


def test_cli_defaults_are_fourier_and_help_is_inert(monkeypatch, capsys):
    monkeypatch.setattr(study, "benchmark_data", lambda *a, **k: pytest.fail("Help opened test"))
    with pytest.raises(SystemExit) as result:
        study.main(["--help"])
    assert result.value.code == 0
    assert "--other-config" in capsys.readouterr().out
    source = SCRIPT.read_text()
    assert "many_rays_fourier_cpu.yaml" in source and "many_rays_fourier_stress_32_8.yaml" in source
    assert "optimizer" not in source and ".backward(" not in source


def test_synthetic_plots_render_all_first_seed_panels_and_fixed_bin_comparisons(tmp_path):
    # Analytic synthetic points only; no dataset factory or learned model calls.
    u = torch.linspace(0.08, 1.0, 64, dtype=torch.float64)
    arm = torch.arange(64) % 16
    theta = math.pi / 2 + 2 * math.pi * arm / 16 + u * 1.6 * math.pi / (0.9 * 16)
    reference = (10 / 11) * u.sqrt()[:, None] * torch.stack((theta.cos(), theta.sin()), 1)
    samples = {"synthetic": reference.roll(1, 0)}
    geometry = study.analyse_rays(samples["synthetic"], reference)
    rows = synthetic_rows()
    for row in rows:
        row.update(sample_key="synthetic", geometry=geometry)
    for grid in ("32_to_16", "32_to_8"):
        rows.extend(
            {
                "grid": grid,
                "method": f"Original N{nfe}",
                "student_seed": None,
                "nfe": nfe,
                "sample_key": "synthetic",
                "geometry": geometry,
            }
            for nfe in study.BASELINE_NFE
        )
    files = study.render_plots(rows, samples, reference, tmp_path, 16)
    assert len(files) == 15
    assert all(path.is_file() and path.stat().st_size > 10000 for path in files)
    assert (tmp_path / "geometry_vs_nfe.png").is_file()
    assert sum("seed101" in path.name for path in files) == 8
    assert not any("seed202" in path.name or "seed303" in path.name for path in files)
