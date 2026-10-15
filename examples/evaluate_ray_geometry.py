"""Post-fixation rotating-ray geometry study; never trains or selects a model.

Run only after BOTH registered 12-student suites and their 18-row core final
evaluations have completed. The two-root preflight happens before test access,
sampling, or report writes. This sidecar intentionally lives outside recheck so
preparing a report cannot alter the frozen algorithm-source fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from recheck.config import config_hash, load_config, method_name
from recheck.distill import experiment_schedule
from recheck.evaluate import paired_statistics
from recheck.fm import benchmark_data, load_model, sample_euler
from recheck.rays_analysis import analyse_rays, plot_rays_diagnostics
from recheck.runtime import algorithm_hash, atomic_json, file_hash, model_hash, tensor_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_NFE = (8, 16, 24, 32, 64, 128)
METHODS = ("M0", "M1", "E0", "E1")
METRICS = (
    "absolute_phase_distance_w1_sector_fraction",
    "nearest_arm_occupancy_tv_outer",
    "radial_w1_raw",
)
COLORS = {"M0": "#3368ad", "M1": "#45a4b6", "E0": "#df7139", "E1": "#9a4c9c"}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message + "; final geometry test remains closed")


def _path(path, repo_root):
    path = Path(path)
    return path if path.is_absolute() else Path(repo_root) / path


def _read(path, evidence):
    path = Path(path)
    _require(path.is_file(), f"Missing prerequisite {path}")
    result = json.loads(path.read_text())
    evidence[str(path.resolve())] = file_hash(path)
    return result


def _native(path, config, expected_hash, evidence):
    """Verify actual native config AND actual exported tensors, without sampling."""
    native = _read(path / "config.json", evidence)
    m = config["model"]
    expected = {
        "_class_name": "FourierFMMLP",
        "input_dim": 2,
        "hidden_dim": m["hidden_dim"],
        "num_layers": m["num_layers"],
        "time_embedding_dim": m["time_embedding_dim"],
        "time_scale": 1000.0,
        "spatial_frequencies": m["spatial_frequencies"],
        "polar_features": m["polar_features"],
        "epsilon": m["spatial_epsilon"],
    }
    _require(
        all(native.get(k) == v for k, v in expected.items()), f"Wrong native architecture {path}"
    )
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    _require(bool(weights), f"Missing native weight files {path}")
    for file in weights:
        evidence[str(file.resolve())] = file_hash(file)
    exported = load_model(path, "cpu").eval().requires_grad_(False)
    _require(model_hash(exported) == expected_hash, f"Native weight hash mismatch {path}")
    _require(exported.sample_shape == (2,), f"Native sample shape mismatch {path}")
    del exported


def preflight(configs, *, repo_root=REPO_ROOT):
    """Check all 24 fixed students and BOTH completed core evaluations first.

    A copied stress teacher legitimately retains its original main config.
    Its own manifest hash and shared data/model/teacher profiles are verified;
    its full config hash is therefore not compared to the stress config hash.
    Missing provenance fields are rejected, without legacy fallback defaults.
    """
    _require(len(configs) == 2, "Exactly two registered grids are required")
    first, second = configs
    for section in ("data", "model", "teacher", "student", "evaluation"):
        _require(first[section] == second[section], f"Shared {section} settings differ")
    _require(first["data"]["name"] == "rotating_rays", "Only rotating rays are supported")
    _require(first["data"]["n_classes"] == 16, "This registered study uses 16 rays")
    common_distillation = [
        {k: v for k, v in c["distillation"].items() if k not in {"student_steps", "groups"}}
        for c in configs
    ]
    _require(
        common_distillation[0] == common_distillation[1], "Shared distillation settings differ"
    )
    _require(
        first["model"]["kind"] == "fourier_mlp",
        "This study requires the registered Fourier teacher",
    )
    _require(first["teacher"]["updates"] == 100000, "The shared teacher requires 100000 updates")
    _require(first["evaluation"]["final_seed"] == 730001, "Final noise seed must be 730001")
    _require(first["evaluation"]["samples"] == 4096, "Final noise bank must have 4096 samples")
    _require(first["evaluation"]["batch_size"] == 128, "Final sampling batch must be 128")
    _require(first["data"]["n_test"] == 8192, "The independent real test must have 8192 samples")
    _require(
        {c["distillation"]["student_steps"] for c in configs} == {8, 16},
        "Both 32→16 and 32→8 grids are required",
    )
    _require(first["output_dir"] != second["output_dir"], "The two suite roots must differ")
    evidence, records = {}, []
    noise = torch.randn((4096, 2), generator=torch.Generator().manual_seed(730001))
    noise_hash = tensor_hash(noise)
    for config in configs:
        root = _path(config["output_dir"], repo_root).resolve()
        digest, d = config_hash(config), config["distillation"]
        _require(
            d["teacher_steps"] == 32 and d["groups"] is None,
            "Only the fixed uniform 32-step parent grids are registered",
        )
        _require(
            d["seeds"] == [101, 202, 303], "The fixed paired training seeds must be 101/202/303"
        )
        pairs = [
            (method_name(loss, rho), loss, rho) for loss in d["losses"] for rho in d["tube_rhos"]
        ]
        _require(
            len(pairs) == 4 and {x[0] for x in pairs} == set(METHODS),
            "All four M0/M1/E0/E1 methods are required",
        )
        _require(
            tuple(sorted(config["evaluation"]["teacher_nfe"])) == BASELINE_NFE,
            "The six registered teacher baselines differ",
        )
        teacher_path = root / "teacher"
        teacher = _read(teacher_path / "manifest.json", evidence)
        _require(
            teacher.get("status") == "completed" and teacher.get("updates_completed") == 100000,
            f"Teacher incomplete {root}",
        )
        _require(
            teacher.get("config_sha256") == config_hash(teacher["config"]),
            f"Teacher manifest config hash mismatch {root}",
        )
        for section in ("data", "model", "teacher"):
            _require(
                teacher["config"][section] == config[section],
                f"Teacher {section} profile mismatch {root}",
            )
        parent = teacher.get("teacher_sha256")
        _require(isinstance(parent, str) and len(parent) == 64, f"Missing teacher hash {root}")
        _native(teacher_path / "ema_model", config, parent, evidence)
        expected_names = {f"seed{seed}_{method}" for seed in d["seeds"] for method in METHODS}
        found_names = {p.name for p in (root / "students").glob("seed*") if p.is_dir()}
        _require(
            found_names == expected_names, f"The registered 12 student directories differ {root}"
        )
        students = {}
        for seed in d["seeds"]:
            for method, loss, rho in pairs:
                name = f"seed{seed}_{method}"
                path = root / "students" / name
                manifest = _read(path / "manifest.json", evidence)
                expected = {
                    "status": "completed",
                    "updates_completed": config["student"]["updates"],
                    "config_sha256": digest,
                    "teacher_sha256": parent,
                    "student_init_sha256": parent,
                    "teacher_sha256_after": parent,
                    "seed": seed,
                    "loss_family": loss,
                    "rho": rho,
                    "schedule": experiment_schedule(config).to_dict(),
                }
                _require(
                    all(k in manifest and manifest[k] == v for k, v in expected.items()),
                    f"Student fixation/provenance mismatch {path}",
                )
                _require(
                    config_hash(manifest["config"]) == digest,
                    f"Student manifest config body mismatch {path}",
                )
                student_hash = manifest.get("student_ema_sha256")
                _require(
                    isinstance(student_hash, str) and len(student_hash) == 64,
                    f"Missing student EMA hash {path}",
                )
                _native(path / "ema_model", config, student_hash, evidence)
                students[name] = manifest
        evaluation = _read(root / "evaluation/results.json", evidence)
        meta, rows = evaluation.get("metadata", {}), evaluation.get("rows", [])
        expected = {
            "status": "completed",
            "evaluation_split": "test",
            "config_sha256": digest,
            "teacher_sha256": parent,
            "initial_noise_sha256": noise_hash,
            "student_checkpoint_hashes": {
                name: m["student_ema_sha256"] for name, m in students.items()
            },
        }
        _require(
            all(k in meta and meta[k] == v for k, v in expected.items()),
            f"Core final evaluation is incomplete/stale {root}",
        )
        _require(
            config_hash(meta["config"]) == digest, f"Core evaluation config body mismatch {root}"
        )
        _require(
            isinstance(meta.get("evaluation_sha256"), str) and len(meta["evaluation_sha256"]) == 64,
            f"Missing core evaluation digest {root}",
        )
        quality = meta.get("quality_context", {})
        _require(
            quality.get("reference_split") == "test"
            and isinstance(quality.get("real_samples_sha256"), str),
            f"Missing independent real-test provenance {root}",
        )
        expected_rows = {(f"Original N{nfe}", None) for nfe in BASELINE_NFE}
        expected_rows |= {(method, seed) for seed in d["seeds"] for method in METHODS}
        actual_rows = [(row.get("method"), row.get("student_seed")) for row in rows]
        _require(
            len(actual_rows) == 18 and set(actual_rows) == expected_rows,
            f"Core final evaluation must contain all 18 unique rows {root}",
        )
        for row in rows:
            seed, method = row["student_seed"], row["method"]
            expected_model = (
                parent if seed is None else students[f"seed{seed}_{method}"]["student_ema_sha256"]
            )
            nfe = int(method.removeprefix("Original N")) if seed is None else d["student_steps"]
            expected = {
                "nfe": nfe,
                "model_sha256": expected_model,
                "initial_noise_sha256": noise_hash,
                "evaluation_sha256": meta["evaluation_sha256"],
            }
            _require(
                all(row.get(k) == v for k, v in expected.items()),
                f"Core final evaluation row mismatch {root}: {method}/{seed}",
            )
            _require(
                row.get("cost", {}).get("generated_samples") == 4096,
                f"Truncated core final samples {root}: {method}/{seed}",
            )
        records.append(
            {
                "root": root,
                "config": config,
                "config_sha256": digest,
                "teacher": teacher,
                "students": students,
                "core": evaluation,
            }
        )
    _require(
        records[0]["teacher"]["teacher_sha256"] == records[1]["teacher"]["teacher_sha256"],
        "The two suites do not share one frozen teacher",
    )
    _require(
        records[0]["core"]["metadata"]["quality_context"]["real_samples_sha256"]
        == records[1]["core"]["metadata"]["quality_context"]["real_samples_sha256"],
        "The core suites used different independent test references",
    )
    return records, evidence


def paired_geometry(rows, seeds=(101, 202, 303)):
    """Paired differences across three trainings, conditional on one teacher."""
    by_grid = {}
    for row in rows:
        if row["student_seed"] is not None:
            by_grid.setdefault(row["grid"], {})[row["method"], row["student_seed"]] = row
    contrasts = {
        "E0-M0": {"E0": 1, "M0": -1},
        "E1-M1": {"E1": 1, "M1": -1},
        "M1-M0": {"M1": 1, "M0": -1},
        "E1-E0": {"E1": 1, "E0": -1},
        "tube_interaction": {"E1": 1, "E0": -1, "M1": -1, "M0": 1},
    }
    results = []
    for grid, group in sorted(by_grid.items()):
        _require(
            set(group) == {(m, s) for m in METHODS for s in seeds},
            f"Incomplete paired geometry rows {grid}",
        )
        for metric in METRICS:
            for contrast, coefficients in contrasts.items():
                values = []
                for seed in seeds:
                    measurements = [
                        group[method, seed]["geometry"]["metrics"][metric]
                        for method in coefficients
                    ]
                    if any(value is None for value in measurements):
                        values = None
                        break
                    values.append(
                        sum(
                            coefficients[m] * group[m, seed]["geometry"]["metrics"][metric]
                            for m in coefficients
                        )
                    )
                results.append(
                    {
                        "grid": grid,
                        "metric": metric,
                        "contrast": contrast,
                        "student_seeds": list(seeds),
                        "lower_is_better_for_simple_differences": contrast != "tube_interaction",
                        "statistics": paired_statistics(values) if values is not None else None,
                        "undefined_reason": "phase/occupancy undefined when no generated outer points"
                        if values is None
                        else None,
                    }
                )
    return results


def _csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_plots(rows, samples, reference, output, n_rays):
    """Fixed bins/axes and first registered seed; no best-looking selection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    paths = []
    for grid in sorted({row["grid"] for row in rows}):
        group = [row for row in rows if row["grid"] == grid]
        student_nfe = next(row["nfe"] for row in group if row["student_seed"] == 101)
        chosen = [
            row
            for row in group
            if row["student_seed"] == 101
            or (row["student_seed"] is None and row["nfe"] in {32, student_nfe})
        ]
        for row in chosen:
            title = f"{grid} · {row['method']}" + (
                " · seed101" if row["student_seed"] == 101 else " · fixed teacher"
            )
            figure, _ = plot_rays_diagnostics(
                samples[row["sample_key"]], reference, n_rays=n_rays, title=title
            )
            file = (
                output / f"{grid}_{row['method'].replace(' ', '_')}_seed{row['student_seed']}.png"
            )
            figure.savefig(file, dpi=160)
            plt.close(figure)
            paths.append(file)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        real = chosen[0]["geometry"]["reference"]
        geometry = chosen[0]["geometry"]["geometry"]
        edges = np.asarray(geometry["phase_histogram_edges_sector_fraction"])
        nodes = geometry["radial_cdf_nodes_normalized"]
        axes[0].stairs(
            real["phase_outer"]["signed_histogram_density"],
            edges,
            color="black",
            label="Real test",
            linewidth=2,
        )
        axes[1].plot(
            range(n_rays), real["nearest_arm_occupancy_outer"], color="black", label="Real test"
        )
        axes[2].step(
            nodes,
            real["radial"]["cdf_at_fixed_normalized_nodes"],
            where="post",
            color="black",
            label="Real test",
        )
        for row in chosen:
            generated, label = row["geometry"]["generated"], row["method"]
            color = COLORS.get(label, "#56616d" if row["nfe"] == 32 else "#b5a341")
            if generated["phase_outer"]["valid"]:
                axes[0].stairs(
                    generated["phase_outer"]["signed_histogram_density"],
                    edges,
                    label=label,
                    color=color,
                )
                axes[1].plot(
                    range(n_rays),
                    generated["nearest_arm_occupancy_outer"],
                    label=label,
                    color=color,
                )
            axes[2].step(
                nodes,
                generated["radial"]["cdf_at_fixed_normalized_nodes"],
                where="post",
                label=label,
                color=color,
            )
        axes[0].set(
            xlim=(-0.5, 0.5),
            ylim=(0, 40),
            xlabel="Signed phase / sector",
            ylabel="Density",
            title="Fixed 40-bin phase histogram",
        )
        axes[1].set(
            xlim=(-0.5, n_rays - 0.5),
            ylim=(0, 1),
            xlabel="Nearest geometric arm",
            ylabel="Outer fraction",
            title="Outer occupancy",
        )
        axes[2].set(
            xlim=(0, 1.25), ylim=(0, 1), xlabel="r/R", ylabel="CDF", title="Fixed radial CDF nodes"
        )
        axes[0].legend(fontsize=7)
        figure.suptitle(f"{grid} · seed101 (first registered) · fixed bins; no smoothing")
        file = output / f"{grid}_geometry_comparison.png"
        figure.savefig(file, dpi=160)
        plt.close(figure)
        paths.append(file)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    first_grid = rows[0]["grid"]
    teachers = sorted(
        (row for row in rows if row["grid"] == first_grid and row["student_seed"] is None),
        key=lambda row: row["nfe"],
    )
    for axis, metric in zip(axes, METRICS):
        axis.plot(
            [r["nfe"] for r in teachers],
            [r["geometry"]["metrics"][metric] for r in teachers],
            "o-",
            color="black",
            label="Fixed teacher",
        )
        for grid in sorted({r["grid"] for r in rows}):
            for method in METHODS:
                group = [
                    r
                    for r in rows
                    if r["grid"] == grid and r["method"] == method and r["student_seed"] is not None
                ]
                values = [r["geometry"]["metrics"][metric] for r in group]
                if any(value is None for value in values):
                    continue
                stats = paired_statistics(values)
                axis.errorbar(
                    group[0]["nfe"],
                    stats["mean"],
                    yerr=[
                        [stats["mean"] - stats["paired_t_ci"][0]],
                        [stats["paired_t_ci"][1] - stats["mean"]],
                    ],
                    fmt="s" if group[0]["nfe"] == 8 else "o",
                    capsize=3,
                    color=COLORS[method],
                    label=f"{method} N{group[0]['nfe']}",
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(BASELINE_NFE, labels=[str(n) for n in BASELINE_NFE])
        axis.set(
            xlabel="Velocity evaluations (NFE)",
            ylabel=metric,
            title="Geometry to independent real test",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Geometry quality, separate from MMD · student means / approximate 95% t CI (3 seeds)"
    )
    file = output / "geometry_vs_nfe.png"
    figure.savefig(file, dpi=160)
    plt.close(figure)
    paths.append(file)
    return paths


def run_study(configs, output, *, repo_root=REPO_ROOT, config_paths=()):
    """Recompute raw endpoints using a common CPU32 noise bank, after both gates."""
    records, evidence = preflight(configs, repo_root=repo_root)
    if config_paths:
        _require(len(config_paths) == 2, "Both source YAML paths are required")
        for path, config in zip(config_paths, configs):
            path = _path(path, repo_root).resolve()
            _require(
                load_config(path) == config,
                f"Source YAML differs from its resolved configuration {path}",
            )
            evidence[str(path)] = file_hash(path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Report directory must be new or empty: {output}")
    for file, digest in evidence.items():
        _require(file_hash(file) == digest, f"Frozen input changed during preflight {file}")
    config = configs[0]
    # This is the FIRST data-factory call. Both 24-student/core-eval gates passed.
    reference = benchmark_data(config, "test").detach().cpu().float()
    _require(tuple(reference.shape) == (8192, 2), "Real test reference shape differs")
    reference_hash = tensor_hash(reference)
    for record in records:
        _require(
            reference_hash == record["core"]["metadata"]["quality_context"]["real_samples_sha256"],
            "Recreated independent real test differs from the fixed core reference",
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "endpoints").mkdir()
    (output / "geometry").mkdir()
    noise = torch.randn((4096, 2), generator=torch.Generator().manual_seed(730001))
    noise_hash = tensor_hash(noise)
    samples, sample_provenance, rows = {}, {}, []
    teacher_hash = records[0]["teacher"]["teacher_sha256"]
    source_hash = file_hash(__file__)
    helper_hash = file_hash(inspect.getfile(analyse_rays))

    def generate(path, expected_hash, key, times):
        model = load_model(path, "cpu").eval().requires_grad_(False)
        _require(model_hash(model) == expected_hash, f"Fixed model changed before sampling {path}")
        generated = sample_euler(model, noise, times=times, batch_size=128, record_history=False)
        _require(
            generated["initial_noise_sha256"] == noise_hash, "Sampling changed the shared noise"
        )
        _require(model_hash(model) == expected_hash, f"Model mutated while sampling {path}")
        raw = generated["samples"].detach().cpu().float()
        _require(
            raw.shape == noise.shape and bool(torch.isfinite(raw).all()),
            "Invalid raw generated endpoints",
        )
        samples[key] = raw
        file = output / "endpoints" / f"{key}.pt"
        torch.save(raw, file)
        sample_provenance[key] = {
            "samples_sha256": tensor_hash(raw),
            "file": str(file),
            "file_sha256": file_hash(file),
            "model_sha256": expected_hash,
            "nfe": len(times) - 1,
            "total_model_calls": generated["total_model_calls"],
        }

    for nfe in BASELINE_NFE:
        generate(
            records[0]["root"] / "teacher/ema_model",
            teacher_hash,
            f"teacher_N{nfe}",
            torch.linspace(1, 0, nfe + 1, dtype=torch.float64),
        )
    for record in records:
        cfg = record["config"]
        grid = f"32_to_{cfg['distillation']['student_steps']}"
        specifications = [
            (f"Original N{nfe}", None, f"teacher_N{nfe}", nfe, teacher_hash) for nfe in BASELINE_NFE
        ]
        for seed in cfg["distillation"]["seeds"]:
            for method in METHODS:
                name, key = f"seed{seed}_{method}", f"{grid}_seed{seed}_{method}"
                digest = record["students"][name]["student_ema_sha256"]
                generate(
                    record["root"] / "students" / name / "ema_model",
                    digest,
                    key,
                    experiment_schedule(cfg).times,
                )
                specifications.append(
                    (method, seed, key, cfg["distillation"]["student_steps"], digest)
                )
        for method, seed, key, nfe, digest in specifications:
            geometry = analyse_rays(samples[key], reference, n_rays=cfg["data"]["n_classes"])
            row = {
                "grid": grid,
                "protocol_id": cfg["protocol_id"],
                "config_sha256": record["config_sha256"],
                "core_evaluation_sha256": record["core"]["metadata"]["evaluation_sha256"],
                "sidecar_source_sha256": source_hash,
                "geometry_helper_sha256": helper_hash,
                "method": method,
                "student_seed": seed,
                "nfe": nfe,
                "model_sha256": digest,
                "sample_key": key,
                "samples_sha256": sample_provenance[key]["samples_sha256"],
                "initial_noise_sha256": noise_hash,
                "reference_samples_sha256": reference_hash,
                "geometry": geometry,
            }
            file = output / "geometry" / f"{grid}_{method.replace(' ', '_')}_seed{seed}.json"
            atomic_json(file, row)
            rows.append(row)
    contrasts = paired_geometry(rows)
    flat = [
        {
            k: row[k]
            for k in (
                "grid",
                "protocol_id",
                "config_sha256",
                "core_evaluation_sha256",
                "sidecar_source_sha256",
                "geometry_helper_sha256",
                "method",
                "student_seed",
                "nfe",
                "model_sha256",
                "samples_sha256",
                "initial_noise_sha256",
                "reference_samples_sha256",
            )
        }
        | row["geometry"]["metrics"]
        for row in rows
    ]
    for row in flat:
        row["off_curve_tail_fraction_differences"] = json.dumps(
            row["off_curve_tail_fraction_differences"]
        )
    _csv(output / "geometry_per_seed.csv", flat)
    atomic_json(output / "paired_geometry.json", contrasts)
    _csv(
        output / "paired_geometry.csv",
        [
            {
                "grid": row["grid"],
                "metric": row["metric"],
                "contrast": row["contrast"],
                "seeds": json.dumps(row["student_seeds"]),
                "paired_values": json.dumps(row["statistics"]["values"])
                if row["statistics"]
                else "null",
                "mean": row["statistics"]["mean"] if row["statistics"] else None,
                "sample_std": row["statistics"]["sample_std"] if row["statistics"] else None,
                "ci95": json.dumps(row["statistics"]["paired_t_ci"])
                if row["statistics"]
                else "null",
            }
            for row in contrasts
        ],
    )
    plots = render_plots(rows, samples, reference, output, config["data"]["n_classes"])
    metadata = {
        "status": "completed",
        "scope": "post_fixation_coordinate_only_geometry_sidecar",
        "test_access_rule": "both 12-student suites fixed and both 18-row core final evaluations completed before data or sampling",
        "algorithm_sha256": algorithm_hash(),
        "sidecar_source_sha256": source_hash,
        "geometry_helper_sha256": helper_hash,
        "configs": [
            {
                "config": r["config"],
                "config_sha256": r["config_sha256"],
                "core_evaluation_sha256": r["core"]["metadata"]["evaluation_sha256"],
                "core_algorithm_sha256": r["core"]["metadata"].get("algorithm_sha256"),
            }
            for r in records
        ],
        "input_files_sha256": evidence,
        "shared_teacher_sha256": teacher_hash,
        "teacher_updates": 100000,
        "noise": {
            "seed": 730001,
            "shape": [4096, 2],
            "dtype": "torch.float32",
            "sampling_batch_size": 128,
            "sha256": noise_hash,
        },
        "reference": {"split": "test", "samples": 8192, "sha256": reference_hash},
        "coordinate_units": "raw Cartesian x/y; no feature scaling, clipping, projection or smoothing",
        "geometry": rows[0]["geometry"]["geometry"],
        "sample_exports": sample_provenance,
        "limitations": [
            "Three paired training seeds are a pilot conditional on one frozen teacher, one independent test reference and common generation noise.",
            "Approximate t intervals describe training-seed variability; images/points are not independent trained-model replicates.",
            "Geometry supplements MMD and fidelity and does not establish a Wasserstein/OT mechanism.",
            "Teacher NFE baselines are one model at different integration budgets, not independent training seeds.",
        ],
        "report_files_sha256": {
            str(file.relative_to(output)): file_hash(file)
            for file in [
                *plots,
                output / "geometry_per_seed.csv",
                output / "paired_geometry.json",
                output / "paired_geometry.csv",
                *(output / "geometry").glob("*.json"),
            ]
        },
    }
    atomic_json(
        output / "geometry_study.json",
        {"metadata": metadata, "rows": rows, "paired_contrasts": contrasts},
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="replication/configs/many_rays_fourier_cpu.yaml")
    parser.add_argument(
        "--other-config", default="replication/configs/many_rays_fourier_stress_32_8.yaml"
    )
    parser.add_argument("--output", default="replication/reports/many_rays/fourier_geometry")
    args = parser.parse_args(argv)
    config_paths = [_path(path, REPO_ROOT) for path in (args.config, args.other_config)]
    configs = [load_config(path) for path in config_paths]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(configs[0]["threads"])
    try:
        metadata = run_study(configs, _path(args.output, REPO_ROOT), config_paths=config_paths)
    finally:
        torch.set_num_threads(previous_threads)
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "output": str(_path(args.output, REPO_ROOT)),
                "shared_teacher_sha256": metadata["shared_teacher_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
