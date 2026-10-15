"""Closed-test assessment of four frozen exploratory wall-navigation students.

The default invocation only prints the registered plan. ``--execute`` is for
the coordinator's explicit final GO, after all four final EMA exports exist.
Manifest checks and actual model hashes are verified before the Test factory
is called. No training, projection, clipping, mode conditioning or path repair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_lab import PointDenoiser
from diffusion_lab.environments.multimodal_navigation import (
    evaluate_navigation,
    make_multimodal_navigation,
)
from diffusion_lab.navigation_visualization import (
    plot_navigation_denoising,
    plot_navigation_paths,
    save_navigation_denoising_gif,
    save_navigation_particle_gif,
    summarize_navigation_benchmark,
)
from recheck.evaluate import fixed_projection_bank, sliced_wasserstein_fixed
from recheck.runtime import atomic_json, model_hash, save_checkpoint, tensor_hash

EXPECTED_TEACHER = "71d77760c37e6a053f324f2c810adb323439ed76ef51f09a57db43a5d2da9e79"
NFES = (1, 2, 4, 8, 16, 32, 64)
CHILDREN = ((4, "M0"), (4, "E0"), (2, "M0"), (2, "E0"))
TEST = {
    "n_scenes": 32,
    "samples_per_scene": 128,
    "path_length": 32,
    "seed": 20061,
    "split": "test",
    "jitter": 0.025,
    "clearance": 0.06,
    "margin": 0.035,
}
NOISE_SEED = 614001
PROJECTION_SEED = 861003


def _read(path):
    return json.loads(Path(path).read_text())


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def registered_plan(root):
    """Data-free plan: this does not read even a validation or Test scene."""
    return {
        "status": "prepared_not_executed",
        "scope": "one_seed_exploratory_closed_test_wall_navigation",
        "root": str(root),
        "teacher_sha256": EXPECTED_TEACHER,
        "parent_nfe": list(NFES),
        "children": [{"student_nfe": nfe, "method": method} for nfe, method in CHILDREN],
        "test": TEST,
        "noise_seed": NOISE_SEED,
        "noise_rule": "same128 interior noises repeated across32 distinct conditions",
        "projection_seed": PROJECTION_SEED,
        "fidelity_reference": "Original N32; descriptive paired fidelity, not geometric truth",
        "gate": "all four completed/fixed EMA exports AND actual hashes verified before Test factory",
        "threads": 4,
        "authoritative_cohort": str(Path(root) / "final_cohort_registration.json"),
    }


def _verify_reference(record, name):
    if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
        raise RuntimeError(f"Closed Test: missing {name} provenance")
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path(__file__).parents[1] / path
    if _file_hash(path) != record["sha256"]:
        raise RuntimeError(f"Closed Test: {name} source SHA mismatch")
    return path.resolve()


def _cohort(root, registration_path, expected_test):
    """Resolve an explicit amended cohort, never choose models by metric/glob."""
    path = root / "final_cohort_registration.json"
    cohort = _read(path)
    if cohort.get("teacher_sha256") != EXPECTED_TEACHER or cohort.get("test") != expected_test:
        raise RuntimeError("Closed Test: final cohort teacher/Test registration mismatch")
    if (
        _verify_reference(cohort.get("original_registration"), "original registration")
        != registration_path.resolve()
    ):
        raise RuntimeError("Closed Test: final cohort references another original registration")
    for name in ("amendment", "solver_backend", "notebook"):
        _verify_reference(cohort.get(name), name)
    backend = cohort["solver_backend"]
    if (
        backend.get("tolerance") != 1e-10
        or backend.get("dtype") != "float64"
        or not backend.get("name")
    ):
        raise RuntimeError("Closed Test: exact float64 backend/tolerance provenance is required")
    expected_trace = {
        "updates": 1000,
        "seed": 161,
        "batch_size": 128,
        "paired_keys": ["scene", "interval", "data_indices", "noise"],
        "equal_within_nfe": True,
    }
    if cohort.get("expected_trace_contract") != expected_trace:
        raise RuntimeError("Closed Test: altered paired training trace contract")
    children = cohort.get("children", [])
    if not isinstance(children, list) or len(children) != 4:
        raise RuntimeError("Closed Test: final cohort must explicitly name exactly four children")
    by_name, directories = {}, set()
    for child in children:
        nfe, method = child.get("student_nfe"), child.get("method")
        if (nfe, method) not in CHILDREN or child.get("name") != f"{method} N{nfe}":
            raise RuntimeError("Closed Test: unknown or mislabeled child in final cohort")
        name = child["name"]
        if name in by_name or not isinstance(child.get("directory"), str):
            raise RuntimeError("Closed Test: duplicate or missing child directory")
        relative = Path(child["directory"])
        directory = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not directory.is_relative_to(root.resolve())
            or directory in directories
        ):
            raise RuntimeError(
                "Closed Test: child paths must be distinct and inside the registered wall root"
            )
        by_name[name] = directory
        directories.add(directory)
    return cohort, by_name, path


def _verify_pairing_traces(records):
    """Read saved training RNG evidence before Test, without running any model."""
    traces = {}
    for nfe, method in CHILDREN:
        name = f"{method} N{nfe}"
        record = records[name]
        path = Path(record["directory"]) / "training.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        trace = saved.get("pairing_trace")
        if (
            saved.get("updates_completed") != 1000
            or saved.get("status") != "completed"
            or not isinstance(trace, list)
            or len(trace) != 1000
            or any(not isinstance(item, str) or len(item) != 64 for item in trace)
        ):
            raise RuntimeError(
                f"Closed Test: incomplete scene/interval/data/noise trace for {name}"
            )
        contract = saved.get("contract", {})
        for key in (
            "teacher_sha256",
            "train_sha256",
            "conditions_sha256",
            "seed",
            "updates",
            "student_steps",
            "teacher_steps",
            "batch_size",
            "calibration_sha256",
        ):
            if contract.get(key) != record["manifest"].get(key):
                raise RuntimeError(
                    f"Closed Test: checkpoint contract differs from manifest {name}/{key}"
                )
        if (
            not isinstance(saved.get("ema"), dict)
            or _state_hash(saved["ema"]) != record["expected_model_sha256"]
        ):
            raise RuntimeError(
                f"Closed Test: checkpoint final EMA differs from manifest for {name}"
            )
        traces[name] = trace
        record["training_checkpoint_sha256"] = _file_hash(path)
        record["paired_trace_sha256"] = hashlib.sha256(
            json.dumps(trace, separators=(",", ":")).encode()
        ).hexdigest()
    for nfe in (4, 2):
        if traces[f"M0 N{nfe}"] != traces[f"E0 N{nfe}"]:
            raise RuntimeError(
                f"Closed Test: unequal paired scene/interval/data/noise training traces at N{nfe}"
            )


def preflight_manifests(root):
    """Reject incomplete/mismatched cohorts without constructing any data/model."""
    root = Path(root)
    registration_path = root / "child_study_registration.json"
    registration = _read(registration_path)
    expected_registration = {
        "teacher_sha256": EXPECTED_TEACHER,
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
    }
    for key, expected in expected_registration.items():
        if registration.get(key) != expected:
            raise RuntimeError(f"Registration mismatch for {key}: expected {expected!r}")
    test_registration = registration.get("test_preregistered_not_yet_accessed", {})
    expected_test = {
        key: TEST[key] for key in ("n_scenes", "samples_per_scene", "path_length", "seed", "split")
    }
    expected_test["noise_seed"] = NOISE_SEED
    if test_registration != expected_test:
        raise RuntimeError("Registered Test bank differs from the frozen evaluator plan")
    cohort, child_directories, cohort_path = _cohort(root, registration_path, expected_test)
    teacher_dir = root / "teacher"
    teacher = _read(teacher_dir / "manifest.json")
    if (
        teacher.get("status") != "completed"
        or teacher.get("updates_completed") != 20000
        or teacher.get("teacher_sha256") != EXPECTED_TEACHER
        or teacher.get("route_ids_used_by_model") is not False
    ):
        raise RuntimeError("Frozen parent must complete20000 updates with no route-ID input")
    records = {
        "Teacher": {
            "directory": str(teacher_dir),
            "manifest": teacher,
            "expected_model_sha256": EXPECTED_TEACHER,
        }
    }
    for nfe, method in CHILDREN:
        directory = child_directories[f"{method} N{nfe}"]
        path = directory / "manifest.json"
        manifest = _read(path)
        expected = {
            "status": "completed",
            "updates_completed": 1000,
            "updates": 1000,
            "teacher_sha256": EXPECTED_TEACHER,
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
            "train_sha256": registration["train_data_sha256"],
            "conditions_sha256": registration["conditions_sha256"],
            "solver_backend": cohort["solver_backend"],
            "solver_amendment_sha256": cohort["amendment"]["sha256"],
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Closed Test: {method} N{nfe} {key} must equal {value!r}")
        model_digest = manifest.get("ema_sha256")
        if not isinstance(model_digest, str) or len(model_digest) != 64:
            raise RuntimeError(f"Closed Test: {method} N{nfe} has no fixed EMA digest")
        records[f"{method} N{nfe}"] = {
            "directory": str(directory),
            "manifest": manifest,
            "expected_model_sha256": model_digest,
        }
    _verify_pairing_traces(records)
    # Only after ALL manifests pass, inspect every native export. A failed E0
    # cannot be hidden by assessing the other three or by allowing partial runs.
    for name, record in records.items():
        directory = Path(record["directory"])
        export = directory / "ema_model"
        config_path = export / "config.json"
        weight_path = export / "diffusion_pytorch_model.safetensors"
        config = _read(config_path)
        if (
            config.get("sample_shape") != [30, 2]
            or config.get("condition_dim") != 8
            or config.get("conditioner") != "mlp"
            or config.get("num_classes") != 0
        ):
            raise RuntimeError(f"Closed Test: incorrect continuous-condition model for {name}")
        record["native_files_sha256"] = {
            "manifest.json": _file_hash(directory / "manifest.json"),
            "ema_model/config.json": _file_hash(config_path),
            "ema_model/diffusion_pytorch_model.safetensors": _file_hash(weight_path),
        }
    return {
        "registration": registration,
        "registration_sha256": _file_hash(registration_path),
        "final_cohort_registration": cohort,
        "final_cohort_registration_sha256": _file_hash(cohort_path),
        "all_four_training_traces_verified": True,
        "models": records,
        "manifest_preflight_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_frozen_models(preflight):
    """Verify actual tensor hashes of all five models before the Test factory."""
    models = {}
    for name, record in preflight["models"].items():
        model = (
            PointDenoiser.from_pretrained(
                Path(record["directory"]) / "ema_model",
                local_files_only=True,
                low_cpu_mem_usage=False,
            )
            .cpu()
            .eval()
            .requires_grad_(False)
        )
        actual = model_hash(model)
        if actual != record["expected_model_sha256"]:
            raise RuntimeError(f"Closed Test: native tensor SHA differs from manifest for {name}")
        record["verified_model_sha256"] = actual
        models[name] = model
    preflight["all_actual_weights_fixed_utc"] = datetime.now(timezone.utc).isoformat()
    return models


def _assert_weights_unchanged(preflight, models):
    for name, model in models.items():
        record = preflight["models"][name]
        if model_hash(model) != record["expected_model_sha256"]:
            raise RuntimeError(f"Model weights changed during assessment for {name}")
        for relative, digest in record["native_files_sha256"].items():
            if _file_hash(Path(record["directory"]) / relative) != digest:
                raise RuntimeError(
                    f"Frozen native export changed during assessment: {name}/{relative}"
                )
        if (
            "training_checkpoint_sha256" in record
            and _file_hash(Path(record["directory"]) / "training.pt")
            != record["training_checkpoint_sha256"]
        ):
            raise RuntimeError(f"Frozen training checkpoint changed during assessment: {name}")
    cohort = preflight["final_cohort_registration"]
    for name in ("original_registration", "amendment", "solver_backend", "notebook"):
        _verify_reference(cohort[name], name)


def shared_scene_noise(scene_indices, *, samples_per_scene=128, seed=NOISE_SEED):
    """Independent CPU generator; the same noise IDs are paired across scenes."""
    base = torch.randn((samples_per_scene, 30, 2), generator=torch.Generator().manual_seed(seed))
    noise = torch.empty((len(scene_indices), 30, 2))
    for scene in torch.unique(scene_indices):
        selected = torch.nonzero(scene_indices == scene).flatten()
        if len(selected) != samples_per_scene:
            raise ValueError("Each registered condition must have exactly128 samples")
        noise[selected] = base
    return noise


@torch.no_grad()
def sample_navigation(model, initial_noise, conditions, *, steps, batch_size=128):
    """Visible plain-Torch Euler loop; raw interior states and exact A/B nodes."""
    if steps not in NFES or initial_noise.ndim != 3 or initial_noise.shape[1:] != (30, 2):
        raise ValueError("Registered positive NFE and noise[B,30,2] are required")
    if conditions.shape != (len(initial_noise), 8):
        raise ValueError("Continuous conditions must be[B,8]")
    times = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    batches = []
    for offset in range(0, len(initial_noise), batch_size):
        state = initial_noise[offset : offset + batch_size].clone()
        context = conditions[offset : offset + batch_size]

        def whole_path():
            return torch.cat((context[:, None, :2], state, context[:, None, 2:4]), dim=1)

        nodes = [whole_path()]
        for t, width in zip(times[:-1], times[:-1] - times[1:]):
            predicted = model(state, float(t) * 1000, conditions=context).sample
            state = state - float(width) * predicted
            if not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite raw Euler state; no clamping is allowed")
            nodes.append(whole_path())
        batches.append(torch.stack(nodes))
    return torch.cat(batches, dim=1)


def distribution_metrics(real_paths, generated_paths, scene_indices, projections):
    """Raw interior60D SWD; conditioning is preserved in the primary average."""
    real = real_paths[:, 1:-1].flatten(1)
    generated = generated_paths[:, 1:-1].flatten(1)
    per_scene = []
    for scene in torch.unique(scene_indices, sorted=True):
        selected = scene_indices == scene
        per_scene.append(sliced_wasserstein_fixed(real[selected], generated[selected], projections))
    return {
        "space": "raw interior60D; endpoints excluded because they are constructed",
        "fixed_projection_bank_sha256": tensor_hash(projections),
        "mean_scene_sliced_w1_interior_real": float(np.mean(per_scene)),
        "scene_sliced_w1_interior_real": per_scene,
        "marginal_sliced_w1_interior_real": sliced_wasserstein_fixed(real, generated, projections),
        "marginal_limitation": "Marginal SWD mixes conditions; mean per-condition SWD is primary.",
    }


def save_public_payload(output, histories, *, noise, conditions, real_paths, scene_indices):
    """Clone views: public endpoints must not serialize full backing histories."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    endpoints = {name: value[-1].clone().contiguous() for name, value in histories.items()}
    endpoint_path = output / "final_endpoints.pt"
    save_checkpoint(
        endpoint_path,
        {
            "endpoints_by_method": endpoints,
            "initial_noise": noise.clone().contiguous(),
            "conditions": conditions.clone().contiguous(),
            "real_paths": real_paths.clone().contiguous(),
            "scene_indices": scene_indices.clone().contiguous(),
        },
    )
    selected = torch.nonzero(scene_indices == 0).flatten()[:32]
    if len(selected) == 0:
        raise ValueError("Registered first scene0 is required for the fixed preview")
    preview = {name: value[:, selected].clone().contiguous() for name, value in histories.items()}
    preview_path = output / "scene0_histories.pt"
    save_checkpoint(
        preview_path,
        {
            "histories_by_method": preview,
            "condition": conditions[selected[0]].clone(),
            "real_paths": real_paths[scene_indices == 0].clone(),
            "selected_global_indices": selected.clone(),
        },
    )
    for path in (endpoint_path, preview_path):
        if path.stat().st_size >= 20 * 1024**2:
            raise RuntimeError(f"Public tensor payload exceeds20MiB: {path}")
    return preview, conditions[selected[0]]


def _csv(output, rows):
    path = Path(output) / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "nfe",
                "success_fraction",
                "collision_free_fraction",
                "bounds_fraction",
                "both_modes_fraction_all_conditions",
                "top_successful",
                "bottom_successful",
                "invalid_or_unassigned",
                "mean_path_length",
                "mean_successful_path_length",
                "mean_scene_sliced_w1_interior_real",
                "paired_teacher32_interior_mse",
                "generation_and_assessment_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            quality = row["quality"]
            writer.writerow(
                {
                    "method": row["method"],
                    "nfe": row["nfe"],
                    "success_fraction": quality["success_fraction"],
                    "collision_free_fraction": quality["collision_free_fraction"],
                    "bounds_fraction": quality["bounds_fraction"],
                    "both_modes_fraction_all_conditions": quality["mode_coverage"][
                        "both_modes_fraction_all_conditions"
                    ],
                    "top_successful": quality["successful_mode_counts"][0],
                    "bottom_successful": quality["successful_mode_counts"][1],
                    "invalid_or_unassigned": quality["unassigned_route_count"],
                    "mean_path_length": quality["mean_path_length"],
                    "mean_successful_path_length": quality["mean_path_length_successful"],
                    "mean_scene_sliced_w1_interior_real": row["distribution"][
                        "mean_scene_sliced_w1_interior_real"
                    ],
                    "paired_teacher32_interior_mse": row["fidelity_to_teacher32"][
                        "paired_parent_interior_mse"
                    ],
                    "generation_and_assessment_seconds": row["generation_and_assessment_seconds"],
                }
            )


def descriptive_contrasts(rows):
    """Paired-bank point differences; never pretend one seed is repeated runs."""
    by_name = {row["method"]: row for row in rows}
    contrasts = []
    for nfe in (4, 2):
        pairs = [(f"E0 N{nfe}", f"M0 N{nfe}")]
        pairs += [(f"{method} N{nfe}", f"Original N{nfe}") for method in ("M0", "E0")]
        for left_name, right_name in pairs:
            left, right = by_name[left_name], by_name[right_name]
            contrasts.append(
                {
                    "contrast": f"{left_name} minus {right_name}",
                    "same_initial_noise": True,
                    "same_condition_bank": True,
                    "success_fraction_difference": left["quality"]["success_fraction"]
                    - right["quality"]["success_fraction"],
                    "collision_free_fraction_difference": left["quality"]["collision_free_fraction"]
                    - right["quality"]["collision_free_fraction"],
                    "both_modes_all_conditions_difference": left["quality"]["mode_coverage"][
                        "both_modes_fraction_all_conditions"
                    ]
                    - right["quality"]["mode_coverage"]["both_modes_fraction_all_conditions"],
                    "conditional_swd_difference": left["distribution"][
                        "mean_scene_sliced_w1_interior_real"
                    ]
                    - right["distribution"]["mean_scene_sliced_w1_interior_real"],
                    "teacher32_interior_mse_difference": left["fidelity_to_teacher32"][
                        "paired_parent_interior_mse"
                    ]
                    - right["fidelity_to_teacher32"]["paired_parent_interior_mse"],
                    "training_seeds": 1,
                    "confidence_interval": None,
                    "limitation": "Descriptive difference on one paired held-out scene/noise bank; no training-seed replication.",
                }
            )
    return contrasts


def render_cached(output, preview, condition):
    """Scene0/first noise IDs only. Actual FM nodes; no success filtering."""
    output = Path(output)
    scene = {
        "start": condition[:2],
        "goal": condition[2:4],
        "walls": condition[4:].reshape(1, 2, 2),
    }
    for nfe in (4, 2):
        selected = {
            "Teacher N32": preview["Original N32"],
            f"Original N{nfe}": preview[f"Original N{nfe}"],
            f"M0 N{nfe}": preview[f"M0 N{nfe}"],
            f"E0 N{nfe}": preview[f"E0 N{nfe}"],
        }
        for name, history in selected.items():
            stem = name.lower().replace(" ", "_")
            figure, _ = plot_navigation_paths(
                history[-1], title=f"{name} · Test scene0 · first32 paired noises", **scene
            )
            for extension in ("png", "svg"):
                figure.savefig(
                    output / f"{stem}_scene0_paths.{extension}", dpi=160, bbox_inches="tight"
                )
            plt.close(figure)
            save_navigation_particle_gif(
                history[-1, :4],
                output / f"{stem}_spatial_particle.gif",
                fps=10,
                substeps=1,
                **scene,
            )
        figure, _ = plot_navigation_denoising(selected, **scene)
        for extension in ("png", "svg"):
            figure.savefig(output / f"nfe{nfe}_fm_final.{extension}", dpi=160, bbox_inches="tight")
        plt.close(figure)
        save_navigation_denoising_gif(
            selected,
            output / f"nfe{nfe}_fm_denoising.gif",
            fps=10,
            max_frames=33,
            **scene,
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    rows = _read(output / "metrics.json")["rows"]
    parent = [row for row in rows if row["kind"] == "original"]
    axes[0].plot(
        [row["nfe"] for row in parent],
        [row["quality"]["success_fraction"] for row in parent],
        "o-",
        label="Frozen original",
    )
    axes[1].plot(
        [row["nfe"] for row in parent],
        [row["distribution"]["mean_scene_sliced_w1_interior_real"] for row in parent],
        "o-",
        label="Frozen original",
    )
    for method, marker in (("M0", "s"), ("E0", "^")):
        selected = [row for row in rows if row["method"].startswith(method)]
        axes[0].plot(
            [row["nfe"] for row in selected],
            [row["quality"]["success_fraction"] for row in selected],
            marker=marker,
            label=method,
        )
        axes[1].plot(
            [row["nfe"] for row in selected],
            [row["distribution"]["mean_scene_sliced_w1_interior_real"] for row in selected],
            marker=marker,
            label=method,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(NFES, [str(nfe) for nfe in NFES])
        axis.set_xlabel("NFE per generated route")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set(ylabel="Exact geometry success fraction", ylim=(-0.02, 1.02))
    axes[1].set_ylabel("Mean per-condition SWD · raw interior60D")
    figure.suptitle("Held-out scenes · one training seed · endpoint A/B reconstructed")
    for extension in ("png", "svg"):
        figure.savefig(output / f"quality_vs_nfe.{extension}", dpi=160)
    plt.close(figure)


def run_assessment(root, output, cache):
    """Closed Test is opened exactly once, only after every final weight passes."""
    root, output, cache = Path(root), Path(output), Path(cache)
    if (output / "metrics.json").exists():
        raise FileExistsError("Preserve the existing fixed final assessment")
    preflight = preflight_manifests(root)
    models = load_frozen_models(preflight)
    _assert_weights_unchanged(preflight, models)
    # This is deliberately the FIRST call that can construct any Test scene.
    real = make_multimodal_navigation(**TEST)
    conditions = real.conditions.cpu()
    reference = evaluate_navigation(real.samples, conditions, margin=TEST["margin"])
    if reference["success_fraction"] != 1.0:
        raise RuntimeError("Reference Test routes must satisfy exact full-segment geometry")
    noise = shared_scene_noise(real.scene_indices)
    projections = fixed_projection_bank(60, 128, PROJECTION_SEED)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "scope": "one_seed_exploratory_closed_test_wall_navigation",
        "test": TEST,
        "test_accessed": True,
        "noise_seed": NOISE_SEED,
        "noise_sha256": tensor_hash(noise),
        "conditions_sha256": tensor_hash(conditions),
        "reference_paths_sha256": tensor_hash(real.samples),
        "projection_seed": PROJECTION_SEED,
        "projection_bank_sha256": tensor_hash(projections),
        "teacher_sha256": EXPECTED_TEACHER,
        "preflight": preflight,
        "reference_quality": reference,
        "threads": 4,
        "preview_rule": "held-out scene0, first32 paired noise IDs; physical playback first4; no selection",
        "source_sha256": _file_hash(__file__),
        "support_source_sha256": {
            "navigation_visualization.py": _file_hash(
                Path(__file__).parents[1] / "src/diffusion_lab/navigation_visualization.py"
            ),
            "multimodal_navigation.py": _file_hash(
                Path(__file__).parents[1]
                / "src/diffusion_lab/environments/multimodal_navigation.py"
            ),
        },
        "rows": [],
    }
    jobs = [(f"Original N{nfe}", nfe, models["Teacher"], "original") for nfe in NFES]
    jobs += [
        (f"{method} N{nfe}", nfe, models[f"{method} N{nfe}"], "student") for nfe, method in CHILDREN
    ]
    histories = {}
    started = time.perf_counter()
    for name, nfe, model, kind in jobs:
        before = time.perf_counter()
        history = sample_navigation(model, noise, conditions, steps=nfe)
        histories[name] = history
        quality = evaluate_navigation(history[-1], conditions, margin=TEST["margin"])
        distribution = distribution_metrics(
            real.samples, history[-1], real.scene_indices, projections
        )
        row = {
            "method": name,
            "kind": kind,
            "nfe": nfe,
            "quality": quality,
            "distribution": distribution,
            "generation_and_assessment_seconds": time.perf_counter() - before,
            "model_sha256": model_hash(model),
            "raw_endpoint_sha256": tensor_hash(history[-1]),
            "nfe_per_sample": nfe,
            "sampler_model_calls": nfe * ((len(noise) + 127) // 128),
        }
        state["rows"].append(row)
        atomic_json(output / "progress.json", state)
        print(
            json.dumps(
                {
                    "method": name,
                    "success": quality["success_fraction"],
                    "both_modes": quality["mode_coverage"]["both_modes_fraction_all_conditions"],
                    "conditional_swd": distribution["mean_scene_sliced_w1_interior_real"],
                }
            ),
            flush=True,
        )
    fidelity = summarize_navigation_benchmark(histories, conditions, parent="Original N32")
    for row, paired in zip(state["rows"], fidelity["rows"]):
        row["fidelity_to_teacher32"] = paired["fidelity"]
    state["descriptive_contrasts"] = descriptive_contrasts(state["rows"])
    _assert_weights_unchanged(preflight, models)
    state.update(
        status="metrics_completed_render_pending",
        assessment_seconds=time.perf_counter() - started,
        limitations=fidelity["limitations"]
        + [
            "One student training seed161, not a three-seed confirmatory comparison.",
            "Teacher32 fidelity is distinct from exact geometry quality and from real path-distribution SWD.",
            "Generated sample counts do not imply independent training repeats or a seed-level confidence interval.",
            "Endpoints A/B are reconstructed exactly at every solver node, not learned.",
            "Shared CPU host; measured stage times are not an isolated latency benchmark.",
            "Failed/colliding routes remain in every metric and preview bank; successful-only mode counts retain ALL-condition coverage denominator.",
        ],
    )
    atomic_json(output / "metrics.json", state)
    _csv(output, state["rows"])
    save_checkpoint(
        cache / "full_histories.pt",
        {
            "histories_by_method": histories,
            "initial_noise": noise,
            "conditions": conditions,
            "real_paths": real.samples,
            "scene_indices": real.scene_indices,
        },
    )
    preview, condition = save_public_payload(
        output,
        histories,
        noise=noise,
        conditions=conditions,
        real_paths=real.samples,
        scene_indices=real.scene_indices,
    )
    print(
        "METRICS_READY: all11 frozen models assessed once; rendering cached scene0 only", flush=True
    )
    render_cached(output, preview, condition)
    state.update(status="completed", elapsed_seconds=time.perf_counter() - started)
    atomic_json(output / "metrics.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--root", default="replication/artifacts/wall_navigation_conditional_v1")
    parser.add_argument("--output", default="replication/reports/wall_navigation/final_test")
    parser.add_argument(
        "--cache", default="replication/artifacts/wall_navigation_conditional_v1/final_test_cache"
    )
    args = parser.parse_args()
    if args.execute and args.preflight_only:
        parser.error("Choose actual execution OR manifest-only preflight")
    if args.preflight_only:
        print(json.dumps(preflight_manifests(args.root), indent=2))
    elif args.execute:
        torch.set_num_threads(4)
        state = run_assessment(args.root, args.output, args.cache)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "elapsed_seconds": state["elapsed_seconds"],
                    "output": args.output,
                }
            ),
            flush=True,
        )
    else:
        print(json.dumps(registered_plan(args.root), indent=2))


if __name__ == "__main__":
    main()
