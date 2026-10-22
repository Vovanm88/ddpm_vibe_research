"""Open the preregistered final banks only after ALL48 native runs are fixed.

This driver performs no training or checkpoint selection. Models, data, noises,
projections, outcomes and actual sampling events carry explicit hashes. Raw
endpoints/full histories stay in the ignored private cache; public previews
contain only the fixed first16 draws of target/scene0, new training seed161.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_lab.conditional_metrics import nearest_ray_ids
from diffusion_lab.environments import make_environment
from diffusion_lab.environments.multimodal_navigation import make_multimodal_navigation
from recheck.evaluate import fixed_projection_bank
from recheck.runtime import atomic_json, choose_device, model_hash, save_checkpoint, tensor_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "replication/reports/method_zoo/final"
DEFAULT_GATE = ROOT / "replication/reports/method_zoo/final_gate.json"
DEFAULT_RUNS = ROOT / "replication/artifacts/method_zoo_v1"
DEFAULT_PRIVATE = DEFAULT_RUNS / "final_evaluation_cache"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_core():
    return _module("method_zoo_quality_core", ROOT / "examples/evaluate_method_zoo.py")


def training_api():
    return _module("method_zoo_native_loader", ROOT / "examples/train_method_zoo.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def final_jobs(gate):
    """Use the gate's exact export mapping; never infer a progressive horizon."""
    jobs = []
    # Parent32 is measured first and supplies the shared paired fidelity target.
    for task in ("rays", "wall"):
        for nfe in (32, 1, 2, 4):
            jobs.append(
                {
                    "task": task,
                    "method": "Original",
                    "seed": None,
                    "nfe": nfe,
                    "league": "parent_baseline",
                    "export": gate["parents"][task],
                    "training_updates": 20000,
                    "export_cumulative_updates": 20000,
                }
            )
    for row in gate["runs"]:
        for nfe in (1, 2, 4):
            export = row["inference_exports"][str(nfe)]
            jobs.append(
                {
                    "task": row["task"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "nfe": nfe,
                    "league": "primary",
                    "export": export,
                    "training_updates": 2000,
                    "export_cumulative_updates": export["cumulative_updates_at_export"],
                }
            )
    for row in gate["historical"]:
        jobs.append(
            {
                "task": row["task"],
                "method": row["method"],
                "seed": row["seed"],
                "nfe": row["nfe"],
                "league": "historical_reference",
                "export": row,
                "training_updates": 1000,
                "export_cumulative_updates": 1000,
            }
        )
    keys = [(j["task"], j["method"], j["seed"], j["nfe"], j["league"]) for j in jobs]
    if len(jobs) != 168 or len(set(keys)) != 168:
        raise ValueError("Expected exactly144 primary,8 parent and16 historical evaluations")
    return jobs


def build_final_banks(registration, core):
    """Called ONLY after verify_final_gate has checked every native export."""
    requested = registration["final"]
    banks = {}
    ray_spec = requested["rays"]
    real = make_environment(
        "rotating_rays",
        n_samples=ray_spec["reference_samples"],
        seed=ray_spec["reference_seed"],
        n_classes=16,
    )
    ray_count = ray_spec["generated_per_condition"]
    target_values = torch.arange(16, dtype=torch.long)
    conditions = target_values.repeat_interleave(ray_count)
    base = torch.randn(
        ray_count, 2, generator=torch.Generator().manual_seed(ray_spec["noise_seed"])
    )
    extra = torch.randn(
        3,
        ray_count,
        2,
        generator=torch.Generator().manual_seed(core.EVALUATION["rays"]["extra_noise_seed"]),
    )
    banks["rays"] = {
        "reference": real.samples,
        "reference_conditions": real.conditions,
        "noise": base.repeat(16, 1),
        "conditions": conditions,
        "condition_ids": conditions.clone(),
        "extra_noise": extra.repeat(1, 16, 1),
        "reference_seed": ray_spec["reference_seed"],
        "noise_seed": ray_spec["noise_seed"],
    }
    wall_spec = requested["wall"]
    real = make_multimodal_navigation(
        n_scenes=wall_spec["n_scenes"],
        samples_per_scene=wall_spec["paths_per_scene"],
        path_length=32,
        seed=wall_spec["scene_seed"],
        split="test",
        jitter=0.025,
        clearance=0.06,
        margin=0.035,
    )
    scene_conditions = real.scene_conditions
    per_scene = wall_spec["paths_per_scene"]
    base = torch.randn(
        per_scene, 30, 2, generator=torch.Generator().manual_seed(wall_spec["noise_seed"])
    )
    extra = torch.randn(
        3,
        per_scene,
        30,
        2,
        generator=torch.Generator().manual_seed(core.EVALUATION["wall"]["extra_noise_seed"]),
    )
    banks["wall"] = {
        "reference": real.samples,
        "reference_conditions": real.conditions,
        "noise": base.repeat(wall_spec["n_scenes"], 1, 1),
        "conditions": scene_conditions.repeat_interleave(per_scene, dim=0),
        "condition_ids": torch.arange(wall_spec["n_scenes"]).repeat_interleave(per_scene),
        "extra_noise": extra.repeat(1, wall_spec["n_scenes"], 1, 1),
        "reference_seed": wall_spec["scene_seed"],
        "noise_seed": wall_spec["noise_seed"],
    }
    for task, bank in banks.items():
        settings = core.EVALUATION[task]
        bank["projections"] = fixed_projection_bank(
            2 if task == "rays" else 60, settings["projection_count"], settings["projection_seed"]
        )
        bank["hashes"] = {
            key: tensor_hash(bank[key])
            for key in (
                "reference",
                "reference_conditions",
                "noise",
                "conditions",
                "condition_ids",
                "extra_noise",
                "projections",
            )
        }
        bank["extra_noise_seed"] = settings["extra_noise_seed"]
        bank["projection_seed"] = settings["projection_seed"]
        if len(bank["noise"]) != requested[task]["total_generated"]:
            raise ValueError("Final bank generated count differs from registration")
    return banks


def _identifier(job):
    seed = "parent" if job["seed"] is None else f"seed{job['seed']}"
    return f"{job['task']}_{job['league']}_{job['method']}_{seed}_nfe{job['nfe']}"


def _public_quality(quality):
    result = copy.deepcopy(quality)
    result.pop("per_sample", None)
    return result


def outcome_arrays(task, samples, bank, quality):
    """Scalar per-sample evidence only; never public endpoint/path coordinates."""
    arrays = {"condition_ids": bank["condition_ids"].cpu().numpy()}
    if task == "wall":
        arrays.update({key: np.asarray(value) for key, value in quality["per_sample"].items()})
    else:
        points = samples.detach().cpu().double()
        targets = bank["conditions"].cpu()
        actual = nearest_ray_ids(points)
        radius = torch.linalg.vector_norm(points, dim=1)
        defined = radius > 0
        phase = torch.atan2(points[:, 1], points[:, 0]) - torch.pi / 2
        phase -= 1.6 * torch.pi * (radius / (10 / 11)).square() / (0.9 * 16)
        requested = (
            torch.remainder(phase - targets * (2 * torch.pi / 16) + torch.pi, 2 * torch.pi)
            - torch.pi
        )
        requested = torch.where(defined, requested, torch.full_like(requested, float("nan")))
        arrays.update(
            {
                "requested_id": targets.numpy(),
                "nearest_geometric_id": actual.numpy(),
                "assignment_defined": defined.numpy(),
                "correct_requested_ray": (actual == targets).numpy(),
                "outer": (radius > 0.25 * (10 / 11)).numpy(),
                "radius": radius.numpy(),
                "requested_phase_error_radians": requested.numpy(),
            }
        )
    return arrays


def per_condition_rows(job, quality):
    prefix = {key: job[key] for key in ("task", "method", "seed", "nfe", "league")}
    distribution = quality["distribution"]
    records = {}
    for condition, value in zip(
        distribution["conditions"], distribution["per_condition"], strict=True
    ):
        key = tuple(condition)
        records[key] = {**prefix, "condition": json.dumps(condition), "conditional_swd": value}
    if job["task"] == "rays":
        for row in quality["routing"]["per_class"]:
            key = (row["target_id"],)
            records[key].update(row)
    else:
        for row in quality["geometry"]["mode_coverage"]["condition_groups"]:
            key = tuple(row["condition"])
            records[key].update(
                {
                    name: json.dumps(value) if isinstance(value, list) else value
                    for name, value in row.items()
                    if name != "condition"
                }
            )
    return list(records.values())


def _fidelity(samples, parent_samples, bank, task, core):
    x = samples if task == "rays" else samples[:, 1:-1]
    y = parent_samples if task == "rays" else parent_samples[:, 1:-1]
    error = x.double() - y.double()
    distribution = core.conditional_swd(
        x, bank["conditions"], y, bank["conditions"], bank["projections"]
    )
    return {
        "paired_endpoint_mse": float(error.square().mean()),
        "paired_endpoint_rmse": float(error.square().mean().sqrt()),
        "conditional_swd_to_parent32": distribution["mean"],
        "state": "raw_xy" if task == "rays" else "raw_interior60_no_fixed_endpoints",
        "scope": "paired same-noise fidelity to one frozen parent32; separate from real-data quality",
    }


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _preview_payload(task, bank):
    chosen = torch.where(bank["condition_ids"] == 0)[0][:16]
    reference_conditions = bank["reference_conditions"]
    if task == "wall":
        selected = (reference_conditions == bank["conditions"][chosen[0]]).all(1)
    else:
        selected = torch.ones(len(reference_conditions), dtype=torch.bool)
    return {
        "task": task,
        "conditions": bank["conditions"][chosen],
        "condition_ids": bank["condition_ids"][chosen],
        "reference_samples": bank["reference"][selected],
        "reference_conditions": reference_conditions[selected],
        "runs": [],
    }, chosen


def _measure_job(job, bank, api, core, *, device, batch_size):
    directory = resolve(job["export"]["path"])
    for name, key in (
        ("config.json", "config_file_sha256"),
        ("diffusion_pytorch_model.safetensors", "file_sha256"),
    ):
        if file_hash(directory / name) != job["export"][key]:
            raise RuntimeError("Native export file changed after final gate")
    model = api.load_zoo_model(directory, device=device)
    if model_hash(model) != job["export"]["sha256"]:
        raise RuntimeError("Native model tensor hash changed after final gate")
    model.zoo_task, model.zoo_method = job["task"], job["method"]
    kwargs = {"nfe": job["nfe"], "batch_size": batch_size}
    if job["method"] == "consistency" and job["nfe"] > 1:
        kwargs["extra_noise"] = bank["extra_noise"][: job["nfe"] - 1]
    if job["nfe"] == 32:
        kwargs["reference"] = True
    started = time.perf_counter()
    result = core.verified_sample(
        api.sample_zoo, model, bank["noise"], bank["conditions"], **kwargs
    )
    sampling_seconds = time.perf_counter() - started
    events = core.sampling_events(result, method=job["method"])
    metadata = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "actual_device": str(model.device),
        "sampling_seconds": sampling_seconds,
        "actual_network_cost": result["actual_network_cost"],
    }
    del model
    return result["samples"].cpu(), events, metadata


def run_evaluation(
    gate_path=DEFAULT_GATE,
    output=DEFAULT_REPORT,
    private_output=DEFAULT_PRIVATE,
    *,
    allow_final=False,
    device="cpu",
    threads=4,
    batch_size=128,
    render=True,
):
    """Explicit final-bank opening; no final factory precedes complete preflight."""
    if not allow_final:
        raise PermissionError(
            "Final bank remains closed; coordinator GO requires --open-final-bank"
        )
    if threads < 1 or batch_size < 1:
        raise ValueError("threads and batch_size must be positive")
    core, api = evaluation_core(), training_api()
    fixed = core.verify_final_gate(gate_path, verify_native=True, api=api)
    gate, registration = fixed["gate"], fixed["registration"]
    jobs = final_jobs(gate)
    torch.set_num_threads(threads)
    chosen_device = choose_device(device)
    output, private_output = Path(output), Path(private_output)
    if output.resolve() == private_output.resolve():
        raise ValueError("Public report and private endpoint cache must be different directories")
    output.mkdir(parents=True, exist_ok=True)
    private_output.mkdir(parents=True, exist_ok=True)
    contract = {
        "gate_sha256": file_hash(gate_path),
        "driver_sha256": file_hash(__file__),
        "core_sha256": file_hash(core.__file__),
        "actual_device": str(chosen_device),
        "batch_size": batch_size,
        "threads": threads,
    }
    contract_path = private_output / "evaluation_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Preserve the previous cache; its source/gate/device contract differs")
    atomic_json(contract_path, contract)
    banks = build_final_banks(registration, core)  # FIRST final-data/noise factory.
    bank_metadata = {
        task: {key: value for key, value in bank.items() if not torch.is_tensor(value)}
        for task, bank in banks.items()
    }
    for task, bank in banks.items():
        save_checkpoint(private_output / f"{task}_final_bank.pt", bank)
    atomic_json(output / "final_banks.json", bank_metadata)
    previews, selected = {}, {}
    for task, bank in banks.items():
        previews[task], selected[task] = _preview_payload(task, bank)
    rows, condition_rows, parent_samples = [], [], {}
    (output / "outcomes").mkdir(exist_ok=True)
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "scope": "all48_fixed_before_final_banks; exploratory3seedcomparison",
        **contract,
        "all48_metadata_fixed": fixed["all48_metadata_fixed"],
        "all144_exports_fixed": fixed["all144_exports_fixed"],
        "actual_native_hashes_verified": fixed["actual_native_hashes_verified"],
        "expected_rows": 168,
        "reference_and_noise_hashes": bank_metadata,
        "raw_endpoints_and_full_histories": "private ignored cache only",
        "public_preview_policy": "target0/scene0 first16 noise draws; newseed161 plus frozenparents",
        "test_opening": "Explicit coordinator-authorized --open-final-bank after native fixation",
    }
    try:
        for index, job in enumerate(jobs):
            task, bank, identity = job["task"], banks[job["task"]], _identifier(job)
            cache = private_output / f"{identity}.pt"
            cache_contract = {**contract, "job": job, "bank_hashes": bank["hashes"]}
            if cache.exists():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if (
                    saved["contract"] != cache_contract
                    or tensor_hash(saved["samples"]) != saved["samples_sha256"]
                    or tensor_hash(saved["events"]["states"]) != saved["event_states_sha256"]
                    or tensor_hash(saved["events"]["times"]) != saved["event_times_sha256"]
                ):
                    raise ValueError(
                        "Cached endpoint evidence changed or belongs to a different contract"
                    )
                samples, events, measure = saved["samples"], saved["events"], saved["measurement"]
            else:
                samples, events, measure = _measure_job(
                    job, bank, api, core, device=chosen_device, batch_size=batch_size
                )
                save_checkpoint(
                    cache,
                    {
                        "contract": cache_contract,
                        "samples": samples,
                        "samples_sha256": tensor_hash(samples),
                        "events": events,
                        "event_states_sha256": tensor_hash(events["states"]),
                        "event_times_sha256": tensor_hash(events["times"]),
                        "measurement": measure,
                    },
                )
            if job["league"] == "parent_baseline" and job["nfe"] == 32:
                parent_samples[task] = samples
            if task == "rays":
                quality = core.rays_quality(
                    samples,
                    bank["conditions"],
                    bank["reference"],
                    bank["reference_conditions"],
                    bank["projections"],
                    initial_noise=bank["noise"],
                )
            else:
                quality = core.wall_quality(
                    samples,
                    bank["conditions"],
                    bank["reference"],
                    bank["reference_conditions"],
                    bank["projections"],
                )
            outcome_path = output / "outcomes" / f"{identity}.npz"
            np.savez_compressed(outcome_path, **outcome_arrays(task, samples, bank, quality))
            row = {
                key: job[key]
                for key in (
                    "task",
                    "method",
                    "seed",
                    "nfe",
                    "league",
                    "training_updates",
                    "export_cumulative_updates",
                )
            }
            row.update(
                {
                    "status": "completed",
                    "model_sha256": job["export"]["sha256"],
                    "native_export": job["export"],
                    "initial_noise_sha256": bank["hashes"]["noise"],
                    "conditions_sha256": bank["hashes"]["conditions"],
                    "extra_noise_sha256": tensor_hash(bank["extra_noise"][: job["nfe"] - 1])
                    if job["method"] == "consistency" and job["nfe"] > 1
                    else None,
                    "samples_sha256": tensor_hash(samples),
                    "private_cache_file_sha256": file_hash(cache),
                    "event_states_sha256": tensor_hash(events["states"]),
                    "event_times": events["times"].tolist(),
                    "event_kinds": events["kinds"],
                    "outcomes_file": str(outcome_path.relative_to(output)),
                    "outcomes_file_sha256": file_hash(outcome_path),
                    "quality": _public_quality(quality),
                    "parent_fidelity": _fidelity(samples, parent_samples[task], bank, task, core),
                    "inference": measure,
                }
            )
            if job["league"] == "primary":
                manifest = fixed["manifests"][(task, job["method"], job["seed"])]
                row["training_cost"] = manifest["training_cost"]
                row["training_seconds"] = manifest["training_seconds"]
                row["training_manifest_sha256"] = file_hash(
                    resolve(
                        next(
                            run["directory"]
                            for run in gate["runs"]
                            if (run["task"], run["method"], run["seed"])
                            == (task, job["method"], job["seed"])
                        )
                    )
                    / "manifest.json"
                )
            rows.append(row)
            condition_rows.extend(per_condition_rows(job, quality))
            if job["league"] == "parent_baseline" or (
                job["league"] == "primary" and job["seed"] == 161
            ):
                preview = {key: job[key] for key in ("method", "seed", "nfe", "league")}
                preview["events"] = {**events, "states": events["states"][:, selected[task]]}
                preview["model_sha256"] = row["model_sha256"]
                previews[task]["runs"].append(preview)
            metadata["completed_rows"] = len(rows)
            atomic_json(output / "results.json", {"metadata": metadata, "rows": rows})
            print(f"[{index + 1}/168] {identity}", flush=True)
        if len(rows) != 168:
            raise RuntimeError("Incomplete fixed evaluation cohort")
        _write_csv(output / "per_condition.csv", condition_rows)
        core.save_ranking_artifacts(rows, output)
        preview_dir = output / "previews"
        preview_dir.mkdir(exist_ok=True)
        preview_refs = {}
        for task, payload in previews.items():
            payload["provenance"] = {**contract, "bank_hashes": banks[task]["hashes"]}
            path = preview_dir / f"{task}_preview.pt"
            save_checkpoint(path, payload)
            preview_refs[task] = {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
            if render:
                core.render_method_zoo_previews(path, preview_dir / task)
        metadata.update(
            status="completed",
            completed_rows=len(rows),
            evaluation_seconds=time.perf_counter() - started,
            public_previews=preview_refs,
        )
        atomic_json(output / "results.json", {"metadata": metadata, "rows": rows})
    except BaseException as error:
        metadata.update(
            status="failed",
            completed_rows=len(rows),
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "results.json", {"metadata": metadata, "rows": rows})
        raise
    return {"metadata": metadata, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "replication/reports/method_zoo/registration.json",
    )
    parser.add_argument("--build-gate", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--open-final-bank", action="store_true", help="Use only after explicit coordinator GO"
    )
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)
    if args.build_gate:
        evaluation_core().build_final_gate(args.run_root, args.registration, args.gate)
    if not args.open_final_bank:
        if args.build_gate:
            print(
                json.dumps({"status": "gate_fixed_final_bank_still_closed", "gate": str(args.gate)})
            )
            return 0
        parser.error(
            "Final bank remains closed; provide --open-final-bank only after coordinator GO"
        )
    run_evaluation(
        args.gate,
        args.output,
        args.private_output,
        allow_final=True,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        render=not args.no_render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
