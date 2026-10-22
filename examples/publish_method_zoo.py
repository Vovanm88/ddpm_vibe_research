"""Publish selected method-zoo EMA exports and independently verify their audit.

Private training checkpoints and full teacher trajectory banks are never copied.
``--verify-only`` needs only the public package and its referenced native parents.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch

from diffusion_lab import PointDenoiser
from diffusion_lab.interval_model import IntervalPointDenoiser
from recheck.runtime import model_hash, tensor_hash

ROOT = Path(__file__).resolve().parents[1]
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
TASKS = ("rays", "wall")
SEEDS = (161, 262, 363)
TRAINING_SOURCE = "69a55a69fa7f6f1658e374b15d841c158b19b16c7b200187c64d5f48a8d91c77"
PARENTS = {
    "rays": {
        "path": "replication/demo/rays16_conditional_v1/teacher/ema_model",
        "sha256": "9bc0dbe39c4cb24c4d9b893bac084c89be66fdd7b29d6879fca786b4020d6ce1",
    },
    "wall": {
        "path": "replication/demo/wall_navigation_conditional_v1/teacher/ema_model",
        "sha256": "71d77760c37e6a053f324f2c810adb323439ed76ef51f09a57db43a5d2da9e79",
    },
}


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def state_sha(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(tensor_hash(tensor).encode())
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def native_info(path, expected):
    path = Path(path)
    native = read(path / "config.json")
    classes = {"PointDenoiser": PointDenoiser, "IntervalPointDenoiser": IntervalPointDenoiser}
    cls = classes[native["_class_name"]]
    model = (
        cls.from_pretrained(
            path, local_files_only=True, low_cpu_mem_usage=False, torch_dtype=torch.float32
        )
        .eval()
        .requires_grad_(False)
    )
    actual = model_hash(model)
    if actual != expected:
        raise ValueError(f"Native tensor hash mismatch: {path}")
    return {
        "model_class": native["_class_name"],
        "sha256": actual,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "buffer_elements": sum(buffer.numel() for buffer in model.buffers()),
    }


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha(source) != file_sha(destination):
        raise ValueError(f"File copy differs: {source}")


def copy_native(source, destination):
    files = {path.name for path in source.iterdir()}
    expected = {"config.json", "diffusion_pytorch_model.safetensors"}
    if files != expected:
        raise ValueError(f"Unexpected native file set: {source}: {files}")
    for name in sorted(files):
        copy_file(source / name, destination / name)


def check_manifest(manifest, task, seed, method):
    expected = {
        "status": "completed",
        "task": task,
        "method": method,
        "seed": seed,
        "updates_planned": 2000,
        "updates_completed": 2000,
        "teacher_sha256": PARENTS[task]["sha256"],
        "source_sha256": TRAINING_SOURCE,
        "initialization": "parent",
        "whole_support_single_condition": True,
        "route_id_input": False,
        "cfg": False,
        "condition_dropout": 0.0,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"Incomplete/mismatched cohort {task}/{seed}/{method}/{name}")


def audit_checkpoint(directory, manifest):
    path = directory / "training.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["completed"] != 2000 or saved["status"] != "completed":
        raise ValueError(f"Incomplete private checkpoint: {path}")
    if state_sha(saved["ema"]) != manifest["ema_sha256"]:
        raise ValueError(f"Private checkpoint EMA differs: {path}")
    if saved["history"] != read(directory / "history.json"):
        raise ValueError(f"Private checkpoint history differs: {path}")
    for key, expected in (
        ("raw_trace", manifest["raw_batch_trace_sha256"]),
        ("pair_trace", manifest["pair_indices_trace_sha256"]),
    ):
        values = saved[key]
        if len(values) not in (0, 2000) or any(
            not isinstance(v, str) or len(v) != 64 for v in values
        ):
            raise ValueError(f"Malformed trace {path}/{key}")
        if digest(values) != expected:
            raise ValueError(f"Trace hash differs: {path}/{key}")
    return {
        "schema_version": 1,
        "task": manifest["task"],
        "method": manifest["method"],
        "seed": manifest["seed"],
        "updates_completed": 2000,
        "source_checkpoint": {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha(path),
            "size_bytes": path.stat().st_size,
            "status": "completed",
        },
        "training_contract_sha256": digest(saved["contract"]),
        "raw_trace": saved["raw_trace"],
        "pair_trace": saved["pair_trace"],
        "raw_batch_trace_sha256": digest(saved["raw_trace"]),
        "pair_indices_trace_sha256": digest(saved["pair_trace"]),
        "raw_trace_definition": "SHA256 of condition ID, selected clean indices, raw Gaussian noise and raw uniform time, before any OT or interval transform",
        "pair_trace_definition": "SHA256 of condition ID and selected shared teacher pairbank indices; endpoint/reflow only",
        "optimizer_state_included": False,
        "rng_state_included": False,
        "exact_resume_available": False,
        "limitation": "Trace digests audit matched streams; they do not reconstruct optimizer or RNG states.",
    }


def verify_publication(destination):
    destination = Path(destination)
    publication = read(destination / "PUBLICATION_MANIFEST.json")
    expected_files = {record["path"] for record in publication["files"]}
    actual_files = {
        str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()
    }
    if actual_files != expected_files | {"PUBLICATION_MANIFEST.json"}:
        raise ValueError("Public file set differs from inventory")
    for record in publication["files"]:
        relative = Path(record["path"])
        path = (destination / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(destination.resolve()):
            raise ValueError("Invalid publication-relative path")
        if path.stat().st_size != record["size_bytes"] or file_sha(path) != record["sha256"]:
            raise ValueError(f"Publication file differs: {path}")
        if path.suffix in (".pt", ".pth", ".pkl", ".bin", ".whl") or path.is_symlink():
            raise ValueError(f"Private checkpoint/binary archive leaked: {path}")
    counts = {"runs": 0, "native_exports": 0, "inference_references": 0}
    verified = {}
    for task in TASKS:
        parent = native_info(ROOT / PARENTS[task]["path"], PARENTS[task]["sha256"])
        for seed in SEEDS:
            manifests = {}
            for method in METHODS:
                directory = destination / task / f"seed{seed}" / method
                manifest = read(directory / "manifest.json")
                check_manifest(manifest, task, seed, method)
                audit = read(directory / "audit_traces.json")
                if (
                    digest(audit["raw_trace"]) != manifest["raw_batch_trace_sha256"]
                    or digest(audit["pair_trace"]) != manifest["pair_indices_trace_sha256"]
                ):
                    raise ValueError("Public audit trace differs from original manifest")
                manifests[method] = manifest
                native_path = directory / "ema_model"
                info = native_info(native_path, manifest["ema_sha256"])
                verified[str(native_path.relative_to(destination))] = info
                counts["native_exports"] += 1
                if method == "progressive":
                    for stage in (1, 2, 4):
                        stage_dir = directory / f"stage{stage}"
                        stage_manifest = read(stage_dir / "manifest.json")
                        verified[str((stage_dir / "ema_model").relative_to(destination))] = (
                            native_info(stage_dir / "ema_model", stage_manifest["ema_sha256"])
                        )
                        counts["native_exports"] += 1
                    if any((directory / f"stage{n}" / "ema_model").exists() for n in (16, 8)):
                        raise ValueError("Stage16/8 weights must remain private")
                for nfe in (1, 2, 4):
                    export = manifest["inference_exports"][str(nfe)]
                    path = (directory / export["path"]).resolve()
                    if not path.is_relative_to(directory.resolve()):
                        raise ValueError("Inference export escapes its method directory")
                    info = verified[str(path.relative_to(destination.resolve()))]
                    if info["sha256"] != export["sha256"]:
                        raise ValueError("Public inference reference hash differs")
                    expected_updates = (
                        {4: 1200, 2: 1600, 1: 2000}[nfe] if method == "progressive" else 2000
                    )
                    if export["cumulative_updates_at_export"] != expected_updates:
                        raise ValueError("Wrong cumulative update count")
                    counts["inference_references"] += 1
                if info["parameter_count"] < parent["parameter_count"]:
                    raise ValueError("Unexpected parameter count")
                counts["runs"] += 1
            for method in ("ot_cfm", "shortcut", "meanflow"):
                if (
                    manifests[method]["raw_batch_trace_sha256"]
                    != manifests["cfm"]["raw_batch_trace_sha256"]
                ):
                    raise ValueError("Public paired raw training traces differ")
            if (
                manifests["endpoint"]["pair_indices_trace_sha256"]
                != manifests["reflow"]["pair_indices_trace_sha256"]
            ):
                raise ValueError("Public endpoint/reflow pair traces differ")
    return counts | {
        "files": len(actual_files),
        "total_bytes": sum((destination / p).stat().st_size for p in actual_files),
        "native_tensor_hashes_verified": len(verified),
        "exact_resume_available": False,
    }


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        return verify_publication(destination)
    # All 48 cases must be complete before any package directory is created.
    records = []
    for task in TASKS:
        for seed in SEEDS:
            for method in METHODS:
                directory = source / task / f"seed{seed}" / method
                manifest = read(directory / "manifest.json")
                check_manifest(manifest, task, seed, method)
                records.append(
                    (task, seed, method, directory, manifest, audit_checkpoint(directory, manifest))
                )
    stage = destination.parent / ("." + destination.name + "_publishing")
    if stage.exists():
        raise FileExistsError(f"Preserve/review incomplete publication directory first: {stage}")
    stage.mkdir(parents=True)
    rows = []
    parent_info = {
        task: native_info(ROOT / PARENTS[task]["path"], PARENTS[task]["sha256"]) for task in TASKS
    }
    for task, seed, method, directory, manifest, audit in records:
        target = stage / task / f"seed{seed}" / method
        for name in ("manifest.json", "history.json"):
            copy_file(directory / name, target / name)
        copy_native(directory / "ema_model", target / "ema_model")
        write(target / "audit_traces.json", audit)
        if method == "progressive":
            for nfe in (16, 8, 4, 2, 1):
                source_stage, target_stage = directory / f"stage{nfe}", target / f"stage{nfe}"
                copy_file(source_stage / "manifest.json", target_stage / "manifest.json")
                if nfe in (4, 2, 1):
                    copy_native(source_stage / "ema_model", target_stage / "ema_model")
        info = native_info(target / "ema_model", manifest["ema_sha256"])
        row = {
            "task": task,
            "method": method,
            "seed": seed,
            "updates_completed": 2000,
            "parameter_count": info["parameter_count"],
            "extra_interval_parameters": info["parameter_count"]
            - parent_info[task]["parameter_count"],
            "training_seconds": manifest["training_seconds"],
            "ema_sha256": manifest["ema_sha256"],
            "pairbank_sha256": manifest["pairbank_sha256"],
        }
        for role in ("student", "teacher", "ema", "stage_teacher"):
            for metric in ("forward_calls", "sample_evals"):
                field = role + "_" + metric
                row[field] = manifest["training_cost"].get(field, 0)
        rows.append(row)
    copy_file(source / "run_config.json", stage / "run_config.json")
    banks = {}
    for task in TASKS:
        copy_file(source / task / "pairbank/manifest.json", stage / task / "pairbank/manifest.json")
        banks[task] = read(stage / task / "pairbank/manifest.json")
    timing = "Two task cohorts trained concurrently on CPU4 each. Elapsed training time is descriptive operational cost, not an isolated latency/speed ranking. Forward/sample counts are not FLOPs; JVP/backward/optimizer work is included in elapsed time."
    write(
        stage / "training_summary.json",
        {
            "schema_version": 1,
            "runs": rows,
            "timing_context": timing,
            "common_parent_updates_per_task": 20000,
            "parents": {task: PARENTS[task] | parent_info[task] for task in TASKS},
            "shared_pairbank_generation_once_per_task": banks,
            "total_new_optimizer_updates": 48 * 2000,
            "exact_resume_available": False,
        },
    )
    with (stage / "training_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (stage / "README.md").write_text("""# Method-zoo selected native EMA exports

48 completed warm-start runs: rays/wall × eight methods × seeds 161/262/363, 2000 updates each.
Read each method's `manifest.json` → `inference_exports['1'|'2'|'4']` for its native path,
tensor SHA, trained/off-target interpretation and cumulative updates. Progressive uses
stage4/2/1 at cumulative1200/1600/2000; stage16/8 publish metadata only. Every full run
finished2000 before the final assessment. Histories contain all2000 optimizer updates.

Native weights use Diffusers `save_pretrained`. `examples/train_method_zoo.py` exposes
`load_zoo_model` and `sample_zoo`; conditional inputs remain mandatory, CFG is off.
Wall models predict30 interior points and reconstruct A/B: exact endpoints are construction.
Consistency uses explicit denoise/re-noise events and an additional Gaussian bank for K>1.

Only selected EMA weights, manifests, loss histories, config and hashed training traces
are public. **Exact resume is unavailable from this package.** Live weights, AdamW,
RNG states, private `training.pt` and full8192-row teacher trajectory banks are excluded.
`audit_traces.json` preserves raw/pair trace lists plus source-checkpoint SHA/size;
the digests audit matched streams but cannot reconstruct optimizer/RNG state.
Pairbank manifests preserve hashes/costs, without full trajectories or latent banks.

The two20k parent exports remain in `replication/demo/rays16_conditional_v1/teacher`
and `replication/demo/wall_navigation_conditional_v1/teacher`; they are not duplicated.
Training source: `replication/reports/method_zoo/training_notebook15.ipynb` (69a55…);
the current notebook has a separately recorded GIF-only UI amendment with identical API.
See `training_summary.json`/`.csv` for parameters and operational costs. Timings came
from concurrent task jobs and do not support speed/efficiency claims.

Verify payload file sizes/SHA, all native tensor hashes and144 inference references:
`python examples/publish_method_zoo.py --verify-only`
""")
    inventory = [
        {
            "path": str(path.relative_to(stage)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    ]
    write(
        stage / "PUBLICATION_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "completed",
            "scope": "selected_ema_only_method_zoo_48_warm_started_runs",
            "exact_resume_available": False,
            "private_training_checkpoints_included": False,
            "full_pairbanks_included": False,
            "parents": PARENTS,
            "training_source_sha256": TRAINING_SOURCE,
            "payload_file_count": len(inventory),
            "payload_total_bytes": sum(row["size_bytes"] for row in inventory),
            "inventory_scope": "All payload files; this self-describing manifest is excluded to avoid self-reference.",
            "files": inventory,
        },
    )
    result = verify_publication(stage)
    stage.rename(destination)
    return result | {"path": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "replication/artifacts/method_zoo_v1")
    parser.add_argument("--destination", type=Path, default=ROOT / "replication/demo/method_zoo_v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    result = (
        verify_publication(args.destination)
        if args.verify_only
        else publish(args.source, args.destination)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
