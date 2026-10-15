"""Run the visible notebook13 student loops at fixed low NFE, without hidden trainers."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from recheck.runtime import atomic_json, model_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfe", type=int, choices=(2, 4), required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--output", default="replication/artifacts/rays16_conditional_low_nfe_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive updates and unique seeds are required")
    notebook_path = Path("notebooks/13_conditional_rays_distillation.ipynb")
    source_bytes = notebook_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    cells = json.loads(source_bytes)["cells"]
    root = Path(args.output) / f"nfe{args.nfe}"
    root.mkdir(parents=True, exist_ok=True)
    parent = json.loads(
        Path("replication/demo/rays16_conditional_v1/teacher/manifest.json").read_text()
    )
    contract = {
        "scope": "registered_conditional_low_nfe_comparison_one_frozen_parent",
        "teacher_sha256": parent["teacher_sha256"],
        "teacher_steps": 32,
        "student_steps": args.nfe,
        "student_updates": args.updates,
        "seeds": args.seeds,
        "methods": ["M0", "M1", "E0", "E1"],
        "batch_size": 128,
        "lr": 1e-4,
        "warmup": max(1, args.updates // 10),
        "ema_decay": 0.99,
        "tube_rho": 0.1,
        "null_probability": 0.1,
        "notebook_sha256": source_sha,
        "loss_support": "one requested condition or trained null per entire OT support",
        "assessment_rule": "No new final noise/reference until all24 models across both grids completed",
        "final_noise_seed": 940001,
        "final_reference_seed": 20047,
    }
    contract_path = root / "registration.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing study contract differs; use a new output directory")
    atomic_json(contract_path, contract)
    status = {"status": "running", "contract": contract, "runs": []}
    atomic_json(root / "suite_status.json", status)
    try:
        for seed in args.seeds:
            namespace = {"__name__": "__conditional_low_nfe_notebook__"}
            started = time.perf_counter()
            for index in [2, 3, 4, 5, 7, 9, 11, 13]:
                cell = cells[index]
                if cell["cell_type"] != "code":
                    raise RuntimeError(f"Notebook cell layout changed at {index}")
                exec(
                    compile("".join(cell["source"]), f"{notebook_path}:cell{index}", "exec"),
                    namespace,
                )
                if index == 3:
                    namespace.update(
                        STUDENT_STEPS=args.nfe,
                        UPDATES=args.updates,
                        WARMUP=max(1, args.updates // 10),
                        SEED=seed,
                        OUTPUT_DIR=root / f"seed{seed}",
                        CHECKPOINT_EVERY=100,
                        RESUME=args.resume,
                        SAVE_GIFS=False,
                        SHOW_WIDGETS=False,
                        PROGRESS=False,
                        TRAIN_TEACHER=False,
                        DEVICE="cpu",
                        THREADS=4,
                    )
                if index == 7 and model_hash(namespace["teacher"]) != contract["teacher_sha256"]:
                    raise RuntimeError("Loaded another parent")
            run = {
                "seed": seed,
                "status": "completed",
                "wall_seconds_four_methods": time.perf_counter() - started,
                "teacher_sha256_after": model_hash(namespace["teacher"]),
                "updates_each": args.updates,
                "methods": {
                    name: model_hash(model) for name, model in namespace["trained"].items()
                },
                "common_rng_traces_equal": all(
                    trace == namespace["pairing_traces"]["M0"]
                    for trace in namespace["pairing_traces"].values()
                ),
            }
            assert run["common_rng_traces_equal"]
            status["runs"].append(run)
            atomic_json(root / "suite_status.json", status)
            print(json.dumps(run), flush=True)
        status["status"] = "completed"
    except BaseException as error:
        status.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        atomic_json(root / "suite_status.json", status)
    print(
        json.dumps({"nfe": args.nfe, "status": "completed", "runs": len(status["runs"]) * 4}),
        flush=True,
    )


if __name__ == "__main__":
    main()
