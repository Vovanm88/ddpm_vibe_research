"""Validation-only, cached whole-route assessment of the frozen wall teacher.

Default invocation is inert. After explicit approval and teacher completion:
python examples/evaluate_wall_navigation.py --execute

No training, test split, mode conditioning, clipping or path repair occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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
    plot_navigation_paths,
    save_navigation_denoising_gif,
    save_navigation_particle_gif,
    summarize_navigation_benchmark,
)
from recheck.evaluate import fixed_projection_bank, sliced_wasserstein_fixed
from recheck.runtime import atomic_json, model_hash, save_checkpoint, tensor_hash

EXPECTED_TEACHER = "71d77760c37e6a053f324f2c810adb323439ed76ef51f09a57db43a5d2da9e79"
VALIDATION = {
    "n_scenes": 32,
    "samples_per_scene": 128,
    "path_length": 32,
    "seed": 10061,
    "split": "validation",
    "jitter": 0.025,
    "clearance": 0.06,
    "margin": 0.035,
}
NFES = (1, 2, 4, 8, 16, 32, 64)
NOISE_SEED = 613001
PROJECTION_SEED = 861003


def reconstruct(interior, conditions):
    return torch.cat((conditions[:, None, :2], interior, conditions[:, None, 2:4]), dim=1)


@torch.no_grad()
def sample_navigation(model, initial_noise, conditions, *, steps, batch_size=128):
    """Explicit Euler on interior points; A/B are reconstructed at every node."""
    if steps < 1 or initial_noise.ndim != 3 or initial_noise.shape[1:] != (30, 2):
        raise ValueError("Positive steps and initial noise[B,30,2] are required")
    if conditions.shape != (len(initial_noise), 8):
        raise ValueError("conditions must be[B,8]")
    times = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    histories = []
    was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(initial_noise), batch_size):
            state = initial_noise[offset : offset + batch_size].to(model.device).clone()
            context = conditions[offset : offset + batch_size].to(model.device)
            nodes = [reconstruct(state, context).cpu()]
            for t, width in zip(times[:-1], times[:-1] - times[1:]):
                predicted = model(state, float(t) * 1000, conditions=context).sample
                state = state - float(width) * predicted
                if not torch.isfinite(state).all():
                    raise FloatingPointError("Nonfinite raw navigation Euler state")
                nodes.append(reconstruct(state, context).cpu())
            histories.append(torch.stack(nodes))
    finally:
        model.train(was_training)
    history = torch.cat(histories, dim=1)
    return {
        "samples": history[-1],
        "history": history,
        "times": times,
        "nfe": steps,
        "initial_noise_sha256": tensor_hash(initial_noise),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--teacher", default="replication/artifacts/wall_navigation_conditional_v1/teacher"
    )
    parser.add_argument("--output", default="replication/reports/wall_navigation/teacher_qc")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "split": "validation_only",
                    "validation": VALIDATION,
                    "nfe": NFES,
                    "noise_seed": NOISE_SEED,
                    "teacher": args.teacher,
                    "output": args.output,
                    "threads": 4,
                }
            )
        )
        return
    torch.set_num_threads(4)
    teacher_dir, output = Path(args.teacher), Path(args.output)
    manifest = json.loads((teacher_dir / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["updates_completed"] != 20000:
        raise RuntimeError("Frozen teacher must complete20000 updates before QC")
    assert manifest["teacher_sha256"] == EXPECTED_TEACHER
    assert manifest["config"]["validation"] == VALIDATION
    assert not manifest["route_ids_used_by_model"]
    teacher = (
        PointDenoiser.from_pretrained(
            teacher_dir / "ema_model", local_files_only=True, low_cpu_mem_usage=False
        )
        .to("cpu")
        .eval()
        .requires_grad_(False)
    )
    assert teacher.sample_shape == (30, 2) and teacher.config.condition_dim == 8
    assert model_hash(teacher) == EXPECTED_TEACHER
    if (output / "metrics.json").is_file():
        raise FileExistsError("Completed QC already exists; preserve its fixed measurements")
    output.mkdir(parents=True, exist_ok=True)
    real = make_multimodal_navigation(**VALIDATION)
    conditions = real.conditions.cpu()
    reference = evaluate_navigation(real.samples, conditions)
    assert reference["success_fraction"] == 1.0
    base_noise = torch.randn((128, 30, 2), generator=torch.Generator().manual_seed(NOISE_SEED))
    noise = torch.empty((len(real), 30, 2))
    for scene in range(VALIDATION["n_scenes"]):
        selected = torch.nonzero(real.scene_indices == scene).flatten()
        assert len(selected) == 128
        noise[selected] = base_noise
    projections = fixed_projection_bank(60, 128, PROJECTION_SEED)
    real_flat = real.samples[:, 1:-1].flatten(1)
    scene0 = torch.nonzero(real.scene_indices == 0).flatten()
    assert len(scene0) == 128
    preview = scene0[:32]
    started = time.perf_counter()
    state = {
        "status": "running",
        "scope": "one_seed_exploratory_validation_teacher_qc",
        "split": "validation",
        "test_accessed": False,
        "teacher_sha256": EXPECTED_TEACHER,
        "teacher_updates": 20000,
        "validation": VALIDATION,
        "noise_seed": NOISE_SEED,
        "noise_sha256": tensor_hash(noise),
        "conditions_sha256": tensor_hash(conditions),
        "reference_paths_sha256": tensor_hash(real.samples),
        "reference_quality": reference,
        "projection_seed": PROJECTION_SEED,
        "projection_bank_sha256": tensor_hash(projections),
        "reference_distribution_metric": "fixed128-direction sliced W1 on raw interior60D; per-scene and marginal",
        "preview_rule": "heldout scene0, first32 noise paths; physical playback first4; no quality selection",
        "endpoint_contract": "A/B reconstructed exactly; endpoint accuracy is not learned",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    histories = {}
    for nfe in NFES:
        before = time.perf_counter()
        sampled = sample_navigation(teacher, noise, conditions, steps=nfe)
        histories[f"Original N{nfe}"] = sampled["history"]
        quality = evaluate_navigation(sampled["samples"], conditions)
        generated = sampled["samples"][:, 1:-1].flatten(1)
        swd = sliced_wasserstein_fixed(real_flat, generated, projections)
        scene_swd = []
        for scene in range(VALIDATION["n_scenes"]):
            selected = real.scene_indices == scene
            scene_swd.append(
                sliced_wasserstein_fixed(real_flat[selected], generated[selected], projections)
            )
        row = {
            "nfe": nfe,
            "quality": quality,
            "sliced_w1_interior_real": swd,
            "mean_scene_sliced_w1_interior_real": float(np.mean(scene_swd)),
            "scene_sliced_w1_interior_real": scene_swd,
            "generation_and_assessment_seconds": time.perf_counter() - before,
        }
        state["rows"].append(row)
        atomic_json(output / "progress.json", state)
        print(
            json.dumps(
                {
                    "nfe": nfe,
                    "success": quality["success_fraction"],
                    "collision_free": quality["collision_free_fraction"],
                    "mode_coverage": quality["mode_coverage"],
                    "sliced_w1_real": swd,
                    "mean_scene_sliced_w1_real": row["mean_scene_sliced_w1_interior_real"],
                }
            ),
            flush=True,
        )
    # Matched noise and exact common FM nodes; Teacher64 is a descriptive reference.
    fidelity = summarize_navigation_benchmark(histories, conditions, parent="Original N64")
    for row, assessment in zip(state["rows"], fidelity["rows"]):
        row["fidelity_to_teacher64"] = assessment["fidelity"]
    assert model_hash(teacher) == EXPECTED_TEACHER
    state.update(
        status="metrics_completed_render_pending",
        assessment_seconds=time.perf_counter() - started,
        limitations=fidelity["limitations"]
        + [
            "Teacher64 is a numerical reference, not ground-truth dynamics.",
            "Generation timing shares the host with other authorized work; not an isolated latency benchmark.",
        ],
    )
    atomic_json(output / "metrics.json", state)
    save_checkpoint(
        output / "teacher_validation_bank.pt",
        {
            "histories_by_method": histories,
            "initial_noise": noise,
            "conditions": conditions,
            "real_paths": real.samples,
            "scene_indices": real.scene_indices,
        },
    )
    print(
        "METRICS_READY: validation-only teacher bank fixed; rendering cached scene0 previews",
        flush=True,
    )
    context = conditions[scene0[0]]
    scene = {"start": context[:2], "goal": context[2:4], "walls": context[4:].reshape(1, 2, 2)}
    for name, paths in (
        ("reference_scene0_paths", real.samples[scene0]),
        ("teacher_n64_scene0_paths", histories["Original N64"][-1, preview]),
    ):
        figure, _ = plot_navigation_paths(paths, title=name.replace("_", " "), **scene)
        for extension in ("png", "svg"):
            figure.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(figure)
    preview_histories = {name: value[:, preview] for name, value in histories.items()}
    save_checkpoint(
        output / "scene0_histories.pt",
        {"histories_by_method": preview_histories, "condition": context},
    )
    save_navigation_denoising_gif(
        {"Teacher N64": preview_histories["Original N64"]},
        output / "teacher_n64_fm_denoising.gif",
        fps=10,
        max_frames=33,
        **scene,
    )
    save_navigation_particle_gif(
        preview_histories["Original N64"][-1, :4],
        output / "teacher_n64_spatial_particle.gif",
        fps=10,
        substeps=1,
        **scene,
    )
    state.update(status="completed", elapsed_seconds=time.perf_counter() - started)
    atomic_json(output / "metrics.json", state)
    print(
        json.dumps(
            {
                "status": "completed",
                "metrics": str(output / "metrics.json"),
                "seconds": state["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
