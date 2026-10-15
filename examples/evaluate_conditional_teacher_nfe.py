"""Posthoc same-noise NFE diagnostic for the published conditional-ray teacher."""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy.stats import wasserstein_distance

from diffusion_lab import PointDenoiser
from diffusion_lab.conditional_metrics import conditional_routing_metrics
from diffusion_lab.environments import make_environment
from recheck.evaluate import fit_feature_reference, fixed_projection_bank, rays_quality
from recheck.runtime import atomic_json, model_hash, tensor_hash


def main():
    parser = argparse.ArgumentParser(
        description="Check conditional ray teacher quality across Euler budgets; never trains."
    )
    parser.add_argument(
        "--output", default="replication/reports/conditional_rays/teacher_nfe_sweep"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path("replication/demo/rays16_conditional_v1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = (
        PointDenoiser.from_pretrained(
            root / "teacher/ema_model", local_files_only=True, low_cpu_mem_usage=False
        )
        .eval()
        .requires_grad_(False)
    )
    expected = json.loads((root / "teacher/manifest.json").read_text())["teacher_sha256"]
    assert model_hash(model) == expected
    train = make_environment("rotating_rays", n_samples=65536, n_classes=16, seed=47)
    real = make_environment("rotating_rays", n_samples=8192, n_classes=16, seed=10047)
    real2 = make_environment("rotating_rays", n_samples=4096, n_classes=16, seed=10048)
    base = torch.randn((256, 2), generator=torch.Generator().manual_seed(830001))
    noise = base.repeat(16, 1)
    targets = torch.arange(16).repeat_interleave(256)
    bank = fixed_projection_bank(2, 1024, 1717)
    refs = {
        c: fit_feature_reference(
            train.samples[train.conditions == c],
            seed=471600 + c,
            bandwidth_samples=1024,
            standardize=False,
        )
        for c in range(16)
    }
    rows = []
    perclass = []
    samples = {}

    @torch.no_grad()
    def sample(steps):
        x = noise.clone()
        for j in range(steps):
            x = x - model(x, (1 - j / steps) * 1000, conditions=targets).sample / steps
        return x

    def assess(x, nfe):
        metrics = conditional_routing_metrics(x, targets, noise, n_classes=16)
        parts = []
        for c in range(16):
            a = x[targets == c]
            b = real.samples[real.conditions == c]
            q = rays_quality(a, b, refs[c], bank, block_size=256)
            q["radial_w1_raw"] = float(
                wasserstein_distance(a.norm(dim=1).numpy(), b.norm(dim=1).numpy())
            )
            parts.append({"nfe": nfe, "condition": c, **q})
        perclass.extend(parts)
        return {
            "nfe": nfe,
            "accuracy_all": metrics["accuracy_all"],
            "accuracy_outer": metrics["accuracy_outer"],
            "outer_coverage": metrics["outer_coverage"],
            "requested_outer_phase_error_sector_fraction": metrics["requested_phase_error"][
                "mean_absolute_sector_fraction_outer"
            ],
            **{
                name: statistics.mean(p[name] for p in parts)
                for name in ["mmd2_real", "sliced_wasserstein", "radial_w1_raw"]
            },
        }

    for n in [1, 2, 4, 8, 16, 32, 64]:
        t0 = time.perf_counter()
        x = sample(n)
        seconds = time.perf_counter() - t0
        samples[n] = x
        row = assess(x, n)
        row["sampling_seconds_full4096_batch"] = seconds
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Independent real points with identical per-condition sample sizes provide an empirical noise floor.
    rr = torch.cat([real2.samples[real2.conditions == c] for c in range(16)])
    baseline = assess(rr, "independent_real")
    metadata = {
        "status": "completed",
        "scope": "posthoc_teacher_step_sensitivity_not_model_selection",
        "teacher_sha256": expected,
        "teacher_updates": 20000,
        "cfg_enabled": False,
        "guidance_scale": 1.0,
        "noise_seed": 830001,
        "samples_per_condition": 256,
        "real_reference_per_condition": 512,
        "conditions": list(range(16)),
        "same_noise_across_nfe_and_conditions": True,
        "noise_sha256": tensor_hash(noise),
        "real_reference_seed": 10047,
        "independent_real_seed": 10048,
        "real_reference_sha256": tensor_hash(real.samples),
        "kernel_reference": "per-condition real train only, raw xy, bandwidth1024, fixed across NFE",
        "swd": "raw xy mean projected W1;1024 fixed directions",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation_limits": [
            "Posthoc diagnostic at one frozen teacher and one noise/reference bank.",
            "Independent-real baseline is an empirical finite-sample reference, not a calibrated hypothesis test.",
            "Correct target arm is not distributional convergence; inspect radial/angular/conditional quality.",
        ],
    }
    atomic_json(
        out / "metrics.json",
        {
            "metadata": metadata,
            "rows": rows,
            "independent_real_baseline": baseline,
            "per_condition": perclass,
        },
    )
    torch.save(
        {
            "samples": samples,
            "initial_noise_per_condition": base,
            "targets": targets,
            "real": real.samples,
            "real_labels": real.conditions,
        },
        out / "samples.pt",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ["sliced_wasserstein", "radial_w1_raw", "mmd2_real"],
        ["Conditional projected W1", "Conditional radial W1", "Conditional MMD squared"],
    ):
        ax.plot(
            [r["nfe"] for r in rows], [r[key] for r in rows], marker="o", label="Frozen teacher"
        )
        ax.axhline(baseline[key], color="gray", ls="--", label="Independent real vs real")
        ax.set(xscale="log", xlabel="Euler steps / NFE", title=title)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"quality_vs_nfe.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    ref = real.samples[real.conditions == 0]
    for ax, n in zip(axes.flat, [1, 2, 4, 8, 16, 32]):
        x = samples[n][:256]
        ax.scatter(ref[:, 0], ref[:, 1], s=6, alpha=0.25, color="gray", label="Real target0")
        ax.scatter(x[:, 0], x[:, 1], s=7, alpha=0.55, color="#7645b5", label="Generated target0")
        ax.set(
            title=f"Teacher N{n} · same256 initial noises",
            xlim=(-0.35, 0.12),
            ylim=(-0.05, 1.05),
            aspect="equal",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"target0_endpoints.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert model_hash(model) == expected
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
