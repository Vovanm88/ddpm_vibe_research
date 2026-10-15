"""Evaluate both low-NFE ray cohorts only after all 24 students are frozen."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy.stats import wasserstein_distance

from diffusion_lab import PointDenoiser
from diffusion_lab.conditional_metrics import conditional_routing_metrics
from diffusion_lab.environments import make_environment
from recheck.evaluate import (
    fit_feature_reference,
    fixed_projection_bank,
    paired_statistics,
    rays_quality,
)
from recheck.reporting import save_comparison_gif
from recheck.runtime import atomic_json, model_hash, tensor_hash


def load_native(path, expected):
    model = (
        PointDenoiser.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=False)
        .eval()
        .requires_grad_(False)
    )
    if model_hash(model) != expected:
        raise ValueError(f"Native weight hash mismatch: {path}")
    return model


def preflight(root, parent_hash):
    records = []
    for nfe in (4, 2):
        status = json.loads((root / f"nfe{nfe}/suite_status.json").read_text())
        if status["status"] != "completed" or len(status["runs"]) != 3:
            raise ValueError(
                "Both complete three-seed cohorts are required before opening final reference"
            )
        assert all(
            r["common_rng_traces_equal"] and r["teacher_sha256_after"] == parent_hash
            for r in status["runs"]
        )
        for seed in (101, 202, 303):
            for method in ("M0", "M1", "E0", "E1"):
                folder = root / f"nfe{nfe}/seed{seed}/{method}"
                m = json.loads((folder / "manifest.json").read_text())
                assert m["status"] == "completed" and m["updates_completed"] == 1000
                assert (
                    m["teacher_sha256"] == parent_hash
                    and m["student_steps"] == nfe
                    and m["teacher_steps"] == 32
                )
                assert m["seed"] == seed and m["whole_support_single_condition"]
                model = load_native(folder / "ema_model", m["ema_sha256"])
                del model
                records.append(
                    {
                        "nfe": nfe,
                        "seed": seed,
                        "method": method,
                        "folder": str(folder),
                        "model_sha256": m["ema_sha256"],
                    }
                )
    assert len(records) == 24
    return records


@torch.no_grad()
def sample(model, noise, targets, steps):
    x = noise.clone()
    preview = [x[:16].clone()]
    for j in range(steps):
        x = x - model(x, (1 - j / steps) * 1000, conditions=targets).sample / steps
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite low-NFE sample")
        preview.append(x[:16].clone())
    return x, torch.stack(preview)


def phase(x, condition):
    r = x.double().norm(dim=1)
    a = torch.atan2(x[:, 1].double(), x[:, 0].double())
    a = (
        a
        - math.pi / 2
        - 1.6 * math.pi * (r / (10 / 11)).square() / (0.9 * 16)
        - condition * 2 * math.pi / 16
    )
    a = torch.remainder(a + math.pi, 2 * math.pi) - math.pi
    return (a.abs() / (2 * math.pi / 16))[r > 0.25 * (10 / 11)].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="replication/artifacts/rays16_conditional_low_nfe_v1")
    parser.add_argument("--output", default="replication/reports/low_nfe_conditional_rays")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, out = Path(args.runs), Path(args.output)
    parent = Path("replication/demo/rays16_conditional_v1/teacher")
    parent_hash = json.loads((parent / "manifest.json").read_text())["teacher_sha256"]
    records = preflight(root, parent_hash)
    # The new final bank is opened only AFTER all 24 native exports pass preflight.
    train = make_environment("rotating_rays", n_samples=65536, n_classes=16, seed=47)
    real = make_environment("rotating_rays", n_samples=8192, n_classes=16, seed=20047)
    base = torch.randn((256, 2), generator=torch.Generator().manual_seed(940001))
    noise, targets = base.repeat(16, 1), torch.arange(16).repeat_interleave(256)
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
    rows, per_condition, endpoints, previews = [], [], {}, {}
    parent32 = None

    def assess(key, model, nfe, method, seed):
        nonlocal parent32
        x, history = sample(model, noise, targets, nfe)
        if method == "Teacher" and nfe == 32:
            parent32 = x.clone()
        routing = conditional_routing_metrics(x, targets, noise, n_classes=16)
        parts = []
        for c in range(16):
            a, b = x[targets == c], real.samples[real.conditions == c]
            q = rays_quality(a, b, refs[c], bank, block_size=256)
            q["radial_w1_raw"] = float(
                wasserstein_distance(a.norm(dim=1).numpy(), b.norm(dim=1).numpy())
            )
            pa, pb = phase(a, c), phase(b, c)
            q["requested_phase_w1_sector"] = (
                float(wasserstein_distance(pa, pb)) if len(pa) and len(pb) else None
            )
            parts.append({"key": key, "condition": c, **q})
        per_condition.extend(parts)
        row = {
            "key": key,
            "method": method,
            "nfe": nfe,
            "student_seed": seed,
            "model_sha256": model_hash(model),
            "accuracy_all": routing["accuracy_all"],
            "accuracy_outer": routing["accuracy_outer"],
            "outer_coverage": routing["outer_coverage"],
            **{
                k: statistics.mean(p[k] for p in parts)
                for k in (
                    "mmd2_real",
                    "sliced_wasserstein",
                    "radial_w1_raw",
                    "requested_phase_w1_sector",
                )
            },
        }
        rows.append(row)
        endpoints[key], previews[key] = x, history
        print(json.dumps(row), flush=True)

    teacher = load_native(parent / "ema_model", parent_hash)
    for nfe in (32, 1, 2, 4, 8, 16, 64):
        assess(f"Teacher_N{nfe}", teacher, nfe, "Teacher", None)
    for record in records:
        model = load_native(Path(record["folder"]) / "ema_model", record["model_sha256"])
        assess(
            f"N{record['nfe']}_{record['method']}_seed{record['seed']}",
            model,
            record["nfe"],
            record["method"],
            record["seed"],
        )
    for row in rows:
        row["paired_mse_to_teacher32"] = float(
            (endpoints[row["key"]].double() - parent32.double()).square().mean()
        )
    contrasts = []
    for nfe in (4, 2):
        for first, second in [("E0", "M0"), ("E1", "M1"), ("M1", "M0"), ("E1", "E0")]:
            for metric in [
                "mmd2_real",
                "sliced_wasserstein",
                "radial_w1_raw",
                "requested_phase_w1_sector",
                "paired_mse_to_teacher32",
            ]:
                a = [
                    next(
                        r
                        for r in rows
                        if r["method"] == first and r["student_seed"] == s and r["nfe"] == nfe
                    )[metric]
                    for s in (101, 202, 303)
                ]
                b = [
                    next(
                        r
                        for r in rows
                        if r["method"] == second and r["student_seed"] == s and r["nfe"] == nfe
                    )[metric]
                    for s in (101, 202, 303)
                ]
                contrasts.append(
                    {
                        "nfe": nfe,
                        "contrast": f"{first}-{second}",
                        "metric": metric,
                        "statistics": paired_statistics(a, b),
                    }
                )
    metadata = {
        "status": "completed",
        "teacher_sha256": parent_hash,
        "models_fixed_before_reference": records,
        "noise_seed": 940001,
        "reference_seed": 20047,
        "samples_per_condition": 256,
        "real_per_condition": 512,
        "initial_noise_sha256": tensor_hash(noise),
        "reference_sha256": tensor_hash(real.samples),
        "teacher_updates": 20000,
        "student_updates": 1000,
        "student_seeds": [101, 202, 303],
        "same_noise_all_methods_nfe_conditions": True,
        "cfg_enabled": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "conditional_low_nfe_pilot_one_frozen_teacher_three_paired_student_seeds",
        "cost_note": "Training walltimes include concurrent CPUload and a documented isolated benchmark pause; no walltime efficiency ranking.",
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "results.json",
        {
            "metadata": metadata,
            "rows": rows,
            "per_condition": per_condition,
            "paired_contrasts": contrasts,
        },
    )
    torch.save(
        {
            "endpoints": endpoints,
            "target0_previews": previews,
            "initial_noise_per_condition": base,
            "targets": targets,
            "reference": real.samples,
            "reference_conditions": real.conditions,
        },
        out / "samples_and_previews.pt",
    )
    for nfe in (4, 2):
        histories = {
            "Teacher N32": previews["Teacher_N32"],
            f"Original N{nfe}": previews[f"Teacher_N{nfe}"],
            **{f"{m} N{nfe}": previews[f"N{nfe}_{m}_seed101"] for m in ("M0", "M1", "E0", "E1")},
        }
        gif = save_comparison_gif(
            histories,
            out / f"comparison_nfe{nfe}.gif",
            particles=16,
            real=real.samples[real.conditions == 0],
            title=f"Conditional rays 32 to {nfe} · target0 · first registered seed101 · identical noises",
        )
        atomic_json(out / f"comparison_nfe{nfe}_manifest.json", gif)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ["sliced_wasserstein", "requested_phase_w1_sector"]):
        rr = sorted([r for r in rows if r["method"] == "Teacher"], key=lambda r: r["nfe"])
        ax.plot(
            [r["nfe"] for r in rr],
            [r[metric] for r in rr],
            color="black",
            marker="o",
            label="Frozen teacher",
        )
        for m in ("M0", "M1", "E0", "E1"):
            for n in (2, 4):
                values = [r[metric] for r in rows if r["method"] == m and r["nfe"] == n]
                ax.scatter([n] * len(values), values, label=m if n == 2 else None, alpha=0.75)
        ax.set(xlabel="Euler steps / NFE", ylabel=metric, xscale="log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
        ax.grid(alpha=0.2)
    axes[0].legend()
    for extension in ("png", "svg"):
        fig.savefig(out / f"quality_vs_nfe.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
