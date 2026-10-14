"""Train rotating rays with an explicit PyTorch loop and export a visual report."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader

from diffusion_lab import PointDenoiser, ResearchDiffusionPipeline
from diffusion_lab.environments import make_environment
from diffusion_lab.metrics import evaluate_samples
from diffusion_lab.reporting import export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/visual_demo_rays"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if (args.output / "history.json").exists():
        raise ValueError("Demo run already exists; choose another --output")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshots").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dataset = make_environment("rotating_rays", n_samples=4096, n_classes=4, seed=args.seed)
    reference = make_environment("rotating_rays", n_samples=4096, n_classes=4, seed=args.seed + 1)
    np.savez_compressed(
        args.output / "reference.npz",
        samples=reference.samples.numpy(),
        conditions=reference.conditions.numpy(),
    )
    loader_generator = torch.Generator().manual_seed(args.seed + 2)
    noise_generator = torch.Generator().manual_seed(args.seed + 3)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=loader_generator)
    model = PointDenoiser(
        sample_shape=dataset.sample_shape,
        hidden_dim=128,
        num_layers=4,
        conditioner="embedding",
        num_classes=4,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
        timestep_spacing="trailing",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    history, snapshots = [], []
    snapshot_epochs = {0, 1, 5, 20, 60, args.epochs}
    conditions = reference.conditions[:1024]

    def snapshot(epoch):
        sampling_scheduler = DDIMScheduler.from_config(scheduler.config)
        pipeline = ResearchDiffusionPipeline(ema, sampling_scheduler)
        output = pipeline(
            batch_size=1024,
            conditions=conditions,
            num_inference_steps=100,
            guidance_scale=1.5,
            generator=torch.Generator().manual_seed(args.seed + 4),
        )
        np.savez_compressed(
            args.output / "snapshots" / f"epoch_{epoch:04d}.npz",
            samples=output.samples.cpu().numpy(),
            conditions=conditions.numpy(),
        )
        metrics = evaluate_samples(reference, output.samples, conditions, seed=args.seed)
        snapshots.append({"epoch": epoch, **metrics})
        (args.output / "snapshot_metrics.json").write_text(json.dumps(snapshots, indent=2))
        print(f"snapshot epoch={epoch} heldout SWD={metrics['sliced_wasserstein']:.5f}", flush=True)

    import yaml

    config = {
        "seed": args.seed,
        "device": "cpu",
        "num_threads": 4,
        "environment": {
            "name": "rotating_rays",
            "n_samples": 4096,
            "n_classes": 4,
            "seed": args.seed,
        },
        "model": {
            "sample_shape": [2],
            "hidden_dim": 128,
            "num_layers": 4,
            "conditioner": "embedding",
            "num_classes": 4,
        },
        "scheduler": {
            "kind": "ddpm",
            "num_train_timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "v_prediction",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "condition_dropout": 0.15,
            "ema_decay": 0.99,
            "loss": "mse",
        },
        "sampling": {"scheduler": "ddim", "steps": 100, "num_samples": 1024, "guidance_scale": 1.5},
    }
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    snapshot(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            clean = batch["samples"].to(device)
            context = batch["conditions"].to(device)
            noise = torch.randn(clean.shape, generator=noise_generator, device=device)
            timesteps = torch.randint(1000, (len(clean),), generator=noise_generator, device=device)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            target = scheduler.get_velocity(clean, noise, timesteps)
            keep = torch.rand(len(clean), generator=noise_generator, device=device) >= 0.15
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, timesteps, conditions=context, condition_mask=keep).sample
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
                    averaged.lerp_(current, 0.01)
            total += loss.item() * len(clean)
            count += len(clean)
        row = {
            "epoch": epoch,
            "step": epoch * len(loader),
            "loss": total / count,
            "regression": total / count,
        }
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:3d}/{args.epochs} loss={row['loss']:.5f}", flush=True)
        if epoch in snapshot_epochs:
            snapshot(epoch)
    pipeline = ResearchDiffusionPipeline(ema, DDIMScheduler.from_config(scheduler.config))
    pipeline.save_pretrained(args.output / "pipeline")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": args.epochs,
            "noise_rng": noise_generator.get_state(),
            "loader_rng": loader_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        args.output / "checkpoint.pt",
    )
    training_seconds = time.perf_counter() - started
    report_metrics = export_report(
        pipeline,
        reference,
        args.output / "report",
        num_samples=1024,
        steps=100,
        guidance_scale=1.5,
        seed=args.seed + 4,
        history=history,
        make_gif=True,
    )
    try:
        checkout = Path(__file__).resolve().parent.parent
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                text=True,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = None, None
    manifest = {
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_seconds": training_seconds,
        "optimizer_steps": args.epochs * len(loader),
        "config": config,
        "training_data_seed": args.seed,
        "heldout_data_seed": args.seed + 1,
        "heldout_metrics": report_metrics,
        "snapshot_metrics": snapshots,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "numpy", "matplotlib")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FINAL", json.dumps(report_metrics), flush=True)
    print("Training seconds:", round(training_seconds, 2), flush=True)


if __name__ == "__main__":
    main()
