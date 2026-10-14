"""Train, resume, sample, report, and sweep from the same research components."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler

from .environments import make_environment
from .experiment import (
    build_experiment,
    endpoint_constraints,
    load_config,
    resolve_device,
    save_run,
    switch_scheduler,
    write_json,
)
from .pipeline import ResearchDiffusionPipeline
from .reporting import export_report
from .visualization import (
    plot_dataset,
    plot_trajectories,
    save_denoising_gif,
    save_figure,
    save_trajectory_gif,
)


def train(config, output, resume=None, report=True):
    output = Path(output)
    if (output / "config.yaml").exists() and resume is None:
        raise ValueError(f"Run already exists: {output}. Choose a new output or pass --resume.")
    dataset, pipeline, trainer = build_experiment(config)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        trainer.load_checkpoint(resume)
    print(
        f"Environment={dataset.metadata['name']} shape={dataset.sample_shape} samples={len(dataset)} device={trainer.device}"
    )
    every = max(1, trainer.config.epochs // 10)

    def log(metrics):
        if metrics["epoch"] % every == 0 or metrics["epoch"] == trainer.config.epochs:
            print(
                f"epoch={metrics['epoch']:4d} loss={metrics['loss']:.5f} regression={metrics['regression']:.5f}",
                flush=True,
            )

    trainer.fit(dataset, callback=log, checkpoint_dir=output)
    save_run(trainer, dataset, config, output)
    metrics = {}
    if report:
        sampling = config.get("sampling", {}).copy()
        kind = sampling.pop(
            "scheduler",
            "flow" if isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler) else "ddim",
        )
        with trainer.ema_weights():
            original_scheduler = pipeline.scheduler
            try:
                switch_scheduler(pipeline, kind)
                metrics = export_report(
                    pipeline,
                    dataset,
                    output / "report",
                    history=trainer.history,
                    seed=config.get("seed", 42),
                    **sampling,
                )
            finally:
                pipeline.scheduler = original_scheduler
        print(json.dumps(metrics, indent=2))
    print(f"Saved run: {output.resolve()}")
    return metrics


def sample(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchDiffusionPipeline.from_pretrained(
        args.pipeline, local_files_only=True, low_cpu_mem_usage=False
    ).to(resolve_device(args.device))
    torch.set_num_threads(4)
    if args.scheduler:
        switch_scheduler(pipeline, args.scheduler)
    metadata_path = Path(args.pipeline) / "research_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    conditions = None
    if args.conditions_file:
        conditions = torch.as_tensor(
            np.load(args.conditions_file, allow_pickle=False), device=pipeline.model.device
        )
    elif args.condition is not None:
        conditions = pipeline._conditions(
            json.loads(args.condition), args.num_samples, pipeline.model.device
        )
    elif metadata and len(pipeline.model.sample_shape) == 2:
        environment = metadata["environment_config"].copy()
        environment["n_samples"] = args.num_samples
        environment["seed"] = args.seed
        conditions = make_environment(**environment).conditions.to(pipeline.model.device)
    if conditions is not None:
        conditions = pipeline._conditions(conditions, args.num_samples, pipeline.model.device)
    constraints = (
        endpoint_constraints(pipeline.model, conditions) if not args.free_endpoints else {}
    )
    result = pipeline(
        batch_size=args.num_samples,
        conditions=conditions,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        eta=args.eta,
        generator=torch.Generator().manual_seed(args.seed),
        record_history=args.gif,
        **constraints,
    )
    write_json(
        output / "sampling_config.json",
        {
            "pipeline": str(Path(args.pipeline).resolve()),
            "seed": args.seed,
            "num_samples": args.num_samples,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "eta": args.eta,
            "scheduler_class": type(pipeline.scheduler).__name__,
            "scheduler_config": dict(pipeline.scheduler.config),
            "pinned_endpoints": bool(constraints),
            "device": str(pipeline.model.device),
        },
    )
    np.save(output / "samples.npy", result.samples.cpu().numpy())
    if conditions is not None:
        np.save(output / "conditions.npy", conditions.cpu().numpy())
    walls = metadata.get("environment", {}).get("walls", [])
    if len(pipeline.model.sample_shape) == 2:
        fig, _ = plot_trajectories(result.samples.cpu().numpy()[:64], walls=walls)
    else:
        fig, _ = plot_dataset(
            result.samples.cpu().numpy(),
            labels=conditions.cpu().numpy()
            if conditions is not None and conditions.ndim == 1
            else None,
        )
    save_figure(fig, output / "samples")
    if args.gif:
        save_denoising_gif(result.history[:, :64], output / "denoising.gif", walls=walls)
        if len(pipeline.model.sample_shape) == 2:
            save_trajectory_gif(
                result.samples[:12],
                output / "generated_trajectories.gif",
                walls=walls,
                title="Generated navigation paths · A → B",
            )
        else:
            save_trajectory_gif(
                result.history[:, :24].permute(1, 0, 2),
                output / "denoising_trajectories.gif",
                title="Denoising trajectories · noise → sample",
                start_label="Noise x_T",
                end_label="Sample x_0",
            )
    print(f"Saved samples: {output.resolve()}")


def report_run(args):
    run = Path(args.run)
    config = load_config(run / "config.yaml")
    environment = config.get("environment", {"name": "rotating_rays"}).copy()
    environment.setdefault("seed", config.get("seed", 42))
    dataset = make_environment(**environment)
    pipeline = ResearchDiffusionPipeline.from_pretrained(
        run / "pipeline", local_files_only=True, low_cpu_mem_usage=False
    ).to(resolve_device(args.device))
    torch.set_num_threads(config.get("num_threads", 4))
    switch_scheduler(
        pipeline,
        args.scheduler
        or ("flow" if isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler) else "ddim"),
    )
    history = json.loads((run / "history.json").read_text())
    metrics = export_report(
        pipeline,
        dataset,
        args.output or run / "report",
        history=history,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        make_gif=args.gif,
    )
    print(json.dumps(metrics, indent=2))


def sweep(args):
    config = load_config(args.config)
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    keys = args.parameter.split(".")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, value in enumerate(args.values):
        experiment = copy.deepcopy(config)
        node = experiment
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        parsed = yaml.safe_load(value)
        node[keys[-1]] = parsed
        metrics = train(experiment, output / f"run_{index:02d}")
        rows.append({"parameter": args.parameter, "value": value, **metrics})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparison.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train or resume a config-driven experiment")
    training.add_argument("--config", default="configs/rays.yaml")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int)
    training.add_argument("--resume")
    training.add_argument("--no-report", action="store_true")
    sampling = sub.add_parser("sample", help="Sample a saved Diffusers pipeline")
    sampling.add_argument("pipeline")
    sampling.add_argument("--output", required=True)
    condition_inputs = sampling.add_mutually_exclusive_group()
    condition_inputs.add_argument(
        "--condition", help="JSON class ID, class batch, or continuous vector"
    )
    condition_inputs.add_argument("--conditions-file", help="NumPy [B] or [B,K] array")
    sampling.add_argument("--free-endpoints", action="store_true")
    sampling.add_argument("--eta", type=float, default=0.0)
    reporting = sub.add_parser("report", help="Regenerate figures and metrics from a saved run")
    reporting.add_argument("run")
    reporting.add_argument("--output")
    for command in (sampling, reporting):
        command.add_argument("--num-samples", type=int, default=512)
        command.add_argument("--steps", type=int, default=100)
        command.add_argument("--guidance", type=float, default=1.5)
        command.add_argument("--scheduler", choices=("ddpm", "ddim", "flow"))
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--gif", action="store_true")
    sweeping = sub.add_parser("sweep", help="Compare one changed parameter with matching seeds")
    sweeping.add_argument("--config", default="configs/rays.yaml")
    sweeping.add_argument("--parameter", required=True)
    sweeping.add_argument("--values", nargs="+", required=True)
    sweeping.add_argument("--output", required=True)
    sweeping.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        if args.epochs is not None:
            config.setdefault("training", {})["epochs"] = args.epochs
        train(config, args.output, args.resume, not args.no_report)
    elif args.command == "sample":
        sample(args)
    elif args.command == "report":
        report_run(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
