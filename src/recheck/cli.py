"""Command-line entry points for the explicitly staged replication protocol.

Examples (from the repository root)::

    python -m recheck.cli doctor
    python -m recheck.cli pilot-rays --config replication/configs/rays_cpu.yaml
    python -m recheck.cli prepare-mnist --dry-run

Loading this module does not import training/data/download implementations.
"""

import argparse
import contextlib
import importlib
import json
import sys
from pathlib import Path

from yaml import YAMLError

from .config import load_config

DEFAULT_CONFIGS = {
    "pilot-rays": "replication/configs/rays_cpu.yaml",
    "pilot-toy": "replication/configs/toy.yaml",
}
MNIST_CONFIG = "replication/configs/mnist_mvp.yaml"
CONFIG_STAGES = {
    "pilot-rays": "pilot",
    "pilot-toy": "pilot",
    "prepare-mnist": "prepare_mnist_stage",
    "train-evaluator": "train_evaluator_stage",
    "train-teacher": "train_teacher_stage",
    "teacher-gate": "teacher_gate",
    "calibrate": "calibrate_stage",
    "benchmark": "benchmark_stage",
    "distill-suite": "distill_suite",
    "evaluate-suite": "evaluate_suite",
}
RESUMABLE = {"pilot-rays", "pilot-toy", "train-evaluator", "train-teacher", "distill-suite"}


def _workflow():
    return importlib.import_module("recheck.workflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recheck.cli",
        description="Controlled FM/OT/tube replication; verify rays and animations before MNIST.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect local hardware and package versions")
    doctor.add_argument("--output", help="Optionally save the hardware report as JSON")
    doctor.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing files"
    )
    verify = commands.add_parser("verify-losses", help="Run numeric loss/schedule contract checks")
    verify.add_argument("--output", default="replication/reports/tests_report.json")
    verify.add_argument(
        "--dry-run", action="store_true", help="Print the plan without executing checks"
    )
    descriptions = {
        "pilot-rays": "Train/evaluate the rays pilot and render trajectory animations",
        "pilot-toy": "Run the independent Gaussian-mixture diagnostic pilot",
        "prepare-mnist": "Prepare deterministic official MNIST splits after the rays gate",
        "train-evaluator": "Fit the independent real-data MNIST evaluator",
        "train-teacher": "Fit the shared unconditional FM teacher",
        "teacher-gate": "Check frozen teacher quality and Euler NFE sensitivity",
        "calibrate": "Fix train-only teacher feature scales before student training",
        "benchmark": "Measure warmed training cost before registering the full budget",
        "distill-suite": "Train the registered paired student methods",
        "evaluate-suite": "Evaluate every registered method on common initial noise",
    }
    for name in CONFIG_STAGES:
        stage = commands.add_parser(name, help=descriptions[name])
        default = DEFAULT_CONFIGS.get(name, MNIST_CONFIG)
        stage.add_argument("--config", help=f"Strict YAML configuration (default: {default})")
        stage.add_argument(
            "--dry-run", action="store_true", help="Inspect the plan without downloads or training"
        )
        if name in RESUMABLE:
            stage.add_argument(
                "--resume", action="store_true", help="Explicitly restore the matching saved run"
            )
    report = commands.add_parser("report", help="Rebuild tables/plots/report from recorded runs")
    report.add_argument("--runs", default="replication/artifacts/")
    report.add_argument("--output", default="replication/reports/REPORT.md")
    report.add_argument(
        "--config", help="Optional strict YAML; otherwise infer config from run metadata"
    )
    report.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing reports"
    )
    return parser


def _load_stage_config(command, supplied):
    path = Path(supplied or DEFAULT_CONFIGS.get(command, MNIST_CONFIG))
    if supplied is None and not path.is_file():
        # Editable installs can also be invoked from a notebook's own directory.
        candidate = Path(__file__).resolve().parents[2] / path
        if candidate.is_file():
            path = candidate
    config = load_config(path)
    expected_data = {"pilot-rays": "rotating_rays", "pilot-toy": "gmm"}.get(command)
    if expected_data is not None and config["data"]["name"] != expected_data:
        raise ValueError(
            f"{command} requires data.name={expected_data}; use the matching pilot configuration"
        )
    return config


def _json_value(value):
    if isinstance(value, dict):
        # Library functions can return a live model to Python callers. The CLI
        # exposes its saved path/manifest, never parameters or a Python repr.
        return {
            str(key): _json_value(item)
            for key, item in value.items()
            if not (key == "model" and callable(getattr(item, "state_dict", None)))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dispatch(args):
    if args.command == "doctor":
        if args.dry_run:
            return {
                "command": "doctor",
                "status": "dry_run",
                "output": args.output,
                "planned_action": "Inspect local hardware and installed package versions",
            }
        result = _json_value(_workflow().doctor())
        if args.output:
            from .runtime import atomic_json

            atomic_json(args.output, result)
        return result
    if args.command == "verify-losses":
        if args.dry_run:
            return {
                "command": "verify-losses",
                "status": "dry_run",
                "output": args.output,
                "planned_action": "Execute float64 numerical contract checks",
            }
        return _workflow().verify_losses(output=args.output)
    if args.command == "report":
        config = load_config(args.config) if args.config else None
        output = Path(args.output)
        if args.dry_run:
            return {
                "command": "report",
                "status": "dry_run",
                "runs": args.runs,
                "output": str(output),
                "configuration_source": args.config or "recorded_run_metadata",
            }
        result = _workflow().report_stage(args.runs, str(output.parent), config)
        standard = output.parent / "REPORT.md"
        if output != standard and standard.is_file():
            standard.replace(output)
            if isinstance(result, dict):
                result = dict(result, report=str(output))
        return result
    config = _load_stage_config(args.command, args.config)
    if not args.dry_run:
        import torch

        torch.set_num_threads(config["threads"])
    options = {"dry_run": args.dry_run}
    if args.command in RESUMABLE:
        options["resume"] = args.resume
    return getattr(_workflow(), CONFIG_STAGES[args.command])(config, **options)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Keep progress visible on stderr and reserve stdout for one finite JSON
        # document, so commands can be logged or consumed programmatically.
        with contextlib.redirect_stdout(sys.stderr):
            result = _dispatch(args)
        document = json.dumps(_json_value(result), indent=2, allow_nan=False)
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "interrupted",
                    "error": "Interrupted; resume the last fully committed checkpoint explicitly",
                }
            ),
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2 if isinstance(error, (ValueError, OSError, RuntimeError, YAMLError)) else 1
    print(document)
    if (
        args.command == "verify-losses"
        and isinstance(result, dict)
        and result.get("passed") is False
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
