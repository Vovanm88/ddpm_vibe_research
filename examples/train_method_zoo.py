"""Run the visible notebook15 algorithms without duplicating their training code.

Importing this module defines wrappers only. Notebook imports and function
definitions are loaded lazily; no data, native parent or optimizer is constructed
until an explicit call. The actual objective and optimizer loops remain in the
notebook's tagged ``zoo-api`` cells.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/15_conditional_method_zoo.ipynb"
_API = None


def notebook_api():
    global _API
    if _API is None:
        namespace = {"__name__": "method_zoo_visible_notebook_api"}
        for index, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
            if cell["cell_type"] == "code" and "zoo-api" in cell.get("metadata", {}).get(
                "tags", []
            ):
                source = "".join(cell["source"])
                exec(compile(source, f"notebook15:cell{index}", "exec"), namespace)
        _API = namespace
    return _API


def resolved_config(config):
    return notebook_api()["resolved_config"](config)


def config_sha256(config):
    api = notebook_api()
    return api["json_digest"](api["resolved_config"](config))


def load_zoo_model(export_dir, *, device="cpu"):
    return notebook_api()["load_zoo_model"](export_dir, device=device)


def sample_zoo(
    model,
    initial_noise,
    conditions,
    *,
    nfe,
    batch_size=128,
    record_history=True,
    extra_noise=None,
    generator=None,
):
    return notebook_api()["sample_zoo"](
        model,
        initial_noise,
        conditions,
        nfe=nfe,
        batch_size=batch_size,
        record_history=record_history,
        extra_noise=extra_noise,
        generator=generator,
    )


def run_zoo(config, **kwargs):
    return notebook_api()["run_zoo"](config, repo_root=ROOT, **kwargs)


def read_config(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        import yaml

        value = yaml.safe_load(path.read_text())
    return resolved_config(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Strict JSON or YAML config")
    parser.add_argument("--task", action="append", choices=["rays", "wall"])
    parser.add_argument(
        "--method",
        action="append",
        choices=[
            "cfm",
            "ot_cfm",
            "endpoint",
            "progressive",
            "reflow",
            "consistency",
            "shortcut",
            "meanflow",
        ],
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze train-only teacher pairbanks; no student updates",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_zoo(
            read_config(args.config),
            tasks=args.task,
            methods=args.method,
            seeds=args.seed,
            resume=args.resume,
            stop_after=args.stop_after,
            prepare_only=args.prepare_only,
            dry_run=args.dry_run,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"method-zoo: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
