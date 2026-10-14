"""Embed a verified wheel in standalone notebooks and prepare downloadable files.

Run after uv build --wheel:
python scripts/build_notebook_bundle.py --wheel dist/diffusion_lab-0.2.0-py3-none-any.whl
"""

import argparse
import base64
import hashlib
import json
import subprocess
import textwrap
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "notebooks/03_standalone_quickstart.ipynb",
    "notebooks/04_change_components.ipynb",
)


def verify_wheel(wheel):
    version = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]["version"]
    if wheel.name != f"diffusion_lab-{version}-py3-none-any.whl":
        raise ValueError("Wheel filename does not match current project version")
    with ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(
            archive.read(f"diffusion_lab-{version}.dist-info/METADATA")
        )
        if metadata["Version"] != version or metadata["Name"] != "diffusion-lab":
            raise ValueError("Wheel internal metadata does not match this project")
        package_files = {
            name
            for name in archive.namelist()
            if name.startswith("diffusion_lab/") and name.endswith(".py")
        }
        source_files = {
            source.relative_to(PROJECT / "src").as_posix()
            for source in (PROJECT / "src/diffusion_lab").rglob("*.py")
        }
        if source_files != package_files:
            raise ValueError("Wheel/source file sets differ; rebuild the wheel")
        for name in source_files:
            if archive.read(name) != (PROJECT / "src" / name).read_bytes():
                raise ValueError(f"Stale wheel contents: {name}; rebuild the wheel")
    return version


def installer_source(wheel, digest):
    encoded = base64.b64encode(wheel.read_bytes()).decode()
    payload = "\n".join(f'    "{line}"' for line in textwrap.wrap(encoded, 100))
    return f'''#@title Установить Diffusion Lab и зависимости
# This cell contains the small, verified Python wheel; no repository URL is needed.
import base64
import hashlib
import importlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.version_info < (3, 11):
    raise RuntimeError("Diffusion Lab requires Python 3.11 or newer")
_diffusion_lab_wheel_data = (
{payload}
)
_diffusion_lab_wheel_bytes = base64.b64decode(_diffusion_lab_wheel_data)
if hashlib.sha256(_diffusion_lab_wheel_bytes).hexdigest() != "{digest}":
    raise RuntimeError("Embedded package checksum mismatch")
if importlib.util.find_spec("pip") is None:
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
with tempfile.TemporaryDirectory(prefix="diffusion-lab-install-") as _install_dir:
    _wheel_path = Path(_install_dir) / "{wheel.name}"
    _wheel_path.write_bytes(_diffusion_lab_wheel_bytes)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--disable-pip-version-check", str(_wheel_path) + "[interactive]"])
    # Replace only the tiny local package, even when its version is already installed.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--disable-pip-version-check", "--force-reinstall", "--no-deps",
                           str(_wheel_path)])
importlib.invalidate_caches()
print("Diffusion Lab is installed in:", sys.executable)
'''


def build(wheel):
    wheel = wheel.resolve()
    version = verify_wheel(wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    source = installer_source(wheel, digest)
    for relative in NOTEBOOKS:
        path = PROJECT / relative
        notebook = json.loads(path.read_text())
        found = 0
        for cell in notebook["cells"]:
            if "diffusion-lab-bootstrap" in cell.get("metadata", {}).get("tags", []):
                cell["source"] = source.splitlines(keepends=True)
                cell["execution_count"] = None
                cell["outputs"] = []
                cell["metadata"].update({"jupyter": {"source_hidden": True}, "cellView": "form"})
                found += 1
        if found != 1:
            raise ValueError(f"Expected one bootstrap cell in {relative}, found {found}")
        notebook["metadata"]["diffusion_lab_bundle"] = {
            "version": version,
            "wheel_filename": wheel.name,
            "wheel_sha256": digest,
        }
        notebook["metadata"]["colab"] = {"name": path.name, "provenance": []}
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    archive_path = PROJECT / "dist" / f"diffusion-lab-{version}-source.zip"
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=PROJECT,
        )
        .decode()
        .split("\0")
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            excluded = {
                ".git",
                ".venv",
                "outputs",
                "artifacts",
                "dist",
                "build",
                ".cache",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                ".aws",
                ".codex",
            }
            parts = Path(name).parts
            if any(part in excluded or part.endswith(".egg-info") for part in parts):
                continue
            if (
                name
                and (PROJECT / name).is_file()
                and not (PROJECT / name).is_symlink()
                and (PROJECT / name).resolve().is_relative_to(PROJECT)
            ):
                archive.write(PROJECT / name, f"diffusion-lab/{name}")
    manifest = {
        "version": version,
        "wheel": wheel.name,
        "wheel_sha256": digest,
        "source_archive": archive_path.name,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "notebooks": {
            relative: hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest()
            for relative in NOTEBOOKS
        },
    }
    (PROJECT / "dist/bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    build(parser.parse_args().wheel)
