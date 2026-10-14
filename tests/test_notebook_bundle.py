"""Portable notebook artifacts must carry exactly the current package sources.

Rebuild after package or version edits:
``uv build --wheel`` then ``python scripts/build_notebook_bundle.py --wheel dist/<wheel>``.
Checks read committed notebook cells directly; ignored distribution files are unnecessary.
"""

import ast
import base64
import hashlib
import io
import json
import tomllib
from email.parser import Parser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest

PROJECT = Path(__file__).resolve().parents[1]
STANDALONE_NOTEBOOKS = (
    "03_standalone_quickstart.ipynb",
    "04_change_components.ipynb",
)


def _source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _read_notebook(name):
    return json.loads((PROJECT / "notebooks" / name).read_text())


def _embedded_wheel(notebook):
    cells = [
        cell
        for cell in notebook["cells"]
        if "diffusion-lab-bootstrap" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1, "Each standalone notebook needs exactly one bootstrap cell"
    tree = ast.parse(_source(cells[0]))
    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_diffusion_lab_wheel_data"
            for target in statement.targets
        )
    ]
    assert len(assignments) == 1, "Rebuild the standalone notebook bundle"
    encoded = ast.literal_eval(assignments[0].value)
    assert isinstance(encoded, str)
    # Decode data only. The bootstrap and its installation commands are never executed.
    wheel = base64.b64decode(encoded, validate=True)
    return wheel, tree


@pytest.mark.parametrize("name", STANDALONE_NOTEBOOKS)
def test_standalone_wheel_matches_current_version_and_every_package_source(name):
    notebook = _read_notebook(name)
    wheel_bytes, bootstrap = _embedded_wheel(notebook)
    bundle = notebook["metadata"].get("diffusion_lab_bundle")
    assert bundle, "Run scripts/build_notebook_bundle.py after rebuilding the wheel"
    version = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]["version"]
    assert bundle["version"] == version
    assert bundle["wheel_filename"] == f"diffusion_lab-{version}-py3-none-any.whl"
    digest = hashlib.sha256(wheel_bytes).hexdigest()
    assert bundle["wheel_sha256"] == digest
    checksum_comparisons = [
        expression
        for expression in ast.walk(bootstrap)
        if isinstance(expression, ast.Compare)
        and isinstance(expression.left, ast.Call)
        and isinstance(expression.left.func, ast.Attribute)
        and expression.left.func.attr == "hexdigest"
    ]
    assert any(
        len(comparison.comparators) == 1
        and isinstance(comparison.ops[0], ast.NotEq)
        and ast.literal_eval(comparison.comparators[0]) == digest
        for comparison in checksum_comparisons
    ), "Bootstrap must check the embedded wheel checksum before installation"
    with ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        names = wheel.namelist()
        assert len(names) == len(set(names)), "Duplicate wheel entries are ambiguous"
        dist_info = f"diffusion_lab-{version}.dist-info/"
        for entry in names:
            path = PurePosixPath(entry)
            assert not path.is_absolute() and ".." not in path.parts
            assert entry.startswith(("diffusion_lab/", dist_info))
        metadata = Parser().parsestr(wheel.read(dist_info + "METADATA").decode())
        assert metadata["Name"] == "diffusion-lab"
        assert metadata["Version"] == version
        package_sources = {
            entry for entry in names if entry.startswith("diffusion_lab/") and entry.endswith(".py")
        }
        actual_sources = {
            source.relative_to(PROJECT / "src").as_posix(): source.read_bytes()
            for source in (PROJECT / "src/diffusion_lab").rglob("*.py")
        }
        assert package_sources == set(actual_sources), "Rebuild the wheel after source file changes"
        for entry, current_bytes in actual_sources.items():
            assert wheel.read(entry) == current_bytes, f"Stale bundled package: {entry}"


@pytest.mark.parametrize("name", STANDALONE_NOTEBOOKS)
def test_new_notebooks_are_clean_and_every_code_cell_compiles(name):
    notebook = _read_notebook(name)
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None, f"{name} cell {index} retains execution state"
        assert cell.get("outputs") == [], f"{name} cell {index} retains outputs"
        compile(_source(cell), f"{name}:cell:{index}", "exec")


def test_standalone_notebooks_embed_the_same_distribution():
    wheels = [_embedded_wheel(_read_notebook(name))[0] for name in STANDALONE_NOTEBOOKS]
    assert wheels[0] == wheels[1]
