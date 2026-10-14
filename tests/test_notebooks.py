import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "03_standalone_quickstart.ipynb",
        "04_change_components.ipynb",
        "05_live_visualizations.ipynb",
        "06_trained_interactive_demo.ipynb",
    ],
)
def test_downloadable_notebooks_compile_without_binary_payloads(name):
    path = ROOT / "notebooks" / name
    notebook = json.loads(path.read_text())
    assert path.stat().st_size < 100_000
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell["outputs"]
        assert cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]), filename=f"{name}:cell{index}")
        compile(tree, f"{name}:cell{index}", "exec")
        assert not any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 20_000
            for node in ast.walk(tree)
        )
