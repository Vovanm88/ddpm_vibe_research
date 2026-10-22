"""Notebook16 reads synthetic caches only; it never trains or samples a model."""

import ast
import importlib.util
import json
from pathlib import Path

import matplotlib
import nbformat
import torch

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "notebooks/16_method_zoo_results.ipynb"


def results_api():
    namespace = {"Path": Path, "importlib": importlib}
    book = nbformat.read(BOOK, 4)
    for cell in book.cells:
        if cell.cell_type == "code" and "results_api" in cell.metadata.get("tags", []):
            exec(compile(cell.source, "notebook16:results_api", "exec"), namespace)
    return namespace


def synthetic_preview(task):
    conditions = torch.zeros(2, dtype=torch.long)
    if task == "rays":
        first = torch.tensor([[0.5, -0.3], [-0.4, -0.2]])
        last = torch.tensor([[0.0, 0.5], [0.02, 0.6]])
    else:
        condition = torch.tensor([-0.8, 0.0, 0.8, 0.0, -0.1, -0.4, 0.1, 0.4])
        conditions = condition.expand(2, -1).clone()
        last = (
            torch.tensor(
                [[-0.8, 0.0], [-0.5, 0.3], [-0.25, 0.55], [0.25, 0.55], [0.5, 0.3], [0.8, 0.0]]
            )
            .expand(2, -1, -1)
            .clone()
        )
        first = last.clone()
        first[:, 1:-1] = 0
    runs = []
    for method in ("cfm", "consistency", "Original"):
        for nfe in (1, 2, 4):
            if method == "consistency":
                times = torch.tensor([v for index in range(nfe) for v in (1 - index / nfe, 0.0)])
                calls = [v for index in range(nfe) for v in (index, index + 1)]
            else:
                times = torch.linspace(1, 0, nfe + 1)
                calls = list(range(nfe + 1))
            states = torch.stack([time * first + (1 - time) * last for time in times])
            runs.append(
                {
                    "method": method,
                    "seed": None if method == "Original" else 161,
                    "league": "parent_baseline" if method == "Original" else "primary",
                    "nfe": nfe,
                    "events": {
                        "states": states,
                        "times": times,
                        "kinds": ["synthetic_cached_event"] * len(times),
                        "network_calls_used": calls,
                    },
                }
            )
    return {
        "task": task,
        "conditions": conditions,
        "condition_ids": torch.zeros(2, dtype=torch.long),
        "reference_samples": last,
        "reference_conditions": conditions.clone(),
        "runs": runs,
    }


def test_notebook_is_clean_compiles_and_has_no_training_or_weight_payload():
    book = nbformat.read(BOOK, 4)
    nbformat.validate(book)
    assert BOOK.stat().st_size < 40000
    assert book.metadata.kernelspec.name == "python3"
    sources = []
    forbidden = {
        "run_zoo",
        "run_evaluation",
        "sample_zoo",
        "from_pretrained",
        "save_pretrained",
        "train_method",
        "make_environment",
        "make_multimodal_navigation",
        "backward",
    }
    for index, cell in enumerate(book.cells):
        assert not cell.get("attachments")
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and cell.outputs == []
        compile(cell.source, f"notebook16:{index}", "exec")
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None)
                )
                assert name not in forbidden
        sources.append(cell.source)
    source = "\n".join(sources)
    assert "diffusion-lab[replication,interactive]" in source
    assert "research/wasserstein-tube-replication" in source
    assert "sampling_preview_widget" in source
    assert "base64" not in source and ".whl" not in source


def test_missing_report_does_not_open_preview_or_factory(tmp_path):
    api = results_api()

    class ClosedCore:
        def sampling_preview_widget(self, path):
            raise AssertionError("A missing final report must not open a cache")

    browser = api["create_results_browser"](tmp_path, core_module=ClosedCore())
    selector = browser.children[1]
    assert selector.value == "rays"
    selector.value = "wall"
    assert selector.value == "wall"
    browser.close()


def test_table_formats_five_significant_digits_keeps_integer_budgets_and_escapes():
    api = results_api()
    rows = [
        {
            "method": "endpoint",
            "nfe": "4",
            "primary_point_rank": "12",
            "score": "0.123456789123",
            "tiny": "0.000000123456789",
            "unsafe": "<script>",
        }
    ]
    original = json.loads(json.dumps(rows))
    table = api["_table"](rows, tuple(rows[0])).data
    assert "0.12346" in table and "0.123456789123" not in table
    assert "1.2346e-07" in table
    assert "<td>4</td>" in table and "<td>12</td>" in table
    assert "endpoint (off-target)" in table
    assert "&lt;script&gt;" in table and "<script>" not in table
    assert rows == original  # display formatting must never alter source rows.
    assert api["_method_label"]("endpoint", "1") == "endpoint"
    assert api["_method_label"]("meanflow", "4") == "meanflow"


def test_preview_caption_marks_only_endpoint_above_one_call():
    api = results_api()
    widgets = api["widgets"]
    method = widgets.Dropdown(options=["cfm", "endpoint"], value="endpoint")
    nfe = widgets.Dropdown(options=[1, 2, 4], value=2)
    view = widgets.VBox([widgets.HBox([method, nfe])])
    annotated = api["_annotate_preview"](view)
    assert "off-target" in annotated.children[0].value
    nfe.value = 1
    assert annotated.children[0].value == ""
    nfe.value = 4
    assert "off-target" in annotated.children[0].value
    method.value = "cfm"
    assert annotated.children[0].value == ""
    annotated.close()
    view.close()


def test_real_cached_widget_task_switch_reuses_widgets_and_actual_cm_events(tmp_path, monkeypatch):
    api = results_api()
    core = api["load_evaluation_core"](ROOT)
    monkeypatch.setattr(
        core, "_training_api", lambda: (_ for _ in ()).throw(AssertionError("Model API called"))
    )
    report = tmp_path / "replication/reports/method_zoo/final"
    (report / "previews").mkdir(parents=True)
    (report / "results.json").write_text(
        json.dumps({"metadata": {"status": "completed"}, "rows": []})
    )
    (report / "primary_summary.csv").write_text(
        "task,method,nfe,conditional_swd\nrays,cfm,2,0.1\nwall,cfm,2,0.2\n"
    )
    for task in ("rays", "wall"):
        torch.save(synthetic_preview(task), report / f"previews/{task}_preview.pt")
    actual = core.sampling_preview_widget
    calls, views = [], []

    def counted(path):
        calls.append(Path(path).name)
        result = actual(path)
        views.append(result)
        return result

    monkeypatch.setattr(core, "sampling_preview_widget", counted)
    browser = api["create_results_browser"](tmp_path, core_module=core)
    selector = browser.children[1]
    assert calls == ["rays_preview.pt"]
    # The actual core widget retains one-event increments and nonmonotone CM nodes.
    choose_method, choose_nfe = views[0].children[0].children[:2]
    choose_method.value = "consistency"
    choose_nfe.value = 4
    play, frame = views[0].children[1].children
    assert play.step == frame.step == 1 and play.max == frame.max == 7
    frame.value = 2  # t=.75 after t=0, not a smoothed physical time axis.
    selector.value = "wall"
    selector.value = "rays"
    assert calls == ["rays_preview.pt", "wall_preview.pt"]
    assert choose_method.value == "consistency" and choose_nfe.value == 4 and frame.value == 2
    for view in views:
        view.close()
    browser.close()
