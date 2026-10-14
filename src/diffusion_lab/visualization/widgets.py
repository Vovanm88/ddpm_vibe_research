"""Live Jupyter controls over ordinary PyTorch / Diffusers pipelines.

Widgets are optional: importing the package does not import ipywidgets. Field
sliders query the model on release; sampling runs only when Generate is pressed.
The history player displays cached, actual scheduler states, without resampling.
"""

import math
from collections.abc import Mapping
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .fields import Grid, as_numpy, evaluate_field, recover_potential
from .plots import plot_dataset, plot_potential, plot_trajectories, plot_vector_field


def _widget_modules():
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            'Jupyter explorers require optional dependencies: pip install "diffusion-lab[interactive]"'
        ) from exc
    return widgets, display


def _condition_choices(pipeline, dataset, options):
    config = pipeline.model.config
    kind = getattr(config, "conditioner", "none")
    dimension = int(getattr(config, "condition_dim", 0))
    categorical = kind in ("one_hot", "embedding")
    continuous = kind != "none" and not categorical and dimension > 0
    entries = [("Unconditional", None)]
    if options is not None:
        if not isinstance(options, Mapping):
            raise ValueError("condition_options must map display labels to class IDs or vectors")
        for name, value in options.items():
            if value is None:
                continue
            if categorical:
                array = np.asarray(as_numpy(value))
                if array.ndim or not np.isfinite(array) or int(array) != array:
                    raise ValueError("Categorical condition options must be integer scalar IDs")
                value = int(array)
                if not 0 <= value < config.num_classes:
                    raise ValueError("Condition option class ID is outside the model vocabulary")
            elif continuous:
                value = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
                if value.shape != (dimension,) or not torch.isfinite(value).all():
                    raise ValueError(f"Condition option vectors must have shape ({dimension},)")
            else:
                raise ValueError("This model has no condition encoder")
            entries.append((str(name), value))
    elif categorical:
        if dataset is not None and dataset.condition_kind == "categorical":
            values = torch.unique(dataset.conditions).tolist()
        else:
            values = range(config.num_classes)
        entries.extend((f"Class {int(value)}", int(value)) for value in values)
    elif continuous and dataset is not None and dataset.conditions is not None:
        if dataset.condition_kind != "continuous" or dataset.condition_dim != dimension:
            raise ValueError(
                "Dataset conditions do not match the model's continuous condition width"
            )
        if dataset.labels is not None:
            representatives = [
                (f"Label {int(label)}", int(torch.nonzero(dataset.labels == label)[0]))
                for label in torch.unique(dataset.labels)[:8]
            ]
        else:
            indices = np.unique(np.linspace(0, len(dataset) - 1, min(8, len(dataset)), dtype=int))
            representatives = [(f"Example {index}", int(index)) for index in indices]
        entries.extend(
            (name, dataset.conditions[index].detach().cpu()) for name, index in representatives
        )
    # Dropdown values are integer keys, so tensor comparisons never enter traitlets.
    choices = [(name, index) for index, (name, _) in enumerate(entries)]
    values = {index: value for index, (_, value) in enumerate(entries)}
    return choices, values


def _point_data(pipeline, dataset, points, labels):
    shape = tuple(pipeline.model.sample_shape)
    if len(shape) != 1:
        raise ValueError(
            "Spatial field explorers require a point model with sample_shape=(D,). "
            "For path models, use notebook_sampling_explorer to inspect trajectories."
        )
    if not 1 <= shape[0] <= 3:
        raise ValueError("Field explorers support one, two or three spatial coordinates")
    if points is None and dataset is not None:
        points = dataset.samples
    if labels is None and dataset is not None:
        labels = dataset.labels
    if points is not None:
        points = np.asarray(as_numpy(points))
        if points.ndim != 2 or points.shape[1] != shape[0] or not len(points):
            raise ValueError(f"points must have nonempty shape (N,{shape[0]})")
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")
        if labels is not None:
            labels = as_numpy(labels)
            if labels.shape != (len(points),):
                raise ValueError("labels must have one entry per point")
    elif labels is not None:
        raise ValueError("labels require points or a dataset")
    return shape[0], points, labels


def _field_bounds(dimension, points, bounds):
    if bounds is None:
        if points is None:
            bounds = [(-2.0, 2.0)] * dimension
        else:
            margin = np.maximum(np.ptp(points, axis=0) * 0.12, 0.1)
            bounds = list(zip(points.min(axis=0) - margin, points.max(axis=0) + margin))
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (dimension, 2) or not np.isfinite(bounds).all():
        raise ValueError("bounds must contain (low, high) for each coordinate")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be strictly increasing")
    return bounds.tolist()


def _show_error(status, state, exception):
    state["error"] = exception
    status.value = f'<span style="color:#b91c1c">{escape(str(exception))}</span>'


def notebook_field_explorer(
    pipeline,
    dataset=None,
    *,
    points=None,
    labels=None,
    bounds=None,
    condition_options=None,
    resolution=32,
    timestep=None,
    guidance_scale=1.5,
):
    """Return a live ``ipywidgets.VBox`` for a 1D/2D/3D model vector field.

    ``bounds=[(xmin,xmax), ...]`` uses coordinate order, just like :class:`Grid`.
    Discrete models expose class IDs; continuous models expose representative
    dataset rows, or explicit ``condition_options={"scene": feature_vector}``.
    Flow fields expose velocity and denoising, with sigma=t/T shown explicitly.
    The returned widget has ``controls``, ``output``, ``state`` and ``refresh()``
    for further notebook customization. Display it as the last cell expression.
    """
    widgets, display = _widget_modules()
    dimension, points, labels = _point_data(pipeline, dataset, points, labels)
    bounds = _field_bounds(dimension, points, bounds)
    maximum = int(pipeline.scheduler.config.num_train_timesteps)
    flow = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    max_time = maximum if flow else maximum - 1
    if int(resolution) != resolution or not 4 <= resolution <= 64:
        raise ValueError("resolution must be an integer between 4 and 64")
    timestep = max_time // 3 if timestep is None else timestep
    if int(timestep) != timestep or not 0 <= timestep <= max_time:
        raise ValueError("timestep must be an integer within the scheduler training range")
    choices, values = _condition_choices(pipeline, dataset, condition_options)
    controls = {
        "condition": widgets.Dropdown(options=choices, description="Condition:"),
        "timestep": widgets.IntSlider(
            value=int(timestep), min=0, max=max_time, description="t:", continuous_update=False
        ),
        "guidance_scale": widgets.FloatSlider(
            value=guidance_scale,
            min=0,
            max=3,
            step=0.1,
            description="CFG:",
            continuous_update=False,
        ),
        "kind": widgets.Dropdown(
            options=["velocity", "denoising"] if flow else ["score", "epsilon", "denoising"],
            description="Field:",
        ),
        "normalize": widgets.Checkbox(value=True, description="Normalize arrows"),
        "streamlines": widgets.Checkbox(
            value=dimension == 2, disabled=dimension != 2, description="Streamlines (2D)"
        ),
        "potential": widgets.Checkbox(value=False, description="Recover potential"),
        "resolution": widgets.IntSlider(
            value=int(resolution), min=4, max=64, description="Grid:", continuous_update=False
        ),
    }
    output, status, state = widgets.Output(), widgets.HTML(), {}
    explorer = widgets.VBox(
        [
            widgets.HBox([controls["condition"], controls["kind"]]),
            widgets.HBox([controls["timestep"], controls["guidance_scale"]]),
            widgets.HBox(
                [
                    controls["resolution"],
                    controls["normalize"],
                    controls["streamlines"],
                    controls["potential"],
                ]
            ),
            status,
            output,
        ]
    )

    def refresh(*_):
        figure = None
        try:
            grid = Grid.from_bounds(bounds, controls["resolution"].value)
            condition = values[controls["condition"].value]
            time = controls["timestep"].value
            field_kind = controls["kind"].value
            vectors = evaluate_field(
                lambda query: pipeline.vector_field(
                    query,
                    time,
                    conditions=condition,
                    kind=field_kind,
                    guidance_scale=controls["guidance_scale"].value,
                ),
                grid,
            )
            title = f"{field_kind.capitalize()} · {controls['condition'].label} · t={time}"
            if flow:
                title += f" · σ={time / maximum:.3f} (t/T)"
            title += f" · CFG={controls['guidance_scale'].value:g}"
            potential, field_ax = None, None
            if controls["potential"].value:
                # Recover from raw model vectors, independent of arrow display scaling.
                potential = recover_potential(grid, vectors)
                figure = plt.figure(figsize=(15, 6), constrained_layout=True)
                field_ax = figure.add_subplot(121, projection="3d" if dimension == 3 else None)
                potential_ax = figure.add_subplot(122)
                plot_potential(grid, potential, ax=potential_ax)
            figure, ax = plot_vector_field(
                grid,
                vectors,
                normalize=controls["normalize"].value,
                streamlines=controls["streamlines"].value,
                stride=max(1, math.ceil(grid.shape[-1] / (10 if dimension == 3 else 22))),
                title=title,
                ax=field_ax,
            )
            if points is not None:
                # Deterministic thinning keeps figures responsive and RNG state intact.
                indices = np.linspace(0, len(points) - 1, min(len(points), 4000), dtype=int)
                plot_dataset(
                    points[indices],
                    None if labels is None else labels[indices],
                    ax=ax,
                    size=6,
                    alpha=0.25,
                    title=ax.get_title(),
                )
                for coordinate, (low, high) in enumerate(bounds):
                    getattr(ax, ("set_xlim", "set_ylim", "set_zlim")[coordinate])(low, high)
            state.update(
                grid=grid,
                vectors=vectors,
                condition=condition,
                timestep=time,
                figure=figure,
                potential=potential,
                error=None,
            )
            status.value = (
                f"{grid.size:,} queried points. Arrow color always shows the original |F|."
            )
            if potential is not None:
                verdict = (
                    "conservative on this grid"
                    if potential.is_conservative
                    else "least-squares projection, not an exact potential"
                )
                status.value += (
                    f" Potential: residual={potential.relative_residual:.2%}, "
                    f"relative curl={potential.relative_curl:.3g}; {verdict}."
                )
            with output:
                output.clear_output(wait=True)
                display(figure)
            return figure
        except Exception as exc:
            _show_error(status, state, exc)
            return None
        finally:
            if figure is not None:
                plt.close(figure)

    for control in controls.values():
        control.observe(refresh, names="value")
    explorer.controls, explorer.output, explorer.state, explorer.refresh = (
        controls,
        output,
        state,
        refresh,
    )
    refresh()
    return explorer


def _navigation_scene(dataset, condition, sample_shape):
    """Only decode endpoints when metadata explicitly declares navigation data."""
    if dataset is None or len(sample_shape) != 2 or condition is None:
        return {}, (), None, None
    metadata = dataset.metadata
    if metadata.get("condition_layout") != "start(D), goal(D), walls(W,2,D) flattened in C order":
        return {}, (), None, None
    dimension = sample_shape[-1]
    condition = torch.as_tensor(condition, dtype=torch.float32)
    if (
        condition.ndim != 1
        or len(condition) < 2 * dimension
        or (len(condition) - 2 * dimension) % (2 * dimension)
    ):
        raise ValueError("Navigation condition does not match the start/goal/walls contract")
    known = torch.zeros(sample_shape)
    mask = torch.zeros(sample_shape, dtype=torch.bool)
    known[0], known[-1] = condition[:dimension], condition[dimension : 2 * dimension]
    mask[0], mask[-1] = True, True
    walls = condition[2 * dimension :].reshape(-1, 2, dimension).numpy()
    return {"known_values": known, "known_mask": mask}, walls, known[:1].numpy(), known[-1:].numpy()


def notebook_sampling_explorer(
    pipeline,
    dataset=None,
    *,
    condition_options=None,
    num_samples=256,
    steps=50,
    seed=42,
    guidance_scale=1.5,
):
    """Generate on button press, then play the actual cached denoising history.

    Condition, CFG, inference steps and seed controls do not start generation
    automatically. ``.generate()`` is an explicit equivalent of button click;
    ``.refresh()`` redraws the cached frame. ``.state`` contains samples, history
    and real scheduler timesteps. Navigation endpoints are pinned only when the
    supplied dataset explicitly declares the navigation condition contract.
    A local ``torch.Generator`` leaves the notebook's global RNG state intact.
    """
    widgets, display = _widget_modules()
    sample_shape = tuple(pipeline.model.sample_shape)
    if len(sample_shape) not in (1, 2) or not 1 <= sample_shape[-1] <= 3:
        raise ValueError("Sampling explorer requires point or path data with 1–3 coordinates")
    maximum = int(pipeline.scheduler.config.num_train_timesteps)
    if int(num_samples) != num_samples or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if int(steps) != steps or not 1 <= steps <= maximum:
        raise ValueError("steps must be within the scheduler training range")
    choices, values = _condition_choices(pipeline, dataset, condition_options)
    controls = {
        "condition": widgets.Dropdown(options=choices, description="Condition:"),
        "guidance_scale": widgets.FloatSlider(
            value=guidance_scale,
            min=0,
            max=3,
            step=0.1,
            description="CFG:",
            continuous_update=False,
        ),
        "steps": widgets.IntSlider(
            value=int(steps), min=1, max=maximum, description="Steps:", continuous_update=False
        ),
        "seed": widgets.BoundedIntText(value=int(seed), min=0, max=2**63 - 1, description="Seed:"),
        "generate": widgets.Button(description="Generate", button_style="primary", icon="play"),
        "frame": widgets.IntSlider(
            value=0, min=0, max=0, description="Frame:", continuous_update=False, disabled=True
        ),
        "play": widgets.Play(value=0, min=0, max=0, interval=100, disabled=True),
    }
    output, status, state = (
        widgets.Output(),
        widgets.HTML(value="Set parameters, then press Generate."),
        {},
    )
    explorer = widgets.VBox(
        [
            widgets.HBox([controls["condition"], controls["guidance_scale"]]),
            widgets.HBox([controls["steps"], controls["seed"], controls["generate"]]),
            widgets.HBox([controls["play"], controls["frame"]]),
            status,
            output,
        ]
    )
    # Kernel-side linking also works in classic Notebook, JupyterLab and Colab.
    explorer.history_link = widgets.link((controls["play"], "value"), (controls["frame"], "value"))

    def refresh(*_):
        if "history" not in state or state.get("updating"):
            return None
        figure = None
        try:
            frame = controls["frame"].value
            total = len(state["history"]) - 1
            # history[k] is the state AFTER k scheduler steps. Its next input
            # noise level is timesteps[k], while final output has no noisy t.
            level = (
                f"next t={float(state['timesteps'][frame]):g}"
                if frame < total
                else "final clean sample"
            )
            title = f"{state['condition_label']} · frame {frame}/{total} · {level} · seed={state['seed']}"
            data = state["history"][frame]
            if len(sample_shape) == 1:
                figure, ax = plot_dataset(data, title=title, size=12, alpha=0.6)
                if dataset is not None:
                    plot_dataset(
                        dataset.samples, dataset.labels, ax=ax, title=title, size=5, alpha=0.1
                    )
            else:
                # Large path batches are costly and obscure the geometry.
                figure, ax = plot_trajectories(
                    data[: min(len(data), 40)],
                    walls=state["walls"],
                    starts=state["starts"],
                    goals=state["goals"],
                    title=title,
                )
            if state.get("limits") is not None:
                for coordinate, (low, high) in enumerate(state["limits"]):
                    if len(sample_shape) == 2 and sample_shape[-1] == 1:
                        ax.set_ylim(low, high)
                    else:
                        getattr(ax, ("set_xlim", "set_ylim", "set_zlim")[coordinate])(low, high)
            state.update(figure=figure, error=None)
            status.value = (
                f"Cached run: {state['condition_label']}, CFG={state['guidance_scale']:g}, "
                f"{total} solver steps. Frame {frame}: {level}. "
                "Changed settings take effect after Generate."
            )
            with output:
                output.clear_output(wait=True)
                display(figure)
            return figure
        except Exception as exc:
            _show_error(status, state, exc)
            return None
        finally:
            if figure is not None:
                plt.close(figure)

    def generate(*_):
        controls["generate"].disabled = True
        try:
            condition = values[controls["condition"].value]
            constraints, walls, starts, goals = _navigation_scene(dataset, condition, sample_shape)
            device = pipeline.model.device
            # torch.Generator does not support MPS; Diffusers transfers its CPU draws.
            generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
            generator.manual_seed(controls["seed"].value)
            result = pipeline(
                batch_size=int(num_samples),
                conditions=condition,
                num_inference_steps=controls["steps"].value,
                guidance_scale=controls["guidance_scale"].value,
                generator=generator,
                record_history=True,
                **constraints,
            )
            history = result.history.detach().cpu()
            limits = None
            if dataset is not None:
                reference = as_numpy(dataset.samples).reshape(-1, sample_shape[-1])
                final = as_numpy(history[-1]).reshape(-1, sample_shape[-1])
                combined = np.concatenate((reference, final))
                limits = _field_bounds(sample_shape[-1], combined, None)
            state.update(
                samples=result.samples.detach().cpu(),
                history=history,
                timesteps=result.timesteps.detach().cpu(),
                condition=condition,
                condition_label=controls["condition"].label,
                seed=controls["seed"].value,
                guidance_scale=controls["guidance_scale"].value,
                walls=walls,
                starts=starts,
                goals=goals,
                limits=limits,
                updating=True,
                error=None,
            )
            total = len(history) - 1
            controls["play"].max = total
            controls["frame"].max = total
            controls["frame"].value = total
            controls["play"].disabled = controls["frame"].disabled = False
            state["updating"] = False
            refresh()
            return result
        except Exception as exc:
            _show_error(status, state, exc)
            return None
        finally:
            controls["generate"].disabled = False

    controls["generate"].on_click(generate)
    controls["frame"].observe(refresh, names="value")
    explorer.controls, explorer.output, explorer.state = controls, output, state
    explorer.refresh, explorer.generate = refresh, generate
    return explorer
