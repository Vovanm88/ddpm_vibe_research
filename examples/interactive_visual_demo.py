"""Build an offline explorer with actual model scores at every training timestep.

Only vector values are cached: no weights, wheel, interpolated time states or
duplicated Plotly animation frames are embedded in the HTML. JavaScript applies
CFG to the cached unconditional / conditional scores, then redraws the arrows.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs

from diffusion_lab.pipeline import ResearchDiffusionPipeline
from diffusion_lab.schedulers import alpha_bar, prediction_to_x0
from diffusion_lab.visualization import Grid
from diffusion_lab.visualization.plots import PALETTE


@torch.no_grad()
def _score_cache(pipeline, grid, batch_size=8192):
    """Return [mode, t, point, xy], querying all integer t in batched forwards."""
    if tuple(pipeline.model.sample_shape) != (2,):
        raise ValueError("This explorer requires a 2D point denoiser")
    if isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError("This score explorer requires a DDPM/DDIM model, not rectified flow")
    if pipeline.model.config.conditioner not in ("one_hot", "embedding"):
        raise ValueError("This rays explorer requires a categorical condition encoder")
    if pipeline.model.config.num_classes != 4:
        raise ValueError("This rays explorer expects four model classes")
    maximum = int(pipeline.scheduler.config.num_train_timesteps)
    model = pipeline.model
    points = torch.as_tensor(grid.points.copy(), device=model.device, dtype=model.dtype)
    scores = np.empty((5, maximum, grid.size, 2), dtype=np.float32)
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for mode, condition in enumerate((None, 0, 1, 2, 3)):
            flat = scores[mode].reshape(-1, 2)
            for begin in range(0, maximum * grid.size, batch_size):
                end = min(begin + batch_size, maximum * grid.size)
                index = torch.arange(begin, end, device=model.device)
                timesteps = torch.div(index, grid.size, rounding_mode="floor")
                samples = points[index.remainder(grid.size)]
                conditions = (
                    None
                    if condition is None
                    else torch.full_like(timesteps, condition, dtype=torch.long)
                )
                prediction = model(samples, timesteps, conditions=conditions).sample
                clean = prediction_to_x0(pipeline.scheduler, samples, prediction, timesteps)
                a = alpha_bar(pipeline.scheduler, timesteps, samples)
                epsilon = (samples - a.sqrt() * clean) / (1 - a).sqrt()
                score = -epsilon / (1 - a).sqrt()
                flat[begin:end] = score.float().cpu().numpy()
            print(
                f"Cached {'Unconditional' if condition is None else f'Class {condition}'}: "
                f"all {maximum} timesteps × {grid.size} points",
                flush=True,
            )
    finally:
        for module, was_training in training_states:
            module.training = was_training
    if not np.isfinite(scores).all():
        raise FloatingPointError("The model returned nonfinite score vectors")
    return scores


def _model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


_APPLICATION_JS = r"""
const SPEC = __SPEC__;
const COMPRESSED_SCORES_BASE64 = "__PAYLOAD__";
const dom = {
  condition: document.getElementById('condition'),
  timestep: document.getElementById('timestep'),
  useCFG: document.getElementById('use-cfg'),
  guidance: document.getElementById('guidance'),
  timeValue: document.getElementById('timestep-value'),
  guidanceValue: document.getElementById('guidance-value'),
  status: document.getElementById('status'),
  plot: document.getElementById('plot')
};
let cache = null;
let rendering = false;
let redrawRequested = false;

async function decodeScores(base64) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This offline explorer needs a current browser with DecompressionStream(gzip).');
  }
  const compressed = Uint8Array.from(atob(base64), value => value.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  if (buffer.byteLength !== SPEC.cache_bytes) {
    throw new Error('Invalid score cache length: ' + buffer.byteLength + ' bytes.');
  }
  const littleEndian = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 4;
  if (littleEndian) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const result = new Float32Array(SPEC.cache_bytes / 4);
  for (let index = 0; index < result.length; index++) result[index] = view.getFloat32(index * 4, true);
  return result;
}

function calculateField(scores, condition, timestep, useCFG, guidance) {
  const count = SPEC.points.length;
  const totalTimes = SPEC.num_timesteps;
  if (!Number.isInteger(timestep) || timestep < 0 || timestep >= totalTimes) {
    throw new Error('t must be an actual integer training timestep.');
  }
  if (!Number.isInteger(condition) || condition < -1 || condition > 3) {
    throw new Error('Condition must be Unconditional or class 0–3.');
  }
  const effective = useCFG ? guidance : 1;
  const unconditionalStart = timestep * count * 2;
  const conditionalStart = ((condition + 1) * totalTimes + timestep) * count * 2;
  const result = new Float32Array(count * 2);
  for (let index = 0; index < result.length; index++) {
    const unconditional = scores[unconditionalStart + index];
    result[index] = condition === -1 ? unconditional :
      effective === 1 ? scores[conditionalStart + index] :
      unconditional + effective * (scores[conditionalStart + index] - unconditional);
  }
  return result;
}

function selectedControls() {
  const condition = Number(dom.condition.value);
  const timestep = Number(dom.timestep.value);
  const guidance = Number(dom.guidance.value);
  const useCFG = dom.useCFG.checked;
  dom.guidance.disabled = !useCFG || condition === -1;
  dom.timeValue.textContent = String(timestep);
  dom.guidanceValue.textContent = guidance.toFixed(1);
  return {condition, timestep, guidance, useCFG};
}

function buildTraces(field, condition) {
  const traces = [];
  SPEC.reference.forEach((values, classId) => {
    if (condition !== -1 && classId !== condition) return;
    traces.push({type: 'scatter', mode: 'markers', name: 'Reference class ' + classId,
      x: values.map(point => point[0]), y: values.map(point => point[1]),
      marker: {size: 4, opacity: 0.35, color: SPEC.class_colors[classId]},
      hovertemplate: 'Reference class ' + classId + '<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>'});
  });
  const count = SPEC.points.length;
  const magnitudes = new Array(count);
  let minimum = Infinity;
  let maximum = 0;
  for (let index = 0; index < count; index++) {
    const magnitude = Math.hypot(field[2 * index], field[2 * index + 1]);
    magnitudes[index] = magnitude;
    if (magnitude > 0) minimum = Math.min(minimum, magnitude);
    maximum = Math.max(maximum, magnitude);
  }
  const low = Math.log10(Number.isFinite(minimum) ? Math.max(minimum, 1e-12) : 1e-12);
  const high = Math.max(low + 0.01, Math.log10(Math.max(maximum, 1e-12)));
  const bins = SPEC.arrow_colors.map(() => ({x: [], y: []}));
  const custom = [];
  const colors = [];
  for (let index = 0; index < count; index++) {
    const [x, y] = SPEC.points[index];
    const fx = field[2 * index];
    const fy = field[2 * index + 1];
    const magnitude = magnitudes[index];
    const colorValue = Math.log10(Math.max(magnitude, 1e-12));
    colors.push(colorValue);
    custom.push([fx, fy, magnitude]);
    if (magnitude === 0) continue;
    const bin = Math.max(0, Math.min(bins.length - 1,
      Math.floor((colorValue - low) / (high - low) * bins.length)));
    const ux = fx / magnitude;
    const uy = fy / magnitude;
    const length = SPEC.arrow_length;
    const tipX = x + length * ux;
    const tipY = y + length * uy;
    const head = length * 0.28;
    // Three coordinate-space segments: shaft plus two arrowhead branches.
    bins[bin].x.push(x, tipX, null,
      tipX - head * ux - head * 0.48 * uy, tipX, tipX - head * ux + head * 0.48 * uy, null);
    bins[bin].y.push(y, tipY, null,
      tipY - head * uy + head * 0.48 * ux, tipY, tipY - head * uy - head * 0.48 * ux, null);
  }
  bins.forEach((values, index) => {
    traces.push({type: 'scatter', mode: 'lines', x: values.x, y: values.y,
      line: {color: SPEC.arrow_colors[index], width: 1.5},
      showlegend: false, hoverinfo: 'skip'});
  });
  const tickvals = Array.from({length: 5}, (_, index) => low + (high - low) * index / 4);
  traces.push({type: 'scatter', mode: 'markers', name: 'Raw model score',
    x: SPEC.points.map(point => point[0]), y: SPEC.points.map(point => point[1]),
    customdata: custom,
    marker: {size: 5, color: colors, colorscale: 'Viridis', cmin: low, cmax: high,
      showscale: true, colorbar: {title: {text: 'Raw |F|<br>log scale'}, tickvals,
        ticktext: tickvals.map(value => (10 ** value).toPrecision(3)), thickness: 16}},
    hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<br>Fₓ=%{customdata[0]:.6g}' +
      '<br>Fᵧ=%{customdata[1]:.6g}<br>|F|=%{customdata[2]:.6g}<extra>Actual cached score</extra>',
    showlegend: false});
  return {traces, minimum: Number.isFinite(minimum) ? minimum : 0, maximum};
}

async function render() {
  if (cache === null) return;
  const controls = selectedControls();
  const field = calculateField(cache, controls.condition, controls.timestep,
    controls.useCFG, controls.guidance);
  const {traces, minimum, maximum} = buildTraces(field, controls.condition);
  const conditionLabel = controls.condition === -1 ? 'Unconditional' : 'Condition ' + controls.condition;
  const guidanceLabel = controls.condition === -1 ? 'unconditional score' :
    controls.useCFG ? 'CFG ' + controls.guidance.toFixed(1) : 'CFG off · conditional score (scale 1)';
  const layout = {
    title: {text: conditionLabel + ' · actual t=' + controls.timestep + ' · ' + guidanceLabel,
      x: 0.05, font: {size: 20}},
    xaxis: {title: {text: 'x'}, range: SPEC.bounds[0], zeroline: false, gridcolor: '#e2e8f0'},
    yaxis: {title: {text: 'y'}, range: SPEC.bounds[1], scaleanchor: 'x', scaleratio: 1,
      zeroline: false, gridcolor: '#e2e8f0'},
    paper_bgcolor: 'white', plot_bgcolor: '#f8fafc',
    font: {family: 'Arial, sans-serif', color: '#14263d'},
    legend: {orientation: 'h', y: -0.12},
    margin: {l: 60, r: 110, t: 70, b: 65}, height: 720,
    uirevision: 'actual-model-field'
  };
  await Plotly.react(dom.plot, traces, layout, {responsive: true, displaylogo: false});
  dom.status.textContent = conditionLabel + ', t=' + controls.timestep + '; ' + guidanceLabel +
    '. Raw |F| range ' + minimum.toPrecision(4) + '–' + maximum.toPrecision(4) +
    '. Fixed arrow lengths show direction; hover shows actual components.';
  window.diffusionExplorer.lastField = field;
  window.diffusionExplorer.lastControls = controls;
}

function requestRender() {
  selectedControls();
  redrawRequested = true;
  if (rendering || cache === null) return;
  rendering = true;
  requestAnimationFrame(async () => {
    try {
      do {
        redrawRequested = false;
        await render();
      } while (redrawRequested);
    } catch (error) {
      dom.status.textContent = error.message;
    } finally {
      rendering = false;
    }
  });
}

window.diffusionExplorer = {SPEC, calculateField, decodeScores, selectedControls, render, requestRender};
['condition', 'timestep', 'useCFG', 'guidance'].forEach(name => {
  dom[name].addEventListener(name === 'timestep' || name === 'guidance' ? 'input' : 'change', requestRender);
});
selectedControls();
window.diffusionExplorer.ready = decodeScores(COMPRESSED_SCORES_BASE64).then(async scores => {
  cache = scores;
  window.diffusionExplorer.cache = scores;
  rendering = true;
  try {
    do {
      redrawRequested = false;
      await render();
    } while (redrawRequested);
  } finally {
    rendering = false;
  }
  return scores;
}).catch(error => {
  dom.status.textContent = error.message;
  throw error;
});
"""


def build_explorer(pipeline, reference, labels, output_path, resolution=17, guidance_scale=1.5):
    """Write one self-contained HTML and its provenance JSON; return metadata."""
    if not isinstance(resolution, int) or not 4 <= resolution <= 33:
        raise ValueError("resolution must be an integer between 4 and 33")
    if not np.isfinite(guidance_scale) or not 0 <= guidance_scale <= 3:
        raise ValueError("guidance_scale must be finite and within [0,3]")
    reference = np.asarray(reference, dtype=np.float32)
    labels = np.asarray(labels)
    if reference.ndim != 2 or reference.shape[1] != 2 or labels.shape != (len(reference),):
        raise ValueError("Expected reference[N,2] and labels[N]")
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Reference points must be nonempty and finite")
    if not np.array_equal(np.unique(labels), np.arange(4)):
        raise ValueError("Reference labels must contain all four classes 0,1,2,3")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    grid = Grid.from_bounds([(-1.15, 1.15), (-1.15, 1.15)], resolution)
    scores = _score_cache(pipeline, grid)
    raw = scores.astype("<f4", copy=False).tobytes()
    payload = base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii")
    restored = np.frombuffer(gzip.decompress(base64.b64decode(payload)), dtype="<f4")
    if not np.array_equal(restored.reshape(scores.shape), scores):
        raise RuntimeError("Score cache compression roundtrip failed")
    classes = []
    for condition in range(4):
        values = reference[labels == condition]
        indices = np.linspace(0, len(values) - 1, min(len(values), 1200), dtype=int)
        classes.append(values[indices].tolist())
    maximum = scores.shape[1]
    spec = {
        "num_timesteps": maximum,
        "cache_bytes": len(raw),
        "cache_shape": list(scores.shape),
        "points": grid.points.tolist(),
        "reference": classes,
        "class_colors": [PALETTE[index % len(PALETTE)] for index in range(4)],
        "arrow_colors": sample_colorscale("Viridis", np.linspace(0, 1, 16).tolist()),
        "arrow_length": float((grid.axes[0][1] - grid.axes[0][0]) * 0.48),
        "bounds": [[-1.15, 1.15], [-1.15, 1.15]],
    }
    application = _APPLICATION_JS.replace("__SPEC__", json.dumps(spec, allow_nan=False)).replace(
        "__PAYLOAD__", payload
    )
    start = int((maximum - 1) * 0.35)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rotating rays · actual model score explorer</title>
<style>
body{{margin:0;background:#edf2f7;font:15px Arial,sans-serif;color:#14263d}}
main{{max-width:1100px;margin:20px auto;background:white;border:1px solid #dbe5ef;border-radius:12px;padding:22px}}
h1{{font-size:23px;margin:0 0 12px}}.controls{{display:flex;gap:24px;align-items:center;flex-wrap:wrap;padding:18px;background:#f8fafc;border-radius:9px}}
label{{display:flex;gap:9px;align-items:center}}select{{padding:7px;font:inherit}}input[type=range]{{width:220px}}
output{{font-variant-numeric:tabular-nums;min-width:32px}}input:disabled{{opacity:.35}}#status{{font-size:13px;color:#5c6e80;min-height:36px;margin:15px 0 0}}.note{{font-size:13px;color:#5c6e80;line-height:1.6}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>Explore every actual noise timestep of the trained model</h1>
<div class="controls">
<label>Condition <select id="condition"><option value="-1">Unconditional</option><option value="0" selected>Condition 0</option><option value="1">Condition 1</option><option value="2">Condition 2</option><option value="3">Condition 3</option></select></label>
<label>Noise timestep <input id="timestep" type="range" min="0" max="{maximum - 1}" step="1" value="{start}"><output id="timestep-value">{start}</output></label>
<label><input id="use-cfg" type="checkbox" checked>Use CFG</label>
<label>Guidance <input id="guidance" type="range" min="0" max="3" step="0.1" value="{guidance_scale}"><output id="guidance-value">{guidance_scale:.1f}</output></label>
</div><p id="status" role="status">Decoding the cached actual model scores…</p><div id="plot"></div>
<p class="note">All {maximum} integer timesteps were queried from the trained network on a {resolution} × {resolution} grid. No time interpolation. CFG uses F = Fᵤ + s(F꜀ − Fᵤ). With CFG off, a selected class uses its conditional score at scale 1; Unconditional always shows Fᵤ. Reference points are clean data, filtered by condition. Arrow directions are normalized only for display; colors and hover values retain the raw score. The logarithmic color scale updates for each selected field; its tick labels show raw magnitudes. Everything needed for this page is inline and works offline in a current browser.</p>
</main><script id="diffusion-explorer-application">{application}</script></body></html>"""
    output_path.write_text(html)
    metadata = {
        "html": output_path.name,
        "field": "score",
        "actual_timesteps": list(range(maximum)),
        "timestep_range": [0, maximum - 1],
        "timestep_step": 1,
        "interpolated": False,
        "conditions": [None, 0, 1, 2, 3],
        "cached_guidance_scale": 1,
        "cfg_formula": "Fu + s * (Fc - Fu)",
        "resolution": resolution,
        "bounds": spec["bounds"],
        "cache_shape": list(scores.shape),
        "cache_dtype": "Float32 little-endian",
        "cache_bytes": len(raw),
        "cache_sha256": hashlib.sha256(raw).hexdigest(),
        "model_sha256": _model_digest(pipeline.model),
        "scheduler_class": type(pipeline.scheduler).__name__,
        "scheduler_config": dict(pipeline.scheduler.config),
        "default_guidance_scale": guidance_scale,
        "arrow_display": "Fixed coordinate-space length; direction from raw score",
        "color_scale": "Log10 magnitude; limits recomputed per selected field; tick labels in raw units",
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "compression_roundtrip_verified": True,
        "html_bytes": output_path.stat().st_size,
    }
    output_path.with_name(output_path.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved offline explorer: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MiB)")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("demo"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=17)
    args = parser.parse_args()
    run = args.run.resolve()
    pipeline = ResearchDiffusionPipeline.from_pretrained(
        run / "pipeline", local_files_only=True, low_cpu_mem_usage=False
    ).to("cpu")
    with np.load(run / "reference.npz", allow_pickle=False) as values:
        reference, labels = values["samples"], values["conditions"]
    output = args.output or run / "figures" / "explorer.html"
    build_explorer(pipeline, reference, labels, output, resolution=args.resolution)


if __name__ == "__main__":
    main()
