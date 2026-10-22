# Method-zoo selected native EMA exports

48 completed warm-start runs: rays/wall × eight methods × seeds 161/262/363, 2000 updates each.
Read each method's `manifest.json` → `inference_exports['1'|'2'|'4']` for its native path,
tensor SHA, trained/off-target interpretation and cumulative updates. Progressive uses
stage4/2/1 at cumulative1200/1600/2000; stage16/8 publish metadata only. Every full run
finished2000 before the final assessment. Histories contain all2000 optimizer updates.

Native weights use Diffusers `save_pretrained`. `examples/train_method_zoo.py` exposes
`load_zoo_model` and `sample_zoo`; conditional inputs remain mandatory, CFG is off.
Wall models predict30 interior points and reconstruct A/B: exact endpoints are construction.
Consistency uses explicit denoise/re-noise events and an additional Gaussian bank for K>1.

Only selected EMA weights, manifests, loss histories, config and hashed training traces
are public. **Exact resume is unavailable from this package.** Live weights, AdamW,
RNG states, private `training.pt` and full8192-row teacher trajectory banks are excluded.
`audit_traces.json` preserves raw/pair trace lists plus source-checkpoint SHA/size;
the digests audit matched streams but cannot reconstruct optimizer/RNG state.
Pairbank manifests preserve hashes/costs, without full trajectories or latent banks.

The two20k parent exports remain in `replication/demo/rays16_conditional_v1/teacher`
and `replication/demo/wall_navigation_conditional_v1/teacher`; they are not duplicated.
Training source: `replication/reports/method_zoo/training_notebook15.ipynb` (69a55…);
the current notebook has a separately recorded GIF-only UI amendment with identical API.
See `training_summary.json`/`.csv` for parameters and operational costs. Timings came
from concurrent task jobs and do not support speed/efficiency claims.

Verify payload file sizes/SHA, all native tensor hashes and144 inference references:
`python examples/publish_method_zoo.py --verify-only`
