# Обученные дистилляции и reflow

Отдельные результаты полного выполнения notebooks 07/08/09 на CPU.
Описание методов, seeds и сравнения — [docs/distillation.md](../../docs/distillation.md).
Ноутбуки содержат исходный код обучения; веса, графики и checkpoints лежат здесь.

- `progressive_distillation/`: teacher-100 / student-50 / student-25 samples и GIF,
  metrics/history, оба stage checkpoints и EMA Diffusers pipelines.
- `model_distillation/`: меньшая student model с fixed guidance=1.5, raw pipeline,
  optimizer checkpoint, manifest с NFE/parity/SWD, samples, PNG/SVG и GIF.
- `reflow/`: 1-RF и 2-RF pipelines / optimizer states, связанные пары с fixed mask,
  training/heldout данные, geometry/quality JSON, actual ODE NPZ и GIF.

В progressive pipeline явно передавайте обученные 50 или 25 шагов (`eta=0`);
model-distillation pipeline запускайте с `guidance_scale=1.0` — guidance teacher
уже включён в условные ответы student. Reflow pipelines используют FlowMatchEuler.
Для loading импортируйте `ResearchDiffusionPipeline` из `diffusion_lab`.

`artifacts.json` содержит относительные пути, размеры и SHA256 всех файлов этого
набора. Сохранены исходные manifests запусков; их локальные пути относятся к
машине обучения. Для новых запусков используются свои каталоги `outputs/`.
