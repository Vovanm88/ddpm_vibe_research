# Обученные вращающиеся лучи

Запуск: 120 эпох, 4096 точек четырёх классов, MLP 128 × 4, embedding conditioner,
v prediction, cosine schedule с 1000 шагами, AdamW, EMA 0.99. Обучение на CPU
заняло 22.75 секунды, 1920 optimizer steps.

Обучающая выборка имеет seed 123, отдельная контрольная — 124. Генерация:
1024 точки, DDIM 100 шагов, CFG 1.5, seed 127. Sliced Wasserstein на 128
проекциях с seed 123: **0.658185 до обучения → 0.057220 после обучения**.

Откройте [обученный интерактивный notebook](../notebooks/06_trained_interactive_demo.ipynb)
или [сразу в Colab](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/main/notebooks/06_trained_interactive_demo.ipynb).
Пакет устанавливается из GitHub, веса Safetensors и контрольная выборка
загружаются из отдельных файлов `demo/`. В notebook нет бинарных вложений.

Поле пересчитывается по модели при изменении condition/timestep/CFG/kind.
Генератор запускается кнопкой Generate; Frame/Play показывают реальные
сохранённые состояния денойзинга. Recover potential показывает проекцию
F≈−∇U и численную диагностику residual/curl.

В `pipeline/` лежат EMA-веса и Diffusers configs. `checkpoint.pt` содержит
live weights, EMA, optimizer и RNG. Для экспериментов из клонированного репозитория:

```python
from diffusion_lab import ResearchDiffusionPipeline
pipeline = ResearchDiffusionPipeline.from_pretrained(
    "demo/pipeline", local_files_only=True, low_cpu_mem_usage=False,
)
```

Данные, snapshots и learning history сохранены для воспроизведения графиков.
`manifest.json` содержит точные параметры и версии. Метрика стандартного
отчёта 0.05693 использует другую последовательность проекций (seed 127);
для сравнения до/после используется одна и та же последовательность seed 123.

- [Данные / до / после](figures/comparison.png)
- [Loss и качество](figures/learning.png)
- [Плотности на общей шкале](figures/density.png)
- [Score на разных t](figures/score_fields.png)
- [Потенциал и curl](figures/potential_diagnostics.png)
- [Автономный HTML explorer](figures/explorer.html)
- [Денойзинг GIF](figures/denoising.gif)

Все статические графики имеют SVG-копии. Потенциал при t=349 имеет невязку
0.78%, но ненулевой curl: это приблизительная проекция, не подтверждённая
точная энергия. Для просмотра HTML можно скачать его и открыть в браузере;
отдельный сервер не требуется.

Повторить обучение и рендер из корня репозитория:

```bash
uv run python examples/train_visual_demo.py --epochs 120 --output outputs/new_visual_demo
uv run python examples/render_visual_demo.py --run outputs/new_visual_demo
```
