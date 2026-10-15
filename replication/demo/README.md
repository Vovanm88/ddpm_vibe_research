# Опубликованные обученные примеры

`rays_cpu_pilot_v1/` — один FM teacher (6000 updates) и 12 студентов M0/M1/E0/E1
(по 500 updates, seeds 101/202/303) для 32→16. `rays_stress_32_8_v1/` — тот же
самый teacher и отдельные 12 студентов для 32→8. Данные unconditional; class labels
не подаются модели. Teacher SHA в обоих наборах совпадает.

Веса находятся в отдельных `ema_model/` / `live_model/` Safetensors. Trusted-local
`training.pt` содержит optimizer, EMA, CPU RNG streams и состояние продолжения.
В notebook нет весов, wheel, outputs или вложенных бинарных объектов.

```python
from recheck.fm import load_model, sample_euler
import torch
teacher = load_model("replication/demo/rays_cpu_pilot_v1/teacher/ema_model")
noise = torch.randn(256, 2, generator=torch.Generator().manual_seed(730001))
result = sample_euler(teacher, noise, steps=32)
```

Таблицы и GIF: [32→16](../reports/REPORT.md),
[32→8](../reports/rays_stress_32_8/REPORT.md). Все анимации показывают первый
зарегистрированный seed и первые общие latents; хорошие примеры не отбирались.
Маленькие эффекты OT на лучах не устанавливают перенос на MNIST.

Для точного CLI resume скопируйте нужный каталог в одноимённый
`replication/artifacts/` и используйте исходный YAML с `--resume`: output path входит
в строгий checkpoint contract. Для нового исследования задайте новый output_dir.
Отчёт можно пересобрать непосредственно из demo через `python -m recheck.cli report
--runs replication/demo/rays_cpu_pilot_v1 --output <report_dir>/REPORT.md`.
