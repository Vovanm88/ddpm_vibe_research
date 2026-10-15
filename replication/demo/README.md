# Опубликованные обученные примеры

`rays16_fourier_teacher_v1/` — **16 тонких вращающихся лучей**, один unconditional
FourierFMMLP teacher (100 000 updates, hidden 256 × 4, time embedding 32) и 12
студентов M0/M1/E0/E1 (по 2500 updates, seeds 101/202/303) для **32→16**.
`rays16_fourier_teacher_stress8_v1/` — тот же самый frozen teacher и 12 отдельных
студентов для **32→8**. Fourier features общие для координат; модель не получает
class labels, число лучей или аналитическую проекцию на кривую.

Оба набора содержат выбранные **финальные EMA**, native Safetensors/config,
исходные manifest/history, train-only calibration и зафиксированные результаты
evaluation/траектории. Общий teacher SHA:
`495ff88057d8a5dccbc1581d887404332e9f24ba4dfad658f730de389323b93a`.
В каждом `PUBLICATION_MANIFEST.json` указаны SHA256 и размеры опубликованных файлов.
Наборы копировались после завершения всех 24 student-прогонов и обеих core evaluation.

В основной C16-серии дополнительно сохранён `snapshots/teacher_50000/`: это
**диагностический EMA snapshot на 50 000 updates**, не родитель student-серии.
Его исходный manifest честно сохраняет `status="interrupted"`: teacher был остановлен
на этом checkpoint и затем продолжен до финальных 100 000 updates. Для основной
серии и notebook используется финальный 100k teacher, без отбора лучшего snapshot.
Геометрическая QC остаётся важной: сам completed status не подтверждает освоение мод.

Новые C16-публикации — **selected EMA only**: `training.pt`, `live_model/`, optimizer
и RNG checkpoints исключены. **Точный resume этих прогонов из demo невозможен.**
Исходные manifests не переписаны; их `checkpoint_sha256` может ссылаться на намеренно
неопубликованный training checkpoint. EMA подходит для inference и инициализации
нового исследования с новым optimizer/output_dir. `datasets.pt` и
`evaluation/trajectories.pt` — зафиксированные данные/траектории, а не состояния optimizer.
Публикация содержит независимый final/test export; для новых изменений используйте
train/validation и новый протокол, а не подбирайте настройки по этому test.

Открытый пример обучения, geometry QC, интерактивные графики и GIF:
[notebook12](../../notebooks/12_many_rays_wasserstein_tube.ipynb),
конфигурации [32→16](../configs/many_rays_fourier_cpu.yaml) и
[32→8](../configs/many_rays_fourier_stress_32_8.yaml). Нужна версия пакета ≥0.4.2.
`TEACHER_PRESET="plain"` в notebook выбирает отдельный plain-MLP C16 baseline:
его 100k teacher оставался underfit по phase residual и не подменяет Fourier-parent.
Веса, wheel и бинарные outputs в notebook не встроены.

```python
from recheck.fm import load_model, sample_euler
import torch

teacher = load_model("replication/demo/rays16_fourier_teacher_v1/teacher/ema_model")
noise = torch.randn(256, 2, generator=torch.Generator().manual_seed(730002))
result = sample_euler(teacher, noise, steps=32)
student = load_model("replication/demo/rays16_fourier_teacher_v1/students/seed101_E1/ema_model")
student_result = sample_euler(student, noise, steps=16)
```

Более ранние примеры на **четырёх лучах** имеют другой состав публикации:

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

Только для этих старых четырёхлучевых наборов с `training.pt`: для точного CLI resume
скопируйте нужный каталог в одноимённый
`replication/artifacts/` и используйте исходный YAML с `--resume`: output path входит
в строгий checkpoint contract. Для нового исследования задайте новый output_dir.
Отчёт можно пересобрать непосредственно из demo через `python -m recheck.cli report
--runs replication/demo/rays_cpu_pilot_v1 --output <report_dir>/REPORT.md`.

## Conditional-пример: выбор одного из 16 лучей

`rays16_conditional_v1/` содержит отдельный обученный conditional teacher:
native `PointDenoiser`, `conditioner="embedding"`, 16 labels, hidden 128 × 3,
333 698 параметров, **20 000 CFM updates**. Модель получает категориальный target
при каждом вызове; время передаётся как `t * 1000`. Это демонстрация управления
генерацией через condition, отдельная от unconditional SPEC-серии выше.

В проверке **тот же набор из 256 initial noises повторён для всех 16 targets**.
При 32 Euler-шагах, **без CFG** (`guidance_scale=1`), правильный луч получен
в 4095/4096 случаев; вне центральной области `r > 0.25R` — в 3903/3903 случаев
при coverage 95.29%. Null-контроль даёт 6.25%. Для одного и того же initial noise
conditional-выходы попадают в среднем на 15.9961 различных лучей из 16,
а null-выход остаётся одним и тем же при смене target.

Открытый пример обучения и дистилляции:
[notebook13](../../notebooks/13_conditional_rays_distillation.ipynb).
[Метрики и ограничения проверки](../reports/conditional_rays/REPORT.md),
[GIF с общим стартом и разными targets](../reports/conditional_rays/condition_routing.gif).
Принадлежность измеряется геометрическим phase proxy, с отдельной проверкой
coverage у центра; это не learned classifier и не евклидов nearest finite curve.

В [каталоге публикации](rays16_conditional_v1) находятся выбранные финальные
native EMA weights/config, registration и manifest/history. **Веса не встроены
в notebook**. Optimizer, RNG и `training.pt` исключены: **точный resume из demo
невозможен**; опубликованный EMA подходит для inference и нового обучения
с новым optimizer и каталогом артефактов. Состав и SHA256 указаны в
`PUBLICATION_MANIFEST.json`.

В `rays16_conditional_v1/students/` также опубликованы M0/M1/E0/E1 из полного
запуска notebook 13: по 200 updates, один paired seed 101. Это короткий пример,
а не официальное сравнение методов. На общем наборе 128 шумов × 16 targets
каждый ученик получил правильный запрошенный луч в 2048/2048 случаев.
Калибровка и validation-метрики сохранены рядом с весами; optimizer/RNG исключены.

## Conditional-дистилляция на 2 и 4 шага

`rays16_conditional_low_nfe_v1/` содержит ещё **24 финальных student EMA**:
M0/M1/E0/E1 × seeds 101/202/303 × NFE 2/4, по 1000 updates. Родитель — тот же
`rays16_conditional_v1/teacher/ema_model`, SHA `9bc0dbe39c4cb24c4d9b893bac084c89be66fdd7b29d6879fca786b4020d6ce1`.
Сохраняются native Safetensors/config, manifests, histories, train-only calibration,
registration и suite status. Optimizer/RNG исключены, точного resume нет.

[Таблица, ограничения и обе GIF](../reports/low_nfe_conditional_rays/REPORT.md).
В [notebook13](../../notebooks/13_conditional_rays_distillation.ipynb) меняются
`STUDENT_STEPS=2/4`, `UPDATES=1000`, `STUDENT_SEED=101/202/303`; всё обучение видно.
В `PUBLICATION_MANIFEST.json` указаны размеры и SHA256 каждого файла.

## Два обхода стены с непрерывным условием

`wall_navigation_conditional_v1/teacher/ema_model` — native `PointDenoiser`
с `conditioner="mlp"`, входом `[Axy, Bxy, wall_low_xy, wall_high_xy]` и 20 000
conditional CFM updates. Для одного условия представлены оба обхода;
route ID используется только при оценке. Teacher SHA:
`71d77760c37e6a053f324f2c810adb323439ed76ef51f09a57db43a5d2da9e79`.

Сеть предсказывает `[30,2]` внутренних координат пути. A/B добавляются из условия
на каждом шаге; точность концов задана конструкцией. Никакого исправления маршрутов
после генерации нет. Допустимость проверяется по всем сегментам с margin 0.035.
Сам teacher допускает столкновения: на validation успешны 97.39% путей при NFE32.

[Notebook14](../../notebooks/14_wall_navigation_distillation.ipynb) показывает
обучение teacher и M0/E0-учеников, native reload, интерактивный выбор сцен и две
разные анимации: FM-денойзинг всего пути и движение частицы по готовому пути.
Веса, wheel и outputs в notebook не встроены; selected EMA и история лежат отдельно.
Optimizer/RNG исключены из публикации, точный resume недоступен.

В `students_continuation_nfe4/` и `students_continuation_nfe2/` опубликованы M0/E0
по 1000 updates, один paired seed 161. `final_cohort_registration.json` перечисляет
точный состав, `solver_amendment_v1.json` фиксирует численное дополнение перед
повторным обучением обоих E0. Новая реализация Sinkhorn использует continuation
с теми же финальными epsilon, float64 и tolerance 1e−10; исходный `recheck` сохранён.
В `solver_attempt_v0/` оставлены исходный notebook, сбойный batch update 17 и
manifests прежних попыток. Они не входят в итоговую таблицу моделей.

`pairing_trace.json` содержит все 1000 сохранённых digest сцены/интервала/данных/шума;
они точно совпадают между M0/E0 внутри каждого NFE. Это метаданные для аудита,
без optimizer и состояний RNG. Строгий первичный evaluator требует исходные
training checkpoints; опубликованные EMA доступны для inference, notebook-оценки
и нового обучения. [Итоговые метрики и анимации](../reports/wall_navigation/REPORT.md).
Файлы и их SHA256 перечислены в `PUBLICATION_MANIFEST.json`.
