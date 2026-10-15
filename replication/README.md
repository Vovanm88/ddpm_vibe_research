# Проверка Wasserstein-дистилляции: сначала лучи, затем MNIST

Независимая реализация математического протокола из
[REPLICATION_SPEC.md](source/REPLICATION_SPEC.md), в ветке
`research/wasserstein-tube-replication`. Проверяется добавочный эффект
распределенческого loss и отдельно tube при сокращении числа Euler-шагов.
Исходные diffusion/reflow-эксперименты проекта сохранены.

Генератор unconditional: метки не входят в сеть, CFG отсутствует. Время
`t=0` соответствует данным, `t=1` — шуму; генерация использует `x ← x − H·v(x,t)`.
Основной протокол 32→16: два teacher microsteps по 1/32 против одного student
macrostep по 1/16. Все ученики инициализируются точной копией одного frozen
teacher EMA; проверяется совпадение хешей state dict.

## Запуск из корня репозитория

```bash
git clone --branch research/wasserstein-tube-replication --single-branch \
  https://github.com/Vovanm88/ddpm_vibe_research.git
cd ddpm_vibe_research
uv sync --extra replication --extra interactive --extra dev

uv run python -m recheck.cli doctor
uv run python -m recheck.cli verify-losses
uv run pytest tests/test_recheck*.py
```

В существующем рабочем Torch-окружении можно установить проект через
`uv pip install -e '.[replication,interactive]'`. Не переустанавливайте готовый
CUDA-стек ради пилота. Проверка `doctor` записывает фактическую доступность
устройств и версии. `device: auto` выбирает CUDA, затем Apple MPS, затем CPU.
На macOS GPU может быть скрыт ограничениями процесса: доступность MPS нужно
проверять из того же терминала, из которого запускается обучение.

На MPS сеть и optimizer работают в float32, сходящийся Sinkhorn — на CPU в
float64 через проверенный differentiable bridge. Это явное разделение устройств;
Metal не поддерживает float64. Время переноса также входит в стоимость метода.

## Обязательный пилот на вращающихся лучах

```bash
uv run python -m recheck.cli pilot-rays \
  --config replication/configs/rays_cpu.yaml --dry-run

uv run python -m recheck.cli pilot-rays \
  --config replication/configs/rays_cpu.yaml

# После прерывания, с тем же конфигом и output_dir:
uv run python -m recheck.cli pilot-rays \
  --config replication/configs/rays_cpu.yaml --resume
```

`rays_cpu.yaml`: четыре вращающихся луча, независимые 8192 train / 4096 validation /
4096 test; MLP с тремя hidden layers ширины 128 и sinusoidal time embedding 32.
Учитель — общий финальный EMA после 6000 обновлений. Исходный план учеников —
500 updates, batch/OT support 128, четыре режима × seeds 101/202/303:

| Режим | Попарный MSE | Дополнительный loss | Tube rho |
|---|---|---|---:|
| M0 | да | нет | 0 |
| M1 | да | нет | 0.10 |
| E0 | да | Sinkhorn endpoint | 0 |
| E1 | да | Sinkhorn endpoint | 0.10 |

До итоговой серии benchmark измеряет MSE и Sinkhorn после прогрева. Если бюджет
меняется, одинаковое новое число updates регистрируется для всех методов до
их сравнения. Значения training loss сами по себе не доказывают лучшую генерацию.

Результаты и анимации собираются из сохранённых метрик и фактических solver states:

```bash
uv run python -m recheck.cli report \
  --config replication/configs/rays_cpu.yaml
```

Смотрите [reports/REPORT.md](reports/REPORT.md), таблицы отдельных seed, paired
differences, quality/fidelity/cost и GIF в каталоге отчёта. В анимациях **красный —
старт, фиолетовый — финиш**; используются первые зарегистрированные latents, без
ручного отбора удачных примеров. Основные сравнительные графики имеют общие оси.
Независимые real-data MMD²/SWD отделены от paired MSE относительно teacher32.
Original N32 и original N16 остаются обязательными baseline.

MNIST-команды проверяют `reports/rays_gate.json`: нужны завершённые M0/M1/E0/E1
на трёх seed, анимации и актуальный хеш алгоритма. Изменения алгоритма делают
gate устаревшим и требуют повторной проверки лучей. Сам gate не означает,
что преимущество OT установлено.

## Jupyter: обучение на поверхности

[10_wasserstein_tube_rays.ipynb](../notebooks/10_wasserstein_tube_rays.ipynb)
сам устанавливает исходный пакет из этой ветки и отдельно загружает teacher.
В его ячейках видны teacher microsteps, tube-retargeting, teacher-only calibration,
признаки endpoint/displacement/joint, student forward, loss, `backward`, clipping,
`optimizer.step`, EMA и сохранение состояния. Нет скрытого Trainer.

По умолчанию notebook обучает **четыре новых student, один paired seed, 200 updates**.
Это отдельный интерактивный пример, а не замена CLI-серии на трёх seed. Меняйте
`DISTRIBUTION_FAMILY`, `REPRESENTATION`, `LAMBDA_W`, rho или target steps, сохраняя
новую абляцию в новом `OUTPUT_DIR`. Есть GIF и Play/Slider с шагом один для
реальных Euler-узлов. Веса, optimizer checkpoints, PNG/SVG и GIF сохраняются
отдельно; notebook не содержит встроенных wheel или бинарных весов.

## Как именно устроен loss

Одна OT-точка — полный state, развёрнутый в `[B,D]`; изображение не превращается
в облако отдельных пикселей. Для общего фактического anchor `a`:

```text
pair = mean((student_endpoint - teacher_endpoint)^2)
endpoint     = vec(endpoint) / sqrt(D)
displacement = vec(endpoint - a) / sqrt(D)
joint        = concat(sqrt(gamma)*vec(a), vec(endpoint-a)) / sqrt(D)
total        = pair + lambda_w * distribution_loss
```

У joint размер признаков `2D`, но нормирующий знаменатель остаётся `sqrt(D)`.
Sinkhorn использует squared Euclidean cost и KL к произведению равномерных
весов; дивергенция вычитает обе self-cost. Градиент student self-term проходит
через оба аргумента. Сходящийся float64 solver проверяется против отдельной
GeomLoss-реализации; его ошибки маргиналей не маскируются clamp/nan replacement.

Для SWD одни направления используются student и teacher; loss равен
`P * mean((sort(phi_student @ theta)-sort(phi_teacher @ theta))^2)`, без корня.
Это нормированный sliced W2² training loss. Evaluation SWD — отдельно
определённый fixed-projection sliced W1, поэтому их численные значения не
следует напрямую сравнивать.

До student training на 256 **чистых train-only** anchors для каждого интервала
и representation фиксируется `epsilon = .05 * max(median_teacher_cost, 1e-8)`.
Таблица одна для tube/no-tube и не пересчитывается по student. В tube-режиме
`a_used = a + rho*H*eta`, `eta ~ N(0,I)` на координату; teacher target заново
интегрируется из `a_used`. Повторное использование чистого target запрещено.

Парные методы используют одинаковые data/time/noise/tube streams. SWD, previews,
evaluation и bootstrap имеют отдельные RNG. Изначальный evaluation noise одинаков
между методами и NFE, без числа шагов в seed. Checkpoints сохраняют optimizer,
EMA и RNG; `--resume` проверяет контракт запуска, а не молча меняет эксперимент.

## MNIST после лучей: подготовка и полный план

```bash
# Только после актуального успешного rays gate и просмотра анимаций:
uv sync --extra replication --extra interactive --extra mnist --extra dev

uv run python -m recheck.cli prepare-mnist \
  --config replication/configs/mnist_mvp.yaml

uv run python -m recheck.cli train-evaluator \
  --config replication/configs/mnist_mvp.yaml --dry-run
uv run python -m recheck.cli train-evaluator \
  --config replication/configs/mnist_mvp.yaml

uv run python -m recheck.cli train-teacher \
  --config replication/configs/mnist_mvp.yaml --dry-run
uv run python -m recheck.cli train-teacher \
  --config replication/configs/mnist_mvp.yaml
uv run python -m recheck.cli teacher-gate \
  --config replication/configs/mnist_mvp.yaml
uv run python -m recheck.cli calibrate \
  --config replication/configs/mnist_mvp.yaml
uv run python -m recheck.cli benchmark \
  --config replication/configs/mnist_mvp.yaml

# Сначала оцените полный бюджет и зарегистрируйте необходимое сокращение.
uv run python -m recheck.cli distill-suite \
  --config replication/configs/mnist_mvp.yaml --dry-run
uv run python -m recheck.cli distill-suite \
  --config replication/configs/mnist_mvp.yaml
uv run python -m recheck.cli evaluate-suite \
  --config replication/configs/mnist_mvp.yaml
uv run python -m recheck.cli report \
  --config replication/configs/mnist_mvp.yaml
```

Разделение официального MNIST train — детерминированное стратифицированное
55000/5000. Его индексы и хеши лежат в
`replication/artifacts/mnist_mvp_v1/data/mnist_split_manifest.json` и
`mnist_split_indices.pt`. Test 10000 закрыт для выбора конфигурации. FM-нормировка
`2*pixel-1`, без augmentation. Оценщик получает `(raw_FM+1)/2`, затем единый clamp;
доля и максимальное отклонение пикселей за диапазон измеряются до clamp.

Учитель и ученик — одинаковые U-Net 28→14→7, channels 32/64/128, два residual
blocks на уровень, GroupNorm/SiLU, time embedding 128, без attention/BatchNorm/
dropout и без H/class conditioning. План: teacher 30000 updates; M0/M1/E0/E1
по 10000 updates на seeds 101/202/303; batch 128, финальная EMA 0.999.

Независимый Conv32→Conv64→FC128→logits10 оценщик обучается только на real train,
выбирается по real validation и должен достичь accuracy ≥99%. Калибровка признаков
использует фиксированные train-only mean/std, std floor 1e-6; bandwidth — медиана
off-diagonal squared distances на фиксированных 2000 train features. Индексы
сохраняются. Accuracy классификатора на test измеряется один раз после фиксации.

Главный заранее выбранный контраст — **E0−M0 по `mmd2_mnist_features_real`, N16**.
Unbiased MMD² может быть отрицательной; она не обрезается. Frequency/coverage,
diversity, nearest train neighbor, fidelity и стоимость показываются отдельно.
Три seed дают пилотную оценку устойчивости при одном teacher, а не универсальный
вывод о ранжировании методов.

Протокол ограничивает новые артефакты 8 GiB и суммарный accelerator budget шестью
часами; одновременно работает один accelerator training process. До финальных
серий требуется timing forecast вместе со стоимостью teacher, evaluator и оценки.
Если исходный план не помещается, регистрируется одинаковый сокращённый бюджет
или выполняется только явно обозначенный smoke/pilot. Ни код команд, ни наличие
конфига не означают, что полный MNIST-план уже выполнен: фактический статус
определяют manifests и итоговый отчёт.

## Дополнительные проверки и устройство проекта

GMM — отдельный `pilot-toy --config replication/configs/toy.yaml`: три равновесные
гауссианы, центры `(-.6,0),(.6,0),(0,.65)`, sigma .15. В исходном протоколе учитель
10000 updates, ученики 3000; это отдельный бюджет. Аналитическое FM-поле проверяет
решатель и learned teacher. Выполнение GMM не подразумевается выполнением лучей.

Расширения SWD, displacement, joint, teacher-rollout anchors, 32→8 и точная
неравномерная группировка 24→16 запускаются отдельно с явными конфигами. Контроль
24→16 использует 24 teacher calls по группам `[2,1]×8`, а не 32 teacher calls под
другим названием. H-conditioning и условный MNIST не включены в основной MVP.

```text
src/recheck/          Torch models, data, schedules, FM/distillation, OT, evaluation
replication/configs/  строгие YAML-конфиги; неизвестные параметры отклоняются
replication/source/   исходное ТЗ и хеши доступных файлов
replication/artifacts/ локальные checkpoints, RNG, метрики и траектории
replication/reports/  source audit, preregistration, deviations, тесты и результаты
replication/demo/     отдельно опубликованные небольшие модели/результаты
notebooks/10_*.ipynb  лучи: открытый student training и интерактивная диагностика
notebooks/11_*.ipynb  MNIST: полный CLI-протокол и открытые image training loops
```

Локальные training outputs и MNIST не встраиваются в notebook. Публикуемые demo
артефакты лежат отдельно. Таблицы и plots пересобираются из сохранённых данных;
ссылка на конкретный run важнее выводов из одной иллюстрации. Аудит доступного PDF и границы проверки
непредоставленного авторского кода, изменения протокола и ограничения находятся в
[source_audit.md](reports/source_audit.md), [preregistration.md](reports/preregistration.md)
и [deviations.md](reports/deviations.md).

Ноутбуки в Colab для этой ветки: [лучи](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/research/wasserstein-tube-replication/notebooks/10_wasserstein_tube_rays.ipynb) · [MNIST](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/research/wasserstein-tube-replication/notebooks/11_mnist_wasserstein_tube.ipynb).
