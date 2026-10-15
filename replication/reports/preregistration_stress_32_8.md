# Регистрация дополнительного stress32→8 до его запуска

Дата: 15.10.2026. Подготовлено `2026-10-15T03:39:35.350784+00:00`. Статус: **prepared_not_started**.
Ни одно обучение ученика этого расширения и ни одна оценка его quality ещё не запущены.
Основной конфиг `rays_cpu.yaml` не меняется; SHA-256: `7d2dabf0fe8b6130c72b1d26b352139ead52f726d92443930e9de26d1f016806`.

Это заранее предусмотренный в основном протоколе дополнительный stress-тест на
вращающихся лучах. Он не заменяет основную таблицу32→16 и главный MNIST-контраст.
Его возможный запуск мотивирован зарегистрированным правилом малой чувствительности
teacher16/32 (SWD-разница <0.005), а не подбором числа шагов по выигрышу OT.
Конфигурация stress зарегистрирована до результатов учеников этого расширения.

## Общий замороженный teacher

Повторно используется точный **финальный EMA** общего teacher из `replication/artifacts/rays_cpu_pilot_v1/teacher`:
`5fcbe5ceee197bf2a257b1347586335d7feb8a15789ba406f918a077cf3e6930`. Новое обучение teacher запрещено в этой серии;
число новых teacher updates равно0. Все файлы скопированы без изменения в
`replication/artifacts/rays_stress_32_8_v1/teacher`; исходный run ID и manifest сохранены буквально.
Происхождение и SHA каждого файла находятся в `replication/artifacts/rays_stress_32_8_v1/teacher_reuse.json`.
Это snapshot общего teacher, а не новый независимо обученный teacher.

## Фиксированный бюджет и контраст

- Конфиг: `replication/configs/rays_stress_32_8.yaml`; config SHA-256 `4727422be6c58b193a43c4a38866b36bcfd5b6f457c1906dd7f0851537fa62d6`.
- Teacher Euler32 → student Euler8: четыре вызова teacher на локальный target,
  восемь macro-step ученика, положительная ширинаH=1/8. Новые ученики начинают
  с исходного teacher, не с обученного16-step student.
- Режимы M0/M1/E0/E1, seeds101/202/303:12 новых обучений, каждый500updates,
  batch/micro-batch/OT support128, lr1e-4, wd1e-4, warmup50, EMA0.99, clip1.
- Данные, архитектура, optimizer, epsilon-alpha0.05, lambda1, gamma1, tube0/0.10,
  solver reference, число threads4 и все правила обучения совпадают с основной
  серией. Состояния interpolant; teacher retarget всегда из реального tube-anchor.
- Калибровка **новая для сетки32→8**, train-only256anchors на интервал,
  общая для tube/no-tube. Перенос epsilon-калибровки32→16 запрещён.
- Final noise seed730001, validation730002, reference730003. Initial noise одинаков
  между методами и всеми NFE; конфигурация и seed не меняются по результатам.
- Отдельная таблица: frozen original8, original32 и исходные дополнительные
  NFE8/16/24/64/128; ученики оцениваются на trained8 grid.
- Метрики: independent real-data MMD² и fixed1024-bank SWD; paired fidelity к
  teacher32, local/rollout/tube/Jv diagnostics; sample-std и paired CI по seed.
  Любой эффект ограничен этим teacher и маленьким CPU-стендом.
- GIF: первый зарегистрированный seed101 и первые16 общих latents, реальные
  сохранённые Euler states; общий масштаб; красный старт, фиолетовый финиш.
- Новые артефакты: `replication/artifacts/rays_stress_32_8_v1`. CPU; платные ресурсы не используются.

## Команды после решения root выполнить расширение

Не использовать `pilot-rays`: snapshot уже содержит общего teacher, и эта команда
пыталась бы обучать его заново. Последовательность:

```bash
uv run python -m recheck.cli teacher-gate --config replication/configs/rays_stress_32_8.yaml
uv run python -m recheck.cli calibrate --config replication/configs/rays_stress_32_8.yaml
uv run python -m recheck.cli distill-suite --config replication/configs/rays_stress_32_8.yaml
uv run python -m recheck.cli evaluate-suite --config replication/configs/rays_stress_32_8.yaml
uv run python -m recheck.cli report --runs replication/artifacts/rays_stress_32_8_v1 --config replication/configs/rays_stress_32_8.yaml --output replication/reports/stress_32_8/REPORT.md
```

Сохранённый teacher manifest описывает исходный training run, поэтому его исходный
config/run ID намеренно не переписаны под новую серию. Студенты и их калибровка
будут иметь новый derived config hash. Невыполненные результаты остаются отсутствующими.
