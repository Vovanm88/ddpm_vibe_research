# Изменения и ограничения

1. По прямому запросу пользователя первым обучаемым стендом служат вращающиеся
   лучи существующего проекта. Предложенный GMM реализован с аналитическим
   oracle и exact-OT тестами; он не подменяет обязательные анимации лучей.
2. Лучи обучаются на CPU; для MNIST доступен Apple MPS вне sandbox.
   Фактическая завершённость каждого этапа определяется manifest, а не наличием
   кода или конфига. Отсутствующие метрики — null/not_tested, не нули.
3. Новый пакет `src/recheck` использует общий корневой pyproject/lockfile;
   configs/source/reports/artifacts находятся под `replication/`. Это сохраняет
   существующие Torch/Diffusers компоненты, среды и визуализации проекта.
4. CPU-бюджет лучей отличается от полного MNIST-бюджета и явно зарегистрирован.
   Batch/OT support, updates, init, grids и RNG совпадают между методами серии.
5. Вместо неконтролируемой конечной GeomLoss annealing используется независимый
   converged float64 log-domain solver с envelope gradient. Canonical objective,
   epsilon и self terms сохраняются. Численные ошибки библиотечного parity
   и tolerance записаны в tests_report.json, а не скрываются clamp/nan_to_num.
6. Строгое воспроизведение авторского кода недоступно. Результаты этого стенда
   не являются проверкой чисел больших Z-Image/TripoSG моделей диплома.
7. Линейный interpolant заданной data/noise пары не считается индивидуальной
   траекторией маргинального FM-поля. Для rollout diagnostics строится настоящий ODE.

## Поправка оборудования до MNIST-запусков

Первый процесс в песочнице не видел MPS. Проверка вне песочницы подтвердила
`mps_built=True`, `mps_available=True` и настоящее Metal-вычисление. CPU-пилот
лучей сохраняет зарегистрированный одинаковый бюджет. После него MNIST
оценивается на MPS по timing forecast и общему пределу 6 accelerator-hours.
Sinkhorn float64 переносится на CPU с проверяемым differentiable bridge;
network/optimizer работают на MPS float32. Полный MNIST пока не выполнен.

8. Реально обнаружен дефект MPS32→CPU64 combined cast/copy в текущем Torch: данные
   и gradients обращались в ноль. Исправлено последовательными `.cpu().double()`
   и обратными cast/copy; независимый Metal/CPU gradient parity пройден.
   Calibration, evaluation и exact-OT проверены тем же способом. Это дефект
   выбранного runtime-пути, не вывод об авторском коде диплома.

9. Первая попытка student-серии остановлена строгим контролем сходимости OT.
   Отдельно найдено, что entrypoint student не применял declared CPU threads.
   Вся попытка (включая завершённые M0/M1) помечена invalidated и архивирована;
   в итоговые таблицы она не входит. После исправления вся группа переиграна заново
   из одного teacher; epsilon, marginal tolerance и зарегистрированные бюджеты не меняются.
