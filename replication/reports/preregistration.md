# Регистрация протокола до результатов

Дата: 15.10.2026. Основной пользовательский стенд этой сессии — unconditional
вращающиеся лучи, четыре CPU threads. На машине доступен Apple MPS;
первый sandbox-процесс его не видел, проверка вне sandbox подтвердила Metal.
MNIST запускается только после лучей, просмотра анимаций и замера бюджета.

## Rays CPU pilot v1

- Данные: исходная среда `rotating_rays`, четыре луча; 8192 train, независимые
  4096 validation и 4096 test. Метки не входят в генератор. Base data seed 41;
  split streams получаются стабильным SHA256.
- Общий учитель: FMMLP, 3 hidden layers ×128, sinusoidal time embedding 32,
  6000 обновлений, batch512, AdamW lr0.001/wd0.0001, warmup500, cosine до 10%,
  clip1, EMA0.999, seed41. Фиксируется **финальный EMA**, не лучший test.
- Ученики: точные копии одного teacher EMA. Предварительный бюджет: 500
  обновлений, batch/OT support128, lr0.0001, wd0.0001, warmup50, clip1,
  EMA0.99. До их итоговых запусков измеряются 100 итераций MSE и Sinkhorn
  после 20 warmup; при необходимости общий бюджет меняется до всей серии,
  с записью причины, без сравнения quality этих timing runs.
- Четыре режима M0/M1/E0/E1, seeds101/202/303: 12 парных обучений.
  rho0/0.10; lambda1; gamma1; epsilon=0.05×train-only median per interval.
  256 чистых anchors на интервал; calibration общая для tube/no-tube.
- Основная сетка32→16, Euler; shared t/H внутри batch. Anchors — interpolants.
  Teacher retarget всегда вычисляется из фактического perturbed anchor.
- Production OT: canonical KL Sinkhorn, сходящийся float64 log-domain solver,
  все self gradients. Причина выбора solver — независимая численная проверка,
  не downstream quality. Approximate GeomLoss не выбирается по результатам.
- Общие initial noise: final seed730001, validation730002. Evaluation streams
  и SWD независимы от data/time/noise/tube training streams. Никакого NFE в seed.
- Метрики лучей: независимые real-data MMD² и fixed-bank SWD, paired endpoint
  и node MSE до teacher32, actual NFE и training/sampling time. Предварительно
  выбранный главный MNIST-контраст E0−M0 **не заменяется** лучами.
- Teacher baselines: Euler8/16/24/32/64/128. Ученики — только их trained16 grid.
  Local diagnostics: interpolant, teacher rollout, student rollout; rho_eval
  0/.05/.10/.20. Дополнительно central derivative при zeta и zeta/2.
- Если разница teacher16 и teacher32 по SWD меньше0.005, стенд помечается как
  малочувствительный; отдельный заранее предусмотренный stress32→8 не меняет
  основную таблицу. Без выбора NFE по тому, где выиграет OT.
- Изображения/GIF: первый seed101, первые16 общих latents, одинаковые оси,
  реальные Euler states. Красный старт, фиолетовый финиш. Удачные точки не отбираются.
- Расширенные losses/SWD/joint/displacement и teacher-rollout/24→16 проверяются
  короткими механическими smoke-тестами; их результаты не выдаются за final matrix.

Это CPU-пилот на одном teacher. Три student seed позволяют показать парные
различия, но не доказывают универсальное ранжирование. Для подтверждения нужны
полный MNIST-протокол и дополнительные независимые teachers.

## MNIST MVP: реализованный план запуска

Конфиг `replication/configs/mnist_mvp.yaml`: unconditional 28×28 U-Net
32/64/128, 2 residual blocks на уровень, GroupNorm/SiLU, без attention/dropout,
time embedding128; teacher30k и student10k обновлений, batch128, EMA0.999.
Четыре метода × seeds101/202/303, target32→16. Train55000/validation5000
стратифицированно; official test10000 закрыт для tuning.

Независимый CNN выбирается только по real validation; gate99%. Main metric:
unbiased MMD² standardized FC128 features к real test, E0−M0 N16. Test evaluator
accuracy измеряется один раз после фиксации. MNIST-команды заблокированы до
успешного rays gate с анимациями; полное обучение требует CUDA или Apple MPS.
Лимиты: один accelerator process, суммарно6 GPU-hours, новые artifacts≤8GiB.
Если timing forecast превышает лимит, confirmatory suite не стартует без
заранее зарегистрированного одинакового сокращения бюджета.

## Поправка оборудования до MNIST-запусков

Первый процесс в песочнице не видел MPS. Проверка вне песочницы подтвердила
`mps_built=True`, `mps_available=True` и настоящее Metal-вычисление. CPU-пилот
лучей сохраняет зарегистрированный одинаковый бюджет. После него MNIST
оценивается на MPS по timing forecast и общему пределу 6 accelerator-hours.
Sinkhorn float64 переносится на CPU с проверяемым differentiable bridge;
network/optimizer работают на MPS float32. Полный MNIST пока не выполнен.

## Подтверждение CPU-бюджета до первого final student update

100 измеренных updates после20warmup: MSE0.001407s/update, canonicalSinkhorn
0.018701s/update (OT0.016651s/update). Измерения — timing-only seed919191,
без сравнения quality и без final seeds. Бюджет500updates×12runs приB128
подтверждён без изменений. Общий teacher —6000updates, финальный EMA.

## Техническое исправление solver до повторной всей серии

Gauge-null modes устраняются spectral pseudoinverse; canonical objective, epsilon и
marginal tolerance1e-10 сохранены. Предыдущая попытка invalidated полностью.
Новый timing100updates после20warmup, seed919192: {"mse": {"warmup_updates": 20, "measured_updates": 100, "seconds_per_update": 0.0013508133310824633}, "sinkhorn_endpoint": {"warmup_updates": 20, "measured_updates": 100, "seconds_per_update": 0.026871752496808766}}
Бюджет500updates/B128/4threads для всех12finalruns подтверждён без изменений.
