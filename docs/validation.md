# Выполненная проверка

Проверено 14 октября 2026 года на CPU в локальном окружении Python 3.11.16, PyTorch 2.14.1, Diffusers 0.39.0, NumPy 2.4.6, SciPy 1.17.1 и Matplotlib 3.11.2. Версии конкретных исследовательских запусков сохраняются в `manifest.json`.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

Актуальный полный прогон v0.3.1: **217 passed in 17.30s**.
Версия v0.3.0 отдельно прошла **171 passed in 12.95s**.
Версия v0.2.0 отдельно прошла **160 passed in 10.19s**.
Исходная версия v0.1.0 отдельно прошла **139 passed in 22.30s**. Ошибок и пропущенных тестов нет. Это проверка корректности реализации и воспроизводимости, а не оценка сходимости всех исследовательских рецептов.

## Что покрыто

- Denoiser сохраняет формы 1D/2D/3D-точек и последовательностей `(L,D)` для MLP и transformer. Параметры проекций существуют до optimizer; элементы батча не становятся общими attention tokens.
- Categorical и continuous conditioners имеют явное unconditional состояние. CFG с scale=0/1/2 согласован с unconditional/conditional prediction; condition dropout действительно убирает условие.
- Известные targets `epsilon`, `sample` и `v_prediction` восстанавливают исходный x₀. Cosine cumulative α монотонна. Терминальный t=0 и редкие inference timesteps не дают NaN; perfect oracle достигает точного чистого состояния через DDPM/DDIM.
- Rectified flow использует правильный линейный мост и velocity target. Семплирование DDPM/DDIM/flow поддерживает все заявленные формы, историю реальных timesteps и фиксацию endpoints.
- Sampling и field queries восстанавливают train/eval режим, включая ошибки. Save/load Diffusers pipeline воспроизводит seeded DDIM samples.
- MSE/Huber обновляют параметры; EMA проверена численно. Resume на границе эпох воспроизводит веса, EMA, историю и RNG непрерывного запуска точно. Проверены отклонение несовместимых scheduler/trainer config и изменённых данных; архитектура модели также проверяется перед загрузкой checkpoint.
- Среды генерируются локальными seeded RNG. Reference navigation routes проверены в 1D/2D/3D; учитываются пересечения **отрезков между waypoint**, тонкие стены, границы и недостижимые 1D-сцены. Loss пересечения стен имеет конечные gradients.
- Визуализация проверяет координаты плотности, нормировку 1D KDE, magnitude colors, nonconsecutive labels, 3D slices, стенки, SVG/PNG/HTML и GIF.
- Восстановление потенциала проверено на conservative и rotational полях, масках и несвязных областях. Диагностика residual/curl отделяет скалярную проекцию от точного потенциала.
- Distribution metrics замечают перепутанные классы даже при одинаковой общей marginal distribution. Navigation metrics замечают пересечение стены между свободными waypoint и смещённые endpoints.

## CLI и учебные примеры

Интеграционный тест выполняет настоящий короткий цикл на Gaussian mixture 1D: две эпохи `v_prediction`, checkpoint и EMA-export, seeded conditional sampling, SVG-графики, KDE, vector field, potential report, повторный report и resume до третьей эпохи. Он проверяет сохранённые параметры sampling и точное совпадение экспортированных весов с EMA checkpoint.

Дополнительно проверены sweep `one_hot`/`embedding`, environment по умолчанию, запрет случайного перезаписывания training run, неправильное число sampling steps, nonfinite/отрицательный guidance, ошибки condition file, взаимоисключающие condition arguments и неизвестные YAML sections. В sweep-тесте обучение настоящее, а тяжёлый report заменён лёгким callback; полный report проверяется отдельным тестом.

Все code cells `notebooks/01_rays_research.ipynb` и `notebooks/02_navigation.ipynb` выполнены в CPU-окружении; также выполнен пример пользовательского расширения conditioner/loss. Эти учебные запуски короткие и подтверждают работоспособность API. Они не доказывают хорошую генерацию или collision-free navigation после обучения.

## Дополнительный обученный пример лучей

Отдельно завершён CPU-run `outputs/rays_baseline`: 80 эпох, 4096 reference samples четырёх классов, MLP с hidden dimension 128 и четырьмя слоями, `embedding` conditioner, `v_prediction`, cosine schedule с 1000 training timesteps, EMA 0.99. Evaluation использует 1024 generated samples, DDIM с 100 шагами, guidance 1.5 и seed 42.

Команда для такого же числа эпох:

```bash
diffusion-lab train --config configs/rays.yaml --epochs 80 --output outputs/rays_reproduction
```

Фактические значения из `outputs/rays_baseline/report/metrics.json`:

| Метрика | Значение |
|---|---:|
| Sliced Wasserstein, все классы | 0.0534510694 |
| Sliced Wasserstein, класс 0 | 0.0569137169 |
| Sliced Wasserstein, класс 1 | 0.0683521516 |
| Sliced Wasserstein, класс 2 | 0.0525881772 |
| Sliced Wasserstein, класс 3 | 0.0674573706 |
| Ошибка среднего | 0.0229903068 |
| Ошибка стандартного отклонения | 0.0560778007 |

Это один seeded toy-distribution run и сравнение с reference dataset, использованным при обучении. Он демонстрирует обучаемость основного примера; независимый test split, несколько training seeds и доказательство сходимости не выполнялись. GPU/MPS и длинное обучение navigation пока не проверены. Фиксация A/B endpoints не гарантирует обхода стен: для generated navigation routes нужно смотреть collision-free fraction и endpoint error.

`outputs/` содержит локальные generated artifacts и исключён из Git. В репозитории сохранены recipes, исходники, тесты, изображения в `docs/assets/` и новый обученный запуск в `demo/`.

## Короткие flow и navigation прогоны

Дополнительно завершены настоящие CLI-запуски по три эпохи:

| Прогон | Обучение / evaluation | Полученный результат |
|---|---|---|
| `outputs/flow_smoke` | Rectified flow + Huber; 4096 лучей, 1024 generated samples | Sliced Wasserstein 0.0759183127 |
| `outputs/nav_smoke` | Transformer + continuous start/goal/walls; 1024 reference routes, 128 generated routes | Collision-free fraction **0.0**; endpoint error **0.0**; mean path length 30.4035 |

Navigation smoke подтвердил обучение, checkpoint, генерацию, geometric losses, фиксацию A/B и экспорт графиков. **После трёх эпох модель ещё не научилась обходить стены: все 128 generated routes пересекают препятствия или выходят за границы.** Нулевой endpoint error получен явной фиксацией endpoints и не является доказательством обучения навигации. Reference routes строит planner с collision checks; generated routes оцениваются отдельно.

Для navigation также выполнена команда `sample` с сохранением истории и GIF. Для получения хороших learned routes требуется дальнейшее обучение и подбор рецепта по collision-free fraction, длине и гладкости пути. Flow smoke и 80-эпоховый diffusion baseline имеют разный бюджет; эти числа не являются сравнением методов на равных условиях.

## Сохранность исходника

В оригинальном `/Users/vvmelnikov/Downloads/diffusion_modeling_1.ipynb` ничего не изменено. У очищенной копии `notebooks/original_diffusion_modeling_1.ipynb` все 51 исходные ячейка имеют те же типы, тексты и порядок. Удалены outputs и сохранённое состояние виджетов; исходный notebook целиком не исполнялся.

SHA-256 исходного файла повторно проверен:

```text
9697400f3d39bd8b4aa8e226d0ace4492ea2fb4551712cf36994d219a03dfaf2
```

Конкретные обнаруженные ошибки с номерами исходных ячеек описаны в `docs/notebook_audit.md`.

## Переносимые ноутбуки и PyTorch-интерфейс (v0.2.0)

После добавления standalone notebooks и notebook API полный тестовый прогон:
**160 passed in 10.19s**. Ruff и форматирование проверены для `src`, `tests`,
`examples` и `scripts`. Этот исторический прогон v0.2 проверял также checksum
встроенного пакета. В текущих ноутбуках бинарных вложений нет.

`03_standalone_quickstart.ipynb` и `04_change_components.ipynb` целиком выполнены,
включая установочную ячейку, вне репозитория в отдельной Python 3.11 среде.
Зависимости предварительно установлены из локального cache, установщик выполнен
с `PIP_NO_INDEX=1`; импорт идёт из установленного wheel в отдельной среде, без
editable install и локальной папки `src`. Запуск в облачном Colab и на GPU не
проводился. Новому окружению нужен интернет для отсутствующих зависимостей.

Основной quickstart содержит явный DataLoader/model/scheduler/AdamW и 20 эпох
ручного `backward → step`, conditional sampling, поля, Diffusers export,
torch optimizer/RNG checkpoint и продолжение ещё на две эпохи. Метрика этого
короткого запуска лучей: sliced Wasserstein 0.12620758.

Ноутбук компонентов выполняет 16 вариантов с реальным обучением: backbone и
своя модель, MSE/Huber/Charbonnier, one-hot/embedding/continuous/Fourier, AdamW/Adam,
радиусный penalty, epsilon/v prediction, rectified flow, 1D/3D и navigation.
Основные сравнения — три эпохи; epsilon/v — по 20, navigation — две.
Все объекты и training loop доступны напрямую. Короткая navigation снова даёт
collision-free fraction 0 при endpoint error 0; успех обучения обходу стен не
заявляется. Обновление класса conditioner требует создания нового model/optimizer.

Опциональный `ResearchExperiment` также проверен: дополнительные эпохи,
conditions, инварианты состояния sampling/EMA и точный portable save/load/resume
для diffusion и flow, включая собственную навигационную сцену. Он не требуется
для основных примеров с ручным PyTorch loop.

## Обученная демонстрация и интерактивные ноутбуки (v0.3.0)

Новый запуск `examples/train_visual_demo.py`: 120 эпох, 1920 optimizer steps,
22.75 секунды на CPU. Reference training seed 123; отдельная контрольная выборка
seed 124; generated samples seed 127. На одинаковых 128 проекциях с seed 123
SWD меняется **0.658185 → 0.057220**. Промежуточные snapshots: эпохи 0/1/5/20/60/120.
Веса EMA, optimizer/RNG checkpoint, данные и графики опубликованы в `demo/`.
Это один запуск синтетической задачи; остальные рецепты не оценивались на
одинаковом бюджете. У поля при t=349 residual 0.78%, ненулевой curl, поэтому
восстановленный U подписан как приблизительная проекция.

В текущем `06_trained_interactive_demo.ipynb` пакет устанавливается из GitHub,
Safetensors/configs/reference берутся из отдельных файлов `demo/` — локального
checkout или через clone. Ноутбуки 03/04/05/06 не содержат wheel или весов,
сохранённых outputs и widget state. Для установки нужны Python 3.11+, Git и
интернет. Облачный Colab и GPU этим запуском не проверены.

Финальный лёгкий `06` целиком выполнен вне проекта после публикации GitHub
commit `fa6e2a9`. В отдельном Python 3.11 окружении зависимости присутствовали,
сам `diffusion-lab` заранее удалён. Первая ячейка установила пакет из GitHub;
`direct_url.json` подтвердил URL и commit. Следующая ячейка клонировала
репозиторий и загрузила отдельные `demo/pipeline` и `demo/reference.npz`.
Commit скачанных весов совпал с commit установленного пакета. Все ячейки и
изменение controls, Generate/Frame завершились успешно.

Проверены реальные callbacks смены condition, timestep, CFG и типа поля,
расчёт потенциала, Generate с изменёнными seed/condition и 101 состояния
DDIM-денойзинга. Смена Frame использует сохранённую историю и не запускает
новую генерацию. Дополнительные widget tests проверяют 1D/2D/3D-поля,
навигационные endpoints, flow и ошибки, сохранность RNG/train mode, разделение
отображения стрелок и raw vectors для U. Полный прогон: **171 passed in 12.95s**;
Ruff и форматирование прошли для `src`, `tests`, `examples`.
Пример обучения отдельно завершил одну эпоху из копии исходников без `.git`:
checkpoint и отчёт сохранены, revision записан как null, hash скрипта сохранён.

Автономный Plotly explorer проверен на всех 24 последовательных комбинациях
переключения condition и timestep. Inline explorer содержит 12 настоящих
полей обученной модели; syntax/IDs, оба controls, сохранение и восстановление
состояния проверены с DOM/D3 mock. Эти проверки взаимодействий не заменяют
визуальный browser QA. Статические научные рисунки просмотрены после рендера.

## Полный timestep, отключение CFG, symlog и траектории (v0.3.1)

В Jupyter и inline/external HTML есть Unconditional, отдельно от класса 0,
целочисленные `t=0..999` с шагом 1 и выключатель CFG. OFF использует scale 1
и сохраняет условие; scale 0 соответствует безусловному предсказанию.
В Jupyter и PNG/SVG `plot_potential(scale="symlog")` отображает отрицательные
U и ноль без изменения данных. Новые tests проверяют эти режимы, all-zero U,
порог шкалы, формы 1D/2D/3D и сохранность gauge/RNG/model train mode.

Offline HTML содержит точный Float32 кэш всех 1000 timesteps и пяти режимов
на сетке 17×17. Gzip roundtrip точен. Node/native DecompressionStream + DOM mock
проверили decoder, controls, CFG OFF/0/1.5 и Unconditional. 20 сравнений кэша с
pipeline дали максимальную разницу округления 9.54e-7. Inline версия использует
сетку 11×11 и int16-квантование с известной ошибкой: базовые компоненты имеют
погрешность не более 0.0061% от пика кадра. Проверены 37 взаимодействий,
legacy saved state и 80 CFG-сравнений с pipeline; ошибка укладывается в
заявленную границу. Временные состояния не интерполируются.

GIF `denoising_trajectories` — настоящая DDIM history после 120 эпох, 16 частиц,
100 шагов, CFG 1.5, seed 127; noise/sample endpoints показаны отдельно.
`navigation_trajectory` — отдельно подписанный A* reference route со стенами,
collision-free=true. Это не результат обучения navigation.
Радуга означает исходный индекс пути; границы осей, цвета сегментов и colorbar
фиксированы между кадрами. Сохранены GIF, PNG/SVG, NPZ history и metadata.
24 animation tests проверяют исходные координаты/endpoints, цвета при разной
длине следа, 1D/2D/3D, изменение GIF-кадров и invalid inputs.
CLI `sample --gif` дополнительно выполнен на trained pipeline с восемью шагами:
обычный GIF и радужные траектории сохранены. Полный прогон: **217 passed in 17.30s**;
Ruff check/format прошли. Легенда GIF вынесена за область данных и проверена
после рендера, чтобы не закрывать стартовые точки.
