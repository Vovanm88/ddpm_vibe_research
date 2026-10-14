# Выполненная проверка

Проверено 14 октября 2026 года на CPU в локальном окружении Python 3.11.16, PyTorch 2.14.1, Diffusers 0.39.0, NumPy 2.4.6, SciPy 1.17.1 и Matplotlib 3.11.2. Версии конкретных исследовательских запусков сохраняются в `manifest.json`.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

Актуальный полный прогон v0.2.0: **160 passed in 10.19s**.
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

`outputs/` содержит локальные generated artifacts и исключён из Git. В репозитории сохранены recipes, исходники, тесты и изображения демонстрации в `docs/assets/`.

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
`examples` и `scripts`. Дополнительно проверены checksum и версия встроенных
wheel, точное совпадение пакета с исходниками и отсутствие сохранённых outputs.

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
