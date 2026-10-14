# Визуализация и потенциальные поля

Визуализаторы работают с точками `[N,D]`, `D=1,2,3`; траектории имеют форму
`[N,T,D]`. Функции Matplotlib возвращают `(figure, axes)` и не вызывают
`plt.show()` автоматически. Jupyter explorers возвращают `ipywidgets.VBox`.
Классы могут иметь любые значения: например, метки `4` и `9` остаются `4` и `9`.

```python
import numpy as np
from diffusion_lab.visualization import (
    Grid, evaluate_field, plot_vector_field, recover_potential,
    plot_potential, save_figure,
)

grid = Grid.from_bounds([(-2, 2), (-1, 1)], resolution=(61, 39))
field = evaluate_field(lambda p: -p * np.array([1, 2]), grid)
fig, ax = plot_vector_field(grid, field, stride=2, streamlines=True)
save_figure(fig, "outputs/field")  # field.png + field.svg
result = recover_potential(grid, field)
fig, ax = plot_potential(grid, result)
save_figure(fig, "outputs/potential")
```

## Координаты, стрелки, плотность

`Grid.axes` всегда идут в порядке `(x,y,z)`, а массивы имеют форму `(nz,ny,nx)`
или `(ny,nx)`. `grid.points` содержит координаты `[N,D]`; `grid.reshape()`
возвращает массив на сетке без ручных транспонирований. Несимметричная сетка
`resolution=(61,39)` имеет форму `(39,61)`. Оси должны строго возрастать.

`evaluate_field(callable, grid, batch_size=4096)` передаёт NumPy-точки батчами и
принимает NumPy или torch-результат той же формы. `pipeline.vector_field()`
принимает такие точки напрямую. При фиксированных времени и условии:

```python
vectors = evaluate_field(
    lambda p: pipeline.vector_field(p, timestep=350, conditions=0, kind="score"),
    grid,
)
```

В `plot_vector_field(..., normalize=True)` длина стрелок фиксирована ради
читаемости, а цвет показывает **исходную** норму вектора. При `normalize=False`
сохраняются относительные длины; общий масштаб отображения указан в заголовке.
`streamlines=True` доступен для равномерной 2D-сетки.

`plot_dataset` рисует точки в 1D/2D/3D; `plot_trajectories` рисует целые пути с
метками A/B. Стены задаются как `[W,2,D]`: нижний и верхний углы каждого бокса.
`draw_walls` добавляет их к существующим осям, включая 3D.

`estimate_density` и `plot_density` используют нормированную Gaussian KDE;
это оценка плотности по выборке, а не аналитическая плотность модели.
3D-визуализация показывает срез полной плотности при фиксированном `z`,
**не** маргинальную плотность `(x,y)`. `slice_index` выбирает срез.
Для вырожденной выборки с сингулярной ковариацией нужно явно выбрать проекцию
меньшей размерности. `labels=` строит отдельную условную KDE для каждого класса.

## Score и denoising

Для DDPM/DDIM поле `kind="score"` означает
`−epsilon_prediction / sqrt(1−alpha_bar)` — оценку `∇ log p_t(x)`.
`kind="denoising"` означает `predicted_x0 − x_t`; это другое поле с другой
нормой. `kind="epsilon"` возвращает сам предсказанный шум. Запрос поля не
добавляет новый шум к точкам сетки.

Для rectified flow доступны `velocity` и `denoising`: в принятой конвенции
`velocity = dx/dsigma`, а направление денойзинга равно `−velocity`.
У flow нет автоматически определённого score; API отклоняет такой запрос.

## Восстановление U

Конвенция едина: **`F = −∇U`**. В 1D используется трапециевидное интегрирование.
В 2D/3D разреженный least squares подбирает разности U между соседними узлами
к среднему значению соответствующего компонента F на ребре. Поэтому результат
для произвольного нейросетевого поля — проекция на градиентное поле.

Поля результата:

| Поле | Значение |
|---|---|
| `potential` | U на сетке; недоступные узлы — NaN |
| `relative_residual` | `||G U + F_edges|| / ||F_edges||` |
| `residual_rms` | RMS невязки на рёбрах, в единицах F |
| `curl_rms` | RMS ротора по доступным элементарным граням |
| `relative_curl` | `curl_rms × средний размер области / field_rms` |
| `is_conservative` | Уложились ли обе относительные диагностики в допуски |
| `component_count` | Число связных компонент доступной сетки |
| `solver_converged` | Сошёлся ли численный решатель |
| `message` | Интерпретация и ограничения результата |

По умолчанию допуски равны `0.05`. Например, для `F=(−x,−2y)` восстанавливается
`U=x²/2+y²+C`, невязка близка к нулю и ротор равен нулю.
У `F=(−y,x)` ротор равен `2`, относительная невязка велика, а рассчитанное U
помечается как **projection only**. Галерея сохраняет фактические значения в JSON.

`mask=True` означает доступный узел. Каждая связная компонента имеет собственную
калибровку: U=0 в её первом узле. Сравнивать абсолютные уровни U между такими
компонентами нельзя. Маска проверяет узлы: стенку тоньше шага сетки можно
пропустить, поэтому разрешение должно учитывать геометрию препятствий.

Малый локальный ротор не доказывает существование глобального потенциала:
вокруг отверстий возможна циркуляция. Невязка учитывает циклы графа, но грубая
сетка всё равно ограничивает вывод. `is_conservative` — численная диагностика
на данной сетке, не математическое доказательство.
Даже аналитическое градиентное поле возле углов прямоугольных стен может
показать заметный численный ротор на конечной сетке: проверьте зависимость
диагностики от разрешения, прежде чем делать вывод о самом поле.

U может быть отрицательным и имеет произвольную аддитивную константу. Это не
нормированная плотность. Только для настоящего консервативного score
`U=−log p_t+C`; применять `exp(−U)` к любому denoising или flow-полю как к
плотности некорректно.

## Живые графики в Jupyter / Colab

Готовый пример с весами после **120 эпох** —
[`06_trained_interactive_demo.ipynb`](../notebooks/06_trained_interactive_demo.ipynb)
([открыть в Colab](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/main/notebooks/06_trained_interactive_demo.ipynb)).
Первая ячейка устанавливает пакет из GitHub. Обученная модель загружается из
`demo/pipeline` в локальном checkout; в отдельном Jupyter / Colab ноутбук
клонирует репозиторий с GitHub. Для загрузки нужен интернет. В
[`05_live_visualizations.ipynb`](../notebooks/05_live_visualizations.ipynb) такой же
интерфейс загружает локальный запуск из `outputs/visual_demo_rays`. Для его
повторения из корня проекта выполните `python examples/train_visual_demo.py`.

После собственного обычного PyTorch обучения передайте pipeline напрямую:

```python
from IPython.display import display
from diffusion_lab.visualization import (
    notebook_field_explorer, notebook_sampling_explorer,
)

fields = notebook_field_explorer(pipeline, dataset, resolution=32)
fields.controls["potential"].value = True
display(fields)

sampling = notebook_sampling_explorer(pipeline, dataset, steps=50, seed=42)
display(sampling)  # генерация начнётся после нажатия Generate
```

`notebook_field_explorer` пересчитывает поле по текущей модели при отпускании
ползунка. Это живой запрос pipeline, поэтому можно исследовать любое допустимое
время, а не только заранее сохранённые состояния.

| Контрол | Что меняет |
|---|---|
| Condition | Класс / unconditional либо непрерывный вектор условий |
| t | Индекс DDPM/DDIM `0..T−1`; для flow также доступен `T`, показана `sigma=t/T` |
| CFG | `guidance_scale` от 0 до 3 |
| Field | `score`, `epsilon`, `denoising`; для flow — `velocity`, `denoising` |
| Grid | Разрешение сетки от 4 до 64 по каждой координате |
| Normalize arrows | Масштаб отображения стрелок; цвет сохраняет исходную норму |
| Streamlines (2D) | Линии потока на равномерной двумерной сетке |
| Recover potential | Второй график U и диагностика residual / relative curl |

Потенциал вычисляется по исходным векторам: нормировка стрелок не меняет U.
Для 3D рядом с объёмным полем показан средний срез U по z. Если поле не проходит
проверку консервативности, интерфейс явно сообщает о least-squares projection.

По умолчанию categorical conditioner получает список классов, а continuous —
несколько реальных строк `dataset.conditions`. Для своих сцен передайте
`condition_options={"Название сцены": vector}`; ширина вектора должна совпадать
с `pipeline.model.config.condition_dim`. Поля доступны для модели точек
`sample_shape=(D,)`, `D=1,2,3`. Для `[L,D]` используйте sampling explorer.

`notebook_sampling_explorer` позволяет выбрать condition, CFG, число шагов и
seed. **Generate** запускает генерацию; изменение параметров само по себе её
не запускает. **Play** и **Frame** показывают сохранённую `history` того же
запуска: frame 0 — стартовый шум, frame k — результат k реальных переходов
scheduler, последний frame — итоговая выборка. Перемещение по истории не
семплирует новый шум. Для навигационного dataset с явной разметкой
`condition_layout` фиксируются концы A/B и отображаются стены из выбранного
условия. Пересечения стен нужно дополнительно оценивать через `collision_free`.

Оба объекта открыты для доработки в notebook:

```python
fields.controls["kind"].value = "denoising"
grid, raw_vectors = fields.state["grid"], fields.state["vectors"]
potential_result = fields.state["potential"]

result = sampling.generate()  # программный эквивалент Generate
history, timesteps = sampling.state["history"], sampling.state["timesteps"]
sampling.controls["frame"].value = 0
```

`.controls` содержит виджеты, `.state` — результат последнего расчёта,
`.output` — область графика, `.refresh()` — явную перерисовку. Для генерации
дополнительно доступна `.generate()`. При ошибке текст отображается рядом с
графиком, а исключение доступно в `.state["error"]`.

Нужны зависимости `[interactive]`; ноутбуки устанавливают пакет из GitHub первой
ячейкой:

```python
%pip install "diffusion-lab[interactive] @ git+https://github.com/Vovanm88/ddpm_vibe_research.git"
```

В Colab они также включают `enable_custom_widget_manager()`. Для своих
ячеек при необходимости выполните:

```python
from google.colab import output
output.enable_custom_widget_manager()
```

## Экспорт и галерея

`save_figure` сохраняет PNG и редактируемый SVG; доступен также PDF через
`formats=("pdf",)`. `save_denoising_gif` принимает истории точек `[steps,N,D]`
или траекторий `[steps,N,T,D]` и использует Pillow без ffmpeg.

`interactive_field(grid, {"condition=0 t=900": vectors, ...}, path="field.html")`
делает автономный Plotly HTML со слайдером состояния и общим масштабом цветов.
Для него нужен optional extra `interactive`. Времена и условия указываются в
ключах словаря: масштаб норм позволяет сравнивать состояния.

```bash
python examples/field_gallery.py --output outputs/gallery
```

Команда без обучения рисует шесть сред, консервативное и вращательное поля,
точный score смеси гауссиан, маршруты A* вокруг стен и сравнение аналитического
U с восстановлением на маске. Притяжение и отталкивание в навигационном U могут
иметь локальные минимумы; гарантии обхода стен даёт проверенный планировщик,
а не само потенциальное поле.

Методы опираются на официальные API
[SciPy Gaussian KDE](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
и [SciPy LSQR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html).
