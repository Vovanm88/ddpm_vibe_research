# Быстрый старт из Jupyter: PyTorch / Diffusers

Откройте [`03_standalone_quickstart.ipynb`](../notebooks/03_standalone_quickstart.ipynb).
Для замены архитектуры, loss, conditioner, optimizer и метода обучения —
[`04_change_components.ipynb`](../notebooks/04_change_components.ipynb).
Оба ноутбука лёгкие: первая ячейка устанавливает пакет из GitHub в Python
текущего ядра. Для установки нужен интернет; эксперименты используют
синтетические данные и обучают модель прямо в ноутбуке.

Прямые ссылки:
[quickstart в Colab](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/main/notebooks/03_standalone_quickstart.ipynb),
[изменение компонентов в Colab](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/main/notebooks/04_change_components.ipynb),
[обученная модель и живые графики в Colab](https://colab.research.google.com/github/Vovanm88/ddpm_vibe_research/blob/main/notebooks/06_trained_interactive_demo.ipynb).

Для просмотра уже обученной диффузии используйте
[`06_trained_interactive_demo.ipynb`](../notebooks/06_trained_interactive_demo.ipynb):
он загружает веса EMA после **120 эпох** из `demo/pipeline` в репозитории.
В локальном checkout используются файлы проекта; в отдельном Jupyter / Colab
ноутбук клонирует репозиторий с GitHub. После загрузки можно исследовать поля и
генерировать точки, затем перейти к своему обучению в `03` / `04`. Отдельный
[`05_live_visualizations.ipynb`](../notebooks/05_live_visualizations.ipynb) —
локальный вариант для `outputs/visual_demo_rays`; этот запуск создаёт команда
`python examples/train_visual_demo.py` из корня репозитория.

Нужен Python 3.11 или новее. В [Google Colab](https://research.google.com/colaboratory/faq.html) загрузите `.ipynb` и выполните ячейки
сверху вниз. В локальном Jupyter откройте файл и выберите нужное ядро.
Короткие примеры по умолчанию работают на CPU; для доступного ускорителя замените
`DEVICE = "cpu"` на `"cuda"` или `"mps"`.

В ноутбуках обычный PyTorch loop с `zero_grad → backward → optimizer.step`.
`Dataset`, `DataLoader`, `model`, `scheduler`, `optimizer` и Diffusers pipeline
создаются отдельно и доступны напрямую. Loss — обычная функция или `nn.Module`;
можно редактировать формулы обучения и добавлять свои loss terms.

В пустом Jupyter можно установить пакет прямо из GitHub:

```python
%pip install "diffusion-lab[interactive] @ git+https://github.com/Vovanm88/ddpm_vibe_research.git"
```

Или скопируйте первую установочную ячейку из `03` / `04`. Выполняйте установку перед
импортами; после обновления уже импортированного пакета перезапустите kernel.
Вот минимальная сборка с реальным обучением:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from diffusion_lab import PointDenoiser, ResearchDiffusionPipeline
from diffusion_lab.environments import make_environment

DEVICE = "cpu"
torch.manual_seed(42)
torch.set_num_threads(4)
dataset = make_environment("rotating_rays", n_samples=256, n_classes=4, seed=42)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = PointDenoiser(
    sample_shape=dataset.sample_shape,
    hidden_dim=32, num_layers=1,
    conditioner="embedding", num_classes=dataset.num_classes,
).to(DEVICE)
scheduler = DDIMScheduler(
    num_train_timesteps=64, prediction_type="v_prediction",
    beta_schedule="squaredcos_cap_v2", clip_sample=False,
    timestep_spacing="trailing",
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.MSELoss()

for epoch in range(3):
    model.train()
    for batch in loader:
        clean = batch["samples"].to(DEVICE)
        conditions = batch["conditions"].to(DEVICE)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(64, (len(clean),), device=DEVICE)
        noisy = scheduler.add_noise(clean, noise, timesteps)
        target = scheduler.get_velocity(clean, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, timesteps, conditions=conditions).sample
        loss = loss_fn(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

pipeline = ResearchDiffusionPipeline(model=model, scheduler=scheduler)
generated = pipeline(
    batch_size=64, conditions=torch.arange(64) % dataset.num_classes,
    num_inference_steps=24, generator=torch.Generator().manual_seed(43),
)
from diffusion_lab.visualization import plot_dataset
plot_dataset(generated.samples, torch.arange(64) % dataset.num_classes)
```

Это короткое обучение, поэтому качество ещё не сошлось. Для исследования
увеличивайте `n_samples`, эпохи, ширину и глубину модели; подбирайте параметры
по метрикам и графикам. Scheduler должен зашумлять данные до распределения,
близкого к стартовому Gaussian noise генерации. В примерах используется cosine
schedule; для сравнения epsilon / v — одинаковый linear schedule с 1000 timesteps.

## Интерактивные поля и генерация после обучения

Передайте те же `pipeline` и `dataset` виджетам — их не нужно пересоздавать через
trainer или experiment:

```python
from IPython.display import display
from diffusion_lab.visualization import (
    notebook_field_explorer, notebook_sampling_explorer,
)

fields = notebook_field_explorer(pipeline, dataset, resolution=24)
fields.controls["potential"].value = True
display(fields)

# Здесь 24 <= num_train_timesteps=64 из минимального примера выше.
sampling = notebook_sampling_explorer(pipeline, dataset, steps=24, seed=43)
display(sampling)
```

В поле доступны condition / unconditional, timestep, CFG, тип поля, разрешение
сетки, нормировка стрелок, 2D streamlines и **Recover potential**. Слайдеры
пересчитывают поле по модели при отпускании ползунка. Цвет стрелок показывает
исходную норму; U вычисляется по исходному полю и сопровождается residual/curl,
чтобы отличать потенциальное поле от его численной проекции.

В генераторе выберите condition, CFG, steps и seed, затем нажмите **Generate**.
**Play** и **Frame** показывают настоящие сохранённые шаги денойзинга этого же
запуска. Изменение настроек не запускает генерацию автоматически; повторный
запуск можно вызвать через `sampling.generate()`.

Виджеты доступны в `.controls`, данные — в `.state`. Например,
`fields.state["vectors"]` содержит исходное поле,
`sampling.state["history"]` — историю, а `sampling.state["timesteps"]` —
реальные времена scheduler. Для continuous conditioner можно передать
`condition_options={"Сцена": feature_vector}` либо использовать
репрезентативные строки dataset. Поддерживаются точки 1D/2D/3D; для моделей
траекторий sampling explorer рисует пути, а для явно размеченной навигации —
стены и фиксированные A/B.

В `03` / `04` есть готовые ячейки с этими контролами. Ноутбуки включают
`enable_custom_widget_manager()` для Colab; локальному Jupyter нужны установленные
`ipywidgets` из `[interactive]`. Полный API и интерпретация полей описаны в
[visualization.md](visualization.md).

## Что можно менять — с обучением

| Изменение | Пример в `04_change_components.ipynb` |
|---|---|
| Архитектура | MLP, Transformer, свой обучаемый residual input adapter |
| Loss callable | `nn.MSELoss()`, `nn.SmoothL1Loss()`, своя функция Charbonnier |
| Conditioner | One-hot, embedding, continuous MLP и свой Fourier encoder |
| Optimizer | `torch.optim.AdamW` и `torch.optim.Adam` |
| Target и метод | Diffusion `epsilon` / `v_prediction`, rectified flow |
| Loss terms | Свой radius penalty, smoothness / collision / endpoint / bounds |
| Данные и размерность | Свой `SampleDataset`, 1D Gaussian mixture, 3D helix, навигация |
| Sampling | Condition, CFG, число шагов, фиксация известных концов траектории |
| Продолжение | Тот же optimizer, Diffusers export, `torch.save` / `torch.load` |

Весь `train_loop(model, scheduler, loader, optimizer, loss_fn, ...)` виден в
ноутбуке. Для сравнений есть короткий helper, возвращающий словарь с этими
обычными объектами, history, generated samples и metrics.
Каждый вариант действительно оптимизирует веса. Большинство запусков выполняет
2–3 эпохи; пара epsilon / v в ноутбуке обучается по 20 эпох.
Значения разных loss и targets имеют разные единицы: сравнивайте также общую
метрику качества генерации. Статистический вывод требует нескольких seed и
полноценного обучения.

В навигации reference paths строит A*. Endpoint constraint фиксирует A и B при
семплировании, поэтому нулевая ошибка концов сама по себе не показывает, что модель
научилась маршрутам. После двух эпох пути могут пересекать стены: проверяйте
`collision_free_fraction`. Loss penalties не гарантируют обход препятствий.

## Свои данные и компоненты

`SampleDataset` принимает float `samples[N,D]` или `[N,L,D]`, категориальные
integer `conditions[N]` либо непрерывные float `conditions[N,K]`.
Его `sample_shape`, `num_classes` и `condition_dim` передаются в модель.
Собственный model наследуется от `PointDenoiser` либо совместимого Diffusers
`ModelMixin`. Conditioner — `nn.Module`, возвращающий `[B,hidden_dim]`;
Fourier-пример регистрирует factory только для выбора encoder по имени в model config.
Loss и regularizers в явном loop принимаются как callables без registry.

Для новой формулы обучения меняйте тело loop. Там прямо показаны:

- diffusion noise target `noise`, clean target `clean`, velocity target `scheduler.get_velocity(...)`;
- rectified flow bridge `(1-sigma)*clean + sigma*noise` и target `noise-clean`;
- восстановление clean sample для дополнительных penalties;
- conditioning dropout, gradient clipping и optimizer step.

Flow scheduler и diffusion scheduler должны соответствовать training target.
Усиливать CFG после обучения имеет смысл, если null-condition обучался через
conditioning dropout, как в полном примере loop.

## Сохранение и продолжение

Pipeline экспортируется обычным Diffusers `pipeline.save_pretrained(...)`.
Отдельный `torch.save(...)` хранит optimizer state, history, данные и состояния
генераторов noise / DataLoader. В ноутбуке есть работающий reload и ещё одна эпоха.
Для custom класса сохраните его Python-код вместе с исследованием. Например:

```python
custom_model = AdapterDenoiser.from_pretrained("my_run/pipeline/model")
pipeline = ResearchDiffusionPipeline.from_pretrained(
    "my_run/pipeline", model=custom_model, local_files_only=True,
)
```

Сначала выполните определение класса и регистрацию custom conditioner. Ячейка
Fourier conditioner заменяет factory при повторном запуске: после редактирования
создайте новую модель и optimizer, чтобы использовать новую реализацию.
Для loading / resume используйте свой loss callable и такой же тип optimizer.

Для повторяемого запуска без Jupyter:

```bash
python examples/notebook_components.py --epochs 3 --samples 256
```

`run_examples()` возвращает обученные model / pipeline / optimizer, datasets,
outputs, history и metrics. CLI применяет `--epochs` ко всем сравнениям компонентов;
навигационный smoke ограничен двумя эпохами.

Для изменения самого пакета клонируйте проект и установите его исходники:

```bash
git clone https://github.com/Vovanm88/ddpm_vibe_research.git
cd ddpm_vibe_research
python -m pip install -e ".[interactive,dev]"
python -m pytest tests/test_notebook_api.py tests/test_widgets.py
```

Редактируйте `src/diffusion_lab`, затем перезапустите kernel для обновления
импортов. Изменения модели, loss и training loop в самих notebook применяются
при повторном выполнении соответствующих ячеек; после смены архитектуры
создавайте новую модель и optimizer.

Для компактных конфигурационных запусков в пакете также остаются опциональные
`DiffusionTrainer` и `create_experiment`. Ноутбуки быстрого старта используют
обычные объекты PyTorch / Diffusers напрямую.
