# Как расширять проект

Новая среда возвращает `SampleDataset`: float32 `samples[N,D]` или `[N,L,D]`,
categorical `conditions[N]` (long) либо continuous `[N,K]` (float), metadata
из JSON-совместимых значений. При необходимости добавьте factory в
`environments.make_environment`. Для произвольной среды можно напрямую вызвать
trainer без registry. Размеры выводятся из датасета; не кодируйте `2` в модели.

Conditioner — `nn.Module`, создаваемый factory `(hidden_dim,num_classes,condition_dim)`.
Его `forward(conditions)` возвращает `[B,hidden_dim]`. Для token/set-conditioning
можно расширить этот контракт и denoiser вместе, либо сначала свернуть токены в
вектор внутри conditioner. Null-вектор и dropout остаются в `PointDenoiser`.
Модуль должен регистрироваться до `PointDenoiser.from_pretrained`; название
conditioner и размерности автоматически сохраняются в config.

Regression loss возвращает **по одному значению на пример**, `[B]`:

```python
from diffusion_lab.losses import register_loss

@register_loss("charbonnier")
def charbonnier(prediction, target):
    return ((prediction - target).square() + 1e-6).sqrt().flatten(1).mean(1)
```

Trainer применяет веса objective и усредняет. Геометрический regularizer
регистрируется через `register_regularizer(name)` и имеет контракт
`function(predicted_clean,conditions,metadata) -> scalar`. Вес задаётся в
`training.regularizers`. У diffusion геометрия считается на нижней половине
уровней шума, чтобы избежать плохо обусловленных x₀-предсказаний при больших t.
Веса между loss разных единиц измерения требуют отдельного подбора.

Новый метод регистрируется через `register_objective(name)` и реализует:

- `__init__(scheduler,snr_gamma=None)` с проверкой совместимости.
- `prepare(clean,generator) -> TrainingPair(noisy,target,timesteps,weights)`.
- `clean_prediction(pair,prediction) -> predicted_clean` того же shape.
- `regularization_weights(pair) -> [B]`, сейчас используемый как маска активных примеров.

При изменении семейства процесса реализуйте также подходящий scheduler/семплинг.
Нельзя обучать velocity flow и затем генерировать DDIM: CLI отклоняет такую
замену. Пример `RectifiedFlowObjective` использует линейный мост и официальный
`FlowMatchEulerDiscreteScheduler` с `shift=1`.

CFG использует `prediction_uncond + scale*(prediction_cond - prediction_uncond)`.
`scale=1` — условная модель, `0` — null-условие. Во время обучения доля примеров
теряет conditioning; истинный класс при этом не меняется на новый ID.

Код добавляется в пакет, а ноутбук остаётся клиентом. Перед исследованием
проверьте seed, shapes, finite loss/gradients, контракт timestep и сохранение.
Особенно полезны численные oracle-тесты: идеальный denoiser должен приводить
к известному x₀ при полном и редком расписании. Для нового conditioner проверяйте
null masking и отсутствие обмена информацией между примерами батча.
