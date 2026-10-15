# Источники и область реализации

Вход: `REPLICATION_SPEC.md`, версия 1.0 от 15.10.2026. Его точная копия и SHA256
сохранены в `replication/source/`. Соседний `bachelors_thesis.pdf` найден локально:
97 страниц, SHA256 `850da2d317567e30373f3615a86bdf98fee24a7157ad7b3abf0b7990f1e4e65c`,
совпадает с ТЗ. PDF не публикуется в репозитории. Полный постраничный аудит и
аудит авторского кода не выполнены; авторский тренировочный код не предоставлен.

Это независимая реализация математического протокола **[P]** из ТЗ. Числа из
диплома не используются как новые измерения или целевые результаты. Ошибка
в этой реализации не приписывается авторскому коду без его проверки.

Прямое требование пользователя имеет приоритет над предложенным порядком ТЗ:
перед MNIST проверяются вращающиеся лучи и показываются анимации. Исходные
эксперименты проекта сохраняются; работа идёт в отдельной ветке
`research/wasserstein-tube-replication`.

Проверенные внешние первичные источники:

- [GeomLoss API](https://www.kernel-operations.io/geomloss/api/pytorch-api.html):
  `p=2` задаёт половину squared Euclidean cost; epsilon=blur²; debias self terms.
- [Sinkhorn divergences](https://arxiv.org/abs/1810.08278): regularized cross/self objectives.
- [Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html):
  официальные train/test datasets.

GeomLoss 0.3.1 установлен отдельно, существующий Torch не заменялся.
Legacy annealing solver проверяется против converged float64 reference;
само название библиотечной функции не считается доказательством сходимости.


## Карта реализации

| Источник утверждения | Реализованный объект | Код / проверка |
|---|---|---|
| [P] §§4–6: unconditional FM, t=0 data, t=1 noise, signed Euler | Torch teacher и точные 32→16 / 32→8 / 24→16 targets | `fm.py`, `schedules.py`; schedule и training tests |
| [P] §7: canonical KL cross/self OT, обе student self gradients | Сходящийся log-domain float64 Sinkhorn | `losses.py`; finite differences, marginal residuals, exact-OT limit, GeomLoss parity |
| [P] §§6–7: tube δ=ρH, retarget, train-only ε | Отдельные RNG; один teacher и immutable calibration | `distill.py`, `calibration.py`; identity/lambda0/resume/common-RNG tests |
| [P] §8 и [M]: GMM oracle, endpoint formulas | Аналитическое поле и RK4 с фактическим NFE | `oracle.py`; numerical report |
| [P] §9: real-only CNN, unbiased feature MMD² | Закрытый test, frozen train-only standardization/kernel | `mnist.py`, `mnist_workflow.py`, `evaluate.py`; leakage/hash tests |
| [P] §§10–13: paired fidelity, robustness, statistics | Отдельные quality/fidelity/cost и парные contrasts | `diagnostics.py`, `reporting.py`; assessment/reporting tests |
| [D]: числа и авторские детали диплома | Перечислены только как исходные утверждения в ТЗ | Не объявлены результатами этого проекта; авторский код не проверен |

Пути к коду указаны относительно `src/recheck`. Программные проверки и пилотные
измерения удостоверяют эту реализацию [P]; они не заменяют аудит авторского кода.
