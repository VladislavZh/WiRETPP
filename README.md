# Active Block Wishart TPP

Исследовательская реализация иерархической смеси временных точечных
процессов с trajectory-level Wishart random effect. Метод реализован как
обёртка над произвольным нейросетевым банком интенсивностей, а не как
расширение конкретной архитектуры.

Поддерживаемые backbone:

- `cotic` — continuous convolution из
  [VladislavZh/COTIC](https://github.com/VladislavZh/COTIC);
- `thp`, `nhp`, `rmtpp` — реализации
  [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess),
  зафиксированные на `easy-tpp==0.2.1`, как в исходных экспериментах.

Один и тот же код Active Block Wishart, variational inference и обучения
используется со всеми backbone. Отличается только adapter, создающий банк
положительных интенсивностей `K x C`.

Подробный вывод модели и каждого шага алгоритма находится в
[docs/MATHEMATICS.md](docs/MATHEMATICS.md).
Порядок независимого аудита реализации находится в
[docs/CODE_AUDIT.md](docs/CODE_AUDIT.md).
Формулы в `MATHEMATICS.md` предварительно отрендерены в локальные SVG, поэтому документ
не зависит от поддержки MathJax конкретным Markdown preview. Для изменения
формул редактируется `docs/MATHEMATICS_SOURCE.md`, после чего выполняется:

```bash
python scripts/render_mathematics.py
```

## Что находится в репозитории

```text
configs/
  all12.yaml                 конфигурация базового all-12 schedule
scripts/
  pack_data.py               проверенная упаковка CSV-наборов в Parquet
  run_real_benchmark.py      matched COTIC/Pure/Wishart real-data protocol
  render_mathematics.py      сборка формул документации
src/wishart_tpp/
  backbones/                 общий API и adapters COTIC/EasyTPP
  cotic/                     локальная читаемая реализация COTIC
  model/                     TPP trace, Wishart math, active-block decoder
  inference/                 local VI, responsibilities, Omega и alpha M-steps
  training/
    experiment.py           абстрактный ExperimentRunner template
    shared_pretrain.py      общий resumable K=1 pretrain
    pure_branch.py          matched COTIC K=1 и Pure K=5 ветви
    wishart_branch.py       fixed-q Wishart ветвь
    cycle_checkpoint.py     exact cycle-boundary persistence
    pure_experiment.py      отдельный PureExperimentRunner
    wishart_experiment.py   отдельный WishartExperimentRunner
    active.py               явный variational-EM цикл Lightning Fabric
  data.py                    единый data module, split и train-only нормализация
  data_adapters.py           только прослойки несовместимых raw-форматов
tests/                       только проверки реальных точек отказа
```

Модули разделены по математической ответственности. Базовый `ExperimentRunner`
не знает деталей Pure или Wishart: его наследники реализуют собственный
training/evaluation path, а CLI создаёт ровно один выбранный runner. Полный
Variational-EM сосредоточен в `training/active.py`; exact-resume, обучение
ветвей, selected evaluation и запись артефактов вынесены в отдельные сущности.

## Модель в одной формуле

Backbone с параметрами `theta` выдаёт базовые интенсивности

```text
h_theta,k(t | H_t) in R_+^C,  k=1,...,K.
```

Для траектории `m`:

```text
z_m ~ Categorical(pi)
U_m | z_m=k ~ Wishart_C(nu, Omega_k / nu)
T_U(h) = [U_m^(1/2) sqrt(h_theta,k(t))]^2
lambda_m(t) = (1-alpha) h_theta,k(t) + alpha T_U(h_theta,k(t)) + 1e-6.
```

Смесь интерполируется в пространстве интенсивностей, а не амплитуд. При
`alpha=0` получается точный no-W/Pure Mixture endpoint с тем же постоянным
floor `1e-6`. Веса компонентов `pi` являются независимыми обучаемыми
параметрами: они никогда не извлекаются из `U`. Pure и Wishart
запускаются отдельно, но при одинаковых seed и конфигурации воспроизводят
один и тот же короткий стартовый checkpoint.

## Установка

Рекомендуется Python 3.11 и отдельное окружение.

```bash
python -m venv .venv
.venv/Scripts/activate             # Windows
pip install -e .
```

Для Linux/macOS команда активации — `source .venv/bin/activate`.

PyTorch можно заранее установить с подходящим CUDA wheel. `Lightning
Fabric` используется только как тонкий слой над явным training loop:
setup устройства/precision, backward и gradient clipping. Lightning Trainer,
callbacks и LightningModule здесь отсутствуют.

## Данные

Рабочий формат хранит одну строку Parquet на траекторию:

```text
data/
  sin_K4_C5/
    events.parquet
```

Строка содержит `source_id`, `label`, `horizon` и вложенные массивы
`times/marks`. Для official real-data наборов вместо `label/horizon` хранится
исходный `split`; это единственная несовместимость, изолированная в raw-format
адаптере. После чтения все наборы проходят один `EventDataModule`: split
фиксируется до статистик, train-only P99 exponential normalizer оценивается
одинаково для synthetic, Age и real, затем одна трансформация применяется к
train/validation/test. Legacy-раскладка из отдельных CSV пока читается для
миграции. Упаковщик пишет
во временный файл, сверяет типизированный SHA-256 и лишь затем, при явном флаге,
удаляет CSV:

```bash
python scripts/pack_data.py K2_C5 sin_K2_C5 --root data --delete-source
```

Amazon защищён от случайной упаковки без `--allow-amazon`, чтобы не менять
источник данных работающего процесса. Synthetic DAN и Age split воспроизводят
прежний протокол буквально, но теперь до общей нормировки:

```python
indices = numpy.arange(N)
numpy.random.RandomState(42).shuffle(indices)
train, validation, test = indices[:80%], indices[80%:90%], indices[90%:]
```

Для каждой траектории общий normalizer стабильно сортирует события в raw-адаптере,
сдвигает первое событие в ноль и умножает времена и оставшийся observation horizon
на один train-only коэффициент. Validation и test не участвуют ни в unit detection,
ни в P99, ни в оценке exponential rate.

## Запуск

Актуальный matched real-data протокол для seed 1 или 2:

```powershell
$env:PYTHONPATH="src"
python scripts/run_real_benchmark.py --dataset retweet --seed 1
python scripts/run_real_benchmark.py --dataset amazon --seed 1
python scripts/run_real_benchmark.py --dataset so --seed 1
```

Он создаёт общий K=1 checkpoint, непрерывную COTIC K=1 ветвь, независимую
Pure K=5 ветвь и fixed-q Wishart K=5 ветвь на одной matched-сетке обновлений.
`configs/real_data.yaml` фиксирует научные настройки, а compute-only shards
выбираются по dataset и могут быть уменьшены без изменения effective batch.

Базовый Wishart all-12 протокол:

```bash
wishart-tpp --config configs/all12.yaml
```

Отдельный Pure-прогон:

```bash
wishart-tpp --config configs/all12.yaml --method pure
```

Один запуск всегда обучает только выбранный `training.method`. Режима
`Pure EM` нет: соответствующая endpoint-абляция запускается через тот же
Wishart schedule с фиксированной нулевой силой смешивания:

```bash
wishart-tpp --config configs/all12.yaml --method wishart --fixed-alpha 0
```

Или без editable entry point:

```bash
PYTHONPATH=src python -m wishart_tpp.cli --config configs/all12.yaml
```

В PowerShell эквивалентная команда:

```powershell
$env:PYTHONPATH="src"
python -m wishart_tpp.cli --config configs/all12.yaml
```

Backbone меняется без изменения алгоритма:

```bash
wishart-tpp --config configs/all12.yaml --backbone thp
wishart-tpp --config configs/all12.yaml --backbone nhp
wishart-tpp --config configs/all12.yaml --backbone rmtpp
```

Можно ограничить запуск одним набором:

```bash
wishart-tpp --config configs/all12.yaml \
  --dataset sin_K4_C5 \
  --backbone cotic \
  --output-root runs/debug
```

Метод интегрирования компенсатора выбирается независимо от backbone:

```bash
wishart-tpp --config configs/all12.yaml \
  --integral-method monte_carlo \
  --integral-samples 20
```

Во время обучения равномерные MC-точки пересэмплируются, а validation/test
используют фиксированный `integral_seed`. Это сохраняет стохастический training
objective, но не позволяет шуму интеграла выбирать checkpoint.

## Базовый schedule

`configs/all12.yaml` фиксирует восстановленный production EM-режим, от которого
строятся дальнейшие schedule-эксперименты:

- общий checkpoint: 10 Pure minibatch-шагов;
- Pure при `method: pure`: ещё 600 neural steps;
- Wishart при `method: wishart`: 75 outer cycles по 8 neural steps;
- local Wishart VI: 10 шагов, 4 train-сэмпла, 8 evaluation-сэмплов;
- `df=16` фиксирован, `alpha` стартует с `0.1`;
- exact trace-constrained target для `Omega` с damping `0.25`;
- полный bounded search `alpha` на `[0,1]` с damping `0.5`;
- independent mixture weights обновляются по responsibilities с damping `0.25`;
- balanced responsibilities только первые 2 цикла;
- один persistent Adam для neural M-step;
- полный train или один детерминированный EM block без posterior cache;
- COTIC compensator использует 50 Monte Carlo time samples;
- checkpoint выбирается только по validation prior-predictive NLL.

Следующие модификации не входят в поддерживаемый код:

- обучение population `df`;
- повторного E-step после изменения `alpha`;
- удлинённого local warm-up;
- early stopping;
- auxiliary loss.

Таким образом, конфигурация задаёт общую воспроизводимую точку отсчёта, а не
устаревшую версию модели.

## Результаты запуска

Артефакты разных методов не пересекаются. Pure создаёт:

```text
runs/all12/pure/<dataset>/
  shared_checkpoint.pt
  pure_checkpoint.pt
  shared_history.csv
  pure_history.csv
  result.json
```

Wishart создаёт:

```text
runs/all12/wishart/<dataset>/
  shared_checkpoint.pt
  wishart_checkpoint.pt
  shared_history.csv
  wishart_history.csv
  result.json
```

В `result.json` записываются только метрики выбранного метода:

- Pure/no-W test NLL, purity и ARI для Pure;
- prior-predictive test NLL, purity и ARI для Wishart;
- test-only posterior clustering и `alpha` только для Wishart;
- полная конфигурация и размеры split.

Test не участвует в выборе checkpoint.

## Как добавить новый backbone

Нужно реализовать один класс `IntensityBank`:

```python
class MyBank(IntensityBank):
    n_components: int
    n_marks: int
    mixture_logits: torch.nn.Parameter

    def expand_components(self, count, noise, seed): ...
    def forward(self, sequences) -> TPPTrace: ...
```

`TPPTrace` содержит только то, что требуется обычному marked-TPP likelihood:

- интенсивности в наблюдённых событиях, shape `events x K x C`;
- интенсивности в точках численного интегрирования, shape `points x K x C`;
- marks, path indices и веса интегрирования.

Active decoder не знает, были эти интенсивности получены свёрткой,
Transformer, CT-LSTM или RNN.

Для state-based моделей есть более узкий `StateIntensityBank`: adapter задаёт
только кодирование истории и отображение `(state, elapsed) -> K x C`.

## Тесты

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Проверяются точки, где реализация действительно может сломаться:

- точный DAN split 80/10/10 без пересечений;
- общий trace contract для COTIC, THP, NHP и RMTPP;
- точное равенство Active и Pure при `alpha=0`;
- нулевой KL одинаковых Wishart laws;
- trace и objective exact `Omega` M-step;
- один полный Wishart cycle через Lightning Fabric;
- раздельный dispatch Pure и Wishart без запуска второй ветви;
- пропуск alpha M-step при фиксированной `alpha`;
- значения базового all-12 schedule.

## Ссылки и происхождение кода

- COTIC: <https://github.com/VladislavZh/COTIC>, изучавшийся commit
  `362b8ab1f3cbb9e9dced2518e9daacf68235e77a`.
- EasyTPP: <https://github.com/ant-research/EasyTemporalPointProcess>,
  Apache-2.0; в окружении зафиксирован release `0.2.1`.
- Lightning Fabric: <https://lightning.ai/docs/fabric/stable/>.
