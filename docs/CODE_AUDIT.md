# Порядок аудита Active Block Wishart TPP

Этот документ задаёт порядок чтения и проверки текущего алгоритма. Идти нужно
снизу вверх по зависимостям: ошибка в данных или trace-контракте делает
бессмысленным аудит variational-EM, даже если формулы верхнего уровня выглядят
правильно.

## 0. Сначала зафиксировать проверяемый протокол

1. Сохранить `git status --short`, используемый YAML и seed-и.
2. Проверить `ExperimentConfig.from_yaml` и все `__post_init__` в
   `src/wishart_tpp/config.py`. YAML не должен молча создавать неизвестные поля,
   а ограничения должны совпадать с допустимой областью модели.
3. Выписать фактически включённые ветви: backbone, интегратор, `method`, число
   компонент, sharding, EM-block, learned/fixed population df, damping и
   scheduler.
4. Не сравнивать два запуска, пока не совпадают split, cohort, optimization,
   Monte Carlo и neural seed-и.

Результат этапа: одна таблица «параметр — источник — фактическое значение».

## 1. Данные и разбиения

Смотреть `src/wishart_tpp/data.py`, затем узкие raw-format прослойки в
`src/wishart_tpp/data_adapters.py`, `scripts/pack_data.py` и
`tests/test_data.py`.

Проверить по порядку:

1. Одна Parquet-строка действительно соответствует одной траектории;
   `source_id`, `times`, `marks`, `label/split` не меняются при упаковке.
2. Времена отсортированы стабильно, marks остаются привязаны к тем же событиям,
   события лежат в `[0, horizon]`.
3. Synthetic DAN и Age используют ровно seeded 80/10/10 permutation; real-data —
   только заявленный official или deduplicated protocol.
4. Один и тот же P99 exponential normalizer применяется к synthetic, Age и real
   и оценивается только по train. Validation/test не могут влиять на unit scale,
   P99 или exponential rate.
5. `DatasetPartition.select` одинаково переставляет sequences, labels и
   source IDs. Особенно проверить индексы при `trajectory_limit`, sharding и
   EM blocks.
6. Ветвление по dataset name отсутствует в trainer: различия labeled и official
   split форматов заканчиваются внутри двух адаптеров до общей нормировки.

Стоп-условие: при пересечении split-ов, несовпадении source IDs или изменении
числа событий дальнейший аудит не проводить.

## 2. Backbone и единый trace-контракт

Начать с `src/wishart_tpp/model/trace.py`, затем читать
`backbones/integration.py`, `backbones/base.py`, выбранный backbone и только
после этого `backbones/factory.py`.

Для одного игрушечного batch проверить shapes и смысл каждого поля `TPPTrace`:

- event rows: только наблюдавшиеся события, корректные path/mark indices;
- integral rows: полное покрытие каждого межсобытийного интервала и хвоста до
  horizon;
- base rates: положительные и имеют shape `rows x K x C`;
- quadrature weights имеют правильную временную размерность;
- история в event time не содержит текущее или будущее событие.

Отдельно проверить COTIC alignment: нет обучаемого BOS, первое событие
оценивается из нулевой истории, а encoder видит только реальные события.
Gauss--Legendre должен быть детерминирован. Monte Carlo в train пересэмплирует
точки в контролируемом seed-контексте, а validation/test повторяемы.

## 3. Likelihood и точный Pure endpoint

Читать `src/wishart_tpp/model/active_block.py` рядом с разделами 2–3
`docs/MATHEMATICS_SOURCE.md`.

Проверить:

1. Wishart-ветвь равна `(sqrt(U) @ sqrt(h))**2`, после чего выполняется
   convex-интерполяция в пространстве интенсивностей:
   `(1-alpha) h + alpha wishart_branch + 1e-6`.
2. Event term берёт log интенсивности наблюдавшегося mark; compensator суммирует
   все marks и интеграционные точки. В обоих членах обязан использоваться тот
   же постоянный floor `1e-6`.
3. `alpha=0` совпадает с `base_component_scores` до машинной точности без
   зависимости от Wishart draws.
4. Prior predictive усредняет likelihood в log-space, а не среднее log
   likelihood; повторные Monte Carlo оценки объединяются тем же способом.
5. Mixture weights являются независимыми logits. В production-пути не должно
   быть `pi(W)`, block-trace routing или `Corr(W)^2` attention.

Главный тест этого слоя — `tests/test_model_math.py`.

## 4. Wishart-примитивы и параметризация SPD

Читать `src/wishart_tpp/model/wishart.py`, затем
`tests/test_model_math.py`.

Проверить mean/df parameterization во всех формулах: scale всегда равен
`mean / df`. Обязательные инварианты:

- `wishart_kl(q, q) = 0`;
- Bartlett samples симметричны и SPD;
- их Monte Carlo mean стремится к переданному mean;
- `df > C - 1`;
- roundoff repair в `stable_cholesky` не меняет обычный SPD path и не создаёт
  non-finite gradients;
- matrix square root симметричен и восстанавливает исходную матрицу.

Не принимать «исправление» численной проблемы, если оно меняет статистическую
модель или обнуляет весь gradient tensor вместо отдельных non-finite entries.

## 5. Локальный E-step

Читать `src/wishart_tpp/inference/local.py` вместе с
`training/cache.py`.

Последовательность проверки:

1. Neural trace заморожен до оптимизации `q_mk(U)`.
2. Локальные means параметризованы Cholesky, df — softplus над границей
   `C - 1`; warm start индексирован той же траекторией.
3. Objective равен `E_q[-log p(events | U,k)] + KL(q||p_k)` для каждой пары
   trajectory/component.
4. Training и diagnostic evaluation используют раздельные, явно seeded draws.
5. Free energy имеет shape `N x K`; event-count normalization применяется
   только к диагностике сходимости, а не меняет ELBO.
6. Shard concatenation восстанавливает исходный глобальный порядок путей.

Для adaptive test-time VI отдельно проверить minimum steps, patience и три
критерия сходимости. Test-time posterior не должен участвовать в выборе
checkpoint.

## 6. Категориальный и population M-шаги

Читать в таком порядке:

1. `inference/responsibilities.py`: `gamma = softmax(log pi - free_energy)`;
   balancing допустим только в заданных ранних cycles.
2. Начальные `Omega_k` обязаны быть разными детерминированными случайными SPD
   draws с RNG от optimization seed и точной нормировкой `trace(Omega_k)=C`.
   Общая identity-инициализация недопустима: она сохраняет симметрию компонент
   и оставляет identity-смещение при damped population M-step.
3. `inference/population.py`: sufficient mean, полный KKT solver и ограничение
   `trace(Omega_k)=C`. Проверить все ветви знака множителя и то, что выбранный
   кандидат минимизирует objective. Тесты — `tests/test_population_mstep.py`.
4. `inference/mixture_weights.py`: ML при concentration=1, симметричный
   Dirichlet MAP выше единицы. Тесты — `tests/test_mixture_weights.py`.
5. `inference/population_df.py`: профиль строится при фиксированном локальном
   posterior, минимум выбирается по weighted KL, damping идёт в log-distance от
   допустимой границы. Тесты — `tests/test_population_df.py`.
6. `inference/alpha.py`: одни и те же random draws используются для всех
   кандидатов alpha; endpoints 0 и 1 обязательно входят в поиск.
7. `inference/damping.py`: интерполяция Omega, alpha и pi не нарушает SPD,
   `[0,1]` и simplex. Тесты — `tests/test_damping.py`.

После каждого шага проверить device/dtype и отсутствие скрытого переноса
градиента из M-шага в локальный posterior.

## 7. Neural M-step и полный outer cycle

Главный файл — `src/wishart_tpp/training/active.py`. Читать не сверху вниз, а
по цепочке вызовов:

`fit -> _run_cycle -> _e_step -> _population_step -> _population_df_step ->
_global_msteps -> _neural_mstep -> _validation_row`.

Это единственное осознанное исключение из лимита 500 строк: trainer хранит
один последовательный transaction-like EM cycle, где разнесение фаз по
независимым объектам скрыло бы порядок мутаций model/optimizer/population/RNG.
Checkpoint persistence, branch orchestration и artifact I/O из него вынесены.

Фактический цикл должен быть таким:

1. Сбросить train Monte Carlo stream из номера cycle.
2. Выбрать полный train или один детерминированный non-overlapping EM block.
3. Обновить локальные posterior и responsibilities.
4. Обновить Omega, pi и при разрешённом cycle population df.
5. Обновить alpha или оставить строго фиксированным.
6. Сделать короткий neural M-step при фиксированных `q(U)` и gamma.
7. Посчитать prior-predictive validation NLL и только им выбирать checkpoint.
8. Сохранить полное состояние на границе cycle.

Проверить знаменатель neural loss: microbatches суммируются и делятся на
exposure всего effective batch ровно один раз. `mixture_logits` исключены из
Adam ActiveTrainer и обновляются только categorical M-step. При смене EM block
локальные warm starts обязаны очищаться: cache между блоками не поддерживается.

Интеграционные проверки находятся в `tests/test_fabric_smoke.py`. Это самый
дорогой тестовый файл; запускать его после всех математических unit tests.

## 8. Resume, RNG и выбор checkpoint

В `training/active.py` проверить `_resume_signature`, а затем отдельно
`training/cycle_checkpoint.py`.

Checkpoint обязан включать model, optimizer, scheduler, population state,
допустимые локальные warm starts, лучший validation checkpoint, history и состояния
Python/NumPy/Torch/CUDA RNG. Resume с несовместимым schedule должен завершаться
ошибкой. Единственное допустимое послабление должно быть явно мигрировано и
покрыто тестом.

Затем проверить `training/evaluation.py`, `training/artifacts.py` и
`training/wishart_experiment.py`:

- validation выбирает, test только сообщает;
- prior test не адаптируется к тестовой траектории;
- posterior test inference используется только для описательной кластеризации;
- промежуточный training checkpoint не выдаётся за финальный selected
  checkpoint;
- запись результата атомарна там, где процесс может быть прерван.

## 9. Оркестрация проверяется последней

Только после проверки алгоритма читать `training/experiment.py`,
`shared_pretrain.py`, `pure_branch.py`, `wishart_branch.py`,
`selected_evaluation.py`, `wishart_experiment.py` и
`scripts/run_real_benchmark.py`.

Проверить, что generic CLI создаёт ровно один выбранный runner, а matched
real-data CLI явно создаёт Pure- и Wishart-конфигурации. Общий K=1 checkpoint
расширяется до K лишь после shared pretrain; COTIC продолжает его optimizer/RNG,
а Pure и Wishart получают независимые post-boundary optimizer states.

## 10. Команды аудита

```powershell
$env:PYTHONPATH="src"
python -m pip install -e ".[dev]"
ruff check src scripts tests
python -m unittest tests.test_data -v
python -m unittest tests.test_model_math -v
python -m unittest tests.test_population_mstep -v
python -m unittest tests.test_population_df -v
python -m unittest tests.test_mixture_weights tests.test_damping -v
python -m unittest tests.test_base_schedule -v
python -m unittest tests.test_active_controls tests.test_active_resume -v
python -m unittest tests.test_fabric_smoke -v
python -m unittest tests.test_documentation -v
```

Во время длинного GPU-запуска ограничиться CPU unit/smoke tests и не менять его
Amazon CSV, output directory или два source-checkpoint каталога.

## 11. Критерий удаления кода и тестов

Production-модуль остаётся, если он достижим из CLI текущего метода, реализует
заявленный backbone/data format или проверяемый математический примитив.
Отдельный экспериментальный скрипт остаётся только если воспроизводит актуальную
таблицу/рисунок документации. Тест остаётся, если защищает численный,
статистический, split, resume или shape-инвариант; тест существования фабрики,
обёртки или имени класса сам по себе ценности не имеет.

Результат аудита оформлять таблицей: `severity`, `file/symbol`, нарушенный
инвариант, минимальный воспроизводящий тест, предлагаемое исправление. Сначала
исправляются leakage и неверная likelihood, затем resume/numerics, после этого
performance и структура кода.
