# Аудит Amazon COTIC–Wishart, 2026-08-29

## Вывод

Явной ошибки в математике Wishart, параметризации KL, COTIC-архитектуре,
разбиении данных или интеграле интенсивности не найдено. Текущий разрыв с
Pure K=5 объясняется прежде всего не тем, что Wishart ухудшает готовую модель,
а тем, что нейросетевой backbone под variational-EM обучается заметно медленнее.

К cycle 9 исправленного random-Omega запуска:

| Модель/диагностика | Число neural updates | validation NLL |
|---|---:|---:|
| Wishart prior predictive | 72 | 1.489421 |
| Тот же Wishart-trained backbone без Wishart-оператора | 72 | 1.605569 |
| Pure K=5 | 72 | 1.409585 |

Wishart-слой улучшает собственный backbone на 0.116148 NLL, но backbone отстаёт
от Pure на 0.195984. Итоговый разрыв Wishart–Pure равен 0.079836. Это исключает
гипотезу, что основная проблема находится только в Omega, df или prior-predictive
формуле.

## Что проверено и признано корректным

- Реализация paper COTIC совпадает с локальной upstream-реализацией: encoder
  совпал побитно, event-head — с максимальной абсолютной погрешностью
  4.77e-7. Восемь encoder-слоёв плюс финальный intensity-слой соответствуют
  описанной в статье 9-layer architecture 192/512.
- Один новый вектор равномерных MC-точек берётся на каждый forward, общий для
  всех интервалов этого forward. Validation использует повторяемые точки.
- Wishart KL проверен независимо против Monte Carlo из
  `torch.distributions.Wishart`: расхождение 0.00242 при стандартной ошибке
  0.00497.
- Bartlett sampler использует параметризацию `Wishart(df, mean / df)` и имеет
  правильное математическое ожидание.
- Responsibility update `softmax(log pi - free_energy)`, local objective,
  trace-constrained Omega M-step и alpha profile соответствуют записанному ELBO.
- Validation NLL не содержит KL или других variational-слагаемых: это
  prior-predictive point-process NLL, структурно сопоставимый с Pure.
- Amazon split не течёт: official val/test являются дублями, дубль test отброшен,
  оставшийся holdout детерминированно разделён. P99 exponential normalizer
  оценивается только по train; Unix seconds сначала переводятся в дни.
- Random Omega initialization действительно случайная, детерминированная,
  SPD и trace-normalized. В cycle 8 все пять компонентов живы; матрицы Omega
  не являются ни единичными, ни одинаковыми.
- Во время аудита прошли 42 целевых CPU-теста. До запуска данной ветки проходил
  полный набор 66/66 и CUDA smoke.

## Приоритетные проблемы

### P1. Neural M-step имеет гораздо более шумный и устаревающий сигнал, чем Pure

Pure на каждом optimizer update точно пересчитывает discrete mixture marginal
через `logsumexp`. Wishart один раз на outer cycle оценивает q и gamma, затем
держит их фиксированными восемь neural updates. Для каждого такого update
градиент дополнительно оценивается всего по четырём `q(U)` samples.

Это корректный generalized-EM, но равные числа optimizer updates не означают
равные эффективные бюджеты оптимизации. Здесь одновременно действуют:

1. Monte Carlo variance от `local_samples=4` в neural M-step;
2. усреднение градиентов по пяти компонентам и full-matrix transformations;
3. stale q/gamma на протяжении восьми updates;
4. восемь последовательных updates только на текущей половине train
   (`em_batch_size=3009`), тогда как Pure сэмплирует из всего train.

Наблюдаемое отставание backbone является прямым эмпирическим подтверждением
этого bottleneck, хотя относительный вклад каждого из четырёх механизмов пока
не измерен.

### P1. Responsibilities недостаточно диагностируются

Логи содержат только массы и effective K. Они не отличают два принципиально
разных режима:

- пять хорошо разделённых кластеров примерно одинакового размера;
- почти uniform gamma для каждой траектории.

Во втором режиме все головы получают почти одинаковый градиент, а независимый
full 8x8 local posterior q(U) может поглощать различия между траекториями вместо
кластеризации. Это правдоподобная причина того, что K=5 Wishart не получает
выигрыш Pure K=5, несмотря на живые массы.

Дополнительно sharded E-step при склейке `LocalPosterior` теряет агрегированные
флаги convergence, relative objective change и gradient diagnostics. Поэтому
сейчас мы фактически подбираем метапараметры вслепую.

### P1. Конечный Wishart MC даёт не нейтральную к Pure оценку NLL

Код правильно считает `log(mean_s p(sequence | U_s))`, однако логарифм
конечного MC-среднего имеет Jensen bias: ожидаемый log-likelihood занижен, а NLL
завышен. Pure не имеет этого дополнительного интеграла.

На cycle 4 одного и того же random-Omega checkpoint оценка с 64 samples была
1.629012, а объединённая оценка MC64 x 3 — 1.617010. Разница 0.012 включает и
bias, и MC noise; она почти равна тогдашнему разрыву с Pure 1.613213. Текущий
разрыв cycle 9 значительно больше и одной этой причиной не объясняется, но для
публикационного сравнения samples=64 недостаточно.

### P2. Learned df пока не является полноценной непрерывной оптимизацией

Профиль строится по девяти log-spaced точкам между 8 и 256 с добавлением
текущего df. После перехода 64 -> 49.265 текущая точка выигрывает у грубых
соседей 45.255 и 69.793, из-за чего поиск может залипнуть. Это не доказывает,
что непрерывный optimum равен 49.265.

Нужен bounded scalar refinement (Brent/golden/Newton) внутри найденного
интервала по тому же exact KL profile. Это улучшит корректность learned-df, но
по текущей динамике не выглядит главной причиной проигрыша.

### P2. Omega damping не является exact online M-step

Сначала вычисляется constrained optimum Omega на блоке, затем сами матрицы
усредняются арифметическим EMA. Из-за нелинейного KKT-решения это не эквивалентно
EMA sufficient statistics с последующим exact solve. Подход остаётся допустимым
damped generalized-EM, но может замедлять или смещать адаптацию Omega.

Это стоит исправить после проверки gradient SNR и responsibilities, а не до неё:
текущие Omega уже разделены, и Wishart-оператор улучшает собственный backbone.

### P3. Alpha около единицы не является сам по себе ошибкой

Alpha profile корректно оптимизирует текущий variational Q, и прежняя frozen
alpha-grid проверка также предпочитала alpha=1. Возможен variational mismatch:
train ELBO может предпочитать alpha=1 сильнее, чем истинный validation marginal,
но это надо проверять высокоточным frozen-checkpoint alpha-grid, а не ещё одним
длинным обучением.

## Минимальная программа следующих проверок

Все три проверки выполняются на frozen cycle-9 checkpoint и не меняют модель.

1. **Gradient-SNR audit (главный тест).** На одном фиксированном effective batch
   из 140 trajectories получить 20 повторных neural gradients для Wishart
   samples S = 1, 4, 16, 64 и один S = 256 reference. Логировать norm, cosine к
   reference, coordinate CV и долю clipped gradients; рядом вычислить точный
   Pure gradient. Это займёт минуты и прямо скажет, достаточно ли S=4.
2. **Responsibility audit.** Выполнить один E-step без parameter updates и
   сохранить mean normalized entropy, quantiles `max gamma`, mutual information
   `I(path; k)`, pairwise free-energy gaps, hard-assignment proportions и
   расстояние q от population prior. Если normalized entropy близка к 1, K=5
   Wishart практически не кластеризует.
3. **Nested-MC validation audit.** Для одного checkpoint использовать вложенные
   наборы prior draws S = 64, 128, 256, 512, чтобы построить NLL-versus-1/S и
   доверительный интервал. На тех же draws проверить alpha grid
   {0, .25, .5, .75, 1}.

Только после этих трёх проверок оправдан короткий paired training experiment:

- если S=4 шумный — две одинаковые ветки по 16 updates, S=4 против S=16;
- если gamma устаревает, но SNR хороший — одинаковые 16 updates с refresh q/gamma
  каждые 8 против каждых 2 updates;
- если gamma почти uniform — сначала пересмотреть variational family или
  regularization/identifiability, а не увеличивать число циклов;
- если high-S validation убирает разрыв — обучение не менять, а повысить только
  evaluation budget и честно описать estimator.

## Что сейчас не стоит делать

- снова перебирать Omega initialization: random initialization уже работает;
- запускать полный grid по df или learning rate;
- увеличивать число циклов вслепую;
- делать вывод о кластеризации по masses/effective K без entropy/MI;
- публиковать Wishart–Pure NLL по одному MC64 estimate.

## Результаты frozen-checkpoint проверок

Проверки выполнены после остановки обучения на последнем целостном cycle 10
checkpoint. Test split не читался, optimizer updates не выполнялись.

### Gradient SNR

На настоящем effective batch из 140 train trajectories, относительно одного
S=128 reference:

| Wishart samples | repeats | mean cosine | L2 SNR | mean relative L2 error |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0.4198 | 0.9157 | 1.1065 |
| 16 | 8 | 0.4227 | 0.9697 | 1.0308 |

Увеличение S с 4 до 16 практически не меняет направление градиента. При S=4
семь из восьми реализаций превышали global clip threshold 1.0; нормы лежали от
0.998 до 3.297. На pilot batch 32 редкие выбросы сохранялись даже при S=64
(нормы до 9.33), поэтому шум не ведёт себя как безопасный Gaussian `1/sqrt(S)`.

Вероятный механизм — destructive cancellation после signed full-matrix
преобразования амплитуд. При alpha около 1 отдельная координата
`sqrt(U) sqrt(lambda)` может оказаться почти нулевой; последующий square и
`log(rate)` создают тяжёлый градиентный хвост возле нуля. Текущий
`clamp_min(1e-12)` ограничивает только уже пересёкшие floor значения, но не
ограничивает большой градиент непосредственно выше floor.

### Responsibilities

На всех 6018 train trajectories:

- mean normalized entropy: 0.130261;
- median max responsibility: 0.992355;
- normalized path/component mutual information: 0.851735;
- median top-two logit gap: 4.9621;
- hard counts: [1345, 1681, 958, 864, 1170].

Следовательно, компоненты разделены уверенно; гипотеза uniform gamma отвергнута.

### Nested validation MC и alpha

Для alpha=1 combined NLL монотонно стабилизировался:

| Samples на repeat | Repeats | Effective samples | Validation NLL |
|---:|---:|---:|---:|
| 64 | 3 | 192 | 1.456096 |
| 128 | 3 | 384 | 1.451816 |
| 256 | 3 | 768 | 1.447724 |
| 512 | 3 | 1536 | 1.446174 |

При effective MC1536 alpha-grid дал:

| alpha | Validation NLL |
|---:|---:|
| 0.00 | 1.579945 |
| 0.25 | 1.524309 |
| 0.50 | 1.487763 |
| 0.75 | 1.462390 |
| 0.999204 | 1.446213 |
| 1.00 | 1.446174 |

Alpha=1 подтверждён, а MC64 selection score 1.466392 завышал NLL примерно на
0.020. Однако Pure K=5 на тех же 80 neural updates имеет NLL 1.373333, поэтому
даже MC1536 оставляет разрыв 0.072840. Wishart-слой при этом улучшает собственный
backbone с 1.579945 до 1.446174 на 0.133772: основной остаточный разрыв снова
находится в обучении backbone.

### Smooth intensity-floor probe

Paired frozen-gradient floor test выполнен на том же batch 140: текущий
`square().clamp_min(1e-12)` сравнивался с valid-intensity smoothing

`rate = transformed.square() + alpha**2 * epsilon`

для epsilon `{1e-8, 1e-6, 1e-4}`. Для каждого epsilon измерены S4 gradient
cosine/SNR и common-draw MC512 validation NLL без parameter updates. Множитель `alpha**2`
сохраняет точный Pure endpoint при alpha=0, а добавка входит и в event term, и в
compensator, поэтому likelihood остаётся согласованным.

| Floor | Mean cosine | L2 SNR | Norm std | Frozen NLL | Delta NLL |
|---|---:|---:|---:|---:|---:|
| baseline | 0.4198 | 0.9157 | 0.7359 | 1.450264 | 0 |
| epsilon=1e-8 | 0.5603 | 1.5511 | 0.1965 | 1.449911 | -0.000353 |
| epsilon=1e-6 | 0.6129 | 4.0416 | 0.0551 | 1.449357 | -0.000907 |
| epsilon=1e-4 | 0.5648 | 6.5781 | 0.0564 | 1.446506 | -0.003758 |

Результат подтверждает singular-gradient механизм. Консервативный epsilon=1e-6
улучшает SNR в 4.4 раза, заметно улучшает cosine, практически устраняет выбросы
нормы и не платит frozen NLL. Epsilon=1e-4 ещё сильнее стабилизирует норму, но
заметнее меняет intensity family и имеет худший cosine, поэтому первым брать его
не следует.

Следующий единственный оправданный training experiment — двухцикловая paired
continuation от одного cycle-10 checkpoint: неизменный baseline против
`rate = transformed.square() + alpha**2 * 1e-6`. Все остальные параметры,
draws/seeds, blocks и neural updates должны совпадать. Просто S4 -> S16 запускать
не надо.

## Ключевые участки кода

- Neural q(U) sampling и фиксированные gamma:
  `src/wishart_tpp/training/active.py`, строки 168–256 и 826–830.
- Exact Pure marginal на каждом update:
  `src/wishart_tpp/training/pure.py`, строки 79–104.
- Responsibility update и отсутствующие entropy diagnostics:
  `src/wishart_tpp/inference/responsibilities.py`, строки 13–29;
  `src/wishart_tpp/training/active.py`, строки 499–533.
- Потеря local convergence diagnostics при склейке shards:
  `src/wishart_tpp/training/active.py`, строки 377–405.
- Prior-predictive MC:
  `src/wishart_tpp/model/active_block.py`, строки 101–136;
  `src/wishart_tpp/training/evaluation.py`, строки 95–134.
- Coarse df grid:
  `src/wishart_tpp/inference/population_df.py`, строки 23–49.
- Arithmetic Omega EMA:
  `src/wishart_tpp/inference/damping.py`, строки 9–16.
- Train-only P99 normalization и deduplicated split:
  `src/wishart_tpp/data.py`, строки 364–474 и 502–517.
