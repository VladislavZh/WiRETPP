# Shared-encoder latent-Wishart mixtures for neural TPPs

Этот репозиторий — самодостаточный snapshot последней ревизии эксперимента.
Здесь оставлено только сравнение одной и той же непрерывновременной TPP-модели
в трёх архитектурах (NHP, THP, COTIC), двух параметризациях кластеров
(`output_split`, `lal_fixed`) и двух вариантах objective (`no-W`, latent
Wishart). Предыдущие grid-, GP-Wishart-, MAP-​W-, Wishart-only- и random-walk
пилоты намеренно удалены.

Главный результат лежит в
[`artifacts/corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0`](artifacts/corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0).
Там находятся конфигурация, все 12 чекпойнтов, истории обучения, posterior
probabilities, learned Wishart means, сырые метрики и SHA-256 manifest.

## Что именно сравнивается

Данные имеют (K=3) скрытых класса и (C=5) типов событий. Для каждой из
архитектур NHP/THP/COTIC сравниваются две способы получить (K) компонент:

1. `output_split` — основной вариант. Существует ровно один encoder и один
   общий history state (h(t)). Только последний intensity head расширен с
   (C=5) до (K C=15) выходов. Блок (5k:5(k+1)) задаёт базовую
   интенсивность компоненты (k). Никаких трёх полных сетей нет.
2. `lal_fixed` — контроль, близкий к LaL. Encoder и (C)-мерный decoder общие,
   а кластерным является только начальное состояние CT-LSTM в NHP или BOS
   embedding в THP/COTIC. Модель инициализируется двумя multiplicative LaL
   splits из (K=1) в (K=3), затем (K) фиксирован. Split/merge/delete
   random walk во время обучения не используется.

Каждая параметризация обучается:

- как обычная конечная смесь (`no_wishart`);
- с общей латентной Wishart-распределённой attention matrix
  (`latent_wishart_attention`).

Это сравнение одной сети с расширенным выходом против LaL-параметризации, а не
сравнение трёх независимо обучаемых нейросетей. Файл
`architecture_audit.csv` подтверждает один encoder во всех 12 моделях; Wishart
добавляет ровно (15^2=225) параметров learned mean и не добавляет encoder.

## Данные

Генератор воспроизводит трёхкомпонентный пятиразмерный exponential Hawkes DGP.
Для класса (z) интенсивность типа (c) равна

\[
\lambda^{(z)}_c(t)=\mu^{(z)}_c+
\sum_{t_i<t}\alpha^{(z)}_{c,m_i}\beta^{(z)}_{c,m_i}
\exp[-\beta^{(z)}_{c,m_i}(t-t_i)].
\]

Параметры трёх классов генерируются один раз с `parameter_seed=20260746`, а
траектории — с `simulation_seed=20260747`. Горизонт фиксирован: (T=9.4),
число событий не нормируется и не фиксируется. В 1200 сгенерированных путях
среднее число событий равно 49.768 (диапазон 4–151). Разбиение с
`split_seed=20260748` стратифицировано по истинному классу:

- train: 90 путей, по 30 на класс;
- validation: 45, по 15;
- test: 1065, по 355.

Истинные labels используются только для финальных clustering metrics, не при
обучении и не при выборе чекпойнта. Полный аудит DGP записан в
`data_audit.json` внутри artifact-каталога.

## Обычная смесь без Wishart

Пусть нейросеть задаёт (K) положительных marked intensities

\[
\lambda_{\theta,k,c}(t\mid\mathcal H_t)>0.
\]

Для траектории (X=\{(t_i,c_i)\}_{i=1}^n) component log likelihood —
классический непрерывновременной TPP likelihood:

\[
\ell_k(X)=\sum_i\log\lambda_{\theta,k,c_i}(t_i\mid\mathcal H_{t_i})
-\int_0^T\sum_{c=1}^{C}\lambda_{\theta,k,c}(t\mid\mathcal H_t)\,dt.
\]

Интеграл считается Gauss–Legendre quadrature порядка 8 на каждом межсобытийном
интервале. Терминальный интервал от последнего события до (T) включён.
Смешивающие веса π обучаются через `softmax(mixture_logits)`, поэтому

\[
\log p_0(X)=\operatorname{logsumexp}_k[\log\pi_k+\ell_k(X)].
\]

Это полностью дифференцируемое совместное обучение; отдельного EM E-step в
коде нет. Posterior responsibility для кластеризации вычисляется уже из
нормализованного joint score.

## Латентная Wishart-attention

Обозначим (D=KC=15). Для каждой траектории существует новая латентная
матрица

\[
W\sim\operatorname{Wishart}_{D}(\nu,M/\nu),\qquad
\nu=20,\qquad \mathbb E[W]=M.
\]

Это не обучаемая матрица на каждую траекторию и не процесс (W(t)). Обучается
только глобальный параметр распределения (M\in\mathbb S_{++}^{15}). Пусть

\[
M_0=LL^\top,\qquad
M=D\,M_0/\operatorname{tr}(M_0),
\]

где нижний треугольник (L) свободен, а его диагональ проходит через
`softplus + 1e-5`. Trace constraint устраняет неидентифицируемый общий scale.
Для reparameterized sample берутся (G_s\in\mathbb R^{\nu\times D}) с
независимыми (N(0,1)) элементами:

\[
W_s=(G_sL^\top/\sqrt\nu)^\top(G_sL^\top/\sqrt\nu).
\]

### Как (W) входит в intensity

Из (W) строится correlation matrix и неотрицательная attention:

\[
R_{ab}=\frac{W_{ab}}{\sqrt{W_{aa}W_{bb}}},\qquad
A_{ab}=\frac{R_{ab}^{2}}{\sum_jR_{jb}^{2}}.
\]

Столбцы (A) суммируются в единицу. Поэтому attention перераспределяет
интенсивность между всеми (K\times C) каналами, включая cross-type и
cross-cluster связи, но сохраняет их суммарный instantaneous scale:

\[
\widetilde{\boldsymbol\lambda}(t;W)
=A(W)\boldsymbol\lambda_\theta(t),\qquad
\sum_a\widetilde\lambda_a(t;W)=\sum_a\lambda_a(t).
\]

Квадрат корреляции нужен потому, что intensity routing должен быть
неотрицательным, тогда как допустимая positive-definite (W) может иметь
отрицательные off-diagonal элементы. В этой реализации отрицательных
интенсивностей после умножения не возникает.

Одновременно diagonal block mass задаёт cluster prior:

\[
p(z=k\mid W)=
\frac{\sum_{c=1}^{C}W_{(k,c),(k,c)}}{\operatorname{tr}(W)}.
\]

Для фиксированных (W,z=k) используется тот же classical TPP likelihood, но
с (\widetilde\lambda_{k,c}(t;W)). Латентный кластер суммируется точно, а (W)
интегрируется Monte Carlo:

\[
\widehat{\log p_W(X)}=
\log\left[\frac1S\sum_{s=1}^{S}\sum_{k=1}^{K}
p(k\mid W_s)\exp\ell_k(X\mid W_s)\right].
\]

Важно: это не ELBO и variational distribution (q(W\mid X)) здесь нет.
Также это не closed-form objective: лог-маргинал аппроксимируется
reparameterized Monte Carlo; `log` от конечного sample average имеет обычное
MC-смещение. Train/validation/test используют соответственно (S=4/16/64),
а итоговый stability audit — три независимых повтора с (S=256).

Для стабилизации learned mean используется identity-centered penalty

\[
\mathcal R(M)=\frac{\tau}{2}
[\operatorname{tr}(M)-\log\det M-D],\qquad \tau=1.
\]

При trace normalization он минимален в (M=I).

## Как обучается модель

Для каждого из шести сочетаний architecture × parameterization создаётся одна
общая инициализация; её глубокие копии идут в no-W и Wishart branches.

No-W branch:

1. Берётся minibatch траекторий.
2. Для каждой компоненты строятся event log-intensity и quadrature compensator.
3. Кластеры маргинализуются `logsumexp`.
4. Adam обновляет neural parameters и `mixture_logits` с `lr=0.001`.

Wishart branch:

1. Строятся все (KC) базовых intensity channels общей сети.
2. Для каждой траектории и каждого из (S=4) samples генерируется fresh (W_s).
3. (W_s) используется и как attention, и для (p(z\mid W_s)).
4. Считается MC log-marginal выше плюс ℛ(M)/90.
5. Один Adam optimizer обновляет neural parameters с `lr=0.001`, weight decay
   `1e-5`, а raw Cholesky (M) — с отдельным `lr=0.01`, без weight decay.
6. Общий gradient norm clipping равен 20.

`backbone.mixture_logits` в Wishart branch намеренно не оптимизируется: cluster
prior там целиком задаётся (W). Никаких local (W_m), test-time fitting,
KMeans, E-step или LaL random walk нет.

Обучение длится 300 эпох, batch size 90 (то есть весь train), validation
проводится каждые 5 эпох. Выбирается checkpoint с минимальным validation
conditional suffix NLL. Cutoff равен 4.7, ровно половине горизонта.

## Метрики

`full marginal NLL/exposure` — минус полный log marginal, поделённый на
(N_{test}T).

`suffix NLL/exposure` проверяет prediction без утечки из будущего:

\[
\log p(X_{[4.7,9.4)}\mid X_{[0,4.7)})
=\log\sum_{s,k}p(W_s,z=k\mid X_{prefix})
p(X_{suffix}\mid W_s,z=k,X_{prefix}).
\]

Кластер — `argmax` posterior probability после маргинализации (W). Purity —
стандартная доля majority true label внутри каждого predicted cluster; рядом
обязательно смотрятся ARI и `active_k`, потому что purity сама по себе не
штрафует некоторые вырождения достаточно сильно. `prefix_half` использует
только первую половину, `full_sequence` — весь путь.

## Итоговые результаты

Ниже no-W значения взяты из детерминированной оценки сохранённого checkpoint,
Wishart — среднее трёх fresh-​W прогонов по 256 samples.

| architecture | parameterization | no-W suffix NLL | W suffix NLL | no-W purity / ARI | W purity / ARI |
|---|---|---:|---:|---:|---:|
| NHP | output-split | 3.551045 | 3.460095 | 0.9418 / 0.8351 | 0.9243 / 0.7947 |
| NHP | LaL-fixed | 3.577608 | 3.471363 | 0.4977 / 0.1038 | 0.9243 / 0.7936 |
| THP | output-split | 3.628967 | 3.463962 | 0.8789 / 0.6826 | 0.9509 / 0.8601 |
| THP | LaL-fixed | 3.535851 | 3.482394 | 0.5136 / 0.1062 | 0.9352 / 0.8177 |
| COTIC | output-split | 3.899844 | 3.636450 | 0.3333 / 0.0000 | 0.9427 / 0.8392 |
| COTIC | LaL-fixed | 3.872938 | 3.659475 | 0.4000 / 0.0126 | 0.8923 / 0.7268 |

Во всех 18 stability-evaluations Wishart использовал три активных кластера;
SD suffix NLL меньше 0.001. Wishart улучшил predictive suffix NLL во всех
шести paired comparisons. В NHP/output-split no-W кластеризует немного лучше,
поэтому утверждение «Wishart всегда лучше по purity» не делается.

## Воспроизведение

Проверенная среда: Python 3.11, PyTorch 2.12, NumPy 2.3.5, pandas 2.2.3,
EasyTPP wheel 0.2.1. COTIC source vendored из
`VladislavZh/COTIC@362b8ab1f3cbb9e9dced2518e9daacf68235e77a`.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install easy-tpp==0.2.1 --no-deps
.venv\Scripts\python -m pip install -e . --no-deps
$env:PYTHONPATH='src;scripts'
.venv\Scripts\python scripts/verify_artifacts.py
```

`--no-deps` для EasyTPP намерен: финальный код использует только THP и
`ModelConfig`; нерелевантные `datasets` и `tensorboard` в этом эксперименте не
нужны. Их runtime-зависимости (`omegaconf`, `PyYAML`, `packaging`) уже явно
зафиксированы. Linux/macOS эквивалентен (`.venv/bin/python`,
`export PYTHONPATH=src:scripts`).
Для CUDA сначала можно поставить подходящий PyTorch wheel, затем остальные
зависимости. Чекпойнты device-independent и загружаются на CPU.

Быстрый smoke run всех веток:

```powershell
.venv\Scripts\python scripts/run_corrected_shared_wishart_architectures.py `
  --quick --device cpu --outdir artifacts/smoke
```

Полное повторение (нужна CUDA; исходный прогон занял около 17 минут на
использованном устройстве):

```powershell
.venv\Scripts\python scripts/run_corrected_shared_wishart_architectures.py `
  --device cuda --epochs 300 `
  --outdir artifacts/corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0

.venv\Scripts\python scripts/audit_corrected_shared_wishart_mc.py `
  --device cuda --samples 256 --repeats 3 `
  --artifact artifacts/corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0
```

Повторный запуск в основной artifact-каталог перезапишет результаты. Для
проверки переноса сначала используйте другой `--outdir`. Seeds и все
гиперпараметры находятся также в `config.json`. CUDA/BLAS версии исходного
устройства не были записаны, поэтому bit-for-bit совпадение на другом GPU не
гарантируется; структура вывода и статистический результат воспроизводимы.

## Структура репозитория

```text
.
├── README.md                         # теория, алгоритм, команды и результаты
├── pyproject.toml                    # src-layout package metadata
├── requirements.txt                 # зафиксированные Python-зависимости
├── artifacts/
│   └── corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0/
│       ├── config.json               # полный протокол эксперимента
│       ├── environment.json          # версии и граница воспроизводимости
│       ├── data_audit.json           # DGP и статистика данных
│       ├── FINAL_COMPARISON.csv      # главная компактная таблица
│       ├── nll.csv                   # первичная NLL-оценка
│       ├── clustering.csv            # purity/ARI/NMI/entropy/active K
│       ├── architecture_audit.csv    # один encoder и размеры heads
│       ├── distribution_diagnostics.csv
│       ├── mc_stability_256x3*.csv   # fresh-W аудит
│       ├── checkpoints/              # 12 state_dict
│       ├── histories/                # validation traces
│       ├── learned_means/            # шесть M в long CSV
│       ├── predictions/              # posterior probabilities на test
│       ├── REPORT.md                 # автоматически собранный отчёт
│       └── MANIFEST.sha256            # целостность artifact
├── scripts/
│   ├── run_corrected_shared_wishart_architectures.py
│   ├── audit_corrected_shared_wishart_mc.py
│   └── verify_artifacts.py
├── src/lal_wishart/
│   ├── experiment.py                 # DGP splits, tables, manifest
│   ├── metrics.py                    # purity, ARI, NMI без sklearn
│   ├── data/                         # marked sequences и Hawkes primitives
│   ├── reproduction/paper_k3c5.py    # точный synthetic DGP
│   ├── models/
│   │   ├── neural_hawkes.py          # CT-LSTM/NHP
│   │   ├── reference_*               # shared NHP/THP/COTIC + LaL states
│   │   ├── wishart_math.py            # attention, gates, MC posterior
│   │   └── latent_wishart_attention_*.py
│   └── train/fit_latent_wishart_attention_nhp.py
├── reference/COTIC/                  # минимальный vendored upstream snapshot
└── tests/                            # unit/smoke tests последней модели
```

## Где продолжать эксперименты

Чтобы менять способ, которым (W) превращается в attention или cluster gate,
начинайте с `src/lal_wishart/models/wishart_math.py`. Распределение (W), trace
normalization и sampling находятся в `latent_wishart_attention_nhp.py`.
Архитектурные heads — в `reference_output_mixtures.py`, LaL controls — в
`reference_neural_lal.py` и `reference_bos_lal.py`, objective/checkpoint
selection — в `fit_latent_wishart_attention_nhp.py`. Основной runner содержит
только orchestration и сохранение результатов.
