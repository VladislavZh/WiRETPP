# Математика и алгоритм Active Block Wishart TPP

## 1. Архитектурная граница

Active Block Wishart — вероятностная и обучающая обёртка над произвольным
нейросетевым TPP. Единственное требование к backbone — выдать банк
положительных интенсивностей

$$
h_{\theta,k}(t\mid\mathcal H_t)
=
\bigl(h_{\theta,k1}(t),\ldots,h_{\theta,kC}(t)\bigr)
\in\mathbb R_+^C,
\qquad k=1,\ldots,K.
$$

Ни Wishart decoder, ни variational inference не зависят от внутреннего
состояния backbone. В коде граница представлена `IntensityBank -> TPPTrace`.

Поддержанные adapters:

- COTIC: continuous convolution;
- THP: Transformer Hawkes Process;
- NHP: continuous-time LSTM;
- RMTPP: recurrent marked TPP.

Последние три используют нейросетевые блоки EasyTPP 0.2.1.

## 2. Генеративная модель

Для каждой траектории $m$:

$$
z_m\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_K),
$$

$$
U_m\mid z_m=k
\sim
\mathcal W_C\!\left(\nu,\frac{\Omega_k}{\nu}\right),
\qquad
\mathbb E[U_m\mid z_m=k]=\Omega_k.
$$

Для устранения скалярной неоднозначности используется ограничение

$$
\operatorname{tr}\Omega_k=C.
$$

Определим Wishart-преобразование базовой интенсивности

$$
T_U(h)=\left[U^{1/2}\sqrt h\right]^{\circ2}.
$$

Условная интенсивность траектории:

$$
\boxed{
\lambda_m(t\mid z_m=k,U_m)
=
(1-\alpha)h_{\theta,k}(t\mid\mathcal H_t)
+\alpha T_{U_m}\!\left(h_{\theta,k}(t\mid\mathcal H_t)\right)
+\varepsilon,
\qquad \alpha\in[0,1],\quad\varepsilon=10^{-6}
}.
$$

Корень и квадрат внутри $T_U$ берутся покоординатно, $U^{1/2}$ — главный
симметричный матричный корень. Интерполяция выполняется после преобразования,
в пространстве интенсивностей. Одна и та же формула с тем же $\varepsilon$
используется в событийном члене и компенсаторе.

Это active-block конструкция: при гипотезе $z=k$ локальная
$C\times C$ матрица действует только на $k$-й выходной блок. Размер
random effect не растёт с числом кластеров.

### Pure/no-W endpoint

При $\alpha=0$:

$$
A_0(U)=I_C,
\qquad
\lambda_m(t\mid z_m=k,U_m)=h_{\theta,k}(t\mid\mathcal H_t).
$$

Латентная матрица исчезает из likelihood, и модель точно совпадает с обычной
смесью выбранного backbone. Это вложенность модели, а не приблизительная
абляция.

## 3. Обычный marked-TPP likelihood

Для последовательности

$$
s_m=\{(t_{mi},c_{mi})\}_{i=1}^{N_m},
\qquad 0<t_{mi}\le T_m,
$$

компонентный log likelihood равен

$$
\ell_{mk}(U)
=
\sum_{i=1}^{N_m}
\log\lambda_{mk,c_{mi}}(t_{mi};U)
-
\int_0^{T_m}\sum_{c=1}^C\lambda_{mkc}(t;U)\,dt.
$$

В реализации интегратор выбирается независимо от backbone. Контрольный режим
использует Gauss--Legendre quadrature, а Monte Carlo режим сэмплирует на каждом
межсобытийном интервале $[a,b]$

$$
u_r\sim\operatorname{Uniform}(0,1),
\qquad
\widehat{\int_a^b \lambda(t)\,dt}
=
\frac{b-a}{R}\sum_{r=1}^R
\lambda\!\left(a+(b-a)u_r\right).
$$

Эта оценка несмещённая. Во время neural training точки пересэмплируются;
validation и test используют фиксированные common random numbers, чтобы шум
интеграла не влиял на выбор checkpoint и парное сравнение методов.
`TPPTrace` хранит:

$$
h_{\theta,k}(t_{mi}),
\qquad
h_{\theta,k}(\tau_{mr}),
\qquad
w_{mr},
$$

где $\tau_{mr}$ и $w_{mr}$ — точки и веса выбранного интегратора. После
построения trace локальный VI больше не вызывает backbone.

## 4. Вариационное семейство

Для каждой пары «траектория--компонента»:

$$
q_{mk}(U)
=
\mathcal W_C\!\left(
\kappa_{mk},\frac{Q_{mk}}{\kappa_{mk}}
\right),
\qquad
\mathbb E_q U=Q_{mk}.
$$

Локальная free energy:

$$
\mathcal F_{mk}
=
-\mathbb E_{q_{mk}}\ell_{mk}(U)
+
\operatorname{KL}\!\left[
q_{mk}(U)\,\|\,p(U\mid z=k)
\right].
$$

Responsibilities:

$$
\gamma_{mk}
=
\frac{\pi_k\exp(-\mathcal F_{mk})}
{\sum_j\pi_j\exp(-\mathcal F_{mj})}
=
\operatorname{softmax}_k(\log\pi_k-\mathcal F_{mk}).
$$

Полный ELBO:

$$
\mathcal L
=
\sum_{m,k}\gamma_{mk}
\left[
\mathbb E_q\ell_{mk}(U)
-\operatorname{KL}(q_{mk}\|p_k)
+\log\pi_k
-\log\gamma_{mk}
\right].
$$

### Параметризация local posterior

$Q_{mk}$ параметризуется Cholesky factor:

$$
Q=LL^\top,
\qquad
L_{ii}=\operatorname{softplus}(r_{ii})+10^{-4}.
$$

Degrees of freedom:

$$
\kappa=C-1+10^{-3}+\operatorname{softplus}(\rho),
$$

что гарантирует существование Wishart law. Градиенты проходят через
Bartlett decomposition; для диагональных gamma variables используется
pathwise реализация PyTorch.

## 5. KL между Wishart laws

Для

$$
q=\mathcal W_C(\kappa,Q/\kappa),
\qquad
p=\mathcal W_C(\nu,\Omega/\nu)
$$

используется аналитический KL. Его $\Omega$-зависимая часть:

$$
\frac{\nu}{2}
\left[
\log|\Omega|+\operatorname{tr}(\Omega^{-1}Q)
\right]
$$

с точностью до множителей и констант, не зависящих от $\Omega$.

Тест отдельно проверяет

$$
\operatorname{KL}(p\|p)=0.
$$

## 6. Exact population M-step для Omega

При фиксированных $q$ и $\gamma$ sufficient mean компоненты:

$$
S_k
=
\frac{\sum_m\gamma_{mk}Q_{mk}}
{\sum_m\gamma_{mk}}.
$$

Population M-step решает

$$
\min_{\Omega_k\succ0}
\left\{
\log|\Omega_k|
+\operatorname{tr}(\Omega_k^{-1}S_k)
\right\}
\quad
\text{s.t.}\quad
\operatorname{tr}\Omega_k=C.
$$

Оптимум коммутирует с $S_k$. Пусть $s_i$ и $\omega_i$ — их
соответствующие собственные значения. KKT condition:

$$
\frac1{\omega_i}
-\frac{s_i}{\omega_i^2}
+\lambda=0,
$$

то есть

$$
\boxed{s_i=\omega_i+\lambda\omega_i^2},
\qquad
\sum_i\omega_i=C.
$$

При $\lambda\ge0$ выбирается малая положительная ветвь quadratic root.
При $\lambda<0$ стационарная система может содержать одну большую ветвь;
solver перебирает допустимые branch configurations и выбирает кандидата с
минимальным objective. После решения eigenvalues ограничиваются снизу
$10^{-4}$ и повторно нормируются по trace для устойчивости float32.

Поэтому update

$$
\Omega\leftarrow CS/\operatorname{tr}S
$$

не используется: он сохраняет trace, но в общем случае не решает указанную
constrained задачу.

## 7. M-step для pi и alpha

Mixture weights имеют обычное closed-form обновление:

$$
\pi_k\leftarrow\frac1M\sum_m\gamma_{mk}.
$$

Для $\alpha$ фиксируются draws

$$
U_{mk}^{(r)}\sim q_{mk}
$$

и минимизируется common-random-number Monte Carlo objective

$$
J(\alpha)
=
-\sum_{m,k}\gamma_{mk}
\frac1R\sum_{r=1}^R\ell_{mk}(U_{mk}^{(r)};\alpha),
\qquad \alpha\in[0,1].
$$

Это одномерный bounded golden-section search. Обе границы $0$ и $1$,
а также предыдущее значение $\alpha$, всегда оцениваются явно.

## 8. Neural M-step

При фиксированных $q,\gamma,\Omega,\alpha,\pi$ backbone максимизирует

$$
\sum_{m,k}\gamma_{mk}
\mathbb E_{q_{mk}}\ell_{mk}(U;\theta).
$$

Локальные draws и responsibilities отсоединены от autograd; градиент идёт
только в параметры intensity bank. `mixture_logits` исключены из Adam,
поскольку $\pi$ уже получила отдельное closed-form обновление.

Lightning Fabric выполняет только:

- device/precision setup;
- backward;
- gradient clipping.

Цикл оптимизации остаётся обычным Python-кодом без LightningModule,
callbacks и скрытых optimizer hooks.

## 9. Базовый all-12 алгоритм

Pure и Wishart запускаются независимо. При одинаковых config и seed каждый
запуск воспроизводит один и тот же K-компонентный checkpoint после 10
minibatch-шагов. Далее neural budget одинаков:

$$
600=75\times8.
$$

Один Wishart outer cycle:

1. Заморозить backbone и построить train trace cache.
2. Выполнить 10 Adam-шагов local VI с двумя Wishart draws.
3. Посчитать $\gamma$; первые 6 циклов применить Sinkhorn-like balancing.
4. Полностью заменить $\Omega$ exact trace-constrained optimum.
5. Обновить $\pi$ средними responsibilities.
6. Если $\alpha$ не зафиксирована конфигурацией, найти её 25 итерациями
   bounded search, используя 4 draws.
7. Выполнить 8 neural M-steps с одним persistent Adam.
8. Оценить validation prior-predictive NLL с 64 prior draws.
9. Сохранить checkpoint, если validation NLL улучшился.

В базовом режиме нет damping, alpha trust region и повторного E-step после
изменения $\alpha$. Эти варианты могут добавляться дальше как отдельные
schedule-абляции относительно общей исходной точки.

## 10. Prior и posterior predictive

Для новой траектории:

$$
p(s_{new})
=
\sum_k\pi_k
\int p(s_{new}\mid k,U,\theta)
p(U\mid k,\Omega_k)\,dU.
$$

Validation checkpoint выбирается только по Monte Carlo оценке этого
prior-predictive likelihood. Локальный posterior validation/test траектории
не используется для выбора модели.

После выбора checkpoint можно оценить

$$
q_{new,k}(U)\approx p(U\mid s_{new},z=k)
$$

и posterior responsibilities. Это отдельная задача персонализации и
кластеризации, а не prior-predictive test score.

## 11. Что идентифицируется

Наблюдаемой является композиция

$$
(h_\theta,\alpha,U)
\longmapsto
\lambda(t),
$$

а не отдельная координатная запись $\Omega$. Backbone способен менять
базовый банк, а $\alpha$ частично обменивается с функциональным отклонением
$T_U(h)-h$. Поэтому raw Frobenius error $\Omega$ является вторичной
диагностикой.

Основные функциональные критерии:

- prior-predictive NLL gain относительно Pure;
- posterior/prefix-to-suffix prediction;
- ошибка восстановленной интенсивности;
- count и mark prediction;
- ARI и purity;
- устойчивость при разбиении контекста при общем trajectory-level $U$.

Wishart здесь следует интерпретировать как контролируемое стохастическое
расширение пространства функций интенсивности с persistent SPD random
effect. Физическое совпадение конкретной $\Omega$ не требуется для
функциональной корректности модели.

## 12. DAN protocol

Для каждого набора используется независимый deterministic split:

$$
80\%/10\%/10\%,\qquad seed=42,
$$

созданный `numpy.random.RandomState(42).shuffle`. Validation выбирает
checkpoint. Test используется один раз после выбора. Pure и Wishart получают:

- одинаковый split;
- одинаковую инициализацию;
- один shared short checkpoint;
- одинаковое число neural updates;
- одинаковый backbone adapter.

Один train-запуск оптимизирует только один метод. Дополнительного Pure EM
trainer нет: EM-подобная endpoint-абляция использует обычный Wishart schedule
с фиксированной $\alpha=0$.
