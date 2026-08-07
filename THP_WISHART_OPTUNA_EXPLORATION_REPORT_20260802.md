# THP-Wishart: Optuna and frozen-Wishart exploration

Date: 2026-08-02

## Scope and protocol

- The broad 12-dataset THP run was stopped on request after 39/120 jobs.
- Hyperparameter diagnosis was moved to `sin_K5_C5`, the hardest dataset by
  the stable gap to the published LaL result (`0.92 +/- 0.05` test purity).
- Original timestamps and the fixed observation horizon 20 were preserved.
- Model selection maximized validation purity, breaking ties by lower
  validation NLL.
- Test labels and the test split were not used by Optuna, candidate ranking,
  the beta sweeps, or the annealing sweeps.
- All main tuning runs used 600 epochs, validation every 25 epochs, and 8
  fixed-seed Monte Carlo validation samples.

## Optuna study

The persistent SQLite study contains 24 trials: 23 completed and one failed
the positive-definiteness guard at `lr_omega=0.0773`. The current baseline was
enqueued as trial 0.

| configuration | nu | lr omega | lr alpha | alpha temperature | prior | best validation purity |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25 | 1e-2 | 1e-3 | 0.5 | 1.0 | 0.785 |
| trial 3 | 100 | 2.521e-3 | 9.966e-4 | 0.725 | 0.796 | 0.795 |
| trial 10 | 154 | 1.013e-2 | 7.740e-4 | 1.384 | 0.285 | 0.795 |
| trial 14 | 81 | 2.779e-3 | 5.824e-3 | 1.836 | 0.801 | 0.795 |

No later TPE trial exceeded 0.795. The only common signal among the three
different optima was a larger learned-Wishart `nu` (81--154 rather than 25).
The gain was only 0.01 and identical reruns showed GPU/objective variation up
to roughly 0.02, so seed-0 ranking alone was not trusted.

On validation seeds 1 and 2, trial 3 scored `0.810` and `0.770` versus the
baseline's `0.785` and `0.785`: mean `0.790` versus `0.785`, with substantially
higher variability. Trials 10 and 14 did not improve the mean. The Optuna
result therefore did not establish a robust hyperparameter win.

## Frozen additive Wishart process

The model now supports an independent, isotropic, non-trainable process

`W_effective = W_learned + beta * W_random`,

where `W_random ~ Wishart(nu_random, I / nu_random)`. The sum remains SPD and
the exploration law has no trainable parameters. `beta=0` follows the exact
old sampling path.

Both the signed coupling and the cluster gates are globally scale invariant,
so dividing `W_effective` by `1 + beta` is mathematically redundant. This was
verified by tests.

A constant-beta pilot on validation seeds 0 and 1 gave:

| beta | mean best validation purity |
|---:|---:|
| 0 | 0.7800 |
| 0.01 | 0.7850 |
| 0.03 | 0.7825 |
| 0.10 | 0.7900 |
| 0.30 | 0.7800 |

The non-monotone maximum at beta 0.1 was consistent with a small regularizer,
but constant beta changes inference and did not prevent late purity decline.

## Gradual convex exploration

The train-only curriculum uses

`W_t = (1 - beta_t) * W_learned + beta_t * W_random`,

with linear `beta_t -> 0`. Validation is always evaluated with `beta=0`, and
the returned final model also has `beta=0`; the extra process therefore changes
only the optimization path, not the final WiRE-TPP definition.

The initial grid used `beta_0 in {0.5, 0.9}`, annealing endpoints
`{150, 300, 600}`, and `nu_random=200`. Seed 0 showed no gain; seed 1 favored
the longest strong curriculum by 0.01. A paired five-seed check of the best
schedule (`beta_0=0.9`, endpoint 600) gave:

| configuration | mean best validation purity | paired delta |
|---|---:|---:|
| baseline | 0.768 | -- |
| gradual convex exploration | 0.771 | +0.003 |

The paired signs were `0, +0.010, +0.010, 0, -0.005`, which is too small and
inconsistent to claim improvement. Varying `nu_random` over `{25, 50, 100,
200}` on seed 0 also failed to beat the paired baseline.

## ReduceLROnPlateau control

The LaL paper source specifies factor 0.5 and tolerance/patience 25, but its
linked implementation is no longer publicly reachable. Applying that policy
to the per-epoch minibatch MC train loss was too aggressive: all parameter
group learning rates fell to approximately `1e-6`, and validation purity fell
from 0.790 to 0.735 (0.730 with gradual beta). This scheduler mode is optional
and disabled by default.

Stepping on validation NLL is not expected to solve the observed failure by
itself: validation NLL frequently keeps improving while purity degrades, which
is the central likelihood/clustering mismatch diagnosed in these runs.

## Conclusion

1. Validation-purity checkpoint selection is required for a comparison that
   matches LaL's supervised model-selection protocol, but it is noisy and does
   not robustly close the gap on `sin_K5_C5`.
2. Larger learned-Wishart `nu` is the only recurring Optuna signal, but its
   seed robustness is weak.
3. The frozen-Wishart exploration idea is mathematically valid and now has an
   exact train-only annealed implementation. Its measured effect is small
   (`+0.003` mean validation purity) in the tested form.
4. The remaining gap to LaL is likely structural: LaL's EM procedure, random
   walk in the number of clusters, deletion of extra clusters, and large-subset
   optimization change the basin-selection dynamics more strongly than the
   tested Wishart metaparameters.
5. Test evaluation of the beta schedules was intentionally not performed,
   because no candidate won convincingly on validation.

## Artifacts

- Optuna database and trials: `artifacts/optuna_thp_wishart_sin_K5_C5_20260802`
- Optuna multi-seed validation: `artifacts/optuna_thp_wishart_multiseed_validation_20260802`
- Constant beta: `artifacts/thp_wishart_additive_beta_sweep_seed0_20260802`
  and `artifacts/thp_wishart_additive_beta_sweep_multiseed_20260802`
- Annealing grids: `artifacts/thp_wishart_annealed_convex_seed0_20260802`,
  `artifacts/thp_wishart_annealed_convex_multiseed_20260802`, and
  `artifacts/thp_wishart_annealed_convex_finalists_20260802`
- Exploration-nu grid: `artifacts/thp_wishart_annealed_exploration_nu_seed0_20260802`
- Scheduler ablation: `artifacts/thp_wishart_scheduler_beta_ablation_seed0_20260802`

The final test suite passed 55/55 tests.
