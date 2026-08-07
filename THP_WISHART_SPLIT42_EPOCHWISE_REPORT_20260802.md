# WiRE-THP on `sin_K5_C5`: fixed-split, epoch-wise report

Date: 2026-08-02

## Correct protocol

- Data split is fixed once with `numpy.RandomState(42).shuffle` and 80/10/10 proportions.
- Model seeds 0--4 change initialization, minibatches, and Wishart sampling, not the split.
- Original time is preserved. `sin_K5_C5` uses the fixed observation horizon 20.
- Training lasts 600 optimizer epochs.
- Validation purity is evaluated every optimizer epoch.
- The best checkpoint maximizes validation purity; validation NLL only breaks purity ties.
- Test is evaluated once after restoring the best validation checkpoint. Test never enters training, checkpoint selection, or the Optuna objective.
- Default Wishart Monte Carlo counts are train/validation/test = 2/8/16.

The cloned LaL repository confirms `check_val_every_n_epoch: 1` and checkpoint monitoring of `val/pur`. Its custom EM-era learning-rate callback is not transferred to joint WiRE training.

## Five-seed comparison

| Variant | Validation purity | Test purity | Test ARI | One-to-one accuracy | Seeds with 5/5 true modes | Test NLL/time |
|---|---:|---:|---:|---:|---:|---:|
| THP no-W | 0.808 +/- 0.010 | 0.710 +/- 0.014 | 0.507 | 0.649 +/- 0.016 | 0/5 | 3.631 |
| WiRE-THP | **0.819 +/- 0.004** | 0.729 +/- 0.007 | 0.547 | 0.656 +/- 0.015 | 0/5 | 3.701 |
| WiRE-THP + InfoMax 5 | 0.810 +/- 0.010 | 0.740 +/- 0.017 | **0.555** | 0.705 +/- 0.049 | 2/5 | 3.768 |
| WiRE-THP + InfoMax 10 | 0.811 +/- 0.007 | **0.749 +/- 0.019** | 0.543 | **0.723 +/- 0.052** | **3/5** | 3.888 |

Published external references for this dataset are LaL/Cohortney `0.92 +/- 0.05` test purity and the paper's Transformer Hawkes Process `0.51 +/- 0.01`. These are not paired implementations, but they provide scale.

The plain Wishart layer improves paired test purity over no-W by +0.019 on average. Its per-seed improvements are +0.030, +0.015, +0.010, +0.030, and +0.010. It also approximately halves the test-purity standard deviation.

## Main diagnosis

Both no-W and plain WiRE-THP use five predicted components but cover only four true processes on every model seed. One predicted component duplicates another true mode. Processes 0 and 4 are separated reliably; processes 1 and 2 are systematically confused.

Purity hides part of this collapse because it allows multiple predicted clusters to map to the same true class. Strict one-to-one matching exposes it: plain WiRE-THP has only 0.656 mean one-to-one accuracy despite 0.729 purity.

This explains why tuning Wishart parameters has a low ceiling: Wishart stabilizes the four-mode local solution but does not itself force all five output components to occupy different modes.

## Negative and neutral results

### Monte Carlo counts

For a fixed seed-0 checkpoint, validation purity is identical at 8, 16, 32, 64, and 128 Wishart samples. Test purity is identical at 16, 32, 64, and 128 samples. Test NLL Monte Carlo standard deviation is already approximately 3.7e-4 at 16 samples.

Increasing training samples also does not help on model seed 0:

| Train samples | Best validation purity | Test purity | Test ARI |
|---:|---:|---:|---:|
| 2 | 0.820 | 0.730 | 0.544 |
| 4 | 0.820 | 0.725 | 0.527 |
| 8 | 0.820 | 0.730 | 0.542 |

The efficient 2/8/16 setting is therefore retained.

### Fixed-split Optuna

Eight new trials plus the imported baseline were evaluated on model seed 0. The best trial reached validation purity 0.825 with:

- `nu = 34`
- omega learning rate `0.010096`
- alpha learning rate `0.000590`
- alpha temperature `1.943`
- mean hyperprior strength `1.566`

Its test purity remained 0.730, exactly the baseline value. The +0.005 validation gain is one validation object and did not transfer to test. The search nevertheless confirms omega learning rate near 1e-2.

### Frozen-Wishart exploration

Convex frozen exploration with beta 0.9 annealed linearly to zero over all 600 epochs produced validation/test purity 0.820/0.730 on seed 0, exactly the baseline. It is not scaled to more seeds under the corrected protocol.

## InfoMax anti-collapse diagnostic

The optional unlabeled regularizer is

`loss = negative log likelihood - lambda * (H(mean q(z|x)) - mean H(q(z|x)))`.

It uses model posterior assignments only; cluster labels are not used in the training loss. On this balanced synthetic dataset it tests whether explicit component utilization is the missing pressure.

- Lambda 5 restores all five modes on 2/5 seeds and improves one-to-one accuracy to 0.705.
- Lambda 10 restores all five modes on 3/5 seeds and improves one-to-one accuracy to 0.723.
- Stronger InfoMax worsens likelihood and increases seed variance. It is therefore evidence for the collapse diagnosis, not yet a finished WiRE variant.

The next theoretically coherent experiment is an early InfoMax curriculum: start strong enough to occupy all modes and anneal lambda to zero, so the final phase again optimizes the unmodified TPP marginal likelihood.

## Valid artifacts

- `artifacts/lal_epochwise_split42_validation_20260802/baseline_5seeds`
- `artifacts/thp_no_wishart_sin_K5_C5_split42_epochwise_5seeds_20260802`
- `artifacts/wishart_mc_convergence_sin_K5_C5_split42_seed0_20260802`
- `artifacts/optuna_thp_wishart_sin_K5_C5_split42_epochwise_20260802`
- `artifacts/wishart_convex_b09_t600_sin_K5_C5_split42_seed0_20260802`
- `artifacts/wishart_infomax5_sin_K5_C5_split42_5seeds_20260802`
- `artifacts/wishart_infomax10_sin_K5_C5_split42_5seeds_20260802`
- `artifacts/lal_epochwise_split42_validation_20260802/FINAL_SIN_K5_C5_COMPARISON.csv`

Artifacts made with validation intervals greater than one or data splits tied to model seeds are retained only for audit and are not used in this report.
