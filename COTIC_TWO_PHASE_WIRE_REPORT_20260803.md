# Two-phase plug-in COTIC WiRE experiment (2026-08-03)

## Dataset and selection protocol

- Dataset: `K5_C5`.
- Split: NumPy shuffle with seed 42, then 80/10/10.
- Initialization seed: 2026123100 (experiment seed 0).
- Model selection: validation purity at every optimizer step, enabled only
  after backward elimination reaches K=5. Validation NLL breaks purity ties.
- Test is evaluated once from the selected validation checkpoint.

## Universal two-phase protocol

1. Fit an ordinary K=1 COTIC for 300 optimizer steps.
2. Clone only its C-channel affine intensity output into Kmax=10 blocks.
   The shared encoder and fitted per-mark intensity scale are preserved;
   sibling affine blocks receive noise with standard deviation 0.01.
3. Freeze the shared COTIC encoder while heads, Wishart parameters, alpha,
   beta exploration, DPP repulsion, and the global UOT dual train.
4. Prune one component at steps 300, 400, 500, 600, and 700 using the
   smallest increase in full-train physically-pruned marginal NLL.
5. At K=5, decay DPP from 0.1 to zero over 200 steps and linearly ramp the
   encoder learning rate from zero to 1e-5 over the same interval. The head
   learning rate remains 1e-3.
6. Continue for 600 post-pruning optimizer steps while rho rises 5 -> 50.

The functional DPP signature is the standardized vector of observed-mark
log intensities on the common batch histories. Its Gaussian kernel uses
bandwidth 1 and jitter 1e-4. The penalty is `-logdet(G + eps I) / K`.

Wishart settings: nu=int(1.5*K*C), hence 75 -> 37; alpha0=0.5,
alpha temperature 1.0, alpha lr 5.9e-4, Omega lr 0.010096; convex frozen
Wishart exploration beta0=0.9 with exponential decay to exactly zero.

## Result

| Metric | Value |
|---|---:|
| Best validation purity | 0.995 |
| Best step | 1300 |
| Test purity | 1.000 |
| Test ARI | 1.000 |
| Test NMI | 1.000 |
| Test NLL / exposure | 2.0708 |
| Test cluster sizes | 41, 45, 41, 41, 32 |
| Final alpha | 0.6146 |
| Train marginal L1 to uniform | 0.0159 |
| Full-train UOT stationarity error | 0.00281 |

The test cluster sizes exactly match the true K5 sizes up to permutation.
The validation purity remained 0.995 after DPP reached zero and the encoder
reached its full 1e-5 learning rate.

## Comparisons on the same split/seed

| COTIC configuration | Val purity | Test purity | Test ARI |
|---|---:|---:|---:|
| no-W, direct joint training | 0.690 | 0.665 | -- |
| Wishart, direct joint training | 0.780 | 0.825 | 0.697 |
| K=1 pretrain + frozen/DPP plug-in WiRE | 0.995 | 1.000 | 1.000 |

The result is also competitive with the THP overcomplete controls:
THP no-W obtained test purity 0.985, and THP WiRE with alpha temperature
1.943 obtained 0.980.

## Alpha-temperature ablation on THP

| Alpha temperature | Best val | Test purity | Test ARI | Selected alpha |
|---:|---:|---:|---:|---:|
| 1.943 | 0.990 | 0.980 | 0.951 | 0.586 |
| 1.0 | 0.985 | 0.975 | 0.940 | 0.663 |
| 0.5 | 0.825 | 0.810 | 0.720 | 0.788 |

Temperature 1.0 is a safe, more responsive alternative. Temperature 0.5
amplifies the initial raw-alpha derivative by about 3.9x relative to 1.943,
drives alpha too high too early, and locks in an inferior partition.

## Gradient audit

| Phase | Clip fraction | Median raw norm | Maximum raw norm |
|---|---:|---:|---:|
| K=1 density pretrain (300 steps) | 0.620 | 22.29 | 775.69 |
| Frozen clustering, K=10 -> 5 (700 steps) | 0.000 | 7.06 | 15.12 |
| Low-lr unfrozen K=5 (600 steps) | 0.548 | 20.74 | 61.70 |
| Previous direct joint COTIC WiRE | 0.621 | 21.75 | 6641.90 |

The important stabilization occurs during cluster formation: no frozen-phase
step is clipped. After unfreezing, the median norm lies only slightly above
the threshold 20 and purity does not degrade. Direct joint training had
rare extreme gradients two orders of magnitude larger.

## Artifacts

- Main run: `artifacts/dan_cotic_k5_k1pretrain300_frozen_dpp0p1_wishart_overcomplete10to5_nu1p5_betaexp_rho5to50_alphatemp1p0_acc2_seed0_20260803`
- COTIC direct WiRE control: `artifacts/dan_cotic_k5_signed_wishart_overcomplete10to5_nu1p5_betaexp_rho5to50_alphatemp0p5_acc2_seed0_20260803`
- THP temperature 1.0: `artifacts/dan_thp_k5_signed_wishart_overcomplete10to5_nu1p5_betaexp_rho5to50_alphatemp1p0_acc2_seed0_20260803`
- THP temperature 0.5: `artifacts/dan_thp_k5_signed_wishart_overcomplete10to5_nu1p5_betaexp_rho5to50_alphatemp0p5_acc2_seed0_20260803`

All 70 unit tests pass after the implementation changes.

## Scope of the conclusion

This is one dataset and one model seed. The mechanism is strongly supported
by the trajectory and ablation controls, but the numerical result needs
multi-seed replication and a factorized ablation of K=1 scale preservation,
encoder freezing, and DPP before it is treated as a general benchmark result.
