# Fixed training protocol

This release contains only Direct Pure and the non-centered Bartlett/all-gradient
Active Wishart method. It is a code cleanup, not a new experimental result.
Historical results from different protocols must not be relabelled as results of
this release. In particular, the completed temperature-based DAN comparison used
a different training protocol.

## Common boundary and neural bank

Both methods receive the same seeded K=1 model after exactly 10 optimizer steps,
then the same native noisy K-head expansion. Each active method starts a new Adam;
pretraining moments are not transferred. No data-dependent head calibration occurs.
COTIC defaults are 192 input channels, 512 hidden units, 8 layers and zero dropout.
The optional THP adapter is the corrected residual/LayerNorm encoder, not the
historical EasyTPP implementation. Changing the backbone does not change the mixture
or random-effect law.

Time normalization is fitted on train only using the common train-P99 exponential
procedure. Synthetic splits use seed42. Real splits must be disjoint. StackOverflow
requires the approved separately prepared leakage-free grouped split; do not reuse
the original overlapping file. Raw datasets are not distributed in Git.

## Objectives

For component k, the bank produces positive intensities h_k. The random effect is
U ~ Wishart_C(df, Omega_k/df); each Omega_k is positive definite and has trace C.
The intensity is

    lambda = (1-alpha) h + alpha (sqrt(U) sqrt(h))^2 + 1e-6.

The matrix square root is symmetric. Mixture logits are independent trainable
parameters: no routing weights are computed from U. Direct Pure optimizes the
ordinary log-sum-exp mixture marginal at every update.

Wishart fits local q(U|trajectory,k), then holds responsibilities and **relative**
posterior coordinates fixed during M. At the start of M, A=chol(Omega_anchor).
For the current B=chol(Omega), anchored Bartlett draws are transported by B A^-1.
Likelihood gradients consequently reach Omega. The full analytic Wishart KL is
invariant to this common congruence; its local value remains in the ELBO.
This is not fixed-absolute-q spectral EM, nor IWAE, nor differentiation through E.

## Defaults

| Setting | Value |
|---|---|
| Active budget | 60 cycles x 8 updates = 480 per method |
| Total neural updates including shared | 490 |
| Neural / head / mixture LR | 1e-4 |
| Omega LR / alpha LR | 1e-4 / 0.01 |
| Neural / mixture Adam | beta=(0.9, 0.999), eps=1e-8 |
| Omega / alpha Adam | beta=(0.9, 0.95), eps=1e-8 |
| Weight decay | neural 1e-5; Omega/alpha/pi zero in Wishart |
| Joint gradient clipping | norm500 before Adam; not a step-displacement cap |
| Alpha | start0.1, projected to [0,1] after Adam |
| Omega initialization | symmetric trace-zero noise0.05 around identity |
| Effective batch | 256 |
| Local E | L8, 4 fit draws, 256 scoring draws |
| M samples | 64 |
| E fit grouping | 64 trajectories with native sampling chunk order |
| EM block | 4096, fresh local q between blocks |
| Compensator | 50 Monte Carlo time samples, strict FP32 |
| Ordinary validation | MC64 every cycle |
| Selected Wishart validation/test | MC64 x 3, no reselection |
| Fixed df | 16 Retweet/Amazon; 32 StackOverflow/DAN |

Compute shards do not alter the effective batch or truncate trajectories. There
are no scheduler, temperature, Sinkhorn balance, head untying, freeze, population
warm-up, damping, learned population df, full KCxKC matrix, oracle branch or baseline
training mode. For a full-train EM block, local means may warm-start the next E step;
across changing blocks, no posterior or responsibilities are retained.

Pure selection includes cycle0. Native Wishart selection starts at cycle1; cycle0
is recorded as a diagnostic. Test never selects a checkpoint and has no local-q
fitting. Pure/Wishart comparisons must use equal-update curves or explicitly
identified, independently validation-selected states. Real labels=-1 have no
interpretable purity/ARI; unavailable metrics are written as null.

## Compatibility and verification

`active.pt` is a new, versioned checkpoint format. It binds configuration, ordered
data, package source, and the common boundary, and stores model, raw population
coordinates, Adam, integration seed/counter, and Python/NumPy/PyTorch RNG states. Old experiment checkpoints
are not silently accepted. Selected states are separate from active resume states.
Completion seals active/selected/result hashes; repeating a completed run verifies
these artifacts and returns the result without rescoring test.

The cleanup preserves two neutral FP32 operations in Direct Pure (the beta-one
log-marginal expression and the unit-specialization gradient roundtrip). They are
algebraic identities, not exposed temperature or head-coupling controls. Removing
them changed Adam moments by roundoff, so they are retained for numerical parity.

See [AUDIT.md](AUDIT.md) for the actual verification scope. This release has not
been rerun on all production datasets and does not assert stable Wishart superiority.
