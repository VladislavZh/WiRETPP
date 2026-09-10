# Cleanup verification

The original research tree remains local and ignored to keep the running baseline
queue's source hashes valid. The distributed package is `active_wishart_tpp` only;
it imports neither that tree nor the ignored experiment scripts or baselines.

CPU checks compare the cleaned implementation with the frozen native E64/fast-v
implementation, using a small COTIC fixture. They check seeded bank expansion,
ten direct Pure optimizer steps, and two Wishart cycles including neural weights,
Omega/alpha, physical Omega, and complete Adam state. Comparisons are bit-exact.
The initial port exposed neutral Pure gradient-roundtrip roundoff; restoring the
native unit-factor arithmetic resolved that mismatch without adding a control.

Standalone tests cover paired shared initialization, Pure and Wishart interrupted
versus uninterrupted training (including the integration seed/counter), optimizer
groups, deterministic EM blocks, rejection of experimental switches, and rejection
of incompatible settings/source at resume. Completed-result reuse is checked with
scoring disabled. Mathematical likelihood and fixed-budget local-posterior checks
complement these tests.

All 9 standalone CPU tests passed both from source and from an extracted wheel in
an isolated subprocess that rejects private research/baseline imports. The wheel
contains only `active_wishart_tpp` and its distribution metadata. Static checks
found no missing/private imports or Python files exceeding 500 lines (maximum385).
The local removal inventory verifies all216 former tracked files remain unchanged
on disk, and all439 frozen source pins of the running baseline queue still match.

The inherited local nonfinite-gradient repair remains unchanged for numerical
parity; it is not evidence that every numerical failure is harmless. The fixed
protocol has no adaptive E stopping, alternate quadrature, or spectral M branch.

No new CUDA worker or full production cycle was launched for cleanup. GPU parity
and performance of this refactored entrypoint remain to be checked after the active
baseline queue ends. Existing experiment results/checkpoints are not modified or
claimed as freshly generated paper results. Git history is not rewritten.

For review, start with `training/updates.py`, `inference/relative_posterior.py`,
`inference/regrouped_local.py`, `model/active_block.py`, and `model/wishart.py`.
Then inspect `training/runner.py` and `training/checkpoint.py` for selection/resume.
