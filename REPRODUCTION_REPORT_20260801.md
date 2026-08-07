# WiRE-TPP reproduction report

Date: 2026-08-01

Repository revision: `6309735` (`Publish reproducible shared-Wishart TPP experiments`)

## Scope

The experiment was reconstructed from the current repository and checked against
Chapter 4 of `synopsis.pdf` (pages 25-31). Chapter 4 introduces a hierarchical TPP
mixture with one trajectory-level Wishart random matrix shared by component
routing and intensity transformation.

The repository is a newer corrected snapshot, not an exact implementation of the
legacy Table 3 protocol in the synopsis. In particular:

- Table 3 used the legacy nonnegative transform from Equation 25, one seed, and
  `nu = P = 15`.
- The repository uses one shared encoder, `nu = 20`, squared-correlation
  attention, validation checkpoint selection, two component parameterizations,
  and a 256-sample three-repeat Monte Carlo audit.
- The signed-correlation model from Equation 13 of the synopsis is not the model
  evaluated by the repository snapshot.

## Environment

- Windows, NVIDIA GeForce RTX 4060 Laptop GPU
- Python 3.12.13 (the project permits 3.11-3.12; the published artifact records
  Python 3.11.14)
- PyTorch 2.12.0+cu126, CUDA runtime 12.6
- NumPy 2.3.5, pandas 2.2.3, EasyTPP 0.2.1

The original training device, CUDA version, and BLAS versions were not recorded
in the published artifact, so bit-for-bit retraining on another GPU is not an
available reproducibility target.

## Validation of the published snapshot

- 28/28 unit and smoke tests passed.
- All 12 published checkpoints loaded with `strict=True`.
- Parameter counts and the one-encoder architecture audit passed.
- The published `MANIFEST.sha256` was stale for `environment.json`,
  `FINAL_COMPARISON.csv`, and `REPORT.md`. The three manifest entries were updated
  to the hashes of the tracked files; the strict verifier now passes.
- Re-evaluating the published checkpoints with 256 fresh Wishart samples and
  three repeats reproduced the published suffix NLLs within 0.00061 and full
  purities within 0.00188. Every repeat used all three clusters.

This separates checkpoint portability from retraining variability: evaluation of
fixed weights is stable in the reconstructed environment.

## Full retraining

The complete 300-epoch protocol ran on CUDA in 1104 seconds. The data audit
reproduced 1200 trajectories, mean event count 49.7683, event-count range 4-151,
and the 90/45/1065 train/validation/test split. The architecture audit is
byte-identical to the published one. The only data-audit differences are three
last-bit floating-point representations of the class spectral radii.

The table compares the published 256-sample audit with the newly trained
checkpoints. Lower NLL is better. `rep W-noW` is the paired effect in the new run.

| model | pub no-W NLL | rep no-W NLL | pub W NLL | rep W NLL | rep W-noW | pub W purity | rep W purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHP/output_split | 3.5510 | 3.5515 | 3.4601 | 3.4580 | -0.0935 | 0.924 | 0.936 |
| NHP/lal_fixed | 3.5776 | 3.5776 | 3.4714 | 3.4719 | -0.1057 | 0.924 | 0.923 |
| THP/output_split | 3.6290 | 3.6236 | 3.4640 | 3.5128 | -0.1108 | 0.951 | 0.661 |
| THP/lal_fixed | 3.5359 | 3.5368 | 3.4824 | 3.4969 | -0.0399 | 0.935 | 0.925 |
| COTIC/output_split | 3.8998 | 3.9159 | 3.6365 | 3.6408 | -0.2751 | 0.943 | 0.934 |
| COTIC/lal_fixed | 3.8729 | 3.8771 | 3.6595 | 3.6732 | -0.2039 | 0.892 | 0.709 |

The main predictive claim was reproduced: Wishart improved suffix NLL in all six
paired comparisons. The key COTIC/output-split collapse control was also
reproduced: no-W assigned all test trajectories to one cluster (purity 0.333,
ARI 0), while Wishart used three clusters and reached full purity 0.934 and ARI
0.817.

Clustering was not fully stable under retraining. THP/output-split Wishart purity
was 0.661 instead of 0.951, and COTIC/lal_fixed Wishart purity was 0.709 instead
of 0.892. Their 256-sample repeat SDs are small, so this is training-path
variability rather than Monte Carlo evaluation noise. The runner seeds PyTorch
but does not require deterministic algorithms, and the original GPU/software
stack was not recorded.

## Generated artifacts

- `artifacts/reproduction_smoke_20260801`: two-epoch end-to-end smoke run.
- `artifacts/reproduction_full_300ep_seed0_20260801`: new 300-epoch training,
  checkpoints, predictions, histories, learned means, and 256x3 MC audit.
- `artifacts/reproduction_published_checkpoint_audit_20260801`: a copy of the
  published checkpoints re-evaluated in the reconstructed CUDA environment.

Both full artifact directories pass `scripts/verify_artifacts.py` after their
audits.

## Commands

```powershell
$env:PYTHONPATH='src;scripts'
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts\verify_artifacts.py

.venv\Scripts\python scripts\run_corrected_shared_wishart_architectures.py `
  --quick --device cuda --outdir artifacts/reproduction_smoke_20260801

.venv\Scripts\python scripts\run_corrected_shared_wishart_architectures.py `
  --device cuda --epochs 300 `
  --outdir artifacts/reproduction_full_300ep_seed0_20260801

.venv\Scripts\python scripts\audit_corrected_shared_wishart_mc.py `
  --device cuda --samples 256 --repeats 3 `
  --artifact artifacts/reproduction_full_300ep_seed0_20260801

.venv\Scripts\python scripts\verify_artifacts.py `
  --artifact artifacts/reproduction_full_300ep_seed0_20260801
```

## Conclusion

The repository is executable and its published checkpoints and evaluations are
recoverable. The corrected protocol's strongest result - lower predictive suffix
NLL with the shared latent-Wishart effect in all six paired comparisons - was
reproduced by full retraining. Exact clustering metrics are not a robust
single-seed result across environments, so future experiments should add
independent training seeds, deterministic-mode diagnostics, and the signed-model
and no-interaction ablations described in Chapter 4.
