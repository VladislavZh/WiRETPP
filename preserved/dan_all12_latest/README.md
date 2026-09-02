# PRESERVE: final DAN all-12 experiment

Do not delete this directory during repository cleanup.

This is the latest DAN all-12 result identified on 2026-08-29 from the
workbook protocol and file timestamps:

- experiment: `DAN all-12 Dirichlet mixture-weight prior`;
- report timestamp: 2026-08-22 17:37:50;
- seeds: `0, 1, 2`;
- split seed: `42`;
- baseline Wishart `a_pi = 1`;
- Dirichlet Wishart `a_pi = 81`;
- coverage: `33 new + 3 reused = 36`.

Preserved files:

- `dirichlet_weight_all12_summary.xlsx` — final workbook;
- `dirichlet_weight_all12_summary.xlsx.inspect.ndjson` — extracted workbook
  content used to verify the protocol and results.

The workbook names these source roots:

- `runs/dan12_corrected_3seeds`;
- `runs/dirichlet_weight_all12`;
- `runs/dirichlet_weight_sweep/alpha_81` (reused `sin_K4`).

Those source roots were removed before the cleanup stop request. Restore the
first two (and the third if needed) from OneDrive Recycle Bin to recover the
full checkpoints. The final workbook and its inspection export above were
copied from `outputs/corrected-dan-12-three-seed-monitor` and verified against
the source files with SHA-256 after copying.
