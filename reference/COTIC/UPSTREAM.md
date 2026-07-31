# Vendored COTIC subset

The files under `src/models/components/cotic/` are the minimal runtime subset
used by this experiment. They come from:

- repository: https://github.com/VladislavZh/COTIC
- commit: `362b8ab1f3cbb9e9dced2518e9daacf68235e77a`
- retained classes: `COTIC`, `ContinuousConv1D`, `ContinuousConv1DSim`,
  `IntensityHeadLinear`, and `Predictions`

Training pipelines, datasets, Hydra configuration, checkpoints, pictures, and
the nested Git history were removed from this reproducibility snapshot because
the final experiment does not import them.
