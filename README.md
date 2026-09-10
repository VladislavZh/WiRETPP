# Active Wishart TPP

Direct Pure mixture and Active Wishart random effects over a shared neural
intensity bank. This repository contains the fixed **Bartlett/all-gradient**
protocol: shared pretrain10, fast-v population Adam, L8/M64/E64, without temperature,
balancing, scheduler, freezing or head untying.

The mathematical and training contract is in [PROTOCOL.md](docs/paper/PROTOCOL.md).
The scope and limitations of cleanup verification are in [AUDIT.md](docs/paper/AUDIT.md).

## Install and run

Use Python 3.11 or 3.12 and a PyTorch installation suitable for your GPU.

```bash
pip install -e .
wishart-tpp --config configs/paper/dan.yaml --dataset sin_K4_C5 --seed 0
```

The default is **Pure then Wishart**, sharing exactly the same pretrain and expanded
bank. To run one method, add `--method pure` or `--method wishart`. Run seeds0/1/2
sequentially. Do not start another CUDA worker alongside a running experiment.

Real-data examples:

```bash
wishart-tpp --config configs/paper/retweet.yaml --seed 0
wishart-tpp --config configs/paper/amazon.yaml --seed 0
wishart-tpp --config configs/paper/stackoverflow.yaml --seed 0
```

StackOverflow requires the separately prepared leakage-free grouped split, not the
historical overlapping data. Data, checkpoints, generated results and baselines
are not included in Git. No experiment starts automatically on installation.

## Structure

```text
src/active_wishart_tpp/
  config.py             fixed protocol, dataset and compute settings
  cli.py                paired or single-method entrypoint
  data*.py              common train-only normalization and format adapters
  backbones/, cotic/    COTIC and corrected THP intensity banks
  model/                marked likelihood and Wishart matrix algebra
  inference/            local posterior, relative Bartlett transport, caches
  training/             E/M updates, shared initialization, selection, exact resume
configs/paper/          four reviewed example configurations
tests/paper/            standalone CPU contracts
docs/paper/             protocol and audit notes
```

The new checkpoint format stores active model/population/Adam/RNG state separately
from the validation-selected state. It does not silently import old experimental
checkpoints. Test is evaluated only after training and never selects a model.
For unlabeled real data, purity/ARI are unavailable, not evidence of collapse.

## Tests

PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:CUDA_VISIBLE_DEVICES = "-1"
python -m unittest discover -s tests/paper -v
```

POSIX shell:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python -m unittest discover -s tests/paper -v
```

CPU step parity and exact resume are checked; the refactored entrypoint still
requires a separate CUDA preflight before a new production launch. Historical
temperature-based DAN results are not results of this no-temperature release.

## Provenance

COTIC follows [VladislavZh/COTIC](https://github.com/VladislavZh/COTIC), studied
commit `362b8ab1f3cbb9e9dced2518e9daacf68235e77a`. The corrected THP adapter implements
the shared-bank interface locally; NTPP-MIX, ST-TPP, and other baseline trainers
are deliberately excluded from this distribution. Lightning Fabric provides
device setup and backward/gradient clipping, not a hidden training schedule.

In the original working directory, the ignored `src/wishart_tpp`, old scripts,
configs and reports remain local so that the already running baseline queue can
finish with its frozen source hashes. They are not packaged or published. The
cleanup does not rewrite past Git commits or delete local experiment evidence.
