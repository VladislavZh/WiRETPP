"""Article-ready validation curves with trajectory-bootstrap intervals.

The training losses are normalized by total observation exposure, so the
bootstrap must resample complete trajectories and recompute the ratio of sums.
Taking a percentile of per-trajectory ratios estimates a different quantity
and is intentionally avoided here.
"""

from __future__ import annotations
import os
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import pandas as pd
from active_wishart_tpp.data import DatasetPartition
from active_wishart_tpp.training.evaluation import Evaluation


def trajectory_bootstrap_nll(
    log_likelihoods: np.ndarray,
    horizons: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
    chunk_size: int = 250,
) -> dict[str, float | int | str]:
    """Return a percentile CI for exposure-normalized validation NLL."""
    scores = np.asarray(log_likelihoods, dtype=np.float64)
    exposure = np.asarray(horizons, dtype=np.float64)
    if scores.ndim != 1 or exposure.ndim != 1 or scores.shape != exposure.shape:
        raise ValueError("scores and horizons must be aligned one-dimensional arrays")
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("log likelihoods must be finite and non-empty")
    if not np.all(np.isfinite(exposure)) or np.any(exposure <= 0.0):
        raise ValueError("trajectory horizons must be finite and positive")
    if draws < 1 or chunk_size < 1:
        raise ValueError("draws and chunk_size must be positive")
    random = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    count = scores.size
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = random.integers(0, count, size=(stop - start, count))
        bootstrap[start:stop] = -scores[indices].sum(axis=1) / exposure[indices].sum(
            axis=1
        )
    return {
        "nll_per_exposure": float(-scores.sum() / exposure.sum()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "ci_method": "trajectory_percentile_bootstrap_ratio_of_sums",
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


class ValidationCurveWriter:
    """Persist aggregate and path-level values after every validation point."""

    def __init__(
        self,
        output: Path,
        *,
        dataset: str,
        seed: int,
        method: str,
        bootstrap_draws: int = 2000,
    ) -> None:
        self.output = Path(output)
        self.dataset = dataset
        self.seed = int(seed)
        self.method = method
        self.bootstrap_draws = int(bootstrap_draws)
        self.summary_path = self.output / "validation_curve.csv"
        self.paths_path = self.output / "validation_curve_paths.csv"

    def completed_cycles(self) -> set[int]:
        if not self.summary_path.is_file():
            return set()
        frame = pd.read_csv(self.summary_path)
        return set(frame.loc[frame["method"] == self.method, "cycle"].astype(int))

    def record(
        self,
        cycle: int,
        updates: int,
        evaluation: Evaluation,
        validation: DatasetPartition,
        *,
        extras: Mapping[str, float | int | str | bool] | None = None,
    ) -> dict[str, float | int | str]:
        scores = evaluation.marginal_scores.detach().cpu().numpy().astype(np.float64)
        horizons = np.asarray(
            [sequence.horizon for sequence in validation.sequences], dtype=np.float64
        )
        if scores.shape != horizons.shape:
            raise ValueError("evaluation and validation partition are misaligned")
        bootstrap_seed = 8000021 + self.seed * 1000003 + int(cycle) * 10007
        summary = trajectory_bootstrap_nll(
            scores, horizons, seed=bootstrap_seed, draws=self.bootstrap_draws
        )
        if not np.isclose(
            float(summary["nll_per_exposure"]),
            evaluation.nll_per_exposure,
            rtol=1e-06,
            atol=2e-06,
        ):
            raise RuntimeError("path-level curve NLL does not reproduce Evaluation")
        row: dict[str, float | int | str | bool] = {
            "dataset": self.dataset,
            "seed": self.seed,
            "method": self.method,
            "cycle": int(cycle),
            "neural_updates_after_shared": int(updates),
            "validation_paths": int(scores.size),
            "total_exposure": float(horizons.sum()),
            **summary,
        }
        if extras:
            row.update(dict(extras))
        if self.summary_path.is_file():
            summary_frame = pd.read_csv(self.summary_path)
            keep = ~(
                (summary_frame["method"] == self.method)
                & (summary_frame["cycle"].astype(int) == int(cycle))
            )
            summary_frame = summary_frame.loc[keep]
        else:
            summary_frame = pd.DataFrame()
        summary_frame = pd.concat(
            [summary_frame, pd.DataFrame([row])], ignore_index=True
        ).sort_values(["method", "cycle"], kind="stable")
        path_rows = pd.DataFrame(
            {
                "dataset": self.dataset,
                "seed": self.seed,
                "method": self.method,
                "cycle": int(cycle),
                "neural_updates_after_shared": int(updates),
                "source_id": validation.source_ids,
                "horizon": horizons,
                "log_likelihood": scores,
                "nll_per_path_exposure": -scores / horizons,
            }
        )
        if self.paths_path.is_file():
            paths_frame = pd.read_csv(self.paths_path)
            keep = ~(
                (paths_frame["method"] == self.method)
                & (paths_frame["cycle"].astype(int) == int(cycle))
            )
            paths_frame = paths_frame.loc[keep]
        else:
            paths_frame = pd.DataFrame()
        paths_frame = pd.concat(
            [paths_frame, path_rows], ignore_index=True
        ).sort_values(["method", "cycle", "source_id"], kind="stable")
        _atomic_csv(paths_frame, self.paths_path)
        _atomic_csv(summary_frame, self.summary_path)
        return summary
