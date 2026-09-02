"""Create standalone figures for the completed seed-0 real-data retrain.

The figures stay with the run artifacts and are not copied into the dissertation.
Validation bands use the persisted trajectory-bootstrap intervals. Selected
validation/test intervals and paired gains resample complete held-out paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
from wishart_tpp.training.validation_curves import trajectory_bootstrap_nll

RUN = ROOT / "runs/cotic_real_full_retrain_pretrain10_ls5s2_batchaligned_50cycle_seed0"
OUT = RUN / "figures"
STYLE = ROOT / "tmp/thesis_edit/scientific.mplstyle"

DATASETS = ("retweet", "amazon", "so")
DATASET_LABELS = {
    "retweet": "Retweet",
    "amazon": "Amazon",
    "so": "StackOverflow",
}
METHODS = ("cotic_k1", "pure_k5", "wishart_k5")
METHOD_LABELS = {
    "cotic_k1": r"COTIC ($K=1$)",
    "pure_k5": r"Pure ($K=5$)",
    "wishart_k5": r"Wishart ($K=5$)",
}
COLORS = {
    "cotic_k1": "#3264a8",
    "pure_k5": "#df7f20",
    "wishart_k5": "#238b57",
}
MARKERS = {"cotic_k1": "o", "pure_k5": "s", "wishart_k5": "D"}
BOOTSTRAP_DRAWS = 10_000


def configure_style() -> None:
    plt.style.use("default")
    if STYLE.exists():
        plt.style.use(STYLE)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titleweight": "regular",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def load_curve(dataset: str, method: str) -> pd.DataFrame:
    path = RUN / dataset / "seed_0" / method / "validation_curve.csv"
    frame = pd.read_csv(path)
    if method == "wishart_k5":
        frame = frame.loc[frame.method == "wishart_k5"].copy()
    frame = frame.sort_values("cycle", kind="stable").reset_index(drop=True)
    expected = list(range(51))
    if frame.cycle.astype(int).tolist() != expected:
        raise RuntimeError(f"incomplete curve: {dataset}/{method}")
    if frame.neural_updates_after_shared.astype(int).tolist() != [
        8 * c for c in expected
    ]:
        raise RuntimeError(f"misaligned updates: {dataset}/{method}")
    return frame


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            markerfacecolor=COLORS[method],
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def padded_limits(
    low: np.ndarray, high: np.ndarray, fraction: float = 0.10
) -> tuple[float, float]:
    minimum = float(np.nanmin(low))
    maximum = float(np.nanmax(high))
    span = max(maximum - minimum, 1e-3)
    return minimum - fraction * span, maximum + fraction * span


def plot_validation_curves(*, first_cycle: int, stem: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=False)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for method in METHODS:
            local = load_curve(dataset, method)
            local = local.loc[local.cycle >= first_cycle]
            x = local.cycle.to_numpy(dtype=float)
            y = local.nll_per_exposure.to_numpy(dtype=float)
            low = local.ci95_low.to_numpy(dtype=float)
            high = local.ci95_high.to_numpy(dtype=float)
            lows.append(low)
            highs.append(high)
            axis.fill_between(
                x, low, high, color=COLORS[method], alpha=0.12, linewidth=0
            )
            axis.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=2.8 if first_cycle == 0 else 3.8,
                markevery=5 if first_cycle == 0 else 2,
                linewidth=1.25,
            )
            best = local.loc[local.nll_per_exposure.idxmin()]
            axis.scatter(
                [best.cycle],
                [best.nll_per_exposure],
                s=38,
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
        axis.set_title(f"({chr(97 + panel)}) {DATASET_LABELS[dataset]}")
        axis.set_xlim(first_cycle - 0.7, 50.7)
        axis.set_xlabel("Post-pretrain cycle (8 updates each)")
        axis.set_ylim(*padded_limits(np.concatenate(lows), np.concatenate(highs)))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.22, linewidth=0.55)
    axes[0].set_ylabel("Validation NLL / exposure")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.19, top=0.76, wspace=0.24)
    fig.suptitle(title, y=0.985)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
    )
    save(fig, stem)


def load_paths(dataset: str, method: str, split: str) -> pd.DataFrame:
    name = (
        "selected_validation_paths.csv" if split == "validation" else "test_paths.csv"
    )
    path = RUN / dataset / "seed_0" / method / name
    frame = (
        pd.read_csv(path).sort_values("source_id", kind="stable").reset_index(drop=True)
    )
    if frame.empty or not {"source_id", "horizon", "log_likelihood"}.issubset(
        frame.columns
    ):
        raise RuntimeError(f"invalid path artifact: {path}")
    return frame


def paired_gain(
    baseline: pd.DataFrame,
    wishart: pd.DataFrame,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float, float]:
    if not np.array_equal(baseline.source_id.to_numpy(), wishart.source_id.to_numpy()):
        raise RuntimeError("paired artifacts have different source IDs")
    horizon = baseline.horizon.to_numpy(dtype=np.float64)
    if not np.allclose(
        horizon, wishart.horizon.to_numpy(dtype=np.float64), rtol=0, atol=1e-10
    ):
        raise RuntimeError("paired artifacts have different horizons")
    baseline_ll = baseline.log_likelihood.to_numpy(dtype=np.float64)
    wishart_ll = wishart.log_likelihood.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    chunk = 250
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        idx = rng.integers(0, len(horizon), size=(stop - start, len(horizon)))
        exposure = horizon[idx].sum(axis=1)
        values[start:stop] = (
            -baseline_ll[idx].sum(axis=1) / exposure
            + wishart_ll[idx].sum(axis=1) / exposure
        )
    point = -baseline_ll.sum() / horizon.sum() + wishart_ll.sum() / horizon.sum()
    return (
        float(point),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def selected_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute: list[dict[str, object]] = []
    gains: list[dict[str, object]] = []
    for d_idx, dataset in enumerate(DATASETS):
        for s_idx, split in enumerate(("validation", "test")):
            frames = {method: load_paths(dataset, method, split) for method in METHODS}
            for m_idx, method in enumerate(METHODS):
                frame = frames[method]
                summary = trajectory_bootstrap_nll(
                    frame.log_likelihood.to_numpy(dtype=np.float64),
                    frame.horizon.to_numpy(dtype=np.float64),
                    seed=51_000_041 + d_idx * 100_003 + s_idx * 10_007 + m_idx * 1_009,
                    draws=BOOTSTRAP_DRAWS,
                )
                absolute.append(
                    {"dataset": dataset, "split": split, "method": method, **summary}
                )
            for b_idx, baseline in enumerate(("cotic_k1", "pure_k5")):
                point, low, high = paired_gain(
                    frames[baseline],
                    frames["wishart_k5"],
                    seed=61_000_051 + d_idx * 100_003 + s_idx * 10_007 + b_idx * 1_009,
                )
                gains.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "baseline": baseline,
                        "gain": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(absolute), pd.DataFrame(gains)


def plot_selected_nll(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.25), constrained_layout=True)
    for row, split in enumerate(("validation", "test")):
        for col, dataset in enumerate(DATASETS):
            axis = axes[row, col]
            local = (
                frame.loc[(frame.dataset == dataset) & (frame.split == split)]
                .set_index("method")
                .loc[list(METHODS)]
            )
            x = np.arange(3, dtype=float)
            y = local.nll_per_exposure.to_numpy(dtype=float)
            low = local.ci95_low.to_numpy(dtype=float)
            high = local.ci95_high.to_numpy(dtype=float)
            for index, method in enumerate(METHODS):
                axis.errorbar(
                    x[index],
                    y[index],
                    yerr=[[y[index] - low[index]], [high[index] - y[index]]],
                    fmt=MARKERS[method],
                    color=COLORS[method],
                    markerfacecolor=COLORS[method],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    markersize=6.2,
                    capsize=3,
                    elinewidth=1.25,
                    zorder=3,
                )
            axis.set_xticks(x, ["COTIC", "Pure", "Wishart"])
            axis.set_ylim(*padded_limits(low, high, fraction=0.22))
            axis.set_title(DATASET_LABELS[dataset] if row == 0 else "")
            axis.grid(axis="y", alpha=0.22, linewidth=0.55)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            if col == 0:
                axis.set_ylabel(f"Selected {split} NLL / exposure")
    fig.suptitle("Selected-checkpoint NLL with trajectory-bootstrap 95% intervals")
    save(fig, "selected_nll_ci")


def plot_gains(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.15), constrained_layout=True)
    baselines = ("cotic_k1", "pure_k5")
    for row, split in enumerate(("validation", "test")):
        for col, dataset in enumerate(DATASETS):
            axis = axes[row, col]
            local = (
                frame.loc[(frame.dataset == dataset) & (frame.split == split)]
                .set_index("baseline")
                .loc[list(baselines)]
            )
            x = np.arange(2, dtype=float)
            y = local.gain.to_numpy(dtype=float)
            low = local.ci95_low.to_numpy(dtype=float)
            high = local.ci95_high.to_numpy(dtype=float)
            for index, baseline in enumerate(baselines):
                axis.errorbar(
                    x[index],
                    y[index],
                    yerr=[[y[index] - low[index]], [high[index] - y[index]]],
                    fmt=MARKERS[baseline],
                    color=COLORS[baseline],
                    markerfacecolor=COLORS[baseline],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    markersize=6.2,
                    capsize=3,
                    elinewidth=1.25,
                    zorder=3,
                )
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.set_xticks(x, ["vs COTIC", "vs Pure"])
            axis.set_ylim(
                *padded_limits(np.minimum(low, 0), np.maximum(high, 0), fraction=0.20)
            )
            axis.set_title(DATASET_LABELS[dataset] if row == 0 else "")
            axis.grid(axis="y", alpha=0.22, linewidth=0.55)
            if col == 0:
                axis.set_ylabel(f"{split.title()} gain\n(baseline − Wishart)")
    fig.suptitle("Paired NLL gains; positive values favor Wishart")
    save(fig, "wishart_paired_gains_ci")


def plot_wishart_diagnostics() -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(10.2, 5.2), constrained_layout=True, sharex=True
    )
    for col, dataset in enumerate(DATASETS):
        local = load_curve(dataset, "wishart_k5")
        cycle = local.cycle.to_numpy(dtype=float)
        axes[0, col].plot(cycle, local.alpha, color="#7b4ab5", linewidth=1.45)
        axes[0, col].scatter(
            [cycle[-1]], [local.alpha.iloc[-1]], color="#7b4ab5", s=24, zorder=3
        )
        axes[0, col].set_title(DATASET_LABELS[dataset])
        axes[0, col].grid(True, alpha=0.22, linewidth=0.55)
        axes[1, col].plot(
            cycle, local.effective_k, color=COLORS["wishart_k5"], linewidth=1.45
        )
        axes[1, col].axhline(5, color="#777777", linewidth=0.7, linestyle="--")
        axes[1, col].scatter(
            [cycle[-1]],
            [local.effective_k.iloc[-1]],
            color=COLORS["wishart_k5"],
            s=24,
            zorder=3,
        )
        axes[1, col].set_xlabel("Cycle")
        axes[1, col].set_ylim(1, 5.15)
        axes[1, col].grid(True, alpha=0.22, linewidth=0.55)
    axes[0, 0].set_ylabel(r"Mixing strength $\alpha$")
    axes[1, 0].set_ylabel("Effective number of components")
    fig.suptitle("Wishart mixture diagnostics")
    save(fig, "wishart_alpha_effective_k")


def plot_wishart_weights() -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.15), constrained_layout=False, sharey=True
    )
    palette = ["#3264a8", "#df7f20", "#238b57", "#7b4ab5", "#b34a4a"]
    for col, dataset in enumerate(DATASETS):
        local = load_curve(dataset, "wishart_k5")
        weights = np.vstack(
            [
                local[f"mixture_weight_{index}"].to_numpy(dtype=float)
                for index in range(5)
            ]
        )
        axes[col].stackplot(
            local.cycle, weights, colors=palette, alpha=0.82, linewidth=0
        )
        axes[col].set_title(DATASET_LABELS[dataset])
        axes[col].set_xlabel("Cycle")
        axes[col].set_xlim(0, 50)
        axes[col].set_ylim(0, 1)
        axes[col].grid(axis="x", alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Mixture weight")
    handles = [
        Line2D(
            [0], [0], color=palette[index], linewidth=6, label=f"Component {index + 1}"
        )
        for index in range(5)
    ]
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.20, top=0.74, wspace=0.18)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
    )
    fig.suptitle("Wishart mixture-weight trajectories", y=0.985)
    save(fig, "wishart_mixture_weights")


def main() -> None:
    configure_style()
    plot_validation_curves(
        first_cycle=0,
        stem="validation_nll_curves_full",
        title="Matched validation trajectories (all 400 post-pretrain updates)",
    )
    plot_validation_curves(
        first_cycle=30,
        stem="validation_nll_curves_late",
        title="Matched validation trajectories, cycles 30–50",
    )
    absolute, gains = selected_summaries()
    absolute.to_csv(OUT / "selected_nll_ci.csv", index=False)
    gains.to_csv(OUT / "wishart_paired_gains_ci.csv", index=False)
    plot_selected_nll(absolute)
    plot_gains(gains)
    plot_wishart_diagnostics()
    plot_wishart_weights()
    manifest = {
        "run": str(RUN),
        "seed": 0,
        "main_text_integration": False,
        "curve_ci": "persisted trajectory-bootstrap 95% intervals",
        "selected_ci": "10000 trajectory-bootstrap resamples conditional on seed-0 checkpoint",
        "gain_ci": "10000 paired trajectory-bootstrap resamples",
        "figures": [
            "validation_nll_curves_full",
            "validation_nll_curves_late",
            "selected_nll_ci",
            "wishart_paired_gains_ci",
            "wishart_alpha_effective_k",
            "wishart_mixture_weights",
        ],
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
