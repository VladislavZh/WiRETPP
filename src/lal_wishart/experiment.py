"""Dataset, reporting, and integrity helpers for the final experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.metrics import (
    adjusted_rand_index,
    cluster_purity,
    normalized_mutual_information,
)
from lal_wishart.reproduction.paper_k3c5 import generate_paper_k3c5
from lal_wishart.reproduction.signed_k3c5 import generate_signed_k3c5


@dataclass(frozen=True)
class ExperimentSplit:
    """One deterministic split of simulated marked event sequences."""

    sequences: tuple[MarkedSequence, ...]
    labels: np.ndarray


def make_splits(
    *,
    parameter_seed: int,
    simulation_seed: int,
    split_seed: int,
    horizon: float,
    generated_per_class: int,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
) -> tuple[ExperimentSplit, ExperimentSplit, ExperimentSplit, dict]:
    """Regenerate and deterministically split the K=3, C=5 Hawkes DGP."""

    required_per_class = (
        train_per_class + validation_per_class + test_per_class
    )
    if generated_per_class < required_per_class:
        raise ValueError(
            "generated_per_class must cover train/validation/test"
        )
    paper = generate_paper_k3c5(
        parameter_seed=parameter_seed,
        simulation_seed=simulation_seed,
        n_per_cluster=generated_per_class,
        n_clusters=3,
        n_marks=5,
        horizon=horizon,
        max_jumps=1000,
        shuffle=False,
    )
    rng = np.random.default_rng(split_seed)
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for cluster in range(3):
        indices = np.flatnonzero(paper.labels == cluster)
        indices = indices[rng.permutation(len(indices))]
        first = train_per_class
        second = first + validation_per_class
        third = second + test_per_class
        split_indices["train"].extend(indices[:first].tolist())
        split_indices["validation"].extend(indices[first:second].tolist())
        split_indices["test"].extend(indices[second:third].tolist())

    def build(name: str) -> ExperimentSplit:
        indices = np.asarray(split_indices[name], dtype=np.int64)
        indices = indices[rng.permutation(len(indices))]
        return ExperimentSplit(
            sequences=tuple(paper.sequences[index] for index in indices),
            labels=paper.labels[indices],
        )

    counts = np.asarray([sequence.count for sequence in paper.sequences])
    audit = {
        "parameter_seed": parameter_seed,
        "simulation_seed": simulation_seed,
        "split_seed": split_seed,
        "horizon": horizon,
        "n_sequences": int(len(counts)),
        "generated_per_class": int(generated_per_class),
        "used_per_class": int(required_per_class),
        "unused_per_class": int(generated_per_class - required_per_class),
        "mean_event_count": float(counts.mean()),
        "sd_event_count": float(counts.std(ddof=1)),
        "minimum_event_count": int(counts.min()),
        "maximum_event_count": int(counts.max()),
        "mean_event_count_by_class": [
            float(counts[paper.labels == cluster].mean())
            for cluster in range(3)
        ],
        "spectral_radius_by_class": [
            parameters.spectral_radius for parameters in paper.parameters
        ],
    }
    return build("train"), build("validation"), build("test"), audit


def make_signed_splits(
    *,
    heterogeneity: str,
    parameter_seed: int,
    simulation_seed: int,
    split_seed: int,
    horizon: float,
    generated_per_class: int,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    true_alpha: float = 1.0,
) -> tuple[ExperimentSplit, ExperimentSplit, ExperimentSplit, dict]:
    """Generate and split the controlled signed random-effect DGP."""

    required_per_class = (
        train_per_class + validation_per_class + test_per_class
    )
    if generated_per_class < required_per_class:
        raise ValueError(
            "generated_per_class must cover train/validation/test"
        )
    dataset = generate_signed_k3c5(
        heterogeneity=heterogeneity,
        parameter_seed=parameter_seed,
        simulation_seed=simulation_seed,
        n_per_cluster=generated_per_class,
        n_clusters=3,
        n_marks=5,
        horizon=horizon,
        true_alpha=true_alpha,
        max_jumps=1000,
    )
    rng = np.random.default_rng(split_seed)
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for cluster in range(3):
        indices = np.flatnonzero(dataset.labels == cluster)
        indices = indices[rng.permutation(len(indices))]
        first = train_per_class
        second = first + validation_per_class
        third = second + test_per_class
        split_indices["train"].extend(indices[:first].tolist())
        split_indices["validation"].extend(indices[first:second].tolist())
        split_indices["test"].extend(indices[second:third].tolist())

    def build(name: str) -> ExperimentSplit:
        indices = np.asarray(split_indices[name], dtype=np.int64)
        indices = indices[rng.permutation(len(indices))]
        return ExperimentSplit(
            sequences=tuple(dataset.sequences[index] for index in indices),
            labels=dataset.labels[indices],
        )

    counts = np.asarray([sequence.count for sequence in dataset.sequences])
    identity = np.eye(dataset.mean_matrix.shape[0])
    distances = np.linalg.norm(
        dataset.random_effects - dataset.mean_matrix,
        axis=(1, 2),
    )
    audit = {
        "dgp": "signed_wishart_hawkes",
        "heterogeneity": heterogeneity,
        "true_alpha": true_alpha,
        "true_degrees_of_freedom": dataset.degrees_of_freedom,
        "parameter_seed": parameter_seed,
        "simulation_seed": simulation_seed,
        "split_seed": split_seed,
        "horizon": horizon,
        "n_sequences": int(len(counts)),
        "generated_per_class": int(generated_per_class),
        "used_per_class": int(required_per_class),
        "unused_per_class": int(generated_per_class - required_per_class),
        "mean_event_count": float(counts.mean()),
        "sd_event_count": float(counts.std(ddof=1)),
        "minimum_event_count": int(counts.min()),
        "maximum_event_count": int(counts.max()),
        "mean_event_count_by_class": [
            float(counts[dataset.labels == cluster].mean())
            for cluster in range(3)
        ],
        "spectral_radius_by_class": [
            parameters.spectral_radius for parameters in dataset.parameters
        ],
        "mean_random_effect_frobenius_distance": float(distances.mean()),
        "sd_random_effect_frobenius_distance": float(
            distances.std(ddof=1)
        ),
        "mean_matrix_distance_from_identity": float(
            np.linalg.norm(dataset.mean_matrix - identity)
        ),
        "generation_proposals": dataset.proposals,
    }
    return build("train"), build("validation"), build("test"), audit


def markdown_table(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    digits: int = 5,
) -> str:
    """Render selected DataFrame columns as a compact Markdown table."""

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def clustering_row(
    labels: np.ndarray,
    probabilities: torch.Tensor,
    *,
    model: str,
    degrees_of_freedom: int | None,
    representation: str,
) -> tuple[dict[str, object], np.ndarray]:
    """Compute label-permutation-invariant clustering diagnostics."""

    values = probabilities.detach().cpu().numpy()
    prediction = values.argmax(axis=1)
    sizes = np.bincount(prediction, minlength=values.shape[1])
    entropy = -np.sum(values * np.log(values + 1e-12), axis=1)
    return ({
        "model": model,
        "degrees_of_freedom": degrees_of_freedom,
        "representation": representation,
        "purity": cluster_purity(labels, prediction),
        "ari": adjusted_rand_index(labels, prediction),
        "nmi": normalized_mutual_information(labels, prediction),
        "mean_entropy": float(entropy.mean()),
        "active_k": int(np.count_nonzero(sizes)),
        "cluster_sizes": sizes.tolist(),
        "mean_probabilities": values.mean(axis=0).tolist(),
    }, prediction)


def save_probabilities(
    path: Path,
    *,
    labels: np.ndarray,
    prefix: torch.Tensor,
    full: torch.Tensor,
) -> None:
    """Save posterior cluster probabilities and hard assignments."""

    prefix_values = prefix.detach().cpu().numpy()
    full_values = full.detach().cpu().numpy()
    columns: dict[str, object] = {
        "label": labels,
        "prefix_cluster": prefix_values.argmax(axis=1),
        "full_cluster": full_values.argmax(axis=1),
    }
    for component in range(prefix_values.shape[1]):
        columns[f"prefix_probability_{component}"] = prefix_values[:, component]
        columns[f"full_probability_{component}"] = full_values[:, component]
    pd.DataFrame(columns).to_csv(path, index=False)


def save_mean_matrix(path: Path, matrix: torch.Tensor) -> None:
    """Save a square tensor in long-form CSV."""

    values = matrix.detach().cpu().numpy()
    rows = [
        {"row": row, "column": column, "value": values[row, column]}
        for row in range(values.shape[0])
        for column in range(values.shape[1])
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(outdir: Path) -> None:
    """Write SHA-256 hashes for every artifact file except the manifest."""

    paths = sorted(
        path
        for path in outdir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    (outdir / "MANIFEST.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(outdir).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
