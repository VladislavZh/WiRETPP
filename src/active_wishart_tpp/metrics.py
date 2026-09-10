"""Small dependency-free clustering metrics."""

from __future__ import annotations
import numpy as np


def _contingency(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be aligned vectors")
    _, row = np.unique(labels, return_inverse=True)
    _, column = np.unique(predictions, return_inverse=True)
    table = np.zeros((row.max() + 1, column.max() + 1), dtype=np.int64)
    np.add.at(table, (row, column), 1)
    return table


def purity(labels: np.ndarray, predictions: np.ndarray) -> float:
    table = _contingency(labels, predictions)
    return float(table.max(axis=0).sum() / table.sum())


def adjusted_rand_index(labels: np.ndarray, predictions: np.ndarray) -> float:
    table = _contingency(labels, predictions).astype(np.float64)
    choose_two = lambda values: np.sum(values * (values - 1.0) / 2.0)
    agreement = choose_two(table)
    row_pairs = choose_two(table.sum(axis=1))
    column_pairs = choose_two(table.sum(axis=0))
    total_pairs = table.sum() * (table.sum() - 1.0) / 2.0
    if total_pairs == 0.0:
        return 1.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    return float((agreement - expected) / max(maximum - expected, 1e-12))


def clustering_summary(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if np.asarray(labels).size and np.all(np.asarray(labels) < 0):
        return {"purity": -1.0, "ari": -1.0}
    predictions = probabilities.argmax(axis=1)
    return {
        "purity": purity(labels, predictions),
        "ari": adjusted_rand_index(labels, predictions),
    }
