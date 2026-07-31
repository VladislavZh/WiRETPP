"""Clustering metrics used by the end-to-end comparison runners."""

from __future__ import annotations

import numpy as np


def adjusted_rand_index(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return the Hubert--Arabie adjusted Rand index without sklearn."""

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shape")
    if labels.size == 0:
        raise ValueError("ARI is undefined for an empty sample")
    _, label_ids = np.unique(labels, return_inverse=True)
    _, prediction_ids = np.unique(predictions, return_inverse=True)
    contingency = np.zeros(
        (int(label_ids.max()) + 1, int(prediction_ids.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(contingency, (label_ids, prediction_ids), 1)

    def choose_two(values: np.ndarray) -> float:
        values = values.astype(np.float64, copy=False)
        return float(np.sum(values * (values - 1.0) / 2.0))

    agreement = choose_two(contingency)
    label_pairs = choose_two(contingency.sum(axis=1))
    prediction_pairs = choose_two(contingency.sum(axis=0))
    total_pairs = labels.size * (labels.size - 1.0) / 2.0
    if total_pairs == 0:
        return 1.0
    expected = label_pairs * prediction_pairs / total_pairs
    maximum = 0.5 * (label_pairs + prediction_pairs)
    denominator = maximum - expected
    if abs(denominator) < np.finfo(float).eps:
        return 1.0
    return (agreement - expected) / denominator


def normalized_mutual_information(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return NMI with the arithmetic-mean entropy normalization."""

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shape")
    if labels.size == 0:
        raise ValueError("NMI is undefined for an empty sample")
    _, label_ids = np.unique(labels, return_inverse=True)
    _, prediction_ids = np.unique(predictions, return_inverse=True)
    contingency = np.zeros(
        (int(label_ids.max()) + 1, int(prediction_ids.max()) + 1),
        dtype=np.float64,
    )
    np.add.at(contingency, (label_ids, prediction_ids), 1.0)
    joint = contingency / labels.size
    label_probability = joint.sum(axis=1)
    prediction_probability = joint.sum(axis=0)
    positive = joint > 0
    independent = label_probability[:, None] * prediction_probability[None, :]
    mutual_information = float(
        np.sum(joint[positive] * np.log(joint[positive] / independent[positive]))
    )
    label_entropy = float(
        -np.sum(
            label_probability[label_probability > 0]
            * np.log(label_probability[label_probability > 0])
        )
    )
    prediction_entropy = float(
        -np.sum(
            prediction_probability[prediction_probability > 0]
            * np.log(prediction_probability[prediction_probability > 0])
        )
    )
    denominator = 0.5 * (label_entropy + prediction_entropy)
    if denominator <= np.finfo(float).eps:
        return 1.0
    return mutual_information / denominator


def cluster_purity(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return standard hard-assignment cluster purity.

    Each predicted cluster is matched to its most frequent true label. Unlike
    ARI, purity is not chance-corrected and therefore must be interpreted
    together with the number of active clusters and a majority-class baseline.
    """

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shape")
    if labels.size == 0:
        raise ValueError("purity is undefined for an empty sample")

    majority_sum = 0
    for cluster in np.unique(predictions):
        cluster_labels = labels[predictions == cluster]
        _, counts = np.unique(cluster_labels, return_counts=True)
        majority_sum += int(counts.max())
    return majority_sum / labels.size
