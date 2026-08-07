"""Controlled signed WiRE-TPP K=3, C=5 synthetic data generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.reproduction.paper_k3c5 import (
    _sample_parameters,
    ExponentialHawkesParameters,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SignedK3C5Dataset:
    sequences: tuple[MarkedSequence, ...]
    labels: IntArray
    random_effects: FloatArray
    parameters: tuple[ExponentialHawkesParameters, ...]
    mean_matrix: FloatArray
    heterogeneity: str
    degrees_of_freedom: int | None
    true_alpha: float
    parameter_seed: int
    simulation_seed: int
    horizon: float
    proposals: int


def signed_mean_matrix(
    n_components: int,
    n_marks: int,
    *,
    suppressive: bool,
    correlation: float = 0.6,
) -> FloatArray:
    """Return an SPD, trace-normalized mean with optional signed pairs."""

    if n_components <= 0 or n_marks <= 0:
        raise ValueError("dimensions must be positive")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must lie in [0, 1)")
    dimension = n_components * n_marks
    mean = np.eye(dimension, dtype=float)
    if suppressive:
        for component in range(n_components):
            offset = component * n_marks
            if n_marks >= 2:
                mean[offset, offset + 1] = -correlation
                mean[offset + 1, offset] = -correlation
            if n_marks >= 4:
                mean[offset + 2, offset + 3] = correlation
                mean[offset + 3, offset + 2] = correlation
    eigenvalues = np.linalg.eigvalsh(mean)
    if float(eigenvalues.min()) <= 0.0:
        raise RuntimeError("constructed mean matrix is not positive definite")
    return dimension * mean / float(np.trace(mean))


def sample_wishart_effect(
    rng: np.random.RandomState,
    mean: FloatArray,
    degrees_of_freedom: int | None,
) -> FloatArray:
    """Sample ``Wishart(nu, mean/nu)`` or return deterministic ``mean``."""

    dimension = int(mean.shape[0])
    if mean.shape != (dimension, dimension):
        raise ValueError("mean matrix must be square")
    if degrees_of_freedom is None:
        return np.asarray(mean, dtype=float).copy()
    if degrees_of_freedom < dimension:
        raise ValueError("degrees of freedom must be at least the dimension")
    cholesky = np.linalg.cholesky(mean)
    standard = rng.normal(size=(degrees_of_freedom, dimension))
    transformed = standard @ cholesky.T / np.sqrt(degrees_of_freedom)
    return np.asarray(transformed.T @ transformed, dtype=float)


def _cluster_probabilities(
    matrix: FloatArray,
    n_components: int,
    n_marks: int,
) -> FloatArray:
    diagonal = np.diag(matrix).reshape(n_components, n_marks)
    masses = diagonal.sum(axis=1)
    return masses / masses.sum()


def _inverse_softplus(values: FloatArray) -> FloatArray:
    return values + np.log(-np.expm1(-values))


def _signed_rates(
    base: FloatArray,
    matrix: FloatArray,
    alpha: float,
) -> FloatArray:
    diagonal = np.diag(matrix)
    correlation = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    coupling = correlation - np.eye(matrix.shape[0])
    activity = base / (1e-8 + base.sum())
    logits = _inverse_softplus(base)
    return np.logaddexp(0.0, logits + alpha * (coupling @ activity))


def simulate_signed_hawkes_path(
    rng: np.random.RandomState,
    parameters: tuple[ExponentialHawkesParameters, ...],
    *,
    component: int,
    matrix: FloatArray,
    alpha: float,
    horizon: float,
    max_jumps: int = 1000,
) -> MarkedSequence:
    """Simulate the exact signed transform by bounded Ogata thinning."""

    if not 0 <= component < len(parameters):
        raise IndexError("component index out of range")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if horizon <= 0.0 or max_jumps <= 0:
        raise ValueError("invalid horizon or event cap")
    n_components = len(parameters)
    n_marks = parameters[0].n_marks
    if any(item.n_marks != n_marks for item in parameters):
        raise ValueError("all components must have the same mark count")
    dimension = n_components * n_marks
    if matrix.shape != (dimension, dimension):
        raise ValueError("random effect must have shape (K*C, K*C)")
    baselines = np.stack([item.baseline for item in parameters])
    adjacency = np.stack([item.adjacency for item in parameters])
    decays = np.stack([item.decays for item in parameters])
    excitation = np.zeros_like(adjacency)
    time_now = 0.0
    times: list[float] = []
    marks: list[int] = []

    while time_now < horizon and len(times) < max_jumps:
        base = baselines + excitation.sum(axis=2)
        # softplus(h + delta) <= softplus(h) + |delta| and |delta| <= alpha.
        upper = float(base[component].sum() + n_marks * alpha)
        if not np.isfinite(upper) or upper <= 0.0:
            break
        lag = float(rng.exponential(scale=1.0 / upper))
        candidate_time = time_now + lag
        if candidate_time > horizon:
            break
        excitation *= np.exp(-decays * lag)
        candidate_base = baselines + excitation.sum(axis=2)
        transformed = _signed_rates(
            candidate_base.reshape(dimension),
            matrix,
            alpha,
        ).reshape(n_components, n_marks)
        intensities = transformed[component]
        total = float(intensities.sum())
        time_now = candidate_time
        if float(rng.uniform()) * upper > total:
            continue
        threshold = float(rng.uniform()) * total
        mark = int(np.searchsorted(
            np.cumsum(intensities), threshold, side="right"
        ))
        mark = min(mark, n_marks - 1)
        times.append(time_now)
        marks.append(mark)
        excitation[:, :, mark] += (
            adjacency[:, :, mark] * decays[:, :, mark]
        )

    return MarkedSequence(
        times=np.asarray(times, dtype=float),
        marks=np.asarray(marks, dtype=np.int64),
        horizon=float(horizon),
    )


def generate_signed_k3c5(
    *,
    heterogeneity: str,
    parameter_seed: int,
    simulation_seed: int,
    n_per_cluster: int = 400,
    n_clusters: int = 3,
    n_marks: int = 5,
    horizon: float = 9.4,
    true_alpha: float = 1.0,
    max_jumps: int = 1000,
) -> SignedK3C5Dataset:
    """Generate balanced paths at zero, moderate, strong, or suppressive level.

    Zero uses deterministic ``W=I``. Moderate and strong use the same identity
    mean with ``nu=60`` and ``nu=K*C`` respectively. Suppressive uses ``nu=60``
    and a mean correlation with negative mark-0/mark-1 couplings and positive
    mark-2/mark-3 controls inside every component block.
    """

    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be positive")
    dimension = n_clusters * n_marks
    if heterogeneity == "zero":
        degrees_of_freedom = None
        suppressive = False
    elif heterogeneity == "moderate":
        degrees_of_freedom = 4 * dimension
        suppressive = False
    elif heterogeneity == "strong":
        degrees_of_freedom = dimension
        suppressive = False
    elif heterogeneity == "suppressive":
        degrees_of_freedom = 4 * dimension
        suppressive = True
    else:
        raise ValueError(f"unknown heterogeneity level: {heterogeneity}")
    parameter_rng = np.random.RandomState(parameter_seed)
    simulation_rng = np.random.RandomState(simulation_seed)
    parameters = _sample_parameters(parameter_rng, n_clusters, n_marks)
    mean = signed_mean_matrix(
        n_clusters,
        n_marks,
        suppressive=suppressive,
    )
    sequences: list[MarkedSequence] = []
    labels: list[int] = []
    matrices: list[FloatArray] = []
    counts = np.zeros(n_clusters, dtype=np.int64)
    proposals = 0
    while bool(np.any(counts < n_per_cluster)):
        proposals += 1
        matrix = sample_wishart_effect(
            simulation_rng,
            mean,
            degrees_of_freedom,
        )
        probabilities = _cluster_probabilities(
            matrix,
            n_clusters,
            n_marks,
        )
        component = int(simulation_rng.choice(n_clusters, p=probabilities))
        if counts[component] >= n_per_cluster:
            continue
        sequence = simulate_signed_hawkes_path(
            simulation_rng,
            parameters,
            component=component,
            matrix=matrix,
            alpha=true_alpha,
            horizon=horizon,
            max_jumps=max_jumps,
        )
        sequences.append(sequence)
        labels.append(component)
        matrices.append(matrix)
        counts[component] += 1
    return SignedK3C5Dataset(
        sequences=tuple(sequences),
        labels=np.asarray(labels, dtype=np.int64),
        random_effects=np.stack(matrices),
        parameters=parameters,
        mean_matrix=mean,
        heterogeneity=heterogeneity,
        degrees_of_freedom=degrees_of_freedom,
        true_alpha=float(true_alpha),
        parameter_seed=int(parameter_seed),
        simulation_seed=int(simulation_seed),
        horizon=float(horizon),
        proposals=proposals,
    )
