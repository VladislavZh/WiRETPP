"""K3_C5 data reconstruction from the LaL article's stated exponential DGP.

The original Dropbox archive referenced by ``sequence_clusterers`` has been
deleted.  This module therefore reconstructs the documented data-generating
process rather than claiming byte-for-byte identity with the unavailable
archive.

The article specifies, for target mark ``c`` and source mark ``c_i``,

    lambda_c(t) = mu_c
                  + sum_i a[c, c_i] * delta[c, c_i]
                    * exp(-delta[c, c_i] * (t - t_i)),

with cluster-level ``mu ~ U(0, 1)`` and ``a, delta ~ U(0, 0.6)``.  The
simulation starts from an empty history and uses Ogata thinning.  A maximum of
1000 jumps and a horizon of 50 reproduce the conventions visible in the
authors' historical K3_C5 result metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from lal_wishart.data.hawkes_branching import MarkedSequence


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExponentialHawkesParameters:
    """Parameters of one multivariate exponential Hawkes component."""

    baseline: FloatArray
    adjacency: FloatArray
    decays: FloatArray

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline, dtype=float)
        adjacency = np.asarray(self.adjacency, dtype=float)
        decays = np.asarray(self.decays, dtype=float)
        n_marks = baseline.size
        if baseline.ndim != 1 or n_marks == 0:
            raise ValueError("baseline must be a non-empty vector")
        if adjacency.shape != (n_marks, n_marks):
            raise ValueError("adjacency must be square")
        if decays.shape != adjacency.shape:
            raise ValueError("decays must match adjacency")
        if np.any(baseline < 0) or np.any(adjacency < 0):
            raise ValueError("baseline and adjacency must be non-negative")
        if np.any(decays <= 0):
            raise ValueError("decays must be strictly positive")

    @property
    def n_marks(self) -> int:
        return int(self.baseline.size)

    @property
    def spectral_radius(self) -> float:
        # Because a*delta*exp(-delta*t) integrates to a, the branching
        # matrix is exactly the adjacency matrix.
        return float(np.max(np.abs(np.linalg.eigvals(self.adjacency))))


@dataclass(frozen=True)
class PaperK3C5Dataset:
    """Balanced marked sequences and their cluster-level DGP parameters."""

    sequences: tuple[MarkedSequence, ...]
    labels: IntArray
    parameters: tuple[ExponentialHawkesParameters, ...]
    parameter_seed: int
    simulation_seed: int
    horizon: float
    max_jumps: int


def simulate_exponential_hawkes(
    rng: np.random.RandomState,
    parameters: ExponentialHawkesParameters,
    *,
    horizon: float = 50.0,
    max_jumps: int = 1000,
) -> MarkedSequence:
    """Simulate one path by exact exponential-kernel Ogata thinning."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_jumps <= 0:
        raise ValueError("max_jumps must be positive")

    mu = np.asarray(parameters.baseline, dtype=float)
    adjacency = np.asarray(parameters.adjacency, dtype=float)
    decays = np.asarray(parameters.decays, dtype=float)
    excitation = np.zeros_like(adjacency)
    time_now = 0.0
    times: list[float] = []
    marks: list[int] = []

    while time_now < horizon and len(times) < max_jumps:
        intensities = mu + excitation.sum(axis=1)
        upper = float(intensities.sum())
        if not np.isfinite(upper) or upper <= 0:
            break

        lag = float(rng.exponential(scale=1.0 / upper))
        candidate_time = time_now + lag
        if candidate_time > horizon:
            break

        excitation *= np.exp(-decays * lag)
        candidate_intensities = mu + excitation.sum(axis=1)
        candidate_total = float(candidate_intensities.sum())
        time_now = candidate_time
        if float(rng.uniform()) * upper > candidate_total:
            continue

        threshold = float(rng.uniform()) * candidate_total
        mark = int(np.searchsorted(
            np.cumsum(candidate_intensities),
            threshold,
            side="right",
        ))
        mark = min(mark, parameters.n_marks - 1)
        times.append(time_now)
        marks.append(mark)
        excitation[:, mark] += adjacency[:, mark] * decays[:, mark]

    return MarkedSequence(
        times=np.asarray(times, dtype=float),
        marks=np.asarray(marks, dtype=np.int64),
        horizon=float(horizon),
    )


def _sample_parameters(
    rng: np.random.RandomState,
    n_clusters: int,
    n_marks: int,
) -> tuple[ExponentialHawkesParameters, ...]:
    parameters = []
    for _ in range(n_clusters):
        baseline = rng.uniform(0.0, 1.0, size=n_marks)
        adjacency = rng.uniform(0.0, 0.6, size=(n_marks, n_marks))
        # A zero decay has probability zero but clipping makes the numerical
        # contract explicit for deterministic/random-state edge cases.
        decays = np.maximum(
            rng.uniform(0.0, 0.6, size=(n_marks, n_marks)),
            np.finfo(float).tiny,
        )
        parameters.append(
            ExponentialHawkesParameters(
                baseline=np.asarray(baseline, dtype=float),
                adjacency=np.asarray(adjacency, dtype=float),
                decays=np.asarray(decays, dtype=float),
            )
        )
    return tuple(parameters)


def generate_paper_k3c5(
    *,
    parameter_seed: int = 0,
    simulation_seed: int = 1,
    n_per_cluster: int = 400,
    n_clusters: int = 3,
    n_marks: int = 5,
    horizon: float = 50.0,
    max_jumps: int = 1000,
    shuffle: bool = False,
) -> PaperK3C5Dataset:
    """Generate a deterministic reconstruction of the documented K3_C5 DGP."""

    if n_per_cluster <= 0 or n_clusters <= 0 or n_marks <= 0:
        raise ValueError("dataset dimensions must be positive")
    parameter_rng = np.random.RandomState(parameter_seed)
    simulation_rng = np.random.RandomState(simulation_seed)
    parameters = _sample_parameters(parameter_rng, n_clusters, n_marks)
    sequences: list[MarkedSequence] = []
    labels: list[int] = []
    for cluster, component in enumerate(parameters):
        for _ in range(n_per_cluster):
            sequence = simulate_exponential_hawkes(
                simulation_rng,
                component,
                horizon=horizon,
                max_jumps=max_jumps,
            )
            sequences.append(sequence)
            labels.append(cluster)
    label_array = np.asarray(labels, dtype=np.int64)
    if shuffle:
        order = simulation_rng.permutation(len(sequences))
        sequences = [sequences[index] for index in order]
        label_array = label_array[order]
    return PaperK3C5Dataset(
        sequences=tuple(sequences),
        labels=label_array,
        parameters=parameters,
        parameter_seed=int(parameter_seed),
        simulation_seed=int(simulation_seed),
        horizon=float(horizon),
        max_jumps=int(max_jumps),
    )
