"""Exact branching-representation simulator for multivariate Hawkes processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hawkes_kernels import HawkesKernel


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class MarkedSequence:
    times: FloatArray
    marks: IntArray
    horizon: float

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.marks.ndim != 1:
            raise ValueError("times and marks must be one-dimensional")
        if self.times.size != self.marks.size:
            raise ValueError("times and marks must have the same size")
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be sorted")
        if np.any(self.times < 0) or np.any(self.times > self.horizon):
            raise ValueError("observed times must lie in [0, horizon]")

    @property
    def count(self) -> int:
        return int(self.times.size)


@dataclass(frozen=True)
class BranchingSimulation:
    sequence: MarkedSequence
    n_immigrants: int
    n_parents_processed: int
    n_offspring_proposed: int
    n_offspring_retained: int
    max_generation: int
    burn_in: float

    @property
    def proposed_branching_ratio(self) -> float:
        if self.n_parents_processed == 0:
            return float("nan")
        return self.n_offspring_proposed / self.n_parents_processed


def validate_hawkes_parameters(
    baseline: ArrayLike,
    coupling: ArrayLike,
    kernel: HawkesKernel,
) -> tuple[FloatArray, FloatArray, float]:
    mu = np.asarray(baseline, dtype=float)
    matrix = np.asarray(coupling, dtype=float)
    if mu.ndim != 1 or mu.size == 0 or np.any(mu < 0):
        raise ValueError("baseline must be a non-negative one-dimensional array")
    if matrix.shape != (mu.size, mu.size):
        raise ValueError("coupling must be square with one row per mark")
    if np.any(matrix < 0):
        raise ValueError("coupling entries must be non-negative")
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(kernel.branching_ratio * matrix)))
    )
    if spectral_radius >= 1.0:
        raise ValueError(
            f"unstable Hawkes process: branching spectral radius={spectral_radius:.6g}"
        )
    return mu, matrix, spectral_radius


def stationary_mean_rate(
    baseline: ArrayLike,
    coupling: ArrayLike,
    kernel: HawkesKernel,
) -> FloatArray:
    mu, matrix, _ = validate_hawkes_parameters(baseline, coupling, kernel)
    branching = kernel.branching_ratio * matrix
    return np.linalg.solve(np.eye(mu.size) - branching, mu)


def simulate_hawkes_branching(
    rng: np.random.Generator,
    baseline: ArrayLike,
    coupling: ArrayLike,
    kernel: HawkesKernel,
    *,
    horizon: float,
    burn_in: float = 20.0,
    max_events: int = 1_000_000,
) -> BranchingSimulation:
    """Simulate on ``[-burn_in, horizon]`` and return events in ``[0, horizon]``.

    No event-count truncation is performed. ``max_events`` is a fail-fast
    safety bound that raises instead of silently changing the data law.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    mu, matrix, _ = validate_hawkes_parameters(baseline, coupling, kernel)

    interval_start = -float(burn_in)
    interval_length = horizon + burn_in
    times: list[float] = []
    marks: list[int] = []
    generations: list[int] = []
    for mark, rate in enumerate(mu):
        count = int(rng.poisson(rate * interval_length))
        if count:
            times.extend(rng.uniform(interval_start, horizon, size=count).tolist())
            marks.extend([mark] * count)
            generations.extend([0] * count)

    n_immigrants = len(times)
    n_offspring_proposed = 0
    n_offspring_retained = 0
    max_generation = 0
    parent_index = 0
    mass = kernel.branching_ratio
    while parent_index < len(times):
        parent_time = times[parent_index]
        parent_mark = marks[parent_index]
        child_generation = generations[parent_index] + 1
        for child_mark, weight in enumerate(matrix[:, parent_mark]):
            count = int(rng.poisson(weight * mass))
            n_offspring_proposed += count
            if count == 0:
                continue
            child_times = parent_time + kernel.sample_lags(rng, count)
            retained = child_times <= horizon
            kept = child_times[retained]
            if kept.size:
                times.extend(kept.tolist())
                marks.extend([child_mark] * kept.size)
                generations.extend([child_generation] * kept.size)
                n_offspring_retained += int(kept.size)
                max_generation = max(max_generation, child_generation)
                if len(times) > max_events:
                    raise RuntimeError(
                        "simulation exceeded max_events; refusing to truncate"
                    )
        parent_index += 1

    all_times = np.asarray(times, dtype=float)
    all_marks = np.asarray(marks, dtype=np.int64)
    observed = (all_times >= 0.0) & (all_times <= horizon)
    observed_times = all_times[observed]
    observed_marks = all_marks[observed]
    order = np.argsort(observed_times, kind="stable")
    sequence = MarkedSequence(
        times=np.asarray(observed_times[order], dtype=float),
        marks=np.asarray(observed_marks[order], dtype=np.int64),
        horizon=float(horizon),
    )
    return BranchingSimulation(
        sequence=sequence,
        n_immigrants=n_immigrants,
        n_parents_processed=parent_index,
        n_offspring_proposed=n_offspring_proposed,
        n_offspring_retained=n_offspring_retained,
        max_generation=max_generation,
        burn_in=float(burn_in),
    )

