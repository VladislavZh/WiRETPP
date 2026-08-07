"""Sinus-Hawkes K=5,C=5 data with an explicit Wishart random effect.

For every trajectory with class ``z`` we sample

    W_i ~ Wishart_{K*C}(nu, Omega_z / nu).

``Omega_z`` has a strengthened true-cluster block, a common within-cluster
factor, and a weaker global between-cluster factor.  The same draw controls
the block-mass routing signal and a signed convex transformation of the
sinus-Hawkes intensities.  This deliberately matches the structural claim
tested by WiRE instead of adding unrelated observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.data.hawkes_kernels import HawkesKernel, cotic_sinus
from lal_wishart.metrics import (
    adjusted_rand_index,
    normalized_mutual_information,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# Empirical mark-count profiles of the repository's reference sin_K5_C5
# dataset.  Only the normalized profiles are used: the new paths are still
# generated from the explicit sinus-Hawkes/Wishart model below.  The previous
# hand-written profiles were almost exchangeable and made the observed paths
# essentially unidentifiable even though the hidden W matrices were easy to
# classify.
SIN_K5_C5_MARK_COUNT_PROFILES = np.asarray([
    [9.280, 7.278, 7.040, 2.960, 11.732],
    [25.692, 29.548, 30.642, 21.745, 20.732],
    [30.618, 22.160, 36.102, 19.278, 22.740],
    [50.948, 30.112, 47.268, 59.435, 43.275],
    [17.038, 8.185, 23.535, 11.300, 7.868],
], dtype=float)
SIN_K5_C5_MARK_PROFILES = (
    SIN_K5_C5_MARK_COUNT_PROFILES
    / SIN_K5_C5_MARK_COUNT_PROFILES.sum(axis=1, keepdims=True)
)
SIN_K5_C5_BASELINE_TOTALS = np.asarray(
    [0.8, 6.5, 6.5, 16.0, 2.3], dtype=float
)


@dataclass(frozen=True)
class WishartRESinusDataset:
    sequences: tuple[MarkedSequence, ...]
    labels: IntArray
    random_effects: FloatArray
    cluster_mean_matrices: FloatArray
    baselines: FloatArray
    couplings: FloatArray
    degrees_of_freedom: int
    true_alpha: float
    block_local_activity: bool
    within_block_gain: float
    between_block_gain: float
    parameter_seed: int
    simulation_seed: int
    horizon: float
    proposal_counts: IntArray

    @property
    def n_components(self) -> int:
        return int(self.baselines.shape[0])

    @property
    def n_marks(self) -> int:
        return int(self.baselines.shape[1])


@dataclass(frozen=True)
class WishartRESinusSplit:
    train: tuple[MarkedSequence, ...]
    validation: tuple[MarkedSequence, ...]
    test: tuple[MarkedSequence, ...]
    train_labels: IntArray
    validation_labels: IntArray
    test_labels: IntArray
    train_indices: IntArray
    validation_indices: IntArray
    test_indices: IntArray


def cluster_mean_matrices(
    n_components: int,
    n_marks: int,
    *,
    within_cluster_base_strength: float,
    true_cluster_block_boost: float,
    between_cluster_strength: float,
) -> FloatArray:
    """Construct trace-normalized SPD means with explicit block structure."""

    if n_components <= 1 or n_marks <= 1:
        raise ValueError("at least two clusters and marks are required")
    if min(
        within_cluster_base_strength,
        true_cluster_block_boost,
        between_cluster_strength,
    ) < 0.0:
        raise ValueError("Wishart factor strengths must be non-negative")
    dimension = n_components * n_marks
    global_factor = np.ones(dimension, dtype=float)
    means = []
    for true_cluster in range(n_components):
        mean = np.eye(dimension, dtype=float)
        mean += between_cluster_strength * np.outer(
            global_factor, global_factor
        )
        for cluster in range(n_components):
            block = np.zeros(dimension, dtype=float)
            first = cluster * n_marks
            block[first : first + n_marks] = 1.0
            strength = within_cluster_base_strength
            if cluster == true_cluster:
                strength += true_cluster_block_boost
            mean += strength * np.outer(block, block)
        mean *= dimension / float(np.trace(mean))
        if float(np.linalg.eigvalsh(mean).min()) <= 0.0:
            raise RuntimeError("constructed Wishart mean is not SPD")
        means.append(mean)
    return np.stack(means)


def sample_wishart(
    rng: np.random.Generator,
    mean: FloatArray,
    degrees_of_freedom: int,
) -> FloatArray:
    """Sample ``Wishart(nu, mean / nu)`` with expectation ``mean``."""

    dimension = int(mean.shape[0])
    if mean.shape != (dimension, dimension):
        raise ValueError("mean must be square")
    if degrees_of_freedom < dimension:
        raise ValueError("Wishart degrees of freedom must be at least K*C")
    cholesky = np.linalg.cholesky(mean)
    standard = rng.normal(size=(degrees_of_freedom, dimension))
    transformed = standard @ cholesky.T / np.sqrt(degrees_of_freedom)
    return np.asarray(transformed.T @ transformed, dtype=float)


def sinus_hawkes_parameters(
    n_components: int,
    n_marks: int,
    *,
    parameter_seed: int,
    cluster_baseline_totals: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return stable cluster-level sinus-Hawkes parameters.

    For K=5,C=5 the default profiles are calibrated to the observed mark
    proportions of the reference ``sin_K5_C5`` data.  This preserves the
    intended difficulty scale without copying any trajectories.
    """

    if n_components <= 1 or n_marks <= 1:
        raise ValueError("at least two clusters and marks are required")
    rng = np.random.default_rng(parameter_seed)
    marks = np.arange(n_marks, dtype=float)
    if cluster_baseline_totals is not None:
        totals = np.asarray(cluster_baseline_totals, dtype=float)
        if totals.shape != (n_components,) or np.any(totals <= 0.0):
            raise ValueError(
                "cluster baseline totals must have shape (K,) and be positive"
            )
    elif (n_components, n_marks) == (5, 5):
        totals = SIN_K5_C5_BASELINE_TOTALS.copy()
    else:
        totals = None
    baselines = []
    couplings = []
    kernel = cotic_sinus()
    for cluster in range(n_components):
        phase = 2.0 * np.pi * cluster / n_components
        if (n_components, n_marks) == (5, 5):
            baseline = SIN_K5_C5_MARK_PROFILES[cluster].copy()
            baseline *= float(totals[cluster])
            baseline += rng.normal(
                scale=0.002 * float(totals[cluster]), size=n_marks
            )
            baseline = np.maximum(baseline, 0.025)
        else:
            baseline = (
                0.105
                + 0.020 * np.cos(2.0 * np.pi * marks / n_marks + phase)
                + 0.008 * np.sin(4.0 * np.pi * marks / n_marks - phase)
            )
            baseline += rng.normal(scale=0.002, size=n_marks)
            baseline = np.maximum(baseline, 0.025)
            if totals is not None:
                baseline *= float(totals[cluster]) / float(baseline.sum())
        coupling = np.full((n_marks, n_marks), 0.025, dtype=float)
        np.fill_diagonal(coupling, 0.36)
        for source in range(n_marks):
            coupling[(source + 1 + cluster) % n_marks, source] += 0.15
            coupling[(source + 2) % n_marks, source] += 0.06
        coupling += rng.uniform(0.0, 0.01, size=coupling.shape)
        spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(
                kernel.branching_ratio * coupling
            )))
        )
        if spectral_radius >= 0.8:
            coupling *= 0.78 / spectral_radius
        baselines.append(baseline)
        couplings.append(coupling)
    return np.stack(baselines), np.stack(couplings)


def _inverse_softplus(values: FloatArray) -> FloatArray:
    return values + np.log(-np.expm1(-values))


def _signed_convex_rates(
    base: FloatArray,
    matrix: FloatArray,
    alpha: float,
    *,
    n_components: int | None = None,
    n_marks: int | None = None,
    block_local_activity: bool = False,
    within_block_gain: float = 1.0,
    between_block_gain: float = 1.0,
) -> FloatArray:
    if (
        not np.isfinite(within_block_gain)
        or not np.isfinite(between_block_gain)
        or within_block_gain < 0.0
        or between_block_gain < 0.0
    ):
        raise ValueError(
            "within- and between-block gains must be finite and non-negative"
        )
    diagonal = np.diag(matrix)
    correlation = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    coupling = correlation - np.eye(matrix.shape[0])
    global_activity = base / (1e-8 + float(base.sum()))
    if block_local_activity:
        if n_components is None or n_marks is None:
            raise ValueError(
                "block-local activity requires n_components and n_marks"
            )
        if n_components <= 0 or n_marks <= 0:
            raise ValueError("component and mark counts must be positive")
        if n_components * n_marks != len(base):
            raise ValueError("block-local dimensions must equal K*C")
        blocks = base.reshape(n_components, n_marks)
        local_activity = blocks / (
            1e-8 + blocks.sum(axis=1, keepdims=True)
        )
        block_index = np.arange(len(base)) // n_marks
        within_mask = block_index[:, None] == block_index[None, :]
        perturbation = (
            within_block_gain
            * ((coupling * within_mask) @ local_activity.reshape(-1))
            + between_block_gain
            * ((coupling * ~within_mask) @ global_activity)
        )
    else:
        if within_block_gain != 1.0 or between_block_gain != 1.0:
            raise ValueError("custom block gains require block-local activity")
        perturbation = coupling @ global_activity
    logits = (1.0 - alpha) * _inverse_softplus(base) + alpha * perturbation
    return np.logaddexp(0.0, logits)


def _advance_sinus_state(
    exponential: FloatArray,
    sine: FloatArray,
    cosine: FloatArray,
    lag: float,
    kernel: HawkesKernel,
) -> None:
    scaled_lag = kernel.time_scale * lag
    decay = np.exp(-kernel.beta * scaled_lag)
    angle = kernel.omega * scaled_lag
    old_sine = sine.copy()
    old_cosine = cosine.copy()
    exponential *= decay
    sine[:] = decay * (
        old_sine * np.cos(angle) + old_cosine * np.sin(angle)
    )
    cosine[:] = decay * (
        old_cosine * np.cos(angle) - old_sine * np.sin(angle)
    )


def simulate_wishart_re_sinus_path(
    rng: np.random.Generator,
    baselines: FloatArray,
    couplings: FloatArray,
    *,
    component: int,
    matrix: FloatArray,
    alpha: float,
    block_local_activity: bool = False,
    within_block_gain: float = 1.0,
    between_block_gain: float = 1.0,
    horizon: float,
    max_jumps: int = 1000,
) -> tuple[MarkedSequence, int]:
    """Simulate one exact path by bounded thinning of the matched DGP."""

    if not 0 <= component < baselines.shape[0]:
        raise IndexError("component is out of range")
    if baselines.ndim != 2 or couplings.shape != (
        baselines.shape[0], baselines.shape[1], baselines.shape[1]
    ):
        raise ValueError("invalid sinus-Hawkes parameter shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("convex alpha must lie in [0, 1]")
    if (
        not np.isfinite(within_block_gain)
        or not np.isfinite(between_block_gain)
        or within_block_gain < 0.0
        or between_block_gain < 0.0
    ):
        raise ValueError(
            "within- and between-block gains must be finite and non-negative"
        )
    if not block_local_activity and (
        within_block_gain != 1.0 or between_block_gain != 1.0
    ):
        raise ValueError("custom block gains require block-local activity")
    if horizon <= 0.0 or max_jumps <= 0:
        raise ValueError("horizon and event cap must be positive")
    n_components, n_marks = baselines.shape
    dimension = n_components * n_marks
    if matrix.shape != (dimension, dimension):
        raise ValueError("Wishart matrix must have shape (K*C, K*C)")
    kernel = cotic_sinus()
    amplitude = (
        kernel.amplitude_scale
        * kernel.time_scale
        * kernel.alpha
        * kernel.beta
    )
    exponential = np.zeros(n_marks, dtype=float)
    sine = np.zeros(n_marks, dtype=float)
    cosine = np.zeros(n_marks, dtype=float)
    time_now = 0.0
    times: list[float] = []
    marks: list[int] = []
    proposals = 0
    while time_now < horizon and len(times) < max_jumps:
        envelope = amplitude * (1.0 + kernel.rho) * exponential
        upper_base = baselines + np.einsum(
            "kij,j->ki", couplings, envelope
        )
        perturbation_upper = (
            within_block_gain + between_block_gain
            if block_local_activity
            else 1.0
        )
        upper_logits = (
            (1.0 - alpha) * _inverse_softplus(upper_base.reshape(-1))
            + alpha * perturbation_upper
        )
        upper_rates = np.logaddexp(0.0, upper_logits).reshape(
            n_components, n_marks
        )[component]
        upper = float(upper_rates.sum())
        if not np.isfinite(upper) or upper <= 0.0:
            break
        lag = float(rng.exponential(1.0 / upper))
        candidate_time = time_now + lag
        if candidate_time > horizon:
            break
        proposals += 1
        _advance_sinus_state(
            exponential, sine, cosine, lag, kernel
        )
        time_now = candidate_time
        exact_kernel = amplitude * (exponential + kernel.rho * sine)
        base = baselines + np.einsum(
            "kij,j->ki", couplings, exact_kernel
        )
        base = np.maximum(base, np.finfo(float).tiny)
        transformed = _signed_convex_rates(
            base.reshape(-1),
            matrix,
            alpha,
            n_components=n_components,
            n_marks=n_marks,
            block_local_activity=block_local_activity,
            within_block_gain=within_block_gain,
            between_block_gain=between_block_gain,
        ).reshape(n_components, n_marks)
        intensities = transformed[component]
        total = float(intensities.sum())
        if total > upper * (1.0 + 1e-9):
            raise RuntimeError(
                f"invalid thinning bound: total={total}, upper={upper}"
            )
        if float(rng.uniform()) * upper > total:
            continue
        threshold = float(rng.uniform()) * total
        mark = min(
            int(np.searchsorted(
                np.cumsum(intensities), threshold, side="right"
            )),
            n_marks - 1,
        )
        times.append(time_now)
        marks.append(mark)
        exponential[mark] += 1.0
        cosine[mark] += 1.0
    if len(times) >= max_jumps:
        raise RuntimeError("simulation reached max_jumps; refusing to truncate")
    return (
        MarkedSequence(
            times=np.asarray(times, dtype=float),
            marks=np.asarray(marks, dtype=np.int64),
            horizon=float(horizon),
        ),
        proposals,
    )


def generate_wishart_re_sin_k5c5(
    *,
    parameter_seed: int,
    simulation_seed: int,
    n_per_cluster: int = 400,
    n_components: int = 5,
    n_marks: int = 5,
    horizon: float = 20.0,
    degrees_of_freedom: int = 30,
    within_cluster_base_strength: float = 0.7,
    true_cluster_block_boost: float = 2.4,
    between_cluster_strength: float = 0.12,
    true_alpha: float = 0.35,
    block_local_activity: bool = False,
    within_block_gain: float = 1.0,
    between_block_gain: float = 1.0,
    cluster_baseline_totals: FloatArray | None = None,
    max_jumps: int = 1000,
) -> WishartRESinusDataset:
    """Generate a balanced matched-Wishart sinus dataset."""

    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be positive")
    dimension = n_components * n_marks
    if degrees_of_freedom < dimension:
        raise ValueError("degrees_of_freedom must be at least K*C")
    means = cluster_mean_matrices(
        n_components,
        n_marks,
        within_cluster_base_strength=within_cluster_base_strength,
        true_cluster_block_boost=true_cluster_block_boost,
        between_cluster_strength=between_cluster_strength,
    )
    baselines, couplings = sinus_hawkes_parameters(
        n_components,
        n_marks,
        parameter_seed=parameter_seed,
        cluster_baseline_totals=cluster_baseline_totals,
    )
    rng = np.random.default_rng(simulation_seed)
    sequences = []
    labels = []
    effects = []
    proposal_counts = []
    for component in range(n_components):
        for _ in range(n_per_cluster):
            matrix = sample_wishart(
                rng, means[component], degrees_of_freedom
            )
            sequence, proposals = simulate_wishart_re_sinus_path(
                rng,
                baselines,
                couplings,
                component=component,
                matrix=matrix,
                alpha=true_alpha,
                block_local_activity=block_local_activity,
                within_block_gain=within_block_gain,
                between_block_gain=between_block_gain,
                horizon=horizon,
                max_jumps=max_jumps,
            )
            sequences.append(sequence)
            labels.append(component)
            effects.append(matrix)
            proposal_counts.append(proposals)
    return WishartRESinusDataset(
        sequences=tuple(sequences),
        labels=np.asarray(labels, dtype=np.int64),
        random_effects=np.stack(effects),
        cluster_mean_matrices=means,
        baselines=baselines,
        couplings=couplings,
        degrees_of_freedom=int(degrees_of_freedom),
        true_alpha=float(true_alpha),
        block_local_activity=bool(block_local_activity),
        within_block_gain=float(within_block_gain),
        between_block_gain=float(between_block_gain),
        parameter_seed=int(parameter_seed),
        simulation_seed=int(simulation_seed),
        horizon=float(horizon),
        proposal_counts=np.asarray(proposal_counts, dtype=np.int64),
    )


def stratified_split(
    dataset: WishartRESinusDataset,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> WishartRESinusSplit:
    """Create an exact per-cluster split; the test set is the remainder."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("invalid validation_fraction")
    rng = np.random.default_rng(seed)
    train_rows = []
    validation_rows = []
    test_rows = []
    for cluster in range(dataset.n_components):
        indices = np.flatnonzero(dataset.labels == cluster)
        if len(indices) < 3:
            raise ValueError(
                "each cluster needs at least three paths for train/val/test"
            )
        rng.shuffle(indices)
        train_count = max(1, int(round(train_fraction * len(indices))))
        validation_count = max(
            1, int(round(validation_fraction * len(indices)))
        )
        if train_count + validation_count >= len(indices):
            train_count = len(indices) - validation_count - 1
        train_end = train_count
        validation_end = train_end + validation_count
        train_rows.extend(indices[:train_end])
        validation_rows.extend(indices[train_end:validation_end])
        test_rows.extend(indices[validation_end:])
    train = np.asarray(train_rows, dtype=np.int64)
    validation = np.asarray(validation_rows, dtype=np.int64)
    test = np.asarray(test_rows, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    def select(indices: IntArray) -> tuple[MarkedSequence, ...]:
        return tuple(dataset.sequences[int(index)] for index in indices)

    return WishartRESinusSplit(
        train=select(train),
        validation=select(validation),
        test=select(test),
        train_labels=dataset.labels[train],
        validation_labels=dataset.labels[validation],
        test_labels=dataset.labels[test],
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
    )


def _observable_sequence_features(
    sequence: MarkedSequence,
    n_marks: int,
) -> FloatArray:
    """Return label-free count and transition features for DGP QA."""

    marks = sequence.marks
    counts = np.bincount(marks, minlength=n_marks).astype(float)
    transitions = np.asarray([
        np.sum((marks[:-1] == source) & (marks[1:] == target))
        for source in range(n_marks)
        for target in range(n_marks)
    ], dtype=float)
    time_bins = np.histogram(
        sequence.times,
        bins=n_marks,
        range=(0.0, sequence.horizon),
    )[0].astype(float)
    return np.concatenate((counts, transitions, time_bins))


def observable_feature_audit(
    dataset: WishartRESinusDataset,
    *,
    split_seed: int = 42,
    ridge: float = 0.25,
    kmeans_restarts: int = 10,
) -> dict[str, object]:
    """Measure whether the hidden random effect reaches observed events.

    The supervised ridge-LDA number is a preflight identifiability ceiling,
    not a benchmark result.  K-means is a cheap label-free sanity check.  Both
    prevent a strong oracle-W audit from masking an uninformative event DGP.
    """

    if ridge <= 0.0 or kmeans_restarts <= 0:
        raise ValueError("observable audit settings must be positive")
    split = stratified_split(dataset, seed=split_seed)
    features = np.stack([
        _observable_sequence_features(sequence, dataset.n_marks)
        for sequence in dataset.sequences
    ])
    train = split.train_indices
    validation = split.validation_indices
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    centroids = np.stack([
        standardized[train][split.train_labels == cluster].mean(axis=0)
        for cluster in range(dataset.n_components)
    ])
    pooled = np.cov(standardized[train].T)
    pooled = np.atleast_2d(pooled) + ridge * np.eye(standardized.shape[1])
    inverse = np.linalg.inv(pooled)
    delta = standardized[validation, None, :] - centroids[None, :, :]
    distances = np.einsum("nkd,df,nkf->nk", delta, inverse, delta)
    validation_prediction = distances.argmin(axis=1)

    # Unsupervised check on all paths.  It is intentionally diagnostic-only;
    # it is never used to tune or select a trained model.
    all_mean = features.mean(axis=0)
    all_scale = features.std(axis=0)
    all_scale[all_scale < 1e-8] = 1.0
    all_standardized = (features - all_mean) / all_scale
    rng = np.random.default_rng(split_seed)
    best_inertia = np.inf
    best_labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(kmeans_restarts):
        centers = all_standardized[
            rng.choice(
                len(all_standardized),
                dataset.n_components,
                replace=False,
            )
        ].copy()
        for _ in range(100):
            labels = (
                (all_standardized[:, None, :] - centers[None, :, :]) ** 2
            ).sum(axis=2).argmin(axis=1)
            updated = np.stack([
                all_standardized[labels == cluster].mean(axis=0)
                if np.any(labels == cluster)
                else centers[cluster]
                for cluster in range(dataset.n_components)
            ])
            if np.allclose(updated, centers):
                break
            centers = updated
        inertia = float(
            ((all_standardized - centers[labels]) ** 2).sum()
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return {
        "observable_validation_ridge_lda_accuracy": float(
            np.mean(validation_prediction == split.validation_labels)
        ),
        "observable_validation_ridge_lda_ari": float(
            adjusted_rand_index(
                split.validation_labels, validation_prediction
            )
        ),
        "observable_validation_ridge_lda_nmi": float(
            normalized_mutual_information(
                split.validation_labels, validation_prediction
            )
        ),
        "observable_all_kmeans_ari": float(
            adjusted_rand_index(dataset.labels, best_labels)
        ),
        "observable_all_kmeans_nmi": float(
            normalized_mutual_information(dataset.labels, best_labels)
        ),
        "observable_all_kmeans_cluster_sizes": np.bincount(
            best_labels, minlength=dataset.n_components
        ).tolist(),
    }


def dataset_audit(dataset: WishartRESinusDataset) -> dict[str, object]:
    """Return oracle checks that distinguish within and between effects."""

    k = dataset.n_components
    c = dataset.n_marks
    block_mass = np.diagonal(
        dataset.random_effects, axis1=1, axis2=2
    ).reshape(len(dataset.labels), k, c).sum(axis=2)
    routed = block_mass.argmax(axis=1)
    true_mass = block_mass[np.arange(len(dataset.labels)), dataset.labels]
    other_mass = (
        (block_mass.sum(axis=1) - true_mass) / (k - 1)
    )
    within_true = []
    within_other = []
    between = []
    for label, matrix in zip(
        dataset.labels, dataset.random_effects, strict=True
    ):
        diagonal = np.diag(matrix)
        correlation = matrix / np.sqrt(
            diagonal[:, None] * diagonal[None, :]
        )
        for left in range(k):
            rows = slice(left * c, (left + 1) * c)
            block = correlation[rows, rows]
            values = block[~np.eye(c, dtype=bool)]
            (within_true if left == label else within_other).extend(values)
            for right in range(left + 1, k):
                columns = slice(right * c, (right + 1) * c)
                between.extend(correlation[rows, columns].reshape(-1))
    counts = np.asarray([sequence.count for sequence in dataset.sequences])
    selected_rates = []
    for label, matrix in zip(
        dataset.labels, dataset.random_effects, strict=True
    ):
        rates = _signed_convex_rates(
            dataset.baselines.reshape(-1),
            matrix,
            dataset.true_alpha,
            n_components=k,
            n_marks=c,
            block_local_activity=dataset.block_local_activity,
            within_block_gain=dataset.within_block_gain,
            between_block_gain=dataset.between_block_gain,
        ).reshape(k, c)
        selected_rates.append(rates[label])
    selected_rates = np.stack(selected_rates)
    relative_rate_sd = []
    for cluster in range(k):
        values = selected_rates[dataset.labels == cluster]
        relative_rate_sd.append(float(np.mean(
            values.std(axis=0) / np.maximum(values.mean(axis=0), 1e-12)
        )))
    result = {
        "n_sequences": len(dataset.sequences),
        "cluster_counts": np.bincount(
            dataset.labels, minlength=k
        ).tolist(),
        "mean_events": float(counts.mean()),
        "minimum_events": int(counts.min()),
        "maximum_events": int(counts.max()),
        "mean_thinning_proposals": float(dataset.proposal_counts.mean()),
        "oracle_block_mass_routing_accuracy": float(
            np.mean(routed == dataset.labels)
        ),
        "mean_true_block_mass": float(true_mass.mean()),
        "mean_other_block_mass": float(other_mass.mean()),
        "mean_within_true_block_correlation": float(np.mean(within_true)),
        "mean_within_other_block_correlation": float(np.mean(within_other)),
        "mean_between_block_correlation": float(np.mean(between)),
        "random_effect_rate_relative_sd_by_cluster": relative_rate_sd,
        "random_effect_rate_relative_sd_mean": float(
            np.mean(relative_rate_sd)
        ),
        "wishart_degrees_of_freedom": dataset.degrees_of_freedom,
        "true_alpha": dataset.true_alpha,
        "block_local_activity": dataset.block_local_activity,
        "within_block_gain": dataset.within_block_gain,
        "between_block_gain": dataset.between_block_gain,
        "parameter_seed": dataset.parameter_seed,
        "simulation_seed": dataset.simulation_seed,
        "horizon": dataset.horizon,
    }
    result.update(observable_feature_audit(dataset))
    return result


def save_dataset(path: Path, dataset: WishartRESinusDataset) -> None:
    """Save the generated paths and oracle random effects as one NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    times = []
    marks = []
    for sequence in dataset.sequences:
        times.append(sequence.times)
        marks.append(sequence.marks)
        offsets.append(offsets[-1] + sequence.count)
    metadata = {
        "degrees_of_freedom": dataset.degrees_of_freedom,
        "true_alpha": dataset.true_alpha,
        "block_local_activity": dataset.block_local_activity,
        "within_block_gain": dataset.within_block_gain,
        "between_block_gain": dataset.between_block_gain,
        "parameter_seed": dataset.parameter_seed,
        "simulation_seed": dataset.simulation_seed,
        "horizon": dataset.horizon,
    }
    np.savez_compressed(
        path,
        times=np.concatenate(times) if times else np.empty(0),
        marks=np.concatenate(marks) if marks else np.empty(0, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels=dataset.labels,
        random_effects=dataset.random_effects.astype(np.float32),
        cluster_mean_matrices=dataset.cluster_mean_matrices,
        baselines=dataset.baselines,
        couplings=dataset.couplings,
        proposal_counts=dataset.proposal_counts,
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_saved_dataset(path: Path) -> WishartRESinusDataset:
    """Load a dataset written by :func:`save_dataset`."""

    with np.load(path, allow_pickle=False) as values:
        offsets = values["offsets"]
        times = values["times"]
        marks = values["marks"]
        metadata = json.loads(str(values["metadata"]))
        sequences = tuple(
            MarkedSequence(
                times=np.asarray(times[offsets[i] : offsets[i + 1]], dtype=float),
                marks=np.asarray(marks[offsets[i] : offsets[i + 1]], dtype=np.int64),
                horizon=float(metadata["horizon"]),
            )
            for i in range(len(offsets) - 1)
        )
        return WishartRESinusDataset(
            sequences=sequences,
            labels=np.asarray(values["labels"], dtype=np.int64),
            random_effects=np.asarray(values["random_effects"], dtype=float),
            cluster_mean_matrices=np.asarray(
                values["cluster_mean_matrices"], dtype=float
            ),
            baselines=np.asarray(values["baselines"], dtype=float),
            couplings=np.asarray(values["couplings"], dtype=float),
            degrees_of_freedom=int(metadata["degrees_of_freedom"]),
            true_alpha=float(metadata["true_alpha"]),
            block_local_activity=bool(
                metadata.get("block_local_activity", False)
            ),
            within_block_gain=float(metadata.get("within_block_gain", 1.0)),
            between_block_gain=float(
                metadata.get("between_block_gain", 1.0)
            ),
            parameter_seed=int(metadata["parameter_seed"]),
            simulation_seed=int(metadata["simulation_seed"]),
            horizon=float(metadata["horizon"]),
            proposal_counts=np.asarray(
                values["proposal_counts"], dtype=np.int64
            ),
        )
