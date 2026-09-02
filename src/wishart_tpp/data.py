"""One data contract for every marked-event dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKED_EVENTS = "events.parquet"
REAL_DATASETS = ("amazon", "linkedin", "meme", "mimic_2", "retweet", "so")
SYNTHETIC_DATASET_NAME = re.compile(r"^(?:(sin|trunc)_)?K([2-5])_C5$")


@dataclass(frozen=True)
class EventSequence:
    """Represent one observed marked-event trajectory."""

    times: np.ndarray
    marks: np.ndarray
    horizon: float

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.marks.ndim != 1:
            raise ValueError("times and marks must be one-dimensional")
        if self.times.shape != self.marks.shape or self.times.size == 0:
            raise ValueError("times and marks must have equal non-zero length")
        if not np.all(np.isfinite(self.times)) or not np.isfinite(self.horizon):
            raise ValueError("times and horizon must be finite")
        if np.any(np.diff(self.times) < 0.0):
            raise ValueError("event times must be sorted")
        if np.any(self.times < 0.0) or np.any(self.times > self.horizon):
            raise ValueError("events must lie inside the observation horizon")

    @property
    def count(self) -> int:
        """Return the number of observed events."""

        return int(self.times.size)


@dataclass(frozen=True)
class TimeNormalization:
    """Record the train-only time transformation applied to every split."""

    unit_rate: float
    exponential_rate: float
    mode: str

    @property
    def rate(self) -> float:
        """Return the complete multiplicative time scale."""

        return self.unit_rate * self.exponential_rate

    def as_dict(self) -> dict[str, float | str]:
        """Serialize the fitted normalization without losing either factor."""

        return {
            "mode": self.mode,
            "unit_rate": self.unit_rate,
            "exponential_rate": self.exponential_rate,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class EventDataset:
    """Hold one normalized dataset and its immutable split assignment."""

    name: str
    n_components: int
    n_marks: int
    sequences: tuple[EventSequence, ...]
    labels: np.ndarray
    source_ids: np.ndarray
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    normalization: TimeNormalization


@dataclass(frozen=True)
class DatasetPartition:
    """Keep aligned trajectories, labels, and source identifiers."""

    sequences: tuple[EventSequence, ...]
    labels: np.ndarray
    source_ids: np.ndarray

    @property
    def event_count(self) -> int:
        """Return the number of events in the partition."""

        return sum(sequence.count for sequence in self.sequences)

    @property
    def exposure(self) -> float:
        """Return the total normalized observation horizon."""

        return float(sum(sequence.horizon for sequence in self.sequences))

    def select(self, indices: np.ndarray) -> DatasetPartition:
        """Select aligned rows by positional index."""

        return DatasetPartition(
            sequences=tuple(self.sequences[int(index)] for index in indices),
            labels=self.labels[indices],
            source_ids=self.source_ids[indices],
        )


@dataclass(frozen=True)
class DatasetSplit:
    """Expose the train, validation, and test partitions."""

    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition


@dataclass(frozen=True)
class PreparedDataset:
    """Return dataset metadata and partitions as one preparation result."""

    dataset: EventDataset
    split: DatasetSplit


@dataclass(frozen=True)
class DatasetSource:
    """Carry adapter output before common normalization and partitioning."""

    name: str
    n_components: int
    n_marks: int
    sequences: tuple[EventSequence, ...]
    labels: np.ndarray
    source_ids: np.ndarray
    fixed_split: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


class P99TimeNormalizer:
    """Fit and apply the train-only P99 exponential COTIC normalization."""

    @staticmethod
    def _mean_ratio(value: float) -> float:
        """Evaluate the truncated-exponential mean ratio stably."""

        if value < 1e-4:
            return 0.5 - value / 12.0 + value**3 / 720.0
        if value > 50.0:
            return 1.0 / value
        return 1.0 / value - 1.0 / np.expm1(value)

    @classmethod
    def fit(cls, train: tuple[EventSequence, ...]) -> TimeNormalization:
        """Estimate both unit conversion and exponential rate from train only."""

        maximum_absolute = max(
            float(np.abs(sequence.times).max()) for sequence in train
        )
        if maximum_absolute >= 100_000_000.0:
            unit_rate = 1.0 / 86_400.0
            unit_mode = "unix_seconds_to_days"
        else:
            unit_rate = 1.0
            unit_mode = "already_relative_or_day_scaled"
        delta_parts = [
            np.diff(sequence.times) * unit_rate
            for sequence in train
            if sequence.count > 1
        ]
        if not delta_parts:
            raise ValueError("P99 normalization requires at least one return time")
        deltas = np.concatenate(delta_parts)
        percentile = float(np.quantile(deltas, 0.99))
        if not np.isfinite(percentile) or percentile <= 0.0:
            raise ValueError("P99 return time must be finite and positive")
        truncated_mean = float(deltas[deltas <= percentile].mean())
        ratio = truncated_mean / percentile
        if not 0.0 < ratio:
            raise ValueError("P99 exponential fit requires positive return times")
        if ratio >= 0.5:
            exponential_rate = 1.0 / truncated_mean
        else:
            low, high = 1e-12, max(2.0, 2.0 / ratio)
            for _ in range(100):
                middle = 0.5 * (low + high)
                if cls._mean_ratio(middle) > ratio:
                    low = middle
                else:
                    high = middle
            exponential_rate = (0.5 * (low + high)) / percentile
        return TimeNormalization(
            unit_rate,
            exponential_rate,
            f"{unit_mode}+train_p99_exponential",
        )

    @staticmethod
    def transform(
        sequence: EventSequence, normalization: TimeNormalization
    ) -> EventSequence:
        """Shift one path to zero and scale its events and observation horizon."""

        origin = float(sequence.times[0])
        times = (sequence.times - origin) * normalization.rate
        horizon = max(
            (float(sequence.horizon) - origin) * normalization.rate,
            float(times[-1]),
            1e-8,
        )
        return EventSequence(times, sequence.marks.copy(), horizon)


class EventDataModule:
    """Prepare every supported dataset through one split and normalization path."""

    def __init__(
        self,
        root: Path,
        split_seed: int = 42,
        mixture_components: int = 5,
        split_protocol: str = "official",
        packed_filename: str = PACKED_EVENTS,
    ) -> None:
        if mixture_components < 1:
            raise ValueError("mixture_components must be positive")
        if split_protocol not in {"official", "deduplicated_validation_half"}:
            raise ValueError(f"unsupported real-data split protocol: {split_protocol}")
        if not packed_filename or Path(packed_filename).name != packed_filename:
            raise ValueError("packed filename must be a non-empty local filename")
        self.root = root
        self.split_seed = split_seed
        self.mixture_components = mixture_components
        self.split_protocol = split_protocol
        self.packed_filename = packed_filename

    @staticmethod
    def _adapters():
        """Return only the adapters required by incompatible raw layouts."""

        from wishart_tpp.data_adapters import LabeledAdapter, OfficialSplitAdapter

        return LabeledAdapter(), OfficialSplitAdapter()

    def names(self, group: str | None = None) -> tuple[str, ...]:
        """List datasets exposed by the requested adapter group."""

        return tuple(
            sorted(
                name
                for adapter in self._adapters()
                for name in adapter.names(self.root, self.packed_filename)
                if group is None or adapter.group(name) == group
            )
        )

    def _read(self, name: str) -> DatasetSource:
        """Delegate only incompatible raw-format parsing to one adapter."""

        for adapter in self._adapters():
            if adapter.supports(name):
                return adapter.read(
                    self.root,
                    name,
                    self.packed_filename,
                    self.mixture_components,
                    self.split_seed,
                    self.split_protocol,
                )
        raise ValueError(f"unsupported dataset: {name}")

    def _random_split(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the historical seeded 80/10/10 assignment."""

        indices = np.arange(size, dtype=np.int64)
        np.random.RandomState(self.split_seed).shuffle(indices)
        train_end = int(0.8 * size)
        validation_end = train_end + int(0.1 * size)
        return (
            indices[:train_end],
            indices[train_end:validation_end],
            indices[validation_end:],
        )

    @staticmethod
    def _partition(dataset: EventDataset, indices: np.ndarray) -> DatasetPartition:
        """Build one aligned partition from immutable dataset rows."""

        return DatasetPartition(
            tuple(dataset.sequences[int(index)] for index in indices),
            dataset.labels[indices],
            dataset.source_ids[indices],
        )

    @staticmethod
    def _subset_source(source: DatasetSource, selected: np.ndarray) -> DatasetSource:
        """Restrict one source while preserving aligned row fields."""

        return DatasetSource(
            source.name,
            source.n_components,
            source.n_marks,
            tuple(source.sequences[int(index)] for index in selected),
            source.labels[selected],
            source.source_ids[selected],
        )

    def _limit(
        self, source: DatasetSource, limit: int | None, cohort_seed: int
    ) -> DatasetSource:
        """Apply a deterministic cohort limit before common normalization."""

        if limit is None or limit >= len(source.sequences):
            return source
        random = np.random.RandomState(cohort_seed)
        if source.fixed_split is None:
            return self._subset_source(
                source, random.permutation(len(source.sequences))[:limit]
            )
        sizes = np.array([len(indices) for indices in source.fixed_split])
        raw_counts = sizes * limit / sizes.sum()
        counts = np.floor(raw_counts).astype(int)
        for index in np.argsort(-(raw_counts - counts))[: limit - counts.sum()]:
            counts[index] += 1
        selected_parts = [
            random.permutation(indices)[:count]
            for indices, count in zip(source.fixed_split, counts)
        ]
        selected = np.concatenate(selected_parts)
        limited = self._subset_source(source, selected)
        ends = np.cumsum(counts)
        return DatasetSource(
            limited.name,
            limited.n_components,
            limited.n_marks,
            limited.sequences,
            limited.labels,
            limited.source_ids,
            (
                np.arange(0, ends[0], dtype=np.int64),
                np.arange(ends[0], ends[1], dtype=np.int64),
                np.arange(ends[1], ends[2], dtype=np.int64),
            ),
        )

    def prepare(
        self,
        name: str,
        trajectory_limit: int | None = None,
        cohort_seed: int = 20260824,
    ) -> PreparedDataset:
        """Read, split, fit on train, and transform one dataset."""

        source = self._limit(self._read(name), trajectory_limit, cohort_seed)
        split_indices = source.fixed_split or self._random_split(len(source.sequences))
        train = tuple(source.sequences[int(index)] for index in split_indices[0])
        normalization = P99TimeNormalizer.fit(train)
        dataset = EventDataset(
            source.name,
            source.n_components,
            source.n_marks,
            tuple(
                P99TimeNormalizer.transform(sequence, normalization)
                for sequence in source.sequences
            ),
            source.labels,
            source.source_ids,
            split_indices,
            normalization,
        )
        split = DatasetSplit(
            self._partition(dataset, split_indices[0]),
            self._partition(dataset, split_indices[1]),
            self._partition(dataset, split_indices[2]),
        )
        return PreparedDataset(dataset, split)
