"""Raw-format adapters used by the unified event data module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wishart_tpp.data import (
    REAL_DATASETS,
    SYNTHETIC_DATASET_NAME,
    DatasetSource,
    EventSequence,
)


def _packed(directory: Path, filename: str, columns: tuple[str, ...]):
    """Read required columns from a non-empty packed dataset when present."""

    path = directory / filename
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"empty packed dataset: {path}")
    return frame.loc[:, columns]


def _array(value: object, dtype: np.dtype) -> np.ndarray:
    """Convert one Arrow list scalar to an independent flat array."""

    return np.asarray(value, dtype=dtype).reshape(-1).copy()


def _sequence(times, marks, horizon: float | None = None) -> EventSequence:
    """Sort one raw path stably while keeping marks aligned."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    marks = np.asarray(marks, dtype=np.int64).reshape(-1)
    if times.size == 0 or times.shape != marks.shape:
        raise ValueError("raw trajectory must contain aligned events")
    order = np.argsort(times, kind="stable")
    times, marks = times[order], marks[order]
    return EventSequence(times, marks, float(times[-1]) if horizon is None else horizon)


def _csv_sequence(path: Path, horizon: float | None = None) -> EventSequence:
    """Read one trajectory-per-CSV legacy path."""

    frame = pd.read_csv(path, usecols=["time", "event"])
    return _sequence(frame["time"], frame["event"], horizon)


class LabeledAdapter:
    """Read DAN and Age datasets that share labels and explicit horizons."""

    @staticmethod
    def group(name: str) -> str:
        """Classify Age separately from the synthetic DAN family."""

        return "age" if name == "age" else "synthetic"

    @staticmethod
    def supports(name: str) -> bool:
        """Recognize datasets using the labeled raw layout."""

        return name == "age" or SYNTHETIC_DATASET_NAME.fullmatch(name) is not None

    @staticmethod
    def names(root: Path, packed_filename: str) -> tuple[str, ...]:
        """List available labeled datasets."""

        return tuple(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir()
                and LabeledAdapter.supports(path.name)
                and (
                    (path / packed_filename).is_file()
                    or (path / "clusters.csv").is_file()
                )
            )
        )

    @staticmethod
    def _metadata(name: str, labels: np.ndarray) -> tuple[int, int, float | None]:
        """Validate label cardinality and return dataset dimensions."""

        if name == "age":
            unique = np.unique(labels)
            if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
                raise ValueError("age labels must be contiguous and zero-based")
            return len(unique), 0, 730.0
        match = SYNTHETIC_DATASET_NAME.fullmatch(name)
        assert match is not None
        n_components = int(match.group(2))
        expected = 400 * n_components
        if len(labels) != expected:
            raise ValueError(f"{name} has {len(labels)} labels, expected {expected}")
        horizon = 20.0 if match.group(1) in {"sin", "trunc"} else None
        return n_components, 5, horizon

    def read(
        self,
        root: Path,
        name: str,
        packed_filename: str,
        mixture_components: int,
        split_seed: int,
        split_protocol: str,
    ) -> DatasetSource:
        """Read one labeled dataset without splitting or normalizing it."""

        directory = root / name
        packed = _packed(
            directory,
            packed_filename,
            ("source_id", "label", "horizon", "times", "marks"),
        )
        if packed is None:
            labels = pd.read_csv(directory / "clusters.csv")["cluster_id"].to_numpy(
                dtype=np.int64
            )
            source_ids = np.arange(1, len(labels) + 1, dtype=np.int64)
        else:
            packed = packed.sort_values("source_id", kind="stable")
            labels = packed["label"].to_numpy(dtype=np.int64)
            source_ids = packed["source_id"].to_numpy(dtype=np.int64)
        n_components, n_marks, fixed_horizon = self._metadata(name, labels)
        expected_ids = np.arange(1, len(labels) + 1, dtype=np.int64)
        if not np.array_equal(source_ids, expected_ids):
            raise ValueError(f"{name} source ids must be contiguous and one-based")
        if packed is None:
            sequences = tuple(
                _csv_sequence(directory / f"{source_id}.csv", fixed_horizon)
                for source_id in source_ids
            )
        else:
            sequences = tuple(
                _sequence(
                    _array(row.times, np.float64),
                    _array(row.marks, np.int64),
                    float(row.horizon),
                )
                for row in packed.itertuples(index=False)
            )
            if fixed_horizon is not None and any(
                sequence.horizon != fixed_horizon for sequence in sequences
            ):
                raise ValueError(f"{name} packed horizons must equal {fixed_horizon}")
        observed_marks = {
            int(mark) for sequence in sequences for mark in sequence.marks
        }
        if not observed_marks or min(observed_marks) < 0:
            raise ValueError(f"{name} marks must be non-negative")
        if name == "age":
            n_marks = max(observed_marks) + 1
            if observed_marks != set(range(n_marks)):
                raise ValueError("age marks must be contiguous and zero-based")
        return DatasetSource(
            name,
            n_components,
            n_marks,
            sequences,
            labels,
            source_ids,
        )


class OfficialSplitAdapter:
    """Read public COTIC datasets whose source owns train/val/test splits."""

    @staticmethod
    def group(name: str) -> str:
        """Classify every official-split source as real data."""

        return "real"

    @staticmethod
    def supports(name: str) -> bool:
        """Recognize public COTIC real-data names."""

        return name in REAL_DATASETS

    @staticmethod
    def names(root: Path, packed_filename: str) -> tuple[str, ...]:
        """List available official-split datasets."""

        return tuple(
            name
            for name in REAL_DATASETS
            if (root / name / packed_filename).is_file()
            or all((root / name / split).is_dir() for split in ("train", "val", "test"))
        )

    @staticmethod
    def _numbered_files(directory: Path) -> list[Path]:
        """Sort trajectory CSV files by numeric source identifier."""

        return sorted(
            (path for path in directory.glob("*.csv") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )

    @staticmethod
    def _parts(directory: Path, packed_filename: str) -> list[list[EventSequence]]:
        """Read the three official partitions from packed or legacy storage."""

        packed = _packed(
            directory,
            packed_filename,
            ("split", "source_id", "times", "marks"),
        )
        if packed is None:
            return [
                [
                    _csv_sequence(path)
                    for path in OfficialSplitAdapter._numbered_files(directory / split)
                ]
                for split in ("train", "val", "test")
            ]
        found = set(packed["split"].astype(str))
        if found != {"train", "val", "test"}:
            raise ValueError("packed real dataset must contain train, val, and test")
        parts = []
        for split in ("train", "val", "test"):
            rows = packed.loc[packed["split"] == split].sort_values(
                "source_id", kind="stable"
            )
            if rows.empty or rows["source_id"].duplicated().any():
                raise ValueError(f"packed {split} split has invalid source ids")
            parts.append(
                [
                    _sequence(
                        _array(row.times, np.float64),
                        _array(row.marks, np.int64),
                    )
                    for row in rows.itertuples(index=False)
                ]
            )
        return parts

    @staticmethod
    def _deduplicate(
        parts: list[list[EventSequence]], split_seed: int
    ) -> tuple[list[EventSequence], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split one duplicated official holdout into disjoint validation and test."""

        validation, test = parts[1], parts[2]
        identical = len(validation) == len(test) and all(
            np.array_equal(left.times, right.times)
            and np.array_equal(left.marks, right.marks)
            for left, right in zip(validation, test)
        )
        if not identical:
            raise ValueError("official validation/test are not exact duplicates")
        train_size, holdout_size = len(parts[0]), len(validation)
        holdout = train_size + np.random.RandomState(split_seed).permutation(
            holdout_size
        )
        validation_size = holdout_size // 2
        sequences = parts[0] + validation
        return sequences, (
            np.arange(train_size, dtype=np.int64),
            holdout[:validation_size].astype(np.int64),
            holdout[validation_size:].astype(np.int64),
        )

    def read(
        self,
        root: Path,
        name: str,
        packed_filename: str,
        mixture_components: int,
        split_seed: int,
        split_protocol: str,
    ) -> DatasetSource:
        """Read one official-split dataset without normalizing it."""

        parts = self._parts(root / name, packed_filename)
        if any(not part for part in parts):
            raise ValueError(f"{name} must have non-empty train/val/test splits")
        if split_protocol == "deduplicated_validation_half":
            sequences, split_indices = self._deduplicate(parts, split_seed)
        else:
            sequences = [sequence for part in parts for sequence in part]
            train_size, validation_size, test_size = map(len, parts)
            split_indices = (
                np.arange(train_size, dtype=np.int64),
                np.arange(train_size, train_size + validation_size, dtype=np.int64),
                np.arange(
                    train_size + validation_size,
                    train_size + validation_size + test_size,
                    dtype=np.int64,
                ),
            )
        observed_marks = {
            int(mark) for sequence in sequences for mark in sequence.marks
        }
        if not observed_marks or min(observed_marks) < 0:
            raise ValueError(f"{name} marks must be non-negative")
        total = len(sequences)
        return DatasetSource(
            name,
            mixture_components,
            max(observed_marks) + 1,
            tuple(sequences),
            np.full(total, -1, dtype=np.int64),
            np.arange(total, dtype=np.int64),
            split_indices,
        )
