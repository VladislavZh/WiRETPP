"""Loader for the twelve synthetic datasets from No Two Users are Alike."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lal_wishart.data.hawkes_branching import MarkedSequence


DATASET_PATTERN = re.compile(r"^(?:(sin|trunc)_)?K([2-5])_C5$")
FIXED_HORIZON = 20.0


@dataclass(frozen=True)
class DANSyntheticDataset:
    name: str
    n_components: int
    n_marks: int
    sequences: tuple[MarkedSequence, ...]
    labels: np.ndarray
    source_ids: np.ndarray
    source_horizon: str


@dataclass(frozen=True)
class DANSyntheticSplit:
    train: tuple[MarkedSequence, ...]
    validation: tuple[MarkedSequence, ...]
    test: tuple[MarkedSequence, ...]
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray


def dataset_names(root: Path) -> tuple[str, ...]:
    names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and DATASET_PATTERN.fullmatch(path.name)
    )
    if len(names) != 12:
        raise ValueError(f"expected 12 DAN datasets, found {len(names)}")
    return tuple(names)


def _cluster_labels(path: Path, expected: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if "cluster_id" not in frame:
        raise ValueError(f"cluster_id column missing from {path}")
    labels = frame["cluster_id"].to_numpy(dtype=np.int64)
    if len(labels) != expected:
        raise ValueError(
            f"{path} contains {len(labels)} labels, expected {expected}"
        )
    return labels


def load_dataset(root: Path, name: str) -> DANSyntheticDataset:
    match = DATASET_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid DAN dataset name: {name}")
    family, component_text = match.groups()
    n_components = int(component_text)
    expected = 400 * n_components
    directory = root / name
    labels = _cluster_labels(directory / "clusters.csv", expected)
    expected_labels = set(range(n_components))
    if set(labels.tolist()) != expected_labels:
        raise ValueError(f"unexpected labels in {name}: {sorted(set(labels))}")
    counts = np.bincount(labels, minlength=n_components)
    if not np.array_equal(counts, np.full(n_components, 400)):
        raise ValueError(f"{name} is not balanced: {counts.tolist()}")

    numeric_files = {
        int(path.stem): path
        for path in directory.glob("*.csv")
        if path.stem.isdigit()
    }
    expected_ids = set(range(1, expected + 1))
    if set(numeric_files) != expected_ids:
        missing = sorted(expected_ids - set(numeric_files))
        extra = sorted(set(numeric_files) - expected_ids)
        raise ValueError(f"file id mismatch in {name}: missing={missing}, extra={extra}")

    sequences: list[MarkedSequence] = []
    fixed_horizon = family in {"sin", "trunc"}
    for source_id in range(1, expected + 1):
        frame = pd.read_csv(numeric_files[source_id])
        if not {"time", "event"}.issubset(frame.columns):
            raise ValueError(f"invalid sequence columns in {numeric_files[source_id]}")
        times = frame["time"].to_numpy(dtype=np.float64)
        events = frame["event"].to_numpy(dtype=np.float64)
        if len(times) == 0 or np.any(~np.isfinite(times)):
            raise ValueError(f"invalid times in {numeric_files[source_id]}")
        if np.any(np.diff(times) < 0.0):
            raise ValueError(f"unsorted times in {numeric_files[source_id]}")
        marks = events.astype(np.int64)
        if np.any(events != marks) or np.any((marks < 0) | (marks >= 5)):
            raise ValueError(f"invalid marks in {numeric_files[source_id]}")
        if fixed_horizon:
            if times[-1] > FIXED_HORIZON + 1e-8:
                raise ValueError(f"event beyond horizon in {numeric_files[source_id]}")
            horizon = FIXED_HORIZON
        else:
            if len(times) != 50 or times[-1] <= 0.0:
                raise ValueError(f"invalid fixed-count path in {numeric_files[source_id]}")
            # These paths were stopped after their fiftieth event.  Preserve
            # the common physical time scale and discard the unobserved tail:
            # the likelihood and compensator both end at that last event.
            horizon = float(times[-1])
        sequences.append(
            MarkedSequence(
                times=times,
                marks=marks,
                horizon=horizon,
            )
        )

    return DANSyntheticDataset(
        name=name,
        n_components=n_components,
        n_marks=5,
        sequences=tuple(sequences),
        labels=labels,
        source_ids=np.arange(1, expected + 1, dtype=np.int64),
        source_horizon=(
            "fixed_20"
            if fixed_horizon
            else "sequence_ends_at_fiftieth_event_original_time"
        ),
    )


def shuffled_split(
    dataset: DANSyntheticDataset,
    *,
    seed: int,
) -> DANSyntheticSplit:
    indices = np.arange(len(dataset.sequences), dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_end = int(0.8 * len(indices))
    validation_end = train_end + int(0.1 * len(indices))
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    def select_sequences(selected: np.ndarray) -> tuple[MarkedSequence, ...]:
        return tuple(dataset.sequences[int(index)] for index in selected)

    return DANSyntheticSplit(
        train=select_sequences(train_indices),
        validation=select_sequences(validation_indices),
        test=select_sequences(test_indices),
        train_labels=dataset.labels[train_indices],
        validation_labels=dataset.labels[validation_indices],
        test_labels=dataset.labels[test_indices],
        train_ids=dataset.source_ids[train_indices],
        validation_ids=dataset.source_ids[validation_indices],
        test_ids=dataset.source_ids[test_indices],
    )
