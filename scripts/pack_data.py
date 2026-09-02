"""Pack trajectory-per-CSV datasets into one verified Parquet file each."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wishart_tpp.data import REAL_DATASETS, SYNTHETIC_DATASET_NAME

PACKED_NAME = "events.parquet"
LABELED_SCHEMA = pa.schema(
    [
        ("source_id", pa.int64()),
        ("label", pa.int64()),
        ("horizon", pa.float64()),
        ("times", pa.list_(pa.float64())),
        ("marks", pa.list_(pa.int64())),
    ]
)
REAL_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("source_id", pa.int64()),
        ("times", pa.list_(pa.float64())),
        ("marks", pa.list_(pa.int64())),
    ]
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset directory names")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-amazon",
        action="store_true",
        help="permit packing Amazon after any active Amazon run has stopped",
    )
    return parser


def _numbered_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def _csv_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["time", "event"])
    times = frame["time"].to_numpy(dtype=np.float64)
    marks = frame["event"].to_numpy(dtype=np.int64)
    if times.size == 0 or times.shape != marks.shape:
        raise ValueError(f"invalid trajectory: {path}")
    return times, marks


def _labeled_rows(directory: Path) -> Iterator[dict[str, object]]:
    labels = pd.read_csv(directory / "clusters.csv")["cluster_id"].to_numpy(
        dtype=np.int64
    )
    match = SYNTHETIC_DATASET_NAME.fullmatch(directory.name)
    if directory.name == "age":
        horizon = 730.0
    elif match is not None:
        expected = 400 * int(match.group(2))
        if len(labels) != expected:
            raise ValueError(
                f"{directory.name} has {len(labels)} labels, expected {expected}"
            )
        horizon = 20.0 if match.group(1) in {"sin", "trunc"} else None
    else:
        raise ValueError(f"not a labeled dataset: {directory.name}")

    for source_id, label in enumerate(labels, start=1):
        times, marks = _csv_arrays(directory / f"{source_id}.csv")
        yield {
            "source_id": source_id,
            "label": int(label),
            "horizon": float(times[-1]) if horizon is None else horizon,
            "times": times,
            "marks": marks,
        }


def _real_rows(directory: Path) -> Iterator[dict[str, object]]:
    for split in ("train", "val", "test"):
        files = _numbered_files(directory / split)
        if not files:
            raise ValueError(f"empty real-data split: {directory / split}")
        for path in files:
            times, marks = _csv_arrays(path)
            yield {
                "split": split,
                "source_id": int(path.stem),
                "times": times,
                "marks": marks,
            }


def _update_digest(hasher, row: dict[str, object], schema: pa.Schema) -> None:
    for field in schema:
        value = row[field.name]
        if pa.types.is_string(field.type):
            encoded = str(value).encode("utf-8")
            hasher.update(struct.pack("<Q", len(encoded)))
            hasher.update(encoded)
        elif pa.types.is_int64(field.type):
            hasher.update(struct.pack("<q", int(value)))
        elif pa.types.is_float64(field.type):
            hasher.update(struct.pack("<d", float(value)))
        elif pa.types.is_list(field.type):
            dtype = "<f8" if pa.types.is_float64(field.type.value_type) else "<i8"
            values = np.asarray(value, dtype=dtype).reshape(-1)
            hasher.update(struct.pack("<Q", len(values)))
            hasher.update(values.tobytes())
        else:
            raise TypeError(f"unsupported digest field: {field}")


def _write_verified(
    directory: Path,
    rows: Iterable[dict[str, object]],
    schema: pa.Schema,
    *,
    force: bool,
) -> tuple[int, int]:
    target = directory / PACKED_NAME
    temporary = directory / f".{PACKED_NAME}.tmp"
    if target.exists() and not force:
        raise FileExistsError(f"use --force to replace {target}")
    temporary.unlink(missing_ok=True)

    expected = hashlib.sha256()
    count = 0
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        batch: list[dict[str, object]] = []
        for row in rows:
            _update_digest(expected, row, schema)
            batch.append(row)
            count += 1
            if len(batch) == 512:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"no trajectories found in {directory}")

    actual = hashlib.sha256()
    actual_count = 0
    for record_batch in pq.ParquetFile(temporary).iter_batches(batch_size=512):
        for row in record_batch.to_pylist():
            _update_digest(actual, row, schema)
            actual_count += 1
    if actual_count != count or actual.digest() != expected.digest():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet verification failed for {directory.name}")
    temporary.replace(target)
    return count, target.stat().st_size


def _validated_directory(root: Path, name: str) -> Path:
    root = root.resolve()
    directory = (root / name).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ValueError(f"unsafe or missing dataset directory: {directory}")
    return directory


def _delete_csv_source(directory: Path, real: bool) -> int:
    removed = 0
    if real:
        for split in ("train", "val", "test"):
            target = (directory / split).resolve()
            if target.parent != directory.resolve() or not target.is_dir():
                raise ValueError(f"unsafe or missing split directory: {target}")
            removed += sum(1 for _ in target.rglob("*") if _.is_file())
            shutil.rmtree(target)
    else:
        for path in directory.glob("*.csv"):
            path.unlink()
            removed += 1
    return removed


def _discover(root: Path) -> list[str]:
    names = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        if directory.name == "age" or SYNTHETIC_DATASET_NAME.fullmatch(directory.name):
            if (directory / "clusters.csv").is_file():
                names.append(directory.name)
        elif directory.name in REAL_DATASETS and all(
            (directory / split).is_dir() for split in ("train", "val", "test")
        ):
            names.append(directory.name)
    return sorted(names)


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    names = args.datasets or _discover(root)
    if "amazon" in names and not args.allow_amazon:
        raise ValueError(
            "Amazon is protected; wait for the active run and pass --allow-amazon"
        )

    for name in names:
        directory = _validated_directory(root, name)
        real = name in REAL_DATASETS
        schema = REAL_SCHEMA if real else LABELED_SCHEMA
        rows = _real_rows(directory) if real else _labeled_rows(directory)
        count, size = _write_verified(directory, rows, schema, force=args.force)
        removed = _delete_csv_source(directory, real) if args.delete_source else 0
        print(
            f"{name}: trajectories={count} parquet_bytes={size} "
            f"removed_csv_files={removed}"
        )


if __name__ == "__main__":
    main()
