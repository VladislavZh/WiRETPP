"""Atomic serialization helpers for benchmark artifacts."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
import torch


def cpu_tree(value: Any) -> Any:
    """Clone every tensor in a nested value onto CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple((cpu_tree(item) for item in value))
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write an indented JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_torch(path: Path, payload: Any) -> None:
    """Atomically write a CPU-owned PyTorch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_tree(payload), temporary)
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_tree_equal(left: Any, right: Any) -> bool:
    """Return whether two nested checkpoint values are exactly equal."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            (tensor_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            (tensor_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right
