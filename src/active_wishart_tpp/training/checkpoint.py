"""Atomic model/Adam/random-state persistence for the cleaned protocol."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from active_wishart_tpp.training.artifact_io import sha256, write_json, write_torch
from active_wishart_tpp.training.state import clone_state_dict


def source_digest():
    """Bind resume to package source, independently of checkout paths and newlines."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def seal_completion(directory, signature, cycles):
    """Seal the committed active, selected and result artifacts after final scoring."""
    write_json(
        directory / "completion.json",
        dict(
            completed=True,
            cycles=cycles,
            signature=signature,
            sha256={
                name: sha256(directory / name)
                for name in ("active.pt", "selected.pt", "result.json")
            },
        ),
    )


def completed_result(directory, signature):
    """Reuse a verified completed result without reading or scoring test again."""
    path = directory / "completion.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not receipt.get("completed") or receipt.get("signature") != signature:
        raise ValueError("Completed run protocol/data/initialization mismatch")
    for name in ("active.pt", "selected.pt", "result.json"):
        if sha256(directory / name) != receipt["sha256"].get(name):
            raise ValueError(f"Completed artifact changed: {name}")
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def data_digest(split):
    """Bind checkpoints to the actual ordered trajectories of all three splits."""
    digest = hashlib.sha256()
    for partition in (split.train, split.validation, split.test):
        for sequence, source, label in zip(
            partition.sequences, partition.source_ids, partition.labels
        ):
            for value in (
                sequence.times,
                sequence.marks,
                np.asarray([sequence.horizon]),
                np.asarray([source, label]),
            ):
                digest.update(str((value.dtype.str, value.shape)).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def neural_digest(state):
    """Hash the named neural tensors independently of the checkpoint container."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CheckpointStore:
    """Restore only a matching fixed-protocol run, including its integration counter."""

    def __init__(self, path, signature):
        self.path, self.signature = path, signature

    def save(self, model, optimizer, random_generator, payload, population=None):
        bank = getattr(model, "module", model)
        value = dict(
            payload,
            format="active-wishart-paper-v1",
            signature=self.signature,
            model=clone_state_dict(model),
            optimizer=optimizer.state_dict(),
            population=None if population is None else population.state_dict(),
            integration_draw=bank.integration_rule._training_draw,
            integration_seed=bank.integration_rule._training_seed,
            numpy_generator=random_generator.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        )
        write_torch(self.path, value)

    def restore(self, model, optimizer, random_generator, population=None):
        if not self.path.exists():
            return None
        value = torch.load(self.path, map_location="cpu", weights_only=False)
        if (
            value.get("format") != "active-wishart-paper-v1"
            or value["signature"] != self.signature
        ):
            raise ValueError("Checkpoint protocol/data/initialization mismatch")
        model.load_state_dict(value["model"])
        optimizer.load_state_dict(value["optimizer"])
        if population is not None:
            population.load_state_dict(value["population"])
        random_generator.bit_generator.state = value["numpy_generator"]
        torch.set_rng_state(value["torch_rng"])
        np.random.set_state(value["numpy_rng"])
        random.setstate(value["python_rng"])
        if torch.cuda.is_available() and value["cuda_rng"]:
            torch.cuda.set_rng_state_all(value["cuda_rng"])
        rule = getattr(model, "module", model).integration_rule
        rule._training_draw = value["integration_draw"]
        rule._training_seed = value["integration_seed"]
        return value
