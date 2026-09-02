"""Atomic persistence for resumable Active-EM cycle boundaries."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wishart_tpp.training.state import ActiveCheckpoint, clone_state_dict


class CycleCheckpointStore:
    """Save and restore every state needed for exact cycle-boundary resume."""

    @staticmethod
    def _cpu_clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: CycleCheckpointStore._cpu_clone(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [CycleCheckpointStore._cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(CycleCheckpointStore._cpu_clone(item) for item in value)
        return value

    @staticmethod
    def _best_payload(checkpoint: ActiveCheckpoint) -> dict[str, Any]:
        return {
            "cycle": checkpoint.cycle,
            "validation_nll": checkpoint.validation_nll,
            "model_state": checkpoint.model_state,
            "population_means": checkpoint.population_means,
            "log_weights": checkpoint.log_weights,
            "alpha": checkpoint.alpha,
            "population_df": checkpoint.population_df,
        }

    @staticmethod
    def _best_from_payload(payload: dict[str, Any]) -> ActiveCheckpoint:
        population_df = payload.get("population_df")
        return ActiveCheckpoint(
            int(payload["cycle"]),
            float(payload["validation_nll"]),
            payload["model_state"],
            payload["population_means"],
            payload["log_weights"],
            float(payload["alpha"]),
            float(population_df) if population_df is not None else None,
        )

    def save(
        self,
        path: Path,
        model,
        optimizer,
        active_state: dict[str, Any],
        best: ActiveCheckpoint,
        history: list[dict[str, float]],
        completed_cycle: int,
        signature: dict[str, Any],
        scheduler=None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        payload = {
            "format": "wishart_active_cycle_checkpoint",
            "version": 1,
            "completed_cycle": completed_cycle,
            "signature": signature,
            "model_state": clone_state_dict(model),
            "optimizer_state": self._cpu_clone(optimizer.state_dict()),
            "scheduler_state": (
                self._cpu_clone(scheduler.state_dict())
                if scheduler is not None
                else None
            ),
            "active_state": self._cpu_clone(active_state),
            "best": self._best_payload(best),
            "history": history,
            "torch_rng_state": torch.get_rng_state().cpu().clone(),
            "cuda_rng_state_all": (
                [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(
        self,
        path: Path,
        model,
        optimizer,
        scheduler,
        signature: dict[str, Any],
        target_cycles: int,
    ) -> tuple[int, dict[str, Any], ActiveCheckpoint, list[dict[str, float]]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "wishart_active_cycle_checkpoint":
            raise ValueError(f"unsupported active checkpoint format: {path}")
        if int(payload.get("version", -1)) != 1:
            raise ValueError(f"unsupported active checkpoint version: {path}")
        saved_signature = payload.get("signature")
        legacy_scheduler_resume = False
        if saved_signature != signature:
            legacy_signature = dict(signature)
            legacy_signature.pop("reduce_lr_on_plateau", None)
            legacy_scheduler_resume = (
                scheduler is not None and saved_signature == legacy_signature
            )
            if not legacy_scheduler_resume:
                raise ValueError(
                    "active checkpoint protocol does not match the requested "
                    f"run: {path}"
                )
        completed_cycle = int(payload["completed_cycle"])
        if completed_cycle > target_cycles:
            raise ValueError(
                f"active checkpoint is already at cycle {completed_cycle}, "
                f"beyond requested cycle {target_cycles}"
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = payload["history"]
        if scheduler is not None:
            if payload.get("scheduler_state") is not None:
                scheduler.load_state_dict(payload["scheduler_state"])
            elif legacy_scheduler_resume:
                for row in history:
                    scheduler.step(float(row["validation_nll_per_exposure"]))
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        return (
            completed_cycle,
            payload["active_state"],
            self._best_from_payload(payload["best"]),
            history,
        )
