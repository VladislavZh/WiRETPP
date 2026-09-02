"""Small immutable checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


def clone_state_dict(model) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


@dataclass(frozen=True)
class PureCheckpoint:
    validation_nll: float
    step: int
    model_state: dict[str, Tensor]


@dataclass(frozen=True)
class ActiveCheckpoint:
    cycle: int
    validation_nll: float
    model_state: dict[str, Tensor]
    population_means: Tensor
    log_weights: Tensor
    alpha: float
    population_df: float | None = None
