"""Differentiable event and compensator rows produced by an intensity bank."""

from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor


@dataclass
class TPPTrace:
    n_paths: int
    n_components: int
    n_marks: int
    event_times: Tensor
    event_marks: Tensor
    event_paths: Tensor
    event_base_rates: Tensor
    integral_times: Tensor
    integral_weights: Tensor
    integral_paths: Tensor
    integral_base_rates: Tensor
