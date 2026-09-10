"""Continuous intensity head for a K by C output bank."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class CoticIntensityHead(nn.Module):
    """Evaluate all component intensities from a history state and elapsed time."""

    def __init__(
        self, hidden_size: int, output_size: int, *, base_mark_count: int
    ) -> None:
        super().__init__()
        self.time_slope = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_bias = nn.Parameter(
            torch.full((hidden_size, hidden_size), 1.0 / hidden_size)
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.log_scale = nn.Parameter(
            torch.full((output_size,), math.log(base_mark_count))
        )

    def forward(self, states: Tensor, elapsed: Tensor) -> Tensor:
        hidden = torch.nn.functional.leaky_relu(
            elapsed[..., None] * self.time_slope(states) + states @ self.time_bias,
            negative_slope=0.1,
        )
        return torch.nn.functional.softplus(self.output(hidden)) * self.log_scale.exp()
