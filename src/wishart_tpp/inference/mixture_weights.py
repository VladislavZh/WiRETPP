"""Closed-form mixture-weight M-step with an optional Dirichlet prior."""

from __future__ import annotations

import torch


def dirichlet_map_weights(
    responsibilities: torch.Tensor,
    concentration: float = 1.0,
) -> torch.Tensor:
    """Return the symmetric-Dirichlet MAP estimate of mixture weights.

    ``concentration=1`` deliberately follows the historical ML update,
    including its numerical floor, so the default protocol is unchanged.
    Concentrations above one add ``concentration - 1`` pseudo-counts to
    every component.
    """

    if responsibilities.ndim != 2 or responsibilities.shape[0] < 1:
        raise ValueError("responsibilities must have shape [items, components]")
    if concentration < 1.0:
        raise ValueError("Dirichlet concentration must be >= 1")

    if concentration == 1.0:
        weights = responsibilities.mean(0).clamp_min(1e-8)
        return weights / weights.sum()

    pseudo_count = responsibilities.new_tensor(concentration - 1.0)
    counts = responsibilities.sum(0)
    denominator = responsibilities.shape[0] + responsibilities.shape[1] * pseudo_count
    weights = (counts + pseudo_count) / denominator
    return weights / weights.sum()
