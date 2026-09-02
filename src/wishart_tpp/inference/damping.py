"""Damped coordinate updates for the alternating Wishart schedule."""

from __future__ import annotations

import torch
from torch import Tensor


def damp_population(current: Tensor, optimum: Tensor, rate: float) -> Tensor:
    """Apply one arithmetic EMA step to trace-normalized SPD means."""
    if rate == 0.0:
        return current
    if rate == 1.0:
        return optimum
    updated = torch.lerp(current, optimum, rate)
    return 0.5 * (updated + updated.transpose(-1, -2))


def damp_probability(current: float, optimum: float, rate: float) -> float:
    """Apply one arithmetic EMA step to a scalar in the probability interval."""
    if rate == 0.0:
        return current
    if rate == 1.0:
        return optimum
    return (1.0 - rate) * current + rate * optimum


def damp_simplex(current: Tensor, optimum: Tensor, rate: float) -> Tensor:
    """Apply an arithmetic EMA update to categorical probabilities.

    Online EM must smooth the mixture-weight sufficient statistics just like
    the population matrices and alpha.  Interpolating probabilities (rather
    than logits) preserves the simplex and leaves the historical exact update
    unchanged at ``rate=1``.
    """
    if rate == 0.0:
        return current
    if rate == 1.0:
        return optimum
    updated = torch.lerp(current, optimum, rate).clamp_min(0.0)
    return updated / updated.sum().clamp_min(torch.finfo(updated.dtype).tiny)
