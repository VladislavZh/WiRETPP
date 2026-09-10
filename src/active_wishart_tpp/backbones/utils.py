"""Shared head-expansion helpers for backbone adapters."""

from __future__ import annotations
import torch
from torch import Tensor


def repeat_with_simplex_noise(
    values: Tensor, repeats: int, noise: float, generator: torch.Generator
) -> Tensor:
    """Repeat one head along a centered regular simplex at relative RMS scale."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if noise < 0.0:
        raise ValueError("noise must be non-negative")
    base = values.detach()
    copies = base.unsqueeze(0).repeat((repeats,) + (1,) * base.ndim)
    if repeats == 1 or noise == 0.0:
        return copies.flatten(0, 1)
    dimensions = base.numel()
    if dimensions < repeats - 1:
        perturbations = torch.randn(
            repeats,
            dimensions,
            device=base.device,
            dtype=base.dtype,
            generator=generator,
        )
        perturbations -= perturbations.mean(0, keepdim=True)
        perturbations *= (
            noise
            * base.square().mean().sqrt()
            / perturbations.square().mean().sqrt().clamp_min(1e-12)
        )
        return (copies + perturbations.reshape_as(copies)).flatten(0, 1)
    random = torch.randn(
        dimensions,
        repeats - 1,
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    )
    basis = torch.linalg.qr(random, mode="reduced").Q
    simplex = base.new_zeros((repeats, repeats - 1))
    for column in range(repeats - 1):
        scale = ((column + 1) * (column + 2)) ** (-0.5)
        simplex[: column + 1, column] = scale
        simplex[column + 1, column] = -(column + 1) * scale
    simplex *= (repeats / (repeats - 1)) ** 0.5
    directions = simplex @ basis.T
    relative_scale = noise * base.square().mean().sqrt() * dimensions**0.5
    perturbations = (directions * relative_scale).reshape_as(copies)
    return (copies + perturbations).flatten(0, 1)
