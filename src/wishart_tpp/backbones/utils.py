"""Shared head-expansion helpers for backbone adapters."""

from __future__ import annotations

import torch
from torch import Tensor


def repeat_with_noise(
    values: Tensor,
    repeats: int,
    n_marks: int,
    noise: float,
    generator: torch.Generator,
) -> Tensor:
    copies = values.detach().repeat((repeats,) + (1,) * (values.ndim - 1))
    if noise:
        perturbation = torch.randn(
            copies.shape,
            device=copies.device,
            dtype=copies.dtype,
            generator=generator,
        )
        perturbation[:n_marks].zero_()
        copies = copies + noise * perturbation
    return copies
