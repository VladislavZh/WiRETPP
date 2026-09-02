"""Categorical variational posterior over mixture components."""

from __future__ import annotations

import torch
from torch import Tensor


class ResponsibilityUpdater:
    def __init__(self, balance_iterations: int = 2_000) -> None:
        self.balance_iterations = balance_iterations

    def update(
        self, free_energy: Tensor, log_weights: Tensor, balanced: bool
    ) -> Tensor:
        logits = log_weights[None] - free_energy
        if not balanced:
            return torch.softmax(logits, dim=1)
        target = free_energy.new_full(
            (free_energy.shape[1],), free_energy.shape[0] / free_energy.shape[1]
        )
        # Balanced warm-up is a rectangular Sinkhorn projection.  Performing
        # it after a probability-space softmax is numerically wrong for
        # large-NLL real datasets: losing components can underflow to exact
        # zero and multiplicative rescaling can never revive them.  Keep all
        # iterates in log space, alternating the target column masses and
        # unit row masses, and exponentiate only at the end.
        log_responsibilities = logits
        log_target = target.log()
        for _ in range(self.balance_iterations):
            log_responsibilities = (
                log_responsibilities
                + (log_target - torch.logsumexp(log_responsibilities, dim=0))[None]
            )
            log_responsibilities = log_responsibilities - torch.logsumexp(
                log_responsibilities, dim=1, keepdim=True
            )
        return log_responsibilities.exp()
