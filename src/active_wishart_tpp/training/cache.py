"""Frozen neural traces used by variational and predictive Monte Carlo."""

from __future__ import annotations
from contextlib import nullcontext
import torch
from active_wishart_tpp.data import DatasetPartition
from active_wishart_tpp.model.active_block import CachedTrace


class TraceCacheBuilder:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @torch.no_grad()
    def build(
        self, model, partition: DatasetPartition, *, resample_integration: bool = False
    ) -> tuple[CachedTrace, ...]:
        model.eval()
        batches = []
        bank = getattr(model, "module", model)
        rule = bank.integration_rule
        context = rule.training_draws() if resample_integration else nullcontext()
        with context:
            for start in range(0, len(partition.sequences), self.batch_size):
                stop = min(start + self.batch_size, len(partition.sequences))
                batches.append(
                    CachedTrace(start, stop, model(partition.sequences[start:stop]))
                )
        return tuple(batches)
