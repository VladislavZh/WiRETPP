"""Validation and test metrics with no checkpoint-selection side effects."""

from __future__ import annotations
import gc
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from active_wishart_tpp.data import DatasetPartition
from active_wishart_tpp.metrics import clustering_summary
from active_wishart_tpp.model.active_block import ActiveBlockDecoder
from active_wishart_tpp.training.cache import TraceCacheBuilder


@dataclass(frozen=True)
class Evaluation:
    component_scores: Tensor
    marginal_scores: Tensor
    probabilities: Tensor
    nll_per_exposure: float
    nll_per_event: float
    purity: float
    ari: float

    def as_dict(self) -> dict[str, float]:
        return {
            "nll_per_exposure": self.nll_per_exposure,
            "nll_per_event": self.nll_per_event,
            "purity": self.purity,
            "ari": self.ari,
        }


class ModelEvaluator:
    def __init__(
        self,
        decoder: ActiveBlockDecoder,
        cache_batch_size: int,
        path_shard_size: int | None = None,
    ) -> None:
        self.decoder = decoder
        self.cache_builder = TraceCacheBuilder(cache_batch_size)
        self.path_shard_size = path_shard_size

    def _shards(self, partition: DatasetPartition):
        size = self.path_shard_size or len(partition.sequences)
        for start in range(0, len(partition.sequences), size):
            stop = min(start + size, len(partition.sequences))
            indices = torch.arange(start, stop).numpy()
            yield partition.select(indices)

    def _summarize(
        self, component_scores: Tensor, log_weights: Tensor, partition: DatasetPartition
    ) -> Evaluation:
        log_weights = log_weights.to(component_scores.device)
        logits = component_scores + log_weights[None]
        marginal = torch.logsumexp(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        cluster = clustering_summary(
            partition.labels, probabilities.detach().cpu().numpy()
        )
        return Evaluation(
            component_scores=component_scores,
            marginal_scores=marginal,
            probabilities=probabilities,
            nll_per_exposure=float(-marginal.sum().cpu() / partition.exposure),
            nll_per_event=float(-marginal.sum().cpu() / max(1, partition.event_count)),
            purity=cluster["purity"],
            ari=cluster["ari"],
        )

    @torch.no_grad()
    def pure(self, model, partition: DatasetPartition) -> Evaluation:
        scores = []
        for shard in self._shards(partition):
            cache = self.cache_builder.build(model, shard)
            shard_scores = torch.cat(
                [self.decoder.base_component_scores(batch.trace) for batch in cache]
            )
            scores.append(shard_scores.cpu())
            del cache, shard_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores = torch.cat(scores)
        return self._summarize(scores, model.mixture_log_weights(), partition)

    @torch.no_grad()
    def active(
        self,
        model,
        partition: DatasetPartition,
        population_means: Tensor,
        log_weights: Tensor,
        *,
        population_df: float,
        alpha: float,
        samples: int,
        seed: int,
        repeats: int = 1,
    ) -> Evaluation:
        shard_scores = []
        offset = 0
        for shard in self._shards(partition):
            cache = self.cache_builder.build(model, shard)
            repeated_scores = [
                self.decoder.prior_predictive(
                    cache,
                    population_means,
                    population_df,
                    samples,
                    seed + repeat * 1000003 + offset * 10007,
                    alpha,
                )
                for repeat in range(repeats)
            ]
            scores = torch.logsumexp(torch.stack(repeated_scores), dim=0)
            shard_scores.append((scores - math.log(repeats)).cpu())
            offset += len(shard.sequences)
            del cache, repeated_scores, scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores = torch.cat(shard_scores)
        return self._summarize(scores, log_weights, partition)
