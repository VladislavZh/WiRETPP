"""Disposable, regrouped local inference over deterministic EM blocks."""

import numpy as np
import torch

from active_wishart_tpp.inference.local import LocalPosterior
from active_wishart_tpp.inference.regrouped_local import RegroupedLocalInference
from active_wishart_tpp.training.cache import TraceCacheBuilder


def em_block_indices(total, block_size, cycle, seed):
    """Partition an epoch-wise permutation into near-equal non-overlapping blocks."""
    count = (total + block_size - 1) // block_size
    epoch, block = divmod(cycle - 1, count)
    permutation = np.random.default_rng(seed + 500_009 + epoch * 1_000_003).permutation(
        total
    )
    base, remainder = divmod(total, count)
    start = block * base + min(block, remainder)
    stop = start + base + (1 if block < remainder else 0)
    return permutation[start:stop], epoch + 1, block + 1, count


class ExpectationStep:
    """Fit fresh local posteriors with cached traces and native chunk-wise random draws."""

    def __init__(self, fabric, decoder, config):
        self.config = config
        self.cache = TraceCacheBuilder(config.compute.trace_batch)
        self.inference = RegroupedLocalInference(
            fabric, decoder, fit_batch_size=config.compute.e_fit_batch
        )

    def fit(self, model, train, omega, alpha, cycle, previous=None):
        t, c = self.config.training, self.config.compute
        outputs = []
        for start in range(0, len(train.sequences), c.path_shard):
            stop = min(start + c.path_shard, len(train.sequences))
            shard = train.select(np.arange(start, stop, dtype=np.int64))
            cache = self.cache.build(model, shard, resample_integration=True)
            local = self.inference.fit(
                cache,
                omega.to(model.device),
                t.population_df,
                alpha=alpha,
                steps=t.local_steps,
                samples=t.e_fit_samples,
                evaluation_samples=t.e_score_samples,
                seed=self.config.runtime.seed + cycle * 10_007 + start * 1_000_003,
                initial_means=None
                if previous is None
                else previous[0][start:stop].to(model.device),
                initial_df=None
                if previous is None
                else previous[1][start:stop].to(model.device),
            )
            outputs.append(
                LocalPosterior(
                    local.means.cpu(),
                    local.degrees_of_freedom.cpu(),
                    local.expected_negative_log_likelihood.cpu(),
                    local.kl_to_prior.cpu(),
                    local.steps_taken,
                )
            )
        return LocalPosterior(
            *(
                torch.cat([getattr(item, key) for item in outputs])
                for key in (
                    "means",
                    "degrees_of_freedom",
                    "expected_negative_log_likelihood",
                    "kl_to_prior",
                )
            ),
            steps_taken=t.local_steps,
        )
