"""Grouped E-step over unchanged frozen neural traces and sampling chunks."""

from dataclasses import fields
import torch
from active_wishart_tpp.inference.fast_local import CachedLocalInference
from active_wishart_tpp.inference.fixed_prior_kl import FixedPriorKL
from active_wishart_tpp.inference.frozen_likelihood import FrozenDecoder
from active_wishart_tpp.inference.local import LocalWishartInference
from active_wishart_tpp.model.active_block import CachedTrace
from active_wishart_tpp.model.trace import TPPTrace
from active_wishart_tpp.model.wishart import sample_wishart


def join_traces(batches):
    """Concatenate consecutive frozen traces without resampling or reordering rows."""
    if len(batches) == 1:
        return batches[0]
    first, last = (batches[0], batches[-1])
    values = dict(
        n_paths=last.stop - first.start,
        n_components=first.trace.n_components,
        n_marks=first.trace.n_marks,
    )
    for field in fields(TPPTrace):
        if field.name in values:
            continue
        rows = []
        for batch in batches:
            item = getattr(batch.trace, field.name)
            if field.name in ("event_paths", "integral_paths"):
                item = item + batch.start - first.start
            rows.append(item)
        values[field.name] = torch.cat(rows, dim=0)
    return CachedTrace(first.start, last.stop, TPPTrace(**values))


def regroup_cache(cache, maximum_paths):
    """Pack whole contiguous neural batches into bounded local-inference batches."""
    if not cache or maximum_paths < 1:
        raise ValueError("A nonempty cache and positive batch size are required")
    result = []
    pending = []
    previous = 0
    for batch in cache:
        if (
            batch.start != previous
            or batch.stop - batch.start != batch.trace.n_paths
            or batch.stop <= batch.start
        ):
            raise ValueError(
                "Neural cache must have contiguous nonempty path intervals"
            )
        if (batch.trace.n_components, batch.trace.n_marks) != (
            cache[0].trace.n_components,
            cache[0].trace.n_marks,
        ):
            raise ValueError("Incompatible neural traces")
        if batch.stop - batch.start > maximum_paths:
            raise ValueError("Local regrouping cannot split an original neural batch")
        if pending and batch.stop - pending[0].start > maximum_paths:
            result.append(join_traces(pending))
            pending = []
        pending.append(batch)
        previous = batch.stop
    if pending:
        result.append(join_traces(pending))
    return tuple(result)


def legacy_chunk_draws(means, degrees, samples, generator, bounds):
    """Keep each original gamma/Gaussian call shape and order before concatenation."""
    draws = [
        sample_wishart(means[start:stop], degrees[start:stop], samples, generator)
        for start, stop in bounds
    ]
    return draws[0] if len(draws) == 1 else torch.cat(draws, dim=0)


class RegroupedLocalInference(CachedLocalInference):
    """Vectorize local likelihood/KL/backward while retaining neural and RNG grouping."""

    def __init__(self, *args, fit_batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_batch_size = fit_batch_size

    def fit(self, cache, population_means, population_df, **kwargs):
        """Own grouped statistics only until this local fit finishes or fails."""
        if hasattr(self, "fixed_prior"):
            raise RuntimeError("Local inference fits must not overlap")
        grouped = regroup_cache(cache, self.fit_batch_size)
        self.original_cache = cache
        self.sample_bounds = {
            batch.start: tuple(
                (
                    (old.start - batch.start, old.stop - batch.start)
                    for old in cache
                    if batch.start <= old.start < batch.stop
                )
            )
            for batch in grouped
        }
        decoder = self.decoder
        try:
            self.fixed_prior = FixedPriorKL(population_means, population_df)
            unique = {id(batch.trace): batch for batch in (*cache, *grouped)}
            self.decoder = FrozenDecoder(decoder, tuple(unique.values()))
            return LocalWishartInference.fit(
                self, grouped, population_means, population_df, **kwargs
            )
        finally:
            self.decoder = decoder
            for attribute in ("fixed_prior", "original_cache", "sample_bounds"):
                if hasattr(self, attribute):
                    delattr(self, attribute)

    def _optimization_step(
        self,
        cache,
        priors,
        parameters,
        optimizer,
        *,
        population_df,
        alpha,
        samples,
        generator,
    ):
        """Accumulate the same shard-normalized objective using fewer backward calls."""
        optimizer.zero_grad(set_to_none=True)
        normalizer = priors.shape[0] * priors.shape[1]
        for batch in cache:
            means, degrees = parameters(slice(batch.start, batch.stop))
            draws = legacy_chunk_draws(
                means, degrees, samples, generator, self.sample_bounds[batch.start]
            )
            score = self.decoder.component_scores(batch.trace, draws, alpha)
            loss = (
                -score.mean(2) + self.fixed_prior(means, degrees)
            ).sum() / normalizer
            self.fabric.backward(loss)
        repaired = self._repair_nonfinite_gradients(parameters)
        if repaired:
            self.fabric.print(
                f"[local-numeric] repaired_nonfinite_gradients={repaired}", flush=True
            )
        self.fabric.clip_gradients(parameters, optimizer, max_norm=self.gradient_clip)
        optimizer.step()

    @torch.no_grad()
    def _evaluate(self, cache, priors, parameters, **kwargs):
        """Retain the original independent scoring draws and neural-trace grouping."""
        return super()._evaluate(self.original_cache, priors, parameters, **kwargs)
