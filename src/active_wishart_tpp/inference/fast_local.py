"""Cached E-step with the original Adam, samples, schedule and full KL objective."""

import torch
from active_wishart_tpp.inference.fixed_prior_kl import FixedPriorKL
from active_wishart_tpp.inference.frozen_likelihood import FrozenDecoder
from active_wishart_tpp.inference.local import LocalPosterior, LocalWishartInference
from active_wishart_tpp.model.wishart import sample_wishart


class CachedLocalInference(LocalWishartInference):
    """Reuse immutable trace/prior statistics during one local variational fit."""

    def fit(self, cache, population_means, population_df, **kwargs):
        """Build disposable caches and clear them even when inference raises."""
        if hasattr(self, "fixed_prior"):
            raise RuntimeError("Local inference fits must not overlap")
        decoder = self.decoder
        try:
            self.fixed_prior = FixedPriorKL(population_means, population_df)
            self.decoder = FrozenDecoder(decoder, cache)
            return super().fit(cache, population_means, population_df, **kwargs)
        finally:
            self.decoder = decoder
            if hasattr(self, "fixed_prior"):
                del self.fixed_prior

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
        """Accumulate the unchanged local objective before one clipped Adam update."""
        optimizer.zero_grad(set_to_none=True)
        normalizer = priors.shape[0] * priors.shape[1]
        for batch in cache:
            means, degrees = parameters(slice(batch.start, batch.stop))
            draws = sample_wishart(means, degrees, samples, generator)
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
    def _evaluate(
        self, cache, priors, parameters, *, population_df, alpha, samples, seed
    ):
        """Re-estimate free energies with the original independent evaluation draws."""
        means, degrees = parameters()
        expected_nll = means.new_empty(means.shape[:2])
        generator = torch.Generator(device=means.device).manual_seed(seed)
        for batch in cache:
            selection = slice(batch.start, batch.stop)
            draws = sample_wishart(
                means[selection], degrees[selection], samples, generator
            )
            expected_nll[selection] = -self.decoder.component_scores(
                batch.trace, draws, alpha
            ).mean(2)
        kl = self.fixed_prior(means, degrees)
        return LocalPosterior(
            means.detach(), degrees.detach(), expected_nll.detach(), kl.detach()
        )
