"""TPP likelihood with convex active-block Wishart intensities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from wishart_tpp.model.trace import TPPTrace
from wishart_tpp.model.wishart import matrix_square_root, sample_wishart


@dataclass(frozen=True)
class CachedTrace:
    start: int
    stop: int
    trace: TPPTrace


class ActiveBlockDecoder:
    """Transform and score only the candidate component's C-dimensional block.

    The production likelihood is defined in intensity space:

    ``lambda = (1-alpha) * h + alpha * (sqrt(W) @ sqrt(h))**2 + epsilon``.

    ``W`` never defines a mixture weight or a route.  Mixture membership is
    represented by the model's independent logits and is coupled to ``W`` only
    by the trajectory likelihood.
    """

    intensity_formula = "convex_intensity_v1"

    def __init__(
        self,
        intensity_floor: float = 1e-6,
        intensity_floor_scaling: str = "constant",
        prior_predictive_sample_shard_size: int = 64,
    ) -> None:
        if not math.isfinite(intensity_floor) or intensity_floor < 0.0:
            raise ValueError("intensity_floor must be finite and non-negative")
        if intensity_floor_scaling != "constant":
            raise ValueError(
                "the production active-block likelihood requires a constant "
                "event-and-compensator intensity floor"
            )
        self.intensity_floor = float(intensity_floor)
        self.intensity_floor_scaling = intensity_floor_scaling
        if prior_predictive_sample_shard_size < 1:
            raise ValueError("prior_predictive_sample_shard_size must be positive")
        self.prior_predictive_sample_shard_size = prior_predictive_sample_shard_size

    def rates(
        self,
        base_rates: Tensor,
        draws: Tensor,
        path_indices: Tensor,
        alpha: float | Tensor,
        roots: Tensor | None = None,
    ) -> Tensor:
        roots = matrix_square_root(draws) if roots is None else roots
        selected = roots.index_select(0, path_indices)
        base = base_rates.clamp_min(0.0)
        transformed = torch.einsum("rksij,rkj->rksi", selected, base.sqrt()).square()
        alpha_tensor = torch.as_tensor(alpha, device=base.device, dtype=base.dtype)
        rates = (
            (1.0 - alpha_tensor) * base[:, :, None, :]
            + alpha_tensor * transformed
            + self.intensity_floor
        )
        return rates.clamp_min(1e-12)

    def component_scores(
        self,
        trace: TPPTrace,
        draws: Tensor,
        alpha: float | Tensor,
        roots: Tensor | None = None,
    ) -> Tensor:
        """Return conditional scores with shape paths x K x samples."""

        scores = draws.new_zeros(trace.n_paths, trace.n_components, draws.shape[2])
        # Accumulate marked event log-intensities path by path.
        if trace.event_times.numel():
            event_rates = self.rates(
                trace.event_base_rates, draws, trace.event_paths, alpha, roots
            )
            marked = event_rates.gather(
                3,
                trace.event_marks[:, None, None, None].expand(
                    -1, trace.n_components, draws.shape[2], 1
                ),
            ).squeeze(3)
            scores.index_add_(0, trace.event_paths, marked.log())

        # Subtract the all-mark compensator at the configured integration points.
        integral_rates = self.rates(
            trace.integral_base_rates,
            draws,
            trace.integral_paths,
            alpha,
            roots,
        )
        compensator = trace.integral_weights[:, None, None] * integral_rates.sum(3)
        scores.index_add_(0, trace.integral_paths, -compensator)
        return scores

    def base_component_scores(self, trace: TPPTrace) -> Tensor:
        scores = trace.event_base_rates.new_zeros(trace.n_paths, trace.n_components)
        # Alpha=0 bypasses sampling and is the exact backbone-mixture endpoint
        # under the same constant event-and-compensator floor.
        base_floor = self.intensity_floor
        if trace.event_times.numel():
            event_rates = trace.event_base_rates + base_floor
            marked = event_rates.gather(
                2,
                trace.event_marks[:, None, None].expand(-1, trace.n_components, 1),
            ).squeeze(2)
            scores.index_add_(0, trace.event_paths, marked.clamp_min(1e-12).log())
        integral = trace.integral_weights[:, None] * (
            trace.integral_base_rates + base_floor
        ).sum(2)
        scores.index_add_(0, trace.integral_paths, -integral)
        return scores

    @torch.no_grad()
    def prior_predictive(
        self,
        cache: tuple[CachedTrace, ...],
        population_means: Tensor,
        population_df: float,
        samples: int,
        seed: int,
        alpha: float,
    ) -> Tensor:
        generator = torch.Generator(device=population_means.device).manual_seed(seed)
        rows = []
        for batch in cache:
            paths = batch.stop - batch.start
            means = population_means[None].expand(paths, -1, -1, -1)
            degrees = population_means.new_full(
                (paths, population_means.shape[0]), population_df
            )
            log_total = None
            remaining = samples

            # Keep the batched eigendecomposition well below cuSolver's 65,536
            # matrix boundary while preserving the same Monte Carlo estimator.
            while remaining > 0:
                chunk = min(self.prior_predictive_sample_shard_size, remaining)
                draws = sample_wishart(means, degrees, chunk, generator)
                conditional = self.component_scores(batch.trace, draws, alpha)
                chunk_total = torch.logsumexp(conditional, dim=2)
                log_total = (
                    chunk_total
                    if log_total is None
                    else torch.logaddexp(log_total, chunk_total)
                )
                remaining -= chunk

            assert log_total is not None
            rows.append(log_total - math.log(samples))
        return torch.cat(rows)
