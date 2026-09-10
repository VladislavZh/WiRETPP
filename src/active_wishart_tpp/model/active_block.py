"""TPP likelihood with convex active-block Wishart intensities."""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from active_wishart_tpp.model.trace import TPPTrace
from active_wishart_tpp.model.wishart import matrix_square_root, sample_wishart


def _segment_sum(values: Tensor, paths: Tensor, path_count: int) -> Tensor:
    """Sum path-sorted rows without scatter-based backward operations."""
    lengths = torch.bincount(paths, minlength=path_count)
    return torch.segment_reduce(values, "sum", lengths=lengths, initial=0.0)


class _RepeatPathRows(torch.autograd.Function):
    """Repeat path rows forward and reduce their gradients in path order."""

    @staticmethod
    def forward(context, values: Tensor, paths: Tensor) -> Tensor:
        context.save_for_backward(paths)
        context.path_count = values.shape[0]
        return values.index_select(0, paths)

    @staticmethod
    def backward(context, gradient: Tensor) -> tuple[Tensor, None]:
        (paths,) = context.saved_tensors
        return (_segment_sum(gradient, paths, context.path_count), None)


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
    maximum_intermediate_elements = 1 << 26

    def __init__(
        self,
        intensity_floor: float = 1e-06,
        intensity_floor_scaling: str = "constant",
        prior_predictive_sample_shard_size: int = 64,
    ) -> None:
        if not math.isfinite(intensity_floor) or intensity_floor < 0.0:
            raise ValueError("intensity_floor must be finite and non-negative")
        if intensity_floor_scaling != "constant":
            raise ValueError(
                "the production active-block likelihood requires a constant event-and-compensator intensity floor"
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
        selected = _RepeatPathRows.apply(roots, path_indices)
        base = base_rates.clamp_min(0.0)
        transformed = torch.einsum("rksij,rkj->rksi", selected, base.sqrt()).square()
        alpha_tensor = torch.as_tensor(alpha, device=base.device, dtype=base.dtype)
        rates = (
            (1.0 - alpha_tensor) * base[:, :, None, :]
            + alpha_tensor * transformed
            + self.intensity_floor
        )
        return rates.clamp_min(1e-12)

    def _row_shard_size(self, roots: Tensor, *, alpha_count: int = 1) -> int:
        """Bound selected-root and alpha-grid intermediates by element count."""
        root_elements = roots[0].numel()
        score_elements = roots.shape[1] * roots.shape[2] * alpha_count
        return max(
            1,
            self.maximum_intermediate_elements
            // max(root_elements + score_elements, 1),
        )

    @staticmethod
    def _marked_rates(rates: Tensor, marks: Tensor) -> Tensor:
        """Select observed marks without scatter-based gather gradients."""
        indicators = torch.nn.functional.one_hot(marks, num_classes=rates.shape[-1]).to(
            rates.dtype
        )
        while indicators.ndim < rates.ndim:
            indicators = indicators.unsqueeze(1)
        return (rates * indicators).sum(-1)

    def _integrated_rates(
        self, trace: TPPTrace, roots: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Contract sampled matrices with path-integrated sqrt(h) outer products."""
        paths, components, marks = (trace.n_paths, trace.n_components, trace.n_marks)
        moment_size = components * marks * marks
        width = moment_size + components + 1
        integrated = roots.new_zeros(paths, width)
        row_size = max(1, self.maximum_intermediate_elements // (2 * width))
        for start in range(0, len(trace.integral_paths), row_size):
            stop = min(start + row_size, len(trace.integral_paths))
            base = trace.integral_base_rates[start:stop].clamp_min(0.0)
            amplitudes = base.sqrt()
            weights = trace.integral_weights[start:stop]
            indices = trace.integral_paths[start:stop]
            outer = amplitudes[..., :, None] * amplitudes[..., None, :]
            packed = torch.cat(
                (
                    (weights[:, None, None, None] * outer).flatten(1),
                    weights[:, None] * base.sum(-1),
                    (weights * marks * self.intensity_floor)[:, None],
                ),
                dim=1,
            )
            integrated = integrated + _segment_sum(packed, indices, paths)
        moment = integrated[:, :moment_size].reshape(paths, components, marks, marks)
        direct = integrated[:, moment_size : moment_size + components]
        floor = integrated[:, -1]
        gram = roots.transpose(-1, -2) @ roots
        transformed = torch.einsum("nksij,nkij->nks", gram, moment)
        return (direct, transformed, floor)

    def _clamped_compensator(self, trace, draws, roots, alpha):
        """Preserve per-mark numerical clamping for a sub-1e-12 intensity floor."""
        total = draws.new_zeros(trace.n_paths, trace.n_components, draws.shape[2])
        size = self._row_shard_size(roots)
        for start in range(0, len(trace.integral_paths), size):
            stop = min(start + size, len(trace.integral_paths))
            rates = self.rates(
                trace.integral_base_rates[start:stop],
                draws,
                trace.integral_paths[start:stop],
                alpha,
                roots,
            )
            total = total + _segment_sum(
                trace.integral_weights[start:stop, None, None] * rates.sum(3),
                trace.integral_paths[start:stop],
                trace.n_paths,
            )
        return total

    def component_scores(
        self,
        trace: TPPTrace,
        draws: Tensor,
        alpha: float | Tensor,
        roots: Tensor | None = None,
    ) -> Tensor:
        """Return conditional scores with shape paths x K x samples."""
        roots = matrix_square_root(draws) if roots is None else roots
        scores = draws.new_zeros(trace.n_paths, trace.n_components, draws.shape[2])
        shard_size = self._row_shard_size(roots)
        for start in range(0, len(trace.event_paths), shard_size):
            stop = min(start + shard_size, len(trace.event_paths))
            event_rates = self.rates(
                trace.event_base_rates[start:stop],
                draws,
                trace.event_paths[start:stop],
                alpha,
                roots,
            )
            marked = self._marked_rates(event_rates, trace.event_marks[start:stop])
            scores = scores + _segment_sum(
                marked.log(), trace.event_paths[start:stop], trace.n_paths
            )
        if self.intensity_floor < 1e-12:
            return scores - self._clamped_compensator(trace, draws, roots, alpha)
        direct, transformed, floor = self._integrated_rates(trace, roots)
        alpha_tensor = torch.as_tensor(alpha, device=draws.device, dtype=draws.dtype)
        return (
            scores
            - (1.0 - alpha_tensor) * direct[:, :, None]
            - alpha_tensor * transformed
            - floor[:, None, None]
        )

    def component_score_grid(
        self,
        trace: TPPTrace,
        draws: Tensor,
        alpha_values: Tensor,
        roots: Tensor | None = None,
    ) -> Tensor:
        """Average component scores over draws for every supplied alpha value."""
        roots = matrix_square_root(draws) if roots is None else roots
        values = alpha_values.to(device=draws.device, dtype=draws.dtype).reshape(
            1, 1, 1, -1
        )
        paths = trace.n_paths
        components = trace.n_components
        samples = draws.shape[2]
        score = draws.new_zeros(paths, components, samples, values.shape[-1])
        shard_size = self._row_shard_size(roots, alpha_count=values.shape[-1])
        for start in range(0, len(trace.event_paths), shard_size):
            stop = min(start + shard_size, len(trace.event_paths))
            base = trace.event_base_rates[start:stop].clamp_min(0.0)
            selected_paths = trace.event_paths[start:stop]
            transformed = torch.einsum(
                "rksij,rkj->rksi",
                _RepeatPathRows.apply(roots, selected_paths),
                base.sqrt(),
            ).square()
            marks = trace.event_marks[start:stop]
            base_marked = self._marked_rates(base, marks)
            transformed_marked = self._marked_rates(transformed, marks)
            marked = (
                (1.0 - values) * base_marked[:, :, None, None]
                + values * transformed_marked[:, :, :, None]
                + self.intensity_floor
            ).clamp_min(1e-12)
            score = score + _segment_sum(marked.log(), selected_paths, paths)
        base_integral, transformed_integral, floor_integral = self._integrated_rates(
            trace, roots
        )
        score = score - (
            (1.0 - values) * base_integral[:, :, None, None]
            + values * transformed_integral[:, :, :, None]
            + floor_integral[:, None, None, None]
        )
        return score.mean(2)

    def base_component_scores(self, trace: TPPTrace) -> Tensor:
        scores = trace.event_base_rates.new_zeros(trace.n_paths, trace.n_components)
        base_floor = self.intensity_floor
        if trace.event_times.numel():
            event_rates = trace.event_base_rates + base_floor
            marked = event_rates.gather(
                2, trace.event_marks[:, None, None].expand(-1, trace.n_components, 1)
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
