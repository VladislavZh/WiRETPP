"""Bounded scalar M-step for the random-effect strength alpha."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from wishart_tpp.inference.local import LocalPosterior
from wishart_tpp.model.active_block import ActiveBlockDecoder, CachedTrace
from wishart_tpp.model.wishart import matrix_square_root, sample_wishart


@dataclass(frozen=True)
class AlphaGridProfile:
    """Per-path component scores for a shared bounded alpha grid."""

    values: Tensor
    scores: Tensor


class AlphaMstep:
    def __init__(self, decoder: ActiveBlockDecoder) -> None:
        self.decoder = decoder

    @staticmethod
    def candidate_grid(initial: float, iterations: int) -> Tensor:
        """Return a deterministic bounded grid including the current value."""

        points = max(3, iterations + 2)
        base = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
        current = torch.tensor([initial], dtype=torch.float64)
        return torch.unique(torch.cat((base, current)), sorted=True)

    @torch.no_grad()
    def profile(
        self,
        cache: tuple[CachedTrace, ...],
        posterior: LocalPosterior,
        values: Tensor,
        *,
        samples: int,
        seed: int,
    ) -> AlphaGridProfile:
        """Score every path/component on one grid using common random draws.

        The returned tensor is small and lives on CPU, so profiles from all
        physical shards can be concatenated before responsibilities are known.
        """

        if not cache:
            raise ValueError("alpha profile requires a non-empty trace cache")
        device = cache[0].trace.event_base_rates.device
        generator = torch.Generator(device=device).manual_seed(seed)
        rows = []
        for batch in cache:
            selection = slice(batch.start, batch.stop)
            draws = sample_wishart(
                posterior.means[selection].to(device),
                posterior.degrees_of_freedom[selection].to(device),
                samples,
                generator,
            ).detach()
            roots = matrix_square_root(draws)
            rows.append(
                torch.stack(
                    [
                        self.decoder.component_scores(
                            batch.trace, draws, float(value), roots
                        )
                        .mean(2)
                        .cpu()
                        for value in values
                    ],
                    dim=2,
                )
            )
        return AlphaGridProfile(values.cpu(), torch.cat(rows))

    @staticmethod
    def select(profile: AlphaGridProfile, responsibilities: Tensor) -> float:
        """Select the grid maximizer after all statistical shards are joined."""

        gamma = responsibilities.detach().cpu()
        if profile.scores.shape[:2] != gamma.shape:
            raise ValueError("alpha profile and responsibilities must align")
        objective = (gamma[:, :, None] * profile.scores).sum((0, 1))
        return float(profile.values[int(objective.argmax())])

    def _fixed_draws(
        self,
        cache: tuple[CachedTrace, ...],
        posterior: LocalPosterior,
        samples: int,
        seed: int,
    ) -> tuple[list[tuple[CachedTrace, Tensor, Tensor]], int]:
        device = cache[0].trace.event_base_rates.device
        generator = torch.Generator(device=device).manual_seed(seed)
        fixed = []
        total_events = 0
        for batch in cache:
            selection = slice(batch.start, batch.stop)
            draws = sample_wishart(
                posterior.means[selection].to(device),
                posterior.degrees_of_freedom[selection].to(device),
                samples,
                generator,
            ).detach()
            fixed.append((batch, draws, matrix_square_root(draws)))
            total_events += len(batch.trace.event_times)
        return fixed, total_events

    def _objective(
        self,
        value: float,
        fixed: list[tuple[CachedTrace, Tensor, Tensor]],
        responsibilities: Tensor,
        total_events: int,
    ) -> float:
        expected = fixed[0][1].new_zeros(())
        for batch, draws, roots in fixed:
            selection = slice(batch.start, batch.stop)
            score = self.decoder.component_scores(
                batch.trace, draws, value, roots
            ).mean(2)
            expected = (
                expected + (responsibilities[selection].to(score.device) * score).sum()
            )
        return float(-expected.cpu() / max(1, total_events))

    @staticmethod
    def _golden_search(
        objective: Callable[[float], float], initial: float, iterations: int
    ) -> float:
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        lower, upper = 0.0, 1.0
        left, right = upper - ratio, lower + ratio
        evaluated = {value: objective(value) for value in (0.0, 1.0, initial)}

        def evaluate(value: float) -> float:
            key = round(min(1.0, max(0.0, value)), 14)
            if key not in evaluated:
                evaluated[key] = objective(key)
            return evaluated[key]

        left_loss, right_loss = evaluate(left), evaluate(right)
        for _ in range(iterations):
            if left_loss <= right_loss:
                upper, right, right_loss = right, left, left_loss
                left = upper - ratio * (upper - lower)
                left_loss = evaluate(left)
            else:
                lower, left, left_loss = left, right, right_loss
                right = lower + ratio * (upper - lower)
                right_loss = evaluate(right)
        return min(evaluated.items(), key=lambda item: item[1])[0]

    @torch.no_grad()
    def update(
        self,
        cache: tuple[CachedTrace, ...],
        posterior: LocalPosterior,
        responsibilities: Tensor,
        *,
        initial: float,
        iterations: int,
        samples: int,
        seed: int,
    ) -> float:
        # Common random numbers make the one-dimensional comparison deterministic.
        fixed, total_events = self._fixed_draws(cache, posterior, samples, seed)
        objective = lambda value: self._objective(
            value, fixed, responsibilities, total_events
        )
        return self._golden_search(objective, initial, iterations)
