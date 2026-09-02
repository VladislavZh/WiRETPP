"""Numerical rules for the compensator in a TPP likelihood."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class IntegrationPoints:
    """Elapsed times and weights for a batch of inter-event intervals."""

    elapsed: Tensor
    weights: Tensor


class IntegrationRule(ABC):
    """Construct points on intervals whose left endpoint is the last event."""

    def __init__(self, samples: int) -> None:
        if samples < 1:
            raise ValueError("integral_samples must be positive")
        self.samples = samples
        self._force_training_draws = 0

    @contextmanager
    def training_draws(self):
        """Request fresh per-forward points without enabling dropout."""

        self._force_training_draws += 1
        try:
            yield
        finally:
            self._force_training_draws -= 1

    def uses_training_draws(self, module_training: bool) -> bool:
        return module_training or self._force_training_draws > 0

    def reset_training_draws(self, seed: int | None = None) -> None:
        """Start a deterministic training-forward sequence when supported."""

        del seed

    @abstractmethod
    def points(self, lengths: Tensor, training: bool) -> IntegrationPoints:
        pass


class GaussLegendreRule(IntegrationRule):
    """Deterministic Gauss--Legendre quadrature on every interval."""

    def __init__(self, samples: int) -> None:
        super().__init__(samples)
        nodes, weights = np.polynomial.legendre.leggauss(samples)
        self._unit_nodes = torch.tensor(0.5 * (nodes + 1.0), dtype=torch.float32)
        self._unit_weights = torch.tensor(0.5 * weights, dtype=torch.float32)

    def points(self, lengths: Tensor, training: bool) -> IntegrationPoints:
        del training
        nodes = self._unit_nodes.to(device=lengths.device, dtype=lengths.dtype)
        weights = self._unit_weights.to(device=lengths.device, dtype=lengths.dtype)
        return IntegrationPoints(
            elapsed=lengths[:, None] * nodes[None],
            weights=lengths[:, None] * weights[None],
        )


class MonteCarloRule(IntegrationRule):
    """Paper COTIC uniform Monte Carlo compensator estimator.

    The upstream implementation draws one sorted vector of unit-interval
    locations for a physical batch and reuses those locations in every
    inter-event interval.  Scaling by each interval length is unbiased, while
    sharing the locations preserves the stochastic estimator used by COTIC.
    Evaluation calls are repeatable; training calls receive a fresh draw.
    """

    def __init__(self, samples: int, seed: int) -> None:
        super().__init__(samples)
        self.seed = seed
        self._training_seed = seed
        self._training_draw = 0

    def reset_training_draws(self, seed: int | None = None) -> None:
        self._training_seed = self.seed if seed is None else int(seed)
        self._training_draw = 0

    def _seed_for_call(self, training: bool) -> int:
        if not training:
            return self.seed
        seed = self._training_seed + self._training_draw
        self._training_draw += 1
        return seed

    def points(self, lengths: Tensor, training: bool) -> IntegrationPoints:
        generator = torch.Generator(device=lengths.device)
        generator.manual_seed(self._seed_for_call(self.uses_training_draws(training)))
        unit_points = (
            torch.rand(
                self.samples,
                generator=generator,
                device=lengths.device,
                dtype=lengths.dtype,
            )
            .sort()
            .values
        )
        return IntegrationPoints(
            elapsed=lengths[:, None] * unit_points[None],
            weights=lengths[:, None].expand(-1, self.samples) / self.samples,
        )


def create_integration_rule(method: str, samples: int, seed: int) -> IntegrationRule:
    if method == "gauss_legendre":
        return GaussLegendreRule(samples)
    if method == "monte_carlo":
        return MonteCarloRule(samples, seed)
    raise ValueError(f"unknown integral method: {method}")
