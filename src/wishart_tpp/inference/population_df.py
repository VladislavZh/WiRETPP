"""Profile M-step for the global population-Wishart degrees of freedom."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from wishart_tpp.inference.local import LocalPosterior
from wishart_tpp.model.wishart import multivariate_digamma, wishart_kl


@dataclass(frozen=True)
class PopulationDfProfile:
    """Exact train-ELBO profile after holding the local posterior fixed."""

    values: Tensor
    weighted_kl: Tensor


class PopulationDfMstep:
    """Generalized-EM update for one shared continuous Wishart df."""

    @staticmethod
    def candidate_grid(
        current: float,
        dimension: int,
        *,
        points: int,
        minimum: float | None,
        maximum: float,
    ) -> Tensor:
        if points < 3:
            raise ValueError("population-df profile requires at least 3 points")
        lower = float(dimension if minimum is None else minimum)
        lower = max(lower, dimension - 1.0 + 1e-3)
        if not lower <= current <= maximum:
            raise ValueError("current population df lies outside its profile bounds")
        values = torch.logspace(
            torch.tensor(lower, dtype=torch.float64).log10(),
            torch.tensor(maximum, dtype=torch.float64).log10(),
            points,
            dtype=torch.float64,
        )
        return torch.unique(
            torch.cat((values, torch.tensor([current], dtype=torch.float64))),
            sorted=True,
        )

    @staticmethod
    def _candidate_dependent_kl(
        posterior: LocalPosterior,
        population_means: Tensor,
        values: Tensor,
    ) -> Tensor:
        """Return the candidate-dependent part of KL(q||p_nu).

        Terms depending only on q are omitted because they are identical for
        every candidate and therefore cannot change the profile minimizer.
        """

        means = posterior.means
        degrees = posterior.degrees_of_freedom
        device = means.device
        dtype = means.dtype
        population = population_means.to(device=device, dtype=dtype)
        dimension = means.shape[-1]
        q_logdet = torch.linalg.slogdet(means).logabsdet
        expected_logdet = (
            multivariate_digamma(0.5 * degrees, dimension)
            + dimension * means.new_tensor(2.0).log()
            + q_logdet
            - dimension * degrees.log()
        )
        trace = torch.einsum(
            "...ii->...",
            torch.linalg.solve(population[None], means),
        )
        population_logdet = torch.linalg.slogdet(population).logabsdet[None]
        candidates = values.to(device=device, dtype=dtype)
        expanded = candidates[:, None, None]
        bracket = (
            trace[None]
            - expected_logdet[None]
            + dimension * means.new_tensor(2.0).log()
            + population_logdet[None]
            - dimension * expanded.log()
        )
        return 0.5 * expanded * bracket + torch.special.multigammaln(
            0.5 * expanded, dimension
        )

    @torch.no_grad()
    def profile(
        self,
        posterior: LocalPosterior,
        population_means: Tensor,
        responsibilities: Tensor,
        values: Tensor,
    ) -> PopulationDfProfile:
        if posterior.means.shape[:2] != responsibilities.shape:
            raise ValueError("population-df posterior and responsibilities must align")
        terms = self._candidate_dependent_kl(posterior, population_means, values)
        gamma = responsibilities.to(device=terms.device, dtype=terms.dtype)
        objective = (terms * gamma[None]).sum((1, 2)) / gamma.sum().clamp_min(1.0)
        return PopulationDfProfile(values.detach().cpu(), objective.detach().cpu())

    @staticmethod
    def select(profile: PopulationDfProfile) -> float:
        return float(profile.values[int(profile.weighted_kl.argmin())])

    @staticmethod
    def damp(
        current: float,
        optimum: float,
        dimension: int,
        *,
        damping: float,
        maximum_ratio: float,
    ) -> float:
        """EMA in log-distance from the Wishart admissibility boundary."""

        boundary = dimension - 1.0 + 1e-3
        old_gap = max(current - boundary, 1e-8)
        optimum_gap = max(optimum - boundary, 1e-8)
        updated_gap = torch.exp(
            (1.0 - damping) * torch.tensor(old_gap).log()
            + damping * torch.tensor(optimum_gap).log()
        ).item()
        updated = boundary + updated_gap
        return float(
            min(current * maximum_ratio, max(current / maximum_ratio, updated))
        )

    @staticmethod
    @torch.no_grad()
    def weighted_kl(
        posterior: LocalPosterior,
        population_means: Tensor,
        responsibilities: Tensor,
        population_df: float,
    ) -> float:
        kl = wishart_kl(
            posterior.means,
            posterior.degrees_of_freedom,
            population_means.to(posterior.means),
            population_df,
        )
        gamma = responsibilities.to(kl)
        return float((gamma * kl).sum().cpu() / gamma.sum().clamp_min(1.0).cpu())
