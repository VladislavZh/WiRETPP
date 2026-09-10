"""Trajectory/component-local Wishart variational inference."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from active_wishart_tpp.model.active_block import ActiveBlockDecoder, CachedTrace
from active_wishart_tpp.model.wishart import (
    raw_cholesky_from_spd,
    sample_wishart,
    spd_from_raw_cholesky,
    wishart_kl,
)


@dataclass(frozen=True)
class LocalPosterior:
    """Detached local law and its independently rescored variational free energy."""

    means: Tensor
    degrees_of_freedom: Tensor
    expected_negative_log_likelihood: Tensor
    kl_to_prior: Tensor
    steps_taken: int = 0

    @property
    def free_energy(self) -> Tensor:
        return self.expected_negative_log_likelihood + self.kl_to_prior


class VariationalParameters(nn.Module):
    """Represent each local mean and df through unconstrained trainable coordinates."""

    def __init__(self, means: Tensor, degrees: Tensor) -> None:
        super().__init__()
        self.dimension = means.shape[-1]
        self.raw_mean = nn.Parameter(raw_cholesky_from_spd(means))
        minimum = self.dimension - 1.0 + 0.001
        raw_df = torch.log(torch.expm1((degrees - minimum).clamp_min(0.0001)))
        self.raw_df = nn.Parameter(raw_df)

    def forward(self, selection: slice | None = None) -> tuple[Tensor, Tensor]:
        raw_mean = self.raw_mean if selection is None else self.raw_mean[selection]
        raw_df = self.raw_df if selection is None else self.raw_df[selection]
        means = spd_from_raw_cholesky(raw_mean)
        degrees = self.dimension - 1.0 + 0.001 + torch.nn.functional.softplus(raw_df)
        return (means, degrees)


class LocalWishartInference:
    """Optimize q_mk(U) while the backbone trace and population law stay fixed."""

    def __init__(
        self,
        fabric: Fabric,
        decoder: ActiveBlockDecoder,
        mean_learning_rate: float = 0.02,
        df_learning_rate: float = 0.025,
        gradient_clip: float = 100.0,
    ) -> None:
        self.fabric = fabric
        self.decoder = decoder
        self.mean_learning_rate = mean_learning_rate
        self.df_learning_rate = df_learning_rate
        self.gradient_clip = gradient_clip

    def _repair_nonfinite_gradients(self, parameters) -> int:
        """Repair only numerically non-finite local-gradient entries.

        The forward objective remains unchanged.  NaN entries carry no usable
        direction and are set to zero; signed infinities retain their direction
        at the same bound used by the subsequent global norm clipping step.
        """
        repaired = 0
        for parameter in parameters.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            invalid = ~torch.isfinite(gradient)
            count = int(invalid.sum().item())
            if count == 0:
                continue
            repaired += count
            gradient.nan_to_num_(
                nan=0.0, posinf=self.gradient_clip, neginf=-self.gradient_clip
            )
        return repaired

    def _initialize(
        self,
        n_paths: int,
        population_means: Tensor,
        population_df: float,
        initial_means: Tensor | None,
        initial_df: Tensor | None,
    ):
        n_components = population_means.shape[0]
        priors = population_means[None].expand(n_paths, -1, -1, -1).detach()
        means = priors if initial_means is None else initial_means
        degrees = (
            priors.new_full((n_paths, n_components), population_df)
            if initial_df is None
            else initial_df
        )
        parameters = VariationalParameters(means, degrees)
        optimizer = torch.optim.Adam(
            [
                {"params": parameters.raw_mean, "lr": self.mean_learning_rate},
                {"params": parameters.raw_df, "lr": self.df_learning_rate},
            ]
        )
        parameters, optimizer = self.fabric.setup(parameters, optimizer)
        return (priors, parameters, optimizer)

    def _optimization_step(
        self,
        cache: tuple[CachedTrace, ...],
        priors: Tensor,
        parameters,
        optimizer,
        *,
        population_df: float,
        alpha: float,
        samples: int,
        generator: torch.Generator,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        normalizer = priors.shape[0] * priors.shape[1]
        for batch in cache:
            selection = slice(batch.start, batch.stop)
            means, degrees = parameters(selection)
            draws = sample_wishart(means, degrees, samples, generator)
            score = self.decoder.component_scores(batch.trace, draws, alpha)
            kl = wishart_kl(means, degrees, priors[selection], population_df)
            loss = (-score.mean(2) + kl).sum() / normalizer
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
        self,
        cache: tuple[CachedTrace, ...],
        priors: Tensor,
        parameters,
        *,
        population_df: float,
        alpha: float,
        samples: int,
        seed: int,
    ) -> LocalPosterior:
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
        kl = wishart_kl(means, degrees, priors, population_df)
        return LocalPosterior(
            means.detach(), degrees.detach(), expected_nll.detach(), kl.detach()
        )

    def fit(
        self,
        cache: tuple[CachedTrace, ...],
        population_means: Tensor,
        population_df: float,
        *,
        alpha: float,
        steps: int,
        samples: int,
        evaluation_samples: int,
        seed: int,
        initial_means: Tensor | None = None,
        initial_df: Tensor | None = None,
    ) -> LocalPosterior:
        """Run the fixed local Adam budget and score with fresh independent draws."""
        n_paths = cache[-1].stop
        priors, parameters, optimizer = self._initialize(
            n_paths, population_means, population_df, initial_means, initial_df
        )
        generator = torch.Generator(device=population_means.device).manual_seed(seed)
        for _ in range(steps):
            self._optimization_step(
                cache,
                priors,
                parameters,
                optimizer,
                population_df=population_df,
                alpha=alpha,
                samples=samples,
                generator=generator,
            )
        posterior = self._evaluate(
            cache,
            priors,
            parameters,
            population_df=population_df,
            alpha=alpha,
            samples=evaluation_samples,
            seed=seed + 1000003,
        )
        return LocalPosterior(
            posterior.means,
            posterior.degrees_of_freedom,
            posterior.expected_negative_log_likelihood,
            posterior.kl_to_prior,
            steps,
        )
