"""Trajectory/component-local Wishart variational inference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from lightning.fabric import Fabric
from torch import Tensor, nn

from wishart_tpp.model.active_block import ActiveBlockDecoder, CachedTrace
from wishart_tpp.model.wishart import (
    raw_cholesky_from_spd,
    sample_wishart,
    spd_from_raw_cholesky,
    wishart_kl,
)


@dataclass(frozen=True)
class LocalPosterior:
    means: Tensor
    degrees_of_freedom: Tensor
    expected_negative_log_likelihood: Tensor
    kl_to_prior: Tensor
    steps_taken: int = 0
    converged: bool = False
    objective_relative_change: float = float("nan")
    kappa_relative_change: float = float("nan")
    gradient_norm_p95: float = float("nan")

    @property
    def free_energy(self) -> Tensor:
        return self.expected_negative_log_likelihood + self.kl_to_prior


class VariationalParameters(nn.Module):
    def __init__(self, means: Tensor, degrees: Tensor) -> None:
        super().__init__()
        self.dimension = means.shape[-1]
        self.raw_mean = nn.Parameter(raw_cholesky_from_spd(means))
        minimum = self.dimension - 1.0 + 1e-3
        raw_df = torch.log(torch.expm1((degrees - minimum).clamp_min(1e-4)))
        self.raw_df = nn.Parameter(raw_df)

    def forward(self, selection: slice | None = None) -> tuple[Tensor, Tensor]:
        raw_mean = self.raw_mean if selection is None else self.raw_mean[selection]
        raw_df = self.raw_df if selection is None else self.raw_df[selection]
        means = spd_from_raw_cholesky(raw_mean)
        degrees = self.dimension - 1.0 + 1e-3 + torch.nn.functional.softplus(raw_df)
        return means, degrees


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
                nan=0.0,
                posinf=self.gradient_clip,
                neginf=-self.gradient_clip,
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
        return priors, parameters, optimizer

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
        event_count: Tensor | None = None,
    ) -> float:
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
                f"[local-numeric] repaired_nonfinite_gradients={repaired}",
                flush=True,
            )
        gradient_p95 = float("nan")
        if event_count is not None:
            mean_gradient = parameters.raw_mean.grad.square().sum(
                dim=(-2, -1)
            ).sqrt() * float(optimizer.param_groups[0]["lr"])
            df_gradient = parameters.raw_df.grad.abs() * float(
                optimizer.param_groups[1]["lr"]
            )
            local_gradient = torch.sqrt(mean_gradient.square() + df_gradient.square())
            normalized = local_gradient * local_gradient.numel() / event_count
            gradient_p95 = float(torch.quantile(normalized.flatten(), 0.95).cpu())
        self.fabric.clip_gradients(parameters, optimizer, max_norm=self.gradient_clip)
        optimizer.step()
        return gradient_p95

    @staticmethod
    def _event_counts(cache: tuple[CachedTrace, ...], n_paths: int) -> Tensor:
        reference = cache[0].trace.event_base_rates
        counts = reference.new_ones((n_paths, 1))
        for batch in cache:
            size = batch.stop - batch.start
            local = torch.bincount(batch.trace.event_paths, minlength=size)
            counts[batch.start : batch.stop, 0] = local.to(reference).clamp_min(1.0)
        return counts

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
        adaptive_tolerance: float | None = None,
        adaptive_kappa_tolerance: float = 5e-3,
        adaptive_gradient_tolerance: float = 2e-3,
        adaptive_minimum_steps: int = 200,
        adaptive_check_interval: int = 50,
        adaptive_patience: int = 3,
        adaptive_monitor_samples: int = 16,
        adaptive_learning_rate_floor: float = 0.02,
    ) -> LocalPosterior:
        n_paths = cache[-1].stop
        priors, parameters, optimizer = self._initialize(
            n_paths, population_means, population_df, initial_means, initial_df
        )
        generator = torch.Generator(device=population_means.device).manual_seed(seed)
        started = time.perf_counter()
        progress_interval = max(1, steps // 20) if steps >= 100 else None
        event_count = self._event_counts(cache, n_paths)
        initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        previous_monitor = previous_degrees = None
        stable_checks = 0
        converged = False
        objective_change = kappa_change = gradient_p95 = float("nan")
        steps_taken = steps

        # Optimize only the trajectory/component variational family.
        for step in range(1, steps + 1):
            if adaptive_tolerance is not None:
                progress = (step - 1) / max(steps - 1, 1)
                factor = adaptive_learning_rate_floor + (
                    1.0 - adaptive_learning_rate_floor
                ) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                    group["lr"] = initial_lr * factor
            gradient_p95 = self._optimization_step(
                cache,
                priors,
                parameters,
                optimizer,
                population_df=population_df,
                alpha=alpha,
                samples=samples,
                generator=generator,
                event_count=event_count if adaptive_tolerance is not None else None,
            )
            if progress_interval is not None and (
                step == 1 or step == steps or step % progress_interval == 0
            ):
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (steps - step)
                self.fabric.print(
                    f"[local] step={step}/{steps} elapsed_seconds={elapsed:.1f} "
                    f"eta_seconds={eta:.1f}",
                    flush=True,
                )
            should_check = (
                adaptive_tolerance is not None
                and step >= adaptive_minimum_steps
                and step % adaptive_check_interval == 0
            )
            if should_check:
                monitor = self._evaluate(
                    cache,
                    priors,
                    parameters,
                    population_df=population_df,
                    alpha=alpha,
                    samples=adaptive_monitor_samples,
                    seed=seed + 70_000_019,
                )
                normalized = monitor.free_energy / event_count
                if previous_monitor is not None and previous_degrees is not None:
                    objective_change = float(
                        (
                            (normalized - previous_monitor).abs()
                            / previous_monitor.abs().clamp_min(1e-6)
                        )
                        .quantile(0.95)
                        .cpu()
                    )
                    kappa_change = float(
                        (
                            (monitor.degrees_of_freedom - previous_degrees).abs()
                            / previous_degrees.abs().clamp_min(1e-6)
                        )
                        .quantile(0.95)
                        .cpu()
                    )
                    stable = (
                        objective_change <= adaptive_tolerance
                        and kappa_change <= adaptive_kappa_tolerance
                        and gradient_p95 <= adaptive_gradient_tolerance
                    )
                    stable_checks = stable_checks + 1 if stable else 0
                    if stable_checks >= adaptive_patience:
                        converged = True
                        steps_taken = step
                        break
                previous_monitor = normalized.clone()
                previous_degrees = monitor.degrees_of_freedom.clone()

        # Re-estimate free energies with a larger, independent Monte Carlo sample.
        posterior = self._evaluate(
            cache,
            priors,
            parameters,
            population_df=population_df,
            alpha=alpha,
            samples=evaluation_samples,
            seed=seed + 1_000_003,
        )
        return LocalPosterior(
            posterior.means,
            posterior.degrees_of_freedom,
            posterior.expected_negative_log_likelihood,
            posterior.kl_to_prior,
            steps_taken,
            converged,
            objective_change,
            kappa_change,
            gradient_p95,
        )
