"""Base variational-EM schedule for the Active Block Wishart model."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.data import DatasetPartition
from wishart_tpp.inference.alpha import AlphaGridProfile, AlphaMstep
from wishart_tpp.inference.damping import (
    damp_population,
    damp_probability,
    damp_simplex,
)
from wishart_tpp.inference.local import LocalPosterior, LocalWishartInference
from wishart_tpp.inference.mixture_weights import dirichlet_map_weights
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.population_df import PopulationDfMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.model.wishart import sample_wishart
from wishart_tpp.training.cache import TraceCacheBuilder
from wishart_tpp.training.cycle_checkpoint import CycleCheckpointStore
from wishart_tpp.training.evaluation import ModelEvaluator
from wishart_tpp.training.state import ActiveCheckpoint, clone_state_dict


@dataclass(frozen=True)
class ActiveSchedule:
    cycles: int
    neural_steps: int
    local_steps: int
    local_samples: int
    local_evaluation_samples: int
    population_df: float
    initial_alpha: float
    alpha_steps: int
    alpha_samples: int
    balanced_cycles: int
    validation_samples: int
    alpha_warmup_cycles: int = 0
    alpha_reactivation_cycles: int = 0
    fixed_alpha: float | None = None
    omega_damping: float = 1.0
    alpha_damping: float = 1.0
    mixture_weight_damping: float = 1.0
    mixture_weight_dirichlet_concentration: float = 1.0
    learn_population_df: bool = False
    population_df_warmup_cycles: int = 0
    population_df_profile_points: int = 9
    population_df_min: float | None = None
    population_df_max: float = 256.0
    population_df_damping: float = 0.25
    population_df_maximum_ratio: float = 2.0


@dataclass
class _ActiveState:
    population_means: torch.Tensor
    log_weights: torch.Tensor
    alpha: float
    population_df: float
    previous_means: torch.Tensor | None = None
    previous_df: torch.Tensor | None = None


@dataclass(frozen=True)
class _CycleOutcome:
    row: dict[str, float]
    validation_nll: float


class ActiveTrainer:
    """Alternate exact coordinate updates and short neural M-steps."""

    def __init__(
        self,
        fabric: Fabric,
        decoder: ActiveBlockDecoder,
        evaluator: ModelEvaluator,
        local_inference: LocalWishartInference,
        population_mstep: PopulationMstep,
        alpha_mstep: AlphaMstep,
        responsibility_updater: ResponsibilityUpdater,
        *,
        batch_size: int,
        effective_batch_size: int,
        cache_batch_size: int,
        path_shard_size: int | None = None,
        em_batch_size: int | None = None,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        update_neural_model: bool = True,
        population_df_mstep: PopulationDfMstep | None = None,
        reduce_lr_on_plateau: bool = False,
        lr_plateau_factor: float = 0.5,
        lr_plateau_patience: int = 5,
        lr_plateau_min_lr: float = 0.0,
    ) -> None:
        self.fabric = fabric
        self.decoder = decoder
        self.evaluator = evaluator
        self.local_inference = local_inference
        self.population_mstep = population_mstep
        self.alpha_mstep = alpha_mstep
        self.responsibility_updater = responsibility_updater
        self.batch_size = batch_size
        self.trace_batch_size = cache_batch_size
        self.effective_batch_size = effective_batch_size
        self.cache_builder = TraceCacheBuilder(cache_batch_size)
        self.path_shard_size = path_shard_size
        self.em_batch_size = em_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.update_neural_model = bool(update_neural_model)
        self.population_df_mstep = population_df_mstep or PopulationDfMstep()
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_plateau_factor = lr_plateau_factor
        self.lr_plateau_patience = lr_plateau_patience
        self.lr_plateau_min_lr = lr_plateau_min_lr

    def setup(self, model):
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name != "mixture_logits"
        ]
        optimizer = torch.optim.Adam(
            parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return self.fabric.setup(model, optimizer)

    def _lr_scheduler(self, optimizer):
        if not self.reduce_lr_on_plateau:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_plateau_factor,
            patience=self.lr_plateau_patience,
            min_lr=self.lr_plateau_min_lr,
        )

    @staticmethod
    def _sample_indices(
        random: np.random.Generator, train: DatasetPartition, count: int
    ) -> np.ndarray:
        return random.choice(
            len(train.sequences), min(count, len(train.sequences)), replace=False
        )

    def _microbatch_loss(
        self,
        model,
        batch: DatasetPartition,
        selected: np.ndarray,
        posterior: LocalPosterior,
        responsibilities: torch.Tensor,
        *,
        alpha: float,
        samples: int,
        exposure: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        index = torch.as_tensor(selected, device=posterior.means.device)
        device = model.device
        proposal_means = posterior.means.index_select(0, index).to(device).detach()
        proposal_df = (
            posterior.degrees_of_freedom.index_select(0, index).to(device).detach()
        )
        draws = sample_wishart(
            proposal_means,
            proposal_df,
            samples,
            generator,
        ).detach()
        objective = model.mixture_logits.new_zeros(())

        gamma = responsibilities.index_select(0, index).to(device).detach()

        # Sample q(U) once for the physical batch, but build neural traces in
        # the smaller chunks used by the stable historical implementation.
        for start in range(0, len(batch.sequences), self.trace_batch_size):
            stop = min(start + self.trace_batch_size, len(batch.sequences))
            score = self.decoder.component_scores(
                model(batch.sequences[start:stop]), draws[start:stop], alpha
            )
            objective = objective + (gamma[start:stop] * score.mean(2)).sum()
        return -objective / exposure

    def _neural_step(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        indices: np.ndarray,
        posterior: LocalPosterior,
        responsibilities: torch.Tensor,
        *,
        alpha: float,
        samples: int,
        generator: torch.Generator,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        exposure = train.select(indices).exposure
        loss_value = 0.0

        # Hold q(U) and gamma fixed while accumulating the neural M-step.
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start : start + self.batch_size]
            loss = self._microbatch_loss(
                model,
                train.select(selected),
                selected,
                posterior,
                responsibilities,
                alpha=alpha,
                samples=samples,
                exposure=exposure,
                generator=generator,
            )
            self.fabric.backward(loss)
            loss_value += float(loss.detach().cpu())
        self.fabric.clip_gradients(model, optimizer, max_norm=self.gradient_clip)
        optimizer.step()
        return loss_value

    def _neural_mstep(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        posterior: LocalPosterior,
        responsibilities: torch.Tensor,
        *,
        alpha: float,
        steps: int,
        samples: int,
        seed: int,
    ) -> float:
        random = np.random.default_rng(seed)
        generator = torch.Generator(device=model.device).manual_seed(seed + 101)
        last_loss = float("nan")
        for _ in range(steps):
            indices = self._sample_indices(random, train, self.effective_batch_size)
            last_loss = self._neural_step(
                model,
                optimizer,
                train,
                indices,
                posterior,
                responsibilities,
                alpha=alpha,
                samples=samples,
                generator=generator,
            )
        return last_loss

    @staticmethod
    def _initial_population_means(
        model,
        population_df: float,
        optimization_seed: int,
    ) -> torch.Tensor:
        """Draw distinct deterministic SPD population means for every component.

        Initializing every population mean at the identity leaves the Wishart
        components exactly symmetric and, under a damped population M-step, keeps
        a large deterministic identity contribution for several cycles.  A
        draw from the configured mean/df Wishart family gives a parameter-free
        random initialization.  Trace normalization enforces the model's
        identifiability constraint exactly.
        """

        dimension = model.n_marks
        identity = torch.eye(dimension, device=model.device, dtype=model.dtype)[
            None
        ].repeat(model.n_components, 1, 1)
        degrees = identity.new_full((model.n_components,), population_df)
        generator = torch.Generator(device=model.device).manual_seed(
            optimization_seed + 50_021
        )
        means = sample_wishart(identity, degrees, 1, generator).squeeze(1)
        trace = means.diagonal(dim1=-2, dim2=-1).sum(-1)
        means = means * (float(dimension) / trace)[:, None, None]
        return 0.5 * (means + means.transpose(-1, -2))

    def _initial_state(
        self,
        model,
        schedule: ActiveSchedule,
        optimization_seed: int,
    ) -> _ActiveState:
        population_means = self._initial_population_means(
            model, schedule.population_df, optimization_seed
        )
        return _ActiveState(
            population_means=population_means,
            log_weights=model.mixture_log_weights().detach(),
            alpha=(
                schedule.fixed_alpha
                if schedule.fixed_alpha is not None
                else (0.0 if schedule.alpha_warmup_cycles else schedule.initial_alpha)
            ),
            population_df=schedule.population_df,
        )

    def _e_step(
        self,
        model,
        train: DatasetPartition,
        schedule: ActiveSchedule,
        state: _ActiveState,
        seed: int,
        alpha_seed: int,
        profile_alpha: bool,
    ) -> tuple[tuple | AlphaGridProfile, LocalPosterior]:
        if self.path_shard_size is None:
            cache = self.cache_builder.build(model, train, resample_integration=True)
            posterior = self.local_inference.fit(
                cache,
                state.population_means,
                state.population_df,
                alpha=state.alpha,
                steps=schedule.local_steps,
                samples=schedule.local_samples,
                evaluation_samples=schedule.local_evaluation_samples,
                seed=seed,
                initial_means=state.previous_means,
                initial_df=state.previous_df,
            )
            return cache, posterior

        outputs: list[LocalPosterior] = []
        total = len(train.sequences)
        alpha_values = (
            None
            if schedule.fixed_alpha is not None or not profile_alpha
            else self.alpha_mstep.candidate_grid(state.alpha, schedule.alpha_steps)
        )
        alpha_profiles = []
        started = time.perf_counter()
        for start in range(0, total, self.path_shard_size):
            stop = min(start + self.path_shard_size, total)
            selected = np.arange(start, stop, dtype=np.int64)
            shard = train.select(selected)
            cache = self.cache_builder.build(model, shard, resample_integration=True)
            initial_means = (
                None
                if state.previous_means is None
                else state.previous_means[start:stop].to(model.device)
            )
            initial_df = (
                None
                if state.previous_df is None
                else state.previous_df[start:stop].to(model.device)
            )
            local = self.local_inference.fit(
                cache,
                state.population_means.to(model.device),
                state.population_df,
                alpha=state.alpha,
                steps=schedule.local_steps,
                samples=schedule.local_samples,
                evaluation_samples=schedule.local_evaluation_samples,
                seed=seed + start * 1_000_003,
                initial_means=initial_means,
                initial_df=initial_df,
            )
            if alpha_values is not None:
                alpha_profiles.append(
                    self.alpha_mstep.profile(
                        cache,
                        local,
                        alpha_values,
                        samples=schedule.alpha_samples,
                        seed=alpha_seed + start * 1_000_033,
                    ).scores
                )
            outputs.append(
                LocalPosterior(
                    local.means.cpu(),
                    local.degrees_of_freedom.cpu(),
                    local.expected_negative_log_likelihood.cpu(),
                    local.kl_to_prior.cpu(),
                    local.steps_taken,
                    local.converged,
                    local.objective_relative_change,
                    local.kappa_relative_change,
                    local.gradient_norm_p95,
                )
            )
            del cache, local, initial_means, initial_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.perf_counter() - started
            eta = elapsed / stop * (total - stop)
            self.fabric.print(
                f"[active-e] paths={stop}/{total} elapsed_seconds={elapsed:.1f} "
                f"eta_seconds={eta:.1f}",
                flush=True,
            )
        posterior = LocalPosterior(
            torch.cat([item.means for item in outputs]),
            torch.cat([item.degrees_of_freedom for item in outputs]),
            torch.cat([item.expected_negative_log_likelihood for item in outputs]),
            torch.cat([item.kl_to_prior for item in outputs]),
            schedule.local_steps,
        )
        alpha_input: tuple | AlphaGridProfile = ()
        if alpha_values is not None:
            alpha_input = AlphaGridProfile(
                alpha_values.cpu(), torch.cat(alpha_profiles)
            )
        return alpha_input, posterior

    def _population_step(
        self,
        model,
        posterior: LocalPosterior,
        state: _ActiveState,
        *,
        balanced: bool,
        damping: float,
        weight_damping: float,
        weight_dirichlet_concentration: float,
    ) -> torch.Tensor:
        posterior_device = posterior.free_energy.device
        gamma = self.responsibility_updater.update(
            posterior.free_energy,
            state.log_weights.to(posterior_device),
            balanced=balanced,
        )
        self._population_step_from_responsibilities(
            model,
            posterior,
            gamma,
            state,
            damping=damping,
            weight_damping=weight_damping,
            weight_dirichlet_concentration=weight_dirichlet_concentration,
        )
        return gamma

    def _population_step_from_responsibilities(
        self,
        model,
        posterior: LocalPosterior,
        gamma: torch.Tensor,
        state: _ActiveState,
        *,
        damping: float,
        weight_damping: float,
        weight_dirichlet_concentration: float,
    ) -> None:
        """Update population parameters from a full cached responsibility table."""

        optimum = self.population_mstep.update(posterior.means, gamma)
        state.population_means = damp_population(
            state.population_means.to(optimum.device), optimum, damping
        ).to(model.device)
        optimum_weights = dirichlet_map_weights(gamma, weight_dirichlet_concentration)
        weights = damp_simplex(
            state.log_weights.exp().to(optimum_weights.device),
            optimum_weights,
            weight_damping,
        )
        state.log_weights = weights.log().to(model.device).detach()
        with torch.no_grad():
            model.mixture_logits.copy_(state.log_weights)

    def _population_df_step(
        self,
        posterior: LocalPosterior,
        responsibilities: torch.Tensor,
        state: _ActiveState,
        schedule: ActiveSchedule,
        cycle: int,
    ) -> float:
        """Update global df from the current train-block ELBO profile."""

        optimum = state.population_df
        if (
            schedule.learn_population_df
            and cycle > schedule.population_df_warmup_cycles
        ):
            dimension = posterior.means.shape[-1]
            values = self.population_df_mstep.candidate_grid(
                state.population_df,
                dimension,
                points=schedule.population_df_profile_points,
                minimum=schedule.population_df_min,
                maximum=schedule.population_df_max,
            )
            profile = self.population_df_mstep.profile(
                posterior,
                state.population_means,
                responsibilities,
                values,
            )
            optimum = self.population_df_mstep.select(profile)
            state.population_df = self.population_df_mstep.damp(
                state.population_df,
                optimum,
                dimension,
                damping=schedule.population_df_damping,
                maximum_ratio=schedule.population_df_maximum_ratio,
            )
        return optimum

    def _validation_row(
        self,
        model,
        validation: DatasetPartition,
        schedule: ActiveSchedule,
        state: _ActiveState,
        posterior: LocalPosterior,
        gamma: torch.Tensor,
        neural_loss: float,
        population_df_optimum: float,
        cycle: int,
        seed: int,
    ) -> tuple[dict[str, float], float]:
        evaluation = self.evaluator.active(
            model,
            validation,
            state.population_means,
            state.log_weights,
            population_df=state.population_df,
            alpha=state.alpha,
            samples=schedule.validation_samples,
            seed=seed,
        )
        # Diagnostic endpoint: score the very same neural intensity bank with
        # the Wishart operator bypassed.  In particular, for K=1 this separates
        # slow backbone optimization from degradation introduced by the latent
        # Wishart layer without adding any term to either training objective.
        backbone_evaluation = self.evaluator.pure(model, validation)
        row = {
            "cycle": cycle,
            "alpha": state.alpha,
            "population_df": state.population_df,
            "population_df_optimum": population_df_optimum,
            "validation_nll_per_exposure": evaluation.nll_per_exposure,
            "backbone_validation_nll_per_exposure": (
                backbone_evaluation.nll_per_exposure
            ),
            "validation_purity_descriptive": evaluation.purity,
            "validation_ari_descriptive": evaluation.ari,
            "backbone_validation_purity_descriptive": (backbone_evaluation.purity),
            "backbone_validation_ari_descriptive": backbone_evaluation.ari,
            "responsibility_min_mass": float(gamma.sum(0).min().cpu()),
            "weighted_kl": self.population_df_mstep.weighted_kl(
                posterior,
                state.population_means,
                gamma,
                state.population_df,
            ),
            "neural_nll_per_exposure": neural_loss,
        }
        component_masses = gamma.sum(0).detach().cpu()
        mixture_weights = state.log_weights.exp().detach().cpu()
        row.update(
            {
                f"responsibility_mass_{index}": float(value)
                for index, value in enumerate(component_masses)
            }
        )
        row.update(
            {
                f"mixture_weight_{index}": float(value)
                for index, value in enumerate(mixture_weights)
            }
        )
        return row, evaluation.nll_per_exposure

    @staticmethod
    def _checkpoint(
        model, state: _ActiveState, cycle: int, nll: float
    ) -> ActiveCheckpoint:
        return ActiveCheckpoint(
            cycle,
            nll,
            clone_state_dict(model),
            state.population_means.detach().cpu().clone(),
            state.log_weights.detach().cpu().clone(),
            state.alpha,
            state.population_df,
        )

    def _resume_signature(
        self,
        schedule: ActiveSchedule,
        train: DatasetPartition,
        validation: DatasetPartition,
        optimization_seed: int,
        monte_carlo_seed: int,
    ) -> dict[str, Any]:
        # ``cycles`` is deliberately excluded: increasing only the number of
        # outer cycles is the supported continuation use case.
        schedule_values = asdict(schedule)
        schedule_values.pop("cycles")
        df_learning = {
            key: schedule_values.pop(key)
            for key in (
                "learn_population_df",
                "population_df_warmup_cycles",
                "population_df_profile_points",
                "population_df_min",
                "population_df_max",
                "population_df_damping",
                "population_df_maximum_ratio",
            )
        }
        signature = {
            "schedule": schedule_values,
            "optimization_seed": optimization_seed,
            "monte_carlo_seed": monte_carlo_seed,
            "population_initialization": {
                "method": "trace_normalized_wishart",
                "seed_offset": 50_021,
            },
            "train_paths": len(train.sequences),
            "validation_paths": len(validation.sequences),
            "effective_batch_size": self.effective_batch_size,
            "em_batch_size": self.em_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "time_integration_resampling": "per_forward_common_uniform",
            "active_block_likelihood": {
                "version": getattr(self.decoder, "intensity_formula", "unknown"),
                "structure": "independent_component_c_by_c_blocks",
                "intensity": ("(1-alpha)*h + alpha*(sqrt(W)@sqrt(h))**2 + epsilon"),
                "mixture_weights": "independent_logits",
                "cluster_w_prior": "independent",
                "cluster_w_coupling": "likelihood_only",
                "w_to_mixture_routing": False,
            },
        }
        if df_learning["learn_population_df"]:
            signature["population_df_learning"] = df_learning
        if self.reduce_lr_on_plateau:
            signature["reduce_lr_on_plateau"] = {
                "factor": self.lr_plateau_factor,
                "patience": self.lr_plateau_patience,
                "min_lr": self.lr_plateau_min_lr,
                "monitor": "validation_nll_per_exposure",
            }
        if not self.update_neural_model:
            signature["update_neural_model"] = False
        intensity_floor = float(getattr(self.decoder, "intensity_floor", 0.0))
        signature["wishart_intensity_floor"] = {
            "epsilon": intensity_floor,
            "scaling": getattr(
                self.decoder,
                "intensity_floor_scaling",
                "constant",
            ),
            "placement": "event_and_compensator_rates",
        }
        return signature

    @staticmethod
    def _state_payload(state: _ActiveState, retain_local_state: bool) -> dict[str, Any]:
        return {
            "population_means": state.population_means,
            "log_weights": state.log_weights,
            "alpha": state.alpha,
            "population_df": state.population_df,
            "previous_means": state.previous_means if retain_local_state else None,
            "previous_df": state.previous_df if retain_local_state else None,
        }

    @staticmethod
    def _state_from_payload(
        payload: dict[str, Any], device: torch.device, default_df: float
    ) -> _ActiveState:
        return _ActiveState(
            population_means=payload["population_means"].to(device),
            log_weights=payload["log_weights"].to(device),
            alpha=float(payload["alpha"]),
            population_df=float(payload.get("population_df", default_df)),
            previous_means=(
                payload["previous_means"].to(device)
                if payload["previous_means"] is not None
                else None
            ),
            previous_df=(
                payload["previous_df"].to(device)
                if payload["previous_df"] is not None
                else None
            ),
        )

    def _global_msteps(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        schedule: ActiveSchedule,
        state: _ActiveState,
        cache: tuple | AlphaGridProfile,
        posterior: LocalPosterior,
        gamma: torch.Tensor,
        cycle: int,
        seed: int,
    ) -> float:
        alpha_trainable = cycle > (
            schedule.alpha_warmup_cycles + schedule.alpha_reactivation_cycles
        )
        if schedule.fixed_alpha is None and alpha_trainable:
            optimum = (
                self.alpha_mstep.select(cache, gamma)
                if isinstance(cache, AlphaGridProfile)
                else self.alpha_mstep.update(
                    cache,
                    posterior,
                    gamma,
                    initial=state.alpha,
                    iterations=schedule.alpha_steps,
                    samples=schedule.alpha_samples,
                    seed=seed + cycle * 20_011,
                )
            )
            state.alpha = damp_probability(state.alpha, optimum, schedule.alpha_damping)
        if not self.update_neural_model:
            return float("nan")
        return self._neural_mstep(
            model,
            optimizer,
            train,
            posterior,
            gamma,
            alpha=state.alpha,
            steps=schedule.neural_steps,
            samples=schedule.local_samples,
            seed=seed + cycle * 30_013,
        )

    def _run_cycle(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        validation: DatasetPartition,
        schedule: ActiveSchedule,
        state: _ActiveState,
        cycle: int,
        optimization_seed: int,
        monte_carlo_seed: int,
    ) -> _CycleOutcome:
        if schedule.fixed_alpha is None:
            if cycle <= schedule.alpha_warmup_cycles:
                state.alpha = 0.0
            elif cycle <= (
                schedule.alpha_warmup_cycles + schedule.alpha_reactivation_cycles
            ):
                state.alpha = schedule.initial_alpha

        # Each outer cycle owns a deterministic sequence of fresh uniform
        # compensator points.  Every forward draws one vector and shares it
        # across all intervals in that forward.  Resetting from the cycle id
        # makes cycle-boundary resume bitwise reproducible.
        bank = getattr(model, "module", model)
        bank.integration_rule.reset_training_draws(monte_carlo_seed + cycle * 1_000_003)
        # E-step: refresh every q_mk against the current backbone and population.
        cache, posterior = self._e_step(
            model,
            train,
            schedule,
            state,
            seed=optimization_seed + cycle * 10_007,
            alpha_seed=optimization_seed + cycle * 20_011,
            profile_alpha=cycle
            > (schedule.alpha_warmup_cycles + schedule.alpha_reactivation_cycles),
        )
        gamma = self._population_step(
            model,
            posterior,
            state,
            balanced=cycle <= schedule.balanced_cycles,
            damping=schedule.omega_damping,
            weight_damping=schedule.mixture_weight_damping,
            weight_dirichlet_concentration=(
                schedule.mixture_weight_dirichlet_concentration
            ),
        )

        population_df_optimum = self._population_df_step(
            posterior, gamma, state, schedule, cycle
        )

        # Sharded E inference returns only CPU posterior/profile tensors, but
        # Python/CUDA temporaries from its largest shard can otherwise survive
        # until the first autograd forward.  Reclaim them at the E->NN phase
        # boundary; model/optimizer state and all deterministic RNGs are intact.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        neural_loss = self._global_msteps(
            model,
            optimizer,
            train,
            schedule,
            state,
            cache,
            posterior,
            gamma,
            cycle,
            optimization_seed,
        )
        # The final neural microbatch leaves gradients attached to every model
        # parameter until the next cycle's first zero_grad().  Validation is
        # inference-only and can be much larger (MC64), so release those
        # buffers at the phase boundary without changing optimizer state/RNG.
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row, validation_nll = self._validation_row(
            model,
            validation,
            schedule,
            state,
            posterior,
            gamma,
            neural_loss,
            population_df_optimum,
            cycle,
            seed=monte_carlo_seed + 40_009,
        )
        state.previous_means = posterior.means
        state.previous_df = posterior.degrees_of_freedom
        return _CycleOutcome(row, validation_nll)

    def fit(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        validation: DatasetPartition,
        schedule: ActiveSchedule,
        *,
        optimization_seed: int,
        monte_carlo_seed: int,
        checkpoint_path: str | Path | None = None,
    ) -> tuple[ActiveCheckpoint, list[dict[str, float]]]:
        state = self._initial_state(model, schedule, optimization_seed)
        best: ActiveCheckpoint | None = None
        history: list[dict[str, float]] = []
        completed_cycle = 0
        path = Path(checkpoint_path) if checkpoint_path is not None else None
        signature = self._resume_signature(
            schedule, train, validation, optimization_seed, monte_carlo_seed
        )
        scheduler = self._lr_scheduler(optimizer)
        checkpoint_store = CycleCheckpointStore()
        if path is not None and path.is_file():
            completed_cycle, state_payload, best, history = checkpoint_store.load(
                path, model, optimizer, scheduler, signature, schedule.cycles
            )
            state = self._state_from_payload(
                state_payload, model.device, schedule.population_df
            )
            self.fabric.print(
                f"[active-checkpoint] resumed_cycle={completed_cycle} "
                f"target_cycles={schedule.cycles} path={path}",
                flush=True,
            )
        started = time.perf_counter()

        self.fabric.print(
            f"[active] cycle={completed_cycle}/{schedule.cycles} "
            "elapsed_seconds=0 eta_seconds=unknown stage=starting",
            flush=True,
        )

        for cycle in range(completed_cycle + 1, schedule.cycles + 1):
            cycle_started = time.perf_counter()
            cycle_learning_rate = float(optimizer.param_groups[0]["lr"])
            cycle_train = train
            epoch = block = block_count = 1
            if self.em_batch_size is not None and self.em_batch_size < len(
                train.sequences
            ):
                indices, epoch, block, block_count = self._em_block_indices(
                    len(train.sequences),
                    self.em_batch_size,
                    cycle,
                    optimization_seed,
                )
                cycle_train = train.select(indices)
                state.previous_means = None
                state.previous_df = None
                self.fabric.print(
                    f"[active-block] cycle={cycle}/{schedule.cycles} "
                    f"epoch={epoch} block={block}/{block_count} "
                    f"paths={len(indices)}",
                    flush=True,
                )
            outcome = self._run_cycle(
                model,
                optimizer,
                cycle_train,
                validation,
                schedule,
                state,
                cycle,
                optimization_seed,
                monte_carlo_seed,
            )
            outcome.row.update(
                {
                    "em_epoch": float(epoch),
                    "em_block": float(block),
                    "em_block_count": float(block_count),
                    "em_paths": float(len(cycle_train.sequences)),
                    "posterior_cache_hits": 0.0,
                    "posterior_cache_hit_rate": 0.0,
                    "learning_rate": cycle_learning_rate,
                }
            )
            if scheduler is not None:
                scheduler.step(outcome.validation_nll)
            outcome.row["neural_model_updated"] = float(self.update_neural_model)
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            outcome.row["next_learning_rate"] = next_learning_rate
            outcome.row["lr_reduced"] = float(next_learning_rate < cycle_learning_rate)
            # Select checkpoints only by held-out prior-predictive likelihood.
            history.append(outcome.row)
            if best is None or outcome.validation_nll < best.validation_nll:
                best = self._checkpoint(model, state, cycle, outcome.validation_nll)
            elapsed = time.perf_counter() - started
            cycle_seconds = time.perf_counter() - cycle_started
            cycles_completed_this_process = cycle - completed_cycle
            eta = elapsed / cycles_completed_this_process * (schedule.cycles - cycle)
            component_count = int(state.log_weights.numel())
            masses = ",".join(
                f"{outcome.row[f'responsibility_mass_{index}']:.1f}"
                for index in range(component_count)
            )
            weights = ",".join(
                f"{outcome.row[f'mixture_weight_{index}']:.6f}"
                for index in range(component_count)
            )
            self.fabric.print(
                f"[active] cycle={cycle}/{schedule.cycles} "
                f"elapsed_seconds={elapsed:.1f} cycle_seconds={cycle_seconds:.1f} "
                f"eta_seconds={eta:.1f} alpha={state.alpha:.6f} "
                f"df={state.population_df:.6f} "
                f"neural_update={'on' if self.update_neural_model else 'off'} "
                "neural_objective=fixed_q_elbo "
                f"lr={cycle_learning_rate:.8g} "
                f"next_lr={next_learning_rate:.8g} "
                f"validation_nll={outcome.validation_nll:.6f} "
                "backbone_validation_nll="
                f"{outcome.row['backbone_validation_nll_per_exposure']:.6f} "
                f"validation_purity="
                f"{outcome.row['validation_purity_descriptive']:.6f} "
                f"validation_ari="
                f"{outcome.row['validation_ari_descriptive']:.6f} "
                f"backbone_purity="
                f"{outcome.row['backbone_validation_purity_descriptive']:.6f} "
                f"backbone_ari="
                f"{outcome.row['backbone_validation_ari_descriptive']:.6f} "
                f"min_mass={outcome.row['responsibility_min_mass']:.1f} "
                f"masses=[{masses}] weights=[{weights}] "
                f"best_cycle={best.cycle}",
                flush=True,
            )
            if path is not None:
                retain_local_state = not (
                    self.em_batch_size is not None
                    and self.em_batch_size < len(train.sequences)
                )
                checkpoint_store.save(
                    path,
                    model,
                    optimizer,
                    self._state_payload(state, retain_local_state),
                    best,
                    history,
                    cycle,
                    signature,
                    scheduler,
                )
                self.fabric.print(
                    f"[active-checkpoint] saved_cycle={cycle} path={path}",
                    flush=True,
                )

        assert best is not None
        model.load_state_dict(best.model_state)
        return best, history

    @staticmethod
    def _em_block_indices(
        total: int, block_size: int, cycle: int, seed: int
    ) -> tuple[np.ndarray, int, int, int]:
        """Return one non-overlapping block from an epoch-wise permutation."""

        if total < 1 or block_size < 1 or cycle < 1:
            raise ValueError("EM block arguments must be positive")
        block_count = (total + block_size - 1) // block_size
        epoch_index, block_index = divmod(cycle - 1, block_count)
        random = np.random.default_rng(seed + 500_009 + epoch_index * 1_000_003)
        permutation = random.permutation(total)
        # Balance a non-divisible epoch across blocks instead of leaving one
        # tiny final block.  This keeps block sufficient-statistic noise
        # and the relative Dirichlet pseudo-count essentially constant.
        base, remainder = divmod(total, block_count)
        start = block_index * base + min(block_index, remainder)
        stop = start + base + (1 if block_index < remainder else 0)
        return (
            permutation[start:stop],
            epoch_index + 1,
            block_index + 1,
            block_count,
        )
