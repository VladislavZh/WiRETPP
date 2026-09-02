"""Standalone Active Block Wishart experiment runner."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wishart_tpp.backbones import IntensityBank
from wishart_tpp.inference.alpha import AlphaMstep
from wishart_tpp.inference.local import LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.metrics import clustering_summary
from wishart_tpp.training.active import ActiveSchedule, ActiveTrainer
from wishart_tpp.training.cache import TraceCacheBuilder
from wishart_tpp.training.evaluation import Evaluation
from wishart_tpp.training.experiment import (
    ExperimentContext,
    ExperimentRunner,
    SharedTraining,
)
from wishart_tpp.training.state import ActiveCheckpoint, clone_state_dict


@dataclass(frozen=True)
class _WishartTraining:
    shared: SharedTraining
    model: IntensityBank
    history: list[dict[str, float]]
    checkpoint: ActiveCheckpoint


@dataclass(frozen=True)
class _WishartTest:
    prior: Evaluation
    posterior_clustering: dict[str, float]


class WishartExperimentRunner(ExperimentRunner):
    """Fit and evaluate only the active-block Wishart hierarchy."""

    @property
    def method(self) -> str:
        return "wishart"

    def _schedule(self) -> ActiveSchedule:
        training = self.config.training
        return ActiveSchedule(
            cycles=training.active_cycles,
            neural_steps=training.neural_steps_per_cycle,
            local_steps=training.local_steps,
            local_samples=training.local_samples,
            local_evaluation_samples=training.local_evaluation_samples,
            population_df=training.population_df,
            learn_population_df=training.learn_population_df,
            population_df_warmup_cycles=(training.population_df_warmup_cycles),
            population_df_profile_points=(training.population_df_profile_points),
            population_df_min=training.population_df_min,
            population_df_max=training.population_df_max,
            population_df_damping=training.population_df_damping,
            population_df_maximum_ratio=(training.population_df_maximum_ratio),
            initial_alpha=training.initial_alpha,
            alpha_warmup_cycles=training.alpha_warmup_cycles,
            alpha_reactivation_cycles=training.alpha_reactivation_cycles,
            alpha_steps=training.alpha_steps,
            alpha_samples=training.alpha_samples,
            omega_damping=training.omega_damping,
            alpha_damping=training.alpha_damping,
            mixture_weight_damping=training.mixture_weight_damping,
            mixture_weight_dirichlet_concentration=(
                training.mixture_weight_dirichlet_concentration
            ),
            balanced_cycles=training.balanced_cycles,
            validation_samples=training.validation_samples,
            fixed_alpha=training.fixed_alpha,
        )

    def _local_inference(self) -> LocalWishartInference:
        return LocalWishartInference(self.fabric, self.decoder)

    def _trainer(self, context: ExperimentContext) -> ActiveTrainer:
        training = self.config.training
        return ActiveTrainer(
            self.fabric,
            self.decoder,
            context.evaluator,
            self._local_inference(),
            PopulationMstep(),
            AlphaMstep(self.decoder),
            ResponsibilityUpdater(),
            batch_size=context.batch_size,
            effective_batch_size=training.effective_batch_size,
            cache_batch_size=context.evaluation_batch_size,
            path_shard_size=training.path_shard_size,
            em_batch_size=training.em_batch_size,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            gradient_clip=training.gradient_clip,
            reduce_lr_on_plateau=training.reduce_lr_on_plateau,
            lr_plateau_factor=training.lr_plateau_factor,
            lr_plateau_patience=training.lr_plateau_patience,
            lr_plateau_min_lr=training.lr_plateau_min_lr,
        )

    def _fit(
        self,
        context: ExperimentContext,
        shared: SharedTraining,
        *,
        cycle_checkpoint_path: str | Path | None = None,
    ) -> _WishartTraining:
        model = self._expanded_model_from_shared(context.dataset, shared)
        trainer = self._trainer(context)
        model, optimizer = trainer.setup(model)
        self._seed_neural_randomness()
        checkpoint, history = trainer.fit(
            model,
            optimizer,
            context.split.train,
            context.split.validation,
            self._schedule(),
            optimization_seed=self.config.runtime.optimization_seed,
            monte_carlo_seed=self.config.runtime.monte_carlo_seed,
            checkpoint_path=cycle_checkpoint_path,
        )
        # The active cycle checkpoint retains the final optimizer/model/RNG
        # state for exact continuation.  Evaluation, however, must use one
        # coherent validation-selected state: neural weights, Omega, mixture
        # weights and alpha from the same cycle.  Mixing final neural weights
        # with the best cycle's statistical parameters is not a valid model.
        model.load_state_dict(checkpoint.model_state)
        return _WishartTraining(shared, model, history, checkpoint)

    def _posterior_clustering(
        self, context: ExperimentContext, training: _WishartTraining
    ) -> dict[str, float]:
        config = self.config.training
        checkpoint = training.checkpoint
        population_df = (
            self.config.training.population_df
            if checkpoint.population_df is None
            else checkpoint.population_df
        )
        omega = checkpoint.population_means.to(training.model.device)
        log_weights = checkpoint.log_weights.to(training.model.device)
        self.fabric.print(
            f"[wishart] stage=posterior_inference steps={config.test_local_steps}",
            flush=True,
        )
        shard_size = config.path_shard_size or len(context.split.test.sequences)
        probability_parts = []
        diagnostics: list[tuple[int, bool, float, float, float]] = []
        for start in range(0, len(context.split.test.sequences), shard_size):
            stop = min(start + shard_size, len(context.split.test.sequences))
            shard = context.split.test.select(np.arange(start, stop))
            cache = TraceCacheBuilder(context.evaluation_batch_size).build(
                training.model, shard
            )
            posterior = self._local_inference().fit(
                cache,
                omega,
                population_df,
                alpha=checkpoint.alpha,
                steps=config.test_local_steps,
                samples=config.test_local_samples,
                evaluation_samples=config.test_local_evaluation_samples,
                seed=(
                    self.config.runtime.optimization_seed
                    + 91_000_009
                    + start * 1_000_003
                ),
                adaptive_tolerance=(
                    config.test_local_tolerance if config.test_local_adaptive else None
                ),
                adaptive_kappa_tolerance=config.test_local_kappa_tolerance,
                adaptive_gradient_tolerance=config.test_local_gradient_tolerance,
                adaptive_minimum_steps=min(200, config.test_local_steps),
                adaptive_check_interval=config.test_local_check_interval,
                adaptive_patience=config.test_local_patience,
                adaptive_monitor_samples=min(
                    config.test_local_monitor_samples,
                    config.test_local_evaluation_samples,
                ),
                adaptive_learning_rate_floor=config.test_local_learning_rate_floor,
            )
            probability_parts.append(
                torch.softmax(log_weights[None] - posterior.free_energy, dim=1).cpu()
            )
            diagnostics.append(
                (
                    posterior.steps_taken,
                    posterior.converged,
                    posterior.objective_relative_change,
                    posterior.kappa_relative_change,
                    posterior.gradient_norm_p95,
                )
            )
            del cache, posterior
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.fabric.print(
                f"[wishart-test-local] paths={stop}/{len(context.split.test.sequences)}",
                flush=True,
            )
        probabilities = torch.cat(probability_parts)
        summary = clustering_summary(context.split.test.labels, probabilities.numpy())
        summary.update(
            {
                "local_steps_taken": max(item[0] for item in diagnostics),
                "local_converged": all(item[1] for item in diagnostics),
                "objective_relative_change": max(item[2] for item in diagnostics),
                "kappa_relative_change": max(item[3] for item in diagnostics),
                "gradient_norm_p95": max(item[4] for item in diagnostics),
            }
        )
        return summary

    def _evaluate(
        self, context: ExperimentContext, training: _WishartTraining
    ) -> _WishartTest:
        config = self.config.training
        checkpoint = training.checkpoint
        population_df = (
            config.population_df
            if checkpoint.population_df is None
            else checkpoint.population_df
        )
        omega = checkpoint.population_means.to(training.model.device)
        log_weights = checkpoint.log_weights.to(training.model.device)

        # Prior-predictive metrics use no trajectory-specific test adaptation.
        self.fabric.print(
            f"[wishart] stage=prior_test samples={config.test_samples} "
            f"repeats={config.test_monte_carlo_repeats}",
            flush=True,
        )
        prior = context.evaluator.active(
            training.model,
            context.split.test,
            omega,
            log_weights,
            population_df=population_df,
            alpha=checkpoint.alpha,
            samples=config.test_samples,
            seed=self.config.runtime.monte_carlo_seed + 90_000_007,
            repeats=config.test_monte_carlo_repeats,
        )
        return _WishartTest(prior, self._posterior_clustering(context, training))

    def _training_only_path(self, name: str):
        return (
            self.config.runtime.output_root
            / self.method
            / name
            / "wishart_training_checkpoint.pt"
        )

    def _load_shared(self, name: str) -> SharedTraining | None:
        output = self._training_only_path(name).parent
        state_path = output / "shared_checkpoint.pt"
        history_path = output / "shared_history.csv"
        if not state_path.is_file() or not history_path.is_file():
            return None
        self.fabric.print(
            f"[wishart] stage=shared_checkpoint_resumed path={state_path}",
            flush=True,
        )
        return SharedTraining(
            torch.load(state_path, map_location="cpu", weights_only=True),
            pd.read_csv(history_path).to_dict(orient="records"),
        )

    def _active_cycle_path(self, name: str):
        return self._training_only_path(name).with_name("active_cycle_checkpoint.pt")

    def _save_training_only(self, name: str, training: _WishartTraining) -> None:
        output = self._training_only_path(name)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "shared_state": training.shared.state,
                "shared_history": training.shared.history,
                "model_state": clone_state_dict(training.model),
                "history": training.history,
                "cycle": training.checkpoint.cycle,
                "validation_nll": training.checkpoint.validation_nll,
                "population_means": training.checkpoint.population_means,
                "log_weights": training.checkpoint.log_weights,
                "alpha": training.checkpoint.alpha,
                "population_df": training.checkpoint.population_df,
            },
            output,
        )
        self.fabric.print(
            f"[wishart] stage=training_checkpoint_saved path={output}",
            flush=True,
        )

    def _load_training_only(
        self, name: str, context: ExperimentContext
    ) -> _WishartTraining | None:
        path = self._training_only_path(name)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = self._model(context.dataset)
        model.load_state_dict(payload["model_state"])
        model = self.fabric.setup(model)
        checkpoint = ActiveCheckpoint(
            int(payload["cycle"]),
            float(payload["validation_nll"]),
            payload["model_state"],
            payload["population_means"],
            payload["log_weights"],
            float(payload["alpha"]),
            (
                float(payload["population_df"])
                if payload.get("population_df") is not None
                else None
            ),
        )
        self.fabric.print(
            f"[wishart] stage=training_checkpoint_resumed cycle={checkpoint.cycle}",
            flush=True,
        )
        return _WishartTraining(
            SharedTraining(payload["shared_state"], payload["shared_history"]),
            model,
            payload["history"],
            checkpoint,
        )

    def _execute_dataset(
        self, name: str, context: ExperimentContext, started: float
    ) -> dict[str, object]:
        # Wishart uses the shared initialization without fitting a Pure branch.
        self.fabric.print(
            f"[wishart] dataset={name} stage=shared_pretrain "
            f"batch_size={self.config.training.shared_batch_size or context.batch_size} "
            "evaluation_batch_size="
            f"{self.config.training.shared_evaluation_batch_size or context.evaluation_batch_size} "
            f"active_batch_size={context.batch_size}",
            flush=True,
        )
        training = self._load_training_only(name, context)
        if training is None:
            shared = self._load_shared(name)
            if shared is None:
                shared = self._train_shared(context)
                self.artifacts.write_shared(
                    self._training_only_path(name).parent,
                    shared.state,
                    shared.history,
                )
            self.fabric.print(
                f"[wishart] dataset={name} stage=active_training", flush=True
            )
            training = self._fit(
                context,
                shared,
                cycle_checkpoint_path=self._active_cycle_path(name),
            )
            self._save_training_only(name, training)
        test = self._evaluate(context, training)
        result = self._base_result(name, context, time.perf_counter() - started)
        result.update(
            {
                "best_cycle": training.checkpoint.cycle,
                "alpha": training.checkpoint.alpha,
                "alpha_is_fixed": self.config.training.fixed_alpha is not None,
                "population_df": (
                    self.config.training.population_df
                    if training.checkpoint.population_df is None
                    else training.checkpoint.population_df
                ),
                "population_df_is_learned": (self.config.training.learn_population_df),
                "prior_test": test.prior.as_dict(),
                "posterior_clustering": test.posterior_clustering,
            }
        )

        self.artifacts.write_wishart(
            self.config.runtime.output_root / self.method / name,
            shared_state=training.shared.state,
            wishart_state=clone_state_dict(training.model),
            checkpoint=training.checkpoint,
            shared_history=training.shared.history,
            wishart_history=training.history,
            result=result,
        )
        self._print_summary(
            name, test.prior.nll_per_exposure, training.checkpoint.alpha
        )
        return result
