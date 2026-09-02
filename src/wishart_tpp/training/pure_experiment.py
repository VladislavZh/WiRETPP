"""Standalone Pure-mixture experiment runner."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import torch

from wishart_tpp.backbones import IntensityBank
from wishart_tpp.training.experiment import (
    ExperimentContext,
    ExperimentRunner,
    SharedTraining,
)
from wishart_tpp.training.pure import PureSchedule
from wishart_tpp.training.state import PureCheckpoint, clone_state_dict


@dataclass(frozen=True)
class _PureTraining:
    shared: SharedTraining
    model: IntensityBank
    history: list[dict[str, float]]
    checkpoint: PureCheckpoint


class PureExperimentRunner(ExperimentRunner):
    """Fit and evaluate only the ordinary finite TPP mixture."""

    @property
    def method(self) -> str:
        return "pure"

    def _fit(self, context: ExperimentContext, shared: SharedTraining) -> _PureTraining:
        model = self._expanded_model_from_shared(context.dataset, shared)
        trainer = self._pure_trainer(context)
        model, optimizer = trainer.setup(model)
        self._seed_neural_randomness()
        schedule = PureSchedule(
            cycles=self.config.training.active_cycles,
            neural_steps=self.config.training.neural_steps_per_cycle,
            balanced_cycles=self.config.training.balanced_cycles,
        )
        if schedule.total_steps != self.config.training.pure_steps:
            raise ValueError(
                "pure_steps must equal active_cycles * neural_steps_per_cycle"
            )
        history, checkpoint = trainer.fit_cycles(
            model,
            optimizer,
            context.split.train,
            context.split.validation,
            schedule,
            seed=self.config.runtime.optimization_seed + 107,
        )
        return _PureTraining(shared, model, history, checkpoint)

    def _shared_training(self, context: ExperimentContext) -> SharedTraining:
        root = self.config.runtime.shared_checkpoint_root
        if root is None:
            return self._train_shared(context)
        source = root / context.dataset.name
        state = torch.load(
            source / "shared_checkpoint.pt", map_location="cpu", weights_only=True
        )
        history = pd.read_csv(source / "shared_history.csv").to_dict("records")
        return SharedTraining(state, history)

    def _execute_dataset(
        self, name: str, context: ExperimentContext, started: float
    ) -> dict[str, object]:
        # Pure continues directly from the common short initialization.
        training = self._fit(context, self._shared_training(context))
        selected_validation = context.evaluator.pure(
            training.model, context.split.validation
        )
        test = context.evaluator.pure(training.model, context.split.test)
        weights = training.model.mixture_log_weights().detach().cpu().exp()
        positive = weights[weights > 0.0]
        result = self._base_result(name, context, time.perf_counter() - started)
        result.update(
            {
                "best_step": training.checkpoint.step,
                "best_cycle": (
                    training.checkpoint.step
                    // self.config.training.neural_steps_per_cycle
                ),
                "selection_validation_nll": training.checkpoint.validation_nll,
                "selected_validation": selected_validation.as_dict(),
                "test": test.as_dict(),
                "mixture_weights": weights.tolist(),
                "effective_k": float(
                    (-(positive * positive.log()).sum()).exp()
                ),
                "balanced_cycles": self.config.training.balanced_cycles,
                "balanced_updates": (
                    self.config.training.balanced_cycles
                    * self.config.training.neural_steps_per_cycle
                ),
                "selection": "minimum validation NLL only",
                "shared_checkpoint_source": (
                    str(self.config.runtime.shared_checkpoint_root / name)
                    if self.config.runtime.shared_checkpoint_root is not None
                    else "trained_in_run"
                ),
                "selected_state_matches_checkpoint": True,
                "test_read": True,
            }
        )

        self.artifacts.write_pure(
            self.config.runtime.output_root / self.method / name,
            shared_state=training.shared.state,
            pure_state=clone_state_dict(training.model),
            shared_history=training.shared.history,
            pure_history=training.history,
            result=result,
        )
        self._print_summary(name, test.nll_per_exposure)
        return result
