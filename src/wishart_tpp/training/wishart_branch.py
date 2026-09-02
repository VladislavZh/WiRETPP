"""Training of the matched fixed-q Active-Wishart branch."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wishart_tpp.data import DatasetPartition
from wishart_tpp.training.artifact_io import write_csv, write_json, write_torch
from wishart_tpp.training.experiment import ExperimentContext, SharedTraining
from wishart_tpp.training.state import clone_state_dict
from wishart_tpp.training.validation_curves import ValidationCurveWriter
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner


class RecordingWishartEvaluator:
    """Delegate evaluation while atomically recording each active cycle."""

    def __init__(
        self,
        base,
        active_writer: ValidationCurveWriter,
        backbone_writer: ValidationCurveWriter,
        validation: DatasetPartition,
        start_cycle: int,
        updates_per_cycle: int,
    ) -> None:
        self.base = base
        self.active_writer = active_writer
        self.backbone_writer = backbone_writer
        self.validation = validation
        self.next_cycle = start_cycle
        self.updates_per_cycle = updates_per_cycle
        self.pending_cycle: int | None = None

    def active(self, model, partition, population_means, log_weights, **kwargs):
        evaluation = self.base.active(
            model, partition, population_means, log_weights, **kwargs
        )
        cycle = self.next_cycle
        weights = log_weights.detach().cpu().exp().numpy()
        positive = weights[weights > 0.0]
        self.active_writer.record(
            cycle,
            cycle * self.updates_per_cycle,
            evaluation,
            partition,
            extras={
                "alpha": float(kwargs["alpha"]),
                "population_df": float(kwargs["population_df"]),
                "effective_k": float(np.exp(-(positive * np.log(positive)).sum())),
                **{
                    f"mixture_weight_{index}": float(value)
                    for index, value in enumerate(weights)
                },
            },
        )
        self.pending_cycle = cycle
        self.next_cycle += 1
        return evaluation

    def pure(self, model, partition):
        evaluation = self.base.pure(model, partition)
        if self.pending_cycle is not None:
            self.backbone_writer.record(
                self.pending_cycle,
                self.pending_cycle * self.updates_per_cycle,
                evaluation,
                partition,
            )
            self.pending_cycle = None
        return evaluation


def fit_wishart_branch(
    runner: WishartExperimentRunner,
    context: ExperimentContext,
    shared: SharedTraining,
    output: Path,
) -> dict[str, Any]:
    complete_path = output / "training_complete.json"
    if complete_path.is_file():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("complete"):
            print("[wishart_k5] stage=reused", flush=True)
            return payload
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "active_cycle_checkpoint.pt"
    completed_cycle = 0
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        completed_cycle = int(payload["completed_cycle"])
    active_writer = ValidationCurveWriter(
        output,
        dataset=context.dataset.name,
        seed=runner.config.runtime.optimization_seed,
        method="wishart_k5",
    )
    backbone_writer = ValidationCurveWriter(
        output,
        dataset=context.dataset.name,
        seed=runner.config.runtime.optimization_seed,
        method="wishart_backbone_k5",
    )
    required = set(range(completed_cycle + 1))
    if completed_cycle and not required.issubset(active_writer.completed_cycles()):
        raise RuntimeError("Wishart checkpoint is missing active validation curves")
    recorder = RecordingWishartEvaluator(
        context.evaluator,
        active_writer,
        backbone_writer,
        context.split.validation,
        completed_cycle + 1,
        runner.config.training.neural_steps_per_cycle,
    )
    recording_context = ExperimentContext(
        context.dataset,
        context.split,
        context.batch_size,
        context.evaluation_batch_size,
        recorder,
    )
    if completed_cycle == 0 and 0 not in active_writer.completed_cycles():
        model = runner._expanded_model_from_shared(context.dataset, shared)
        model = runner.fabric.setup(model)
        trainer = runner._trainer(context)
        state = trainer._initial_state(
            model, runner._schedule(), runner.config.runtime.optimization_seed
        )
        initial = context.evaluator.active(
            model,
            context.split.validation,
            state.population_means,
            state.log_weights,
            population_df=state.population_df,
            alpha=state.alpha,
            samples=runner.config.training.validation_samples,
            seed=runner.config.runtime.monte_carlo_seed + 40_009,
        )
        weights = state.log_weights.detach().cpu().exp().numpy()
        positive = weights[weights > 0.0]
        active_writer.record(
            0,
            0,
            initial,
            context.split.validation,
            extras={
                "alpha": state.alpha,
                "population_df": state.population_df,
                "effective_k": float(np.exp(-(positive * np.log(positive)).sum())),
                **{
                    f"mixture_weight_{index}": float(value)
                    for index, value in enumerate(weights)
                },
            },
        )
        backbone_writer.record(
            0,
            0,
            context.evaluator.pure(model, context.split.validation),
            context.split.validation,
        )
        del model, trainer, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    started = time.perf_counter()
    training = runner._fit(
        recording_context,
        shared,
        cycle_checkpoint_path=checkpoint_path,
    )
    write_csv(output / "history.csv", pd.DataFrame(training.history))
    checkpoint = training.checkpoint
    population_df = float(
        runner.config.training.population_df
        if checkpoint.population_df is None
        else checkpoint.population_df
    )
    write_torch(
        output / "selected_checkpoint.pt",
        {
            "method": "wishart_k5",
            "components": 5,
            "model_state": clone_state_dict(training.model),
            "population_means": checkpoint.population_means,
            "log_weights": checkpoint.log_weights,
            "alpha": checkpoint.alpha,
            "population_df": population_df,
            "best_cycle": checkpoint.cycle,
            "selection_validation_nll": checkpoint.validation_nll,
        },
    )
    device = getattr(training.model, "module", training.model).device
    selected = context.evaluator.active(
        training.model,
        context.split.validation,
        checkpoint.population_means.to(device),
        checkpoint.log_weights.to(device),
        population_df=population_df,
        alpha=checkpoint.alpha,
        samples=runner.config.training.test_samples,
        repeats=runner.config.training.test_monte_carlo_repeats,
        seed=runner.config.runtime.monte_carlo_seed + 121_000_057,
    )
    write_csv(
        output / "selected_validation_paths.csv",
        pd.DataFrame(
            {
                "source_id": context.split.validation.source_ids,
                "horizon": [
                    item.horizon for item in context.split.validation.sequences
                ],
                "log_likelihood": selected.marginal_scores.detach().cpu().numpy(),
            }
        ),
    )
    selected_row = next(
        row for row in training.history if int(row["cycle"]) == checkpoint.cycle
    )
    masses = np.asarray(
        [selected_row[f"responsibility_mass_{index}"] for index in range(5)],
        dtype=np.float64,
    )
    proportions = masses / masses.sum()
    positive = proportions[proportions > 0.0]
    result = {
        "complete": True,
        "method": "wishart_k5",
        "components": 5,
        "active_cycles": runner.config.training.active_cycles,
        "neural_updates_after_shared": (
            runner.config.training.active_cycles
            * runner.config.training.neural_steps_per_cycle
        ),
        "best_cycle": checkpoint.cycle,
        "selection_validation_nll": checkpoint.validation_nll,
        "selected_robust_validation_nll": selected.nll_per_exposure,
        "alpha": checkpoint.alpha,
        "population_df": population_df,
        "mixture_weights": checkpoint.log_weights.exp().tolist(),
        "selected_occupancy": {
            "masses": masses.tolist(),
            "proportions": proportions.tolist(),
            "effective_k": float(np.exp(-(positive * np.log(positive)).sum())),
            "live_components_5pct": int((proportions >= 0.05).sum()),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "test_read": False,
    }
    write_json(complete_path, result)
    print(
        f"[wishart_k5] stage=training_complete best_cycle={checkpoint.cycle} "
        f"selection_nll={checkpoint.validation_nll:.6f} "
        f"robust_validation_nll={selected.nll_per_exposure:.6f} "
        f"alpha={checkpoint.alpha:.6f} df={population_df:.6f}",
        flush=True,
    )
    del training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
