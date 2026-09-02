"""Ordinary finite-mixture backbone training with an explicit Fabric loop."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.data import DatasetPartition
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.evaluation import ModelEvaluator
from wishart_tpp.training.state import PureCheckpoint, clone_state_dict


@dataclass(frozen=True)
class PureSchedule:
    """Describe cycle-aligned Pure training and its balanced warm-up."""

    cycles: int
    neural_steps: int
    balanced_cycles: int

    def __post_init__(self) -> None:
        if self.cycles < 1 or self.neural_steps < 1:
            raise ValueError("Pure cycles and neural steps must be positive")
        if not 0 <= self.balanced_cycles <= self.cycles:
            raise ValueError("Pure balanced cycles must lie in [0, cycles]")

    @property
    def total_steps(self) -> int:
        return self.cycles * self.neural_steps


class PureMixtureTrainer:
    def __init__(
        self,
        fabric: Fabric,
        decoder: ActiveBlockDecoder,
        evaluator: ModelEvaluator,
        *,
        batch_size: int,
        trace_batch_size: int,
        effective_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        reduce_lr_on_plateau: bool = False,
        lr_plateau_factor: float = 0.5,
        lr_plateau_patience: int = 5,
        lr_plateau_min_lr: float = 0.0,
        responsibility_updater: ResponsibilityUpdater | None = None,
    ) -> None:
        self.fabric = fabric
        self.decoder = decoder
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.trace_batch_size = trace_batch_size
        self.effective_batch_size = effective_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_plateau_factor = lr_plateau_factor
        self.lr_plateau_patience = lr_plateau_patience
        self.lr_plateau_min_lr = lr_plateau_min_lr
        self.responsibility_updater = responsibility_updater or ResponsibilityUpdater()

    def setup(self, model):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
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

    def _sample_indices(
        self, random: np.random.Generator, train: DatasetPartition
    ) -> np.ndarray:
        return random.choice(
            len(train.sequences),
            min(self.effective_batch_size, len(train.sequences)),
            replace=False,
        )

    def _training_step(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        indices: np.ndarray,
        responsibilities: torch.Tensor | None = None,
    ) -> float:
        effective = train.select(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        # Accumulate one effective batch through memory-sized microbatches.
        for start in range(0, len(indices), self.batch_size):
            micro = train.select(indices[start : start + self.batch_size])
            score = model.mixture_logits.new_zeros(())

            # Preserve the historical COTIC stochastic path: a physical
            # optimizer microbatch is itself traced in smaller chunks.
            for trace_start in range(0, len(micro.sequences), self.trace_batch_size):
                sequences = micro.sequences[
                    trace_start : trace_start + self.trace_batch_size
                ]
                component = self.decoder.base_component_scores(model(sequences))
                logits = component + model.mixture_log_weights()[None]
                if responsibilities is None:
                    score = score + torch.logsumexp(logits, dim=1).sum()
                else:
                    selected = responsibilities[
                        indices[
                            start + trace_start : start + trace_start + len(sequences)
                        ]
                    ].to(logits.device)
                    entropy = -(selected * selected.clamp_min(1e-30).log()).sum(1)
                    score = score + ((selected * logits).sum(1) + entropy).sum()
            loss = -score / effective.exposure
            self.fabric.backward(loss)
            loss_value += float(loss.detach().cpu())

        self.fabric.clip_gradients(model, optimizer, max_norm=self.gradient_clip)
        optimizer.step()
        return loss_value

    def _balanced_responsibilities(
        self, model, train: DatasetPartition
    ) -> torch.Tensor:
        evaluation = self.evaluator.pure(model, train)
        log_weights = model.mixture_log_weights().detach().cpu()
        return self.responsibility_updater.update(
            -evaluation.component_scores, log_weights, balanced=True
        )

    def _validate(
        self, model, validation: DatasetPartition, step: int
    ) -> PureCheckpoint:
        evaluation = self.evaluator.pure(model, validation)
        return PureCheckpoint(
            evaluation.nll_per_exposure, step, clone_state_dict(model)
        )

    def fit(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        validation: DatasetPartition,
        *,
        steps: int,
        seed: int,
        select_best: bool,
    ) -> tuple[list[dict[str, float]], PureCheckpoint]:
        random = np.random.default_rng(seed)
        best = self._validate(model, validation, step=0)
        history = []
        scheduler = self._lr_scheduler(optimizer)
        validation_interval = max(1, steps // 12)
        started = time.perf_counter()
        self.fabric.print(
            f"[pure] step=0/{steps} elapsed_seconds=0 eta_seconds=unknown "
            f"validation_nll={best.validation_nll:.6f} best_step=0",
            flush=True,
        )

        for step in range(1, steps + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            indices = self._sample_indices(random, train)
            loss_value = self._training_step(model, optimizer, train, indices)
            row = {
                "step": step,
                "train_nll_per_exposure": loss_value,
                "learning_rate": learning_rate,
            }

            # Validation is sparse because each pass traces the complete split.
            if step == steps or step % validation_interval == 0:
                candidate = self._validate(model, validation, step)
                row["validation_nll_per_exposure"] = candidate.validation_nll
                if candidate.validation_nll < best.validation_nll:
                    best = candidate
                if scheduler is not None:
                    scheduler.step(candidate.validation_nll)
                next_learning_rate = float(optimizer.param_groups[0]["lr"])
                row["next_learning_rate"] = next_learning_rate
                row["lr_reduced"] = float(next_learning_rate < learning_rate)
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (steps - step)
                self.fabric.print(
                    f"[pure] step={step}/{steps} elapsed_seconds={elapsed:.1f} "
                    f"eta_seconds={eta:.1f} train_nll={loss_value:.6f} "
                    f"validation_nll={candidate.validation_nll:.6f} "
                    f"lr={learning_rate:.8g} next_lr={next_learning_rate:.8g} "
                    f"best_step={best.step}",
                    flush=True,
                )
            history.append(row)

        if select_best:
            model.load_state_dict(best.model_state)
        return history, best

    def fit_cycles(
        self,
        model,
        optimizer,
        train: DatasetPartition,
        validation: DatasetPartition,
        schedule: PureSchedule,
        *,
        seed: int,
    ) -> tuple[list[dict[str, float]], PureCheckpoint]:
        """Fit Pure in validation-aligned cycles with an optional balanced E-step."""

        random = np.random.default_rng(seed)
        initial = self.evaluator.pure(model, validation)
        best = PureCheckpoint(initial.nll_per_exposure, 0, clone_state_dict(model))
        scheduler = self._lr_scheduler(optimizer)
        history = [self._cycle_row(model, initial, 0, 0, None, best.step)]
        started = time.perf_counter()
        self._print_cycle(0, schedule, history[0], 0.0, float("nan"))

        for cycle in range(1, schedule.cycles + 1):
            balanced = cycle <= schedule.balanced_cycles
            responsibilities = (
                self._balanced_responsibilities(model, train) if balanced else None
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            losses = []
            for _ in range(schedule.neural_steps):
                indices = self._sample_indices(random, train)
                losses.append(
                    self._training_step(
                        model, optimizer, train, indices, responsibilities
                    )
                )

            step = cycle * schedule.neural_steps
            evaluation = self.evaluator.pure(model, validation)
            candidate = PureCheckpoint(
                evaluation.nll_per_exposure, step, clone_state_dict(model)
            )
            if candidate.validation_nll < best.validation_nll:
                best = candidate
            if scheduler is not None:
                scheduler.step(candidate.validation_nll)
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            row = self._cycle_row(
                model, evaluation, cycle, step, responsibilities, best.step
            )
            row.update(
                {
                    "train_nll_per_exposure": float(np.mean(losses)),
                    "learning_rate": learning_rate,
                    "next_learning_rate": next_learning_rate,
                    "lr_reduced": float(next_learning_rate < learning_rate),
                }
            )
            history.append(row)
            elapsed = time.perf_counter() - started
            eta = elapsed / cycle * (schedule.cycles - cycle)
            self._print_cycle(cycle, schedule, row, elapsed, eta)

        model.load_state_dict(best.model_state)
        return history, best

    @staticmethod
    def _cycle_row(
        model,
        evaluation,
        cycle: int,
        step: int,
        responsibilities: torch.Tensor | None,
        best_step: int,
    ) -> dict[str, float]:
        weights = model.mixture_log_weights().detach().cpu().exp().numpy()
        positive = weights[weights > 0.0]
        row = {
            "cycle": float(cycle),
            "step": float(step),
            "neural_updates_after_shared": float(step),
            "balanced": float(responsibilities is not None),
            "validation_nll_per_exposure": evaluation.nll_per_exposure,
            "validation_nll_per_event": evaluation.nll_per_event,
            "validation_purity": evaluation.purity,
            "validation_ari": evaluation.ari,
            "best_step_so_far": float(best_step),
            "effective_k": float(np.exp(-(positive * np.log(positive)).sum())),
        }
        row.update(
            {f"mixture_weight_{index}": float(value) for index, value in enumerate(weights)}
        )
        if responsibilities is not None:
            masses = responsibilities.sum(0).numpy()
            target = responsibilities.shape[0] / responsibilities.shape[1]
            row["balance_max_relative_mass_error"] = float(
                np.max(np.abs(masses - target)) / target
            )
            row.update(
                {f"responsibility_mass_{index}": float(value) for index, value in enumerate(masses)}
            )
        return row

    def _print_cycle(
        self,
        cycle: int,
        schedule: PureSchedule,
        row: dict[str, float],
        elapsed: float,
        eta: float,
    ) -> None:
        self.fabric.print(
            f"[pure] cycle={cycle}/{schedule.cycles} "
            f"updates={int(row['step'])}/{schedule.total_steps} "
            f"balanced={bool(row['balanced'])} "
            f"validation_nll={row['validation_nll_per_exposure']:.6f} "
            f"purity={row['validation_purity']:.6f} ari={row['validation_ari']:.6f} "
            f"balance_error={row.get('balance_max_relative_mass_error', float('nan')):.6g} "
            f"best_step={int(row['best_step_so_far'])} "
            f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}",
            flush=True,
        )
