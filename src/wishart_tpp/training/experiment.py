"""Abstract orchestration shared by all experiment runners."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from lightning.fabric import Fabric
from torch import Tensor

from wishart_tpp.backbones import IntensityBank
from wishart_tpp.backbones.factory import BackboneFactory
from wishart_tpp.config import ExperimentConfig
from wishart_tpp.data import DatasetSplit, EventDataModule, EventDataset
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.artifacts import RunArtifactWriter
from wishart_tpp.training.evaluation import ModelEvaluator
from wishart_tpp.training.pure import PureMixtureTrainer
from wishart_tpp.training.state import clone_state_dict


@dataclass(frozen=True)
class ExperimentContext:
    dataset: EventDataset
    split: DatasetSplit
    batch_size: int
    evaluation_batch_size: int
    evaluator: ModelEvaluator


@dataclass(frozen=True)
class SharedTraining:
    state: dict[str, Tensor]
    history: list[dict[str, float]]


class ExperimentRunner(ABC):
    """Template method for one independently configured experiment family."""

    def __init__(self, config: ExperimentConfig, fabric: Fabric) -> None:
        if config.training.method != self.method:
            raise ValueError(
                f"{type(self).__name__} requires training.method={self.method!r}"
            )
        self.config = config
        self.fabric = fabric
        self.data = EventDataModule(
            config.runtime.data_root,
            config.runtime.split_seed,
            mixture_components=config.runtime.mixture_components or 5,
            split_protocol=config.runtime.real_split_protocol,
            packed_filename=config.runtime.packed_filename,
        )
        self.decoder = ActiveBlockDecoder(
            config.model.wishart_intensity_floor,
            config.model.wishart_intensity_floor_scaling,
            config.training.evaluation_sample_shard_size,
        )
        self.artifacts = RunArtifactWriter()

    @property
    @abstractmethod
    def method(self) -> str:
        """Stable method name used in reports and artifact paths."""

    @staticmethod
    def _batch_sizes(dataset: EventDataset, configured: int | None) -> tuple[int, int]:
        maximum = max(sequence.count for sequence in dataset.sequences)
        if configured is not None:
            training = configured
        elif maximum <= 100:
            training = 64
        elif maximum <= 250:
            training = 24
        elif maximum <= 500:
            training = 12
        else:
            training = 8
        if maximum <= 100:
            evaluation = 32
        elif maximum <= 250:
            evaluation = 16
        elif maximum <= 500:
            evaluation = 8
        else:
            evaluation = 4
        return training, evaluation

    def _model(
        self,
        dataset: EventDataset,
        *,
        n_components: int | None = None,
    ) -> IntensityBank:
        model_config = self.config.model
        return BackboneFactory.create(
            model_config,
            dataset.n_components if n_components is None else n_components,
            dataset.n_marks,
            2026082200 + self.config.runtime.optimization_seed,
            2026082300 + self.config.runtime.optimization_seed,
        )

    def _expanded_model_from_shared(
        self,
        dataset: EventDataset,
        shared: SharedTraining,
    ) -> IntensityBank:
        """Load the fitted K=1 backbone, then create nearby mixture heads."""

        model = self._model(dataset, n_components=1)
        model.load_state_dict(shared.state)
        if dataset.n_components > 1:
            model.expand_components(
                dataset.n_components,
                self.config.model.component_noise,
                2026082300 + self.config.runtime.optimization_seed,
            )
        return model

    @property
    def neural_random_seed(self) -> int:
        configured = self.config.runtime.neural_seed
        return (
            2026082200 + self.config.runtime.optimization_seed
            if configured is None
            else configured
        )

    def _seed_neural_randomness(self) -> None:
        """Make dropout and other global neural randomness explicit."""

        torch.manual_seed(self.neural_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.neural_random_seed)

    def _pure_trainer(self, context: ExperimentContext) -> PureMixtureTrainer:
        training = self.config.training
        return PureMixtureTrainer(
            self.fabric,
            self.decoder,
            context.evaluator,
            batch_size=context.batch_size,
            trace_batch_size=context.evaluation_batch_size,
            effective_batch_size=training.effective_batch_size,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            gradient_clip=training.gradient_clip,
            reduce_lr_on_plateau=training.reduce_lr_on_plateau,
            lr_plateau_factor=training.lr_plateau_factor,
            lr_plateau_patience=training.lr_plateau_patience,
            lr_plateau_min_lr=training.lr_plateau_min_lr,
        )

    def _prepare(self, name: str) -> ExperimentContext:
        prepared = self.data.prepare(
            name,
            self.config.runtime.trajectory_limit,
            self.config.runtime.cohort_seed,
        )
        dataset, split = prepared.dataset, prepared.split
        training = self.config.training
        batch_size, default_evaluation = self._batch_sizes(dataset, training.batch_size)
        evaluation_batch_size = training.evaluation_batch_size or default_evaluation
        return ExperimentContext(
            dataset,
            split,
            batch_size,
            evaluation_batch_size,
            ModelEvaluator(
                self.decoder,
                evaluation_batch_size,
                path_shard_size=training.path_shard_size,
            ),
        )

    def _train_shared(self, context: ExperimentContext) -> SharedTraining:
        context = self._shared_context(context)
        trainer = self._pure_trainer(context)
        # The common initialization is genuinely a one-head COTIC/TPP fit.
        # Mixture heads are created only after this checkpoint has been fitted.
        model, optimizer = trainer.setup(self._model(context.dataset, n_components=1))
        self._seed_neural_randomness()
        history, _ = trainer.fit(
            model,
            optimizer,
            context.split.train,
            context.split.validation,
            steps=self.config.training.shared_pretrain_steps,
            seed=self.config.runtime.optimization_seed + 103,
            select_best=False,
        )
        return SharedTraining(clone_state_dict(model), history)

    def _shared_context(self, context: ExperimentContext) -> ExperimentContext:
        """Apply independent compute shards to the reproducible shared boundary."""

        training = self.config.training
        if training.shared_batch_size is None:
            return context
        evaluation_batch_size = (
            training.shared_evaluation_batch_size or context.evaluation_batch_size
        )
        return ExperimentContext(
            context.dataset,
            context.split,
            training.shared_batch_size,
            evaluation_batch_size,
            ModelEvaluator(
                self.decoder,
                evaluation_batch_size,
                path_shard_size=(
                    training.shared_path_shard_size or training.path_shard_size
                ),
            ),
        )

    def _base_result(
        self,
        name: str,
        context: ExperimentContext,
        elapsed_seconds: float,
    ) -> dict[str, object]:
        return {
            "dataset": name,
            "method": self.method,
            "split_seed": self.config.runtime.split_seed,
            "split_sizes": {
                "train": len(context.split.train.sequences),
                "validation": len(context.split.validation.sequences),
                "test": len(context.split.test.sequences),
            },
            "elapsed_seconds": elapsed_seconds,
            "neural_random_seed": self.neural_random_seed,
            "time_normalization": context.dataset.normalization.as_dict(),
            "config": self.config.as_dict(),
        }

    def _print_summary(
        self, name: str, nll_per_exposure: float, alpha: float | None = None
    ) -> None:
        summary = {
            "dataset": name,
            "method": self.method,
            "nll": nll_per_exposure,
        }
        if alpha is not None:
            summary["alpha"] = alpha
        self.fabric.print(json.dumps(summary))

    @abstractmethod
    def _execute_dataset(
        self, name: str, context: ExperimentContext, started: float
    ) -> dict[str, object]:
        """Train, evaluate, and persist one dataset for this runner family."""

    def run_dataset(self, name: str) -> dict[str, object]:
        started = time.perf_counter()
        return self._execute_dataset(name, self._prepare(name), started)

    def run(self) -> list[dict[str, object]]:
        names = (
            self.data.names("synthetic")
            if self.config.runtime.dataset == "all12"
            else (self.config.runtime.dataset,)
        )
        return [self.run_dataset(name) for name in names]
