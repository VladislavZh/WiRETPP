from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.backbones.integration import GaussLegendreRule
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.inference.alpha import AlphaMstep
from wishart_tpp.inference.local import LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.active import ActiveSchedule, ActiveTrainer
from wishart_tpp.training.evaluation import ModelEvaluator
from wishart_tpp.training.state import clone_state_dict


class ActiveResumeTest(unittest.TestCase):
    @staticmethod
    def _partition() -> tuple[DatasetPartition, DatasetPartition]:
        sequences = tuple(
            EventSequence(
                np.array([0.2 + 0.02 * index, 0.7]),
                np.array([index % 2, (index + 1) % 2]),
                1.0,
            )
            for index in range(6)
        )
        train = DatasetPartition(sequences[:4], np.array([0, 1, 0, 1]), np.arange(4))
        validation = DatasetPartition(sequences[4:], np.array([0, 1]), np.arange(4, 6))
        return train, validation

    @staticmethod
    def _model() -> CoticIntensityBank:
        model = CoticIntensityBank(
            1,
            2,
            input_channels=4,
            hidden_size=8,
            layers=1,
            kernel_size=2,
            dropout=0.0,
            integration_rule=GaussLegendreRule(2),
        )
        model.expand_components(2, noise=0.02, seed=2)
        return model

    @staticmethod
    def _assert_tree_equal(left, right) -> None:
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                ActiveResumeTest._assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                ActiveResumeTest._assert_tree_equal(left_item, right_item)
        else:
            assert left == right

    def _trainer_and_model(self, initial_state):
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = ActiveTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )
        model = self._model()
        model.load_state_dict(initial_state)
        model, optimizer = trainer.setup(model)
        return trainer, model, optimizer

    def test_cycle_checkpoint_resumes_exact_optimizer_and_em_state(self) -> None:
        train, validation = self._partition()
        initial_state = clone_state_dict(self._model())
        schedule = ActiveSchedule(
            cycles=2,
            neural_steps=1,
            local_steps=1,
            local_samples=1,
            local_evaluation_samples=1,
            population_df=5.0,
            initial_alpha=0.3,
            alpha_steps=2,
            alpha_samples=1,
            balanced_cycles=1,
            validation_samples=2,
            learn_population_df=True,
            population_df_warmup_cycles=1,
            population_df_profile_points=5,
            population_df_max=16.0,
        )
        with TemporaryDirectory() as directory:
            continuous_path = Path(directory) / "continuous.pt"
            resumed_path = Path(directory) / "resumed.pt"
            trainer, model, optimizer = self._trainer_and_model(initial_state)
            torch.manual_seed(91)
            continuous_best, continuous_history = trainer.fit(
                model,
                optimizer,
                train,
                validation,
                schedule,
                optimization_seed=7,
                monte_carlo_seed=11,
                checkpoint_path=continuous_path,
            )
            trainer, model, optimizer = self._trainer_and_model(initial_state)
            torch.manual_seed(91)
            trainer.fit(
                model,
                optimizer,
                train,
                validation,
                replace(schedule, cycles=1),
                optimization_seed=7,
                monte_carlo_seed=11,
                checkpoint_path=resumed_path,
            )
            trainer, model, optimizer = self._trainer_and_model(initial_state)
            resumed_best, resumed_history = trainer.fit(
                model,
                optimizer,
                train,
                validation,
                schedule,
                optimization_seed=7,
                monte_carlo_seed=11,
                checkpoint_path=resumed_path,
            )
            continuous = torch.load(
                continuous_path, map_location="cpu", weights_only=False
            )
            resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
            self.assertEqual(continuous_history, resumed_history)
            self.assertEqual(continuous_best.cycle, resumed_best.cycle)
            self._assert_tree_equal(continuous["model_state"], resumed["model_state"])
            self._assert_tree_equal(
                continuous["optimizer_state"], resumed["optimizer_state"]
            )
            self._assert_tree_equal(continuous["active_state"], resumed["active_state"])


if __name__ == "__main__":
    unittest.main()
