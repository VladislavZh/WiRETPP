from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.backbones.integration import GaussLegendreRule
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.inference.alpha import AlphaGridProfile
from wishart_tpp.inference.local import LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.active import ActiveSchedule, ActiveTrainer
from wishart_tpp.training.evaluation import ModelEvaluator


class ActiveControlTest(unittest.TestCase):
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

    def _trainer(self, alpha_mstep, *, path_shard_size=None) -> ActiveTrainer:
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        return ActiveTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            alpha_mstep,
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=2,
            path_shard_size=path_shard_size,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )

    @staticmethod
    def _schedule(**changes) -> ActiveSchedule:
        values = {
            "cycles": 1,
            "neural_steps": 1,
            "local_steps": 1,
            "local_samples": 1,
            "local_evaluation_samples": 1,
            "population_df": 5.0,
            "initial_alpha": 0.3,
            "alpha_steps": 2,
            "alpha_samples": 1,
            "balanced_cycles": 1,
            "validation_samples": 2,
        }
        values.update(changes)
        return ActiveSchedule(**values)

    def test_fixed_alpha_skips_scalar_mstep(self) -> None:
        train, validation = self._partition()
        alpha_mstep = Mock()
        trainer = self._trainer(alpha_mstep)
        model, optimizer = trainer.setup(self._model())
        best, _ = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            self._schedule(fixed_alpha=0.0),
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        self.assertEqual(best.alpha, 0.0)
        alpha_mstep.update.assert_not_called()

    def test_sharded_alpha_profiles_are_combined_before_damping(self) -> None:
        train, validation = self._partition()
        alpha_mstep = Mock()
        values = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64)
        alpha_mstep.candidate_grid.return_value = values
        alpha_mstep.profile.side_effect = lambda cache, posterior, *_a, **_k: (
            AlphaGridProfile(values, torch.zeros(len(posterior.means), 2, len(values)))
        )
        alpha_mstep.select.return_value = 0.7
        trainer = self._trainer(alpha_mstep, path_shard_size=2)
        model, optimizer = trainer.setup(self._model())
        best, _ = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            self._schedule(alpha_damping=0.25),
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        self.assertAlmostEqual(best.alpha, 0.4, places=7)
        self.assertEqual(alpha_mstep.profile.call_count, 2)
        retained_profile = alpha_mstep.select.call_args.args[0]
        self.assertEqual(retained_profile.scores.shape, (4, 2, 4))
        alpha_mstep.update.assert_not_called()

    def test_alpha_warmup_reactivation_then_learning(self) -> None:
        train, validation = self._partition()
        alpha_mstep = Mock()
        values = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64)
        alpha_mstep.candidate_grid.return_value = values
        alpha_mstep.profile.side_effect = lambda cache, posterior, *_a, **_k: (
            AlphaGridProfile(values, torch.zeros(len(posterior.means), 2, len(values)))
        )
        alpha_mstep.select.return_value = 0.7
        trainer = self._trainer(alpha_mstep, path_shard_size=2)
        model, optimizer = trainer.setup(self._model())
        _best, history = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            self._schedule(
                cycles=3,
                alpha_warmup_cycles=1,
                alpha_reactivation_cycles=1,
            ),
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        self.assertEqual([row["alpha"] for row in history], [0.0, 0.3, 0.7])
        alpha_mstep.candidate_grid.assert_called_once()
        self.assertEqual(alpha_mstep.profile.call_count, 2)
        alpha_mstep.select.assert_called_once()
        alpha_mstep.update.assert_not_called()

    def test_em_blocks_cover_each_epoch_without_overlap(self) -> None:
        first = [
            ActiveTrainer._em_block_indices(12, 4, cycle, 7)[0] for cycle in range(1, 4)
        ]
        second = [
            ActiveTrainer._em_block_indices(12, 4, cycle, 7)[0] for cycle in range(4, 7)
        ]
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(12))
        self.assertTrue(
            all(
                len(np.intersect1d(left, right)) == 0
                for index, left in enumerate(first)
                for right in first[index + 1 :]
            )
        )
        self.assertFalse(np.array_equal(np.concatenate(first), np.concatenate(second)))
        uneven = [
            ActiveTrainer._em_block_indices(10, 4, cycle, 7)[0] for cycle in range(1, 4)
        ]
        self.assertEqual([len(block) for block in uneven], [4, 3, 3])
        np.testing.assert_array_equal(np.sort(np.concatenate(uneven)), np.arange(10))


if __name__ == "__main__":
    unittest.main()
