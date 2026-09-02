from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.backbones.integration import MonteCarloRule
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.inference.alpha import AlphaMstep
from wishart_tpp.inference.local import LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.active import ActiveSchedule, ActiveTrainer
from wishart_tpp.training.evaluation import ModelEvaluator


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RestoredEmCudaSmokeTest(unittest.TestCase):
    @staticmethod
    def _partition() -> tuple[DatasetPartition, DatasetPartition]:
        sequences = tuple(
            EventSequence(
                np.array([0.15 + 0.01 * index, 0.55, 0.9]),
                np.array([index % 2, (index + 1) % 2, index % 2]),
                1.0,
            )
            for index in range(8)
        )
        train = DatasetPartition(
            sequences[:6], np.array([0, 1, 0, 1, 0, 1]), np.arange(6)
        )
        validation = DatasetPartition(sequences[6:], np.array([0, 1]), np.arange(6, 8))
        return train, validation

    def test_restored_em_uses_convex_likelihood_mc50_and_resume_contract(self) -> None:
        train, validation = self._partition()
        fabric = Fabric(accelerator="cuda", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        evaluator = ModelEvaluator(decoder, cache_batch_size=2)
        trainer = ActiveTrainer(
            fabric,
            decoder,
            evaluator,
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=4,
            effective_batch_size=6,
            cache_batch_size=2,
            path_shard_size=3,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=100.0,
        )
        model = CoticIntensityBank(
            1,
            2,
            input_channels=4,
            hidden_size=8,
            layers=1,
            kernel_size=2,
            dropout=0.0,
            integration_rule=MonteCarloRule(50, 0),
        )
        model.expand_components(2, noise=0.02, seed=2)
        model, optimizer = trainer.setup(model)
        schedule = ActiveSchedule(
            cycles=1,
            neural_steps=1,
            local_steps=1,
            local_samples=1,
            local_evaluation_samples=1,
            population_df=16.0,
            initial_alpha=0.1,
            alpha_steps=2,
            alpha_samples=1,
            omega_damping=0.25,
            alpha_damping=0.5,
            mixture_weight_damping=0.25,
            balanced_cycles=1,
            validation_samples=2,
        )
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "active.pt"
            best, history = trainer.fit(
                model,
                optimizer,
                train,
                validation,
                schedule,
                optimization_seed=0,
                monte_carlo_seed=0,
                checkpoint_path=checkpoint_path,
            )
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

        self.assertEqual(model.integration_rule.samples, 50)
        self.assertEqual(decoder.intensity_formula, "convex_intensity_v1")
        self.assertEqual(payload["completed_cycle"], 1)
        contract = payload["signature"]["active_block_likelihood"]
        self.assertEqual(contract["mixture_weights"], "independent_logits")
        self.assertEqual(contract["cluster_w_coupling"], "likelihood_only")
        self.assertFalse(contract["w_to_mixture_routing"])
        self.assertEqual(
            payload["signature"]["wishart_intensity_floor"]["epsilon"],
            1e-6,
        )
        self.assertEqual(best.cycle, 1)
        self.assertEqual(len(history), 1)
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for key, value in history[0].items()
                if key not in {"population_df_optimum"}
            )
        )


if __name__ == "__main__":
    unittest.main()
