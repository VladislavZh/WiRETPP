from __future__ import annotations

import unittest

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.backbones.integration import GaussLegendreRule
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.inference.local import LocalWishartInference
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.cache import TraceCacheBuilder


class AdaptiveLocalInferenceTest(unittest.TestCase):
    def test_reports_early_convergence(self) -> None:
        train = DatasetPartition(
            tuple(
                EventSequence(
                    np.array([0.2 + 0.02 * index, 0.7]),
                    np.array([index % 2, (index + 1) % 2]),
                    1.0,
                )
                for index in range(4)
            ),
            np.array([0, 1, 0, 1]),
            np.arange(4),
        )
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
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
        cache = TraceCacheBuilder(2).build(model, train)

        posterior = LocalWishartInference(fabric, decoder).fit(
            cache,
            torch.eye(2)[None].repeat(2, 1, 1),
            5.0,
            alpha=0.3,
            steps=4,
            samples=1,
            evaluation_samples=1,
            seed=7,
            adaptive_tolerance=1e9,
            adaptive_kappa_tolerance=1e9,
            adaptive_gradient_tolerance=1e9,
            adaptive_minimum_steps=1,
            adaptive_check_interval=1,
            adaptive_patience=1,
            adaptive_monitor_samples=1,
        )

        self.assertTrue(posterior.converged)
        self.assertEqual(posterior.steps_taken, 2)


if __name__ == "__main__":
    unittest.main()
