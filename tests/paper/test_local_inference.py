from __future__ import annotations

import unittest

import numpy as np
import torch
from lightning.fabric import Fabric

from active_wishart_tpp.backbones.integration import MonteCarloRule
from active_wishart_tpp.cotic import CoticIntensityBank
from active_wishart_tpp.data import DatasetPartition, EventSequence
from active_wishart_tpp.inference.local import LocalWishartInference
from active_wishart_tpp.model.active_block import ActiveBlockDecoder
from active_wishart_tpp.training.cache import TraceCacheBuilder


class FixedLocalInferenceTest(unittest.TestCase):
    def test_fixed_budget_and_detached_posterior(self) -> None:
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
            integration_rule=MonteCarloRule(2, seed=7),
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
        )

        self.assertEqual(posterior.steps_taken, 4)
        self.assertEqual(tuple(posterior.means.shape), (4, 2, 2, 2))
        self.assertFalse(posterior.means.requires_grad)
        self.assertFalse(posterior.free_energy.requires_grad)
        self.assertTrue(torch.isfinite(posterior.free_energy).all())


if __name__ == "__main__":
    unittest.main()
