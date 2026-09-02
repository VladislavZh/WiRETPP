from __future__ import annotations

import unittest

import torch

from wishart_tpp.config import TrainingConfig
from wishart_tpp.inference.mixture_weights import dirichlet_map_weights


class MixtureWeightPriorTest(unittest.TestCase):
    def test_default_config_uses_damped_ml_target(self) -> None:
        self.assertEqual(TrainingConfig().mixture_weight_dirichlet_concentration, 1.0)
        self.assertEqual(TrainingConfig().mixture_weight_damping, 0.25)
        gamma = torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.4, 0.0]])
        historical = gamma.mean(0).clamp_min(1e-8)
        historical = historical / historical.sum()
        torch.testing.assert_close(
            dirichlet_map_weights(gamma), historical, rtol=0.0, atol=0.0
        )

    def test_matches_analytic_map_formula(self) -> None:
        gamma = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        concentration = 3.0
        counts = gamma.sum(0)
        expected = (counts + concentration - 1.0) / (
            gamma.shape[0] + gamma.shape[1] * (concentration - 1.0)
        )
        torch.testing.assert_close(
            dirichlet_map_weights(gamma, concentration), expected
        )

    def test_prior_revives_dead_component(self) -> None:
        gamma = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = dirichlet_map_weights(gamma, concentration=2.0)
        self.assertGreater(float(weights[1]), 0.0)

    def test_weights_are_nonnegative_and_normalized(self) -> None:
        gamma = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        weights = dirichlet_map_weights(gamma, concentration=21.0)
        self.assertTrue(torch.all(weights >= 0.0))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    def test_concentration_below_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, ">= 1"):
            TrainingConfig(mixture_weight_dirichlet_concentration=0.9)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            dirichlet_map_weights(torch.ones(2, 1), concentration=0.9)

    def test_weight_damping_outside_probability_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixture_weight_damping"):
            TrainingConfig(mixture_weight_damping=-0.1)
        with self.assertRaisesRegex(ValueError, "mixture_weight_damping"):
            TrainingConfig(mixture_weight_damping=1.1)


if __name__ == "__main__":
    unittest.main()
