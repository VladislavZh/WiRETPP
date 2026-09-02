from __future__ import annotations

import unittest

import torch

from wishart_tpp.inference.responsibilities import ResponsibilityUpdater


class ResponsibilityUpdaterTests(unittest.TestCase):
    def test_unbalanced_matches_softmax(self) -> None:
        free_energy = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        log_weights = torch.tensor([0.2, -0.2])
        actual = ResponsibilityUpdater().update(
            free_energy, log_weights, balanced=False
        )
        expected = torch.softmax(log_weights[None] - free_energy, dim=1)
        torch.testing.assert_close(actual, expected)

    def test_balanced_log_sinkhorn_revives_underflowed_components(self) -> None:
        # Probability-space softmax makes four columns exactly zero here.
        free_energy = torch.zeros(100, 5)
        free_energy[:, 1:] = 10_000.0
        log_weights = torch.log(torch.full((5,), 0.2))
        actual = ResponsibilityUpdater(balance_iterations=40).update(
            free_energy, log_weights, balanced=True
        )
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.sum(1), torch.ones(100), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            actual.sum(0), torch.full((5,), 20.0), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
