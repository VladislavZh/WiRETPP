from __future__ import annotations

import unittest

import torch

from wishart_tpp.inference.damping import (
    damp_population,
    damp_probability,
    damp_simplex,
)


class DampingTest(unittest.TestCase):
    def test_population_endpoints_are_exact(self) -> None:
        current = torch.eye(2)[None]
        optimum = torch.tensor([[[1.5, 0.2], [0.2, 0.5]]])
        self.assertIs(damp_population(current, optimum, 0.0), current)
        self.assertIs(damp_population(current, optimum, 1.0), optimum)

    def test_population_interpolation_stays_spd_and_preserves_trace(self) -> None:
        current = torch.eye(2)[None]
        optimum = torch.tensor([[[1.5, 0.2], [0.2, 0.5]]])
        updated = damp_population(current, optimum, 0.25)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(updated) > 0.0))
        torch.testing.assert_close(
            updated.diagonal(dim1=-2, dim2=-1).sum(-1), torch.tensor([2.0])
        )

    def test_probability_endpoints_are_exact(self) -> None:
        self.assertEqual(damp_probability(0.2, 0.8, 0.0), 0.2)
        self.assertEqual(damp_probability(0.2, 0.8, 1.0), 0.8)

    def test_probability_uses_arithmetic_ema(self) -> None:
        updated = damp_probability(0.2, 0.8, 0.25)
        self.assertAlmostEqual(updated, 0.35, places=12)

    def test_repeated_updates_are_exponential_smoothing(self) -> None:
        updated = damp_probability(0.0, 1.0, 0.25)
        updated = damp_probability(updated, 1.0, 0.25)
        self.assertAlmostEqual(updated, 0.4375, places=12)

    def test_simplex_endpoints_are_exact(self) -> None:
        current = torch.tensor([0.8, 0.2])
        optimum = torch.tensor([0.1, 0.9])
        self.assertIs(damp_simplex(current, optimum, 0.0), current)
        self.assertIs(damp_simplex(current, optimum, 1.0), optimum)

    def test_simplex_uses_arithmetic_ema_and_stays_normalized(self) -> None:
        current = torch.tensor([0.8, 0.2])
        optimum = torch.tensor([0.1, 0.9])
        updated = damp_simplex(current, optimum, 0.25)
        torch.testing.assert_close(updated, torch.tensor([0.625, 0.375]))
        torch.testing.assert_close(updated.sum(), torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
