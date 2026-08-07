from __future__ import annotations

import unittest

import numpy as np

from lal_wishart.reproduction.paper_k3c5 import _sample_parameters
from lal_wishart.reproduction.signed_k3c5 import (
    generate_signed_k3c5,
    sample_wishart_effect,
    signed_mean_matrix,
    simulate_signed_hawkes_path,
)


class SignedK3C5Tests(unittest.TestCase):
    def test_zero_level_is_balanced_and_uses_identity_effects(self) -> None:
        dataset = generate_signed_k3c5(
            heterogeneity="zero",
            parameter_seed=3,
            simulation_seed=4,
            n_per_cluster=2,
            horizon=2.0,
            max_jumps=30,
        )
        np.testing.assert_array_equal(np.bincount(dataset.labels), [2, 2, 2])
        expected = np.eye(15)[None].repeat(6, axis=0)
        np.testing.assert_allclose(dataset.random_effects, expected)
        self.assertIsNone(dataset.degrees_of_freedom)

    def test_signed_simulator_is_deterministic_and_respects_cap(self) -> None:
        parameters = _sample_parameters(np.random.RandomState(5), 3, 5)
        matrix = signed_mean_matrix(3, 5, suppressive=True)
        first = simulate_signed_hawkes_path(
            np.random.RandomState(7),
            parameters,
            component=1,
            matrix=matrix,
            alpha=1.0,
            horizon=3.0,
            max_jumps=25,
        )
        second = simulate_signed_hawkes_path(
            np.random.RandomState(7),
            parameters,
            component=1,
            matrix=matrix,
            alpha=1.0,
            horizon=3.0,
            max_jumps=25,
        )
        np.testing.assert_array_equal(first.times, second.times)
        np.testing.assert_array_equal(first.marks, second.marks)
        self.assertLessEqual(first.count, 25)

    def test_suppressive_mean_is_spd_and_has_observable_signs(self) -> None:
        mean = signed_mean_matrix(3, 5, suppressive=True, correlation=0.6)
        self.assertGreater(float(np.linalg.eigvalsh(mean).min()), 0.0)
        self.assertAlmostEqual(float(np.trace(mean)), 15.0)
        for offset in (0, 5, 10):
            self.assertAlmostEqual(float(mean[offset, offset + 1]), -0.6)
            self.assertAlmostEqual(float(mean[offset + 2, offset + 3]), 0.6)

    def test_strong_wishart_level_has_more_matrix_variation(self) -> None:
        mean = np.eye(15)
        moderate_rng = np.random.RandomState(11)
        strong_rng = np.random.RandomState(11)
        moderate = np.stack([
            sample_wishart_effect(moderate_rng, mean, 60)
            for _ in range(100)
        ])
        strong = np.stack([
            sample_wishart_effect(strong_rng, mean, 15)
            for _ in range(100)
        ])
        moderate_distance = np.linalg.norm(moderate - mean, axis=(1, 2)).mean()
        strong_distance = np.linalg.norm(strong - mean, axis=(1, 2)).mean()
        self.assertGreater(strong_distance, moderate_distance)


if __name__ == "__main__":
    unittest.main()
