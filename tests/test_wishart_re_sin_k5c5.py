from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lal_wishart.reproduction.wishart_re_sin_k5c5 import (
    SIN_K5_C5_BASELINE_TOTALS,
    cluster_mean_matrices,
    dataset_audit,
    generate_wishart_re_sin_k5c5,
    load_saved_dataset,
    save_dataset,
    sinus_hawkes_parameters,
    stratified_split,
)


class WishartRESinusK5C5Tests(unittest.TestCase):
    def test_k5_baselines_use_reference_sin_profiles_and_activity_scale(self) -> None:
        baselines, _ = sinus_hawkes_parameters(
            5, 5, parameter_seed=2026080701
        )
        np.testing.assert_allclose(
            baselines.sum(axis=1),
            SIN_K5_C5_BASELINE_TOTALS,
            rtol=0.04,
            atol=0.04,
        )
        self.assertGreater(
            float(np.linalg.norm(baselines[0] / baselines[0].sum()
                                 - baselines[1] / baselines[1].sum())),
            0.05,
        )

    def test_cluster_means_are_spd_trace_normalized_and_block_separated(self) -> None:
        means = cluster_mean_matrices(
            5,
            5,
            within_cluster_base_strength=0.7,
            true_cluster_block_boost=2.4,
            between_cluster_strength=0.12,
        )
        self.assertEqual(means.shape, (5, 25, 25))
        for cluster, mean in enumerate(means):
            self.assertGreater(float(np.linalg.eigvalsh(mean).min()), 0.0)
            self.assertAlmostEqual(float(np.trace(mean)), 25.0)
            masses = np.diag(mean).reshape(5, 5).sum(axis=1)
            self.assertEqual(int(masses.argmax()), cluster)
            self.assertGreater(masses[cluster], max(np.delete(masses, cluster)))

    def test_small_generation_is_balanced_reproducible_and_auditable(self) -> None:
        kwargs = {
            "parameter_seed": 11,
            "simulation_seed": 12,
            "n_per_cluster": 3,
            "n_components": 3,
            "n_marks": 3,
            "horizon": 2.0,
            "degrees_of_freedom": 10,
            "block_local_activity": True,
            "within_block_gain": 3.0,
            "between_block_gain": 0.25,
        }
        first = generate_wishart_re_sin_k5c5(**kwargs)
        second = generate_wishart_re_sin_k5c5(**kwargs)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_allclose(first.random_effects, second.random_effects)
        for left, right in zip(first.sequences, second.sequences, strict=True):
            np.testing.assert_allclose(left.times, right.times)
            np.testing.assert_array_equal(left.marks, right.marks)
        audit = dataset_audit(first)
        self.assertEqual(audit["cluster_counts"], [3, 3, 3])
        self.assertGreater(audit["mean_true_block_mass"], audit["mean_other_block_mass"])
        self.assertGreater(
            audit["mean_within_true_block_correlation"],
            audit["mean_within_other_block_correlation"],
        )
        self.assertGreater(audit["oracle_block_mass_routing_accuracy"], 0.5)
        self.assertIn("observable_validation_ridge_lda_accuracy", audit)
        self.assertIn("random_effect_rate_relative_sd_mean", audit)
        self.assertGreaterEqual(
            audit["random_effect_rate_relative_sd_mean"], 0.0
        )
        self.assertTrue(audit["block_local_activity"])
        self.assertEqual(audit["within_block_gain"], 3.0)
        self.assertEqual(audit["between_block_gain"], 0.25)

    def test_stratified_split_and_npz_roundtrip(self) -> None:
        dataset = generate_wishart_re_sin_k5c5(
            parameter_seed=21,
            simulation_seed=22,
            n_per_cluster=10,
            n_components=2,
            n_marks=2,
            horizon=1.5,
            degrees_of_freedom=5,
            block_local_activity=True,
            within_block_gain=2.0,
            between_block_gain=0.25,
        )
        split = stratified_split(
            dataset,
            seed=42,
            train_fraction=0.6,
            validation_fraction=0.2,
        )
        np.testing.assert_array_equal(np.bincount(split.train_labels), [6, 6])
        np.testing.assert_array_equal(
            np.bincount(split.validation_labels), [2, 2]
        )
        np.testing.assert_array_equal(np.bincount(split.test_labels), [2, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            save_dataset(path, dataset)
            loaded = load_saved_dataset(path)
        np.testing.assert_array_equal(loaded.labels, dataset.labels)
        np.testing.assert_allclose(
            loaded.cluster_mean_matrices, dataset.cluster_mean_matrices
        )
        self.assertEqual(len(loaded.sequences), len(dataset.sequences))
        self.assertTrue(loaded.block_local_activity)
        self.assertEqual(loaded.within_block_gain, 2.0)
        self.assertEqual(loaded.between_block_gain, 0.25)
        for left, right in zip(loaded.sequences, dataset.sequences, strict=True):
            np.testing.assert_allclose(left.times, right.times)
            np.testing.assert_array_equal(left.marks, right.marks)


if __name__ == "__main__":
    unittest.main()
