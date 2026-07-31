import unittest

import numpy as np

from lal_wishart.metrics import (
    adjusted_rand_index,
    cluster_purity,
    normalized_mutual_information,
)


class ClusteringMetricTests(unittest.TestCase):
    def test_adjusted_rand_index_boundaries(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(adjusted_rand_index(labels, labels), 1.0)
        collapsed = np.zeros_like(labels)
        self.assertAlmostEqual(adjusted_rand_index(labels, collapsed), 0.0)
        self.assertAlmostEqual(
            normalized_mutual_information(labels, labels),
            1.0,
        )
        self.assertAlmostEqual(
            normalized_mutual_information(labels, collapsed),
            0.0,
        )

    def test_perfect_up_to_permutation(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = np.asarray([2, 2, 0, 0, 1, 1])
        self.assertEqual(cluster_purity(labels, predictions), 1.0)

    def test_single_cluster_is_majority_fraction(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = np.zeros(6, dtype=int)
        self.assertAlmostEqual(cluster_purity(labels, predictions), 1.0 / 3.0)

    def test_rejects_incompatible_inputs(self) -> None:
        with self.assertRaises(ValueError):
            cluster_purity(np.asarray([0, 1]), np.asarray([0]))
        with self.assertRaises(ValueError):
            cluster_purity(np.asarray([]), np.asarray([]))
