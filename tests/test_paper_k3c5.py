from __future__ import annotations

import unittest

import numpy as np

from lal_wishart.experiment import make_splits
from lal_wishart.reproduction.paper_k3c5 import (
    ExponentialHawkesParameters,
    generate_paper_k3c5,
    simulate_exponential_hawkes,
)


class PaperK3C5Tests(unittest.TestCase):
    def test_simulator_is_deterministic_and_respects_cap(self) -> None:
        parameters = ExponentialHawkesParameters(
            baseline=np.asarray([0.7, 0.4]),
            adjacency=np.asarray([[0.4, 0.1], [0.2, 0.3]]),
            decays=np.asarray([[0.5, 0.6], [0.4, 0.7]]),
        )
        first = simulate_exponential_hawkes(
            np.random.RandomState(11), parameters, horizon=5.0, max_jumps=20
        )
        second = simulate_exponential_hawkes(
            np.random.RandomState(11), parameters, horizon=5.0, max_jumps=20
        )
        np.testing.assert_array_equal(first.times, second.times)
        np.testing.assert_array_equal(first.marks, second.marks)
        self.assertLessEqual(first.count, 20)

    def test_balanced_dataset_and_reproducible_split(self) -> None:
        dataset = generate_paper_k3c5(
            parameter_seed=3,
            simulation_seed=4,
            n_per_cluster=4,
            horizon=5.0,
            max_jumps=50,
        )
        np.testing.assert_array_equal(np.bincount(dataset.labels), [4, 4, 4])
        first = make_splits(
            parameter_seed=3,
            simulation_seed=4,
            split_seed=5,
            horizon=5.0,
            generated_per_class=4,
            train_per_class=2,
            validation_per_class=1,
            test_per_class=1,
        )
        second = make_splits(
            parameter_seed=3,
            simulation_seed=4,
            split_seed=5,
            horizon=5.0,
            generated_per_class=4,
            train_per_class=2,
            validation_per_class=1,
            test_per_class=1,
        )
        for left, right in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(left.labels, right.labels)
            for left_sequence, right_sequence in zip(left.sequences, right.sequences):
                np.testing.assert_array_equal(left_sequence.times, right_sequence.times)
                np.testing.assert_array_equal(left_sequence.marks, right_sequence.marks)


if __name__ == "__main__":
    unittest.main()
