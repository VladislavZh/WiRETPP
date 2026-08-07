from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from lal_wishart.reproduction.dan_synthetic import (
    FIXED_HORIZON,
    dataset_names,
    load_dataset,
    shuffled_split,
)


class DANSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1] / "data"

    def test_discovers_all_twelve_datasets(self) -> None:
        names = dataset_names(self.root)
        self.assertEqual(len(names), 12)
        self.assertIn("K2_C5", names)
        self.assertIn("sin_K5_C5", names)
        self.assertIn("trunc_K5_C5", names)

    def test_loader_ignores_non_numeric_duplicate_files(self) -> None:
        dataset = load_dataset(self.root, "sin_K5_C5")
        self.assertEqual(len(dataset.sequences), 2000)
        self.assertEqual(dataset.labels.shape, (2000,))
        np.testing.assert_array_equal(
            np.bincount(dataset.labels), np.full(5, 400)
        )

    def test_plain_paths_keep_original_time_and_end_at_last_event(self) -> None:
        dataset = load_dataset(self.root, "K2_C5")
        self.assertEqual(
            dataset.source_horizon,
            "sequence_ends_at_fiftieth_event_original_time",
        )
        self.assertTrue(
            all(sequence.horizon == sequence.times[-1] for sequence in dataset.sequences)
        )
        source_times = np.loadtxt(
            self.root / "K2_C5" / "1.csv",
            delimiter=",",
            skiprows=1,
            usecols=1,
        )
        np.testing.assert_allclose(dataset.sequences[0].times, source_times)

    def test_fixed_horizon_families_keep_horizon_twenty(self) -> None:
        for name in ("sin_K2_C5", "trunc_K2_C5"):
            dataset = load_dataset(self.root, name)
            self.assertTrue(
                all(
                    sequence.horizon == FIXED_HORIZON
                    for sequence in dataset.sequences
                )
            )

    def test_numpy_shuffle_split_is_exact_and_reproducible(self) -> None:
        dataset = load_dataset(self.root, "K2_C5")
        first = shuffled_split(dataset, seed=17)
        second = shuffled_split(dataset, seed=17)
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (640, 80, 80))
        np.testing.assert_array_equal(first.train_ids, second.train_ids)
        combined = np.concatenate((first.train_ids, first.validation_ids, first.test_ids))
        np.testing.assert_array_equal(np.sort(combined), np.arange(1, 801))


if __name__ == "__main__":
    unittest.main()
