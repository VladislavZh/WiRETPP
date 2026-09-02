from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from wishart_tpp.config import RuntimeConfig
from wishart_tpp.data import EventDataModule
from wishart_tpp.metrics import clustering_summary


def _write_labeled(
    root: Path,
    name: str,
    count: int,
    *,
    delta: float = 2.0,
    filename: str = "events.parquet",
) -> None:
    """Write one packed labeled fixture with a constant return time."""

    directory = root / name
    directory.mkdir()
    offsets = np.arange(count, dtype=float) / 100.0
    horizons = [730.0] * count if name == "age" else offsets + delta + 10.0
    pd.DataFrame(
        {
            "source_id": np.arange(1, count + 1),
            "label": np.arange(count) % 2,
            "horizon": horizons,
            "times": [[offset, offset + delta] for offset in offsets],
            "marks": [[0, 1] for _ in range(count)],
        }
    ).to_parquet(directory / filename, index=False)


def _write_real(
    root: Path,
    name: str = "amazon",
    *,
    train=None,
    validation=None,
    test=None,
) -> None:
    """Write one official-split CSV fixture."""

    parts = {
        "train": train or [([0.0, 2.0], [0, 1]), ([1.0, 3.0], [1, 0])],
        "val": validation or [([4.0, 6.0], [0, 1])],
        "test": test or [([7.0, 9.0], [1, 0])],
    }
    for split, paths in parts.items():
        directory = root / name / split
        directory.mkdir(parents=True)
        for index, (times, marks) in enumerate(paths):
            pd.DataFrame({"time": times, "event": marks}).to_csv(
                directory / f"{index}.csv", index=False
            )


class UnifiedDataModuleTest(unittest.TestCase):
    def test_labeled_data_uses_seeded_split_and_common_normalization(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_labeled(root, "age", 20)

            prepared = EventDataModule(root, split_seed=42).prepare("age")
            dataset, split = prepared.dataset, prepared.split

            self.assertEqual(dataset.n_components, 2)
            self.assertEqual(dataset.n_marks, 2)
            self.assertEqual(
                [
                    len(split.train.sequences),
                    len(split.validation.sequences),
                    len(split.test.sequences),
                ],
                [16, 2, 2],
            )
            combined = np.concatenate(
                (
                    split.train.source_ids,
                    split.validation.source_ids,
                    split.test.source_ids,
                )
            )
            self.assertEqual(len(np.unique(combined)), 20)
            self.assertAlmostEqual(dataset.normalization.rate, 0.5)
            self.assertAlmostEqual(dataset.sequences[0].times[-1], 1.0)
            self.assertAlmostEqual(dataset.sequences[0].horizon, 365.0)

    def test_labeled_adapter_accepts_custom_packed_filename(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_labeled(root, "age", 20, filename="custom.parquet")

            prepared = EventDataModule(root, packed_filename="custom.parquet").prepare(
                "age"
            )

            self.assertEqual(len(prepared.dataset.sequences), 20)
            self.assertTrue(
                np.array_equal(prepared.dataset.source_ids, np.arange(1, 21))
            )

    def test_real_data_preserves_official_splits_and_sorts_stably(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_real(
                root,
                train=[([3.0, 1.0, 2.0], [2, 0, 1]), ([4.0, 6.0], [1, 2])],
            )

            prepared = EventDataModule(root, mixture_components=3).prepare("amazon")
            dataset, split = prepared.dataset, prepared.split

            self.assertEqual(dataset.n_components, 3)
            self.assertEqual(dataset.n_marks, 3)
            self.assertEqual(
                [
                    len(split.train.sequences),
                    len(split.validation.sequences),
                    len(split.test.sequences),
                ],
                [2, 1, 1],
            )
            self.assertTrue(np.array_equal(split.train.sequences[0].marks, [0, 1, 2]))
            self.assertEqual(split.train.sequences[0].times[0], 0.0)
            self.assertTrue(np.all(split.train.labels == -1))

    def test_real_adapter_accepts_packed_parquet(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "linkedin"
            directory.mkdir()
            pd.DataFrame(
                {
                    "split": ["train", "train", "val", "test"],
                    "source_id": [0, 1, 0, 0],
                    "times": [
                        [3.0, 1.0, 2.0],
                        [4.0, 6.0],
                        [10.0, 12.0],
                        [7.0, 8.0],
                    ],
                    "marks": [[2, 0, 1], [1, 2], [0, 1], [2, 0]],
                }
            ).to_parquet(directory / "custom.parquet", index=False)

            module = EventDataModule(root, packed_filename="custom.parquet")
            prepared = module.prepare("linkedin")

            self.assertEqual(module.names("real"), ("linkedin",))
            self.assertEqual(len(prepared.split.train.sequences), 2)
            self.assertTrue(
                np.array_equal(prepared.split.train.sequences[0].marks, [0, 1, 2])
            )

    def test_deduplicated_protocol_halves_one_official_holdout(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            holdout = [([0.0, float(index + 1)], [0, index % 2]) for index in range(6)]
            _write_real(root, validation=holdout, test=holdout)

            prepared = EventDataModule(
                root,
                split_seed=42,
                split_protocol="deduplicated_validation_half",
            ).prepare("amazon")

            self.assertEqual(len(prepared.dataset.sequences), 8)
            self.assertEqual(
                [
                    len(prepared.split.train.sequences),
                    len(prepared.split.validation.sequences),
                    len(prepared.split.test.sequences),
                ],
                [2, 3, 3],
            )
            self.assertEqual(
                len(
                    np.intersect1d(
                        prepared.split.validation.source_ids,
                        prepared.split.test.source_ids,
                    )
                ),
                0,
            )

    def test_deduplication_rejects_distinct_official_test(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_real(
                root,
                validation=[([0.0, 2.0], [0, 1])],
                test=[([0.0, 3.0], [0, 1])],
            )

            module = EventDataModule(
                root, split_protocol="deduplicated_validation_half"
            )

            with self.assertRaisesRegex(ValueError, "not exact duplicates"):
                module.prepare("amazon")

    def test_unix_seconds_are_converted_before_p99_fit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [([1_600_000_000.0, 1_600_086_400.0], [0, 1])]
            _write_real(root, train=paths, validation=paths, test=paths)

            dataset = EventDataModule(root).prepare("amazon").dataset

            self.assertEqual(
                dataset.normalization.mode,
                "unix_seconds_to_days+train_p99_exponential",
            )
            self.assertEqual(dataset.normalization.unit_rate, 1.0 / 86_400.0)
            self.assertGreater(dataset.normalization.exponential_rate, 0.0)
            self.assertAlmostEqual(dataset.sequences[0].times[-1], 1.0)

    def test_synthetic_and_real_use_the_same_normalizer(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_labeled(root, "K2_C5", 800, delta=2.0)
            _write_real(root)
            module = EventDataModule(root)

            synthetic = module.prepare("K2_C5").dataset
            real = module.prepare("amazon").dataset

            self.assertEqual(synthetic.normalization, real.normalization)
            self.assertAlmostEqual(synthetic.sequences[0].times[-1], 1.0)
            self.assertAlmostEqual(real.sequences[0].times[-1], 1.0)

    def test_validation_and_test_cannot_change_the_fitted_rate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_real(
                root,
                validation=[([0.0, 2_000.0], [0, 1])],
                test=[([0.0, 4_000.0], [0, 1])],
            )

            dataset = EventDataModule(root).prepare("amazon").dataset

            self.assertAlmostEqual(dataset.normalization.rate, 0.5)
            self.assertAlmostEqual(dataset.sequences[2].times[-1], 1_000.0)
            self.assertAlmostEqual(dataset.sequences[3].times[-1], 2_000.0)

    def test_trajectory_limit_preserves_official_split_disjointness(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [([0.0, 2.0], [0, 1]) for _ in range(10)]
            validation = [([0.0, 2.0], [0, 1]) for _ in range(4)]
            test = [([0.0, 2.0], [0, 1]) for _ in range(4)]
            _write_real(root, train=train, validation=validation, test=test)

            prepared = EventDataModule(root).prepare(
                "amazon", trajectory_limit=9, cohort_seed=7
            )
            source_ids = [
                set(partition.source_ids.tolist())
                for partition in (
                    prepared.split.train,
                    prepared.split.validation,
                    prepared.split.test,
                )
            ]

            self.assertEqual(sum(map(len, source_ids)), 9)
            self.assertTrue(source_ids[0].isdisjoint(source_ids[1]))
            self.assertTrue(source_ids[0].isdisjoint(source_ids[2]))
            self.assertTrue(source_ids[1].isdisjoint(source_ids[2]))

    def test_runtime_and_missing_label_contracts_remain_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory_limit"):
            RuntimeConfig(trajectory_limit=0)
        with self.assertRaisesRegex(ValueError, "mixture_components"):
            RuntimeConfig(mixture_components=0)
        with self.assertRaisesRegex(ValueError, "real_split_protocol"):
            RuntimeConfig(real_split_protocol="invented")
        with self.assertRaisesRegex(ValueError, "packed_filename"):
            RuntimeConfig(packed_filename="nested/events.parquet")
        self.assertEqual(
            clustering_summary(
                np.full(3, -1),
                np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
            ),
            {"purity": -1.0, "ari": -1.0},
        )


if __name__ == "__main__":
    unittest.main()
