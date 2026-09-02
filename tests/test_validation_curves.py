import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.training.evaluation import Evaluation
from wishart_tpp.training.validation_curves import (
    ValidationCurveWriter,
    trajectory_bootstrap_nll,
)


class ValidationCurveTests(unittest.TestCase):
    def test_ratio_of_sums_and_seed_are_reproducible(self) -> None:
        scores = np.array([-2.0, -9.0, -5.0])
        horizons = np.array([1.0, 3.0, 2.0])
        first = trajectory_bootstrap_nll(scores, horizons, seed=7, draws=200)
        second = trajectory_bootstrap_nll(scores, horizons, seed=7, draws=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["nll_per_exposure"], 16.0 / 6.0)

    def test_writer_replaces_an_existing_cycle(self) -> None:
        partition = DatasetPartition(
            sequences=(
                EventSequence(np.array([0.5]), np.array([0]), 1.0),
                EventSequence(np.array([0.2]), np.array([0]), 2.0),
            ),
            labels=np.array([-1, -1]),
            source_ids=np.array([10, 20]),
        )

        def evaluation(scores: list[float]) -> Evaluation:
            values = torch.tensor(scores)
            return Evaluation(
                component_scores=values[:, None],
                marginal_scores=values,
                probabilities=torch.ones(2, 1),
                nll_per_exposure=float(-values.sum() / 3.0),
                nll_per_event=float(-values.sum() / 2.0),
                purity=-1.0,
                ari=-1.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            writer = ValidationCurveWriter(
                Path(directory),
                dataset="amazon",
                seed=0,
                method="pure_k5",
                bootstrap_draws=20,
            )
            writer.record(1, 8, evaluation([-2.0, -3.0]), partition)
            writer.record(1, 8, evaluation([-1.0, -2.0]), partition)
            self.assertEqual(writer.completed_cycles(), {1})
            import pandas as pd

            summary = pd.read_csv(writer.summary_path)
            paths = pd.read_csv(writer.paths_path)
            self.assertEqual(len(summary), 1)
            self.assertEqual(len(paths), 2)
            self.assertAlmostEqual(summary.iloc[0]["nll_per_exposure"], 1.0)


if __name__ == "__main__":
    unittest.main()
