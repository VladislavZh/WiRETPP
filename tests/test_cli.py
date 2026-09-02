"""Command-line override tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from wishart_tpp.cli import _configured_experiment, _parser


class CliOverrideTests(unittest.TestCase):
    def test_replication_and_compute_overrides_are_coherent(self) -> None:
        args = _parser().parse_args(
            [
                "--config",
                str(Path("configs/dan12_common_p99.yaml")),
                "--replication-seed",
                "2",
                "--compute-batch-size",
                "128",
                "--shared-compute-batch-size",
                "64",
                "--shared-checkpoint-root",
                "runs/source/wishart",
            ]
        )

        config = _configured_experiment(args)

        self.assertEqual(config.model.integral_seed, 2)
        self.assertEqual(config.runtime.optimization_seed, 2)
        self.assertEqual(config.runtime.monte_carlo_seed, 2)
        self.assertEqual(config.runtime.shared_checkpoint_root, Path("runs/source/wishart"))
        self.assertEqual(config.training.batch_size, 128)
        self.assertEqual(config.training.evaluation_batch_size, 64)
        self.assertEqual(config.training.path_shard_size, 512)
        self.assertEqual(config.training.shared_batch_size, 64)
        self.assertEqual(config.training.shared_evaluation_batch_size, 32)
        self.assertEqual(config.training.shared_path_shard_size, 256)
        self.assertEqual(config.training.effective_batch_size, 256)


if __name__ == "__main__":
    unittest.main()
