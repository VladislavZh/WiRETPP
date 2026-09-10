"""CPU contracts for the standalone paper package (no private baseline imports)."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from lightning.fabric import Fabric

from active_wishart_tpp.config import (
    ComputeConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from active_wishart_tpp.training.artifact_io import tensor_tree_equal
from active_wishart_tpp.training.runner import ExperimentRunner
from active_wishart_tpp.training.expectation import em_block_indices
from active_wishart_tpp.training.updates import WishartUpdate


def fixture(root):
    """Write a small deterministic labeled event table without production data."""
    target = root / "age"
    target.mkdir()
    rows = [
        dict(
            source_id=i + 1,
            label=i % 2,
            horizon=730.0,
            times=[0.0, 0.2 + 0.003 * i, 0.8 + 0.004 * i],
            marks=[i % 2, 1, 0],
        )
        for i in range(40)
    ]
    pd.DataFrame(rows).to_parquet(target / "events.parquet", index=False)


def config(root, output):
    return ExperimentConfig(
        ModelConfig(input_channels=4, hidden_size=4, layers=1, integral_samples=2),
        TrainingConfig(
            cycles=2,
            population_df=5.0,
            updates_per_cycle=2,
            effective_batch=4,
            local_steps=2,
            e_fit_samples=2,
            e_score_samples=4,
            m_samples=4,
            validation_samples=4,
            selected_samples=4,
            selected_repeats=1,
        ),
        ComputeConfig(
            physical_batch=2,
            trace_batch=2,
            path_shard=8,
            e_fit_batch=4,
            em_block=16,
            mc_draw_shard=4,
        ),
        RuntimeConfig(
            dataset="age",
            data_root=root,
            output_root=output,
            accelerator="cpu",
            components=2,
        ),
    )


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_no_experimental_switches(self):
        for key in (
            "balanced_cycles",
            "scheduler",
            "temperature",
            "head_freeze_cycles",
            "ntpp_mix",
        ):
            with self.assertRaises(TypeError):
                TrainingConfig(**{key: 1})

    def test_equal_blocks(self):
        blocks = [em_block_indices(19, 8, c, 7)[0] for c in range(1, 4)]
        self.assertEqual(sorted(np.concatenate(blocks).tolist()), list(range(19)))
        self.assertLessEqual(max(map(len, blocks)) - min(map(len, blocks)), 1)

    def test_optimizer_groups(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            runner = ExperimentRunner(
                config(root, root / "out"), Fabric(accelerator="cpu")
            )
            update = WishartUpdate(runner.fabric, runner.decoder, runner.config)
            _, optimizer = update.setup(runner.new_model())
            roles = {g["role"]: g for g in optimizer.param_groups}
            self.assertEqual(
                tuple(roles),
                ("backbone", "component_heads", "omega", "alpha", "mixture"),
            )
            for role, group in roles.items():
                self.assertEqual(
                    group["betas"],
                    (0.9, 0.95) if role in ("omega", "alpha") else (0.9, 0.999),
                )
            self.assertEqual(roles["alpha"]["lr"], 0.01)
            self.assertEqual(roles["omega"]["lr"], 1e-4)

    def test_pair_and_exact_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cfg = config(root, root / "full")
            full = ExperimentRunner(cfg, Fabric(accelerator="cpu"))
            shared = full.shared()
            self.assertTrue(tensor_tree_equal(shared, full.shared()))
            for method in ("pure", "wishart"):
                with self.subTest(method=method):
                    result = full.fit(method, shared)
                    self.assertTrue(result["selected_state_matches_checkpoint"])
                    self.assertTrue(result["test_read"])
                    with patch.object(
                        full, "evaluate", side_effect=AssertionError("Repeated scoring")
                    ):
                        self.assertEqual(full.fit(method, shared), result)
                    split_config = replace(
                        cfg, runtime=replace(cfg.runtime, output_root=root / method)
                    )
                    interrupted = ExperimentRunner(
                        split_config, Fabric(accelerator="cpu")
                    )
                    interrupted.fit(method, shared, stop_after=1)
                    resumed = ExperimentRunner(split_config, Fabric(accelerator="cpu"))
                    resumed.fit(method, shared)
                    a = torch.load(
                        full.output / method / "active.pt", weights_only=False
                    )
                    b = torch.load(
                        resumed.output / method / "active.pt", weights_only=False
                    )
                    for key in (
                        "model",
                        "optimizer",
                        "population",
                        "best",
                        "previous",
                        "torch_rng",
                        "numpy_generator",
                        "integration_draw",
                        "integration_seed",
                        "physical_omega",
                    ):
                        self.assertTrue(tensor_tree_equal(a[key], b[key]), key)
                    wrong = replace(
                        split_config, training=replace(cfg.training, omega_lr=1e-3)
                    )
                    with self.assertRaises(ValueError):
                        ExperimentRunner(wrong, Fabric(accelerator="cpu")).fit(
                            method, shared
                        )
                    with patch(
                        "active_wishart_tpp.training.runner.source_digest",
                        return_value="wrong",
                    ):
                        with self.assertRaises(ValueError):
                            ExperimentRunner(
                                split_config, Fabric(accelerator="cpu")
                            ).fit(method, shared)


if __name__ == "__main__":
    unittest.main()
