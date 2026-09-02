from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from wishart_tpp.config import ModelConfig, RuntimeConfig, TrainingConfig
from wishart_tpp.real_protocol import RealProtocol


class BaseScheduleTest(unittest.TestCase):
    def test_all12_defaults_define_the_base_schedule(self) -> None:
        schedule = TrainingConfig()
        self.assertEqual(schedule.method, "wishart")
        self.assertEqual(schedule.shared_pretrain_steps, 10)
        self.assertEqual(schedule.active_cycles, 75)
        self.assertEqual(schedule.neural_steps_per_cycle, 8)
        self.assertEqual(schedule.local_steps, 10)
        self.assertEqual(schedule.local_samples, 4)
        self.assertEqual(schedule.local_evaluation_samples, 8)
        self.assertEqual(schedule.population_df, 16.0)
        self.assertFalse(schedule.learn_population_df)
        self.assertEqual(schedule.balanced_cycles, 2)
        self.assertEqual(schedule.initial_alpha, 0.1)
        self.assertEqual(schedule.alpha_warmup_cycles, 0)
        self.assertEqual(schedule.alpha_reactivation_cycles, 0)
        self.assertEqual(schedule.omega_damping, 0.25)
        self.assertEqual(schedule.alpha_damping, 0.5)
        self.assertEqual(schedule.mixture_weight_damping, 0.25)
        self.assertEqual(schedule.test_monte_carlo_repeats, 3)
        self.assertTrue(schedule.test_local_adaptive)
        self.assertEqual(schedule.test_local_check_interval, 50)
        self.assertIsNone(schedule.fixed_alpha)
        self.assertIsNone(schedule.path_shard_size)
        self.assertIsNone(schedule.em_batch_size)
        model = ModelConfig()
        self.assertEqual(model.integral_method, "monte_carlo")
        self.assertEqual(model.integral_samples, 50)
        self.assertEqual(model.wishart_intensity_floor, 1e-6)
        self.assertEqual(model.wishart_intensity_floor_scaling, "constant")
        self.assertTrue(RuntimeConfig().deterministic)

    def test_fixed_alpha_is_not_a_separate_pure_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Wishart method"):
            replace(TrainingConfig(), method="pure", fixed_alpha=0.0)

    def test_alpha_warmup_is_only_for_learned_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            replace(TrainingConfig(), fixed_alpha=0.0, alpha_warmup_cycles=1)

    def test_alpha_reactivation_requires_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            replace(TrainingConfig(), alpha_reactivation_cycles=1)

    def test_test_monte_carlo_repeats_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            replace(TrainingConfig(), test_monte_carlo_repeats=0)

    def test_path_shard_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            replace(TrainingConfig(), path_shard_size=0)

    def test_em_batch_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            replace(TrainingConfig(), em_batch_size=0)

    def test_real_protocol_uses_fixed_q_and_dataset_safe_shards(self) -> None:
        pure, wishart = RealProtocol(Path("configs/real_data.yaml")).configurations(
            dataset="so",
            seed=2,
            output=Path("runs/test"),
            smoke=False,
        )
        self.assertEqual(pure.training.learning_rate, 1e-4)
        self.assertEqual(wishart.training.learning_rate, 3e-4)
        self.assertEqual(wishart.training.population_df, 32.0)
        self.assertEqual(wishart.training.batch_size, 96)
        self.assertEqual(wishart.training.path_shard_size, 256)
        self.assertEqual(wishart.training.effective_batch_size, 140)
        self.assertEqual(wishart.runtime.optimization_seed, 2)


if __name__ == "__main__":
    unittest.main()
