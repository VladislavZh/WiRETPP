"""Typed construction of the fixed-q three-method real-data protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from wishart_tpp.config import ExperimentConfig


@dataclass(frozen=True)
class ComputeShards:
    """Hold compute-only shard sizes that do not alter statistical batches."""

    batch: int
    evaluation_batch: int
    path: int
    evaluation_samples: int


class RealProtocol:
    """Build seed-specific COTIC, Pure, and fixed-q Wishart configurations."""

    datasets = ("retweet", "amazon", "so")

    def __init__(self, config_path: Path) -> None:
        self.base = ExperimentConfig.from_yaml(config_path)

    @staticmethod
    def default_shards(dataset: str) -> ComputeShards:
        if dataset == "so":
            return ComputeShards(96, 8, 256, 8)
        return ComputeShards(32, 8, 64, 64)

    def configurations(
        self,
        *,
        dataset: str,
        seed: int,
        output: Path,
        smoke: bool,
        shards: ComputeShards | None = None,
    ) -> tuple[ExperimentConfig, ExperimentConfig]:
        if dataset not in self.datasets:
            raise ValueError(f"unsupported real dataset: {dataset}")
        selected_shards = shards or self.default_shards(dataset)
        population_df = 32.0 if dataset == "so" else 16.0
        cycles = 2 if smoke else self.base.training.active_cycles
        pretrain = 2 if smoke else self.base.training.shared_pretrain_steps
        updates_per_cycle = 1 if smoke else self.base.training.neural_steps_per_cycle
        common = replace(
            self.base.training,
            shared_pretrain_steps=pretrain,
            pure_steps=2 if smoke else cycles * updates_per_cycle,
            active_cycles=cycles,
            neural_steps_per_cycle=updates_per_cycle,
            local_steps=1 if smoke else self.base.training.local_steps,
            local_samples=1 if smoke else self.base.training.local_samples,
            local_evaluation_samples=(
                2 if smoke else self.base.training.local_evaluation_samples
            ),
            validation_samples=4 if smoke else self.base.training.validation_samples,
            test_samples=4 if smoke else self.base.training.test_samples,
            test_monte_carlo_repeats=(
                1 if smoke else self.base.training.test_monte_carlo_repeats
            ),
            population_df=population_df,
            learn_population_df=False,
            batch_size=selected_shards.batch,
            evaluation_batch_size=selected_shards.evaluation_batch,
            path_shard_size=selected_shards.path,
            evaluation_sample_shard_size=selected_shards.evaluation_samples,
            effective_batch_size=16 if smoke else 140,
        )
        runtime = replace(
            self.base.runtime,
            output_root=output,
            dataset=dataset,
            optimization_seed=seed,
            monte_carlo_seed=seed,
            neural_seed=2026082800 + seed,
        )
        configured = replace(
            self.base,
            model=replace(self.base.model, integral_seed=seed),
            training=common,
            runtime=runtime,
        )
        pure = replace(
            configured,
            training=replace(
                common,
                learning_rate=1e-4,
                lr_plateau_patience=5,
                lr_plateau_min_lr=0.0,
            ),
        )
        wishart = replace(
            configured,
            training=replace(common, learning_rate=3e-4),
        )
        return pure, wishart
