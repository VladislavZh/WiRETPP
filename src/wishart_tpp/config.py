"""Typed configuration for the reproducible all-12 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "cotic"
    input_channels: int = 192
    hidden_size: int = 512
    layers: int = 8
    attention_heads: int = 4
    use_layer_norm: bool = True
    kernel_size: int = 3
    dropout: float = 0.1
    dilation_factor: float = 1.29
    integral_method: str = "monte_carlo"
    integral_samples: int = 50
    integral_seed: int = 0
    component_noise: float = 0.02
    wishart_intensity_floor: float = 1e-6
    wishart_intensity_floor_scaling: str = "constant"

    def __post_init__(self) -> None:
        if self.integral_method not in {"gauss_legendre", "monte_carlo"}:
            raise ValueError(
                "model.integral_method must be 'gauss_legendre' or 'monte_carlo'"
            )
        if self.integral_samples < 1:
            raise ValueError("model.integral_samples must be positive")
        if self.wishart_intensity_floor < 0.0:
            raise ValueError("model.wishart_intensity_floor must be non-negative")
        if self.wishart_intensity_floor_scaling != "constant":
            raise ValueError("model.wishart_intensity_floor_scaling must be 'constant'")


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "wishart"
    shared_pretrain_steps: int = 10
    shared_batch_size: int | None = None
    shared_evaluation_batch_size: int | None = None
    shared_path_shard_size: int | None = None
    pure_steps: int = 600
    active_cycles: int = 75
    neural_steps_per_cycle: int = 8
    local_steps: int = 10
    local_samples: int = 4
    local_evaluation_samples: int = 8
    population_df: float = 16.0
    learn_population_df: bool = False
    population_df_warmup_cycles: int = 0
    population_df_profile_points: int = 9
    population_df_min: float | None = None
    population_df_max: float = 256.0
    population_df_damping: float = 0.25
    population_df_maximum_ratio: float = 2.0
    initial_alpha: float = 0.1
    alpha_warmup_cycles: int = 0
    alpha_reactivation_cycles: int = 0
    alpha_steps: int = 25
    alpha_samples: int = 4
    omega_damping: float = 0.25
    alpha_damping: float = 0.5
    mixture_weight_damping: float = 0.25
    mixture_weight_dirichlet_concentration: float = 1.0
    balanced_cycles: int = 2
    validation_samples: int = 64
    test_samples: int = 256
    test_monte_carlo_repeats: int = 3
    test_local_steps: int = 800
    test_local_samples: int = 2
    test_local_evaluation_samples: int = 64
    test_local_adaptive: bool = True
    test_local_tolerance: float = 1e-3
    test_local_kappa_tolerance: float = 5e-3
    test_local_gradient_tolerance: float = 2e-3
    test_local_check_interval: int = 50
    test_local_patience: int = 3
    test_local_monitor_samples: int = 16
    test_local_learning_rate_floor: float = 0.02
    batch_size: int | None = None
    effective_batch_size: int = 140
    evaluation_batch_size: int | None = None
    evaluation_sample_shard_size: int = 64
    path_shard_size: int | None = None
    em_batch_size: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 500.0
    reduce_lr_on_plateau: bool = False
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 5
    lr_plateau_min_lr: float = 0.0
    fixed_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"wishart", "pure"}:
            raise ValueError("training.method must be 'wishart' or 'pure'")
        if self.population_df <= 0.0:
            raise ValueError("training.population_df must be positive")
        if self.population_df_warmup_cycles < 0:
            raise ValueError(
                "training.population_df_warmup_cycles must be non-negative"
            )
        if self.population_df_profile_points < 3:
            raise ValueError("training.population_df_profile_points must be >= 3")
        if self.population_df_min is not None and self.population_df_min <= 0.0:
            raise ValueError("training.population_df_min must be positive")
        if self.population_df_max < self.population_df:
            raise ValueError("training.population_df_max must be >= population_df")
        if not 0.0 < self.population_df_damping <= 1.0:
            raise ValueError("training.population_df_damping must lie in (0, 1]")
        if self.population_df_maximum_ratio <= 1.0:
            raise ValueError("training.population_df_maximum_ratio must be > 1")
        if self.fixed_alpha is not None and not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("training.fixed_alpha must lie in [0, 1]")
        if self.alpha_warmup_cycles < 0 or self.alpha_reactivation_cycles < 0:
            raise ValueError("alpha schedule cycles must be non-negative")
        if self.alpha_warmup_cycles + self.alpha_reactivation_cycles > self.active_cycles:
            raise ValueError(
                "alpha warm-up and reactivation must fit within active_cycles"
            )
        if self.alpha_reactivation_cycles and not self.alpha_warmup_cycles:
            raise ValueError("alpha reactivation requires a positive warm-up")
        if self.fixed_alpha is not None and (
            self.alpha_warmup_cycles or self.alpha_reactivation_cycles
        ):
            raise ValueError("alpha warm-up is incompatible with fixed_alpha")
        if not 0.0 <= self.omega_damping <= 1.0:
            raise ValueError("training.omega_damping must lie in [0, 1]")
        if not 0.0 <= self.alpha_damping <= 1.0:
            raise ValueError("training.alpha_damping must lie in [0, 1]")
        if not 0.0 <= self.mixture_weight_damping <= 1.0:
            raise ValueError("training.mixture_weight_damping must lie in [0, 1]")
        if self.mixture_weight_dirichlet_concentration < 1.0:
            raise ValueError(
                "training.mixture_weight_dirichlet_concentration must be >= 1"
            )
        if self.test_monte_carlo_repeats < 1:
            raise ValueError("training.test_monte_carlo_repeats must be positive")
        for name, value in (
            ("shared_batch_size", self.shared_batch_size),
            ("shared_evaluation_batch_size", self.shared_evaluation_batch_size),
            ("shared_path_shard_size", self.shared_path_shard_size),
        ):
            if value is not None and value < 1:
                raise ValueError(f"training.{name} must be positive")
        if self.path_shard_size is not None and self.path_shard_size < 1:
            raise ValueError("training.path_shard_size must be positive")
        if self.evaluation_sample_shard_size < 1:
            raise ValueError("training.evaluation_sample_shard_size must be positive")
        if self.em_batch_size is not None and self.em_batch_size < 1:
            raise ValueError("training.em_batch_size must be positive")
        if not 0.0 < self.lr_plateau_factor < 1.0:
            raise ValueError("training.lr_plateau_factor must lie in (0, 1)")
        if self.lr_plateau_patience < 0:
            raise ValueError("training.lr_plateau_patience must be non-negative")
        if self.lr_plateau_min_lr < 0.0:
            raise ValueError("training.lr_plateau_min_lr must be non-negative")
        if self.test_local_adaptive and (
            self.test_local_tolerance <= 0.0
            or self.test_local_kappa_tolerance <= 0.0
            or self.test_local_gradient_tolerance <= 0.0
            or self.test_local_check_interval < 1
            or self.test_local_patience < 1
            or self.test_local_monitor_samples < 1
            or not 0.0 < self.test_local_learning_rate_floor <= 1.0
        ):
            raise ValueError("invalid adaptive test-local inference schedule")
        if self.method == "pure" and self.fixed_alpha is not None:
            raise ValueError("fixed_alpha belongs to the Wishart method")


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("runs/all12")
    shared_checkpoint_root: Path | None = None
    dataset: str = "all12"
    accelerator: str = "auto"
    precision: str = "32-true"
    split_seed: int = 42
    optimization_seed: int = 0
    monte_carlo_seed: int = 0
    neural_seed: int | None = None
    deterministic: bool = True
    trajectory_limit: int | None = None
    cohort_seed: int = 20260824
    mixture_components: int | None = None
    real_split_protocol: str = "official"
    packed_filename: str = "events.parquet"

    def __post_init__(self) -> None:
        if self.shared_checkpoint_root is not None and not self.shared_checkpoint_root.name:
            raise ValueError("runtime.shared_checkpoint_root must name a directory")
        if self.trajectory_limit is not None and self.trajectory_limit < 1:
            raise ValueError("runtime.trajectory_limit must be positive")
        if self.mixture_components is not None and self.mixture_components < 1:
            raise ValueError("runtime.mixture_components must be positive")
        if self.real_split_protocol not in {
            "official",
            "deduplicated_validation_half",
        }:
            raise ValueError(
                "runtime.real_split_protocol must be 'official' or "
                "'deduplicated_validation_half'"
            )
        if not self.packed_filename:
            raise ValueError("runtime.packed_filename must be non-empty")
        if Path(self.packed_filename).name != self.packed_filename:
            raise ValueError(
                "runtime.packed_filename must not contain directory components"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            runtime=RuntimeConfig(
                **{
                    key: Path(value) if key.endswith("_root") else value
                    for key, value in payload.get("runtime", {}).items()
                }
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data_root", "output_root", "shared_checkpoint_root"):
            if values["runtime"][key] is not None:
                values["runtime"][key] = str(values["runtime"][key])
        return values
