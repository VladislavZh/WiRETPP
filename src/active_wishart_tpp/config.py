"""Configuration of the single supported Bartlett/all-gradient protocol."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Specify the shared intensity bank and its numerical integration."""

    backbone: str = "cotic"
    input_channels: int = 192
    hidden_size: int = 512
    layers: int = 8
    attention_heads: int = 4
    kernel_size: int = 3
    dilation_factor: float = 1.29
    integral_samples: int = 50
    component_noise: float = 0.02

    def __post_init__(self):
        if self.backbone not in ("cotic", "thp"):
            raise ValueError("backbone must be cotic or corrected thp")
        if (
            min(
                self.input_channels,
                self.hidden_size,
                self.layers,
                self.integral_samples,
            )
            < 1
        ):
            raise ValueError("Model dimensions and integration budget must be positive")
        if not math.isfinite(self.component_noise) or self.component_noise < 0:
            raise ValueError("Component noise must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    """Fix the scientific protocol while exposing the run length and fixed df."""

    cycles: int = 60
    population_df: float = 16.0
    shared_steps: int = 10
    updates_per_cycle: int = 8
    effective_batch: int = 256
    local_steps: int = 8
    e_fit_samples: int = 4
    e_score_samples: int = 256
    m_samples: int = 64
    validation_samples: int = 64
    selected_samples: int = 64
    selected_repeats: int = 3
    learning_rate: float = 1e-4
    omega_lr: float = 1e-4
    alpha_lr: float = 0.01
    initial_alpha: float = 0.1
    omega_noise: float = 0.05
    gradient_clip: float = 500.0
    weight_decay: float = 1e-5

    def __post_init__(self):
        positive = (
            self.cycles,
            self.shared_steps,
            self.updates_per_cycle,
            self.effective_batch,
            self.local_steps,
            self.e_fit_samples,
            self.e_score_samples,
            self.m_samples,
            self.validation_samples,
            self.selected_samples,
            self.selected_repeats,
            self.population_df,
            self.learning_rate,
            self.omega_lr,
            self.alpha_lr,
            self.gradient_clip,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Budgets, rates and df must be positive")
        if not 0 <= self.initial_alpha <= 1 or not 0 <= self.omega_noise < 1:
            raise ValueError("Invalid alpha or Omega initialization")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and non-negative")


@dataclass(frozen=True)
class ComputeConfig:
    """Set memory-sized shards without changing the effective neural batch."""

    physical_batch: int = 32
    trace_batch: int = 8
    path_shard: int = 64
    e_fit_batch: int = 64
    em_block: int = 4096
    mc_draw_shard: int = 64

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("Compute shards must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    """Identify one dataset/seed pair and its independent output directory."""

    dataset: str = "K4_C5"
    seed: int = 0
    data_root: Path = Path("data")
    output_root: Path = Path("runs/paper")
    accelerator: str = "cuda"
    components: int = 5
    split_seed: int = 42
    split_protocol: str = "official"
    packed_filename: str = "events.parquet"


@dataclass(frozen=True)
class ExperimentConfig:
    """Bind model, protocol, data and compute settings without experimental switches."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path):
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(payload) - {"model", "training", "compute", "runtime"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        runtime = dict(payload.get("runtime", {}))
        for key in ("data_root", "output_root"):
            if key in runtime:
                runtime[key] = Path(runtime[key])
        return cls(
            ModelConfig(**payload.get("model", {})),
            TrainingConfig(**payload.get("training", {})),
            ComputeConfig(**payload.get("compute", {})),
            RuntimeConfig(**runtime),
        )

    def as_dict(self):
        result = asdict(self)
        for key in ("data_root", "output_root"):
            result["runtime"][key] = str(result["runtime"][key])
        return result
