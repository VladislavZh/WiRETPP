"""Signed and legacy-row Wishart random effects for neural TPP mixtures."""

from __future__ import annotations

import math
from typing import Iterable, Literal

import torch
from torch import Tensor, nn

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_nhp import (
    build_nhp_mixture_batch_trace,
    LatentNHPMixtureBatchTrace,
    LatentWishartAttentionNHP,
    sampled_component_scores_from_transform,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    legacy_row_normalized_attention,
    signed_correlation_coupling,
    signed_intensity_transform,
)


def _build_trace(backbone, sequences: tuple[MarkedSequence, ...]):
    if hasattr(backbone, "build_trace"):
        return backbone.build_trace(sequences)
    return build_nhp_mixture_batch_trace(backbone, sequences)


class LegacyRowWishartTPP(LatentWishartAttentionNHP):
    """Equation (3.9): row-normalized ``(W o W)`` intensity mixing."""

    @property
    def transformation_name(self) -> str:
        return "legacy_row_normalized_w_squared"

    def component_scores_from_trace(
        self,
        trace: LatentNHPMixtureBatchTrace,
        matrices: Tensor,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Tensor:
        def transform(base: Tensor, selected: Tensor) -> Tensor:
            return torch.einsum(
                "...sij,...j->...si",
                legacy_row_normalized_attention(selected),
                base,
            )

        return sampled_component_scores_from_transform(
            trace,
            matrices,
            transform,
            start_time=start_time,
            end_time=end_time,
        )

    def transformation_diagnostic(self, matrices: Tensor) -> Tensor:
        return legacy_row_normalized_attention(matrices).diagonal(
            dim1=-2,
            dim2=-1,
        )

    def sampled_component_scores(
        self,
        sequences: Iterable[MarkedSequence],
        *,
        n_samples: int,
        sample_seed: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence_list = tuple(sequences)
        trace = _build_trace(self.backbone, sequence_list)
        matrices = self.sample_matrices(
            len(sequence_list),
            n_samples,
            sample_seed=sample_seed,
        )
        scores = self.component_scores_from_trace(trace, matrices)
        gates = cluster_log_weights_from_matrices(
            matrices,
            n_components=self.n_components,
            n_marks=self.n_marks,
        )
        return scores, gates, matrices


class SignedWishartTPP(LatentWishartAttentionNHP):
    """Equations (3.4)-(3.8) with a bounded global signed interaction."""

    def __init__(
        self,
        backbone,
        *,
        degrees_of_freedom: int | float,
        learnable_degrees_of_freedom: bool = False,
        maximum_degrees_of_freedom: float | None = None,
        alpha_max: float = 1.0,
        initial_alpha: float = 0.1,
        interaction_temperature: float = 1.0,
        fixed_alpha: float | None = None,
        alpha_parameterization: Literal["sigmoid", "projected"] = "sigmoid",
        deterministic_matrices: bool = False,
        activity_epsilon: float = 1e-8,
        interaction_mode: Literal["residual", "convex"] = "residual",
        block_local_activity: bool = False,
        within_block_gain: float = 1.0,
        between_block_gain: float = 1.0,
        minimum_diagonal: float = 1e-5,
        exploration_beta: float = 0.0,
        exploration_degrees_of_freedom: int | None = None,
        exploration_seed_offset: int = 10_000_019,
        exploration_mode: Literal["additive", "convex"] = "additive",
    ) -> None:
        if alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive")
        if interaction_temperature <= 0.0:
            raise ValueError("interaction_temperature must be positive")
        if not 0.0 <= initial_alpha <= alpha_max:
            raise ValueError("initial_alpha must lie in [0, alpha_max]")
        if fixed_alpha is not None and not 0.0 <= fixed_alpha <= alpha_max:
            raise ValueError("fixed_alpha must lie in [0, alpha_max]")
        if alpha_parameterization not in {"sigmoid", "projected"}:
            raise ValueError(
                "alpha parameterization must be sigmoid or projected"
            )
        if activity_epsilon <= 0.0:
            raise ValueError("activity_epsilon must be positive")
        if interaction_mode not in {"residual", "convex"}:
            raise ValueError("interaction mode must be residual or convex")
        if (
            not math.isfinite(within_block_gain)
            or not math.isfinite(between_block_gain)
            or within_block_gain < 0.0
            or between_block_gain < 0.0
        ):
            raise ValueError(
                "within- and between-block gains must be finite and non-negative"
            )
        if not block_local_activity and (
            within_block_gain != 1.0 or between_block_gain != 1.0
        ):
            raise ValueError("custom block gains require block-local activity")
        if exploration_beta < 0.0:
            raise ValueError("exploration beta must be non-negative")
        if exploration_mode not in {"additive", "convex"}:
            raise ValueError("exploration mode must be additive or convex")
        if exploration_mode == "convex" and exploration_beta > 1.0:
            raise ValueError("convex exploration beta must lie in [0, 1]")
        dimension = backbone.n_components * backbone.n_marks
        exploration_nu = (
            dimension
            if exploration_degrees_of_freedom is None
            else int(exploration_degrees_of_freedom)
        )
        if exploration_nu < dimension:
            raise ValueError(
                "exploration degrees of freedom must be at least K*C"
            )
        if exploration_seed_offset == 0:
            raise ValueError("exploration seed offset must be non-zero")
        if deterministic_matrices and exploration_beta > 0.0:
            raise ValueError(
                "random exploration is incompatible with deterministic matrices"
            )
        super().__init__(
            backbone,
            degrees_of_freedom=degrees_of_freedom,
            learnable_degrees_of_freedom=learnable_degrees_of_freedom,
            maximum_degrees_of_freedom=maximum_degrees_of_freedom,
            minimum_diagonal=minimum_diagonal,
        )
        self.alpha_max = float(alpha_max)
        self.interaction_temperature = float(interaction_temperature)
        self.alpha_parameterization = alpha_parameterization
        self.activity_epsilon = float(activity_epsilon)
        self.interaction_mode = interaction_mode
        self.block_local_activity = bool(block_local_activity)
        self.within_block_gain = float(within_block_gain)
        self.between_block_gain = float(between_block_gain)
        self._uses_deterministic_matrices = bool(deterministic_matrices)
        self.exploration_beta = float(exploration_beta)
        self.exploration_degrees_of_freedom = exploration_nu
        self.exploration_seed_offset = int(exploration_seed_offset)
        self.exploration_mode = exploration_mode
        if fixed_alpha is None:
            if initial_alpha in (0.0, alpha_max):
                raise ValueError(
                    "learned alpha initialization must lie strictly inside bounds"
                )
            raw = (
                initial_alpha
                if self.alpha_parameterization == "projected"
                else self.interaction_temperature
                * math.log(initial_alpha / (alpha_max - initial_alpha))
            )
            self.raw_interaction_strength = nn.Parameter(
                torch.tensor(raw, dtype=self.dtype, device=self.device)
            )
            self.register_buffer("fixed_interaction_strength", None)
        else:
            self.register_parameter("raw_interaction_strength", None)
            self.register_buffer(
                "fixed_interaction_strength",
                torch.tensor(
                    float(fixed_alpha),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )

    @property
    def transformation_name(self) -> str:
        prefix = (
            "signed"
            if self.interaction_mode == "residual"
            else "signed_convex_interaction"
        )
        if self.block_local_activity:
            prefix = f"{prefix}_block_local"
        if self.exploration_beta > 0.0:
            return f"{prefix}_{self.exploration_mode}_frozen_wishart"
        if self.fixed_interaction_strength is not None:
            if float(self.fixed_interaction_strength.detach().cpu()) == 0.0:
                return f"{prefix}_alpha_zero"
            return f"{prefix}_fixed_alpha"
        if self.uses_deterministic_matrices:
            return f"{prefix}_deterministic_mean"
        return prefix

    @property
    def uses_deterministic_matrices(self) -> bool:
        return self._uses_deterministic_matrices

    def interaction_strength(self) -> Tensor:
        if self.fixed_interaction_strength is not None:
            return self.fixed_interaction_strength
        if self.alpha_parameterization == "projected":
            return self.raw_interaction_strength
        return self.alpha_max * torch.sigmoid(
            self.raw_interaction_strength / self.interaction_temperature
        )

    @torch.no_grad()
    def project_interaction_strength_(self) -> None:
        """Project a directly optimized interaction strength onto its bounds."""

        if (
            self.raw_interaction_strength is not None
            and self.alpha_parameterization == "projected"
        ):
            self.raw_interaction_strength.clamp_(0.0, self.alpha_max)

    def distribution_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = [self.raw_mean_cholesky]
        if self.raw_interaction_strength is not None:
            parameters.append(self.raw_interaction_strength)
        return tuple(parameters)

    def interaction_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.raw_interaction_strength is None:
            return ()
        return (self.raw_interaction_strength,)

    def sample_matrices(
        self,
        batch_size: int,
        n_samples: int,
        *,
        sample_seed: int,
    ) -> Tensor:
        if not self.uses_deterministic_matrices:
            learned = super().sample_matrices(
                batch_size,
                n_samples,
                sample_seed=sample_seed,
            )
        else:
            if batch_size <= 0 or n_samples <= 0:
                raise ValueError("batch and sample counts must be positive")
            mean = self.mean_matrix()
            learned = mean[None, None].expand(
                batch_size,
                n_samples,
                self.dimension,
                self.dimension,
            )
        if self.exploration_beta == 0.0:
            return learned
        exploration = self.sample_exploration_matrices(
            batch_size,
            n_samples,
            sample_seed=sample_seed + self.exploration_seed_offset,
        )
        if self.exploration_mode == "additive":
            return learned + self.exploration_beta * exploration
        return (
            (1.0 - self.exploration_beta) * learned
            + self.exploration_beta * exploration
        )

    def sample_exploration_matrices(
        self,
        batch_size: int,
        n_samples: int,
        *,
        sample_seed: int,
    ) -> Tensor:
        """Sample the fixed isotropic Wishart exploration process."""

        if batch_size <= 0 or n_samples <= 0:
            raise ValueError("batch and sample counts must be positive")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sample_seed)
        standard = torch.randn(
            (
                batch_size,
                n_samples,
                self.exploration_degrees_of_freedom,
                self.dimension,
            ),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        ) / math.sqrt(self.exploration_degrees_of_freedom)
        return standard.transpose(-1, -2) @ standard

    def component_scores_from_trace(
        self,
        trace: LatentNHPMixtureBatchTrace,
        matrices: Tensor,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Tensor:
        def transform(base: Tensor, selected: Tensor) -> Tensor:
            return signed_intensity_transform(
                base,
                selected,
                self.interaction_strength(),
                activity_epsilon=self.activity_epsilon,
                interaction_mode=self.interaction_mode,
                n_components=self.n_components,
                n_marks=self.n_marks,
                block_local_activity=self.block_local_activity,
                within_block_gain=self.within_block_gain,
                between_block_gain=self.between_block_gain,
            )

        return sampled_component_scores_from_transform(
            trace,
            matrices,
            transform,
            start_time=start_time,
            end_time=end_time,
        )

    def transformation_diagnostic(self, matrices: Tensor) -> Tensor:
        return signed_correlation_coupling(matrices).abs().mean(dim=-2)

    def sampled_component_scores(
        self,
        sequences: Iterable[MarkedSequence],
        *,
        n_samples: int,
        sample_seed: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence_list = tuple(sequences)
        trace = _build_trace(self.backbone, sequence_list)
        matrices = self.sample_matrices(
            len(sequence_list),
            n_samples,
            sample_seed=sample_seed,
        )
        scores = self.component_scores_from_trace(trace, matrices)
        gates = cluster_log_weights_from_matrices(
            matrices,
            n_components=self.n_components,
            n_marks=self.n_marks,
        )
        return scores, gates, matrices
