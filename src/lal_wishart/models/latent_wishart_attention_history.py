"""Latent-Wishart attention for horizon-aware THP and COTIC mixtures."""

from __future__ import annotations

from typing import Iterable

from torch import Tensor

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_nhp import (
    LatentWishartAttentionNHP,
)
from lal_wishart.models.reference_history_mixtures import (
    ReferenceHistoryMixture,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
)


class LatentWishartAttentionHistory(LatentWishartAttentionNHP):
    """Use the same distributional Wishart law with another TPP backbone."""

    backbone: ReferenceHistoryMixture

    def __init__(
        self,
        backbone: ReferenceHistoryMixture,
        *,
        degrees_of_freedom: int,
        minimum_diagonal: float = 1e-5,
    ) -> None:
        super().__init__(
            backbone,
            degrees_of_freedom=degrees_of_freedom,
            minimum_diagonal=minimum_diagonal,
        )

    def sampled_component_scores(
        self,
        sequences: Iterable[MarkedSequence],
        *,
        n_samples: int,
        sample_seed: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence_list = tuple(sequences)
        trace = self.backbone.build_trace(sequence_list)
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
