"""Distributional latent-Wishart attention for a Neural Hawkes mixture."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.neural_hawkes import NHPState
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    identity_raw_cholesky,
    inverse_softplus,
    monte_carlo_marginal_scores,
    squared_correlation_attention,
)


@dataclass(frozen=True)
class LatentNHPMixtureBatchTrace:
    """Classical TPP terms for all ``K*C`` base intensity channels."""

    n_paths: int
    horizon: float
    path_horizons: Tensor
    n_components: int
    n_marks: int
    event_times: Tensor
    event_marks: Tensor
    event_path_indices: Tensor
    event_base_intensities: Tensor
    quadrature_times: Tensor
    quadrature_weights: Tensor
    quadrature_path_indices: Tensor
    quadrature_base_intensities: Tensor


def build_nhp_mixture_batch_trace(
    backbone: ReferenceNeuralHawkesMixture,
    sequences: Iterable[MarkedSequence],
    *,
    boundary: float | None = None,
) -> LatentNHPMixtureBatchTrace:
    """Build a differentiable trace, vectorized over paths and components."""

    sequence_list = tuple(sequences)
    if not sequence_list:
        raise ValueError("sequences must be non-empty")
    maximum_horizon = max(sequence.horizon for sequence in sequence_list)
    if boundary is not None and not 0.0 < boundary < maximum_horizon:
        raise ValueError("boundary must lie strictly inside the horizon")
    times, marks, mask = backbone._prepare_sequence_batch(sequence_list)
    path_horizons = backbone._sequence_horizons(sequence_list)
    batch_size, maximum_events = times.shape
    n_components = backbone.n_components
    component_indices = torch.arange(
        n_components,
        dtype=torch.int64,
        device=backbone.device,
    )
    state = NHPState(
        cell=backbone.initial_cell[None, :, :].expand(batch_size, -1, -1),
        cell_bar=backbone.initial_cell_bar[None, :, :].expand(
            batch_size,
            -1,
            -1,
        ),
        decay=(
            F.softplus(backbone.initial_raw_decay) + 1e-8
        )[None, :, :].expand(batch_size, -1, -1),
        output_gate=torch.sigmoid(
            backbone.initial_output_logits
        )[None, :, :].expand(batch_size, -1, -1),
    )
    component_scales = backbone.component_scales()[component_indices]

    def intensities(hidden: Tensor) -> Tensor:
        logits = backbone.intensity_linear(hidden)
        scale_shape = (1, n_components) + (1,) * (logits.ndim - 2)
        scale = component_scales.reshape(scale_shape)
        return scale * F.softplus(logits / scale) + 1e-10

    previous_time = times.new_zeros(batch_size)
    path_ids = torch.arange(
        batch_size,
        dtype=torch.int64,
        device=backbone.device,
    )
    event_times: list[Tensor] = []
    event_marks: list[Tensor] = []
    event_paths: list[Tensor] = []
    event_intensities: list[Tensor] = []
    quadrature_times: list[Tensor] = []
    quadrature_weights: list[Tensor] = []
    quadrature_paths: list[Tensor] = []
    quadrature_intensities: list[Tensor] = []

    def append_segment(
        segment_start: Tensor,
        segment_end: Tensor,
        state_origin: Tensor,
        current: NHPState,
        valid: Tensor,
    ) -> None:
        length = (segment_end - segment_start).clamp_min(0.0)
        selected = valid & (length > 1e-12)
        if not bool(selected.any()):
            return
        local_elapsed = (
            0.5
            * length[:, None]
            * (backbone._quadrature_nodes[None, :] + 1.0)
        )
        elapsed_from_state = (
            segment_start - state_origin
        )[:, None] + local_elapsed
        weights = (
            0.5
            * length[:, None]
            * backbone._quadrature_weights[None, :]
        )
        _, hidden = backbone.recurrent.decay(
            current.cell[:, :, None, :],
            current.cell_bar[:, :, None, :],
            current.decay[:, :, None, :],
            current.output_gate[:, :, None, :],
            elapsed_from_state[:, None, :, None],
        )
        base = intensities(hidden)
        quadrature_times.append(
            (segment_start[:, None] + local_elapsed)[selected].reshape(-1)
        )
        quadrature_weights.append(weights[selected].reshape(-1))
        quadrature_paths.append(
            path_ids[selected, None]
            .expand(-1, backbone.quadrature_order)
            .reshape(-1)
        )
        quadrature_intensities.append(
            base[selected]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(-1, n_components, backbone.n_marks)
        )

    def append_interval(
        interval_end: Tensor,
        current: NHPState,
        active: Tensor,
    ) -> None:
        if boundary is None:
            append_segment(
                previous_time,
                interval_end,
                previous_time,
                current,
                active,
            )
            return
        cutoff = previous_time.new_full(previous_time.shape, boundary)
        append_segment(
            previous_time,
            torch.minimum(interval_end, cutoff),
            previous_time,
            current,
            active & (previous_time < cutoff),
        )
        append_segment(
            torch.maximum(previous_time, cutoff),
            interval_end,
            previous_time,
            current,
            active & (interval_end > cutoff),
        )

    for event_index in range(maximum_events):
        active = mask[:, event_index]
        event_time = times[:, event_index]
        effective_end = torch.where(active, event_time, previous_time)
        append_interval(effective_end, state, active)
        gap = effective_end - previous_time
        cell_left, hidden_left = backbone.recurrent.decay(
            state.cell,
            state.cell_bar,
            state.decay,
            state.output_gate,
            gap[:, None, None],
        )
        base_event = intensities(hidden_left)
        if bool(active.any()):
            event_times.append(event_time[active])
            event_marks.append(marks[active, event_index])
            event_paths.append(path_ids[active])
            event_intensities.append(base_event[active])
        embedding = backbone.event_embedding(
            marks[:, event_index]
        )[:, None, :].expand(-1, n_components, -1)
        cell, cell_bar, decay, output_gate = backbone.recurrent(
            embedding,
            hidden_left,
            cell_left,
            state.cell_bar,
        )
        active_column = active[:, None, None]
        state = NHPState(
            cell=torch.where(active_column, cell, state.cell),
            cell_bar=torch.where(active_column, cell_bar, state.cell_bar),
            decay=torch.where(active_column, decay, state.decay),
            output_gate=torch.where(
                active_column,
                output_gate,
                state.output_gate,
            ),
        )
        previous_time = effective_end

    append_interval(
        path_horizons,
        state,
        torch.ones_like(previous_time, dtype=torch.bool),
    )
    if event_times:
        event_times_tensor = torch.cat(event_times)
        event_marks_tensor = torch.cat(event_marks)
        event_paths_tensor = torch.cat(event_paths)
        event_base_tensor = torch.cat(event_intensities)
    else:
        event_times_tensor = times.new_empty(0)
        event_marks_tensor = torch.empty(
            0,
            dtype=torch.int64,
            device=backbone.device,
        )
        event_paths_tensor = event_marks_tensor.clone()
        event_base_tensor = times.new_empty(
            (0, n_components, backbone.n_marks)
        )
    return LatentNHPMixtureBatchTrace(
        n_paths=batch_size,
        horizon=float(maximum_horizon),
        path_horizons=path_horizons,
        n_components=n_components,
        n_marks=backbone.n_marks,
        event_times=event_times_tensor,
        event_marks=event_marks_tensor,
        event_path_indices=event_paths_tensor,
        event_base_intensities=event_base_tensor,
        quadrature_times=torch.cat(quadrature_times),
        quadrature_weights=torch.cat(quadrature_weights),
        quadrature_path_indices=torch.cat(quadrature_paths),
        quadrature_base_intensities=torch.cat(quadrature_intensities),
    )


def base_nhp_component_scores_from_trace(
    trace: LatentNHPMixtureBatchTrace,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> Tensor:
    """Score the unmodified K-component NHP on a trace."""

    stop = trace.horizon if end_time is None else float(end_time)
    if not 0.0 <= start_time < stop <= trace.horizon:
        raise ValueError("invalid scoring interval")
    scores = trace.event_base_intensities.new_zeros(
        (trace.n_paths, trace.n_components)
    )
    event_selection = trace.event_times >= start_time
    if end_time is not None:
        event_selection = event_selection & (trace.event_times < stop)
    if bool(event_selection.any()):
        base = trace.event_base_intensities[event_selection]
        marked = base.gather(
            2,
            trace.event_marks[event_selection, None, None].expand(
                -1,
                trace.n_components,
                1,
            ),
        ).squeeze(2)
        scores = scores.index_add(
            0,
            trace.event_path_indices[event_selection],
            torch.log(marked + 1e-12),
        )
    quadrature_selection = trace.quadrature_times >= start_time
    if end_time is not None:
        quadrature_selection = quadrature_selection & (
            trace.quadrature_times < stop
        )
    if bool(quadrature_selection.any()):
        weighted = (
            trace.quadrature_weights[quadrature_selection, None]
            * trace.quadrature_base_intensities[
                quadrature_selection
            ].sum(dim=2)
        )
        scores = scores.index_add(
            0,
            trace.quadrature_path_indices[quadrature_selection],
            -weighted,
        )
    return scores


def sampled_nhp_component_scores(
    trace: LatentNHPMixtureBatchTrace,
    sampled_matrices: Tensor,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> Tensor:
    """Return scores for the snapshot's squared-correlation transformation."""

    def transform(base: Tensor, matrices: Tensor) -> Tensor:
        return torch.einsum(
            "...sij,...j->...si",
            squared_correlation_attention(matrices),
            base,
        )

    return sampled_component_scores_from_transform(
        trace,
        sampled_matrices,
        transform,
        start_time=start_time,
        end_time=end_time,
    )


def sampled_component_scores_from_transform(
    trace: LatentNHPMixtureBatchTrace,
    sampled_matrices: Tensor,
    transform: Callable[[Tensor, Tensor], Tensor],
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> Tensor:
    """Score transformed positive rates with shape ``(paths, samples, K)``.

    ``transform`` receives flattened base rates with shape ``(rows, K*C)``
    and the corresponding path-specific matrices with shape
    ``(rows, samples, K*C, K*C)``.  It must return transformed rates with
    shape ``(rows, samples, K*C)``.
    """

    stop = trace.horizon if end_time is None else float(end_time)
    if not 0.0 <= start_time < stop <= trace.horizon:
        raise ValueError("invalid scoring interval")
    dimension = trace.n_components * trace.n_marks
    if (
        sampled_matrices.ndim != 4
        or sampled_matrices.shape[0] != trace.n_paths
        or sampled_matrices.shape[2:] != (dimension, dimension)
    ):
        raise ValueError("sampled matrices and trace are incompatible")
    n_samples = sampled_matrices.shape[1]
    scores = sampled_matrices.new_zeros(
        (trace.n_paths, n_samples, trace.n_components)
    )

    event_selection = trace.event_times >= start_time
    if end_time is not None:
        event_selection = event_selection & (trace.event_times < stop)
    if bool(event_selection.any()):
        paths = trace.event_path_indices[event_selection]
        base = trace.event_base_intensities[event_selection].reshape(
            -1,
            dimension,
        )
        transformed = transform(base, sampled_matrices[paths])
        expected = (base.shape[0], n_samples, dimension)
        if transformed.shape != expected:
            raise RuntimeError(
                f"rate transform returned {tuple(transformed.shape)}, "
                f"expected {expected}"
            )
        transformed = transformed.reshape(
            -1,
            n_samples,
            trace.n_components,
            trace.n_marks,
        )
        marked = transformed.gather(
            3,
            trace.event_marks[event_selection, None, None, None].expand(
                -1,
                n_samples,
                trace.n_components,
                1,
            ),
        ).squeeze(3)
        scores = scores.index_add(
            0,
            paths,
            torch.log(marked + 1e-12),
        )

    quadrature_selection = trace.quadrature_times >= start_time
    if end_time is not None:
        quadrature_selection = quadrature_selection & (
            trace.quadrature_times < stop
        )
    if bool(quadrature_selection.any()):
        paths = trace.quadrature_path_indices[quadrature_selection]
        base = trace.quadrature_base_intensities[
            quadrature_selection
        ].reshape(-1, dimension)
        transformed = transform(base, sampled_matrices[paths])
        expected = (base.shape[0], n_samples, dimension)
        if transformed.shape != expected:
            raise RuntimeError(
                f"rate transform returned {tuple(transformed.shape)}, "
                f"expected {expected}"
            )
        transformed = transformed.reshape(
            -1,
            n_samples,
            trace.n_components,
            trace.n_marks,
        )
        weighted = (
            trace.quadrature_weights[
                quadrature_selection, None, None
            ]
            * transformed.sum(dim=3)
        )
        scores = scores.index_add(0, paths, -weighted)
    return scores


class LatentWishartAttentionNHP(nn.Module):
    """Learn a global Wishart law and integrate trajectory-level ``W``."""

    def __init__(
        self,
        backbone: ReferenceNeuralHawkesMixture,
        *,
        degrees_of_freedom: int | float,
        learnable_degrees_of_freedom: bool = False,
        maximum_degrees_of_freedom: float | None = None,
        minimum_diagonal: float = 1e-5,
    ) -> None:
        super().__init__()
        if backbone.n_components <= 1:
            raise ValueError("latent Wishart attention requires K > 1")
        dimension = backbone.n_components * backbone.n_marks
        initial_nu = float(degrees_of_freedom)
        if initial_nu < dimension:
            raise ValueError("degrees of freedom must be at least K*C")
        if not learnable_degrees_of_freedom and not initial_nu.is_integer():
            raise ValueError("fixed degrees of freedom must be an integer")
        self.backbone = copy.deepcopy(backbone)
        self.minimum_diagonal = float(minimum_diagonal)
        self._fixed_degrees_of_freedom = (
            None if learnable_degrees_of_freedom else int(initial_nu)
        )
        self._degrees_of_freedom_lower_bound = float(dimension)
        self.maximum_degrees_of_freedom = float(
            10 * dimension
            if maximum_degrees_of_freedom is None
            else maximum_degrees_of_freedom
        )
        if learnable_degrees_of_freedom:
            if not (
                dimension < initial_nu < self.maximum_degrees_of_freedom
            ):
                raise ValueError(
                    "learnable degrees of freedom must start strictly between "
                    "K*C and its maximum"
                )
            raw_nu = math.log(
                (initial_nu - dimension)
                / (self.maximum_degrees_of_freedom - initial_nu)
            )
            self.raw_degrees_of_freedom = nn.Parameter(
                torch.tensor(
                    raw_nu,
                    dtype=backbone.dtype,
                    device=backbone.device,
                )
            )
        else:
            if self.maximum_degrees_of_freedom < initial_nu:
                raise ValueError(
                    "maximum degrees of freedom cannot be below fixed nu"
                )
            self.register_parameter("raw_degrees_of_freedom", None)
        self.raw_mean_cholesky = nn.Parameter(
            identity_raw_cholesky(
                1,
                dimension,
                dtype=backbone.dtype,
                device=backbone.device,
                minimum_diagonal=minimum_diagonal,
            )[0]
        )

    @property
    def n_components(self) -> int:
        return self.backbone.n_components

    @property
    def n_marks(self) -> int:
        return self.backbone.n_marks

    @property
    def dimension(self) -> int:
        return self.n_components * self.n_marks

    @property
    def device(self) -> torch.device:
        return self.raw_mean_cholesky.device

    @property
    def dtype(self) -> torch.dtype:
        return self.raw_mean_cholesky.dtype

    @property
    def learns_degrees_of_freedom(self) -> bool:
        return self.raw_degrees_of_freedom is not None

    def degrees_of_freedom_tensor(self) -> Tensor:
        """Return continuous ``nu`` while keeping fixed-mode compatibility."""

        if self.raw_degrees_of_freedom is None:
            return self.raw_mean_cholesky.new_tensor(
                float(self._fixed_degrees_of_freedom)
            )
        lower = self.raw_degrees_of_freedom.new_tensor(
            self._degrees_of_freedom_lower_bound
        )
        upper = self.raw_degrees_of_freedom.new_tensor(
            self.maximum_degrees_of_freedom
        )
        return lower + (upper - lower) * torch.sigmoid(
            self.raw_degrees_of_freedom
        )

    @property
    def degrees_of_freedom(self) -> int | float:
        if self.raw_degrees_of_freedom is None:
            return int(self._fixed_degrees_of_freedom)
        return float(self.degrees_of_freedom_tensor().detach().cpu())

    def mean_cholesky(self) -> Tensor:
        lower = torch.tril(self.raw_mean_cholesky, diagonal=-1)
        diagonal = (
            F.softplus(self.raw_mean_cholesky.diagonal())
            + self.minimum_diagonal
        )
        cholesky = lower + torch.diag(diagonal)
        trace = cholesky.square().sum()
        return cholesky * torch.sqrt(
            cholesky.new_tensor(float(self.dimension)) / trace
        )

    def mean_matrix(self) -> Tensor:
        cholesky = self.mean_cholesky()
        return cholesky @ cholesky.transpose(0, 1)

    def prune_component(
        self,
        component_index: int,
        *,
        degrees_of_freedom: int | None = None,
    ) -> tuple[int, ...]:
        """Remove one C-block from the backbone and Wishart mean matrix."""

        if not hasattr(self.backbone, "prune_component"):
            raise TypeError("the Wishart backbone does not support pruning")
        if not 0 <= component_index < self.n_components:
            raise IndexError("component index out of range")
        if self.n_components <= 2:
            raise ValueError("Wishart pruning requires at least K=3")
        kept_components = tuple(
            index
            for index in range(self.n_components)
            if index != component_index
        )
        kept_dimensions = torch.as_tensor(
            [
                component * self.n_marks + mark
                for component in kept_components
                for mark in range(self.n_marks)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        new_dimension = len(kept_dimensions)
        new_nu = float(
            self.degrees_of_freedom
            if degrees_of_freedom is None
            else degrees_of_freedom
        )
        if new_nu < new_dimension:
            raise ValueError("degrees of freedom must be at least new K*C")
        with torch.no_grad():
            principal_mean = self.mean_matrix().index_select(
                0, kept_dimensions
            ).index_select(1, kept_dimensions)
            principal_mean = (
                principal_mean
                * (new_dimension / torch.trace(principal_mean))
            )
            cholesky = torch.linalg.cholesky(principal_mean)
            diagonal = cholesky.diagonal() - self.minimum_diagonal
            if bool(torch.any(diagonal <= 0.0)):
                raise RuntimeError(
                    "pruned Wishart Cholesky violates minimum diagonal"
                )
            raw = torch.tril(cholesky, diagonal=-1) + torch.diag(
                inverse_softplus(diagonal)
            )
        backbone_kept = self.backbone.prune_component(component_index)
        if tuple(backbone_kept) != kept_components:
            raise RuntimeError("backbone and Wishart pruning disagree")
        self.raw_mean_cholesky = nn.Parameter(raw.detach().clone())
        if self.learns_degrees_of_freedom:
            if not new_dimension < new_nu < self.maximum_degrees_of_freedom:
                raise ValueError(
                    "pruned learnable nu must remain strictly within bounds"
                )
            raw_nu = math.log(
                (new_nu - new_dimension)
                / (self.maximum_degrees_of_freedom - new_nu)
            )
            self._degrees_of_freedom_lower_bound = float(new_dimension)
            self.raw_degrees_of_freedom = nn.Parameter(
                raw.new_tensor(raw_nu)
            )
        else:
            if not new_nu.is_integer():
                raise ValueError("fixed degrees of freedom must be an integer")
            self._fixed_degrees_of_freedom = int(new_nu)
        return kept_components

    def mean_matrix_hyperprior_penalty(self, *, strength: float) -> Tensor:
        if strength < 0.0:
            raise ValueError("hyperprior strength must be non-negative")
        mean = self.mean_matrix()
        sign, logdet = torch.linalg.slogdet(mean)
        if bool(sign <= 0):
            raise RuntimeError("Wishart mean matrix lost positive definiteness")
        return 0.5 * strength * (
            torch.trace(mean) - logdet - self.dimension
        )

    def degrees_of_freedom_prior_penalty(
        self,
        *,
        strength: float,
        center: float,
    ) -> Tensor:
        """Weak log-scale prior used only for an explicitly learnable ``nu``."""

        if strength < 0.0:
            raise ValueError("invalid degrees-of-freedom prior")
        if not self.learns_degrees_of_freedom or strength == 0.0:
            return self.raw_mean_cholesky.new_zeros(())
        if center <= self.dimension:
            raise ValueError("learnable nu prior center must exceed K*C")
        log_ratio = torch.log(self.degrees_of_freedom_tensor() / center)
        return 0.5 * strength * log_ratio.square()

    def sample_matrices(
        self,
        batch_size: int,
        n_samples: int,
        *,
        sample_seed: int,
    ) -> Tensor:
        if batch_size <= 0 or n_samples <= 0:
            raise ValueError("batch and sample counts must be positive")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sample_seed)
        if self.learns_degrees_of_freedom:
            nu = self.degrees_of_freedom_tensor()
            shape = (batch_size, n_samples, self.dimension, self.dimension)
            standard = torch.randn(
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            bartlett = torch.tril(standard, diagonal=-1)
            diagonal_df = nu - torch.arange(
                self.dimension,
                dtype=self.dtype,
                device=self.device,
            )
            concentration = (0.5 * diagonal_df).expand(
                batch_size, n_samples, self.dimension
            )
            chi_squared = 2.0 * torch._standard_gamma(
                concentration,
                generator=generator,
            )
            bartlett = bartlett + torch.diag_embed(
                torch.sqrt(chi_squared.clamp_min(torch.finfo(self.dtype).tiny))
            )
            scale_cholesky = self.mean_cholesky() / torch.sqrt(nu)
            factor = torch.matmul(scale_cholesky, bartlett)
            return factor @ factor.transpose(-1, -2)
        standard = torch.randn(
            (
                batch_size,
                n_samples,
                self.degrees_of_freedom,
                self.dimension,
            ),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        transformed = (
            torch.matmul(
                standard,
                self.mean_cholesky().transpose(0, 1),
            )
            / math.sqrt(self.degrees_of_freedom)
        )
        return transformed.transpose(-1, -2) @ transformed

    @property
    def transformation_name(self) -> str:
        return "squared_correlation_attention"

    @property
    def uses_deterministic_matrices(self) -> bool:
        return False

    def distribution_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters optimized with the Wishart-distribution learning rate."""

        return (self.raw_mean_cholesky,)

    def degrees_of_freedom_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.raw_degrees_of_freedom is None:
            return ()
        return (self.raw_degrees_of_freedom,)

    def interaction_parameters(self) -> tuple[nn.Parameter, ...]:
        """Signed-interaction parameters eligible for a separate learning rate."""

        return ()

    def component_scores_from_trace(
        self,
        trace: LatentNHPMixtureBatchTrace,
        matrices: Tensor,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Tensor:
        return sampled_nhp_component_scores(
            trace,
            matrices,
            start_time=start_time,
            end_time=end_time,
        )

    def transformation_diagnostic(self, matrices: Tensor) -> Tensor:
        """Return one channel-level diagnostic per sampled matrix."""

        return squared_correlation_attention(matrices).diagonal(
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
        trace = build_nhp_mixture_batch_trace(
            self.backbone,
            sequence_list,
        )
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

    def marginal_scores(
        self,
        sequences: Iterable[MarkedSequence],
        *,
        n_samples: int,
        sample_seed: int,
    ) -> Tensor:
        component, gates, _ = self.sampled_component_scores(
            sequences,
            n_samples=n_samples,
            sample_seed=sample_seed,
        )
        return monte_carlo_marginal_scores(component, gates)
