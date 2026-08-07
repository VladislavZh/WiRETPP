"""NHP LaL matched to the cluster-specific structure of upstream Grid LaL."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lal_wishart.models.neural_hawkes import (
    NeuralHawkesMixture,
    NHPState,
    inverse_softplus,
)


class ReferenceNeuralHawkesMixture(NeuralHawkesMixture):
    """Grid-free NHP with one scaled-softplus link per LaL component."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Replace the shared EasyTPP mark scale by the upstream LaL
        # cluster-specific scalar link while keeping a shared linear decoder.
        del self.intensity_link
        self.raw_component_scales = nn.Parameter(
            torch.full(
                (self.n_components,),
                inverse_softplus(1.0 - 1e-4),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def component_scales(self) -> Tensor:
        return F.softplus(self.raw_component_scales) + 1e-4

    def intensities_from_hidden(
        self,
        hidden: Tensor,
        component_index: int | None = None,
    ) -> Tensor:
        if component_index is None:
            if self.n_components != 1:
                raise ValueError("component_index is required for K>1")
            component_index = 0
        scale = self.component_scales()[component_index]
        logits = self.intensity_linear(hidden)
        return scale * F.softplus(logits / scale) + 1e-10

    def component_scores_for_indices(
        self,
        sequences,
        component_indices,
    ) -> Tensor:
        """Vectorize exact CT-LSTM scores over the component dimension."""

        sequence_list = tuple(sequences)
        indices = tuple(int(index) for index in component_indices)
        if not indices:
            raise ValueError("component_indices must be non-empty")
        if any(
            index < 0 or index >= self.n_components for index in indices
        ):
            raise IndexError("component index out of range")
        times, marks, mask = self._prepare_sequence_batch(sequence_list)
        horizons = self._sequence_horizons(sequence_list)
        index = torch.as_tensor(
            indices,
            dtype=torch.int64,
            device=self.device,
        )
        batch_size, maximum_events = times.shape
        n_selected = len(indices)
        state = NHPState(
            cell=self.initial_cell[index][None, :, :].expand(
                batch_size,
                -1,
                -1,
            ),
            cell_bar=self.initial_cell_bar[index][None, :, :].expand(
                batch_size,
                -1,
                -1,
            ),
            decay=(
                F.softplus(self.initial_raw_decay[index]) + 1e-8
            )[None, :, :].expand(batch_size, -1, -1),
            output_gate=torch.sigmoid(
                self.initial_output_logits[index]
            )[None, :, :].expand(batch_size, -1, -1),
        )
        previous_time = times.new_zeros(batch_size)
        event_term = times.new_zeros((batch_size, n_selected))
        compensator = times.new_zeros((batch_size, n_selected))
        selected_scales = self.component_scales()[index]

        def intensities(hidden: Tensor) -> Tensor:
            logits = self.intensity_linear(hidden)
            scale_shape = (1, n_selected) + (1,) * (logits.ndim - 2)
            scale = selected_scales.reshape(scale_shape)
            return scale * F.softplus(logits / scale) + 1e-10

        def integrate(gap: Tensor, current: NHPState) -> Tensor:
            elapsed = (
                0.5
                * gap[:, None]
                * (self._quadrature_nodes[None, :] + 1.0)
            )
            weights = (
                0.5
                * gap[:, None]
                * self._quadrature_weights[None, :]
            )
            _, hidden = self.recurrent.decay(
                current.cell[:, :, None, :],
                current.cell_bar[:, :, None, :],
                current.decay[:, :, None, :],
                current.output_gate[:, :, None, :],
                elapsed[:, None, :, None],
            )
            return (
                weights[:, None, :, None] * intensities(hidden)
            ).sum(dim=(2, 3))

        for event_index in range(maximum_events):
            active = mask[:, event_index]
            event_time = times[:, event_index]
            gap = torch.where(
                active,
                event_time - previous_time,
                torch.zeros_like(previous_time),
            )
            compensator = compensator + integrate(gap, state)
            cell_left, hidden_left = self.recurrent.decay(
                state.cell,
                state.cell_bar,
                state.decay,
                state.output_gate,
                gap[:, None, None],
            )
            event_intensity = intensities(hidden_left)
            selected = event_intensity.gather(
                2,
                marks[:, event_index, None, None].expand(
                    -1,
                    n_selected,
                    1,
                ),
            ).squeeze(2)
            event_term = event_term + torch.where(
                active[:, None],
                torch.log(selected + 1e-12),
                torch.zeros_like(selected),
            )
            embedding = self.event_embedding(
                marks[:, event_index]
            )[:, None, :].expand(-1, n_selected, -1)
            cell, cell_bar, decay, output_gate = self.recurrent(
                embedding,
                hidden_left,
                cell_left,
                state.cell_bar,
            )
            active_column = active[:, None, None]
            state = NHPState(
                cell=torch.where(active_column, cell, state.cell),
                cell_bar=torch.where(
                    active_column,
                    cell_bar,
                    state.cell_bar,
                ),
                decay=torch.where(active_column, decay, state.decay),
                output_gate=torch.where(
                    active_column,
                    output_gate,
                    state.output_gate,
                ),
            )
            previous_time = torch.where(
                active,
                event_time,
                previous_time,
            )

        compensator = compensator + integrate(
            horizons - previous_time,
            state,
        )
        return event_term - compensator


def split_reference_nhp_k1(
    fitted_k1: ReferenceNeuralHawkesMixture,
    n_components: int,
    *,
    initialization_seed: int,
) -> ReferenceNeuralHawkesMixture:
    """Create K NHP components using repeated upstream-style splits."""

    if fitted_k1.n_components != 1 or n_components <= 1:
        raise ValueError("split requires a K=1 source and K>1 target")
    current = fitted_k1
    for offset in range(n_components - 1):
        current = split_reference_nhp_once(
            current,
            initialization_seed=initialization_seed + offset,
        )
    return current


def split_reference_nhp_once(
    fitted: ReferenceNeuralHawkesMixture,
    *,
    initialization_seed: int,
) -> ReferenceNeuralHawkesMixture:
    """Split the largest-state component (legacy deterministic convenience)."""

    split_index = int(
        max(
            range(fitted.n_components),
            key=lambda index: float(
                torch.linalg.vector_norm(
                    fitted.initial_cell[index]
                ).detach().cpu()
                + torch.linalg.vector_norm(
                    fitted.initial_cell_bar[index]
                ).detach().cpu()
            ),
        )
    )
    return split_reference_nhp_component(
        fitted,
        split_index,
        initialization_seed=initialization_seed,
    )


def _new_reference_nhp_like(
    fitted: ReferenceNeuralHawkesMixture,
    n_components: int,
    *,
    initialization_seed: int,
) -> ReferenceNeuralHawkesMixture:
    target = ReferenceNeuralHawkesMixture(
        n_components,
        fitted.n_marks,
        horizon=fitted.horizon,
        hidden_size=fitted.hidden_size,
        quadrature_order=fitted.quadrature_order,
        initialization_seed=initialization_seed,
        dtype=fitted.dtype,
    ).to(fitted.device)
    target.event_embedding.load_state_dict(fitted.event_embedding.state_dict())
    target.recurrent.load_state_dict(fitted.recurrent.state_dict())
    target.intensity_linear.load_state_dict(
        fitted.intensity_linear.state_dict()
    )
    with torch.no_grad():
        target.mixture_logits.zero_()
    return target


def split_reference_nhp_component(
    fitted: ReferenceNeuralHawkesMixture,
    component_index: int,
    *,
    initialization_seed: int,
    beta: float | None = None,
) -> ReferenceNeuralHawkesMixture:
    """Apply the LaL multiplicative split to one NHP initial state."""

    if not 0 <= component_index < fitted.n_components:
        raise IndexError("component index out of range")
    if beta is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(initialization_seed)
        beta = float(torch.rand((), generator=generator, dtype=fitted.dtype))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")

    n_components = fitted.n_components + 1
    target = _new_reference_nhp_like(
        fitted,
        n_components,
        initialization_seed=initialization_seed,
    )
    cells = [
        fitted.initial_cell[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    bars = [
        fitted.initial_cell_bar[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    source_cell = cells[component_index]
    source_bar = bars[component_index]
    cells[component_index] = 2.0 * beta * source_cell
    bars[component_index] = 2.0 * beta * source_bar
    cells.append(2.0 * (1.0 - beta) * source_cell)
    bars.append(2.0 * (1.0 - beta) * source_bar)
    raw_decay = [
        fitted.initial_raw_decay[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    raw_decay.append(raw_decay[component_index].clone())
    output_logits = [
        fitted.initial_output_logits[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    output_logits.append(output_logits[component_index].clone())
    scales = [
        fitted.raw_component_scales[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    scales.append(scales[component_index].clone())
    with torch.no_grad():
        target.initial_cell.copy_(torch.stack(cells))
        target.initial_cell_bar.copy_(torch.stack(bars))
        target.initial_raw_decay.copy_(torch.stack(raw_decay))
        target.initial_output_logits.copy_(torch.stack(output_logits))
        target.raw_component_scales.copy_(torch.stack(scales))
    return target

