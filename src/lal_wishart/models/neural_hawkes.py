"""Grid-free Neural Hawkes mixture used as the shared LaL backbone.

The continuous-time LSTM follows the seven-gate Neural Hawkes Process
implementation in EasyTPP.  Unlike the historical ``GridLaL``, observations
are never binned: event terms are evaluated at exact left limits and the
classical TPP compensator is integrated on every exact inter-event interval
with deterministic Gauss--Legendre quadrature.

References
----------
Mei and Eisner (2017), "The Neural Hawkes Process".
EasyTPP ``torch_nhp.py`` (Apache-2.0), equations 4--8.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lal_wishart.data.hawkes_branching import MarkedSequence


def inverse_softplus(value: float) -> float:
    """Stable inverse of softplus for a positive scalar."""

    if value <= 0:
        raise ValueError("softplus inverse requires a positive value")
    if value > 30.0:
        return value
    return math.log(math.expm1(value))


class ScaledSoftplus(nn.Module):
    """Positive mark-specific intensity link used by EasyTPP's NHP."""

    def __init__(
        self,
        n_marks: int,
        *,
        initial_scale: float = 1.0,
        minimum_scale: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_marks <= 0 or initial_scale <= minimum_scale:
            raise ValueError("invalid scaled-softplus configuration")
        self.minimum_scale = float(minimum_scale)
        raw = inverse_softplus(initial_scale - minimum_scale)
        self.raw_scale = nn.Parameter(torch.full((n_marks,), raw, dtype=dtype))

    def scale(self) -> Tensor:
        return F.softplus(self.raw_scale) + self.minimum_scale

    def forward(self, values: Tensor) -> Tensor:
        scale = self.scale()
        return F.softplus(values * scale) / scale


class ContinuousTimeLSTMCell(nn.Module):
    """Seven-gate continuous-time LSTM cell from the Neural Hawkes Process."""

    def __init__(
        self,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            2 * hidden_size,
            7 * hidden_size,
            bias=True,
            dtype=dtype,
        )

    def forward(
        self,
        event_embedding: Tensor,
        hidden_left: Tensor,
        cell_left: Tensor,
        cell_bar_previous: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gates = self.linear(torch.cat((event_embedding, hidden_left), dim=-1))
        (
            input_gate,
            input_bar_gate,
            forget_gate,
            forget_bar_gate,
            candidate,
            output_gate,
            raw_decay,
        ) = gates.chunk(7, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        input_bar_gate = torch.sigmoid(input_bar_gate)
        forget_gate = torch.sigmoid(forget_gate)
        forget_bar_gate = torch.sigmoid(forget_bar_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        decay = F.softplus(raw_decay) + 1e-8
        cell = forget_gate * cell_left + input_gate * candidate
        cell_bar = (
            forget_bar_gate * cell_bar_previous
            + input_bar_gate * candidate
        )
        return cell, cell_bar, decay, output_gate

    @staticmethod
    def decay(
        cell: Tensor,
        cell_bar: Tensor,
        decay: Tensor,
        output_gate: Tensor,
        elapsed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cell_at_time = cell_bar + (cell - cell_bar) * torch.exp(
            -decay * elapsed
        )
        hidden_at_time = output_gate * torch.tanh(cell_at_time)
        return cell_at_time, hidden_at_time


@dataclass(frozen=True)
class NHPState:
    cell: Tensor
    cell_bar: Tensor
    decay: Tensor
    output_gate: Tensor


@dataclass(frozen=True)
class NHPComponentTrace:
    """All terms needed for a classical marked-TPP component score."""

    event_times: Tensor
    event_marks: Tensor
    event_base_intensities: Tensor
    quadrature_times: Tensor
    quadrature_weights: Tensor
    quadrature_base_intensities: Tensor

    @property
    def log_likelihood(self) -> Tensor:
        if self.event_marks.numel():
            event_term = torch.log(
                self.event_base_intensities[
                    torch.arange(
                        self.event_marks.numel(),
                        device=self.event_marks.device,
                    ),
                    self.event_marks,
                ]
                + 1e-12
            ).sum()
        else:
            event_term = self.quadrature_base_intensities.new_zeros(())
        compensator = (
            self.quadrature_weights[:, None]
            * self.quadrature_base_intensities
        ).sum()
        return event_term - compensator


class NeuralHawkesMixture(nn.Module):
    """LaL mixture with one cluster-specific initial NHP state.

    The event embedding, continuous-time LSTM transition, and intensity
    decoder are shared by all components.  Only the initial state and mixture
    weights are component-specific.
    """

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        horizon: float,
        hidden_size: int = 24,
        quadrature_order: int = 12,
        initial_total_rate: float = 0.7,
        initialization_seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_components <= 0 or n_marks <= 0 or hidden_size <= 0:
            raise ValueError("component, mark, and hidden counts must be positive")
        if horizon <= 0 or quadrature_order <= 0 or initial_total_rate <= 0:
            raise ValueError("horizon, quadrature order, and rate must be positive")
        self.n_components = int(n_components)
        self.n_marks = int(n_marks)
        self.horizon = float(horizon)
        self.hidden_size = int(hidden_size)
        self.quadrature_order = int(quadrature_order)
        self.dtype = dtype

        self.event_embedding = nn.Embedding(
            n_marks,
            hidden_size,
            dtype=dtype,
        )
        self.recurrent = ContinuousTimeLSTMCell(hidden_size, dtype=dtype)
        self.intensity_linear = nn.Linear(
            hidden_size,
            n_marks,
            bias=True,
            dtype=dtype,
        )
        self.intensity_link = ScaledSoftplus(n_marks, dtype=dtype)

        self.initial_cell = nn.Parameter(
            torch.zeros((n_components, hidden_size), dtype=dtype)
        )
        self.initial_cell_bar = nn.Parameter(
            torch.zeros((n_components, hidden_size), dtype=dtype)
        )
        self.initial_raw_decay = nn.Parameter(
            torch.full(
                (n_components, hidden_size),
                inverse_softplus(1.0),
                dtype=dtype,
            )
        )
        self.initial_output_logits = nn.Parameter(
            torch.full((n_components, hidden_size), 2.0, dtype=dtype)
        )
        self.mixture_logits = nn.Parameter(
            torch.zeros(n_components, dtype=dtype)
        )

        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.register_buffer(
            "_quadrature_nodes",
            torch.as_tensor(nodes, dtype=dtype),
        )
        self.register_buffer(
            "_quadrature_weights",
            torch.as_tensor(weights, dtype=dtype),
        )
        self.reset_parameters(
            initialization_seed=initialization_seed,
            initial_total_rate=initial_total_rate,
        )

    def reset_parameters(
        self,
        *,
        initialization_seed: int,
        initial_total_rate: float,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(initialization_seed)
        with torch.no_grad():
            self.event_embedding.weight.copy_(
                0.08
                * torch.randn(
                    self.event_embedding.weight.shape,
                    dtype=self.dtype,
                    generator=generator,
                )
            )
            self.recurrent.linear.weight.copy_(
                0.06
                * torch.randn(
                    self.recurrent.linear.weight.shape,
                    dtype=self.dtype,
                    generator=generator,
                )
            )
            self.recurrent.linear.bias.zero_()
            self.intensity_linear.weight.copy_(
                0.05
                * torch.randn(
                    self.intensity_linear.weight.shape,
                    dtype=self.dtype,
                    generator=generator,
                )
            )
            rate_per_mark = initial_total_rate / self.n_marks
            self.intensity_linear.bias.fill_(inverse_softplus(rate_per_mark))
            initial = 0.03 * torch.randn(
                self.initial_cell.shape,
                dtype=self.dtype,
                generator=generator,
            )
            self.initial_cell.copy_(initial)
            self.initial_cell_bar.copy_(initial)
            self.initial_raw_decay.fill_(inverse_softplus(1.0))
            self.initial_output_logits.fill_(2.0)
            self.mixture_logits.zero_()

    @property
    def device(self) -> torch.device:
        return self.mixture_logits.device

    def mixture_log_weights(self) -> Tensor:
        centered = self.mixture_logits - self.mixture_logits.mean()
        return torch.log_softmax(centered, dim=0)

    def initial_state(self, component_index: int) -> NHPState:
        if not 0 <= component_index < self.n_components:
            raise IndexError("component index out of range")
        return NHPState(
            cell=self.initial_cell[component_index],
            cell_bar=self.initial_cell_bar[component_index],
            decay=F.softplus(self.initial_raw_decay[component_index]) + 1e-8,
            output_gate=torch.sigmoid(
                self.initial_output_logits[component_index]
            ),
        )

    def intensities_from_hidden(
        self,
        hidden: Tensor,
        component_index: int | None = None,
    ) -> Tensor:
        return self.intensity_link(self.intensity_linear(hidden)) + 1e-10

    def _interval_quadrature(
        self,
        state: NHPState,
        *,
        component_index: int,
        start_time: float,
        interval_length: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if interval_length < -1e-12:
            raise ValueError("event times must be sorted")
        length = max(float(interval_length), 0.0)
        elapsed = 0.5 * length * (self._quadrature_nodes + 1.0)
        weights = 0.5 * length * self._quadrature_weights
        _, hidden = self.recurrent.decay(
            state.cell[None, :],
            state.cell_bar[None, :],
            state.decay[None, :],
            state.output_gate[None, :],
            elapsed[:, None],
        )
        intensities = self.intensities_from_hidden(hidden, component_index)
        times = elapsed + float(start_time)
        return times, weights, intensities

    def _left_limit(
        self,
        state: NHPState,
        elapsed: float,
    ) -> tuple[Tensor, Tensor]:
        elapsed_tensor = state.cell.new_tensor(float(elapsed))
        return self.recurrent.decay(
            state.cell,
            state.cell_bar,
            state.decay,
            state.output_gate,
            elapsed_tensor,
        )

    def component_trace(
        self,
        sequence: MarkedSequence,
        component_index: int,
    ) -> NHPComponentTrace:
        if not math.isclose(sequence.horizon, self.horizon, abs_tol=1e-10):
            raise ValueError("sequence horizon does not match model horizon")
        if np.any((sequence.marks < 0) | (sequence.marks >= self.n_marks)):
            raise ValueError("sequence contains an invalid event mark")

        state = self.initial_state(component_index)
        previous_time = 0.0
        quadrature_times: list[Tensor] = []
        quadrature_weights: list[Tensor] = []
        quadrature_intensities: list[Tensor] = []
        event_intensities: list[Tensor] = []

        for event_time, mark in zip(sequence.times, sequence.marks):
            gap = float(event_time) - previous_time
            q_times, q_weights, q_intensities = self._interval_quadrature(
                state,
                component_index=component_index,
                start_time=previous_time,
                interval_length=gap,
            )
            quadrature_times.append(q_times)
            quadrature_weights.append(q_weights)
            quadrature_intensities.append(q_intensities)

            cell_left, hidden_left = self._left_limit(state, gap)
            event_intensities.append(
                self.intensities_from_hidden(hidden_left, component_index)
            )
            embedding = self.event_embedding(
                torch.as_tensor(
                    int(mark),
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            cell, cell_bar, decay, output_gate = self.recurrent(
                embedding,
                hidden_left,
                cell_left,
                state.cell_bar,
            )
            state = NHPState(cell, cell_bar, decay, output_gate)
            previous_time = float(event_time)

        q_times, q_weights, q_intensities = self._interval_quadrature(
            state,
            component_index=component_index,
            start_time=previous_time,
            interval_length=self.horizon - previous_time,
        )
        quadrature_times.append(q_times)
        quadrature_weights.append(q_weights)
        quadrature_intensities.append(q_intensities)

        if event_intensities:
            event_tensor = torch.stack(event_intensities)
        else:
            event_tensor = self.mixture_logits.new_empty((0, self.n_marks))
        return NHPComponentTrace(
            event_times=torch.as_tensor(
                sequence.times,
                dtype=self.dtype,
                device=self.device,
            ),
            event_marks=torch.as_tensor(
                sequence.marks,
                dtype=torch.int64,
                device=self.device,
            ),
            event_base_intensities=event_tensor,
            quadrature_times=torch.cat(quadrature_times),
            quadrature_weights=torch.cat(quadrature_weights),
            quadrature_base_intensities=torch.cat(quadrature_intensities),
        )

    def component_scores(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        return self.component_scores_for_indices(
            sequences,
            range(self.n_components),
        )

    def component_scores_for_indices(
        self,
        sequences: Iterable[MarkedSequence],
        component_indices: Iterable[int],
    ) -> Tensor:
        """Score selected components while preprocessing the batch only once."""

        sequence_list = tuple(sequences)
        indices = tuple(int(index) for index in component_indices)
        if not indices:
            raise ValueError("component_indices must be non-empty")
        if any(
            index < 0 or index >= self.n_components for index in indices
        ):
            raise IndexError("component index out of range")
        times, marks, mask = self._prepare_sequence_batch(sequence_list)
        component_rows = [
            self._batched_component_score(times, marks, mask, component)
            for component in indices
        ]
        return torch.stack(component_rows, dim=1)

    def _prepare_sequence_batch(
        self,
        sequence_list: tuple[MarkedSequence, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not sequence_list:
            raise ValueError("sequences must be non-empty")
        if any(
            not math.isclose(sequence.horizon, self.horizon, abs_tol=1e-10)
            for sequence in sequence_list
        ):
            raise ValueError("sequence horizon does not match model horizon")
        if any(
            np.any(
                (sequence.marks < 0)
                | (sequence.marks >= self.n_marks)
            )
            for sequence in sequence_list
        ):
            raise ValueError("sequence contains an invalid event mark")

        batch_size = len(sequence_list)
        maximum_events = max(sequence.count for sequence in sequence_list)
        times = torch.full(
            (batch_size, maximum_events),
            self.horizon,
            dtype=self.dtype,
            device=self.device,
        )
        marks = torch.zeros(
            (batch_size, maximum_events),
            dtype=torch.int64,
            device=self.device,
        )
        mask = torch.zeros(
            (batch_size, maximum_events),
            dtype=torch.bool,
            device=self.device,
        )
        for row, sequence in enumerate(sequence_list):
            count = sequence.count
            if count:
                times[row, :count] = torch.as_tensor(
                    sequence.times,
                    dtype=self.dtype,
                    device=self.device,
                )
                marks[row, :count] = torch.as_tensor(
                    sequence.marks,
                    dtype=torch.int64,
                    device=self.device,
                )
                mask[row, :count] = True
        return times, marks, mask

    def _batched_component_score(
        self,
        times: Tensor,
        marks: Tensor,
        mask: Tensor,
        component_index: int,
    ) -> Tensor:
        """Vectorized exact-interval score for one mixture component."""

        batch_size, maximum_events = times.shape
        initial = self.initial_state(component_index)
        state = NHPState(
            cell=initial.cell[None, :].expand(batch_size, -1),
            cell_bar=initial.cell_bar[None, :].expand(batch_size, -1),
            decay=initial.decay[None, :].expand(batch_size, -1),
            output_gate=initial.output_gate[None, :].expand(batch_size, -1),
        )
        previous_time = times.new_zeros(batch_size)
        event_term = times.new_zeros(batch_size)
        compensator = times.new_zeros(batch_size)

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
                current.cell[:, None, :],
                current.cell_bar[:, None, :],
                current.decay[:, None, :],
                current.output_gate[:, None, :],
                elapsed[:, :, None],
            )
            intensity = self.intensities_from_hidden(
                hidden,
                component_index,
            )
            return (weights[:, :, None] * intensity).sum(dim=(1, 2))

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
                gap[:, None],
            )
            intensity = self.intensities_from_hidden(
                hidden_left,
                component_index,
            )
            selected = intensity.gather(
                1,
                marks[:, event_index, None],
            ).squeeze(1)
            event_term = event_term + torch.where(
                active,
                torch.log(selected + 1e-12),
                torch.zeros_like(selected),
            )
            embedding = self.event_embedding(marks[:, event_index])
            cell, cell_bar, decay, output_gate = self.recurrent(
                embedding,
                hidden_left,
                cell_left,
                state.cell_bar,
            )
            active_column = active[:, None]
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
            self.horizon - previous_time,
            state,
        )
        return event_term - compensator

    def log_likelihoods(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        component = self.component_scores(sequences)
        return torch.logsumexp(
            component + self.mixture_log_weights()[None, :],
            dim=1,
        )

    def negative_log_likelihood(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        return -self.log_likelihoods(sequences).sum()

    def responsibilities(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        component = self.component_scores(sequences)
        return torch.softmax(
            component + self.mixture_log_weights()[None, :],
            dim=1,
        )
