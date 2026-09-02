"""Backbone boundary required by the Active Block Wishart wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import Tensor, nn

from wishart_tpp.backbones.integration import IntegrationRule
from wishart_tpp.data import EventSequence
from wishart_tpp.model.trace import TPPTrace


@dataclass(frozen=True)
class _EventRows:
    times: Tensor
    marks: Tensor
    paths: Tensor
    rates: Tensor


@dataclass(frozen=True)
class _IntegralRows:
    times: Tensor
    weights: Tensor
    paths: Tensor
    rates: Tensor


class IntensityBank(nn.Module, ABC):
    """A history model that exposes K candidate C-mark intensities."""

    n_components: int
    n_marks: int
    mixture_logits: nn.Parameter

    @property
    def device(self) -> torch.device:
        return self.mixture_logits.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mixture_logits.dtype

    def mixture_log_weights(self) -> Tensor:
        return torch.log_softmax(self.mixture_logits, dim=0)

    @abstractmethod
    def expand_components(self, count: int, noise: float, seed: int) -> None:
        """Turn a fitted K=1 head into K lightly perturbed output blocks."""

    @abstractmethod
    def forward(self, sequences: tuple[EventSequence, ...]) -> TPPTrace:
        """Return event and quadrature rates with shape rows x K x C."""


class StateIntensityBank(IntensityBank, ABC):
    """Common likelihood-trace builder for state-based backbones."""

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        integration_rule: IntegrationRule,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_marks = n_marks
        self.integration_rule = integration_rule
        self.mixture_logits = nn.Parameter(torch.zeros(n_components))

    @abstractmethod
    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        pass

    @abstractmethod
    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        pass

    @abstractmethod
    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        pass

    def _histories(
        self, sequences: tuple[EventSequence, ...]
    ) -> tuple[Tensor, Tensor, Tensor]:
        length = max(sequence.count for sequence in sequences) + 1
        shape = (len(sequences), length)
        times = torch.zeros(shape, device=self.device, dtype=self.dtype)
        marks = torch.zeros(shape, device=self.device, dtype=torch.long)
        valid = torch.zeros(shape, device=self.device, dtype=torch.bool)
        valid[:, 0] = True
        for path, sequence in enumerate(sequences):
            stop = sequence.count + 1
            times[path, 1:stop] = torch.as_tensor(
                sequence.times, device=self.device, dtype=self.dtype
            )
            marks[path, 1:stop] = torch.as_tensor(sequence.marks, device=self.device)
            valid[path, 1:stop] = True
        return times, self.encoded_types(marks, valid), valid

    def _event_rows(
        self, sequences: tuple[EventSequence, ...], states: Tensor
    ) -> _EventRows:
        times, marks, paths, histories, elapsed = [], [], [], [], []
        for path, sequence in enumerate(sequences):
            previous = 0.0
            for history, (time, mark) in enumerate(zip(sequence.times, sequence.marks)):
                times.append(float(time))
                marks.append(int(mark))
                paths.append(path)
                histories.append(history)
                elapsed.append(float(time) - previous)
                previous = float(time)

        path_index = torch.tensor(paths, device=self.device, dtype=torch.long)
        history_index = torch.tensor(histories, device=self.device, dtype=torch.long)
        elapsed_tensor = torch.tensor(elapsed, device=self.device, dtype=self.dtype)
        rates = self.rates_from_states(
            states[path_index, history_index], elapsed_tensor
        )
        return _EventRows(
            torch.tensor(times, device=self.device, dtype=self.dtype),
            torch.tensor(marks, device=self.device, dtype=torch.long),
            path_index,
            rates,
        )

    def _integral_rows(
        self, sequences: tuple[EventSequence, ...], states: Tensor
    ) -> _IntegralRows:
        starts, stops, paths, histories = [], [], [], []
        for path, sequence in enumerate(sequences):
            boundaries = np.concatenate(([0.0], sequence.times, [sequence.horizon]))
            intervals = pairwise(boundaries)
            for history, (start, stop) in enumerate(intervals):
                if stop > start:
                    starts.append(float(start))
                    stops.append(float(stop))
                    paths.append(path)
                    histories.append(history)

        start = torch.tensor(starts, device=self.device, dtype=self.dtype)
        stop = torch.tensor(stops, device=self.device, dtype=self.dtype)
        path_index = torch.tensor(paths, device=self.device, dtype=torch.long)
        history_index = torch.tensor(histories, device=self.device, dtype=torch.long)
        lengths = stop - start
        points = self.integration_rule.points(lengths, self.training)
        elapsed = points.elapsed
        sample_count = self.integration_rule.samples
        expanded_states = states[path_index, history_index][:, None].expand(
            -1, sample_count, -1
        )
        return _IntegralRows(
            times=(start[:, None] + elapsed).reshape(-1),
            weights=points.weights.reshape(-1),
            paths=path_index[:, None].expand(-1, sample_count).reshape(-1),
            rates=self.rates_from_states(expanded_states, elapsed).reshape(
                -1, self.n_components, self.n_marks
            ),
        )

    def forward(self, sequences: tuple[EventSequence, ...]) -> TPPTrace:
        if not sequences:
            raise ValueError("a trace requires at least one sequence")

        # Encode each history once, then query events and integration points.
        history_times, history_types, valid = self._histories(sequences)
        states = self.encode_histories(history_times, history_types, valid)
        events = self._event_rows(sequences, states)
        integral = self._integral_rows(sequences, states)

        # The wrapper consumes only this backbone-independent likelihood trace.
        return TPPTrace(
            n_paths=len(sequences),
            n_components=self.n_components,
            n_marks=self.n_marks,
            event_times=events.times,
            event_marks=events.marks,
            event_paths=events.paths,
            event_base_rates=events.rates,
            integral_times=integral.times,
            integral_weights=integral.weights,
            integral_paths=integral.paths,
            integral_base_rates=integral.rates,
        )
