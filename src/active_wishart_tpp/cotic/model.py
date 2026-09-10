"""COTIC adapter for the generic K by C intensity-bank interface."""

from __future__ import annotations
import torch
from torch import Tensor, nn
from active_wishart_tpp.backbones.base import StateIntensityBank
from active_wishart_tpp.backbones.integration import IntegrationRule, MonteCarloRule
from active_wishart_tpp.backbones.utils import repeat_with_simplex_noise
from active_wishart_tpp.cotic.encoder import CoticEncoder
from active_wishart_tpp.cotic.head import CoticIntensityHead
from active_wishart_tpp.data import EventSequence
from active_wishart_tpp.model.trace import TPPTrace


class CoticIntensityBank(StateIntensityBank):
    """VladislavZh/COTIC encoder with a shared K*C output head."""

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        input_channels: int = 192,
        hidden_size: int = 512,
        layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_factor: float = 1.29,
        integration_rule: IntegrationRule | None = None,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(
            n_components,
            n_marks,
            MonteCarloRule(50, seed=0)
            if integration_rule is None
            else integration_rule,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = CoticEncoder(
                n_marks,
                input_channels,
                hidden_size,
                layers,
                kernel_size,
                dropout,
                dilation_factor,
            )
            self.intensity_head = CoticIntensityHead(
                hidden_size, n_components * n_marks, base_mark_count=n_marks
            )

    def expand_components(self, count: int, noise: float, seed: int) -> None:
        """Place copies of a fitted head at centered simplex perturbations."""
        if self.n_components != 1:
            raise ValueError("only a one-component bank can be expanded")
        old_output = self.intensity_head.output
        old_scale = self.intensity_head.log_scale
        generator = torch.Generator(device=self.device).manual_seed(seed)
        expanded_output = nn.Linear(
            old_output.in_features,
            count * self.n_marks,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            expanded_output.weight.copy_(
                repeat_with_simplex_noise(old_output.weight, count, noise, generator)
            )
            expanded_output.bias.copy_(
                repeat_with_simplex_noise(old_output.bias, count, noise, generator)
            )
        self.intensity_head.output = expanded_output
        self.intensity_head.log_scale = nn.Parameter(old_scale.detach().repeat(count))
        self.mixture_logits = nn.Parameter(
            torch.zeros(count, device=self.device, dtype=self.dtype)
        )
        self.n_components = count

    def component_head_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the component-indexed COTIC intensity parameters."""
        return (
            self.intensity_head.output.weight,
            self.intensity_head.output.bias,
            self.intensity_head.log_scale,
        )

    def _histories(
        self, sequences: tuple[EventSequence, ...]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the exact upstream COTIC event tensor, without a BOS row."""
        length = max((sequence.count for sequence in sequences))
        shape = (len(sequences), length)
        times = torch.zeros(shape, device=self.device, dtype=self.dtype)
        marks = torch.zeros(shape, device=self.device, dtype=torch.long)
        valid = torch.zeros(shape, device=self.device, dtype=torch.bool)
        for path, sequence in enumerate(sequences):
            stop = sequence.count
            times[path, :stop] = torch.as_tensor(
                sequence.times, device=self.device, dtype=self.dtype
            )
            marks[path, :stop] = torch.as_tensor(
                sequence.marks, device=self.device, dtype=torch.long
            )
            valid[path, :stop] = True
        return (times, self.encoded_types(marks, valid), valid)

    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        shifted = marks + 1
        return torch.where(valid, shifted, torch.zeros_like(shifted))

    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        return self.encoder(times, types, valid)

    def states_after_events(
        self, sequences: tuple[EventSequence, ...]
    ) -> tuple[Tensor, Tensor]:
        """Return the paper COTIC state attached to each observed event."""
        history_times, history_types, valid = self._histories(sequences)
        return (self.encode_histories(history_times, history_types, valid), valid)

    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        return (
            self.intensity_head(states, elapsed)
            .reshape(*elapsed.shape, self.n_components, self.n_marks)
            .clamp_min(1e-10)
        )

    def forward(self, sequences: tuple[EventSequence, ...]) -> TPPTrace:
        """Build a likelihood trace with the exact paper event alignment.

        The encoder sees only real events, as in the official repository.  The
        intensity of the first event is evaluated from an external zero state;
        every later event and interval uses the encoder state after the previous
        event.  This is equivalent to ``ContinuousConv1DSim(kernel_size=1)``
        without inserting a learned BOS event into the convolutional history.
        """
        if not sequences:
            raise ValueError("a trace requires at least one sequence")
        encoded, _ = self.states_after_events(sequences)
        zero = encoded.new_zeros(encoded.shape[0], 1, encoded.shape[2])
        states_before_events = torch.cat((zero, encoded), dim=1)
        events = self._event_rows(sequences, states_before_events)
        integral = self._integral_rows(sequences, states_before_events)
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
