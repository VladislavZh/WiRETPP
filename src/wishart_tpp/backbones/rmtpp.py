"""EasyTPP RMTPP adapter with a shared RNN and K*C intensity head."""

from __future__ import annotations

import math

import torch
from easy_tpp.model.torch_model.torch_rmtpp import RMTPP
from torch import Tensor, nn

from wishart_tpp.backbones.base import StateIntensityBank
from wishart_tpp.backbones.easytpp_config import easytpp_config
from wishart_tpp.backbones.integration import IntegrationRule
from wishart_tpp.backbones.utils import repeat_with_noise


class RMTPPIntensityBank(StateIntensityBank):
    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        hidden_size: int,
        integration_rule: IntegrationRule,
        initialization_seed: int,
    ) -> None:
        super().__init__(n_components, n_marks, integration_rule)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = RMTPP(
                easytpp_config(
                    "RMTPP",
                    n_marks,
                    hidden_size,
                    1,
                    1,
                    0.0,
                    False,
                    integration_rule.samples,
                )
            )
        self.intensity_linear = self.encoder.hidden_to_intensity_logits
        self.intensity_base = self.encoder.b_t
        self.intensity_decay = self.encoder.w_t
        del self.encoder.hidden_to_intensity_logits
        del self.encoder.b_t
        del self.encoder.w_t

    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        marks = marks.clone()
        marks[:, 0] = self.n_marks
        return torch.where(valid, marks, marks.new_full(marks.shape, self.n_marks + 1))

    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        del valid
        mark_embedding = self.encoder.layer_type_emb(types)
        time_embedding = self.encoder.layer_temporal_emb(times[..., None])
        states, _ = self.encoder.layer_rnn(mark_embedding + time_embedding)
        return states

    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        logits = (
            self.intensity_linear(states)
            + self.intensity_base
            + elapsed[..., None] * self.intensity_decay
        ).clamp(max=math.log(1e5))
        return (
            logits.exp()
            .reshape(*elapsed.shape, self.n_components, self.n_marks)
            .clamp_min(1e-10)
        )

    def expand_components(self, count: int, noise: float, seed: int) -> None:
        if self.n_components != 1:
            raise ValueError("only a one-component RMTPP head can be expanded")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output_size = count * self.n_marks
        linear = nn.Linear(
            self.intensity_linear.in_features,
            output_size,
            device=self.device,
            dtype=self.dtype,
        )
        # Expand only the mark-intensity parameters; keep the fitted RNN shared.
        with torch.no_grad():
            linear.weight.copy_(
                repeat_with_noise(
                    self.intensity_linear.weight, count, self.n_marks, noise, generator
                )
            )
            linear.bias.copy_(
                repeat_with_noise(
                    self.intensity_linear.bias, count, self.n_marks, noise, generator
                )
            )
            base = repeat_with_noise(
                self.intensity_base.reshape(-1), count, self.n_marks, noise, generator
            )[None]
            decay = repeat_with_noise(
                self.intensity_decay.reshape(-1), count, self.n_marks, noise, generator
            )[None]
        self.intensity_linear = linear
        self.intensity_base = nn.Parameter(base)
        self.intensity_decay = nn.Parameter(decay)
        self.mixture_logits = nn.Parameter(torch.zeros(count, device=self.device))
        self.n_components = count
