"""EasyTPP NHP adapter with a shared CT-LSTM and K*C intensity head."""

from __future__ import annotations

import torch
from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_nhp import NHP
from torch import Tensor, nn

from wishart_tpp.backbones.base import StateIntensityBank
from wishart_tpp.backbones.easytpp_config import easytpp_config
from wishart_tpp.backbones.integration import IntegrationRule
from wishart_tpp.backbones.utils import repeat_with_noise


class NHPIntensityBank(StateIntensityBank):
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
            self.encoder = NHP(
                easytpp_config(
                    "NHP",
                    n_marks,
                    hidden_size,
                    1,
                    1,
                    0.0,
                    False,
                    integration_rule.samples,
                )
            )
        self.intensity_linear = self.encoder.layer_intensity[0]
        self.intensity_link = self.encoder.layer_intensity[1]
        del self.encoder.layer_intensity

    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        marks = marks.clone()
        marks[:, 0] = self.n_marks
        return torch.where(valid, marks, marks.new_full(marks.shape, self.n_marks + 1))

    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        deltas = torch.zeros_like(times)
        deltas[:, 1:] = times[:, 1:] - times[:, :-1]
        deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
        _, right_states = self.encoder((times, deltas, types, None, None))
        return right_states

    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        cell, cell_bar, decay, output_gate = states.chunk(4, dim=-1)
        _, hidden = self.encoder.rnn_cell.decay(
            cell, cell_bar, decay, output_gate, elapsed[..., None]
        )
        return (
            self.intensity_link(self.intensity_linear(hidden))
            .reshape(*elapsed.shape, self.n_components, self.n_marks)
            .clamp_min(1e-10)
        )

    def expand_components(self, count: int, noise: float, seed: int) -> None:
        if self.n_components != 1:
            raise ValueError("only a one-component NHP head can be expanded")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output_size = count * self.n_marks
        linear = nn.Linear(
            self.intensity_linear.in_features,
            output_size,
            bias=self.intensity_linear.bias is not None,
            device=self.device,
            dtype=self.dtype,
        )
        link = ScaledSoftplus(output_size).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            linear.weight.copy_(
                repeat_with_noise(
                    self.intensity_linear.weight, count, self.n_marks, noise, generator
                )
            )
            if linear.bias is not None:
                linear.bias.copy_(
                    repeat_with_noise(
                        self.intensity_linear.bias,
                        count,
                        self.n_marks,
                        noise,
                        generator,
                    )
                )
            link.log_beta.copy_(self.intensity_link.log_beta.detach().repeat(count))
        self.intensity_linear = linear
        self.intensity_link = link
        self.mixture_logits = nn.Parameter(torch.zeros(count, device=self.device))
        self.n_components = count
