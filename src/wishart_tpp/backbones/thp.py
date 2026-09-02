"""EasyTPP THP adapter with one shared encoder and a K*C output bank."""

from __future__ import annotations

import torch
from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_thp import THP
from torch import Tensor, nn

from wishart_tpp.backbones.base import StateIntensityBank
from wishart_tpp.backbones.easytpp_config import easytpp_config
from wishart_tpp.backbones.integration import IntegrationRule
from wishart_tpp.backbones.utils import repeat_with_noise


class THPIntensityBank(StateIntensityBank):
    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        hidden_size: int,
        layers: int,
        heads: int,
        dropout: float,
        layer_norm: bool,
        integration_rule: IntegrationRule,
        initialization_seed: int,
    ) -> None:
        super().__init__(n_components, n_marks, integration_rule)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = THP(
                easytpp_config(
                    "THP",
                    n_marks,
                    hidden_size,
                    layers,
                    heads,
                    dropout,
                    layer_norm,
                    integration_rule.samples,
                )
            )
        # Detach EasyTPP's intensity layer so it can expose a K by C bank.
        self.intensity_linear = self.encoder.layer_intensity_hidden
        self.intensity_base = self.encoder.factor_intensity_base
        self.intensity_decay = self.encoder.factor_intensity_decay
        self.intensity_link = self.encoder.softplus
        del self.encoder.layer_intensity_hidden
        del self.encoder.factor_intensity_base
        del self.encoder.factor_intensity_decay
        del self.encoder.softplus

    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        marks = marks.clone()
        marks[:, 0] = self.n_marks
        return torch.where(valid, marks, marks.new_full(marks.shape, self.n_marks + 1))

    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        length = times.shape[1]
        causal = torch.triu(
            torch.ones(length, length, device=self.device, dtype=torch.bool), diagonal=1
        )[None]
        return self.encoder(times, types, causal | (~valid)[:, None, :])

    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        logits = (
            self.intensity_linear(states)
            + self.intensity_base
            + elapsed[..., None] * self.intensity_decay
        )
        return (
            self.intensity_link(logits)
            .reshape(*elapsed.shape, self.n_components, self.n_marks)
            .clamp_min(1e-10)
        )

    def expand_components(self, count: int, noise: float, seed: int) -> None:
        if self.n_components != 1:
            raise ValueError("only a one-component THP head can be expanded")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output_size = count * self.n_marks
        linear = nn.Linear(
            self.intensity_linear.in_features,
            output_size,
            device=self.device,
            dtype=self.dtype,
        )
        link = ScaledSoftplus(output_size).to(device=self.device, dtype=self.dtype)
        # Expand only the intensity parameters; keep the fitted Transformer shared.
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
            link.log_beta.copy_(self.intensity_link.log_beta.detach().repeat(count))
        self.intensity_linear = linear
        self.intensity_base = nn.Parameter(base)
        self.intensity_decay = nn.Parameter(decay)
        self.intensity_link = link
        self.mixture_logits = nn.Parameter(torch.zeros(count, device=self.device))
        self.n_components = count
