"""Shared COTIC history encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wishart_tpp.cotic.layers import ContinuousConv1d


class CoticEncoder(nn.Module):
    """Map padded marked histories to one state after every event."""

    def __init__(
        self,
        n_marks: int,
        input_channels: int,
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        dilation_factor: float,
    ) -> None:
        super().__init__()
        # Paper COTIC uses exactly one padding id plus the C event ids.  There
        # is no learned beginning-of-stream token in the upstream data path.
        self.embedding = nn.Embedding(n_marks + 1, input_channels, padding_idx=0)
        input_sizes = [input_channels] + [hidden_size] * (layers - 1)
        dilations = [int(dilation_factor**index) for index in range(layers)]
        self.layers = nn.ModuleList(
            [
                ContinuousConv1d(kernel_size, width, hidden_size, dilation)
                for width, dilation in zip(input_sizes, dilations)
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])

    def forward(self, times: Tensor, event_types: Tensor, valid: Tensor) -> Tensor:
        states = self.embedding(event_types)
        for convolution, dropout in zip(self.layers, self.dropouts):
            states = dropout(
                torch.nn.functional.leaky_relu(
                    convolution(times, states, valid), negative_slope=0.1
                )
            )
        return states
