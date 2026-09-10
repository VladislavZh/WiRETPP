"""Minimal continuous convolution used by the original COTIC encoder.

The organization follows VladislavZh/COTIC at commit 362b8ab, while naming,
validation and tensor construction are kept local and explicit.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContinuousConv1d(nn.Module):
    """Causal continuous convolution with a linear time kernel."""

    def __init__(
        self, kernel_size: int, input_channels: int, output_channels: int, dilation: int
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        preparation = (
            torch.eye(kernel_size).unsqueeze(1).repeat(2 * output_channels + 1, 1, 1)
        )
        self.register_buffer("preparation_kernel", preparation)
        self.time_weight = nn.Linear(input_channels, output_channels, bias=False)
        self.time_bias = nn.Parameter(
            torch.full((input_channels, output_channels), 1.0 / output_channels)
        )
        self.skip = nn.Linear(input_channels, output_channels)
        self.normalization = nn.LayerNorm(output_channels)

    def _history_windows(
        self, times: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = torch.cat((times.unsqueeze(-1), features), dim=-1)
        prepared = F.conv1d(
            values.transpose(1, 2),
            self.preparation_kernel,
            padding=self.padding,
            dilation=self.dilation,
            groups=2 * self.output_channels + 1,
        )
        if self.padding:
            prepared = prepared[:, :, : -self.padding]
        prepared = prepared.reshape(
            times.shape[0],
            2 * self.output_channels + 1,
            self.kernel_size,
            times.shape[1],
        )
        history_times = prepared[:, 0]
        history_features = prepared[:, 1:].permute(0, 2, 3, 1)
        return (times.unsqueeze(1) - history_times, history_features)

    def forward(self, times: Tensor, features: Tensor, valid: Tensor) -> Tensor:
        slopes = self.time_weight(features)
        intercepts = features @ self.time_bias
        elapsed, windows = self._history_windows(
            times, torch.cat((slopes, intercepts), dim=-1)
        )
        elapsed = elapsed / self.dilation - 1.0
        slope_windows = windows[..., : self.output_channels]
        intercept_windows = windows[..., self.output_channels :]
        output = (elapsed.unsqueeze(-1) * slope_windows + intercept_windows).sum(dim=1)
        output = output + self.skip(features)
        output[valid] = self.normalization(output[valid])
        return output
