"""THP encoder from Zuo et al. (ICML 2020) and its equation-6 intensity bank.

Architecture follows the authors' post-norm encoder, without their optional RNN
or auxiliary prediction tasks. BOS extends the likelihood to the full window.
Reference: github.com/SimiaoZuo/Transformer-Hawkes-Process/transformer
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from active_wishart_tpp.backbones.base import StateIntensityBank
from active_wishart_tpp.backbones.integration import IntegrationRule
from active_wishart_tpp.backbones.utils import repeat_with_simplex_noise


class PaperAttention(nn.Module):
    """Apply causal multi-head attention, output projection, and post-norm residual."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.key_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
        self.norm = nn.LayerNorm(width, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, forbidden: Tensor) -> Tensor:
        batch, length, width = inputs.shape
        projected = [
            layer(inputs)
            .reshape(batch, length, self.heads, self.key_width)
            .transpose(1, 2)
            for layer in (self.query, self.key, self.value)
        ]
        query, key, value = projected
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=(~forbidden)[:, None],
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        return self.norm(inputs + self.dropout(self.output(merged)))


class PaperEncoderLayer(nn.Module):
    """Apply one independent THP attention and GELU feed-forward block."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = PaperAttention(width, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width, eps=1e-06)

    def forward(self, inputs: Tensor, valid: Tensor, forbidden: Tensor) -> Tensor:
        mask = valid[..., None]
        states = self.attention(inputs, forbidden) * mask
        return self.norm(states + self.feed_forward(states)) * mask


class PaperTHPEncoder(nn.Module):
    """Encode marked histories with the authors' repeated temporal embeddings."""

    def __init__(
        self, marks: int, width: int, layers: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        if width < 2 or heads < 1 or width % heads or (layers < 1):
            raise ValueError(
                "THP needs positive layers/heads and width divisible by heads"
            )
        self.event_emb = nn.Embedding(marks + 2, width, padding_idx=marks + 1)
        exponent = 2 * (torch.arange(width) // 2) / width
        self.register_buffer("position_vec", 10000.0**exponent)
        self.layers = nn.ModuleList(
            (PaperEncoderLayer(width, heads, dropout) for _ in range(layers))
        )

    def forward(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        phase = times.masked_fill(~valid, 0)[..., None] / self.position_vec
        temporal = torch.empty_like(phase)
        temporal[..., 0::2] = phase[..., 0::2].sin()
        temporal[..., 1::2] = phase[..., 1::2].cos()
        temporal = temporal * valid[..., None]
        length = times.shape[1]
        future = torch.ones(length, length, device=times.device, dtype=torch.bool).triu(
            1
        )
        forbidden = future[None] | (~valid)[:, None, :]
        safe_types = types.masked_fill(~valid, self.event_emb.padding_idx)
        states = self.event_emb(safe_types)
        for layer in self.layers:
            states = layer(states + temporal, valid, forbidden)
        return states


class StableScaledSoftplus(nn.Module):
    """Map logits to positive rates with a learned positive inverse softness."""

    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.log_beta = nn.Parameter(torch.zeros(outputs))

    def forward(self, logits: Tensor) -> Tensor:
        beta = self.log_beta.exp()
        return F.softplus(beta * logits, threshold=20.0) / beta


class PaperTHPIntensityBank(StateIntensityBank):
    """Expose K copies of the paper's marked intensity with one shared encoder."""

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
        if not layer_norm:
            raise ValueError(
                "paper THP requires LayerNorm; use the legacy variant explicitly"
            )
        super().__init__(n_components, n_marks, integration_rule)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = PaperTHPEncoder(n_marks, hidden_size, layers, heads, dropout)
            self.intensity_linear = nn.Linear(hidden_size, n_components * n_marks)
        self.intensity_decay = nn.Parameter(torch.full((n_components * n_marks,), -0.1))
        self.intensity_link = StableScaledSoftplus(n_components * n_marks)

    def encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        types = marks.clone()
        types[:, 0] = self.n_marks
        return types.masked_fill(~valid, self.n_marks + 1)

    def encode_histories(self, times: Tensor, types: Tensor, valid: Tensor) -> Tensor:
        return self.encoder(times, types, valid)

    def rates_from_states(self, states: Tensor, elapsed: Tensor) -> Tensor:
        """Evaluate equation 6 given its already normalized temporal argument."""
        logits = (
            self.intensity_linear(states) + elapsed[..., None] * self.intensity_decay
        )
        return (
            self.intensity_link(logits)
            .reshape(*elapsed.shape, self.n_components, self.n_marks)
            .clamp_min(1e-10)
        )

    def rates_at_times(self, states, elapsed, times, paths, sequences):
        """Normalize elapsed time by the last event time, with unit-scale BOS."""
        previous = times - elapsed
        scale = torch.where(previous > 0, previous, torch.ones_like(previous))
        return self.rates_from_states(states, elapsed / scale)

    def expand_components(self, count: int, noise: float, seed: int) -> None:
        """Copy the shared head with simplex perturbations, preserving its encoder."""
        if self.n_components != 1:
            raise ValueError("only a one-component THP head can be expanded")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        linear = nn.Linear(
            self.intensity_linear.in_features,
            count * self.n_marks,
            device=self.device,
            dtype=self.dtype,
        )
        link = StableScaledSoftplus(count * self.n_marks).to(self.device, self.dtype)
        with torch.no_grad():
            linear.weight.copy_(
                repeat_with_simplex_noise(
                    self.intensity_linear.weight, count, noise, generator
                )
            )
            linear.bias.copy_(
                repeat_with_simplex_noise(
                    self.intensity_linear.bias, count, noise, generator
                )
            )
            decay = repeat_with_simplex_noise(
                self.intensity_decay, count, noise, generator
            )
            link.log_beta.copy_(self.intensity_link.log_beta.repeat(count))
        self.intensity_linear = linear
        self.intensity_decay = nn.Parameter(decay)
        self.intensity_link = link
        self.mixture_logits = nn.Parameter(torch.zeros(count, device=self.device))
        self.n_components = count

    def component_head_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.intensity_linear.weight,
            self.intensity_linear.bias,
            self.intensity_decay,
            self.intensity_link.log_beta,
        )
