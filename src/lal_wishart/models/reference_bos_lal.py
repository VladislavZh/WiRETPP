"""Fixed-K LaL controls for THP and COTIC through a latent BOS code.

The original LaL architecture shares its recurrent network and decoder across
clusters and makes only the recurrent initial state cluster-specific. THP and
COTIC have no analogous persistent initial state, so the cluster code is the
beginning-of-stream embedding. All encoder and intensity-head weights remain
shared. Only the two initialization splits used by the final experiment are
implemented; there is no random-walk training logic in this repository.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model.torch_model.torch_thp import THP

from lal_wishart.models.reference_history_mixtures import (
    _load_cotic_classes,
    ReferenceHistoryMixture,
)


class ReferenceTHPBOSLaL(ReferenceHistoryMixture):
    """EasyTPP THP with cluster-specific BOS and shared network weights."""

    architecture_name = "easytpp_thp_0.2.1_bos_lal"
    embedding_scope = "bos"

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        horizon: float,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        quadrature_order: int = 8,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(
            n_components,
            n_marks,
            horizon=horizon,
            quadrature_order=quadrature_order,
        )
        if hidden_size % num_heads:
            raise ValueError("THP hidden size must be divisible by heads")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            config = ModelConfig(
                model_id="THP",
                hidden_size=hidden_size,
                time_emb_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_rate=dropout,
                use_ln=use_layer_norm,
                loss_integral_num_sample_per_step=quadrature_order,
                use_mc_samples=False,
                num_event_types=n_marks,
                num_event_types_pad=n_marks + 2,
                event_pad_index=n_marks + 1,
                gpu=-1,
            )
            self.backbone = THP(config)
        initial_bos = self.backbone.layer_type_emb.weight[
            self._bos_type()
        ].detach()
        self.bos_embeddings = nn.Parameter(
            initial_bos[None].repeat(n_components, 1)
        )

    def _bos_type(self) -> int:
        return self.n_marks

    def _pad_type(self) -> int:
        return self.n_marks + 1

    def _encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        return torch.where(
            valid,
            marks,
            marks.new_full(marks.shape, self._pad_type()),
        )

    def encode_histories(
        self,
        times: Tensor,
        encoded_types: Tensor,
        valid: Tensor,
    ) -> Tensor:
        length = times.shape[1]
        causal = torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=self.device,
            ),
            diagonal=1,
        )[None]
        key_padding = (~valid)[:, None, :]
        attention_mask = causal | key_padding
        temporal = self.backbone.layer_temporal_encoding(times)
        shared_types = self.backbone.layer_type_emb(encoded_types)
        rows = []
        for component in range(self.n_components):
            output = shared_types.clone()
            output[:, 0, :] = self.bos_embeddings[component]
            for layer in self.backbone.stack_layers:
                output = output + temporal
                output = layer(output, mask=attention_mask)
            rows.append(output)
        return torch.stack(rows)

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        del component_index
        logits = (
            self.backbone.layer_intensity_hidden(states)
            + self.backbone.factor_intensity_base
            + elapsed[..., None] * self.backbone.factor_intensity_decay
        )
        return self.backbone.softplus(logits) + 1e-10


class ReferenceCOTICBOSLaL(ReferenceHistoryMixture):
    """COTIC with cluster-specific BOS and a shared CNN/intensity head."""

    architecture_name = "vladislavzh_cotic_362b8ab_bos_lal"
    reference_commit = "362b8ab1f3cbb9e9dced2518e9daacf68235e77a"
    embedding_scope = "bos"

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        horizon: float,
        input_channels: int = 32,
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_factor: float = 1.29,
        quadrature_order: int = 8,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(
            n_components,
            n_marks,
            horizon=horizon,
            quadrature_order=quadrature_order,
        )
        COTIC, IntensityHeadLinear = _load_cotic_classes()
        self.input_channels = int(input_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.dilation_factor = float(dilation_factor)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = COTIC(
                in_channels=input_channels,
                kernel_size=kernel_size,
                nb_filters=hidden_size,
                nb_layers=num_layers,
                num_types=n_marks + 1,
                dropout=dropout,
                dilation_factor=dilation_factor,
            )
            self.intensity_head = IntensityHeadLinear(
                kernel_size=1,
                nb_filters=hidden_size,
                num_types=n_marks,
            )
        initial_bos = self.encoder.event_emb.weight[
            self._bos_type()
        ].detach()
        self.bos_embeddings = nn.Parameter(
            initial_bos[None].repeat(n_components, 1)
        )

    def _bos_type(self) -> int:
        return self.n_marks + 1

    def _pad_type(self) -> int:
        return 0

    def _encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        shifted = marks + 1
        shifted[:, 0] = self._bos_type()
        return torch.where(valid, shifted, torch.zeros_like(shifted))

    def encode_histories(
        self,
        times: Tensor,
        encoded_types: Tensor,
        valid: Tensor,
    ) -> Tensor:
        del valid
        non_pad_mask = encoded_types.ne(0)
        shared_types = self.encoder.event_emb(encoded_types)
        rows = []
        for component in range(self.n_components):
            output = shared_types.clone()
            output[:, 0, :] = self.bos_embeddings[component]
            for dropout, convolution in zip(
                self.encoder.dropouts,
                self.encoder.continuous_convolutions,
            ):
                output = dropout(
                    F.leaky_relu(
                        convolution(times, output, non_pad_mask),
                        0.1,
                    )
                )
            rows.append(output)
        return torch.stack(rows)

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        del component_index
        convolution = self.intensity_head.convolution
        linear_part = convolution.kernel_network_weight(states)
        bias_part = states @ convolution.kernel_network_bias
        hidden = self.intensity_head.activation(
            elapsed[..., None] * linear_part + bias_part
        )
        logits = self.intensity_head.layer(hidden)
        scale = self.intensity_head.softplus_params.exp().reshape(self.n_marks)
        return self.intensity_head.softplus(logits) * scale + 1e-10


ReferenceBOSLaL = ReferenceTHPBOSLaL | ReferenceCOTICBOSLaL


def _new_bos_lal_like(
    fitted: ReferenceBOSLaL,
    n_components: int,
    *,
    initialization_seed: int,
) -> ReferenceBOSLaL:
    if isinstance(fitted, ReferenceTHPBOSLaL):
        target: ReferenceBOSLaL = ReferenceTHPBOSLaL(
            n_components,
            fitted.n_marks,
            horizon=fitted.horizon,
            hidden_size=fitted.hidden_size,
            num_layers=fitted.num_layers,
            num_heads=fitted.num_heads,
            dropout=fitted.dropout,
            use_layer_norm=fitted.use_layer_norm,
            quadrature_order=fitted.quadrature_order,
            initialization_seed=initialization_seed,
        )
        target.backbone.load_state_dict(fitted.backbone.state_dict())
    else:
        target = ReferenceCOTICBOSLaL(
            n_components,
            fitted.n_marks,
            horizon=fitted.horizon,
            input_channels=fitted.input_channels,
            hidden_size=fitted.hidden_size,
            num_layers=fitted.num_layers,
            kernel_size=fitted.kernel_size,
            dropout=fitted.dropout,
            dilation_factor=fitted.dilation_factor,
            quadrature_order=fitted.quadrature_order,
            initialization_seed=initialization_seed,
        )
        target.encoder.load_state_dict(fitted.encoder.state_dict())
        target.intensity_head.load_state_dict(
            fitted.intensity_head.state_dict()
        )
    target.to(fitted.device)
    with torch.no_grad():
        target.mixture_logits.zero_()
    return target


def split_reference_bos_lal_component(
    fitted: ReferenceBOSLaL,
    component_index: int,
    *,
    initialization_seed: int,
    beta: float,
) -> ReferenceBOSLaL:
    """Apply LaL's multiplicative split to one cluster BOS code."""

    if not 0 <= component_index < fitted.n_components:
        raise IndexError("component index out of range")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    target = _new_bos_lal_like(
        fitted,
        fitted.n_components + 1,
        initialization_seed=initialization_seed,
    )
    embeddings = [
        fitted.bos_embeddings[index].detach().clone()
        for index in range(fitted.n_components)
    ]
    source = embeddings[component_index]
    embeddings[component_index] = 2.0 * beta * source
    embeddings.append(2.0 * (1.0 - beta) * source)
    with torch.no_grad():
        target.bos_embeddings.copy_(torch.stack(embeddings))
    return target
