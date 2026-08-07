"""Single-encoder TPP mixtures whose only K-way expansion is the output head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model.torch_model.torch_baselayer import (
    ScaledSoftplus as EasyTPPScaledSoftplus,
)
from easy_tpp.model.torch_model.torch_thp import THP

from lal_wishart.models.neural_hawkes import (
    NeuralHawkesMixture,
    NHPState,
    ScaledSoftplus,
)
from lal_wishart.models.reference_history_mixtures import (
    _load_cotic_classes,
    ReferenceHistoryMixture,
)


def _repeat_with_noise(
    values: Tensor,
    repeats: int,
    *,
    generator: torch.Generator,
    noise_scale: float,
) -> Tensor:
    repeated = values.repeat((repeats,) + (1,) * (values.ndim - 1))
    if repeats == 1 or noise_scale == 0.0:
        return repeated
    return repeated + noise_scale * torch.randn(
        repeated.shape,
        dtype=repeated.dtype,
        device="cpu",
        generator=generator,
    ).to(repeated.device)


class ReferenceNHPOutputMixture(ReferenceHistoryMixture):
    """One CT-LSTM and a single ``hidden -> K*C`` intensity head."""

    architecture_name = "easytpp_nhp_shared_encoder_kc_output"

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        horizon: float,
        hidden_size: int = 16,
        quadrature_order: int = 8,
        initial_total_rate: float = 0.7,
        initialization_seed: int = 0,
        output_noise_scale: float = 0.01,
    ) -> None:
        super().__init__(
            n_components,
            n_marks,
            horizon=horizon,
            quadrature_order=quadrature_order,
        )
        self.hidden_size = int(hidden_size)
        self.initial_total_rate = float(initial_total_rate)
        self.output_noise_scale = float(output_noise_scale)
        source = NeuralHawkesMixture(
            1,
            n_marks,
            horizon=horizon,
            hidden_size=hidden_size,
            quadrature_order=quadrature_order,
            initial_total_rate=initial_total_rate,
            initialization_seed=initialization_seed,
            dtype=torch.float32,
        )
        self.event_embedding = source.event_embedding
        self.recurrent = source.recurrent
        self.initial_cell = nn.Parameter(
            source.initial_cell[0].detach().clone()
        )
        self.initial_cell_bar = nn.Parameter(
            source.initial_cell_bar[0].detach().clone()
        )
        self.initial_raw_decay = nn.Parameter(
            source.initial_raw_decay[0].detach().clone()
        )
        self.initial_output_logits = nn.Parameter(
            source.initial_output_logits[0].detach().clone()
        )
        dimension = n_components * n_marks
        self.intensity_linear = nn.Linear(
            hidden_size,
            dimension,
            dtype=torch.float32,
        )
        self.intensity_link = ScaledSoftplus(
            dimension,
            dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(initialization_seed + 17)
        with torch.no_grad():
            self.intensity_linear.weight.copy_(
                _repeat_with_noise(
                    source.intensity_linear.weight.detach(),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )
            )
            self.intensity_linear.bias.copy_(
                _repeat_with_noise(
                    source.intensity_linear.bias.detach(),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )
            )
            self.intensity_link.raw_scale.copy_(
                source.intensity_link.raw_scale.detach().repeat(n_components)
            )

    def _bos_type(self) -> int:
        return self.n_marks

    def _pad_type(self) -> int:
        return self.n_marks

    def _encoded_types(self, marks: Tensor, valid: Tensor) -> Tensor:
        return torch.where(valid, marks, torch.zeros_like(marks))

    def encode_histories(
        self,
        times: Tensor,
        encoded_types: Tensor,
        valid: Tensor,
    ) -> Tensor:
        batch_size, history_length = times.shape
        state = NHPState(
            cell=self.initial_cell[None].expand(batch_size, -1),
            cell_bar=self.initial_cell_bar[None].expand(batch_size, -1),
            decay=(F.softplus(self.initial_raw_decay) + 1e-8)[None].expand(
                batch_size,
                -1,
            ),
            output_gate=torch.sigmoid(
                self.initial_output_logits
            )[None].expand(batch_size, -1),
        )
        rows = [
            torch.cat(
                (
                    state.cell,
                    state.cell_bar,
                    state.decay,
                    state.output_gate,
                ),
                dim=1,
            )
        ]
        previous_time = times.new_zeros(batch_size)
        for history_index in range(1, history_length):
            active = valid[:, history_index]
            event_time = times[:, history_index]
            gap = torch.where(
                active,
                event_time - previous_time,
                torch.zeros_like(previous_time),
            )
            cell_left, hidden_left = self.recurrent.decay(
                state.cell,
                state.cell_bar,
                state.decay,
                state.output_gate,
                gap[:, None],
            )
            embedding = self.event_embedding(
                encoded_types[:, history_index]
            )
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
            previous_time = torch.where(active, event_time, previous_time)
            rows.append(
                torch.cat(
                    (
                        state.cell,
                        state.cell_bar,
                        state.decay,
                        state.output_gate,
                    ),
                    dim=1,
                )
            )
        shared = torch.stack(rows, dim=1)
        return shared[None].expand(self.n_components, -1, -1, -1)

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        cell, cell_bar, decay, output_gate = states.chunk(4, dim=-1)
        _, hidden = self.recurrent.decay(
            cell,
            cell_bar,
            decay,
            output_gate,
            elapsed[..., None],
        )
        all_rates = self.intensity_link(
            self.intensity_linear(hidden)
        ) + 1e-10
        return all_rates.reshape(
            *all_rates.shape[:-1],
            self.n_components,
            self.n_marks,
        )[..., component_index, :]

    def prune_component(self, component_index: int) -> tuple[int, ...]:
        """Physically remove one mixture logit and its C-output block."""

        if self.n_components <= 1:
            raise ValueError("cannot prune the final mixture component")
        if not 0 <= component_index < self.n_components:
            raise IndexError("component index out of range")
        kept_components = tuple(
            index
            for index in range(self.n_components)
            if index != component_index
        )
        kept_outputs = torch.as_tensor(
            [
                component * self.n_marks + mark
                for component in kept_components
                for mark in range(self.n_marks)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        new_dimension = len(kept_components) * self.n_marks
        old_linear = self.intensity_linear
        new_linear = nn.Linear(
            old_linear.in_features,
            new_dimension,
            device=self.device,
            dtype=self.dtype,
        )
        new_link = ScaledSoftplus(
            new_dimension,
            dtype=self.dtype,
        ).to(self.device)
        with torch.no_grad():
            new_linear.weight.copy_(
                old_linear.weight.index_select(0, kept_outputs)
            )
            new_linear.bias.copy_(
                old_linear.bias.index_select(0, kept_outputs)
            )
            new_link.raw_scale.copy_(
                self.intensity_link.raw_scale.index_select(0, kept_outputs)
            )
            old_weights = torch.softmax(self.mixture_logits, dim=0)
            kept_index = torch.as_tensor(
                kept_components,
                dtype=torch.int64,
                device=self.device,
            )
            kept_weights = old_weights.index_select(0, kept_index)
            new_logits = torch.log(kept_weights / kept_weights.sum())
        self.intensity_linear = new_linear
        self.intensity_link = new_link
        self.mixture_logits = nn.Parameter(new_logits.detach().clone())
        self.n_components = len(kept_components)
        return kept_components


class ReferenceTHPOutputMixture(ReferenceHistoryMixture):
    """One EasyTPP THP encoder and one ``hidden -> K*C`` THP head."""

    architecture_name = "easytpp_thp_0.2.1_shared_encoder_kc_output"

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
        output_noise_scale: float = 0.01,
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
        self.output_noise_scale = float(output_noise_scale)
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
        source_linear = self.backbone.layer_intensity_hidden
        source_base = self.backbone.factor_intensity_base
        source_decay = self.backbone.factor_intensity_decay
        source_beta = self.backbone.softplus.log_beta
        del self.backbone.layer_intensity_hidden
        del self.backbone.factor_intensity_base
        del self.backbone.factor_intensity_decay
        del self.backbone.softplus
        dimension = n_components * n_marks
        self.layer_intensity_hidden = nn.Linear(hidden_size, dimension)
        self.factor_intensity_base = nn.Parameter(
            torch.empty(1, dimension)
        )
        self.factor_intensity_decay = nn.Parameter(
            torch.empty(1, dimension)
        )
        self.softplus = EasyTPPScaledSoftplus(dimension)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(initialization_seed + 19)
        with torch.no_grad():
            self.layer_intensity_hidden.weight.copy_(
                _repeat_with_noise(
                    source_linear.weight.detach(),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )
            )
            self.layer_intensity_hidden.bias.copy_(
                _repeat_with_noise(
                    source_linear.bias.detach(),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )
            )
            self.factor_intensity_base.copy_(
                _repeat_with_noise(
                    source_base.detach().reshape(-1),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )[None]
            )
            self.factor_intensity_decay.copy_(
                _repeat_with_noise(
                    source_decay.detach().reshape(-1),
                    n_components,
                    generator=generator,
                    noise_scale=output_noise_scale,
                )[None]
            )
            self.softplus.log_beta.copy_(source_beta.repeat(n_components))

    def expand_components(
        self,
        n_components: int,
        *,
        noise_scale: float = 0.01,
        initialization_seed: int = 0,
    ) -> None:
        """Clone a fitted K=1 THP output while preserving its encoder."""

        if self.n_components != 1:
            raise ValueError("component expansion requires a fitted K=1 head")
        if n_components <= 1:
            raise ValueError("expanded component count must exceed one")
        if noise_scale < 0.0:
            raise ValueError("output noise scale must be non-negative")
        old_linear = self.layer_intensity_hidden
        output_count = n_components * self.n_marks
        new_linear = nn.Linear(
            old_linear.in_features,
            output_count,
            device=self.device,
            dtype=self.dtype,
        )
        new_softplus = EasyTPPScaledSoftplus(output_count).to(
            device=self.device,
            dtype=self.dtype,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(initialization_seed)

        def expanded(values: Tensor) -> Tensor:
            copies = values.detach().repeat(n_components, *(
                1 for _ in range(values.ndim - 1)
            ))
            if noise_scale > 0.0:
                noise = torch.randn(
                    copies.shape,
                    dtype=copies.dtype,
                    device=copies.device,
                    generator=generator,
                )
                noise[: self.n_marks].zero_()
                copies = copies + noise_scale * noise
            return copies

        with torch.no_grad():
            new_linear.weight.copy_(expanded(old_linear.weight))
            new_linear.bias.copy_(expanded(old_linear.bias))
            new_base = expanded(
                self.factor_intensity_base.reshape(self.n_marks)
            )[None]
            new_decay = expanded(
                self.factor_intensity_decay.reshape(self.n_marks)
            )[None]
            new_softplus.log_beta.copy_(
                self.softplus.log_beta.detach().repeat(n_components)
            )
        self.layer_intensity_hidden = new_linear
        self.factor_intensity_base = nn.Parameter(new_base)
        self.factor_intensity_decay = nn.Parameter(new_decay)
        self.softplus = new_softplus
        self.mixture_logits = nn.Parameter(
            torch.zeros(
                n_components,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.n_components = int(n_components)

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
        attention_mask = causal | (~valid)[:, None, :]
        shared = self.backbone(times, encoded_types, attention_mask)
        return shared[None].expand(self.n_components, -1, -1, -1)

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        logits = (
            self.layer_intensity_hidden(states)
            + self.factor_intensity_base
            + elapsed[..., None] * self.factor_intensity_decay
        )
        all_rates = self.softplus(logits) + 1e-10
        return all_rates.reshape(
            *all_rates.shape[:-1],
            self.n_components,
            self.n_marks,
        )[..., component_index, :]

    def prune_component(self, component_index: int) -> tuple[int, ...]:
        """Physically remove one mixture logit and its C-output block."""

        if self.n_components <= 1:
            raise ValueError("cannot prune the final mixture component")
        if not 0 <= component_index < self.n_components:
            raise IndexError("component index out of range")
        kept_components = tuple(
            index
            for index in range(self.n_components)
            if index != component_index
        )
        kept_outputs = torch.as_tensor(
            [
                component * self.n_marks + mark
                for component in kept_components
                for mark in range(self.n_marks)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        new_dimension = len(kept_components) * self.n_marks
        old_linear = self.layer_intensity_hidden
        new_linear = nn.Linear(
            old_linear.in_features,
            new_dimension,
            device=self.device,
            dtype=self.dtype,
        )
        new_softplus = EasyTPPScaledSoftplus(new_dimension).to(
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            new_linear.weight.copy_(
                old_linear.weight.index_select(0, kept_outputs)
            )
            new_linear.bias.copy_(
                old_linear.bias.index_select(0, kept_outputs)
            )
            new_softplus.log_beta.copy_(
                self.softplus.log_beta.index_select(0, kept_outputs)
            )
            old_weights = torch.softmax(self.mixture_logits, dim=0)
            kept_weights = old_weights[
                torch.as_tensor(
                    kept_components,
                    dtype=torch.int64,
                    device=self.device,
                )
            ]
            kept_weights = kept_weights / kept_weights.sum()
            new_logits = torch.log(kept_weights)
            new_base = self.factor_intensity_base.index_select(
                1, kept_outputs
            )
            new_decay = self.factor_intensity_decay.index_select(
                1, kept_outputs
            )
        self.layer_intensity_hidden = new_linear
        self.factor_intensity_base = nn.Parameter(new_base.detach().clone())
        self.factor_intensity_decay = nn.Parameter(
            new_decay.detach().clone()
        )
        self.softplus = new_softplus
        self.mixture_logits = nn.Parameter(new_logits.detach().clone())
        self.n_components = len(kept_components)
        return kept_components


class ReferenceCOTICOutputMixture(ReferenceHistoryMixture):
    """One COTIC encoder and one intensity head with ``K*C`` outputs."""

    architecture_name = "vladislavzh_cotic_362b8ab_shared_encoder_kc_output"
    reference_commit = "362b8ab1f3cbb9e9dced2518e9daacf68235e77a"

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
                num_types=n_components * n_marks,
            )

    def expand_components(
        self,
        n_components: int,
        *,
        noise_scale: float = 0.01,
        initialization_seed: int = 0,
    ) -> None:
        """Clone a fitted K=1 output block without changing the encoder.

        The COTIC reference head initializes its intensity scale from the
        total number of output channels.  Constructing a ``K*C`` head from
        scratch therefore changes the inner intensity law as K changes.  A
        post-pretraining expansion instead preserves the fitted C-channel
        scale in every cloned component and perturbs only the affine output
        maps to break symmetry.
        """

        if self.n_components != 1:
            raise ValueError("component expansion requires a fitted K=1 head")
        if n_components <= 1:
            raise ValueError("expanded component count must exceed one")
        if noise_scale < 0.0:
            raise ValueError("output noise scale must be non-negative")
        old_layer = self.intensity_head.layer
        output_count = n_components * self.n_marks
        new_layer = nn.Linear(
            old_layer.in_features,
            output_count,
            device=self.device,
            dtype=self.dtype,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(initialization_seed)
        with torch.no_grad():
            weights = old_layer.weight.detach().repeat(n_components, 1)
            biases = old_layer.bias.detach().repeat(n_components)
            if noise_scale > 0.0:
                weight_noise = torch.randn(
                    weights.shape,
                    dtype=weights.dtype,
                    device=weights.device,
                    generator=generator,
                )
                bias_noise = torch.randn(
                    biases.shape,
                    dtype=biases.dtype,
                    device=biases.device,
                    generator=generator,
                )
                # Keep component zero as the exact pretrained process and
                # perturb only its overcomplete siblings.
                weight_noise[: self.n_marks].zero_()
                bias_noise[: self.n_marks].zero_()
                weights.add_(noise_scale * weight_noise)
                biases.add_(noise_scale * bias_noise)
            new_layer.weight.copy_(weights)
            new_layer.bias.copy_(biases)
            new_softplus_params = (
                self.intensity_head.softplus_params.detach().repeat(
                    1, 1, n_components
                )
            )
        self.intensity_head.layer = new_layer
        self.intensity_head.softplus_params = nn.Parameter(
            new_softplus_params.clone()
        )
        self.intensity_head.num_types = output_count
        self.mixture_logits = nn.Parameter(
            torch.zeros(
                n_components,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.n_components = int(n_components)

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
        shared = self.encoder(times, encoded_types)
        return shared[None].expand(self.n_components, -1, -1, -1)

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        convolution = self.intensity_head.convolution
        linear_part = convolution.kernel_network_weight(states)
        bias_part = states @ convolution.kernel_network_bias
        hidden = self.intensity_head.activation(
            elapsed[..., None] * linear_part + bias_part
        )
        logits = self.intensity_head.layer(hidden)
        scale = self.intensity_head.softplus_params.exp().reshape(
            self.n_components * self.n_marks
        )
        all_rates = self.intensity_head.softplus(logits) * scale + 1e-10
        return all_rates.reshape(
            *all_rates.shape[:-1],
            self.n_components,
            self.n_marks,
        )[..., component_index, :]

    def prune_component(self, component_index: int) -> tuple[int, ...]:
        """Physically remove one mixture logit and its C-output block."""

        if self.n_components <= 1:
            raise ValueError("cannot prune the final mixture component")
        if not 0 <= component_index < self.n_components:
            raise IndexError("component index out of range")
        kept_components = tuple(
            index
            for index in range(self.n_components)
            if index != component_index
        )
        kept_outputs = torch.as_tensor(
            [
                component * self.n_marks + mark
                for component in kept_components
                for mark in range(self.n_marks)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        old_layer = self.intensity_head.layer
        new_layer = nn.Linear(
            old_layer.in_features,
            len(kept_outputs),
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            new_layer.weight.copy_(
                old_layer.weight.index_select(0, kept_outputs)
            )
            new_layer.bias.copy_(
                old_layer.bias.index_select(0, kept_outputs)
            )
            new_softplus_params = (
                self.intensity_head.softplus_params.index_select(
                    2, kept_outputs
                )
            )
            old_weights = torch.softmax(self.mixture_logits, dim=0)
            kept_index = torch.as_tensor(
                kept_components,
                dtype=torch.int64,
                device=self.device,
            )
            kept_weights = old_weights.index_select(0, kept_index)
            new_logits = torch.log(kept_weights / kept_weights.sum())
        self.intensity_head.layer = new_layer
        self.intensity_head.softplus_params = nn.Parameter(
            new_softplus_params.detach().clone()
        )
        self.mixture_logits = nn.Parameter(new_logits.detach().clone())
        self.n_components = len(kept_components)
        return kept_components
