"""Finite TPP mixtures built from reference THP and COTIC architectures."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model.torch_model.torch_thp import THP

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
    LatentNHPMixtureBatchTrace,
)


def _load_cotic_classes() -> tuple[type[nn.Module], type[nn.Module]]:
    """Load the checked-out reference implementation without copying it."""

    project_root = Path(__file__).resolve().parents[3]
    cotic_root = project_root / "reference" / "COTIC"
    if not (cotic_root / "src/models/components/cotic/cotic.py").is_file():
        raise FileNotFoundError(
            "reference/COTIC is required; clone VladislavZh/COTIC there"
        )
    root_text = str(cotic_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.models.components.cotic.cotic import COTIC
    from src.models.components.cotic.head.intensity_head import (
        IntensityHeadLinear,
    )

    return COTIC, IntensityHeadLinear


class ReferenceHistoryMixture(nn.Module):
    """Common horizon-aware likelihood interface for causal history encoders."""

    architecture_name = "abstract"

    def __init__(
        self,
        n_components: int,
        n_marks: int,
        *,
        horizon: float,
        quadrature_order: int,
    ) -> None:
        super().__init__()
        if n_components <= 0 or n_marks <= 0:
            raise ValueError("history mixture requires K > 0 and C > 0")
        if horizon <= 0.0 or quadrature_order <= 0:
            raise ValueError("horizon and quadrature order must be positive")
        self.n_components = int(n_components)
        self.n_marks = int(n_marks)
        self.horizon = float(horizon)
        self.quadrature_order = int(quadrature_order)
        self.mixture_logits = nn.Parameter(torch.zeros(n_components))
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.register_buffer(
            "_quadrature_nodes",
            torch.as_tensor(nodes, dtype=torch.float32),
        )
        self.register_buffer(
            "_quadrature_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return self.mixture_logits.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mixture_logits.dtype

    def mixture_log_weights(self) -> Tensor:
        return torch.log_softmax(self.mixture_logits, dim=0)

    def _encoded_types(
        self,
        marks: Tensor,
        valid: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def _bos_type(self) -> int:
        raise NotImplementedError

    def _pad_type(self) -> int:
        raise NotImplementedError

    def encode_histories(
        self,
        times: Tensor,
        encoded_types: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Return states with shape ``(K, paths, history, hidden)``."""

        raise NotImplementedError

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        """Map post-history states and elapsed time to mark intensities."""

        raise NotImplementedError

    def _prepare_histories(
        self,
        sequences: tuple[MarkedSequence, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        maximum_events = max(sequence.count for sequence in sequences)
        history_length = maximum_events + 1
        times = torch.zeros(
            (len(sequences), history_length),
            dtype=self.dtype,
            device=self.device,
        )
        marks = torch.zeros(
            (len(sequences), history_length),
            dtype=torch.int64,
            device=self.device,
        )
        valid = torch.zeros(
            (len(sequences), history_length),
            dtype=torch.bool,
            device=self.device,
        )
        marks[:, 0] = self._bos_type()
        valid[:, 0] = True
        for path_index, sequence in enumerate(sequences):
            count = sequence.count
            if count:
                times[path_index, 1 : count + 1] = torch.as_tensor(
                    sequence.times,
                    dtype=self.dtype,
                    device=self.device,
                )
                marks[path_index, 1 : count + 1] = torch.as_tensor(
                    sequence.marks,
                    dtype=torch.int64,
                    device=self.device,
                )
                valid[path_index, 1 : count + 1] = True
        return times, self._encoded_types(marks, valid), valid

    def build_trace(
        self,
        sequences: Iterable[MarkedSequence],
        *,
        boundary: float | None = None,
    ) -> LatentNHPMixtureBatchTrace:
        """Build event and deterministic compensator rows for all components."""

        sequence_list = tuple(sequences)
        if not sequence_list:
            raise ValueError("sequences must be non-empty")
        maximum_horizon = max(sequence.horizon for sequence in sequence_list)
        if boundary is not None and not 0.0 < boundary < maximum_horizon:
            raise ValueError("boundary must lie strictly inside the horizon")
        history_times, encoded_types, valid = self._prepare_histories(
            sequence_list
        )
        states = self.encode_histories(
            history_times,
            encoded_types,
            valid,
        )
        if states.shape[:3] != (
            self.n_components,
            len(sequence_list),
            history_times.shape[1],
        ):
            raise RuntimeError("history encoder returned an invalid shape")

        event_times: list[float] = []
        event_marks: list[int] = []
        event_paths: list[int] = []
        event_history: list[int] = []
        event_elapsed: list[float] = []
        segment_starts: list[float] = []
        segment_ends: list[float] = []
        segment_paths: list[int] = []
        segment_history: list[int] = []
        for path_index, sequence in enumerate(sequence_list):
            starts = np.concatenate(([0.0], sequence.times))
            ends = np.concatenate((sequence.times, [sequence.horizon]))
            previous = 0.0
            for event_index, (event_time, mark) in enumerate(
                zip(sequence.times, sequence.marks)
            ):
                event_times.append(float(event_time))
                event_marks.append(int(mark))
                event_paths.append(path_index)
                event_history.append(event_index)
                event_elapsed.append(float(event_time) - previous)
                previous = float(event_time)
            for history_index, (start, stop) in enumerate(zip(starts, ends)):
                cuts = [float(start)]
                if (
                    boundary is not None
                    and float(start) < boundary < float(stop)
                ):
                    cuts.append(float(boundary))
                cuts.append(float(stop))
                for segment_start, segment_stop in zip(cuts[:-1], cuts[1:]):
                    if segment_stop <= segment_start:
                        continue
                    segment_starts.append(segment_start)
                    segment_ends.append(segment_stop)
                    segment_paths.append(path_index)
                    segment_history.append(history_index)

        event_paths_tensor = torch.as_tensor(
            event_paths,
            dtype=torch.int64,
            device=self.device,
        )
        event_history_tensor = torch.as_tensor(
            event_history,
            dtype=torch.int64,
            device=self.device,
        )
        event_elapsed_tensor = torch.as_tensor(
            event_elapsed,
            dtype=self.dtype,
            device=self.device,
        )
        event_base = torch.stack(
            tuple(
                self.component_rates_from_states(
                    component,
                    states[
                        component,
                        event_paths_tensor,
                        event_history_tensor,
                    ],
                    event_elapsed_tensor,
                )
                for component in range(self.n_components)
            ),
            dim=1,
        )

        segment_start_tensor = torch.as_tensor(
            segment_starts,
            dtype=self.dtype,
            device=self.device,
        )
        segment_end_tensor = torch.as_tensor(
            segment_ends,
            dtype=self.dtype,
            device=self.device,
        )
        segment_paths_tensor = torch.as_tensor(
            segment_paths,
            dtype=torch.int64,
            device=self.device,
        )
        segment_history_tensor = torch.as_tensor(
            segment_history,
            dtype=torch.int64,
            device=self.device,
        )
        lengths = segment_end_tensor - segment_start_tensor
        local_elapsed = (
            0.5
            * lengths[:, None]
            * (self._quadrature_nodes[None, :] + 1.0)
        )
        quadrature_times = (
            segment_start_tensor[:, None] + local_elapsed
        ).reshape(-1)
        quadrature_weights = (
            0.5 * lengths[:, None] * self._quadrature_weights[None, :]
        ).reshape(-1)
        quadrature_paths = (
            segment_paths_tensor[:, None]
            .expand(-1, self.quadrature_order)
            .reshape(-1)
        )
        quadrature_base = torch.stack(
            tuple(
                self.component_rates_from_states(
                    component,
                    states[
                        component,
                        segment_paths_tensor,
                        segment_history_tensor,
                    ][:, None, :].expand(
                        -1,
                        self.quadrature_order,
                        -1,
                    ),
                    local_elapsed,
                ).reshape(-1, self.n_marks)
                for component in range(self.n_components)
            ),
            dim=1,
        )
        return LatentNHPMixtureBatchTrace(
            n_paths=len(sequence_list),
            horizon=float(maximum_horizon),
            path_horizons=torch.as_tensor(
                [sequence.horizon for sequence in sequence_list],
                dtype=self.dtype,
                device=self.device,
            ),
            n_components=self.n_components,
            n_marks=self.n_marks,
            event_times=torch.as_tensor(
                event_times,
                dtype=self.dtype,
                device=self.device,
            ),
            event_marks=torch.as_tensor(
                event_marks,
                dtype=torch.int64,
                device=self.device,
            ),
            event_path_indices=event_paths_tensor,
            event_base_intensities=event_base,
            quadrature_times=quadrature_times,
            quadrature_weights=quadrature_weights,
            quadrature_path_indices=quadrature_paths,
            quadrature_base_intensities=quadrature_base,
        )

    def component_scores_for_indices(
        self,
        sequences: Iterable[MarkedSequence],
        component_indices: Iterable[int],
    ) -> Tensor:
        indices = tuple(int(index) for index in component_indices)
        if not indices:
            raise ValueError("component_indices must be non-empty")
        if any(
            index < 0 or index >= self.n_components for index in indices
        ):
            raise IndexError("component index out of range")
        scores = base_nhp_component_scores_from_trace(
            self.build_trace(sequences)
        )
        return scores[:, indices]

    def component_scores(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        """Return classical marked-TPP scores for every component."""

        return self.component_scores_for_indices(
            sequences,
            range(self.n_components),
        )

    def log_likelihoods(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        component = self.component_scores(sequences)
        return torch.logsumexp(
            component + self.mixture_log_weights()[None, :],
            dim=1,
        )

    def negative_log_likelihood(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        return -self.log_likelihoods(sequences).sum()

    def responsibilities(
        self,
        sequences: Iterable[MarkedSequence],
    ) -> Tensor:
        component = self.component_scores(sequences)
        return torch.softmax(
            component + self.mixture_log_weights()[None, :],
            dim=1,
        )


class ReferenceEasyTPPTHPMixture(ReferenceHistoryMixture):
    """K=1 EasyTPP THP oracle retained only for architecture unit tests."""

    architecture_name = "easytpp_thp_0.2.1"

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
        if n_components != 1:
            raise ValueError("independent THP components are not supported")
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
        components: list[THP] = []
        for component in range(n_components):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed + component)
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
                components.append(THP(config))
        self.components = nn.ModuleList(components)

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
        return torch.stack(
            tuple(
                component(times, encoded_types, attention_mask)
                for component in self.components
            )
        )

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        component = self.components[component_index]
        logits = (
            component.layer_intensity_hidden(states)
            + component.factor_intensity_base
            + elapsed[..., None] * component.factor_intensity_decay
        )
        return component.softplus(logits) + 1e-10


class ReferenceCOTICMixture(ReferenceHistoryMixture):
    """K=1 COTIC oracle retained only for architecture unit tests."""

    architecture_name = "vladislavzh_cotic_362b8ab"
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
        if n_components != 1:
            raise ValueError("independent COTIC components are not supported")
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
        encoders: list[nn.Module] = []
        heads: list[nn.Module] = []
        for component in range(n_components):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed + component)
                encoders.append(
                    COTIC(
                        in_channels=input_channels,
                        kernel_size=kernel_size,
                        nb_filters=hidden_size,
                        nb_layers=num_layers,
                        num_types=n_marks + 1,
                        dropout=dropout,
                        dilation_factor=dilation_factor,
                    )
                )
                heads.append(
                    IntensityHeadLinear(
                        kernel_size=1,
                        nb_filters=hidden_size,
                        num_types=n_marks,
                    )
                )
        self.encoders = nn.ModuleList(encoders)
        self.intensity_heads = nn.ModuleList(heads)

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
        return torch.stack(
            tuple(
                encoder(times, encoded_types)
                for encoder in self.encoders
            )
        )

    def component_rates_from_states(
        self,
        component_index: int,
        states: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        """Exact kernel-size-one COTIC intensity at arbitrary elapsed times."""

        head = self.intensity_heads[component_index]
        convolution = head.convolution
        linear_part = convolution.kernel_network_weight(states)
        bias_part = states @ convolution.kernel_network_bias
        hidden = head.activation(
            elapsed[..., None] * linear_part + bias_part
        )
        logits = head.layer(hidden)
        scale = head.softplus_params.exp().reshape(self.n_marks)
        return head.softplus(logits) * scale + 1e-10
