"""Pinned EasyTPP 0.2.1 model configuration."""

from easy_tpp.config_factory import ModelConfig


def easytpp_config(
    model_id: str,
    n_marks: int,
    hidden_size: int,
    layers: int,
    heads: int,
    dropout: float,
    layer_norm: bool,
    integral_samples: int,
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        hidden_size=hidden_size,
        time_emb_size=hidden_size,
        num_layers=layers,
        num_heads=heads,
        dropout_rate=dropout,
        use_ln=layer_norm,
        loss_integral_num_sample_per_step=integral_samples,
        use_mc_samples=False,
        num_event_types=n_marks,
        num_event_types_pad=n_marks + 2,
        event_pad_index=n_marks + 1,
        gpu=-1,
        model_specs={"beta": 1.0, "bias": True},
    )
