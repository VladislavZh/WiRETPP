"""Create a chosen intensity-bank adapter without touching the algorithm."""

from __future__ import annotations

from wishart_tpp.backbones.base import IntensityBank
from wishart_tpp.backbones.integration import create_integration_rule
from wishart_tpp.backbones.nhp import NHPIntensityBank
from wishart_tpp.backbones.rmtpp import RMTPPIntensityBank
from wishart_tpp.backbones.thp import THPIntensityBank
from wishart_tpp.config import ModelConfig
from wishart_tpp.cotic import CoticIntensityBank


class BackboneFactory:
    @staticmethod
    def _single_component(
        config: ModelConfig,
        n_marks: int,
        initialization_seed: int,
    ) -> IntensityBank:
        common = {
            "n_components": 1,
            "n_marks": n_marks,
            "hidden_size": config.hidden_size,
            "integration_rule": create_integration_rule(
                config.integral_method,
                config.integral_samples,
                config.integral_seed,
            ),
            "initialization_seed": initialization_seed,
        }
        if config.backbone == "cotic":
            bank = CoticIntensityBank(
                input_channels=config.input_channels,
                layers=config.layers,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
                dilation_factor=config.dilation_factor,
                **common,
            )
        elif config.backbone == "thp":
            bank = THPIntensityBank(
                layers=config.layers,
                heads=config.attention_heads,
                dropout=config.dropout,
                layer_norm=config.use_layer_norm,
                **common,
            )
        elif config.backbone == "nhp":
            bank = NHPIntensityBank(**common)
        elif config.backbone == "rmtpp":
            bank = RMTPPIntensityBank(**common)
        else:
            raise ValueError(f"unknown backbone: {config.backbone}")
        return bank

    @classmethod
    def create(
        cls,
        config: ModelConfig,
        n_components: int,
        n_marks: int,
        initialization_seed: int,
        expansion_seed: int | None = None,
    ) -> IntensityBank:
        # Fit-compatible construction always starts from one population head.
        bank = cls._single_component(config, n_marks, initialization_seed)
        if n_components > 1:
            # Candidate heads then share the encoder and begin as nearby copies.
            bank.expand_components(
                n_components,
                config.component_noise,
                initialization_seed + 1 if expansion_seed is None else expansion_seed,
            )
        return bank
