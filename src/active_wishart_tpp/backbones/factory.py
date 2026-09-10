"""Construct the COTIC or corrected THP bank used by both methods."""

from active_wishart_tpp.backbones.integration import MonteCarloRule
from active_wishart_tpp.backbones.paper_thp import PaperTHPIntensityBank
from active_wishart_tpp.cotic.model import CoticIntensityBank


def create_bank(config, n_marks, seed):
    """Create the same seeded single-head initialization for either method."""
    common = dict(
        n_components=1,
        n_marks=n_marks,
        hidden_size=config.hidden_size,
        integration_rule=MonteCarloRule(config.integral_samples, 0),
        initialization_seed=2026082200 + seed,
    )
    if config.backbone == "cotic":
        return CoticIntensityBank(
            input_channels=config.input_channels,
            layers=config.layers,
            kernel_size=config.kernel_size,
            dropout=0.0,
            dilation_factor=config.dilation_factor,
            **common,
        )
    return PaperTHPIntensityBank(
        layers=config.layers,
        heads=config.attention_heads,
        dropout=0.0,
        layer_norm=True,
        **common,
    )
