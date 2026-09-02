"""Variational E- and population M-steps."""

from wishart_tpp.inference.local import LocalPosterior, LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep

__all__ = ["LocalPosterior", "LocalWishartInference", "PopulationMstep"]
