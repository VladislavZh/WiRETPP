"""Synthetic K=3, C=5 Hawkes benchmark used in the final experiment."""

from .paper_k3c5 import (
    ExponentialHawkesParameters,
    PaperK3C5Dataset,
    generate_paper_k3c5,
    simulate_exponential_hawkes,
)

__all__ = [
    "ExponentialHawkesParameters",
    "PaperK3C5Dataset",
    "generate_paper_k3c5",
    "simulate_exponential_hawkes",
]
