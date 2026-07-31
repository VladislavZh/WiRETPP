"""Mathematical primitives for latent-Wishart TPP attention.

This module is deliberately independent of the retired grid experiments.  It
contains the complete transformation from the learned Wishart mean to sampled
attention matrices, cluster gates, Monte-Carlo marginal likelihoods, and
posterior responsibilities used by the final NHP/THP/COTIC comparison.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def identity_raw_cholesky(
    n_matrices: int,
    dimension: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    minimum_diagonal: float = 1e-5,
) -> Tensor:
    """Return unconstrained Cholesky parameters representing identity."""

    if n_matrices <= 0 or dimension <= 0:
        raise ValueError("matrix counts and dimensions must be positive")
    if not 0.0 < minimum_diagonal < 1.0:
        raise ValueError("minimum diagonal must lie in (0, 1)")
    target = torch.tensor(
        1.0 - minimum_diagonal,
        dtype=dtype,
        device=device,
    )
    raw_diagonal = torch.log(torch.expm1(target))
    raw = torch.zeros(
        (n_matrices, dimension, dimension),
        dtype=dtype,
        device=device,
    )
    raw.diagonal(dim1=-2, dim2=-1).fill_(raw_diagonal)
    return raw


def correlation_matrices(matrices: Tensor) -> Tensor:
    """Convert positive-definite matrices to unit-diagonal correlations."""

    if matrices.ndim < 2 or matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError("matrices must have shape (..., D, D)")
    diagonal = matrices.diagonal(dim1=-2, dim2=-1)
    if bool(torch.any(diagonal <= 0.0)):
        raise ValueError("positive diagonal is required")
    inverse_scale = diagonal.rsqrt()
    return (
        matrices
        * inverse_scale[..., :, None]
        * inverse_scale[..., None, :]
    )


def squared_correlation_attention(matrices: Tensor) -> Tensor:
    """Return non-negative attention with unit column sums.

    Squaring removes the sign of a correlation.  Column normalization makes
    every source intensity channel distribute exactly one unit of mass, so the
    attention changes routing without changing the total instantaneous rate.
    """

    strength = correlation_matrices(matrices).square()
    return strength / strength.sum(dim=-2, keepdim=True)


def cluster_log_weights_from_matrices(
    matrices: Tensor,
    *,
    n_components: int,
    n_marks: int,
) -> Tensor:
    """Map diagonal block mass to categorical ``log p(z | W)``."""

    dimension = n_components * n_marks
    if matrices.shape[-2:] != (dimension, dimension):
        raise ValueError("matrix dimension must equal K * C")
    diagonal = matrices.diagonal(dim1=-2, dim2=-1)
    block_mass = diagonal.reshape(
        *diagonal.shape[:-1],
        n_components,
        n_marks,
    ).sum(dim=-1)
    return torch.log(block_mass) - torch.log(
        block_mass.sum(dim=-1, keepdim=True)
    )


def monte_carlo_marginal_scores(
    component_scores: Tensor,
    cluster_log_weights: Tensor,
) -> Tensor:
    """Integrate sampled ``W`` and exactly sum the finite cluster."""

    if (
        component_scores.ndim != 3
        or component_scores.shape != cluster_log_weights.shape
    ):
        raise ValueError("scores and gates must have shape (batch, S, K)")
    return (
        torch.logsumexp(
            component_scores + cluster_log_weights,
            dim=(1, 2),
        )
        - math.log(component_scores.shape[1])
    )


def sampled_posterior_log_weights(
    component_scores: Tensor,
    cluster_log_weights: Tensor,
) -> Tensor:
    """Return normalized ``log p(W_s,z=k | observations)``."""

    if (
        component_scores.ndim != 3
        or component_scores.shape != cluster_log_weights.shape
    ):
        raise ValueError("scores and gates must have shape (batch, S, K)")
    joint = component_scores + cluster_log_weights
    return joint - torch.logsumexp(joint, dim=(1, 2), keepdim=True)


def cluster_probabilities_from_sampled_posterior(
    posterior_log_weights: Tensor,
) -> Tensor:
    """Marginalize Wishart samples from posterior ``p(W_s,z | X)``."""

    if posterior_log_weights.ndim != 3:
        raise ValueError("posterior weights must have shape (batch, S, K)")
    return posterior_log_weights.exp().sum(dim=1)


def conditional_suffix_scores_from_samples(
    prefix_component_scores: Tensor,
    suffix_component_scores: Tensor,
    cluster_log_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``log p(suffix | prefix)`` and prefix posterior weights."""

    if prefix_component_scores.shape != suffix_component_scores.shape:
        raise ValueError("prefix and suffix score shapes must match")
    posterior = sampled_posterior_log_weights(
        prefix_component_scores,
        cluster_log_weights,
    )
    return (
        torch.logsumexp(posterior + suffix_component_scores, dim=(1, 2)),
        posterior,
    )
