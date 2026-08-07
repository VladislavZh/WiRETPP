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
from torch.nn import functional as F


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


def legacy_row_normalized_attention(matrices: Tensor) -> Tensor:
    """Return the legacy ``D(W)^{-1} (W o W)`` transformation.

    Rows, rather than columns, are normalized.  This is Equation (3.9) in
    the signed WiRE-TPP chapter and is intentionally kept separate from the
    squared-correlation attention used by the previously published snapshot.
    """

    if matrices.ndim < 2 or matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError("matrices must have shape (..., D, D)")
    strength = matrices.square()
    return strength / strength.sum(dim=-1, keepdim=True)


def signed_correlation_coupling(matrices: Tensor) -> Tensor:
    """Return ``B(W) = Corr(W) - I`` while preserving off-diagonal signs."""

    correlation = correlation_matrices(matrices)
    identity = torch.eye(
        correlation.shape[-1],
        dtype=correlation.dtype,
        device=correlation.device,
    )
    return correlation - identity


def inverse_softplus(values: Tensor) -> Tensor:
    """Stable inverse of the unit-temperature softplus on positive values."""

    if bool(torch.any(values <= 0.0)):
        raise ValueError("inverse softplus requires strictly positive values")
    return values + torch.log(-torch.expm1(-values))


def signed_intensity_transform(
    base_intensities: Tensor,
    matrices: Tensor,
    interaction_strength: Tensor | float,
    *,
    activity_epsilon: float = 1e-8,
    interaction_mode: str = "residual",
    n_components: int | None = None,
    n_marks: int | None = None,
    block_local_activity: bool = False,
    within_block_gain: float = 1.0,
    between_block_gain: float = 1.0,
) -> Tensor:
    """Apply the signed WiRE-TPP transformation from Equations (3.4)-(3.5).

    ``base_intensities`` has shape ``(..., D)`` and ``matrices`` has shape
    ``(..., S, D, D)`` with the same leading dimensions.  The returned tensor
    has shape ``(..., S, D)``.  The canonical logits are defined as the exact
    unit-softplus inverse of the positive base intensities; this preserves any
    existing neural TPP link exactly when ``interaction_strength == 0``.
    """

    if activity_epsilon <= 0.0:
        raise ValueError("activity epsilon must be positive")
    if interaction_mode not in {"residual", "convex"}:
        raise ValueError("interaction mode must be residual or convex")
    if (
        not math.isfinite(within_block_gain)
        or not math.isfinite(between_block_gain)
        or within_block_gain < 0.0
        or between_block_gain < 0.0
    ):
        raise ValueError("within- and between-block gains must be finite and non-negative")
    if matrices.ndim < 3 or matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError("matrices must have shape (..., S, D, D)")
    if base_intensities.shape[:-1] != matrices.shape[:-3]:
        raise ValueError("base intensities and matrices have incompatible rows")
    if base_intensities.shape[-1] != matrices.shape[-1]:
        raise ValueError("base intensity dimension must match matrix dimension")
    alpha = torch.as_tensor(
        interaction_strength,
        dtype=base_intensities.dtype,
        device=base_intensities.device,
    )
    if bool(torch.any(alpha < 0.0)):
        raise ValueError("interaction strength must be non-negative")
    if interaction_mode == "convex" and bool(torch.any(alpha > 1.0)):
        raise ValueError("convex interaction strength must lie in [0, 1]")
    coupling = signed_correlation_coupling(matrices)
    global_activity = base_intensities / (
        activity_epsilon + base_intensities.sum(dim=-1, keepdim=True)
    )
    if block_local_activity:
        if n_components is None or n_marks is None:
            raise ValueError(
                "block-local activity requires n_components and n_marks"
            )
        if n_components <= 0 or n_marks <= 0:
            raise ValueError("component and mark counts must be positive")
        if n_components * n_marks != base_intensities.shape[-1]:
            raise ValueError("block-local dimensions must equal K*C")
        blocks = base_intensities.reshape(
            *base_intensities.shape[:-1], n_components, n_marks
        )
        local_activity = blocks / (
            activity_epsilon + blocks.sum(dim=-1, keepdim=True)
        )
        local_activity = local_activity.reshape_as(base_intensities)
        block_index = torch.arange(
            base_intensities.shape[-1], device=base_intensities.device
        ) // n_marks
        within_mask = block_index[:, None] == block_index[None, :]
        within_coupling = coupling * within_mask.to(coupling.dtype)
        between_coupling = coupling * (~within_mask).to(coupling.dtype)
        perturbation = (
            within_block_gain
            * torch.einsum(
                "...sij,...j->...si", within_coupling, local_activity
            )
            + between_block_gain
            * torch.einsum(
                "...sij,...j->...si", between_coupling, global_activity
            )
        )
    else:
        if within_block_gain != 1.0 or between_block_gain != 1.0:
            raise ValueError("custom block gains require block-local activity")
        perturbation = torch.einsum(
            "...sij,...j->...si",
            coupling,
            global_activity,
        )
    logits = inverse_softplus(base_intensities)
    base_logits = logits.unsqueeze(-2)
    if interaction_mode == "residual":
        transformed_logits = base_logits + alpha * perturbation
    else:
        transformed_logits = (
            (1.0 - alpha) * base_logits + alpha * perturbation
        )
    return F.softplus(transformed_logits)


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
