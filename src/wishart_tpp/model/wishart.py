"""Wishart distribution primitives in mean/df parameterization."""

from __future__ import annotations

import torch
from torch import Tensor


def stable_cholesky(matrix: Tensor) -> Tensor:
    """Cholesky with a deterministic eigenvalue floor for roundoff failures."""

    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    dimension = symmetric.shape[-1]
    scale = symmetric.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    jitter = torch.finfo(symmetric.dtype).eps * 10.0 * dimension * scale
    identity = torch.eye(dimension, device=symmetric.device, dtype=symmetric.dtype)
    regularized = symmetric + jitter * identity
    factor, info = torch.linalg.cholesky_ex(regularized)
    if bool(torch.all(info == 0)):
        return factor
    values, vectors = torch.linalg.eigh(regularized)
    floor = jitter.squeeze(-1)
    stabilized = (vectors * values.clamp_min(floor).unsqueeze(-2)) @ vectors.transpose(
        -1, -2
    )
    stabilized = 0.5 * (stabilized + stabilized.transpose(-1, -2))
    # The repair is a numerical projection, not a model parameterization.
    # A straight-through correction avoids undefined eigenvector gradients at
    # repeated eigenvalues while retaining gradients through the input matrix.
    straight_through = regularized + (stabilized - regularized).detach()
    return torch.linalg.cholesky(straight_through)


def matrix_square_root(matrix: Tensor) -> Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(symmetric)
    scaled = vectors * values.clamp_min(1e-10).sqrt().unsqueeze(-2)
    return scaled @ vectors.transpose(-1, -2)


def inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def raw_cholesky_from_spd(matrix: Tensor, floor: float = 1e-4) -> Tensor:
    cholesky = torch.linalg.cholesky(0.5 * (matrix + matrix.transpose(-1, -2)))
    diagonal = inverse_softplus(
        (cholesky.diagonal(dim1=-2, dim2=-1) - floor).clamp_min(1e-8)
    )
    return torch.tril(cholesky, diagonal=-1) + torch.diag_embed(diagonal)


def spd_from_raw_cholesky(raw: Tensor, floor: float = 1e-4) -> Tensor:
    diagonal = torch.nn.functional.softplus(raw.diagonal(dim1=-2, dim2=-1)) + floor
    cholesky = torch.tril(raw, diagonal=-1) + torch.diag_embed(diagonal)
    matrix = cholesky @ cholesky.transpose(-1, -2)
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def multivariate_digamma(value: Tensor, dimension: int) -> Tensor:
    offsets = 0.5 * torch.arange(dimension, device=value.device, dtype=value.dtype)
    return torch.digamma(value[..., None] - offsets).sum(dim=-1)


def wishart_kl(
    posterior_mean: Tensor,
    posterior_df: Tensor,
    prior_mean: Tensor,
    prior_df: float,
) -> Tensor:
    """KL(q||p) for Wishart(kappa, mean/kappa) laws."""

    dimension = posterior_mean.shape[-1]
    q_df = posterior_df
    p_df = posterior_mean.new_tensor(prior_df)
    q_logdet = torch.linalg.slogdet(posterior_mean).logabsdet
    p_logdet = torch.linalg.slogdet(prior_mean).logabsdet
    q_scale_logdet = q_logdet - dimension * q_df.log()
    p_scale_logdet = p_logdet - dimension * p_df.log()
    log_two = posterior_mean.new_tensor(2.0).log()

    # E_q[log|U|] is the only non-linear Wishart sufficient statistic.
    expected_q_logdet = (
        multivariate_digamma(0.5 * q_df, dimension)
        + dimension * log_two
        + q_scale_logdet
    )
    trace_term = p_df * torch.einsum(
        "...ii->...", torch.linalg.solve(prior_mean, posterior_mean)
    )

    # Combine expected log densities; constants are retained for free energies.
    return (
        0.5 * (q_df - p_df) * expected_q_logdet
        - 0.5 * q_df * dimension
        + 0.5 * trace_term
        - 0.5 * q_df * dimension * log_two
        - 0.5 * q_df * q_scale_logdet
        - torch.special.multigammaln(0.5 * q_df, dimension)
        + 0.5 * p_df * dimension * log_two
        + 0.5 * p_df * p_scale_logdet
        + torch.special.multigammaln(0.5 * p_df, dimension)
    )


def wishart_log_prob(
    draws: Tensor,
    mean: Tensor,
    degrees_of_freedom: Tensor | float,
) -> Tensor:
    """Log density of mean-parameterized Wishart draws.

    ``draws`` must have shape ``... x samples x C x C`` while ``mean`` has
    shape ``... x C x C`` and ``degrees_of_freedom`` broadcasts to ``...``.
    The distribution is ``Wishart(df, scale=mean/df)``.
    """

    if draws.ndim != mean.ndim + 1:
        raise ValueError("draws must add exactly one sample axis to mean")
    if draws.shape[:-3] != mean.shape[:-2]:
        raise ValueError("draw and mean batch shapes must agree")
    if draws.shape[-2:] != mean.shape[-2:] or mean.shape[-1] != mean.shape[-2]:
        raise ValueError("Wishart matrices must be square and dimension-matched")

    dimension = mean.shape[-1]
    degrees = torch.as_tensor(degrees_of_freedom, device=mean.device, dtype=mean.dtype)
    degrees = torch.broadcast_to(degrees, mean.shape[:-2])
    if bool((degrees <= dimension - 1.0).any()):
        raise ValueError("Wishart degrees of freedom must exceed C-1")

    draw_sign, draw_logdet = torch.linalg.slogdet(draws)
    mean_sign, mean_logdet = torch.linalg.slogdet(mean)
    if not bool((draw_sign > 0).all()) or not bool((mean_sign > 0).all()):
        raise ValueError("Wishart draws and means must be positive definite")

    expanded_mean = mean.unsqueeze(-3).expand_as(draws)
    solved = torch.linalg.solve(expanded_mean, draws)
    trace = solved.diagonal(dim1=-2, dim2=-1).sum(-1) * degrees.unsqueeze(-1)
    scale_logdet = mean_logdet - dimension * degrees.log()
    log_two = mean.new_tensor(2.0).log()
    log_normalizer = (
        0.5 * degrees * dimension * log_two
        + 0.5 * degrees * scale_logdet
        + torch.special.multigammaln(0.5 * degrees, dimension)
    )
    return (
        0.5 * (degrees.unsqueeze(-1) - dimension - 1.0) * draw_logdet
        - 0.5 * trace
        - log_normalizer.unsqueeze(-1)
    )


def sample_wishart(
    mean: Tensor,
    degrees_of_freedom: Tensor,
    samples: int,
    generator: torch.Generator,
) -> Tensor:
    """Pathwise Bartlett samples with shape ``batch x samples x C x C``."""

    dimension = mean.shape[-1]
    offsets = torch.arange(dimension, device=mean.device, dtype=mean.dtype)
    concentration = 0.5 * (degrees_of_freedom[..., None] - offsets)
    concentration = concentration[..., None, :].expand(
        *mean.shape[:-2], samples, dimension
    )
    # Bartlett's triangular factor supplies reparameterized Wishart draws.
    diagonal = (
        (2.0 * torch._standard_gamma(concentration, generator=generator))
        .clamp_min(torch.finfo(mean.dtype).tiny)
        .sqrt()
    )
    gaussian = torch.randn(
        (*mean.shape[:-2], samples, dimension, dimension),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    bartlett = torch.tril(gaussian, diagonal=-1) + torch.diag_embed(diagonal)

    # Mean parameterization uses scale = mean / degrees_of_freedom.
    scale_root = stable_cholesky(mean) / degrees_of_freedom.sqrt()[..., None, None]
    draw_root = torch.einsum("...ij,...sjh->...sih", scale_root, bartlett)
    draws = draw_root @ draw_root.transpose(-1, -2)
    return 0.5 * (draws + draws.transpose(-1, -2))
