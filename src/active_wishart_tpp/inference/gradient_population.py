"""Differentiable trace-constrained Wishart population parameters."""

import torch
from torch import nn
from active_wishart_tpp.model.wishart import (
    raw_cholesky_from_spd,
    spd_from_raw_cholesky,
)


class GradientPopulation(nn.Module):
    """Represent global Omega and alpha without coordinate solvers or damping."""

    eigenvalue_floor = 0.0001

    def __init__(self, means, alpha):
        super().__init__()
        self.raw_omega = nn.Parameter(torch.zeros_like(means))
        self.alpha = nn.Parameter(means.new_tensor(alpha))
        self.initialize(means, alpha)

    @torch.no_grad()
    def initialize(self, means, alpha):
        """Encode an existing population with only Cholesky roundoff error."""
        identity = torch.eye(means.shape[-1], device=means.device, dtype=means.dtype)
        unconstrained = (means - self.eigenvalue_floor * identity) / (
            1 - self.eigenvalue_floor
        )
        self.raw_omega.copy_(raw_cholesky_from_spd(unconstrained))
        self.alpha.fill_(alpha)

    def means(self):
        """Return positive-definite matrices with trace C and a numerical spectral floor."""
        matrix = spd_from_raw_cholesky(self.raw_omega)
        dimension = matrix.shape[-1]
        trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
        normalized = matrix * (dimension / trace)[..., None, None]
        identity = torch.eye(dimension, device=matrix.device, dtype=matrix.dtype)
        return (
            1 - self.eigenvalue_floor
        ) * normalized + self.eigenvalue_floor * identity

    @torch.no_grad()
    def project_alpha(self):
        """Keep alpha in the convex-intensity interval without sigmoid boundary trapping."""
        if not torch.isfinite(self.alpha) or not torch.isfinite(self.raw_omega).all():
            raise RuntimeError("Nonfinite joint population parameters")
        self.alpha.clamp_(0.0, 1.0)
