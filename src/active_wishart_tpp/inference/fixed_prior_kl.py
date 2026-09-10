"""Wishart KL with population-only linear algebra cached for one E-step fit."""

import torch
from active_wishart_tpp.model.wishart import multivariate_digamma


class FixedPriorKL:
    """Evaluate full KL(q||p), retaining constants needed for responsibilities."""

    @torch.no_grad()
    def __init__(self, means, degrees):
        self.dimension = means.shape[-1]
        self.degrees = means.new_tensor(degrees)
        self.means = means.detach()
        self.scale_logdet = (
            torch.linalg.slogdet(means.detach()).logabsdet
            - self.dimension * self.degrees.log()
        )
        self.log_two = means.new_tensor(2.0).log()
        self.log_gamma = torch.special.multigammaln(0.5 * self.degrees, self.dimension)

    def __call__(self, means, degrees):
        """Differentiate only the posterior mean and local degrees of freedom."""
        dimension, p_df, q_df = (self.dimension, self.degrees, degrees)
        q_scale_logdet = torch.linalg.slogdet(means).logabsdet - dimension * q_df.log()
        expected_logdet = (
            multivariate_digamma(0.5 * q_df, dimension)
            + dimension * self.log_two
            + q_scale_logdet
        )
        solved = torch.linalg.solve(self.means.expand_as(means), means)
        trace = p_df * torch.einsum("...ii->...", solved)
        return (
            0.5 * (q_df - p_df) * expected_logdet
            - 0.5 * q_df * dimension
            + 0.5 * trace
            - 0.5 * q_df * dimension * self.log_two
            - 0.5 * q_df * q_scale_logdet
            - torch.special.multigammaln(0.5 * q_df, dimension)
            + 0.5 * p_df * dimension * self.log_two
            + 0.5 * p_df * self.scale_logdet
            + self.log_gamma
        )
