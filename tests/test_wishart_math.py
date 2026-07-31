from __future__ import annotations

import unittest

import torch

from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    conditional_suffix_scores_from_samples,
    identity_raw_cholesky,
    monte_carlo_marginal_scores,
    squared_correlation_attention,
)


class WishartMathTests(unittest.TestCase):
    def test_identity_raw_cholesky_and_attention(self) -> None:
        raw = identity_raw_cholesky(
            2,
            6,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        diagonal = torch.nn.functional.softplus(raw.diagonal(dim1=-2, dim2=-1)) + 1e-5
        cholesky = torch.tril(raw, diagonal=-1) + torch.diag_embed(diagonal)
        matrices = cholesky @ cholesky.transpose(-1, -2)
        torch.testing.assert_close(matrices, torch.eye(6, dtype=torch.float64)[None].repeat(2, 1, 1))
        attention = squared_correlation_attention(matrices)
        torch.testing.assert_close(attention, matrices)
        gates = cluster_log_weights_from_matrices(
            matrices,
            n_components=3,
            n_marks=2,
        ).exp()
        torch.testing.assert_close(gates, torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64))

    def test_marginal_and_conditional_match_manual_logsumexp(self) -> None:
        prefix = torch.tensor([[[1.0, 0.0], [0.5, -0.5]]])
        suffix = torch.tensor([[[0.2, 0.4], [-0.1, 0.3]]])
        gates = torch.full_like(prefix, -torch.log(torch.tensor(2.0)))
        full = prefix + suffix
        expected_full = torch.logsumexp(full + gates, dim=(1, 2)) - torch.log(torch.tensor(2.0))
        torch.testing.assert_close(
            monte_carlo_marginal_scores(full, gates), expected_full
        )
        conditional, posterior = conditional_suffix_scores_from_samples(
            prefix, suffix, gates
        )
        torch.testing.assert_close(
            conditional,
            torch.logsumexp(posterior + suffix, dim=(1, 2)),
        )
        torch.testing.assert_close(posterior.exp().sum(dim=(1, 2)), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
