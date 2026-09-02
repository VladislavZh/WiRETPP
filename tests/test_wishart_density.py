from __future__ import annotations

import unittest

import torch

from wishart_tpp.model.wishart import sample_wishart, wishart_log_prob


class WishartDensityTest(unittest.TestCase):
    def test_mean_parameterized_density_matches_torch(self) -> None:
        mean = torch.tensor([[1.4, 0.25], [0.25, 0.9]], dtype=torch.float64)
        degrees = torch.tensor(7.0, dtype=torch.float64)
        generator = torch.Generator().manual_seed(12345)
        draws = sample_wishart(mean, degrees, 11, generator)
        expected = torch.distributions.Wishart(
            df=degrees,
            covariance_matrix=mean / degrees,
            validate_args=False,
        ).log_prob(draws)
        actual = wishart_log_prob(draws, mean, degrees)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_batched_density_keeps_sample_axis(self) -> None:
        means = torch.eye(3, dtype=torch.float64).repeat(2, 4, 1, 1)
        degrees = torch.full((2, 4), 8.0, dtype=torch.float64)
        draws = sample_wishart(means, degrees, 5, torch.Generator().manual_seed(91))
        values = wishart_log_prob(draws, means, degrees)
        self.assertEqual(values.shape, (2, 4, 5))
        self.assertTrue(bool(torch.isfinite(values).all()))


if __name__ == "__main__":
    unittest.main()
