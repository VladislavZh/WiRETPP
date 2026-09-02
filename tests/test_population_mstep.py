from __future__ import annotations

import unittest

import torch

from wishart_tpp.inference.population import PopulationMstep


class PopulationMstepTest(unittest.TestCase):
    def test_roundoff_below_trace_uses_zero_multiplier_limit(self) -> None:
        step = PopulationMstep()
        values = torch.full((64,), 1.0 - 1e-10, dtype=torch.float64)
        optimum = step._solve_eigenvalues(values, 64.0)
        self.assertAlmostEqual(float(optimum.sum()), 64.0, places=10)
        self.assertTrue(bool(torch.all(optimum > 0.0)))

    def test_exact_solution_satisfies_trace_and_beats_scaled_moment(self) -> None:
        means = torch.tensor(
            [
                [[[3.0, 0.4], [0.4, 0.5]]],
                [[[2.0, -0.2], [-0.2, 1.0]]],
                [[[4.0, 0.8], [0.8, 0.7]]],
            ]
        )
        responsibilities = torch.ones(3, 1)
        step = PopulationMstep()
        exact = step.update(means, responsibilities)
        sufficient = step.sufficient_mean(means, responsibilities)
        scaled = (
            sufficient
            * (2.0 / sufficient.diagonal(dim1=-2, dim2=-1).sum(-1))[:, None, None]
        )
        torch.testing.assert_close(
            exact.diagonal(dim1=-2, dim2=-1).sum(-1),
            torch.tensor([2.0]),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertLessEqual(
            float(step.objective(exact, means, responsibilities)),
            float(step.objective(scaled, means, responsibilities)) + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
