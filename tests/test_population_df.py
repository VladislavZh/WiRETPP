from __future__ import annotations

import unittest

import torch

from wishart_tpp.inference.local import LocalPosterior
from wishart_tpp.inference.population_df import PopulationDfMstep
from wishart_tpp.model.wishart import wishart_kl


class PopulationDfMstepTests(unittest.TestCase):
    @staticmethod
    def _posterior() -> tuple[LocalPosterior, torch.Tensor, torch.Tensor]:
        population = torch.tensor(
            [
                [[1.2, 0.1], [0.1, 0.8]],
                [[0.9, -0.05], [-0.05, 1.1]],
            ],
            dtype=torch.float64,
        )
        means = torch.stack((population * 0.9, population * 1.1))
        degrees = torch.tensor([[4.5, 6.0], [7.5, 5.5]], dtype=torch.float64)
        posterior = LocalPosterior(
            means,
            degrees,
            torch.zeros(2, 2, dtype=torch.float64),
            torch.zeros(2, 2, dtype=torch.float64),
        )
        responsibilities = torch.tensor([[0.8, 0.2], [0.35, 0.65]], dtype=torch.float64)
        return posterior, population, responsibilities

    def test_profile_matches_exact_weighted_kl_up_to_one_constant(self) -> None:
        posterior, population, gamma = self._posterior()
        values = torch.tensor([2.2, 4.5, 8.0], dtype=torch.float64)
        profile = PopulationDfMstep().profile(posterior, population, gamma, values)
        exact = torch.stack(
            [
                (
                    gamma
                    * wishart_kl(
                        posterior.means,
                        posterior.degrees_of_freedom,
                        population[None],
                        float(value),
                    )
                ).sum()
                / gamma.sum()
                for value in values
            ]
        )
        torch.testing.assert_close(
            profile.weighted_kl - profile.weighted_kl[0],
            exact - exact[0],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_profile_recovers_matching_prior_df(self) -> None:
        population = torch.eye(3, dtype=torch.float64)[None].repeat(2, 1, 1)
        means = population[None].repeat(4, 1, 1, 1)
        posterior = LocalPosterior(
            means,
            torch.full((4, 2), 7.0, dtype=torch.float64),
            torch.zeros(4, 2, dtype=torch.float64),
            torch.zeros(4, 2, dtype=torch.float64),
        )
        values = torch.tensor([3.0, 5.0, 7.0, 12.0], dtype=torch.float64)
        profile = PopulationDfMstep().profile(
            posterior,
            population,
            torch.full((4, 2), 0.5, dtype=torch.float64),
            values,
        )
        self.assertEqual(PopulationDfMstep.select(profile), 7.0)

    def test_grid_is_global_and_damping_respects_ratio(self) -> None:
        values = PopulationDfMstep.candidate_grid(
            64.0, 16, points=9, minimum=None, maximum=256.0
        )
        self.assertTrue(bool(torch.isclose(values, values.new_tensor(16.0)).any()))
        self.assertTrue(bool(torch.isclose(values, values.new_tensor(64.0)).any()))
        updated = PopulationDfMstep.damp(
            64.0, 16.0, 16, damping=0.25, maximum_ratio=2.0
        )
        self.assertGreaterEqual(updated, 32.0)
        self.assertLess(updated, 64.0)


if __name__ == "__main__":
    unittest.main()
