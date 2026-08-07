from __future__ import annotations

import unittest

import numpy as np
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
)
from lal_wishart.models.reference_output_mixtures import (
    ReferenceNHPOutputMixture,
)
from lal_wishart.models.signed_wishart import (
    LegacyRowWishartTPP,
    SignedWishartTPP,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    inverse_softplus,
    legacy_row_normalized_attention,
    signed_correlation_coupling,
    signed_intensity_transform,
)


class SignedWishartMathTests(unittest.TestCase):
    def test_signed_coupling_preserves_sign_and_scale(self) -> None:
        matrix = torch.tensor(
            [[1.0, -0.5], [-0.5, 1.0]],
            dtype=torch.float64,
        )
        first = signed_correlation_coupling(matrix)
        second = signed_correlation_coupling(7.0 * matrix)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            torch.diagonal(first),
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertEqual(float(first[0, 1]), -0.5)
        torch.testing.assert_close(first, first.T)

    def test_signed_transform_suppresses_and_excites(self) -> None:
        base = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
        negative = torch.tensor(
            [[[[1.0, -0.5], [-0.5, 1.0]]]],
            dtype=torch.float64,
        )
        positive = negative.clone()
        positive[..., 0, 1] = 0.5
        positive[..., 1, 0] = 0.5
        suppressed = signed_intensity_transform(base, negative, 1.0)
        excited = signed_intensity_transform(base, positive, 1.0)
        self.assertTrue(bool(torch.all(suppressed < base[:, None, :])))
        self.assertTrue(bool(torch.all(excited > base[:, None, :])))

    def test_alpha_zero_is_exact_base_embedding(self) -> None:
        base = torch.tensor(
            [[1e-8, 0.2, 1.0, 10.0]],
            dtype=torch.float64,
        )
        raw = torch.tensor(
            [
                [
                    [2.0, -0.2, 0.1, 0.0],
                    [-0.2, 1.5, 0.0, 0.1],
                    [0.1, 0.0, 1.2, -0.1],
                    [0.0, 0.1, -0.1, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        matrices = raw[:, None].repeat(1, 3, 1, 1)
        transformed = signed_intensity_transform(base, matrices, 0.0)
        torch.testing.assert_close(
            transformed,
            base[:, None, :].expand_as(transformed),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_convex_transform_weights_base_and_wishart_terms(self) -> None:
        base = torch.tensor([[0.2, 1.0]], dtype=torch.float64)
        identity = torch.eye(2, dtype=torch.float64)[None, None]
        alpha = 0.25
        transformed = signed_intensity_transform(
            base,
            identity,
            alpha,
            interaction_mode="convex",
        )
        expected = torch.nn.functional.softplus(
            (1.0 - alpha) * inverse_softplus(base)
        )[:, None]
        torch.testing.assert_close(transformed, expected)

    def test_block_local_activity_removes_global_k_attenuation(self) -> None:
        base = torch.ones((1, 4), dtype=torch.float64)
        matrix = torch.tensor(
            [[
                [1.0, 0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 1.0],
            ]],
            dtype=torch.float64,
        )[:, None]
        global_result = signed_intensity_transform(base, matrix, 0.5)
        local_result = signed_intensity_transform(
            base,
            matrix,
            0.5,
            n_components=2,
            n_marks=2,
            block_local_activity=True,
            between_block_gain=0.0,
        )
        amplified = signed_intensity_transform(
            base,
            matrix,
            0.5,
            n_components=2,
            n_marks=2,
            block_local_activity=True,
            within_block_gain=3.0,
            between_block_gain=0.0,
        )
        self.assertTrue(bool(torch.all(local_result > global_result)))
        self.assertTrue(bool(torch.all(amplified > local_result)))

    def test_default_transform_is_exact_legacy_global_path(self) -> None:
        base = torch.tensor([[0.3, 0.7, 1.2, 2.0]], dtype=torch.float64)
        raw = torch.tensor(
            [[
                [1.0, 0.1, -0.1, 0.0],
                [0.1, 1.0, 0.0, 0.2],
                [-0.1, 0.0, 1.0, 0.15],
                [0.0, 0.2, 0.15, 1.0],
            ]],
            dtype=torch.float64,
        )[:, None]
        observed = signed_intensity_transform(base, raw, 0.2)
        activity = base / (1e-8 + base.sum(dim=-1, keepdim=True))
        perturbation = torch.einsum(
            "...sij,...j->...si",
            signed_correlation_coupling(raw),
            activity,
        )
        expected = torch.nn.functional.softplus(
            inverse_softplus(base)[:, None] + 0.2 * perturbation
        )
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_legacy_rows_sum_to_one(self) -> None:
        matrix = torch.tensor(
            [[[2.0, -1.0], [-1.0, 3.0]]],
            dtype=torch.float64,
        )
        attention = legacy_row_normalized_attention(matrix)
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones((1, 2), dtype=torch.float64),
        )


class SignedWishartModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backbone = ReferenceNHPOutputMixture(
            2,
            2,
            horizon=2.0,
            hidden_size=4,
            quadrature_order=3,
            initialization_seed=11,
            initial_total_rate=2.0,
        )
        self.sequences = (
            MarkedSequence(
                times=np.asarray([0.3, 1.1]),
                marks=np.asarray([0, 1], dtype=np.int64),
                horizon=2.0,
            ),
        )

    def test_model_alpha_zero_scores_equal_base(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            fixed_alpha=0.0,
        )
        trace = model.backbone.build_trace(self.sequences)
        matrices = model.sample_matrices(1, 5, sample_seed=19)
        sampled = model.component_scores_from_trace(trace, matrices)
        base = base_nhp_component_scores_from_trace(trace)
        torch.testing.assert_close(
            sampled,
            base[:, None, :].expand_as(sampled),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_model_records_block_local_transform(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            block_local_activity=True,
            within_block_gain=3.0,
            between_block_gain=0.25,
        )
        self.assertIn("block_local", model.transformation_name)
        self.assertEqual(model.within_block_gain, 3.0)
        self.assertEqual(model.between_block_gain, 0.25)

    def test_learned_alpha_initializes_at_point_one_and_has_gradient(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
        )
        torch.testing.assert_close(
            model.interaction_strength(),
            torch.tensor(0.1),
        )
        loss = -model.marginal_scores(
            self.sequences,
            n_samples=3,
            sample_seed=23,
        ).mean()
        loss.backward()
        self.assertIsNotNone(model.raw_interaction_strength.grad)
        self.assertTrue(
            bool(torch.isfinite(model.raw_interaction_strength.grad))
        )
        self.assertIsNotNone(model.raw_mean_cholesky.grad)

    def test_continuous_nu_bartlett_sampling_is_reproducible_and_differentiable(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=6.0,
            learnable_degrees_of_freedom=True,
            maximum_degrees_of_freedom=40.0,
        )
        self.assertTrue(model.learns_degrees_of_freedom)
        self.assertAlmostEqual(model.degrees_of_freedom, 6.0, places=6)
        first = model.sample_matrices(4, 8, sample_seed=71)
        second = model.sample_matrices(4, 8, sample_seed=71)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool((torch.linalg.eigvalsh(first) > 0.0).all()))
        first.square().mean().backward()
        self.assertIsNotNone(model.raw_degrees_of_freedom.grad)
        self.assertTrue(bool(torch.isfinite(
            model.raw_degrees_of_freedom.grad
        )))
        self.assertNotEqual(
            float(model.raw_degrees_of_freedom.grad), 0.0
        )

    def test_continuous_nu_samples_preserve_the_configured_mean(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=6.0,
            learnable_degrees_of_freedom=True,
            maximum_degrees_of_freedom=40.0,
        )
        samples = model.sample_matrices(1000, 1, sample_seed=73)[:, 0]
        torch.testing.assert_close(
            samples.mean(dim=0),
            model.mean_matrix(),
            rtol=0.12,
            atol=0.12,
        )

    def test_fixed_nu_rejects_non_integer_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            SignedWishartTPP(
                self.backbone,
                degrees_of_freedom=6.5,
            )

    def test_temperature_preserves_initial_alpha_and_scales_raw_gradient(self) -> None:
        gradients = []
        for temperature in (1.0, 0.5):
            model = SignedWishartTPP(
                self.backbone,
                degrees_of_freedom=4,
                initial_alpha=0.5,
                interaction_temperature=temperature,
            )
            self.assertAlmostEqual(
                float(model.interaction_strength().detach()), 0.5, places=7
            )
            self.assertAlmostEqual(
                float(model.raw_interaction_strength.detach()), 0.0, places=7
            )
            model.interaction_strength().backward()
            gradients.append(float(model.raw_interaction_strength.grad))
        self.assertAlmostEqual(gradients[1], 2.0 * gradients[0], places=7)

    def test_projected_alpha_has_unit_gradient_and_projects_to_bounds(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            initial_alpha=0.1,
            alpha_parameterization="projected",
        )
        model.interaction_strength().backward()
        self.assertAlmostEqual(
            float(model.raw_interaction_strength.grad), 1.0, places=7
        )
        with torch.no_grad():
            model.raw_interaction_strength.fill_(1.2)
        model.project_interaction_strength_()
        self.assertAlmostEqual(
            float(model.interaction_strength().detach()), 1.0
        )
        with torch.no_grad():
            model.raw_interaction_strength.fill_(-0.2)
        model.project_interaction_strength_()
        self.assertAlmostEqual(
            float(model.interaction_strength().detach()), 0.0
        )

    def test_pruning_preserves_principal_wishart_mean_block(self) -> None:
        backbone = ReferenceNHPOutputMixture(
            3,
            2,
            horizon=2.0,
            hidden_size=4,
            quadrature_order=3,
            initialization_seed=17,
            initial_total_rate=2.0,
        )
        model = SignedWishartTPP(
            backbone,
            degrees_of_freedom=9,
            initial_alpha=0.5,
            interaction_temperature=1.7,
        )
        with torch.no_grad():
            model.raw_mean_cholesky.add_(
                0.05 * torch.tril(torch.ones_like(model.raw_mean_cholesky), -1)
            )
            mean_before = model.mean_matrix().clone()
            alpha_before = model.interaction_strength().clone()
            weights_before = torch.softmax(
                model.backbone.mixture_logits, dim=0
            ).clone()
        kept = model.prune_component(1, degrees_of_freedom=6)
        self.assertEqual(kept, (0, 2))
        self.assertEqual(model.n_components, 2)
        self.assertEqual(model.dimension, 4)
        self.assertEqual(model.degrees_of_freedom, 6)
        self.assertEqual(tuple(model.raw_mean_cholesky.shape), (4, 4))
        kept_dimensions = torch.tensor([0, 1, 4, 5])
        expected = mean_before.index_select(0, kept_dimensions).index_select(
            1, kept_dimensions
        )
        expected = expected * (4.0 / torch.trace(expected))
        torch.testing.assert_close(model.mean_matrix(), expected)
        torch.testing.assert_close(model.interaction_strength(), alpha_before)
        expected_weights = weights_before[[0, 2]]
        expected_weights = expected_weights / expected_weights.sum()
        torch.testing.assert_close(
            torch.softmax(model.backbone.mixture_logits, dim=0),
            expected_weights,
        )

    def test_deterministic_ablation_uses_mean_for_every_sample(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            deterministic_matrices=True,
        )
        first = model.sample_matrices(2, 3, sample_seed=1)
        second = model.sample_matrices(2, 3, sample_seed=999)
        expected = model.mean_matrix()[None, None].expand_as(first)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(first, second)

    def test_zero_beta_is_exact_original_sampling_path(self) -> None:
        original = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
        )
        explicit_zero = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            exploration_beta=0.0,
        )
        explicit_zero.load_state_dict(original.state_dict())
        torch.testing.assert_close(
            original.sample_matrices(2, 3, sample_seed=31),
            explicit_zero.sample_matrices(2, 3, sample_seed=31),
            rtol=0.0,
            atol=0.0,
        )

    def test_additive_exploration_is_frozen_and_positive_definite(self) -> None:
        beta = 0.2
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            exploration_beta=beta,
            exploration_degrees_of_freedom=4,
            exploration_seed_offset=37,
        )
        baseline = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
        )
        baseline.load_state_dict(model.state_dict())
        learned = baseline.sample_matrices(2, 3, sample_seed=41)
        exploration = model.sample_exploration_matrices(
            2,
            3,
            sample_seed=41 + 37,
        )
        observed = model.sample_matrices(2, 3, sample_seed=41)
        torch.testing.assert_close(observed, learned + beta * exploration)
        self.assertTrue(bool((torch.linalg.eigvalsh(observed) > 0.0).all()))
        self.assertFalse(any(
            "exploration" in name for name, _ in model.named_parameters()
        ))

    def test_convex_exploration_endpoints_select_each_process(self) -> None:
        random_only = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            exploration_beta=1.0,
            exploration_mode="convex",
            exploration_seed_offset=47,
        )
        expected_random = random_only.sample_exploration_matrices(
            2,
            3,
            sample_seed=53 + 47,
        )
        torch.testing.assert_close(
            random_only.sample_matrices(2, 3, sample_seed=53),
            expected_random,
        )
        learned_only = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            exploration_beta=0.0,
            exploration_mode="convex",
        )
        baseline = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
        )
        learned_only.load_state_dict(baseline.state_dict())
        torch.testing.assert_close(
            learned_only.sample_matrices(2, 3, sample_seed=53),
            baseline.sample_matrices(2, 3, sample_seed=53),
            rtol=0.0,
            atol=0.0,
        )

    def test_global_normalization_of_additive_wishart_is_invariant(self) -> None:
        model = SignedWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
            exploration_beta=0.3,
        )
        matrices = model.sample_matrices(2, 3, sample_seed=43)
        normalized = matrices / 1.3
        torch.testing.assert_close(
            signed_correlation_coupling(matrices),
            signed_correlation_coupling(normalized),
        )
        torch.testing.assert_close(
            cluster_log_weights_from_matrices(
                matrices,
                n_components=2,
                n_marks=2,
            ),
            cluster_log_weights_from_matrices(
                normalized,
                n_components=2,
                n_marks=2,
            ),
        )

    def test_legacy_row_wrapper_is_finite(self) -> None:
        model = LegacyRowWishartTPP(
            self.backbone,
            degrees_of_freedom=4,
        )
        scores = model.marginal_scores(
            self.sequences,
            n_samples=3,
            sample_seed=29,
        )
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
