from __future__ import annotations

import unittest

import numpy as np
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
    build_nhp_mixture_batch_trace,
    LatentWishartAttentionNHP,
    sampled_nhp_component_scores,
)
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
)
from lal_wishart.models.signed_wishart import SignedWishartTPP
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    _LaLAdjacentLossDecay,
    _selection_is_better,
    _validated_selection_labels,
    balanced_assignment_kl,
    balanced_sinkhorn_assignments,
    evaluate_latent_wishart_nhp,
    fit_direct_nhp_mixture,
    fit_latent_wishart_attention_nhp,
    integrated_wishart_component_scores,
    unbalanced_ot_dual_free_energy,
    component_removal_marginal_nll_deltas,
)


class LatentWishartAttentionNHPTests(unittest.TestCase):
    def test_removal_deltas_use_full_marginal_and_renormalized_weights(
        self,
    ) -> None:
        scores = torch.tensor([
            [2.0, -3.0, 1.9],
            [-4.0, 2.5, -3.5],
            [1.5, -2.0, 1.4],
        ])
        logits = torch.tensor([0.3, 0.1, -2.0])
        full_nll, deltas = component_removal_marginal_nll_deltas(
            scores, logits
        )
        log_weights = torch.log_softmax(logits, dim=0)
        expected_full = -torch.logsumexp(
            scores + log_weights[None, :], dim=1
        ).mean()
        torch.testing.assert_close(full_nll, expected_full)
        for removed in range(3):
            kept = [index for index in range(3) if index != removed]
            kept_log_weights = log_weights[kept]
            kept_log_weights -= torch.logsumexp(
                kept_log_weights, dim=0
            )
            expected_removed = -torch.logsumexp(
                scores[:, kept] + kept_log_weights[None, :], dim=1
            ).mean()
            torch.testing.assert_close(
                deltas[removed], expected_removed - expected_full
            )
        self.assertEqual(int(torch.argmin(deltas)), 2)
    def test_constrained_free_energy_adds_not_subtracts_kl(self) -> None:
        scores = torch.tensor(
            [[1.2, -0.3, 0.7], [-1.0, 0.4, 0.2]],
            dtype=torch.float64,
        )
        q = torch.tensor(
            [[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]],
            dtype=torch.float64,
        )
        posterior = torch.softmax(scores, dim=1)
        mixture_nll = -torch.logsumexp(scores, dim=1).mean()
        kl = (
            q * (torch.log(q) - torch.log(posterior))
        ).sum(dim=1).mean()
        free_energy = (
            -q * scores + q * torch.log(q)
        ).sum(dim=1).mean()
        torch.testing.assert_close(free_energy, mixture_nll + kl)

    def test_unbalanced_ot_dual_has_envelope_gradients(self) -> None:
        generator = torch.Generator().manual_seed(23)
        scores = torch.randn(
            11, 4, generator=generator, dtype=torch.float64,
            requires_grad=True,
        )
        dual = torch.randn(
            4, generator=generator, dtype=torch.float64,
            requires_grad=True,
        )
        value, assignments, reference = unbalanced_ot_dual_free_energy(
            scores,
            dual,
            temperature=1.0,
            marginal_penalty=2.5,
        )
        score_gradient, dual_gradient = torch.autograd.grad(
            value, (scores, dual)
        )
        torch.testing.assert_close(
            score_gradient,
            -assignments / scores.shape[0],
        )
        torch.testing.assert_close(
            dual_gradient,
            assignments.mean(dim=0) - reference,
        )

    def test_unbalanced_ot_dual_matches_primal_at_optimum(self) -> None:
        generator = torch.Generator().manual_seed(29)
        scores = torch.randn(31, 3, generator=generator)
        dual = torch.nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.Adam((dual,), lr=0.05, maximize=True)
        for _ in range(1000):
            optimizer.zero_grad(set_to_none=True)
            value, assignments, reference = unbalanced_ot_dual_free_energy(
                scores,
                dual,
                temperature=1.0,
                marginal_penalty=1.7,
            )
            value.backward()
            optimizer.step()
            with torch.no_grad():
                dual.sub_(dual.mean())
        value, assignments, reference = unbalanced_ot_dual_free_energy(
            scores,
            dual,
            temperature=1.0,
            marginal_penalty=1.7,
        )
        marginal = assignments.mean(dim=0)
        torch.testing.assert_close(marginal, reference, atol=2e-4, rtol=2e-4)
        uniform = torch.full_like(marginal, 1.0 / marginal.numel())
        primal = (
            (-assignments * scores).sum(dim=1).mean()
            + (assignments * torch.log(assignments)).sum(dim=1).mean()
            + 1.7
            * (marginal * (torch.log(marginal) - torch.log(uniform))).sum()
        )
        torch.testing.assert_close(value, primal, atol=2e-4, rtol=2e-4)

    def test_wishart_cluster_scores_integrate_before_ot(self) -> None:
        generator = torch.Generator().manual_seed(31)
        components = torch.randn(7, 5, 3, generator=generator)
        gates = torch.log_softmax(
            torch.randn(7, 5, 3, generator=generator), dim=2
        )
        integrated = integrated_wishart_component_scores(
            components, gates
        )
        expected = torch.logsumexp(
            components + gates, dim=(1, 2)
        ) - np.log(components.shape[1])
        torch.testing.assert_close(
            torch.logsumexp(integrated, dim=1), expected
        )

    def test_sinkhorn_assignments_have_uniform_cluster_marginals(self) -> None:
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(8, 4, generator=generator)
        assignments = balanced_sinkhorn_assignments(
            torch.log_softmax(logits, dim=1),
            temperature=0.1,
            iterations=200,
        )
        torch.testing.assert_close(
            assignments.sum(dim=1), torch.ones(8), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            assignments.sum(dim=0), torch.full((4,), 2.0),
            atol=1e-4, rtol=1e-4,
        )
        self.assertFalse(assignments.requires_grad)

    def test_sinkhorn_kl_penalizes_collapsed_assignments(self) -> None:
        probabilities = torch.tensor(
            [[0.97, 0.01, 0.01, 0.01]] * 8,
            requires_grad=True,
        )
        divergence, target = balanced_assignment_kl(
            probabilities.log(),
            temperature=0.1,
            iterations=200,
        )
        self.assertGreater(float(divergence.detach()), 1.0)
        torch.testing.assert_close(
            target.sum(dim=0), torch.full((4,), 2.0),
            atol=1e-5, rtol=1e-5,
        )
        divergence.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertGreater(float(probabilities.grad.abs().sum()), 0.0)

    def test_lal_lr_decay_matches_published_callback(self) -> None:
        parameters = [
            torch.nn.Parameter(torch.tensor(0.0)),
            torch.nn.Parameter(torch.tensor(0.0)),
            torch.nn.Parameter(torch.tensor(0.0)),
        ]
        optimizer = torch.optim.Adam([
            {"params": [parameters[0]], "lr": 0.001},
            {"params": [parameters[1]], "lr": 0.01},
            {"params": [parameters[2]], "lr": 0.001},
        ])
        decay = _LaLAdjacentLossDecay(
            optimizer,
            factor=0.5,
            tolerance=2,
            min_lr=0.001,
            updated_lr=0.001,
        )
        self.assertFalse(decay.step(3.0))
        self.assertFalse(decay.step(2.0))
        self.assertFalse(decay.step(2.5))
        self.assertFalse(decay.step(2.4))
        self.assertTrue(decay.step(2.6))
        self.assertEqual(decay.checker, 0)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.001, 0.005, 0.001],
        )

    def backbone(self) -> ReferenceNeuralHawkesMixture:
        return ReferenceNeuralHawkesMixture(
            2,
            2,
            horizon=1.0,
            hidden_size=5,
            quadrature_order=4,
            initial_total_rate=2.0,
            initialization_seed=121,
            dtype=torch.float64,
        )

    def model(self) -> LatentWishartAttentionNHP:
        return LatentWishartAttentionNHP(
            self.backbone(),
            degrees_of_freedom=6,
        )

    def sequences(self) -> tuple[MarkedSequence, ...]:
        return (
            MarkedSequence(
                times=np.array([0.1, 0.4, 0.8]),
                marks=np.array([0, 1, 0], dtype=np.int64),
                horizon=1.0,
            ),
            MarkedSequence(
                times=np.array([0.2, 0.55, 0.9]),
                marks=np.array([1, 0, 1], dtype=np.int64),
                horizon=1.0,
            ),
        )

    def test_identity_attention_equals_base_component_scores(self) -> None:
        backbone = self.backbone()
        sequences = self.sequences()
        trace = build_nhp_mixture_batch_trace(backbone, sequences)
        identity = torch.eye(4, dtype=torch.float64)[None, None].repeat(
            len(sequences),
            1,
            1,
            1,
        )
        observed = sampled_nhp_component_scores(trace, identity)[:, 0]
        expected = backbone.component_scores_for_indices(
            sequences,
            (0, 1),
        )
        traced = base_nhp_component_scores_from_trace(trace)
        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close(traced, expected)

    def test_prefix_and_suffix_add_to_full(self) -> None:
        backbone = self.backbone()
        trace = build_nhp_mixture_batch_trace(
            backbone,
            self.sequences(),
            boundary=0.5,
        )
        matrices = torch.eye(4, dtype=torch.float64)[None, None].repeat(
            2,
            3,
            1,
            1,
        )
        prefix = sampled_nhp_component_scores(
            trace,
            matrices,
            end_time=0.5,
        )
        suffix = sampled_nhp_component_scores(
            trace,
            matrices,
            start_time=0.5,
        )
        full = sampled_nhp_component_scores(trace, matrices)
        torch.testing.assert_close(prefix + suffix, full)

    def test_gradients_reach_distribution_and_nhp(self) -> None:
        model = self.model()
        score = model.marginal_scores(
            self.sequences(),
            n_samples=3,
            sample_seed=132,
        )
        (-score.mean()).backward()
        self.assertIsNotNone(model.raw_mean_cholesky.grad)
        self.assertGreater(
            float(model.raw_mean_cholesky.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.backbone.intensity_linear.weight.grad)
        self.assertGreater(
            float(model.backbone.intensity_linear.weight.grad.abs().sum()),
            0.0,
        )

    def test_prefix_posterior_does_not_read_suffix(self) -> None:
        first = MarkedSequence(
            times=np.array([0.1, 0.4, 0.8]),
            marks=np.array([0, 1, 0], dtype=np.int64),
            horizon=1.0,
        )
        second = MarkedSequence(
            times=np.array([0.1, 0.4, 0.6, 0.7, 0.9]),
            marks=np.array([0, 1, 1, 1, 0], dtype=np.int64),
            horizon=1.0,
        )
        model = self.model()
        left = evaluate_latent_wishart_nhp(
            model,
            (first,),
            cutoff=0.5,
            n_samples=5,
            sample_seed=141,
            batch_size=1,
        )
        right = evaluate_latent_wishart_nhp(
            model,
            (second,),
            cutoff=0.5,
            n_samples=5,
            sample_seed=141,
            batch_size=1,
        )
        torch.testing.assert_close(
            left.prefix_cluster_probabilities,
            right.prefix_cluster_probabilities,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            left.prefix_posterior_attention_diagonal,
            right.prefix_posterior_attention_diagonal,
            rtol=0.0,
            atol=0.0,
        )

    def test_one_epoch_has_no_local_w_or_lal_logic(self) -> None:
        model = self.model()
        result = fit_latent_wishart_attention_nhp(
            model,
            self.sequences(),
            self.sequences(),
            max_epochs=1,
            batch_size=2,
            train_samples=2,
            validation_samples=3,
            evaluation_interval=1,
            validation_cutoff=0.5,
            evaluation_batch_size=2,
            distribution_learning_rate=0.01,
            sample_seed=151,
        )
        names = tuple(name for name, _ in result.model.named_parameters())
        self.assertEqual(
            sum(name == "raw_mean_cholesky" for name in names),
            1,
        )
        self.assertFalse(any("trajectory" in name for name in names))
        self.assertFalse(result.history[0]["uses_local_w_parameters"])
        self.assertFalse(result.history[0]["uses_elbo"])
        self.assertFalse(result.history[0]["uses_lal_logic"])

    def test_wishart_gradient_accumulation_reports_effective_batch(self) -> None:
        result = fit_latent_wishart_attention_nhp(
            self.model(),
            self.sequences(),
            self.sequences(),
            max_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=2,
            train_samples=1,
            validation_samples=1,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            distribution_learning_rate=0.01,
            sample_seed=152,
        )
        self.assertEqual(result.history[0]["gradient_accumulation_steps"], 2)
        self.assertEqual(result.history[0]["effective_batch_size"], 2)

    def test_direct_k1_pretraining_can_freeze_uniform_mixture_prior(self) -> None:
        model = ReferenceNeuralHawkesMixture(
            1,
            2,
            horizon=1.0,
            hidden_size=5,
            quadrature_order=4,
            initial_total_rate=2.0,
            initialization_seed=122,
            dtype=torch.float64,
        )
        result = fit_direct_nhp_mixture(
            model,
            self.sequences(),
            self.sequences(),
            max_epochs=1,
            batch_size=2,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            optimize_mixture_logits=False,
        )
        self.assertEqual(result.model.n_components, 1)
        self.assertFalse(result.model.mixture_logits.requires_grad)
        self.assertFalse(result.history[0]["mixture_logits_optimized"])

    def test_assignment_information_regularizer_is_finite(self) -> None:
        result = fit_latent_wishart_attention_nhp(
            self.model(),
            self.sequences(),
            self.sequences(),
            max_epochs=1,
            batch_size=2,
            train_samples=2,
            validation_samples=2,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            distribution_learning_rate=0.01,
            sample_seed=153,
            assignment_information_strength=1.0,
        )
        row = result.history[0]
        self.assertTrue(np.isfinite(row["assignment_mutual_information"]))
        self.assertGreaterEqual(row["assignment_mutual_information"], 0.0)
        self.assertEqual(row["assignment_information_strength"], 1.0)

    def test_purity_selection_uses_nll_only_as_a_tie_breaker(self) -> None:
        self.assertTrue(_selection_is_better(
            "validation_purity",
            validation_nll=2.0,
            validation_purity=0.8,
            best_validation_nll=1.0,
            best_validation_purity=0.7,
        ))
        self.assertTrue(_selection_is_better(
            "validation_purity",
            validation_nll=0.9,
            validation_purity=0.8,
            best_validation_nll=1.0,
            best_validation_purity=0.8,
        ))
        self.assertFalse(_selection_is_better(
            "validation_purity",
            validation_nll=0.5,
            validation_purity=0.7,
            best_validation_nll=1.0,
            best_validation_purity=0.8,
        ))

    def test_purity_selection_requires_aligned_validation_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            _validated_selection_labels(
                None,
                validation_size=2,
                selection_metric="validation_purity",
            )
        with self.assertRaisesRegex(ValueError, "align"):
            _validated_selection_labels(
                np.array([0]),
                validation_size=2,
                selection_metric="validation_purity",
            )

    def test_exploration_anneals_to_zero_and_validation_uses_zero(self) -> None:
        model = SignedWishartTPP(
            self.backbone(),
            degrees_of_freedom=4,
            exploration_beta=0.9,
            exploration_mode="convex",
            exploration_degrees_of_freedom=8,
        )
        result = fit_latent_wishart_attention_nhp(
            model,
            self.sequences(),
            self.sequences(),
            max_epochs=3,
            batch_size=2,
            train_samples=1,
            validation_samples=1,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            validation_labels=np.array([0, 1]),
            selection_metric="validation_purity",
            exploration_beta_anneal_epochs=3,
            validation_exploration_beta=0.0,
        )
        observed = [
            row["training_exploration_beta"] for row in result.history
        ]
        np.testing.assert_allclose(observed, [0.9, 0.45, 0.0])
        self.assertTrue(all(
            row["validation_exploration_beta"] == 0.0
            for row in result.history
        ))
        self.assertEqual(result.model.exploration_beta, 0.0)

    def test_exponential_exploration_anneals_early_and_reaches_zero(self) -> None:
        model = SignedWishartTPP(
            self.backbone(),
            degrees_of_freedom=4,
            exploration_beta=0.9,
            exploration_mode="convex",
            exploration_degrees_of_freedom=8,
        )
        result = fit_latent_wishart_attention_nhp(
            model,
            self.sequences(),
            self.sequences(),
            max_epochs=3,
            batch_size=2,
            train_samples=1,
            validation_samples=1,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            validation_labels=np.array([0, 1]),
            selection_metric="validation_purity",
            exploration_beta_anneal_epochs=3,
            exploration_beta_schedule="exponential",
            exploration_beta_decay_rate=5.0,
            validation_exploration_beta=0.0,
        )
        observed = [
            row["training_exploration_beta"] for row in result.history
        ]
        endpoint = np.exp(-5.0)
        midpoint = 0.9 * (np.exp(-2.5) - endpoint) / (1.0 - endpoint)
        np.testing.assert_allclose(observed, [0.9, midpoint, 0.0])
        self.assertLess(observed[1], 0.1)
        self.assertEqual(result.model.exploration_beta, 0.0)

    def test_learnable_nu_has_its_own_optimizer_group_and_history(self) -> None:
        model = SignedWishartTPP(
            self.backbone(),
            degrees_of_freedom=6.0,
            learnable_degrees_of_freedom=True,
            maximum_degrees_of_freedom=40.0,
        )
        result = fit_latent_wishart_attention_nhp(
            model,
            self.sequences(),
            self.sequences(),
            max_epochs=2,
            batch_size=2,
            train_samples=2,
            validation_samples=2,
            evaluation_interval=1,
            validation_cutoff=None,
            evaluation_batch_size=2,
            validation_labels=np.array([0, 1]),
            selection_metric="validation_purity",
            degrees_of_freedom_learning_rate=1e-3,
            degrees_of_freedom_prior_strength=0.01,
            degrees_of_freedom_prior_center=6.0,
        )
        self.assertTrue(all(
            row["learnable_degrees_of_freedom"]
            for row in result.history
        ))
        self.assertTrue(all(np.isfinite(
            row["degrees_of_freedom_gradient"]
        ) for row in result.history))
        self.assertTrue(all(
            row["degrees_of_freedom_learning_rate"] == 1e-3
            for row in result.history
        ))
        self.assertTrue(any(
            abs(row["degrees_of_freedom"] - 6.0) > 1e-6
            for row in result.history
        ))


if __name__ == "__main__":
    unittest.main()
