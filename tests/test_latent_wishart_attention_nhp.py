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
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_latent_wishart_nhp,
    fit_latent_wishart_attention_nhp,
)


class LatentWishartAttentionNHPTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
