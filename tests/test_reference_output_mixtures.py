from __future__ import annotations

import unittest

import numpy as np
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.latent_wishart_attention_history import (
    LatentWishartAttentionHistory,
)
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
)
from lal_wishart.models.reference_history_mixtures import (
    ReferenceCOTICMixture,
    ReferenceEasyTPPTHPMixture,
)
from lal_wishart.models.reference_output_mixtures import (
    ReferenceCOTICOutputMixture,
    ReferenceNHPOutputMixture,
    ReferenceTHPOutputMixture,
)


class ReferenceOutputMixtureTests(unittest.TestCase):
    def sequences(self) -> tuple[MarkedSequence, ...]:
        return (
            MarkedSequence(
                times=np.array([0.2, 0.7]),
                marks=np.array([0, 1], dtype=np.int64),
                horizon=1.0,
            ),
            MarkedSequence(
                times=np.array([0.1, 0.4, 0.9]),
                marks=np.array([1, 0, 1], dtype=np.int64),
                horizon=1.0,
            ),
        )

    def output_models(self):
        return (
            ReferenceNHPOutputMixture(
                3,
                2,
                horizon=1.0,
                hidden_size=6,
                quadrature_order=4,
                initialization_seed=401,
            ),
            ReferenceTHPOutputMixture(
                3,
                2,
                horizon=1.0,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=402,
            ),
            ReferenceCOTICOutputMixture(
                3,
                2,
                horizon=1.0,
                input_channels=4,
                hidden_size=8,
                num_layers=2,
                kernel_size=3,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=403,
            ),
        )

    def test_only_the_final_head_has_k_times_c_outputs(self) -> None:
        nhp, thp, cotic = self.output_models()
        self.assertEqual(nhp.intensity_linear.out_features, 6)
        self.assertEqual(thp.layer_intensity_hidden.out_features, 6)
        self.assertEqual(cotic.intensity_head.layer.out_features, 6)
        for model in (nhp, thp, cotic):
            names = tuple(name for name, _ in model.named_modules())
            self.assertFalse(any(name.startswith("components.") for name in names))
            self.assertFalse(any(name.startswith("encoders.") for name in names))

    def test_encoder_state_is_identical_across_components(self) -> None:
        for model in self.output_models():
            times, types, valid = model._prepare_histories(
                self.sequences()
            )
            states = model.encode_histories(times, types, valid)
            torch.testing.assert_close(states[0], states[1])
            torch.testing.assert_close(states[1], states[2])

    def test_component_scores_are_finite_and_head_slices_differ(self) -> None:
        for model in self.output_models():
            scores = model.component_scores(self.sequences())
            self.assertEqual(tuple(scores.shape), (2, 3))
            self.assertTrue(bool(torch.isfinite(scores).all()))
            self.assertGreater(
                float(
                    torch.max(torch.abs(scores[:, 0] - scores[:, 1]))
                    .detach()
                    .cpu()
                ),
                1e-7,
            )

    def test_prefix_and_suffix_add_to_full(self) -> None:
        for model in self.output_models():
            trace = model.build_trace(self.sequences(), boundary=0.5)
            prefix = base_nhp_component_scores_from_trace(
                trace,
                end_time=0.5,
            )
            suffix = base_nhp_component_scores_from_trace(
                trace,
                start_time=0.5,
            )
            full = base_nhp_component_scores_from_trace(trace)
            torch.testing.assert_close(prefix + suffix, full)

    def test_thp_and_cotic_use_each_paths_own_horizon(self) -> None:
        sample = (
            MarkedSequence(
                times=np.array([0.2, 0.7]),
                marks=np.array([0, 1], dtype=np.int64),
                horizon=0.7,
            ),
            MarkedSequence(
                times=np.array([0.1, 0.4, 1.4]),
                marks=np.array([1, 0, 1], dtype=np.int64),
                horizon=1.4,
            ),
        )
        models = (
            ReferenceTHPOutputMixture(
                3,
                2,
                horizon=1.4,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=421,
            ),
            ReferenceCOTICOutputMixture(
                3,
                2,
                horizon=1.4,
                input_channels=4,
                hidden_size=8,
                num_layers=2,
                kernel_size=3,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=422,
            ),
        )
        for model in models:
            batched = model.component_scores(sample)
            individual = torch.cat(
                tuple(model.component_scores((sequence,)) for sequence in sample),
                dim=0,
            )
            torch.testing.assert_close(batched, individual)
            trace = model.build_trace(sample)
            self.assertEqual(trace.event_times.numel(), 5)
            torch.testing.assert_close(
                trace.path_horizons,
                torch.tensor([0.7, 1.4], dtype=trace.path_horizons.dtype),
            )

    def test_thp_pruning_preserves_surviving_scores_and_weights(self) -> None:
        model = self.output_models()[1]
        model.eval()
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([0.7, -0.2, 1.1]))
            scores_before = model.component_scores(self.sequences())
            weights_before = torch.softmax(model.mixture_logits, dim=0)
        kept = model.prune_component(1)
        self.assertEqual(kept, (0, 2))
        self.assertEqual(model.n_components, 2)
        self.assertEqual(model.layer_intensity_hidden.out_features, 4)
        self.assertEqual(tuple(model.factor_intensity_base.shape), (1, 4))
        self.assertEqual(tuple(model.softplus.log_beta.shape), (4,))
        with torch.no_grad():
            scores_after = model.component_scores(self.sequences())
            weights_after = torch.softmax(model.mixture_logits, dim=0)
        torch.testing.assert_close(scores_after, scores_before[:, [0, 2]])
        expected_weights = weights_before[[0, 2]]
        expected_weights = expected_weights / expected_weights.sum()
        torch.testing.assert_close(weights_after, expected_weights)

    def test_nhp_pruning_preserves_surviving_scores_and_weights(self) -> None:
        model = self.output_models()[0]
        model.eval()
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([0.7, -0.2, 1.1]))
            scores_before = model.component_scores(self.sequences())
            weights_before = torch.softmax(model.mixture_logits, dim=0)
        kept = model.prune_component(1)
        self.assertEqual(kept, (0, 2))
        self.assertEqual(model.n_components, 2)
        self.assertEqual(model.intensity_linear.out_features, 4)
        self.assertEqual(tuple(model.intensity_link.raw_scale.shape), (4,))
        with torch.no_grad():
            scores_after = model.component_scores(self.sequences())
            weights_after = torch.softmax(model.mixture_logits, dim=0)
        torch.testing.assert_close(scores_after, scores_before[:, [0, 2]])
        expected_weights = weights_before[[0, 2]]
        expected_weights = expected_weights / expected_weights.sum()
        torch.testing.assert_close(weights_after, expected_weights)

    def test_cotic_pruning_preserves_surviving_scores_and_weights(self) -> None:
        model = self.output_models()[2]
        model.eval()
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([0.7, -0.2, 1.1]))
            scores_before = model.component_scores(self.sequences())
            weights_before = torch.softmax(model.mixture_logits, dim=0)
        kept = model.prune_component(1)
        self.assertEqual(kept, (0, 2))
        self.assertEqual(model.n_components, 2)
        self.assertEqual(model.intensity_head.layer.out_features, 4)
        self.assertEqual(
            tuple(model.intensity_head.softplus_params.shape), (1, 1, 4)
        )
        with torch.no_grad():
            scores_after = model.component_scores(self.sequences())
            weights_after = torch.softmax(model.mixture_logits, dim=0)
        torch.testing.assert_close(scores_after, scores_before[:, [0, 2]])
        expected_weights = weights_before[[0, 2]]
        expected_weights = expected_weights / expected_weights.sum()
        torch.testing.assert_close(weights_after, expected_weights)

    def test_cotic_k1_expansion_preserves_scale_and_encoder(self) -> None:
        model = ReferenceCOTICOutputMixture(
            1,
            2,
            horizon=1.0,
            input_channels=4,
            hidden_size=8,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=423,
        )
        model.eval()
        encoder_before = {
            name: parameter.detach().clone()
            for name, parameter in model.encoder.named_parameters()
        }
        scale_before = model.intensity_head.softplus_params.detach().clone()
        score_before = model.component_scores(self.sequences())[:, 0]
        model.expand_components(
            4, noise_scale=0.0, initialization_seed=424
        )
        self.assertEqual(model.n_components, 4)
        self.assertEqual(model.intensity_head.layer.out_features, 8)
        self.assertEqual(model.intensity_head.num_types, 8)
        torch.testing.assert_close(
            model.intensity_head.softplus_params,
            scale_before.repeat(1, 1, 4),
        )
        for name, parameter in model.encoder.named_parameters():
            torch.testing.assert_close(parameter, encoder_before[name])
        scores_after = model.component_scores(self.sequences())
        torch.testing.assert_close(
            scores_after,
            score_before[:, None].expand(-1, 4),
        )

    def test_thp_k1_expansion_preserves_scale_and_encoder(self) -> None:
        model = ReferenceTHPOutputMixture(
            1,
            2,
            horizon=1.0,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=425,
        )
        model.eval()
        encoder_before = {
            name: parameter.detach().clone()
            for name, parameter in model.backbone.named_parameters()
        }
        scale_before = model.softplus.log_beta.detach().clone()
        score_before = model.component_scores(self.sequences())[:, 0]
        model.expand_components(
            4, noise_scale=0.0, initialization_seed=426
        )
        self.assertEqual(model.n_components, 4)
        self.assertEqual(model.layer_intensity_hidden.out_features, 8)
        torch.testing.assert_close(
            model.softplus.log_beta, scale_before.repeat(4)
        )
        for name, parameter in model.backbone.named_parameters():
            torch.testing.assert_close(parameter, encoder_before[name])
        scores_after = model.component_scores(self.sequences())
        torch.testing.assert_close(
            scores_after,
            score_before[:, None].expand(-1, 4),
        )

    def test_k1_thp_and_cotic_match_reference_architectures(self) -> None:
        output_thp = ReferenceTHPOutputMixture(
            1,
            2,
            horizon=1.0,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=411,
        )
        reference_thp = ReferenceEasyTPPTHPMixture(
            1,
            2,
            horizon=1.0,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=411,
        )
        output_cotic = ReferenceCOTICOutputMixture(
            1,
            2,
            horizon=1.0,
            input_channels=4,
            hidden_size=8,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=412,
        )
        reference_cotic = ReferenceCOTICMixture(
            1,
            2,
            horizon=1.0,
            input_channels=4,
            hidden_size=8,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            quadrature_order=4,
            initialization_seed=412,
        )
        for output, reference in (
            (output_thp, reference_thp),
            (output_cotic, reference_cotic),
        ):
            torch.testing.assert_close(
                output.component_scores(self.sequences()),
                reference.component_scores(self.sequences()),
                rtol=0.0,
                atol=1e-6,
            )

    def test_latent_wishart_wraps_output_mixture_without_extra_encoders(self) -> None:
        backbone = self.output_models()[1]
        model = LatentWishartAttentionHistory(
            backbone,
            degrees_of_freedom=6,
        )
        scores, gates, matrices = model.sampled_component_scores(
            self.sequences(),
            n_samples=2,
            sample_seed=421,
        )
        self.assertEqual(tuple(scores.shape), (2, 2, 3))
        self.assertEqual(tuple(gates.shape), (2, 2, 3))
        self.assertEqual(tuple(matrices.shape), (2, 2, 6, 6))
        encoder_count = sum(
            1
            for name, _ in model.named_modules()
            if name == "backbone.backbone"
        )
        self.assertEqual(encoder_count, 1)


if __name__ == "__main__":
    unittest.main()
