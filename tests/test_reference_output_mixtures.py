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
