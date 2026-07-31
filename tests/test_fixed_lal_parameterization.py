from __future__ import annotations

import unittest

import numpy as np
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.reference_bos_lal import (
    ReferenceCOTICBOSLaL,
    ReferenceTHPBOSLaL,
    split_reference_bos_lal_component,
)
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
    split_reference_nhp_k1,
)


class FixedLaLParameterizationTests(unittest.TestCase):
    def sequence(self) -> tuple[MarkedSequence, ...]:
        return (
            MarkedSequence(
                times=np.array([0.2, 0.7]),
                marks=np.array([0, 1], dtype=np.int64),
                horizon=1.0,
            ),
        )

    def test_nhp_repeated_split_keeps_one_shared_network(self) -> None:
        model = ReferenceNeuralHawkesMixture(
            1,
            2,
            horizon=1.0,
            hidden_size=6,
            quadrature_order=4,
            initialization_seed=51,
        )
        split = split_reference_nhp_k1(
            model,
            3,
            initialization_seed=52,
        )
        self.assertEqual(split.n_components, 3)
        self.assertEqual(tuple(split.initial_cell.shape), (3, 6))
        self.assertFalse(
            any(name.startswith("components.") for name, _ in split.named_modules())
        )
        self.assertTrue(bool(torch.isfinite(split.component_scores(self.sequence())).all()))

    def test_thp_and_cotic_split_only_cluster_bos(self) -> None:
        models = (
            ReferenceTHPBOSLaL(
                1,
                2,
                horizon=1.0,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=61,
            ),
            ReferenceCOTICBOSLaL(
                1,
                2,
                horizon=1.0,
                input_channels=4,
                hidden_size=8,
                num_layers=2,
                kernel_size=3,
                dropout=0.0,
                quadrature_order=4,
                initialization_seed=62,
            ),
        )
        for original in models:
            split = split_reference_bos_lal_component(
                original,
                0,
                initialization_seed=63,
                beta=0.35,
            )
            split = split_reference_bos_lal_component(
                split,
                1,
                initialization_seed=64,
                beta=0.40,
            )
            self.assertEqual(split.n_components, 3)
            self.assertEqual(split.bos_embeddings.shape[0], 3)
            names = tuple(name for name, _ in split.named_modules())
            self.assertFalse(any(name.startswith("components.") for name in names))
            self.assertFalse(any(name.startswith("encoders.") for name in names))
            self.assertTrue(
                bool(torch.isfinite(split.component_scores(self.sequence())).all())
            )


if __name__ == "__main__":
    unittest.main()
