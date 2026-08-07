from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.neural_hawkes import (
    ContinuousTimeLSTMCell,
    NeuralHawkesMixture,
    inverse_softplus,
)


def sequences() -> tuple[MarkedSequence, ...]:
    return (
        MarkedSequence(
            times=np.array([0.1, 0.4, 0.85]),
            marks=np.array([0, 1, 0], dtype=np.int64),
            horizon=1.0,
        ),
        MarkedSequence(
            times=np.array([0.2, 0.21, 0.7]),
            marks=np.array([1, 1, 0], dtype=np.int64),
            horizon=1.0,
        ),
        MarkedSequence(
            times=np.array([], dtype=float),
            marks=np.array([], dtype=np.int64),
            horizon=1.0,
        ),
    )


class NeuralHawkesTests(unittest.TestCase):
    def test_batched_scores_match_individual_traces(self) -> None:
        model = NeuralHawkesMixture(
            2,
            2,
            horizon=1.0,
            hidden_size=6,
            quadrature_order=8,
            initialization_seed=43,
        )
        sample = sequences()
        batched = model.component_scores(sample)
        individual = torch.stack(
            [
                torch.stack(
                    [
                        model.component_trace(sequence, component).log_likelihood
                        for component in range(model.n_components)
                    ]
                )
                for sequence in sample
            ]
        )
        torch.testing.assert_close(batched, individual, rtol=1e-11, atol=1e-11)

    def model(self, components: int = 2, order: int = 16) -> NeuralHawkesMixture:
        return NeuralHawkesMixture(
            components,
            2,
            horizon=1.0,
            hidden_size=6,
            quadrature_order=order,
            initialization_seed=7,
        )

    def test_decay_matches_equation_seven(self) -> None:
        cell = torch.tensor([1.0, -0.5], dtype=torch.float64)
        cell_bar = torch.tensor([0.2, 0.1], dtype=torch.float64)
        decay = torch.tensor([0.7, 1.3], dtype=torch.float64)
        output = torch.tensor([0.8, 0.4], dtype=torch.float64)
        elapsed = torch.tensor(0.6, dtype=torch.float64)
        actual_cell, actual_hidden = ContinuousTimeLSTMCell.decay(
            cell, cell_bar, decay, output, elapsed
        )
        expected_cell = cell_bar + (cell - cell_bar) * torch.exp(
            -decay * elapsed
        )
        torch.testing.assert_close(actual_cell, expected_cell)
        torch.testing.assert_close(
            actual_hidden,
            output * torch.tanh(expected_cell),
        )

    def test_classical_homogeneous_tpp_likelihood(self) -> None:
        model = self.model(components=1)
        rates = torch.tensor([0.4, 0.7], dtype=torch.float64)
        with torch.no_grad():
            model.intensity_linear.weight.zero_()
            model.intensity_link.raw_scale.fill_(
                inverse_softplus(1.0 - model.intensity_link.minimum_scale)
            )
            model.intensity_linear.bias.copy_(
                torch.log(torch.expm1(rates))
            )
        score = model.component_scores(sequences()[:1])[0, 0]
        expected = (
            2.0 * math.log(0.4)
            + math.log(0.7)
            - float(rates.sum())
        )
        self.assertAlmostEqual(float(score.detach()), expected, places=9)

    def test_variable_horizons_preserve_last_event_and_compensator_endpoint(self) -> None:
        model = NeuralHawkesMixture(
            1,
            2,
            horizon=2.0,
            hidden_size=6,
            quadrature_order=8,
            initialization_seed=47,
        )
        rates = torch.tensor([0.4, 0.7], dtype=torch.float64)
        with torch.no_grad():
            model.intensity_linear.weight.zero_()
            model.intensity_link.raw_scale.fill_(
                inverse_softplus(1.0 - model.intensity_link.minimum_scale)
            )
            model.intensity_linear.bias.copy_(torch.log(torch.expm1(rates)))
        sample = (
            MarkedSequence(
                times=np.array([0.2, 0.7]),
                marks=np.array([0, 0], dtype=np.int64),
                horizon=0.7,
            ),
            MarkedSequence(
                times=np.array([0.3]),
                marks=np.array([1], dtype=np.int64),
                horizon=1.5,
            ),
        )
        observed = model.component_scores(sample)[:, 0]
        expected = torch.tensor(
            [
                2.0 * math.log(0.4) - 0.7 * 1.1,
                math.log(0.7) - 1.5 * 1.1,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(observed, expected, rtol=1e-9, atol=1e-9)

    def test_empty_sequence_is_negative_compensator(self) -> None:
        model = self.model(components=1)
        trace = model.component_trace(sequences()[-1], 0)
        self.assertEqual(trace.event_marks.numel(), 0)
        self.assertLess(float(trace.log_likelihood.detach()), 0.0)

    def test_duplicated_components_equal_k1(self) -> None:
        single = self.model(components=1)
        mixture = self.model(components=3)
        mixture.event_embedding.load_state_dict(
            copy.deepcopy(single.event_embedding.state_dict())
        )
        mixture.recurrent.load_state_dict(
            copy.deepcopy(single.recurrent.state_dict())
        )
        mixture.intensity_linear.load_state_dict(
            copy.deepcopy(single.intensity_linear.state_dict())
        )
        mixture.intensity_link.load_state_dict(
            copy.deepcopy(single.intensity_link.state_dict())
        )
        with torch.no_grad():
            for name in (
                "initial_cell",
                "initial_cell_bar",
                "initial_raw_decay",
                "initial_output_logits",
            ):
                getattr(mixture, name).copy_(
                    getattr(single, name).repeat(3, 1)
                )
            mixture.mixture_logits.copy_(
                torch.tensor([0.4, -0.9, 1.1], dtype=torch.float64)
            )
        torch.testing.assert_close(
            mixture.log_likelihoods(sequences()),
            single.log_likelihoods(sequences()),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_responsibilities_and_gradients_are_finite(self) -> None:
        model = self.model(components=3)
        responsibilities = model.responsibilities(sequences())
        torch.testing.assert_close(
            responsibilities.sum(dim=1),
            torch.ones(3, dtype=torch.float64),
            rtol=0.0,
            atol=1e-13,
        )
        loss = model.negative_log_likelihood(sequences())
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_quadrature_converges(self) -> None:
        low = self.model(components=1, order=10)
        high = self.model(components=1, order=32)
        high.load_state_dict(
            {
                name: value
                for name, value in low.state_dict().items()
                if not name.startswith("_quadrature_")
            },
            strict=False,
        )
        low_score = low.component_scores(sequences()[:1])[0, 0]
        high_score = high.component_scores(sequences()[:1])[0, 0]
        self.assertAlmostEqual(
            float(low_score.detach()),
            float(high_score.detach()),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
