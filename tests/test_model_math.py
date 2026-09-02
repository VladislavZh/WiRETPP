from __future__ import annotations

import unittest

import numpy as np
import torch

from wishart_tpp.backbones.factory import BackboneFactory
from wishart_tpp.backbones.integration import GaussLegendreRule, MonteCarloRule
from wishart_tpp.config import ModelConfig
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import DatasetPartition, EventSequence
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.model.wishart import sample_wishart, wishart_kl
from wishart_tpp.training.evaluation import ModelEvaluator


class ModelMathTest(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[EventSequence, ...]:
        return (
            EventSequence(np.array([0.2, 0.7]), np.array([0, 1]), 1.0),
            EventSequence(np.array([0.4]), np.array([1]), 1.0),
        )

    def test_default_cotic_matches_official_parameterization(self) -> None:
        model = CoticIntensityBank(1, 8)
        paper_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name != "mixture_logits"
        )
        self.assertEqual(paper_parameters, 6_342_352)
        self.assertEqual(len(model.encoder.layers), 8)
        self.assertEqual(model.encoder.embedding.embedding_dim, 192)
        self.assertEqual(model.intensity_head.output.in_features, 512)
        self.assertIsInstance(model.integration_rule, MonteCarloRule)
        self.assertEqual(model.integration_rule.samples, 50)

    def test_all_backbones_share_the_trace_contract(self) -> None:
        for name in ("cotic", "thp", "nhp", "rmtpp"):
            with self.subTest(backbone=name):
                config = ModelConfig(
                    backbone=name,
                    input_channels=4,
                    hidden_size=8,
                    layers=1,
                    attention_heads=2,
                    kernel_size=2,
                    dropout=0.0,
                    integral_samples=2,
                )
                trace = BackboneFactory.create(config, 2, 2, 11, 12)(self._sequences())
                self.assertEqual(trace.event_base_rates.shape, (3, 2, 2))
                self.assertEqual(trace.integral_base_rates.shape[-2:], (2, 2))

    def test_alpha_zero_is_the_pure_mixture_endpoint(self) -> None:
        model = CoticIntensityBank(
            1,
            2,
            input_channels=4,
            hidden_size=8,
            layers=1,
            kernel_size=2,
            dropout=0.0,
            integration_rule=GaussLegendreRule(3),
        )
        model.expand_components(2, noise=0.01, seed=4)
        trace = model(self._sequences())
        decoder = ActiveBlockDecoder()
        means = torch.eye(2)[None, None].expand(2, 2, -1, -1)
        degrees = torch.full((2, 2), 5.0)
        draws = sample_wishart(means, degrees, 3, torch.Generator().manual_seed(7))
        active = decoder.component_scores(trace, draws, alpha=0.0)
        pure = decoder.base_component_scores(trace)
        torch.testing.assert_close(active, pure[:, :, None].expand_as(active))

    def test_active_rates_are_convex_in_intensity_space(self) -> None:
        base = torch.tensor([[[2.0, 8.0]]])
        draws = torch.diag(torch.tensor([4.0, 0.25])).reshape(1, 1, 1, 2, 2)
        paths = torch.zeros(1, dtype=torch.long)
        decoder = ActiveBlockDecoder(intensity_floor=1e-6)
        transformed = torch.tensor([[[[8.0, 2.0]]]])
        torch.testing.assert_close(
            decoder.rates(base, draws, paths, 0.0), base[:, :, None] + 1e-6
        )
        torch.testing.assert_close(
            decoder.rates(base, draws, paths, 0.5),
            0.5 * base[:, :, None] + 0.5 * transformed + 1e-6,
        )
        torch.testing.assert_close(
            decoder.rates(base, draws, paths, 1.0), transformed + 1e-6
        )

    def test_decoder_rejects_invalid_intensity_floor(self) -> None:
        with self.assertRaises(ValueError):
            ActiveBlockDecoder(intensity_floor=-1e-6)
        with self.assertRaises(ValueError):
            ActiveBlockDecoder(intensity_floor=float("nan"))
        with self.assertRaises(ValueError):
            ActiveBlockDecoder(intensity_floor=1e-6, intensity_floor_scaling="bad")

    def test_equal_wishart_laws_have_zero_kl(self) -> None:
        mean = torch.tensor([[[[1.2, 0.1], [0.1, 0.8]]]])
        degrees = torch.tensor([[5.0]])
        value = wishart_kl(mean, degrees, mean, 5.0)
        torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-6, rtol=0.0)

    def test_wishart_sampling_stabilizes_roundoff_and_gradients(self) -> None:
        singular = torch.ones(1, 1, 3, 3) + 1e-8 * torch.eye(3)
        degrees = torch.full((1, 1), 6.0)
        draws = sample_wishart(singular, degrees, 2, torch.Generator().manual_seed(19))
        self.assertTrue(bool(torch.isfinite(draws).all()))
        self.assertTrue(bool(torch.linalg.eigvalsh(draws).min() > 0.0))
        mean = torch.ones(1, 1, 3, 3, requires_grad=True)
        sample_wishart(
            mean, degrees, 2, torch.Generator().manual_seed(23)
        ).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))

    def test_monte_carlo_rule_resamples_only_training_draws(self) -> None:
        lengths = torch.tensor([1.0, 2.0])
        rule = MonteCarloRule(samples=50_000, seed=17)
        first = rule.points(lengths, training=False)
        second = rule.points(lengths, training=False)
        estimate = (first.weights * first.elapsed.square()).sum(dim=1)
        torch.testing.assert_close(
            estimate, lengths.pow(3) / 3.0, rtol=0.015, atol=0.003
        )
        torch.testing.assert_close(first.elapsed, second.elapsed)
        training_first = rule.points(lengths, training=True)
        training_second = rule.points(lengths, training=True)
        self.assertFalse(torch.equal(training_first.elapsed, training_second.elapsed))

    def test_cotic_uses_paper_event_alignment_without_learned_bos(self) -> None:
        model = CoticIntensityBank(
            1,
            2,
            input_channels=4,
            hidden_size=8,
            layers=1,
            kernel_size=2,
            dropout=0.0,
            integration_rule=MonteCarloRule(4, seed=3),
        )
        model.eval()
        sequence = EventSequence(np.array([0.2, 0.7]), np.array([0, 1]), 0.7)
        encoded, valid = model.states_after_events((sequence,))
        self.assertEqual(model.encoder.embedding.num_embeddings, 3)
        self.assertTrue(bool(valid.all()))
        expected_first = model.rates_from_states(
            encoded.new_zeros(1, encoded.shape[-1]), torch.tensor([0.2])
        )[0]
        expected_second = model.rates_from_states(encoded[:, 0], torch.tensor([0.5]))[0]
        trace = model((sequence,))
        torch.testing.assert_close(trace.event_base_rates[0], expected_first)
        torch.testing.assert_close(trace.event_base_rates[1], expected_second)

    def test_prior_evaluation_combines_independent_mc_repeats(self) -> None:
        class Decoder:
            def __init__(self) -> None:
                self.seeds: list[int] = []

            def prior_predictive(self, *args):
                self.seeds.append(args[4])
                value = float(len(self.seeds))
                return torch.tensor([[value, 0.0], [0.0, value]])

        sequences = (
            EventSequence(np.array([0.2]), np.array([0]), 1.0),
            EventSequence(np.array([0.3]), np.array([0]), 1.0),
        )
        partition = DatasetPartition(sequences, np.array([0, 1]), np.array([0, 1]))
        decoder = Decoder()
        evaluator = ModelEvaluator(decoder, cache_batch_size=2)
        evaluator.cache_builder.build = lambda model, data: ()
        result = evaluator.active(
            object(),
            partition,
            torch.eye(1),
            torch.zeros(2),
            population_df=3.0,
            alpha=0.2,
            samples=4,
            seed=11,
            repeats=3,
        )
        expected = torch.logsumexp(torch.tensor([1.0, 2.0, 3.0]), 0)
        expected -= torch.log(torch.tensor(3.0))
        torch.testing.assert_close(result.component_scores[0, 0], expected)
        self.assertEqual(decoder.seeds, [11, 1_000_014, 2_000_017])


if __name__ == "__main__":
    unittest.main()
