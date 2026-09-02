from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import torch
from lightning.fabric import Fabric

from wishart_tpp.backbones.integration import GaussLegendreRule
from wishart_tpp.config import ExperimentConfig
from wishart_tpp.cotic import CoticIntensityBank
from wishart_tpp.data import (
    DatasetPartition,
    DatasetSplit,
    EventDataset,
    EventSequence,
    TimeNormalization,
)
from wishart_tpp.inference.alpha import AlphaMstep
from wishart_tpp.inference.local import LocalPosterior, LocalWishartInference
from wishart_tpp.inference.population import PopulationMstep
from wishart_tpp.inference.population_df import PopulationDfMstep
from wishart_tpp.inference.responsibilities import ResponsibilityUpdater
from wishart_tpp.model.active_block import ActiveBlockDecoder
from wishart_tpp.training.active import ActiveSchedule, ActiveTrainer
from wishart_tpp.training.evaluation import ModelEvaluator
from wishart_tpp.training.experiment import ExperimentContext
from wishart_tpp.training.pure import PureMixtureTrainer, PureSchedule
from wishart_tpp.training.state import clone_state_dict
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner


class FabricSmokeTest(unittest.TestCase):
    def assert_tensor_tree_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tensor_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_tensor_tree_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)

    def test_local_inference_repairs_only_nonfinite_gradient_entries(self) -> None:
        inference = LocalWishartInference(
            Fabric(accelerator="cpu"), ActiveBlockDecoder(), gradient_clip=10.0
        )
        parameter = torch.nn.Parameter(torch.zeros(4))
        module = torch.nn.ParameterList([parameter])
        parameter.grad = torch.tensor([float("nan"), float("inf"), -float("inf"), 2.0])

        repaired = inference._repair_nonfinite_gradients(module)

        self.assertEqual(repaired, 3)
        torch.testing.assert_close(
            parameter.grad, torch.tensor([0.0, 10.0, -10.0, 2.0])
        )

    @staticmethod
    def _partition():
        sequences = tuple(
            EventSequence(
                np.array([0.2 + 0.02 * index, 0.7]),
                np.array([index % 2, (index + 1) % 2]),
                1.0,
            )
            for index in range(6)
        )
        train = DatasetPartition(sequences[:4], np.array([0, 1, 0, 1]), np.arange(4))
        validation = DatasetPartition(sequences[4:], np.array([0, 1]), np.arange(4, 6))
        return train, validation

    @staticmethod
    def _model():
        model = CoticIntensityBank(
            1,
            2,
            input_channels=4,
            hidden_size=8,
            layers=1,
            kernel_size=2,
            dropout=0.0,
            integration_rule=GaussLegendreRule(2),
        )
        model.expand_components(2, noise=0.02, seed=2)
        return model

    def test_shared_fabric_checkpoint_loads_into_a_fresh_branch(self) -> None:
        train, validation = self._partition()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        evaluator = ModelEvaluator(decoder, cache_batch_size=2)
        trainer = PureMixtureTrainer(
            fabric,
            decoder,
            evaluator,
            batch_size=2,
            trace_batch_size=2,
            effective_batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )
        model, optimizer = trainer.setup(self._model())
        trainer.fit(
            model, optimizer, train, validation, steps=1, seed=0, select_best=False
        )
        fresh = self._model()
        fresh.load_state_dict(clone_state_dict(model))

    def test_runner_pretrains_one_head_before_expansion(self) -> None:
        train, validation = self._partition()
        dataset = EventDataset(
            "smoke",
            2,
            2,
            train.sequences + validation.sequences,
            np.concatenate([train.labels, validation.labels]),
            np.arange(1, 7),
            (np.arange(4), np.arange(4, 6), np.arange(4, 6)),
            TimeNormalization(1.0, 1.0, "test_identity"),
        )
        split = DatasetSplit(train, validation, validation)
        base = ExperimentConfig()
        config = replace(
            base,
            model=replace(
                base.model,
                input_channels=4,
                hidden_size=8,
                layers=1,
                kernel_size=2,
                dropout=0.0,
                integral_samples=2,
            ),
            training=replace(
                base.training,
                method="wishart",
                shared_pretrain_steps=1,
                batch_size=2,
                evaluation_batch_size=2,
                effective_batch_size=4,
            ),
        )
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        runner = WishartExperimentRunner(config, fabric)
        context = ExperimentContext(
            dataset,
            split,
            2,
            2,
            ModelEvaluator(decoder, cache_batch_size=2),
        )
        shared = runner._train_shared(context)

        self.assertEqual(shared.state["mixture_logits"].shape, (1,))
        self.assertEqual(shared.state["intensity_head.output.weight"].shape, (2, 8))

        expanded = runner._expanded_model_from_shared(dataset, shared)
        self.assertEqual(expanded.n_components, 2)
        self.assertEqual(expanded.mixture_logits.shape, (2,))
        self.assertEqual(expanded.intensity_head.output.weight.shape, (4, 8))
        torch.testing.assert_close(
            expanded.intensity_head.output.weight[:2],
            shared.state["intensity_head.output.weight"],
        )

    def test_pure_neural_step_uses_nested_trace_batches(self) -> None:
        train, _validation = self._partition()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = PureMixtureTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            batch_size=2,
            trace_batch_size=1,
            effective_batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )
        source = self._model()
        calls = []
        original_forward = source.forward

        def recorded_forward(sequences):
            calls.append(len(sequences))
            return original_forward(sequences)

        source.forward = recorded_forward
        model, optimizer = trainer.setup(source)
        trainer._training_step(model, optimizer, train, np.arange(4))
        self.assertEqual(calls, [1, 1, 1, 1])

    def test_pure_balanced_responsibilities_have_equal_global_mass(self) -> None:
        train, _ = self._partition()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = PureMixtureTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            batch_size=2,
            trace_batch_size=2,
            effective_batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
            responsibility_updater=ResponsibilityUpdater(balance_iterations=40),
        )
        model, _ = trainer.setup(self._model())

        responsibilities = trainer._balanced_responsibilities(model, train)

        torch.testing.assert_close(responsibilities.sum(1), torch.ones(4))
        torch.testing.assert_close(responsibilities.sum(0), torch.full((2,), 2.0))

    def test_pure_cycle_schedule_balances_only_the_warmup(self) -> None:
        train, validation = self._partition()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = PureMixtureTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            batch_size=2,
            trace_batch_size=2,
            effective_batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
            responsibility_updater=ResponsibilityUpdater(balance_iterations=40),
        )
        model, optimizer = trainer.setup(self._model())

        history, _ = trainer.fit_cycles(
            model,
            optimizer,
            train,
            validation,
            PureSchedule(cycles=2, neural_steps=1, balanced_cycles=1),
            seed=0,
        )

        self.assertEqual([row["balanced"] for row in history], [0.0, 1.0, 0.0])
        self.assertEqual([row["step"] for row in history], [0.0, 1.0, 2.0])

    def test_active_neural_loss_uses_nested_trace_batches(self) -> None:
        train, _ = self._partition()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = ActiveTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )
        model = self._model()
        calls = []
        original_forward = model.forward

        def recorded_forward(sequences):
            calls.append(len(sequences))
            return original_forward(sequences)

        model.forward = recorded_forward
        identity = torch.eye(2)[None, None].repeat(4, 2, 1, 1)
        posterior = LocalPosterior(
            identity,
            torch.full((4, 2), 5.0),
            torch.zeros(4, 2),
            torch.zeros(4, 2),
        )
        trainer._microbatch_loss(
            model,
            train.select(np.arange(2)),
            np.arange(2),
            posterior,
            torch.full((4, 2), 0.5),
            alpha=0.3,
            samples=1,
            exposure=train.select(np.arange(2)).exposure,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(calls, [1, 1])

    def test_one_active_cycle_runs(self) -> None:
        train, validation = self._partition()
        model = self._model()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        evaluator = ModelEvaluator(decoder, cache_batch_size=2)
        local = LocalWishartInference(fabric, decoder)
        trainer = ActiveTrainer(
            fabric,
            decoder,
            evaluator,
            local,
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
        )
        model, optimizer = trainer.setup(model)
        schedule = ActiveSchedule(
            cycles=1,
            neural_steps=1,
            local_steps=1,
            local_samples=1,
            local_evaluation_samples=1,
            population_df=5.0,
            initial_alpha=0.3,
            alpha_steps=2,
            alpha_samples=1,
            balanced_cycles=1,
            validation_samples=2,
        )
        best, history = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            schedule,
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        self.assertEqual(best.cycle, 1)
        self.assertEqual(len(history), 1)

    def test_active_cycle_can_freeze_neural_model(self) -> None:
        train, validation = self._partition()
        model = self._model()
        initial = clone_state_dict(model)
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        trainer = ActiveTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
            update_neural_model=False,
        )
        model, optimizer = trainer.setup(model)
        schedule = ActiveSchedule(
            cycles=1,
            neural_steps=1,
            local_steps=1,
            local_samples=1,
            local_evaluation_samples=1,
            population_df=5.0,
            initial_alpha=0.0,
            alpha_steps=2,
            alpha_samples=1,
            balanced_cycles=1,
            validation_samples=2,
        )
        _, history = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            schedule,
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        current = clone_state_dict(model)
        for name in initial:
            if name != "mixture_logits":
                torch.testing.assert_close(
                    initial[name], current[name], rtol=0.0, atol=0.0
                )
        self.assertEqual(history[0]["neural_model_updated"], 0.0)

    def test_active_population_initialization_is_random_spd_and_seeded(self) -> None:
        model = self._model()
        first = ActiveTrainer._initial_population_means(model, 5.0, 17)
        repeated = ActiveTrainer._initial_population_means(model, 5.0, 17)
        changed = ActiveTrainer._initial_population_means(model, 5.0, 18)

        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.allclose(first, changed))
        self.assertFalse(torch.allclose(first[0], first[1]))
        torch.testing.assert_close(
            first.diagonal(dim1=-2, dim2=-1).sum(-1),
            first.new_full((model.n_components,), float(model.n_marks)),
        )
        self.assertTrue(bool((torch.linalg.eigvalsh(first) > 0.0).all()))
        identity = torch.eye(model.n_marks, dtype=first.dtype)[None].expand_as(first)
        self.assertFalse(torch.allclose(first, identity))

    def test_population_df_learning_starts_after_warmup(self) -> None:
        train, validation = self._partition()
        model = self._model()
        fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
        decoder = ActiveBlockDecoder()
        df_mstep = Mock(spec=PopulationDfMstep)
        df_mstep.candidate_grid.return_value = torch.tensor([4.0, 5.0, 8.0])
        df_mstep.profile.return_value = object()
        df_mstep.select.return_value = 4.0
        df_mstep.damp.return_value = 4.5
        df_mstep.weighted_kl.return_value = 0.0
        trainer = ActiveTrainer(
            fabric,
            decoder,
            ModelEvaluator(decoder, cache_batch_size=2),
            LocalWishartInference(fabric, decoder),
            PopulationMstep(),
            AlphaMstep(decoder),
            ResponsibilityUpdater(),
            batch_size=2,
            effective_batch_size=4,
            cache_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip=10.0,
            population_df_mstep=df_mstep,
        )
        model, optimizer = trainer.setup(model)
        schedule = ActiveSchedule(
            cycles=2,
            neural_steps=1,
            local_steps=1,
            local_samples=1,
            local_evaluation_samples=1,
            population_df=5.0,
            initial_alpha=0.3,
            alpha_steps=2,
            alpha_samples=1,
            balanced_cycles=1,
            validation_samples=2,
            learn_population_df=True,
            population_df_warmup_cycles=1,
        )
        _, history = trainer.fit(
            model,
            optimizer,
            train,
            validation,
            schedule,
            optimization_seed=0,
            monte_carlo_seed=0,
        )
        self.assertEqual(history[0]["population_df"], 5.0)
        self.assertEqual(history[1]["population_df"], 4.5)
        df_mstep.profile.assert_called_once()
        df_mstep.select.assert_called_once()
        df_mstep.damp.assert_called_once()

if __name__ == "__main__":
    unittest.main()
