"""Exact resumable K=1 pretraining shared by every benchmark branch."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wishart_tpp.training.artifact_io import sha256, write_csv, write_torch
from wishart_tpp.training.experiment import ExperimentContext, SharedTraining
from wishart_tpp.training.state import clone_state_dict
from wishart_tpp.training.validation_curves import ValidationCurveWriter
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner


class SharedPretrainer:
    """Fit one K=1 model for a fixed number of validated optimizer updates."""

    def __init__(self, runner: WishartExperimentRunner) -> None:
        self.runner = runner

    def fit(
        self,
        context: ExperimentContext,
        output: Path,
        *,
        steps: int,
    ) -> SharedTraining:
        final_state = output / "shared_checkpoint.pt"
        final_history = output / "shared_history.csv"
        resume_path = output / "shared_pretrain_checkpoint.pt"
        seed = self.runner.config.runtime.optimization_seed
        curve = ValidationCurveWriter(
            output, dataset=context.dataset.name, seed=seed, method="shared_k1"
        )
        if final_state.is_file() and final_history.is_file():
            if not set(range(steps + 1)).issubset(curve.completed_cycles()):
                raise RuntimeError("completed shared pretrain lacks validation points")
            return SharedTraining(
                torch.load(final_state, map_location="cpu", weights_only=False),
                pd.read_csv(final_history).to_dict(orient="records"),
            )

        trainer = self.runner._pure_trainer(context)
        model = self.runner._model(context.dataset, n_components=1)
        model, optimizer = trainer.setup(model)
        sample_seed = seed + 103
        generator = np.random.default_rng(sample_seed)
        signature = {
            "format": "fixed_shared_pretrain",
            "version": 1,
            "dataset": context.dataset.name,
            "seed": seed,
            "steps": steps,
            "sample_seed": sample_seed,
            "train_paths": len(context.split.train.sequences),
            "validation_paths": len(context.split.validation.sequences),
            "effective_batch_size": self.runner.config.training.effective_batch_size,
            "learning_rate": self.runner.config.training.learning_rate,
            "validation_every_step": True,
        }
        history: list[dict[str, float]] = []
        completed_step = 0

        if resume_path.is_file():
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if payload.get("signature") != signature:
                raise ValueError("shared pretrain resume signature mismatch")
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            history = payload["history"]
            completed_step = int(payload["completed_step"])
            generator.bit_generator.state = payload["numpy_generator_state"]
            self._restore_random_state(model, payload)
            if not set(range(completed_step + 1)).issubset(curve.completed_cycles()):
                raise RuntimeError("shared checkpoint lacks validation points")
        else:
            self.runner._seed_neural_randomness()
            getattr(model, "module", model).integration_rule.reset_training_draws(
                self.runner.config.model.integral_seed
            )
            initial = context.evaluator.pure(model, context.split.validation)
            curve.record(
                0,
                0,
                initial,
                context.split.validation,
                extras={"phase": "shared_pretrain", "learning_rate": 1e-4},
            )

        started = time.perf_counter()
        for step in range(completed_step + 1, steps + 1):
            indices = trainer._sample_indices(generator, context.split.train)
            train_nll = trainer._training_step(
                model, optimizer, context.split.train, indices
            )
            evaluation = context.evaluator.pure(model, context.split.validation)
            history.append(
                {
                    "step": float(step),
                    "train_nll_per_exposure": float(train_nll),
                    "validation_nll_per_exposure": float(evaluation.nll_per_exposure),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            curve.record(
                step,
                step,
                evaluation,
                context.split.validation,
                extras={"phase": "shared_pretrain", "learning_rate": 1e-4},
            )
            write_torch(
                resume_path,
                self._resume_payload(
                    signature, step, model, optimizer, history, generator
                ),
            )
            elapsed = time.perf_counter() - started
            eta = elapsed / max(step - completed_step, 1) * (steps - step)
            print(
                f"[shared] step={step}/{steps} train_nll={train_nll:.6f} "
                f"validation_nll={evaluation.nll_per_exposure:.6f} "
                f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}",
                flush=True,
            )

        write_torch(final_state, clone_state_dict(model))
        write_csv(final_history, pd.DataFrame(history))
        print(f"[shared] complete sha256={sha256(final_state)}", flush=True)
        return SharedTraining(clone_state_dict(model), history)

    @staticmethod
    def _resume_payload(signature, step, model, optimizer, history, generator):
        return {
            "signature": signature,
            "completed_step": step,
            "model_state": clone_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "numpy_generator_state": generator.bit_generator.state,
            "integration_training_draw": getattr(
                model, "module", model
            ).integration_rule._training_draw,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

    @staticmethod
    def _restore_random_state(model, payload) -> None:
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        getattr(model, "module", model).integration_rule._training_draw = int(
            payload["integration_training_draw"]
        )
