"""Training of matched COTIC K=1 and Pure K=5 branches."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wishart_tpp.training.artifact_io import (
    pure_checkpoint_from_payload,
    pure_checkpoint_payload,
    sha256,
    write_csv,
    write_json,
    write_torch,
)
from wishart_tpp.training.experiment import ExperimentContext, SharedTraining
from wishart_tpp.training.state import PureCheckpoint, clone_state_dict
from wishart_tpp.training.validation_curves import ValidationCurveWriter
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner


def _curve_extras(model, best_step: int, learning_rate: float) -> dict[str, Any]:
    weights = model.mixture_log_weights().detach().cpu().exp().numpy()
    positive = weights[weights > 0.0]
    effective_k = float(np.exp(-(positive * np.log(positive)).sum()))
    extras: dict[str, Any] = {
        "best_step_so_far": int(best_step),
        "learning_rate": float(learning_rate),
        "effective_k": effective_k,
    }
    extras.update(
        {f"mixture_weight_{index}": float(value) for index, value in enumerate(weights)}
    )
    return extras


def fit_pure_branch(
    runner: WishartExperimentRunner,
    context: ExperimentContext,
    shared: SharedTraining,
    output: Path,
    *,
    method: str,
    components: int,
    validation_interval: int | None = None,
    seed_scheduler_with_initial_validation: bool = False,
    continuous_optimizer_checkpoint: Path | None = None,
) -> dict[str, Any]:
    complete_path = output / "training_complete.json"
    if complete_path.is_file():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("complete"):
            print(f"[{method}] stage=reused", flush=True)
            return payload
    output.mkdir(parents=True, exist_ok=True)
    if components == 1:
        model = runner._model(context.dataset, n_components=1)
        model.load_state_dict(shared.state)
    else:
        model = runner._expanded_model_from_shared(context.dataset, shared)
    trainer = runner._pure_trainer(context)
    model, optimizer = trainer.setup(model)
    scheduler = trainer._lr_scheduler(optimizer)
    steps = runner.config.training.pure_steps
    interval = (
        max(1, steps // 12) if validation_interval is None else int(validation_interval)
    )
    if interval < 1:
        raise ValueError("validation_interval must be positive")
    sample_seed = runner.config.runtime.optimization_seed + 107
    curve = ValidationCurveWriter(
        output,
        dataset=context.dataset.name,
        seed=runner.config.runtime.optimization_seed,
        method=method,
    )
    signature = {
        "format": "three_method_pure_branch",
        "version": 1,
        "dataset": context.dataset.name,
        "method": method,
        "components": components,
        "steps": steps,
        "validation_interval": interval,
        "sample_seed": sample_seed,
        "train_paths": len(context.split.train.sequences),
        "validation_paths": len(context.split.validation.sequences),
        "shared_sha256": sha256(output.parent / "shared" / "shared_checkpoint.pt"),
    }
    if seed_scheduler_with_initial_validation:
        signature["scheduler_seeded_with_initial_validation"] = True
    if continuous_optimizer_checkpoint is not None:
        signature["continuous_optimizer_checkpoint_sha256"] = sha256(
            continuous_optimizer_checkpoint
        )
    resume_path = output / "step_checkpoint.pt"
    random_generator = np.random.default_rng(sample_seed)
    history: list[dict[str, float]] = []
    completed_step = 0
    draw_owner = getattr(model, "module", model).integration_rule

    if resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(f"{method} resume signature mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        history = payload["history"]
        completed_step = int(payload["completed_step"])
        best = pure_checkpoint_from_payload(payload["best"])
        random_generator.bit_generator.state = payload["numpy_generator_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        draw_owner._training_draw = int(payload["integration_training_draw"])
        required = set(range(completed_step // interval + 1))
        if not required.issubset(curve.completed_cycles()):
            raise RuntimeError(
                f"{method} checkpoint is missing validation curve points"
            )
        print(
            f"[{method}-checkpoint] resumed_step={completed_step}/{steps} "
            f"best_step={best.step}",
            flush=True,
        )
    else:
        if continuous_optimizer_checkpoint is None:
            runner._seed_neural_randomness()
            draw_owner.reset_training_draws(runner.config.model.integral_seed)
        else:
            shared_payload = torch.load(
                continuous_optimizer_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if int(shared_payload.get("completed_step", -1)) < 1:
                raise ValueError("continuous optimizer checkpoint is incomplete")
            optimizer.load_state_dict(shared_payload["optimizer_state"])
            torch.set_rng_state(shared_payload["torch_rng_state"])
            if torch.cuda.is_available() and shared_payload["cuda_rng_state_all"]:
                torch.cuda.set_rng_state_all(shared_payload["cuda_rng_state_all"])
            np.random.set_state(shared_payload["numpy_rng_state"])
            random.setstate(shared_payload["python_rng_state"])
            draw_owner._training_draw = int(shared_payload["integration_training_draw"])
        evaluation = context.evaluator.pure(model, context.split.validation)
        best = PureCheckpoint(evaluation.nll_per_exposure, 0, clone_state_dict(model))
        # ReduceLROnPlateau must see the selection baseline.  Otherwise the
        # first (possibly worse) post-update validation is treated as its best
        # value and an unstable branch can retain an excessive learning rate.
        if scheduler is not None and seed_scheduler_with_initial_validation:
            scheduler.step(evaluation.nll_per_exposure)
        curve.record(
            0,
            0,
            evaluation,
            context.split.validation,
            extras=_curve_extras(model, 0, float(optimizer.param_groups[0]["lr"])),
        )

    started = time.perf_counter()
    print(
        f"[{method}] step={completed_step}/{steps} validation_nll="
        f"{best.validation_nll:.6f} best_step={best.step}",
        flush=True,
    )
    for step in range(completed_step + 1, steps + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        indices = trainer._sample_indices(random_generator, context.split.train)
        loss = trainer._training_step(model, optimizer, context.split.train, indices)
        row: dict[str, float] = {
            "step": float(step),
            "train_nll_per_exposure": loss,
            "learning_rate": learning_rate,
        }
        if step == steps or step % interval == 0:
            evaluation = context.evaluator.pure(model, context.split.validation)
            candidate = PureCheckpoint(
                evaluation.nll_per_exposure, step, clone_state_dict(model)
            )
            if candidate.validation_nll < best.validation_nll:
                best = candidate
            row["validation_nll_per_exposure"] = candidate.validation_nll
            if scheduler is not None:
                scheduler.step(candidate.validation_nll)
            next_lr = float(optimizer.param_groups[0]["lr"])
            row["next_learning_rate"] = next_lr
            row["lr_reduced"] = float(next_lr < learning_rate)
            curve.record(
                step // interval,
                step,
                evaluation,
                context.split.validation,
                extras=_curve_extras(model, best.step, learning_rate),
            )
            elapsed = time.perf_counter() - started
            eta = elapsed / max(step - completed_step, 1) * (steps - step)
            print(
                f"[{method}] step={step}/{steps} validation_nll="
                f"{candidate.validation_nll:.6f} best_step={best.step} "
                f"lr={learning_rate:.8g} elapsed_seconds={elapsed:.1f} "
                f"eta_seconds={eta:.1f}",
                flush=True,
            )
            history.append(row)
            write_torch(
                resume_path,
                {
                    "signature": signature,
                    "completed_step": step,
                    "model_state": clone_state_dict(model),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "best": pure_checkpoint_payload(best),
                    "history": history,
                    "numpy_generator_state": random_generator.bit_generator.state,
                    "integration_training_draw": draw_owner._training_draw,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else []
                    ),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
            )
            print(f"[{method}-checkpoint] saved_step={step}/{steps}", flush=True)
        else:
            history.append(row)

    model.load_state_dict(best.model_state)
    write_csv(output / "history.csv", pd.DataFrame(history))
    write_torch(
        output / "selected_checkpoint.pt",
        {
            "method": method,
            "components": components,
            "model_state": best.model_state,
            "best_step": best.step,
            "selection_validation_nll": best.validation_nll,
        },
    )
    selected = context.evaluator.pure(model, context.split.validation)
    selected_paths = pd.DataFrame(
        {
            "source_id": context.split.validation.source_ids,
            "horizon": [item.horizon for item in context.split.validation.sequences],
            "log_likelihood": selected.marginal_scores.detach().cpu().numpy(),
        }
    )
    write_csv(output / "selected_validation_paths.csv", selected_paths)
    result = {
        "complete": True,
        "method": method,
        "components": components,
        "neural_updates_after_shared": steps,
        "best_step": best.step,
        "selection_validation_nll": best.validation_nll,
        "selected_validation_nll": selected.nll_per_exposure,
        "mixture_weights": model.mixture_log_weights().detach().cpu().exp().tolist(),
        "test_read": False,
    }
    write_json(complete_path, result)
    print(
        f"[{method}] stage=training_complete best_step={best.step} "
        f"validation_nll={selected.nll_per_exposure:.6f}",
        flush=True,
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
