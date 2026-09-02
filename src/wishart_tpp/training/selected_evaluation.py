"""Robust selected-checkpoint evaluation for matched branches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wishart_tpp.training.artifact_io import write_csv, write_json
from wishart_tpp.training.experiment import ExperimentContext, SharedTraining
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner

METHODS = ("cotic_k1", "pure_k5", "wishart_k5")


def evaluate_selected(
    runner: WishartExperimentRunner,
    context: ExperimentContext,
    shared: SharedTraining,
    group: Path,
    *,
    skip_test: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for method in METHODS:
        output = group / method
        result_path = output / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("complete") and (skip_test or result.get("test_read")):
                results[method] = result
                print(f"[{method}] stage=evaluation_reused", flush=True)
                continue
        training = json.loads(
            (output / "training_complete.json").read_text(encoding="utf-8")
        )
        checkpoint = torch.load(
            output / "selected_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if method == "cotic_k1":
            model = runner._model(context.dataset, n_components=1)
            model.load_state_dict(checkpoint["model_state"])
            model = runner.fabric.setup(model)
            validation = context.evaluator.pure(model, context.split.validation)
            test = (
                None if skip_test else context.evaluator.pure(model, context.split.test)
            )
        elif method == "pure_k5":
            model = runner._expanded_model_from_shared(context.dataset, shared)
            model.load_state_dict(checkpoint["model_state"])
            model = runner.fabric.setup(model)
            validation = context.evaluator.pure(model, context.split.validation)
            test = (
                None if skip_test else context.evaluator.pure(model, context.split.test)
            )
        else:
            model = runner._expanded_model_from_shared(context.dataset, shared)
            model.load_state_dict(checkpoint["model_state"])
            model = runner.fabric.setup(model)
            device = getattr(model, "module", model).device
            kwargs = {
                "population_df": float(checkpoint["population_df"]),
                "alpha": float(checkpoint["alpha"]),
                "samples": runner.config.training.test_samples,
                "repeats": runner.config.training.test_monte_carlo_repeats,
            }
            validation = context.evaluator.active(
                model,
                context.split.validation,
                checkpoint["population_means"].to(device),
                checkpoint["log_weights"].to(device),
                seed=runner.config.runtime.monte_carlo_seed + 121_000_057,
                **kwargs,
            )
            test = (
                None
                if skip_test
                else context.evaluator.active(
                    model,
                    context.split.test,
                    checkpoint["population_means"].to(device),
                    checkpoint["log_weights"].to(device),
                    seed=runner.config.runtime.monte_carlo_seed + 131_000_063,
                    **kwargs,
                )
            )
        if test is not None:
            write_csv(
                output / "test_paths.csv",
                pd.DataFrame(
                    {
                        "source_id": context.split.test.source_ids,
                        "horizon": [
                            item.horizon for item in context.split.test.sequences
                        ],
                        "log_likelihood": test.marginal_scores.detach().cpu().numpy(),
                    }
                ),
            )
        result = {
            **training,
            "selected_validation": validation.as_dict(),
            "test": None if test is None else test.as_dict(),
            "test_read": test is not None,
            "complete": True,
        }
        write_json(result_path, result)
        results[method] = result
        print(
            f"[{method}] stage=evaluation_complete validation_nll="
            f"{validation.nll_per_exposure:.6f} test_nll="
            f"{'skipped' if test is None else f'{test.nll_per_exposure:.6f}'}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def combine_group_curves(group: Path) -> None:
    summaries = []
    paths = []
    for method in METHODS:
        output = group / method
        summaries.append(pd.read_csv(output / "validation_curve.csv"))
        paths.append(pd.read_csv(output / "validation_curve_paths.csv"))
    write_csv(group / "validation_curves.csv", pd.concat(summaries, ignore_index=True))
    write_csv(group / "validation_curve_paths.csv", pd.concat(paths, ignore_index=True))
