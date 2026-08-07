#!/usr/bin/env python3
"""Tune Wishart-specific THP hyperparameters on one DAN validation split."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
import torch

from lal_wishart.metrics import (
    adjusted_rand_index,
    cluster_purity,
    normalized_mutual_information,
)
from lal_wishart.models.signed_wishart import SignedWishartTPP
from lal_wishart.reproduction.dan_synthetic import (
    dataset_names,
    load_dataset,
    shuffled_split,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_latent_wishart_nhp,
    fit_latent_wishart_attention_nhp,
)
from run_corrected_shared_wishart_architectures import (
    _initial_backbone,
    _set_seed,
)
from run_dan_synthetic_benchmark import (
    _dynamic_batch_size,
    _evaluation_batch_size,
)


BASELINE_PARAMETERS = {
    "wishart_nu": 25,
    "omega_learning_rate": 1e-2,
    "alpha_learning_rate": 1e-3,
    "alpha_temperature": 0.5,
    "mean_hyperprior_strength": 1.0,
}

OMEGA_LR_RANGE = (1e-3, 5e-2)
ALPHA_LR_RANGE = (1e-4, 3e-3)
ALPHA_TEMPERATURE_RANGE = (0.2, 2.0)
MEAN_HYPERPRIOR_RANGE = (0.1, 10.0)


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _study_storage(database: Path) -> str:
    return f"sqlite:///{database.resolve().as_posix()}"


def _trial_parameters(
    trial: optuna.Trial,
    *,
    dimension: int,
) -> dict[str, float | int]:
    return {
        "wishart_nu": trial.suggest_int(
            "wishart_nu",
            dimension,
            8 * dimension,
            log=True,
        ),
        "omega_learning_rate": trial.suggest_float(
            "omega_learning_rate",
            *OMEGA_LR_RANGE,
            log=True,
        ),
        "alpha_learning_rate": trial.suggest_float(
            "alpha_learning_rate",
            *ALPHA_LR_RANGE,
            log=True,
        ),
        "alpha_temperature": trial.suggest_float(
            "alpha_temperature",
            *ALPHA_TEMPERATURE_RANGE,
            log=True,
        ),
        "mean_hyperprior_strength": trial.suggest_float(
            "mean_hyperprior_strength",
            *MEAN_HYPERPRIOR_RANGE,
            log=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="sin_K5_C5")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/optuna_thp_wishart_sin_K5_C5"),
    )
    parser.add_argument("--study-name", default="thp_wishart_sin_K5_C5")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--evaluation-interval", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=20260802)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        help=(
            "Import an already completed, protocol-identical baseline result "
            "into a new study instead of recomputing it."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if (
        args.n_trials <= 0
        or args.epochs <= 0
        or args.evaluation_interval <= 0
        or args.train_samples <= 0
        or args.validation_samples <= 0
        or args.test_samples <= 0
        or args.seed < 0
        or args.split_seed < 0
    ):
        raise ValueError("trials and epochs must be positive; seed must be non-negative")

    names = dataset_names(args.data_root)
    if args.dataset not in names:
        raise ValueError(f"unknown dataset: {args.dataset}")
    dataset_index = names.index(args.dataset)
    dataset = load_dataset(args.data_root, args.dataset)
    split = shuffled_split(dataset, seed=args.split_seed)
    train = split.train
    validation = split.validation
    validation_labels = split.validation_labels
    test = split.test
    test_labels = split.test_labels
    test_ids = split.test_ids
    if args.quick:
        train = train[: min(32, len(train))]
        validation = validation[: min(16, len(validation))]
        validation_labels = validation_labels[: len(validation)]
        test = test[: min(16, len(test))]
        test_labels = test_labels[: len(test)]
        test_ids = test_ids[: len(test)]

    args.outdir.mkdir(parents=True, exist_ok=True)
    database = args.outdir / "study.sqlite3"
    storage = _study_storage(database)
    try:
        existing = optuna.load_study(
            study_name=args.study_name,
            storage=storage,
        )
        existing_trials = len(existing.trials)
    except KeyError:
        existing_trials = 0
    effective_sampler_seed = args.sampler_seed + existing_trials
    sampler = optuna.samplers.TPESampler(
        seed=effective_sampler_seed,
        multivariate=True,
        n_startup_trials=5,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    dimension = dataset.n_components * dataset.n_marks
    if not study.trials and args.baseline_result is not None:
        baseline_result = json.loads(
            args.baseline_result.read_text(encoding="utf-8")
        )
        baseline = dict(BASELINE_PARAMETERS)
        baseline["wishart_nu"] = dimension
        if baseline_result.get("parameters") != baseline:
            raise ValueError(
                "imported baseline parameters do not match this study"
            )
        if (
            baseline_result.get("split_seed") != args.split_seed
            or baseline_result.get("model_seed") != args.seed
        ):
            raise ValueError(
                "imported baseline split/model seeds do not match this study"
            )
        study.add_trial(optuna.trial.create_trial(
            params=baseline,
            distributions={
                "wishart_nu": optuna.distributions.IntDistribution(
                    dimension,
                    8 * dimension,
                    log=True,
                ),
                "omega_learning_rate": (
                    optuna.distributions.FloatDistribution(
                        *OMEGA_LR_RANGE,
                        log=True,
                    )
                ),
                "alpha_learning_rate": (
                    optuna.distributions.FloatDistribution(
                        *ALPHA_LR_RANGE,
                        log=True,
                    )
                ),
                "alpha_temperature": (
                    optuna.distributions.FloatDistribution(
                        *ALPHA_TEMPERATURE_RANGE,
                        log=True,
                    )
                ),
                "mean_hyperprior_strength": (
                    optuna.distributions.FloatDistribution(
                        *MEAN_HYPERPRIOR_RANGE,
                        log=True,
                    )
                ),
            },
            value=float(baseline_result["best_validation_purity"]),
            user_attrs={
                "role": "imported_current_baseline",
                "source_result": str(args.baseline_result.resolve()),
                "best_epoch": int(baseline_result["best_epoch"]),
                "best_validation_nll_per_time": float(
                    baseline_result["best_validation_nll_per_time"]
                ),
                "final_validation_purity": float(
                    baseline_result["final_validation_purity"]
                ),
                "learned_alpha": float(
                    baseline_result["learned_alpha"]
                ),
                "mean_abs_omega_minus_identity": float(
                    baseline_result["mean_abs_omega_minus_identity"]
                ),
                "runtime_seconds": float(
                    baseline_result["runtime_seconds"]
                ),
            },
        ))
    elif not study.trials:
        baseline = dict(BASELINE_PARAMETERS)
        baseline["wishart_nu"] = dimension
        study.enqueue_trial(
            baseline,
            user_attrs={"role": "current_baseline"},
        )

    device = torch.device(args.device)
    maximum_events = max(sequence.count for sequence in dataset.sequences)
    batch_size = _dynamic_batch_size(maximum_events)
    evaluation_batch_size = _evaluation_batch_size(maximum_events)
    if args.quick:
        batch_size = min(batch_size, 8)
        evaluation_batch_size = min(evaluation_batch_size, 2)
    epochs = 2 if args.quick else args.epochs
    evaluation_interval = 1 if args.quick else args.evaluation_interval
    initial_total_rate = sum(sequence.count for sequence in train) / sum(
        sequence.horizon for sequence in train
    )
    config = {
        "n_components": dataset.n_components,
        "n_marks": dataset.n_marks,
        "horizon": max(sequence.horizon for sequence in dataset.sequences),
        "quadrature_order": 4,
        "nhp_hidden_size": 16,
        "thp_hidden_size": 32,
        "thp_num_layers": 2,
        "thp_num_heads": 4,
        "cotic_input_channels": 32,
        "cotic_hidden_size": 64,
        "cotic_num_layers": 4,
        "cotic_kernel_size": 3,
        "cotic_dilation_factor": 1.29,
        "dropout": 0.1,
    }
    init_seed = (
        2026090100
        + args.seed * 1_000_000
        + dataset_index * 10_000
        + 1_000
    )
    sample_seed = init_seed + 701
    experiment_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "model_seed": args.seed,
        "sampler_seed": args.sampler_seed,
        "effective_sampler_seed": effective_sampler_seed,
        "imported_baseline_result": (
            str(args.baseline_result.resolve())
            if args.baseline_result is not None
            else None
        ),
        "epochs": epochs,
        "selection_metric": "validation_purity",
        "tie_breaker": "validation_nll",
        "test_labels_used_during_optimization": False,
        "test_evaluated_after_each_trial": True,
        "fixed_parameters": {
            "architecture": "thp",
            "initial_alpha": 0.5,
            "alpha_max": 1.0,
            "neural_learning_rate": 1e-3,
            "train_samples": args.train_samples if not args.quick else 1,
            "validation_samples": (
                args.validation_samples if not args.quick else 2
            ),
            "test_samples": args.test_samples if not args.quick else 2,
            "evaluation_interval": evaluation_interval,
        },
        "search_space": {
            "wishart_nu": [dimension, 8 * dimension, "integer_log"],
            "omega_learning_rate": [*OMEGA_LR_RANGE, "log"],
            "alpha_learning_rate": [*ALPHA_LR_RANGE, "log"],
            "alpha_temperature": [*ALPHA_TEMPERATURE_RANGE, "log"],
            "mean_hyperprior_strength": [*MEAN_HYPERPRIOR_RANGE, "log"],
        },
        "storage": str(database.resolve()),
        "study_name": args.study_name,
    }
    _json_write(args.outdir / "config.json", experiment_config)
    study.set_user_attr("experiment_config", experiment_config)

    def objective(trial: optuna.Trial) -> float:
        parameters = _trial_parameters(trial, dimension=dimension)
        trial_dir = args.outdir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _json_write(trial_dir / "parameters.json", parameters)
        started = time.time()
        _set_seed(init_seed)
        initial = _initial_backbone(
            "thp",
            "output_split",
            config,
            initial_total_rate=initial_total_rate,
            initialization_seed=init_seed,
            device=device,
        )
        model = SignedWishartTPP(
            copy.deepcopy(initial),
            degrees_of_freedom=int(parameters["wishart_nu"]),
            alpha_max=1.0,
            initial_alpha=0.5,
            interaction_temperature=float(parameters["alpha_temperature"]),
        ).to(device)
        fit = fit_latent_wishart_attention_nhp(
            model,
            train,
            validation,
            max_epochs=epochs,
            batch_size=batch_size,
            train_samples=(1 if args.quick else args.train_samples),
            validation_samples=(
                2 if args.quick else args.validation_samples
            ),
            neural_learning_rate=1e-3,
            distribution_learning_rate=float(
                parameters["omega_learning_rate"]
            ),
            interaction_learning_rate=float(
                parameters["alpha_learning_rate"]
            ),
            neural_weight_decay=1e-5,
            mean_hyperprior_strength=float(
                parameters["mean_hyperprior_strength"]
            ),
            evaluation_interval=evaluation_interval,
            validation_cutoff=None,
            evaluation_batch_size=min(
                evaluation_batch_size,
                max(
                    1,
                    evaluation_batch_size
                    * 8
                    // (
                        2
                        if args.quick
                        else args.validation_samples
                    ),
                ),
            ),
            gradient_clip=20.0,
            batch_seed=init_seed + 301,
            sample_seed=sample_seed,
            validation_labels=validation_labels,
            selection_metric="validation_purity",
            show_progress=True,
            progress_description=(
                f"trial {trial.number} {args.dataset} Wishart-THP"
            ),
        )
        model = fit.model
        history = pd.DataFrame(fit.history)
        history.to_csv(trial_dir / "history.csv", index=False)
        torch.save(model.state_dict(), trial_dir / "checkpoint.pt")
        test_samples = 2 if args.quick else args.test_samples
        test_evaluation = evaluate_latent_wishart_nhp(
            model,
            test,
            cutoff=None,
            n_samples=test_samples,
            sample_seed=sample_seed + 900_000,
            batch_size=min(
                evaluation_batch_size,
                max(1, evaluation_batch_size * 8 // test_samples),
            ),
        )
        test_probabilities = (
            test_evaluation.full_cluster_probabilities.numpy()
        )
        test_predictions = test_probabilities.argmax(axis=1)
        test_exposure = sum(sequence.horizon for sequence in test)
        pd.DataFrame({
            "source_id": test_ids,
            "true_cluster": test_labels,
            "predicted_cluster": test_predictions,
            **{
                f"probability_{component}": test_probabilities[:, component]
                for component in range(dataset.n_components)
            },
        }).to_csv(trial_dir / "test_predictions.csv", index=False)
        mean = model.mean_matrix().detach()
        identity = torch.eye(
            model.dimension,
            device=mean.device,
            dtype=mean.dtype,
        )
        result = {
            "status": "complete",
            "trial": trial.number,
            "model_seed": args.seed,
            "split_seed": args.split_seed,
            "objective_best_validation_purity": fit.best_validation_purity,
            "best_validation_nll_per_time": (
                fit.best_validation_suffix_nll_per_exposure
            ),
            "best_epoch": fit.best_epoch,
            "final_validation_purity": float(
                history.iloc[-1]["validation_purity"]
            ),
            "test_purity": cluster_purity(
                test_labels,
                test_predictions,
            ),
            "test_ari": adjusted_rand_index(
                test_labels,
                test_predictions,
            ),
            "test_nmi": normalized_mutual_information(
                test_labels,
                test_predictions,
            ),
            "test_full_nll_per_time": float(
                -test_evaluation.full_marginal_scores.sum() / test_exposure
            ),
            "test_samples": test_samples,
            "test_used_for_objective": False,
            "learned_alpha": float(
                model.interaction_strength().detach().cpu()
            ),
            "mean_abs_omega_minus_identity": float(
                (mean - identity).abs().mean().cpu()
            ),
            "runtime_seconds": time.time() - started,
            "parameters": parameters,
        }
        _json_write(trial_dir / "result.json", result)
        for key, value in result.items():
            if key not in {"parameters", "status"}:
                trial.set_user_attr(key, value)
        return fit.best_validation_purity

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
        catch=(RuntimeError,),
    )
    study.trials_dataframe().to_csv(
        args.outdir / "OPTUNA_TRIALS.csv",
        index=False,
    )
    complete = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    summary: dict[str, object] = {
        "study_name": study.study_name,
        "completed_trials": len(complete),
        "total_trials": len(study.trials),
    }
    if complete:
        summary.update({
            "best_trial": study.best_trial.number,
            "best_validation_purity": study.best_value,
            "best_parameters": study.best_params,
        })
    _json_write(args.outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
