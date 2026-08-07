#!/usr/bin/env python3
"""Validate fixed Wishart-THP candidates across DAN split/model seeds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

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


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="sin_K5_C5")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--evaluation-interval", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.evaluation_interval <= 0
        or args.train_samples <= 0
        or args.validation_samples <= 0
        or args.test_samples <= 0
        or any(seed < 0 for seed in args.seeds)
        or args.split_seed < 0
    ):
        raise ValueError("epochs must be positive and seeds non-negative")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("candidates must be a non-empty JSON object")

    names = dataset_names(args.data_root)
    if args.dataset not in names:
        raise ValueError(f"unknown dataset: {args.dataset}")
    dataset_index = names.index(args.dataset)
    dataset = load_dataset(args.data_root, args.dataset)
    split = shuffled_split(dataset, seed=args.split_seed)
    maximum_events = max(sequence.count for sequence in dataset.sequences)
    batch_size = _dynamic_batch_size(maximum_events)
    evaluation_batch_size = _evaluation_batch_size(maximum_events)
    device = torch.device(args.device)
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
    args.outdir.mkdir(parents=True, exist_ok=True)
    _json_write(args.outdir / "config.json", {
        "dataset": args.dataset,
        "model_seeds": args.seeds,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "candidates": candidates,
        "selection_metric": "validation_purity",
        "tie_breaker": "validation_nll",
        "test_split_evaluated": True,
        "test_used_for_selection": False,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "test_samples": args.test_samples,
        "evaluation_interval": args.evaluation_interval,
    })

    for candidate_name, parameters in candidates.items():
        required = {
            "wishart_nu",
            "omega_learning_rate",
            "alpha_learning_rate",
            "alpha_temperature",
            "mean_hyperprior_strength",
        }
        optional = {
            "exploration_beta",
            "exploration_mode",
            "exploration_degrees_of_freedom",
            "exploration_beta_anneal_epochs",
            "lr_plateau_factor",
            "lr_plateau_patience",
            "lal_lr_decay_factor",
            "lal_lr_decay_tolerance",
            "lal_lr_min",
            "lal_lr_updated",
            "assignment_information_strength",
        }
        if not required.issubset(parameters) or not set(parameters).issubset(
            required | optional
        ):
            raise ValueError(
                f"candidate {candidate_name!r} must define {sorted(required)} "
                f"and may define {sorted(optional)}"
            )
        for seed in args.seeds:
            job_dir = args.outdir / candidate_name / f"seed_{seed}"
            result_path = job_dir / "result.json"
            if result_path.is_file():
                try:
                    previous = json.loads(result_path.read_text())
                    if (
                        previous["status"] == "pass"
                        and "test_purity" in previous
                    ):
                        print(f"SKIP {candidate_name} seed={seed}", flush=True)
                        continue
                except (KeyError, OSError, json.JSONDecodeError):
                    pass
            job_dir.mkdir(parents=True, exist_ok=True)
            train = split.train
            validation = split.validation
            validation_labels = split.validation_labels
            test = split.test
            test_labels = split.test_labels
            test_ids = split.test_ids
            initial_total_rate = sum(
                sequence.count for sequence in train
            ) / sum(sequence.horizon for sequence in train)
            init_seed = (
                2026090100
                + seed * 1_000_000
                + dataset_index * 10_000
                + 1_000
            )
            sample_seed = init_seed + 701
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
                interaction_temperature=float(
                    parameters["alpha_temperature"]
                ),
                exploration_beta=float(
                    parameters.get("exploration_beta", 0.0)
                ),
                exploration_degrees_of_freedom=(
                    int(parameters["exploration_degrees_of_freedom"])
                    if "exploration_degrees_of_freedom" in parameters
                    else dataset.n_components * dataset.n_marks
                ),
                exploration_mode=str(
                    parameters.get("exploration_mode", "additive")
                ),
            ).to(device)
            started = time.time()
            print(f"RUN {candidate_name} seed={seed}", flush=True)
            fit = fit_latent_wishart_attention_nhp(
                model,
                train,
                validation,
                max_epochs=args.epochs,
                batch_size=batch_size,
                train_samples=args.train_samples,
                validation_samples=args.validation_samples,
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
                evaluation_interval=args.evaluation_interval,
                validation_cutoff=None,
                evaluation_batch_size=min(
                    evaluation_batch_size,
                    max(
                        1,
                        evaluation_batch_size
                        * 8
                        // args.validation_samples,
                    ),
                ),
                gradient_clip=20.0,
                batch_seed=init_seed + 301,
                sample_seed=sample_seed,
                validation_labels=validation_labels,
                selection_metric="validation_purity",
                show_progress=True,
                progress_description=f"{candidate_name} s{seed}",
                exploration_beta_anneal_epochs=(
                    int(parameters["exploration_beta_anneal_epochs"])
                    if "exploration_beta_anneal_epochs" in parameters
                    else None
                ),
                validation_exploration_beta=(
                    0.0
                    if "exploration_beta_anneal_epochs" in parameters
                    else None
                ),
                lr_plateau_factor=(
                    float(parameters["lr_plateau_factor"])
                    if "lr_plateau_factor" in parameters
                    else None
                ),
                lr_plateau_patience=int(
                    parameters.get("lr_plateau_patience", 25)
                ),
                lal_lr_decay_factor=(
                    float(parameters["lal_lr_decay_factor"])
                    if "lal_lr_decay_factor" in parameters
                    else None
                ),
                lal_lr_decay_tolerance=int(
                    parameters.get("lal_lr_decay_tolerance", 25)
                ),
                lal_lr_min=(
                    float(parameters.get("lal_lr_min", 0.001))
                    if parameters.get("lal_lr_min", 0.001) is not None
                    else None
                ),
                lal_lr_updated=(
                    float(parameters.get("lal_lr_updated", 0.001))
                    if parameters.get("lal_lr_updated", 0.001) is not None
                    else None
                ),
                assignment_information_strength=float(
                    parameters.get("assignment_information_strength", 0.0)
                ),
            )
            history = pd.DataFrame(fit.history)
            history.to_csv(job_dir / "history.csv", index=False)
            torch.save(fit.model.state_dict(), job_dir / "checkpoint.pt")
            test_evaluation = evaluate_latent_wishart_nhp(
                fit.model,
                test,
                cutoff=None,
                n_samples=args.test_samples,
                sample_seed=sample_seed + 900_000,
                batch_size=min(
                    evaluation_batch_size,
                    max(
                        1,
                        evaluation_batch_size
                        * 8
                        // args.test_samples,
                    ),
                ),
            )
            test_probabilities = (
                test_evaluation.full_cluster_probabilities.numpy()
            )
            test_predictions = test_probabilities.argmax(axis=1)
            test_exposure = sum(sequence.horizon for sequence in test)
            test_nll = float(
                -test_evaluation.full_marginal_scores.sum() / test_exposure
            )
            pd.DataFrame({
                "source_id": test_ids,
                "true_cluster": test_labels,
                "predicted_cluster": test_predictions,
                **{
                    f"probability_{component}": (
                        test_probabilities[:, component]
                    )
                    for component in range(dataset.n_components)
                },
            }).to_csv(job_dir / "test_predictions.csv", index=False)
            mean = fit.model.mean_matrix().detach()
            identity = torch.eye(
                fit.model.dimension,
                device=mean.device,
                dtype=mean.dtype,
            )
            result = {
                "status": "pass",
                "candidate": candidate_name,
                "seed": seed,
                "model_seed": seed,
                "split_seed": args.split_seed,
                "best_validation_purity": fit.best_validation_purity,
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
                "test_full_nll_per_time": test_nll,
                "test_samples": args.test_samples,
                "learned_alpha": float(
                    fit.model.interaction_strength().detach().cpu()
                ),
                "mean_abs_omega_minus_identity": float(
                    (mean - identity).abs().mean().cpu()
                ),
                "runtime_seconds": time.time() - started,
                "parameters": parameters,
                "exploration_beta": float(
                    parameters.get("exploration_beta", 0.0)
                ),
                "final_model_exploration_beta": float(
                    fit.model.exploration_beta
                ),
                "test_split_evaluated": True,
                "test_used_for_selection": False,
            }
            _json_write(result_path, result)
            print(
                f"  best_val_purity={fit.best_validation_purity:.4f} "
                f"epoch={fit.best_epoch} "
                f"test_purity={result['test_purity']:.4f}",
                flush=True,
            )

    rows = []
    for result_path in args.outdir.glob("*/seed_*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "pass":
            rows.append(result)
    frame = pd.DataFrame(rows).sort_values(["candidate", "seed"])
    frame.to_csv(args.outdir / "VALIDATION_RESULTS.csv", index=False)
    aggregate = (
        frame.groupby("candidate", as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            validation_purity_mean=("best_validation_purity", "mean"),
            validation_purity_sd=("best_validation_purity", "std"),
            final_validation_purity_mean=("final_validation_purity", "mean"),
            test_purity_mean=("test_purity", "mean"),
            test_purity_sd=("test_purity", "std"),
            test_purity_min=("test_purity", "min"),
            test_ari_mean=("test_ari", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values("validation_purity_mean", ascending=False)
    )
    aggregate.to_csv(args.outdir / "VALIDATION_AGGREGATE.csv", index=False)
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
