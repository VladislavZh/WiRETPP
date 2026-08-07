#!/usr/bin/env python3
"""Benchmark no-W and signed-Wishart NHP/THP/COTIC on DAN synthetic data."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from lal_wishart.experiment import clustering_row
from lal_wishart.models.signed_wishart import SignedWishartTPP
from lal_wishart.reproduction.dan_synthetic import (
    dataset_names,
    load_dataset,
    shuffled_split,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_direct_nhp_mixture,
    evaluate_latent_wishart_nhp,
    fit_direct_nhp_mixture,
    fit_latent_wishart_attention_nhp,
)
from run_corrected_shared_wishart_architectures import (
    _environment_snapshot,
    _initial_backbone,
    _set_seed,
)


ARCHITECTURES = ("nhp", "thp", "cotic")
VARIANTS = ("no_wishart", "wishart")
ARCHITECTURE_OMEGA_LEARNING_RATES = {
    "nhp_K2": 0.1,
    "nhp_K3_to_K5": 0.01,
    "thp": 0.01,
    "cotic": 3e-5,
}


PUBLISHED_PURITY = {
    "K2_C5": (0.97, 0.04, 1.00, 0.00, 0.65, 0.17, 1.00, 0.00),
    "K3_C5": (0.85, 0.11, 0.92, 0.04, 0.64, 0.07, 0.60, 0.01),
    "K4_C5": (0.90, 0.07, 0.89, 0.14, 0.45, 0.04, 0.67, 0.06),
    "K5_C5": (0.84, 0.09, 0.66, 0.06, 0.46, 0.02, 0.57, 0.04),
    "sin_K2_C5": (0.99, 0.00, 0.93, 0.15, 0.89, 0.03, 0.89, 0.00),
    "sin_K3_C5": (0.99, 0.01, 0.95, 0.02, 0.80, 0.01, 0.82, 0.00),
    "sin_K4_C5": (0.92, 0.06, 0.81, 0.08, 0.62, 0.07, 0.55, 0.00),
    "sin_K5_C5": (0.92, 0.05, 0.70, 0.03, 0.47, 0.01, 0.51, 0.01),
    "trunc_K2_C5": (1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 0.88, 0.17),
    "trunc_K3_C5": (0.96, 0.01, 0.96, 0.01, 0.59, 0.09, 0.61, 0.00),
    "trunc_K4_C5": (0.99, 0.00, 0.97, 0.05, 0.75, 0.07, 0.67, 0.02),
    "trunc_K5_C5": (0.94, 0.06, 0.91, 0.08, 0.64, 0.10, 0.60, 0.02),
}


def _reference_frame() -> pd.DataFrame:
    columns = (
        "dan_moitpp_mean",
        "dan_moitpp_sd",
        "dan_dmhp_mean",
        "dan_dmhp_sd",
        "dan_autoencoder_mean",
        "dan_autoencoder_sd",
        "dan_thp_mean",
        "dan_thp_sd",
    )
    return pd.DataFrame.from_dict(
        PUBLISHED_PURITY, orient="index", columns=columns
    ).rename_axis("dataset").reset_index()


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _dynamic_batch_size(maximum_events: int) -> int:
    if maximum_events <= 100:
        return 64
    if maximum_events <= 250:
        return 24
    if maximum_events <= 500:
        return 12
    return 8


def _evaluation_batch_size(maximum_events: int) -> int:
    if maximum_events <= 100:
        return 32
    if maximum_events <= 250:
        return 16
    if maximum_events <= 500:
        return 8
    return 4


def _architecture_omega_learning_rate(
    architecture: str,
    n_components: int,
) -> float:
    if architecture == "nhp":
        return (
            ARCHITECTURE_OMEGA_LEARNING_RATES["nhp_K2"]
            if n_components == 2
            else ARCHITECTURE_OMEGA_LEARNING_RATES["nhp_K3_to_K5"]
        )
    return ARCHITECTURE_OMEGA_LEARNING_RATES[architecture]


def _write_summary(root: Path) -> None:
    rows = []
    for path in root.glob("seed_*/*/*/*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "pass":
            rows.append(result)
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(
        ["seed", "dataset", "architecture", "variant"]
    )
    reference = _reference_frame()
    comparison = frame.merge(reference, on="dataset", how="left")
    comparison["purity_minus_dan_moitpp"] = (
        comparison["test_purity"] - comparison["dan_moitpp_mean"]
    )
    comparison["purity_minus_dan_thp"] = (
        comparison["test_purity"] - comparison["dan_thp_mean"]
    )
    comparison.to_csv(root / "ALL_RESULTS.csv", index=False)

    aggregate = (
        comparison.groupby(["dataset", "architecture", "variant"], as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            purity_mean=("test_purity", "mean"),
            purity_sd=("test_purity", "std"),
            ari_mean=("test_ari", "mean"),
            ari_sd=("test_ari", "std"),
            nmi_mean=("test_nmi", "mean"),
            nmi_sd=("test_nmi", "std"),
            nll_mean=("test_full_nll_per_time", "mean"),
            nll_sd=("test_full_nll_per_time", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            dan_moitpp_mean=("dan_moitpp_mean", "first"),
            dan_moitpp_sd=("dan_moitpp_sd", "first"),
            dan_thp_mean=("dan_thp_mean", "first"),
            dan_thp_sd=("dan_thp_sd", "first"),
        )
    )
    aggregate["purity_minus_dan_moitpp"] = (
        aggregate["purity_mean"] - aggregate["dan_moitpp_mean"]
    )
    aggregate.to_csv(root / "AGGREGATE_RESULTS.csv", index=False)

    paired = aggregate.pivot(
        index=["dataset", "architecture"],
        columns="variant",
        values=["purity_mean", "ari_mean", "nll_mean"],
    )
    paired.columns = ["_".join(column) for column in paired.columns]
    paired = paired.reset_index()
    paired_columns = {
        "purity_mean_wishart",
        "purity_mean_no_wishart",
        "ari_mean_wishart",
        "ari_mean_no_wishart",
        "nll_mean_wishart",
        "nll_mean_no_wishart",
    }
    if paired_columns.issubset(paired.columns):
        paired["wishart_minus_no_wishart_purity"] = (
            paired["purity_mean_wishart"]
            - paired["purity_mean_no_wishart"]
        )
        paired["wishart_minus_no_wishart_ari"] = (
            paired["ari_mean_wishart"] - paired["ari_mean_no_wishart"]
        )
        paired["wishart_minus_no_wishart_nll"] = (
            paired["nll_mean_wishart"] - paired["nll_mean_no_wishart"]
        )
        paired.to_csv(root / "PAIRED_WISHART_EFFECTS.csv", index=False)

    architecture_summary = (
        frame.groupby(["architecture", "variant"], as_index=False)
        .agg(
            datasets_and_seeds=("dataset", "count"),
            purity_mean=("test_purity", "mean"),
            ari_mean=("test_ari", "mean"),
            nll_mean=("test_full_nll_per_time", "mean"),
        )
    )
    architecture_summary.to_csv(
        root / "ARCHITECTURE_SUMMARY.csv",
        index=False,
    )

    completed = len(frame)
    expected = int(json.loads((root / "benchmark_config.json").read_text())["expected_jobs"])
    best = aggregate.sort_values(
        ["dataset", "purity_mean"], ascending=[True, False]
    ).groupby("dataset", as_index=False).first()
    lines = [
        "# DAN synthetic WiRE-TPP benchmark",
        "",
        f"Completed jobs: {completed}/{expected}.",
        "",
        "Original timestamps are preserved. Plain K*_C5 paths end at their "
        "fiftieth (last observed) event; sin/trunc paths retain the fixed "
        "observation horizon 20. No sequence-dependent time warping is used.",
        "",
        "Mean over completed datasets and seeds:",
        "",
        "| architecture | variant | purity | ARI | NLL / time |",
        "|---|---|---:|---:|---:|",
    ]
    for row in architecture_summary.itertuples(index=False):
        lines.append(
            f"| {row.architecture} | {row.variant} | "
            f"{row.purity_mean:.4f} | {row.ari_mean:.4f} | "
            f"{row.nll_mean:.4f} |"
        )
    lines.extend([
        "",
        "Best completed WiRE-TPP configuration per dataset:",
        "",
        "| dataset | architecture | variant | purity | DAN MoITPP | delta |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.architecture} | {row.variant} | "
            f"{row.purity_mean:.4f} | {row.dan_moitpp_mean:.4f} | "
            f"{row.purity_minus_dan_moitpp:+.4f} |"
        )
    best_vs_moitpp = int(
        (best["purity_mean"] >= best["dan_moitpp_mean"]).sum()
    )
    best_vs_thp = int(
        (best["purity_mean"] >= best["dan_thp_mean"]).sum()
    )
    lines.extend([
        "",
        f"Best configuration reaches or exceeds DAN MoITPP mean on "
        f"{best_vs_moitpp}/{len(best)} datasets and DAN THP mean on "
        f"{best_vs_thp}/{len(best)} datasets.",
        "",
        "Paired Wishart-minus-no-W effects are in "
        "`PAIRED_WISHART_EFFECTS.csv`.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _job_result_exists(job_dir: Path) -> bool:
    path = job_dir / "result.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    except (OSError, json.JSONDecodeError):
        return False


def _run_job(
    *,
    root: Path,
    dataset,
    split,
    seed: int,
    split_seed: int,
    dataset_index: int,
    architecture: str,
    variant: str,
    device: torch.device,
    epochs: int,
    quick: bool,
    omega_learning_rate: float,
    alpha_learning_rate: float,
    alpha_temperature: float,
    mean_hyperprior_strength: float,
    wishart_degrees_of_freedom: int | None,
    wishart_nu_multiplier: float,
    train_samples: int,
    validation_samples: int,
    wishart_test_samples: int,
    exploration_beta: float,
    exploration_degrees_of_freedom: int,
    exploration_mode: str,
    exploration_beta_anneal_epochs: int | None,
    exploration_beta_schedule: str,
    exploration_beta_decay_rate: float,
    assignment_balance_strength: float,
    assignment_balance_temperature: float,
    assignment_balance_iterations: int,
    assignment_ot_marginal_penalty: float,
    assignment_ot_temperature: float,
    assignment_ot_dual_learning_rate: float,
    gradient_clip: float,
    optimization_objective_normalization: str,
    gradient_accumulation_steps: int,
    cotic_num_layers: int,
    cotic_dilation_factor: float,
    selection_metric: str,
    evaluation_interval_override: int | None,
) -> dict[str, object]:
    job_dir = root / f"seed_{seed}" / dataset.name / architecture / variant
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    maximum_events = max(sequence.count for sequence in dataset.sequences)
    batch_size = _dynamic_batch_size(maximum_events)
    evaluation_batch_size = _evaluation_batch_size(maximum_events)
    train = split.train
    train_labels = split.train_labels
    train_ids = split.train_ids
    validation = split.validation
    validation_labels = split.validation_labels
    test = split.test
    test_labels = split.test_labels
    test_ids = split.test_ids
    if quick:
        train = train[: min(32, len(train))]
        train_labels = train_labels[: len(train)]
        train_ids = train_ids[: len(train)]
        validation = validation[: min(16, len(validation))]
        validation_labels = validation_labels[: len(validation)]
        test = test[: min(16, len(test))]
        test_labels = test_labels[: len(test)]
        test_ids = test_ids[: len(test)]
        batch_size = min(batch_size, 8)
        evaluation_batch_size = min(evaluation_batch_size, 2)

    optimization_objective_scale = (
        float(np.mean([sequence.count for sequence in train]))
        if optimization_objective_normalization == "mean_train_events"
        else 1.0
    )

    architecture_index = ARCHITECTURES.index(architecture)
    init_seed = 2026090100 + seed * 1_000_000 + dataset_index * 10_000 + architecture_index * 1_000
    sample_seed = init_seed + 701
    initial_total_rate = sum(sequence.count for sequence in train) / sum(
        sequence.horizon for sequence in train
    )
    model_horizon = max(sequence.horizon for sequence in dataset.sequences)
    config: dict[str, object] = {
        "n_components": dataset.n_components,
        "n_marks": dataset.n_marks,
        "horizon": model_horizon,
        "quadrature_order": 4,
        "nhp_hidden_size": 16,
        "thp_hidden_size": 32,
        "thp_num_layers": 2,
        "thp_num_heads": 4,
        "cotic_input_channels": 32,
        "cotic_hidden_size": 64,
        "cotic_num_layers": cotic_num_layers,
        "cotic_kernel_size": 3,
        "cotic_dilation_factor": cotic_dilation_factor,
        "dropout": 0.1,
    }
    _set_seed(init_seed)
    initial = _initial_backbone(
        architecture,
        "output_split",
        config,
        initial_total_rate=initial_total_rate,
        initialization_seed=init_seed,
        device=device,
    )
    model_epochs = 2 if quick else epochs
    evaluation_interval = (
        evaluation_interval_override
        if evaluation_interval_override is not None
        else 1
    )
    cutoff = None
    if variant == "no_wishart":
        fit = fit_direct_nhp_mixture(
            copy.deepcopy(initial),
            train,
            validation,
            max_epochs=model_epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            neural_weight_decay=1e-5,
            evaluation_interval=evaluation_interval,
            validation_cutoff=cutoff,
            evaluation_batch_size=evaluation_batch_size,
            gradient_clip=gradient_clip,
            optimization_objective_scale=optimization_objective_scale,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_seed=init_seed + 301,
            validation_labels=validation_labels,
            selection_metric=selection_metric,
            show_progress=True,
            progress_description=(
                f"{dataset.name} s{seed} {architecture.upper()} no-W"
            ),
            assignment_balance_strength=assignment_balance_strength,
            assignment_balance_temperature=assignment_balance_temperature,
            assignment_balance_iterations=assignment_balance_iterations,
            assignment_ot_marginal_penalty=(
                assignment_ot_marginal_penalty
            ),
            assignment_ot_temperature=assignment_ot_temperature,
            assignment_ot_dual_learning_rate=(
                assignment_ot_dual_learning_rate
            ),
        )
        model = fit.model
        evaluation = evaluate_direct_nhp_mixture(
            model,
            test,
            cutoff=cutoff,
            batch_size=evaluation_batch_size,
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        train_evaluation = evaluate_direct_nhp_mixture(
            model,
            train,
            cutoff=cutoff,
            batch_size=evaluation_batch_size,
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        alpha = float("nan")
        mean_distance = float("nan")
        test_samples = 0
    else:
        wishart_nu = (
            int(wishart_degrees_of_freedom)
            if wishart_degrees_of_freedom is not None
            else int(
                wishart_nu_multiplier
                * dataset.n_components
                * dataset.n_marks
            )
        )
        model = SignedWishartTPP(
            copy.deepcopy(initial),
            degrees_of_freedom=wishart_nu,
            alpha_max=1.0,
            initial_alpha=0.5,
            interaction_temperature=alpha_temperature,
            exploration_beta=exploration_beta,
            exploration_degrees_of_freedom=(
                exploration_degrees_of_freedom
            ),
            exploration_mode=exploration_mode,
        ).to(device)
        fit = fit_latent_wishart_attention_nhp(
            model,
            train,
            validation,
            max_epochs=model_epochs,
            batch_size=batch_size,
            train_samples=(1 if quick else train_samples),
            validation_samples=(2 if quick else validation_samples),
            neural_learning_rate=1e-3,
            distribution_learning_rate=omega_learning_rate,
            interaction_learning_rate=alpha_learning_rate,
            neural_weight_decay=1e-5,
            mean_hyperprior_strength=mean_hyperprior_strength,
            evaluation_interval=evaluation_interval,
            validation_cutoff=cutoff,
            evaluation_batch_size=evaluation_batch_size,
            gradient_clip=gradient_clip,
            optimization_objective_scale=optimization_objective_scale,
            batch_seed=init_seed + 301,
            sample_seed=sample_seed,
            validation_labels=validation_labels,
            selection_metric=selection_metric,
            show_progress=True,
            progress_description=(
                f"{dataset.name} s{seed} {architecture.upper()} Wishart"
            ),
            exploration_beta_anneal_epochs=(
                exploration_beta_anneal_epochs
            ),
            exploration_beta_schedule=exploration_beta_schedule,
            exploration_beta_decay_rate=exploration_beta_decay_rate,
            validation_exploration_beta=(
                0.0
                if exploration_beta_anneal_epochs is not None
                else None
            ),
            assignment_balance_strength=assignment_balance_strength,
            assignment_balance_temperature=assignment_balance_temperature,
            assignment_balance_iterations=assignment_balance_iterations,
            assignment_ot_marginal_penalty=(
                assignment_ot_marginal_penalty
            ),
            assignment_ot_temperature=assignment_ot_temperature,
            assignment_ot_dual_learning_rate=(
                assignment_ot_dual_learning_rate
            ),
        )
        model = fit.model
        test_samples = 4 if quick else wishart_test_samples
        evaluation = evaluate_latent_wishart_nhp(
            model,
            test,
            cutoff=cutoff,
            n_samples=test_samples,
            sample_seed=sample_seed + 900_000,
            batch_size=evaluation_batch_size,
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        train_evaluation = evaluate_latent_wishart_nhp(
            model,
            train,
            cutoff=cutoff,
            n_samples=test_samples,
            sample_seed=sample_seed + 800_000,
            batch_size=evaluation_batch_size,
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        alpha = float(model.interaction_strength().detach().cpu())
        identity = torch.eye(model.dimension, device=model.device, dtype=model.dtype)
        mean_distance = float((model.mean_matrix() - identity).abs().mean().detach().cpu())

    probability_tensor = evaluation.full_cluster_probabilities
    cluster, prediction = clustering_row(
        test_labels,
        probability_tensor,
        model=architecture,
        degrees_of_freedom=(
            wishart_nu if variant == "wishart" else None
        ),
        representation=variant,
    )
    probabilities = probability_tensor.numpy()
    train_probability_tensor = train_evaluation.full_cluster_probabilities
    train_probabilities = train_probability_tensor.numpy()
    train_predicted_components = train_probability_tensor.argmax(dim=1).numpy()
    train_soft_marginal = train_probability_tensor.mean(dim=0)
    train_hard_counts = torch.bincount(
        torch.from_numpy(train_predicted_components),
        minlength=dataset.n_components,
    )
    train_hard_marginal = train_hard_counts.to(
        dtype=train_soft_marginal.dtype
    ) / len(train)
    target_marginal = torch.full_like(
        train_soft_marginal,
        1.0 / dataset.n_components,
    )
    if fit.assignment_dual is not None:
        assignment_dual = fit.assignment_dual.to(
            dtype=train_soft_marginal.dtype
        )
        dual_reference_marginal = torch.softmax(
            torch.log(target_marginal)
            + assignment_dual / assignment_ot_marginal_penalty,
            dim=0,
        )
        full_train_stationarity_error = float(
            (train_soft_marginal - dual_reference_marginal).abs().max()
        )
    else:
        dual_reference_marginal = None
        full_train_stationarity_error = float("nan")
    full_exposure = sum(sequence.horizon for sequence in test)
    full_nll = float(-evaluation.full_marginal_scores.sum() / full_exposure)
    gradient_norms = np.asarray(
        [float(row["gradient_norm"]) for row in fit.history],
        dtype=np.float64,
    )
    gradient_clip_coefficients = np.minimum(
        1.0,
        gradient_clip / np.maximum(gradient_norms, 1e-30),
    )
    pd.DataFrame(fit.history).to_csv(job_dir / "history.csv", index=False)
    torch.save(model.state_dict(), job_dir / "checkpoint.pt")
    if fit.assignment_dual is not None:
        np.save(
            job_dir / "assignment_dual.npy",
            fit.assignment_dual.numpy(),
        )
    pd.DataFrame({
        "source_id": test_ids,
        "true_cluster": test_labels,
        "predicted_cluster": prediction,
        **{
            f"probability_{component}": probabilities[:, component]
            for component in range(dataset.n_components)
        },
    }).to_csv(job_dir / "predictions.csv", index=False)
    pd.DataFrame({
        "source_id": train_ids,
        "true_cluster": train_labels,
        "predicted_component": train_predicted_components,
        **{
            f"probability_{component}": train_probabilities[:, component]
            for component in range(dataset.n_components)
        },
    }).to_csv(job_dir / "train_predictions.csv", index=False)
    if variant == "wishart":
        np.save(job_dir / "learned_mean.npy", model.mean_matrix().detach().cpu().numpy())

    result: dict[str, object] = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.name,
        "seed": seed,
        "model_seed": seed,
        "initialization_seed": init_seed,
        "wishart_sample_seed": sample_seed,
        "split_seed": split_seed,
        "architecture": architecture,
        "variant": variant,
        "n_components": dataset.n_components,
        "n_marks": dataset.n_marks,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "epochs": model_epochs,
        "best_epoch": fit.best_epoch,
        "selection_metric": fit.selection_metric,
        "best_validation_nll_per_time": (
            fit.best_validation_suffix_nll_per_exposure
        ),
        "best_validation_purity": fit.best_validation_purity,
        "batch_size": batch_size,
        "maximum_events": maximum_events,
        "evaluation_interval": evaluation_interval,
        "gradient_clip": gradient_clip,
        "optimization_objective_normalization": (
            optimization_objective_normalization
        ),
        "optimization_objective_scale": optimization_objective_scale,
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
            if variant == "no_wishart"
            else 1
        ),
        "effective_batch_size": (
            batch_size * gradient_accumulation_steps
            if variant == "no_wishart"
            else batch_size
        ),
        "gradient_clipping_fraction": float(
            np.mean(gradient_norms > gradient_clip)
        ),
        "gradient_norm_median": float(np.median(gradient_norms)),
        "gradient_norm_maximum": float(np.max(gradient_norms)),
        "gradient_clip_coefficient_median": float(
            np.median(gradient_clip_coefficients)
        ),
        "gradient_clip_coefficient_minimum": float(
            np.min(gradient_clip_coefficients)
        ),
        "test_samples": test_samples,
        "test_full_nll_per_time": full_nll,
        "test_purity": cluster["purity"],
        "test_ari": cluster["ari"],
        "test_nmi": cluster["nmi"],
        "test_active_clusters": cluster["active_k"],
        "train_true_cluster_counts": np.bincount(
            train_labels,
            minlength=dataset.n_components,
        ).tolist(),
        "train_predicted_component_counts": train_hard_counts.tolist(),
        "train_soft_assignment_marginal": train_soft_marginal.tolist(),
        "train_hard_assignment_marginal": train_hard_marginal.tolist(),
        "train_soft_marginal_l1_to_uniform": float(
            (train_soft_marginal - target_marginal).abs().sum()
        ),
        "train_hard_marginal_l1_to_uniform": float(
            (train_hard_marginal - target_marginal).abs().sum()
        ),
        "assignment_ot_reference_marginal": (
            dual_reference_marginal.tolist()
            if dual_reference_marginal is not None
            else None
        ),
        "assignment_ot_full_train_stationarity_error": (
            full_train_stationarity_error
        ),
        "alpha": alpha,
        "mean_abs_omega_minus_identity": mean_distance,
        "runtime_seconds": time.time() - started,
        "split_protocol": (
            f"numpy.RandomState({split_seed}).shuffle; 80/10/10"
        ),
        "time_axis": dataset.source_horizon,
        "minimum_sequence_horizon": min(sequence.horizon for sequence in dataset.sequences),
        "maximum_sequence_horizon": model_horizon,
        "cotic_num_layers": (
            cotic_num_layers if architecture == "cotic" else None
        ),
        "cotic_dilation_factor": (
            cotic_dilation_factor if architecture == "cotic" else None
        ),
        "cotic_receptive_field": (
            1
            + 2
            * sum(
                int(cotic_dilation_factor**index)
                for index in range(cotic_num_layers)
            )
            if architecture == "cotic"
            else None
        ),
        "wishart_nu": (
            wishart_nu if variant == "wishart" else None
        ),
        "wishart_nu_multiplier": (
            wishart_nu_multiplier
            if variant == "wishart"
            and wishart_degrees_of_freedom is None
            else None
        ),
        "alpha_initial": 0.5 if variant == "wishart" else None,
        "alpha_temperature": (
            alpha_temperature if variant == "wishart" else None
        ),
        "omega_learning_rate": (
            omega_learning_rate if variant == "wishart" else None
        ),
        "alpha_learning_rate": (
            alpha_learning_rate if variant == "wishart" else None
        ),
        "mean_hyperprior_strength": (
            mean_hyperprior_strength if variant == "wishart" else None
        ),
        "train_samples": train_samples if variant == "wishart" else 0,
        "validation_samples": (
            validation_samples if variant == "wishart" else 0
        ),
        "exploration_beta_initial": (
            exploration_beta if variant == "wishart" else None
        ),
        "exploration_degrees_of_freedom": (
            exploration_degrees_of_freedom
            if variant == "wishart"
            else None
        ),
        "exploration_mode": (
            exploration_mode if variant == "wishart" else None
        ),
        "exploration_beta_anneal_epochs": (
            exploration_beta_anneal_epochs
            if variant == "wishart"
            else None
        ),
        "exploration_beta_schedule": (
            exploration_beta_schedule if variant == "wishart" else None
        ),
        "exploration_beta_decay_rate": (
            exploration_beta_decay_rate if variant == "wishart" else None
        ),
        "final_exploration_beta": (
            float(model.exploration_beta)
            if variant == "wishart"
            else None
        ),
        "assignment_balance_method": (
            "sinkhorn_kl" if assignment_balance_strength > 0.0 else None
        ),
        "assignment_balance_strength": assignment_balance_strength,
        "assignment_balance_temperature": assignment_balance_temperature,
        "assignment_balance_iterations": assignment_balance_iterations,
        "assignment_objective": fit.assignment_objective,
        "assignment_ot_marginal_penalty": (
            assignment_ot_marginal_penalty
        ),
        "assignment_ot_temperature": assignment_ot_temperature,
        "assignment_ot_dual_learning_rate": (
            assignment_ot_dual_learning_rate
        ),
        "assignment_ot_dual": (
            fit.assignment_dual.tolist()
            if fit.assignment_dual is not None
            else None
        ),
    }
    stale_failure = job_dir / "failure.json"
    if stale_failure.is_file():
        stale_failure.unlink()
    _json_write(job_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/dan_synthetic_benchmark"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--omega-learning-rate", type=float)
    parser.add_argument("--alpha-learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha-temperature", type=float, default=0.5)
    parser.add_argument("--mean-hyperprior-strength", type=float, default=1.0)
    parser.add_argument("--wishart-degrees-of-freedom", type=int)
    parser.add_argument("--wishart-nu-multiplier", type=float, default=1.5)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--exploration-beta", type=float, default=0.0)
    parser.add_argument(
        "--exploration-degrees-of-freedom", type=int, default=200
    )
    parser.add_argument(
        "--exploration-mode",
        choices=("additive", "convex"),
        default="additive",
    )
    parser.add_argument("--exploration-beta-anneal-epochs", type=int)
    parser.add_argument(
        "--exploration-beta-schedule",
        choices=("linear", "exponential"),
        default="linear",
    )
    parser.add_argument(
        "--exploration-beta-decay-rate", type=float, default=5.0
    )
    parser.add_argument(
        "--assignment-balance-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--assignment-balance-temperature", type=float, default=0.1
    )
    parser.add_argument(
        "--assignment-balance-iterations", type=int, default=200
    )
    parser.add_argument(
        "--assignment-ot-marginal-penalty", type=float, default=0.0,
        help=(
            "Rho in the global unbalanced-OT train objective; zero disables "
            "the strict OT objective."
        ),
    )
    parser.add_argument(
        "--assignment-ot-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--assignment-ot-dual-learning-rate", type=float, default=0.05
    )
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument(
        "--optimization-objective-normalization",
        choices=("none", "mean_train_events"),
        default="none",
        help=(
            "Divide the complete train objective by one dataset-level "
            "constant; this does not normalize individual sequences."
        ),
    )
    parser.add_argument("--cotic-num-layers", type=int, default=4)
    parser.add_argument("--cotic-dilation-factor", type=float, default=1.29)
    parser.add_argument(
        "--selection-metric",
        choices=("validation_nll", "validation_purity"),
        default="validation_nll",
    )
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        help="Stop cleanly after this many previously unfinished jobs.",
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or any(seed < 0 for seed in args.seeds)
        or args.split_seed < 0
        or (
            args.omega_learning_rate is not None
            and args.omega_learning_rate <= 0.0
        )
        or args.alpha_learning_rate <= 0.0
        or args.alpha_temperature <= 0.0
        or args.mean_hyperprior_strength < 0.0
        or (
            args.wishart_degrees_of_freedom is not None
            and args.wishart_degrees_of_freedom <= 0
        )
        or args.wishart_nu_multiplier < 1.0
        or args.train_samples <= 0
        or args.validation_samples <= 0
        or args.test_samples <= 0
        or args.exploration_beta < 0.0
        or (
            args.exploration_mode == "convex"
            and args.exploration_beta > 1.0
        )
        or args.exploration_degrees_of_freedom <= 0
        or (
            args.exploration_beta_anneal_epochs is not None
            and args.exploration_beta_anneal_epochs <= 1
        )
        or args.exploration_beta_decay_rate <= 0.0
        or args.assignment_balance_strength < 0.0
        or args.assignment_balance_temperature <= 0.0
        or args.assignment_balance_iterations <= 0
        or args.assignment_ot_marginal_penalty < 0.0
        or args.assignment_ot_temperature <= 0.0
        or args.assignment_ot_dual_learning_rate <= 0.0
        or args.gradient_clip <= 0.0
        or args.gradient_accumulation_steps <= 0
        or (
            args.assignment_balance_strength > 0.0
            and args.assignment_ot_marginal_penalty > 0.0
        )
        or args.cotic_num_layers <= 0
        or args.cotic_dilation_factor < 1.0
        or (args.max_new_jobs is not None and args.max_new_jobs <= 0)
        or (
            args.evaluation_interval is not None
            and args.evaluation_interval <= 0
        )
    ):
        raise ValueError("epochs must be positive and seeds non-negative")

    available = dataset_names(args.data_root)
    selected = tuple(args.datasets) if args.datasets else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    root = args.outdir
    root.mkdir(parents=True, exist_ok=True)
    expected_jobs = len(selected) * len(args.architectures) * len(args.variants) * len(args.seeds)
    benchmark_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": selected,
        "architectures": args.architectures,
        "variants": args.variants,
        "seeds": args.seeds,
        "epochs": 2 if args.quick else args.epochs,
        "quick": args.quick,
        "expected_jobs": expected_jobs,
        "split": (
            f"numpy.RandomState({args.split_seed}).shuffle; 80/10/10; "
            "fixed across model seeds"
        ),
        "split_seed": args.split_seed,
        "model_seeds": args.seeds,
        "time_normalization": "none",
        "selection_metric": args.selection_metric,
        "evaluation_interval": args.evaluation_interval,
        "gradient_clip": args.gradient_clip,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimization_objective_normalization": (
            args.optimization_objective_normalization
        ),
        "plain_likelihood_horizon": "time of the fiftieth (last observed) event",
        "fixed_horizon_families": ["sin", "trunc"],
        "architecture": {
            "cotic_num_layers": args.cotic_num_layers,
            "cotic_dilation_factor": args.cotic_dilation_factor,
            "cotic_receptive_field": (
                1
                + 2
                * sum(
                    int(args.cotic_dilation_factor**index)
                    for index in range(args.cotic_num_layers)
                )
            ),
        },
        "assignment_balance": {
            "method": (
                "sinkhorn_kl"
                if args.assignment_balance_strength > 0.0
                else None
            ),
            "strength": args.assignment_balance_strength,
            "temperature": args.assignment_balance_temperature,
            "iterations": args.assignment_balance_iterations,
            "target_cluster_marginal": "uniform",
            "uses_labels": False,
        },
        "assignment_ot": {
            "method": (
                "global_unbalanced_ot_dual"
                if args.assignment_ot_marginal_penalty > 0.0
                else None
            ),
            "temperature": args.assignment_ot_temperature,
            "marginal_penalty_rho": (
                args.assignment_ot_marginal_penalty
            ),
            "dual_learning_rate": args.assignment_ot_dual_learning_rate,
            "target_train_marginal": "uniform",
            "minibatch_is_hard_balanced": False,
            "uses_labels": False,
        },
        "wishart": {
            "nu": (
                args.wishart_degrees_of_freedom
                if args.wishart_degrees_of_freedom is not None
                else f"{args.wishart_nu_multiplier} * K*C"
            ),
            "initial_alpha": 0.5,
            "alpha_temperature": args.alpha_temperature,
            "omega_learning_rate": (
                args.omega_learning_rate
                if args.omega_learning_rate is not None
                else ARCHITECTURE_OMEGA_LEARNING_RATES
            ),
            "alpha_learning_rate": args.alpha_learning_rate,
            "mean_hyperprior_strength": args.mean_hyperprior_strength,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "exploration_beta": args.exploration_beta,
            "exploration_degrees_of_freedom": (
                args.exploration_degrees_of_freedom
            ),
            "exploration_mode": args.exploration_mode,
            "exploration_beta_anneal_epochs": (
                args.exploration_beta_anneal_epochs
            ),
            "exploration_beta_schedule": args.exploration_beta_schedule,
            "exploration_beta_decay_rate": (
                args.exploration_beta_decay_rate
            ),
            "validation_exploration_beta": (
                0.0
                if args.exploration_beta_anneal_epochs is not None
                else None
            ),
        },
    }
    _json_write(root / "benchmark_config.json", benchmark_config)
    _json_write(root / "environment.json", _environment_snapshot(torch.device(args.device)))
    _reference_frame().to_csv(root / "PUBLISHED_DAN_PURITY.csv", index=False)

    data_audits = []
    split_audits = []
    failures = []
    total_index = 0
    new_jobs_started = 0
    stop_requested = False
    canonical_dataset_indices = {
        name: index for index, name in enumerate(available)
    }
    for name in selected:
        dataset_index = canonical_dataset_indices[name]
        print(f"Loading {name}", flush=True)
        dataset = load_dataset(args.data_root, name)
        counts = np.asarray([sequence.count for sequence in dataset.sequences])
        data_audits.append({
            "dataset": name,
            "n_components": dataset.n_components,
            "n_sequences": len(dataset.sequences),
            "mean_events": float(counts.mean()),
            "minimum_events": int(counts.min()),
            "maximum_events": int(counts.max()),
            "source_horizon": dataset.source_horizon,
            "minimum_sequence_horizon": float(
                min(sequence.horizon for sequence in dataset.sequences)
            ),
            "maximum_sequence_horizon": float(
                max(sequence.horizon for sequence in dataset.sequences)
            ),
        })
        for seed in args.seeds:
            split = shuffled_split(dataset, seed=args.split_seed)
            audit = {
                "dataset": name,
                "model_seed": seed,
                "split_seed": args.split_seed,
            }
            for split_name, labels in (
                ("train", split.train_labels),
                ("validation", split.validation_labels),
                ("test", split.test_labels),
            ):
                audit[f"{split_name}_size"] = len(labels)
                for component in range(dataset.n_components):
                    audit[f"{split_name}_cluster_{component}"] = int(
                        np.sum(labels == component)
                    )
            split_audits.append(audit)
            for architecture in args.architectures:
                for variant in args.variants:
                    if (
                        args.max_new_jobs is not None
                        and new_jobs_started >= args.max_new_jobs
                    ):
                        stop_requested = True
                        break
                    total_index += 1
                    job_dir = root / f"seed_{seed}" / name / architecture / variant
                    if not args.force and _job_result_exists(job_dir):
                        print(f"[{total_index}/{expected_jobs}] SKIP {name} {architecture} {variant}", flush=True)
                        continue
                    new_jobs_started += 1
                    print(f"[{total_index}/{expected_jobs}] RUN {name} {architecture} {variant}", flush=True)
                    try:
                        omega_learning_rate = (
                            args.omega_learning_rate
                            if args.omega_learning_rate is not None
                            else _architecture_omega_learning_rate(
                                architecture,
                                dataset.n_components,
                            )
                        )
                        result = _run_job(
                            root=root,
                            dataset=dataset,
                            split=split,
                            seed=seed,
                            split_seed=args.split_seed,
                            dataset_index=dataset_index,
                            architecture=architecture,
                            variant=variant,
                            device=torch.device(args.device),
                            epochs=args.epochs,
                            quick=args.quick,
                            omega_learning_rate=omega_learning_rate,
                            alpha_learning_rate=args.alpha_learning_rate,
                            alpha_temperature=args.alpha_temperature,
                            mean_hyperprior_strength=(
                                args.mean_hyperprior_strength
                            ),
                            wishart_degrees_of_freedom=(
                                args.wishart_degrees_of_freedom
                            ),
                            wishart_nu_multiplier=args.wishart_nu_multiplier,
                            train_samples=args.train_samples,
                            validation_samples=args.validation_samples,
                            wishart_test_samples=args.test_samples,
                            exploration_beta=args.exploration_beta,
                            exploration_degrees_of_freedom=(
                                args.exploration_degrees_of_freedom
                            ),
                            exploration_mode=args.exploration_mode,
                            exploration_beta_anneal_epochs=(
                                args.exploration_beta_anneal_epochs
                            ),
                            exploration_beta_schedule=(
                                args.exploration_beta_schedule
                            ),
                            exploration_beta_decay_rate=(
                                args.exploration_beta_decay_rate
                            ),
                            assignment_balance_strength=(
                                args.assignment_balance_strength
                            ),
                            assignment_balance_temperature=(
                                args.assignment_balance_temperature
                            ),
                            assignment_balance_iterations=(
                                args.assignment_balance_iterations
                            ),
                            assignment_ot_marginal_penalty=(
                                args.assignment_ot_marginal_penalty
                            ),
                            assignment_ot_temperature=(
                                args.assignment_ot_temperature
                            ),
                            assignment_ot_dual_learning_rate=(
                                args.assignment_ot_dual_learning_rate
                            ),
                            gradient_clip=args.gradient_clip,
                            optimization_objective_normalization=(
                                args.optimization_objective_normalization
                            ),
                            gradient_accumulation_steps=(
                                args.gradient_accumulation_steps
                            ),
                            cotic_num_layers=args.cotic_num_layers,
                            cotic_dilation_factor=(
                                args.cotic_dilation_factor
                            ),
                            selection_metric=args.selection_metric,
                            evaluation_interval_override=(
                                args.evaluation_interval
                            ),
                        )
                        print(
                            f"  purity={result['test_purity']:.4f} "
                            f"ARI={result['test_ari']:.4f} "
                            f"NLL={result['test_full_nll_per_time']:.4f}",
                            flush=True,
                        )
                    except Exception as error:
                        job_dir.mkdir(parents=True, exist_ok=True)
                        failure = {
                            "status": "failed",
                            "dataset": name,
                            "seed": seed,
                            "architecture": architecture,
                            "variant": variant,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                        _json_write(job_dir / "failure.json", failure)
                        failures.append(failure)
                        print(f"  FAILED: {error!r}", flush=True)
                    _write_summary(root)
                if stop_requested:
                    break
            if stop_requested:
                break
        if stop_requested:
            break

    pd.DataFrame(data_audits).to_csv(root / "DATA_AUDIT.csv", index=False)
    pd.DataFrame(split_audits).to_csv(root / "SPLIT_AUDIT.csv", index=False)
    _write_summary(root)
    completed_jobs = sum(
        1
        for path in root.glob("seed_*/*/*/*/result.json")
        if _job_result_exists(path.parent)
    )
    final = {
        "status": (
            "failed"
            if failures
            else "pass" if completed_jobs == expected_jobs else "partial"
        ),
        "expected_jobs": expected_jobs,
        "completed_jobs": completed_jobs,
        "failures": failures,
    }
    _json_write(root / "result.json", final)
    if failures:
        raise SystemExit(f"{len(failures)} benchmark jobs failed")


if __name__ == "__main__":
    main()
