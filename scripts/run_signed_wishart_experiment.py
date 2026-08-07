#!/usr/bin/env python3
"""Run the signed WiRE-TPP experiment from dissertation Chapter 3."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from lal_wishart.experiment import (
    clustering_row,
    make_splits,
    make_signed_splits,
    markdown_table,
    save_mean_matrix,
    save_probabilities,
    write_manifest,
)
from lal_wishart.models.signed_wishart import (
    LegacyRowWishartTPP,
    SignedWishartTPP,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    signed_correlation_coupling,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_direct_nhp_mixture,
    evaluate_latent_wishart_nhp,
    fit_direct_nhp_mixture,
    fit_latent_wishart_attention_nhp,
)
from run_corrected_shared_wishart_architectures import (
    _architecture_audit,
    _environment_snapshot,
    _initial_backbone,
    _nll_and_cluster_rows,
    _parameter_count,
    _set_seed,
    sequences,
)


VARIANTS = (
    "legacy_row",
    "signed",
    "signed_alpha_zero",
    "signed_deterministic",
)


def _wrapper(variant: str, backbone, config: dict[str, object]):
    common = {
        "degrees_of_freedom": int(config["degrees_of_freedom"]),
    }
    if variant == "legacy_row":
        return LegacyRowWishartTPP(backbone, **common)
    if variant == "signed":
        return SignedWishartTPP(
            backbone,
            alpha_max=float(config["alpha_max"]),
            initial_alpha=float(config["initial_alpha"]),
            interaction_temperature=float(config.get("interaction_temperature", 1.0)),
            **common,
        )
    if variant == "signed_alpha_zero":
        return SignedWishartTPP(
            backbone,
            alpha_max=float(config["alpha_max"]),
            interaction_temperature=float(config.get("interaction_temperature", 1.0)),
            fixed_alpha=0.0,
            **common,
        )
    if variant == "signed_deterministic":
        return SignedWishartTPP(
            backbone,
            alpha_max=float(config["alpha_max"]),
            initial_alpha=float(config["initial_alpha"]),
            interaction_temperature=float(config.get("interaction_temperature", 1.0)),
            deterministic_matrices=True,
            **common,
        )
    raise ValueError(f"unknown variant: {variant}")


def _alpha(model) -> float:
    if hasattr(model, "interaction_strength"):
        return float(model.interaction_strength().detach().cpu())
    return float("nan")


def _audit_rows(
    *,
    model,
    architecture: str,
    variant: str,
    test_sequences,
    labels: np.ndarray,
    cutoff: float,
    suffix_exposure: float,
    full_exposure: float,
    n_samples: int,
    repeats: int,
    sample_seed: int,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        evaluation = evaluate_latent_wishart_nhp(
            model,
            test_sequences,
            cutoff=cutoff,
            n_samples=n_samples,
            sample_seed=sample_seed + 100_000 * repeat,
            batch_size=batch_size,
        )
        prefix, _ = clustering_row(
            labels,
            evaluation.prefix_cluster_probabilities,
            model=variant,
            degrees_of_freedom=model.degrees_of_freedom,
            representation="prefix_half",
        )
        full, _ = clustering_row(
            labels,
            evaluation.full_cluster_probabilities,
            model=variant,
            degrees_of_freedom=model.degrees_of_freedom,
            representation="full_sequence",
        )
        rows.append({
            "architecture": architecture,
            "variant": variant,
            "repeat": repeat,
            "samples": n_samples,
            "suffix_nll": float(
                -evaluation.conditional_suffix_scores.sum()
                / suffix_exposure
            ),
            "full_nll": float(
                -evaluation.full_marginal_scores.sum() / full_exposure
            ),
            "prefix_purity": prefix["purity"],
            "prefix_ari": prefix["ari"],
            "prefix_nmi": prefix["nmi"],
            "prefix_active_k": prefix["active_k"],
            "full_purity": full["purity"],
            "full_ari": full["ari"],
            "full_nmi": full["nmi"],
            "full_active_k": full["active_k"],
            "alpha": _alpha(model),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("nhp", "thp", "cotic"),
        default=["nhp", "thp", "cotic"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--wishart-nu", type=int)
    parser.add_argument("--omega-learning-rate", type=float)
    parser.add_argument("--alpha-learning-rate", type=float)
    parser.add_argument("--initial-alpha", type=float)
    parser.add_argument("--alpha-temperature", type=float)
    parser.add_argument("--audit-samples", type=int)
    parser.add_argument("--audit-repeats", type=int)
    parser.add_argument(
        "--heterogeneity",
        choices=("baseline", "zero", "moderate", "strong", "suppressive"),
        default="baseline",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/signed_wishart_pilot_seed0"),
    )
    args = parser.parse_args()
    seed_offset = 10_000_000 * args.training_seed
    config: dict[str, object] = {
        "protocol": "dissertation_chapter_3_signed_pilot",
        "heterogeneity": args.heterogeneity,
        "dgp_true_alpha": 1.0,
        "parameter_seed": 20260746,
        "simulation_seed": 20260747 + 1_000 * args.training_seed,
        "split_seed": 20260748 + 1_000 * args.training_seed,
        "init_seed": 2026080701 + seed_offset,
        "sample_seed": 2026080702 + seed_offset,
        "training_seed": args.training_seed,
        "architectures": args.architectures,
        "parameterizations": ["output_split"],
        "variants": ["no_wishart", *args.variants],
        "n_components": 3,
        "n_marks": 5,
        "wishart_dimension": 15,
        "degrees_of_freedom": 15,
        "routing_temperature": 1.0,
        "intensity_temperature": 1.0,
        "alpha_max": 1.0,
        "initial_alpha": 0.1,
        "interaction_temperature": 1.0,
        "activity_epsilon": 1e-8,
        "horizon": 9.4,
        "cutoff": 4.7,
        "generated_per_class": 400,
        "train_per_class": 30,
        "validation_per_class": 15,
        "test_per_class": 355,
        "quadrature_order": 8,
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
        "epochs": args.epochs,
        "batch_size": 90,
        "evaluation_interval": 5,
        "train_samples": 4,
        "validation_samples": 16,
        "test_samples": 64,
        "audit_samples": 256,
        "audit_repeats": 3,
        "evaluation_batch_size": 16,
        "audit_evaluation_batch_size": 4,
        "neural_learning_rate": 0.001,
        "distribution_learning_rate": 0.01,
        "interaction_learning_rate": 0.01,
        "neural_weight_decay": 1e-5,
        "mean_hyperprior_strength": 1.0,
        "gradient_clip": 20.0,
        "uses_random_walk": False,
        "uses_local_w_parameters": False,
        "uses_test_time_w_optimization": False,
        "uses_elbo": False,
        "device": args.device,
    }
    if args.quick:
        config.update({
            "generated_per_class": 12,
            "train_per_class": 4,
            "validation_per_class": 4,
            "test_per_class": 4,
            "quadrature_order": 4,
            "nhp_hidden_size": 6,
            "thp_hidden_size": 8,
            "thp_num_layers": 1,
            "thp_num_heads": 2,
            "cotic_input_channels": 4,
            "cotic_hidden_size": 8,
            "cotic_num_layers": 2,
            "dropout": 0.0,
            "epochs": 2,
            "batch_size": 12,
            "evaluation_interval": 1,
            "train_samples": 2,
            "validation_samples": 4,
            "test_samples": 8,
            "audit_samples": 16,
            "audit_repeats": 2,
            "evaluation_batch_size": 4,
            "audit_evaluation_batch_size": 2,
        })
    if args.audit_samples is not None:
        if args.audit_samples <= 0:
            raise ValueError("audit samples must be positive")
        config["audit_samples"] = args.audit_samples
    if args.audit_repeats is not None:
        if args.audit_repeats <= 0:
            raise ValueError("audit repeats must be positive")
        config["audit_repeats"] = args.audit_repeats
    if args.wishart_nu is not None:
        if args.wishart_nu < int(config["wishart_dimension"]):
            raise ValueError(
                "Wishart nu must be at least the Wishart dimension "
                f"({config['wishart_dimension']})"
            )
        config["degrees_of_freedom"] = args.wishart_nu
    if args.alpha_learning_rate is not None:
        if args.alpha_learning_rate <= 0.0:
            raise ValueError("alpha learning rate must be positive")
        config["interaction_learning_rate"] = args.alpha_learning_rate
    if args.omega_learning_rate is not None:
        if args.omega_learning_rate <= 0.0:
            raise ValueError("omega learning rate must be positive")
        config["distribution_learning_rate"] = args.omega_learning_rate
    if args.initial_alpha is not None:
        if not 0.0 < args.initial_alpha < float(config["alpha_max"]):
            raise ValueError("initial alpha must lie strictly inside its bounds")
        config["initial_alpha"] = args.initial_alpha
    if args.alpha_temperature is not None:
        if args.alpha_temperature <= 0.0:
            raise ValueError("alpha temperature must be positive")
        config["interaction_temperature"] = args.alpha_temperature

    device = torch.device(args.device)
    outdir = args.outdir
    histories = outdir / "histories"
    checkpoints = outdir / "checkpoints"
    means = outdir / "learned_means"
    predictions = outdir / "predictions"
    for directory in (outdir, histories, checkpoints, means, predictions):
        directory.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (outdir / "environment.json").write_text(
        json.dumps(_environment_snapshot(device), indent=2) + "\n",
        encoding="utf-8",
    )

    started = time.time()
    split_kwargs = {
        "parameter_seed": int(config["parameter_seed"]),
        "simulation_seed": int(config["simulation_seed"]),
        "split_seed": int(config["split_seed"]),
        "horizon": float(config["horizon"]),
        "generated_per_class": int(config["generated_per_class"]),
        "train_per_class": int(config["train_per_class"]),
        "validation_per_class": int(config["validation_per_class"]),
        "test_per_class": int(config["test_per_class"]),
    }
    if args.heterogeneity == "baseline":
        train, validation, test, data_audit = make_splits(**split_kwargs)
    else:
        train, validation, test, data_audit = make_signed_splits(
            heterogeneity=args.heterogeneity,
            true_alpha=float(config["dgp_true_alpha"]),
            **split_kwargs,
        )
    train_sequences = sequences(train)
    validation_sequences = sequences(validation)
    test_sequences = sequences(test)
    data_audit["actual_split_sizes"] = {
        "train": len(train.labels),
        "validation": len(validation.labels),
        "test": len(test.labels),
    }
    (outdir / "data_audit.json").write_text(
        json.dumps(data_audit, indent=2) + "\n",
        encoding="utf-8",
    )
    initial_total_rate = float(
        np.mean([sequence.count for sequence in train_sequences])
        / float(config["horizon"])
    )
    cutoff = float(config["cutoff"])
    suffix_exposure = len(test_sequences) * (
        float(config["horizon"]) - cutoff
    )
    full_exposure = len(test_sequences) * float(config["horizon"])
    nll_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    architecture_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    for architecture_index, architecture in enumerate(args.architectures):
        print(f"[{architecture_index + 1}/{len(args.architectures)}] {architecture}", flush=True)
        init_seed = int(config["init_seed"]) + 10_000 * architecture_index
        initial = _initial_backbone(
            architecture,
            "output_split",
            config,
            initial_total_rate=initial_total_rate,
            initialization_seed=init_seed,
            device=device,
        )
        audit = _architecture_audit(initial, "output_split")
        audit["architecture"] = architecture
        audit["backbone_parameter_count"] = _parameter_count(initial)
        architecture_rows.append(audit)
        stem = f"{architecture}_output_split"

        print("  no_wishart", flush=True)
        _set_seed(init_seed + 30_000)
        direct = fit_direct_nhp_mixture(
            copy.deepcopy(initial),
            train_sequences,
            validation_sequences,
            max_epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["neural_learning_rate"]),
            neural_weight_decay=float(config["neural_weight_decay"]),
            evaluation_interval=int(config["evaluation_interval"]),
            validation_cutoff=cutoff,
            evaluation_batch_size=int(config["evaluation_batch_size"]),
            gradient_clip=float(config["gradient_clip"]),
            batch_seed=init_seed + 300,
        )
        pd.DataFrame(direct.history).to_csv(
            histories / f"{stem}_no_wishart.csv", index=False
        )
        torch.save(
            direct.model.state_dict(),
            checkpoints / f"{stem}_no_wishart.pt",
        )
        direct_evaluation = evaluate_direct_nhp_mixture(
            direct.model,
            test_sequences,
            cutoff=cutoff,
            batch_size=int(config["evaluation_batch_size"]),
        )
        direct_nll, direct_cluster = _nll_and_cluster_rows(
            architecture=architecture,
            parameterization="output_split",
            model_name="no_wishart",
            evaluation=direct_evaluation,
            labels=test.labels,
            suffix_exposure=suffix_exposure,
            full_exposure=full_exposure,
            parameter_count=_parameter_count(direct.model),
            best_epoch=direct.best_epoch,
        )
        direct_nll["variant"] = "no_wishart"
        direct_nll["alpha"] = float("nan")
        direct_nll["delta_suffix_vs_no_w"] = 0.0
        nll_rows.append(direct_nll)
        for row in direct_cluster:
            row["variant"] = "no_wishart"
        cluster_rows.extend(direct_cluster)
        save_probabilities(
            predictions / f"{stem}_no_wishart.csv",
            labels=test.labels,
            prefix=direct_evaluation.prefix_cluster_probabilities,
            full=direct_evaluation.full_cluster_probabilities,
        )

        for variant_index, variant in enumerate(args.variants):
            print(f"  {variant}", flush=True)
            _set_seed(init_seed + 30_000)
            model = _wrapper(
                variant,
                copy.deepcopy(initial),
                config,
            ).to(device)
            fit = fit_latent_wishart_attention_nhp(
                model,
                train_sequences,
                validation_sequences,
                max_epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                train_samples=int(config["train_samples"]),
                validation_samples=int(config["validation_samples"]),
                neural_learning_rate=float(config["neural_learning_rate"]),
                distribution_learning_rate=float(config["distribution_learning_rate"]),
                interaction_learning_rate=float(config["interaction_learning_rate"]),
                neural_weight_decay=float(config["neural_weight_decay"]),
                mean_hyperprior_strength=float(config["mean_hyperprior_strength"]),
                evaluation_interval=int(config["evaluation_interval"]),
                validation_cutoff=cutoff,
                evaluation_batch_size=int(config["evaluation_batch_size"]),
                gradient_clip=float(config["gradient_clip"]),
                batch_seed=init_seed + 300,
                sample_seed=(
                    int(config["sample_seed"])
                    + 1_000_000 * architecture_index
                    + 100_000 * variant_index
                ),
            )
            pd.DataFrame(fit.history).to_csv(
                histories / f"{stem}_{variant}.csv", index=False
            )
            torch.save(
                fit.model.state_dict(),
                checkpoints / f"{stem}_{variant}.pt",
            )
            save_mean_matrix(
                means / f"{stem}_{variant}_mean.csv",
                fit.model.mean_matrix(),
            )
            evaluation = evaluate_latent_wishart_nhp(
                fit.model,
                test_sequences,
                cutoff=cutoff,
                n_samples=int(config["test_samples"]),
                sample_seed=(
                    int(config["sample_seed"])
                    + 500_000
                    + 1_000_000 * architecture_index
                    + 100_000 * variant_index
                ),
                batch_size=int(config["evaluation_batch_size"]),
            )
            row_nll, rows_cluster = _nll_and_cluster_rows(
                architecture=architecture,
                parameterization="output_split",
                model_name=variant,
                evaluation=evaluation,
                labels=test.labels,
                suffix_exposure=suffix_exposure,
                full_exposure=full_exposure,
                parameter_count=_parameter_count(fit.model),
                best_epoch=fit.best_epoch,
            )
            row_nll["variant"] = variant
            row_nll["alpha"] = _alpha(fit.model)
            row_nll["delta_suffix_vs_no_w"] = (
                row_nll["suffix_nll_per_exposure"]
                - direct_nll["suffix_nll_per_exposure"]
            )
            nll_rows.append(row_nll)
            for row in rows_cluster:
                row["variant"] = variant
                row["alpha"] = _alpha(fit.model)
            cluster_rows.extend(rows_cluster)
            save_probabilities(
                predictions / f"{stem}_{variant}.csv",
                labels=test.labels,
                prefix=evaluation.prefix_cluster_probabilities,
                full=evaluation.full_cluster_probabilities,
            )

            with torch.no_grad():
                mean = fit.model.mean_matrix()
                eigenvalues = torch.linalg.eigvalsh(mean)
                mean_gate = cluster_log_weights_from_matrices(
                    mean[None],
                    n_components=int(config["n_components"]),
                    n_marks=int(config["n_marks"]),
                ).exp()[0]
                coupling = signed_correlation_coupling(mean)
                off_diagonal = ~torch.eye(
                    fit.model.dimension,
                    dtype=torch.bool,
                    device=fit.model.device,
                )
                signed_values = coupling[off_diagonal]
            distribution_rows.append({
                "architecture": architecture,
                "variant": variant,
                "transformation": fit.model.transformation_name,
                "best_epoch": fit.best_epoch,
                "alpha": _alpha(fit.model),
                "deterministic_w": fit.model.uses_deterministic_matrices,
                "minimum_mean_eigenvalue": float(eigenvalues.min().cpu()),
                "maximum_mean_eigenvalue": float(eigenvalues.max().cpu()),
                "mean_matrix_gate": mean_gate.cpu().tolist(),
                "mean_abs_signed_coupling": float(signed_values.abs().mean().cpu()),
                "fraction_negative_signed_coupling": float((signed_values < 0).float().mean().cpu()),
                "normalized_prefix_joint_ess": float(
                    evaluation.prefix_joint_effective_sample_size.mean()
                    / (int(config["test_samples"]) * int(config["n_components"]))
                ),
                "normalized_full_joint_ess": float(
                    evaluation.full_joint_effective_sample_size.mean()
                    / (int(config["test_samples"]) * int(config["n_components"]))
                ),
            })
            stability_rows.extend(_audit_rows(
                model=fit.model,
                architecture=architecture,
                variant=variant,
                test_sequences=test_sequences,
                labels=test.labels,
                cutoff=cutoff,
                suffix_exposure=suffix_exposure,
                full_exposure=full_exposure,
                n_samples=int(config["audit_samples"]),
                repeats=int(config["audit_repeats"]),
                sample_seed=(
                    int(config["sample_seed"])
                    + 9_000_000
                    + 1_000_000 * architecture_index
                    + 100_000 * variant_index
                ),
                batch_size=int(config["audit_evaluation_batch_size"]),
            ))

    nll = pd.DataFrame(nll_rows)
    clustering = pd.DataFrame(cluster_rows)
    architecture_audit = pd.DataFrame(architecture_rows)
    distribution = pd.DataFrame(distribution_rows)
    stability = pd.DataFrame(stability_rows)
    summary = (
        stability.groupby(["architecture", "variant"], as_index=False)
        .agg(
            suffix_nll_mean=("suffix_nll", "mean"),
            suffix_nll_sd=("suffix_nll", "std"),
            full_nll_mean=("full_nll", "mean"),
            prefix_purity_mean=("prefix_purity", "mean"),
            prefix_ari_mean=("prefix_ari", "mean"),
            full_purity_mean=("full_purity", "mean"),
            full_ari_mean=("full_ari", "mean"),
            minimum_full_active_k=("full_active_k", "min"),
            alpha=("alpha", "mean"),
        )
    )
    baseline = nll.loc[nll["variant"] == "no_wishart", [
        "architecture", "suffix_nll_per_exposure"
    ]].rename(columns={"suffix_nll_per_exposure": "no_w_suffix_nll"})
    final = summary.merge(baseline, on="architecture", how="left")
    final["delta_suffix_vs_no_w"] = (
        final["suffix_nll_mean"] - final["no_w_suffix_nll"]
    )

    nll.to_csv(outdir / "nll.csv", index=False)
    clustering.to_csv(outdir / "clustering.csv", index=False)
    architecture_audit.to_csv(outdir / "architecture_audit.csv", index=False)
    distribution.to_csv(outdir / "distribution_diagnostics.csv", index=False)
    stability.to_csv(outdir / "mc_stability.csv", index=False)
    summary.to_csv(outdir / "mc_stability_summary.csv", index=False)
    final.to_csv(outdir / "FINAL_COMPARISON.csv", index=False)

    final_table = markdown_table(
        final,
        (
            "architecture",
            "variant",
            "no_w_suffix_nll",
            "suffix_nll_mean",
            "suffix_nll_sd",
            "delta_suffix_vs_no_w",
            "prefix_purity_mean",
            "full_purity_mean",
            "full_ari_mean",
            "minimum_full_active_k",
            "alpha",
        ),
        digits=6,
    )
    report = f"""# Signed WiRE-TPP pilot

This run implements Equations (3.4)-(3.8) of the dissertation chapter.
The signed model uses `B(W) = Corr(W) - I`, unit-temperature softplus,
`alpha = alpha_max * sigmoid(raw_alpha / T_alpha)`,
`alpha_max = {config["alpha_max"]}`, `alpha_0 = {config["initial_alpha"]}`,
and `T_alpha = {config["interaction_temperature"]}`.
The same trajectory-level matrix controls block-trace routing and intensities.

The comparison includes the ordinary finite mixture, the exact legacy
row-normalized `W o W` transformation from Equation (3.9), signed WiRE-TPP,
the `alpha = 0` routing-only ablation, and deterministic `W = Omega`.

## Stable evaluation

{final_table}
"""
    (outdir / "REPORT.md").write_text(report, encoding="utf-8")
    result = {
        "status": "pass",
        "runtime_seconds": time.time() - started,
        "architectures": args.architectures,
        "variants": ["no_wishart", *args.variants],
        "training_seed": args.training_seed,
        "heterogeneity": args.heterogeneity,
        "signed_equations": ["3.4", "3.5", "3.6", "3.7", "3.8"],
        "legacy_equation": "3.9",
        "uses_same_w_for_routing_and_intensity": True,
        "uses_common_random_numbers_for_suffix": True,
    }
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(outdir)
    print(final_table, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
