#!/usr/bin/env python3
"""Corrected Wishart comparison with one shared encoder per architecture."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.experiment import (
    clustering_row,
    make_splits,
    markdown_table,
    save_mean_matrix,
    save_probabilities,
    write_manifest,
)
from lal_wishart.models.latent_wishart_attention_history import (
    LatentWishartAttentionHistory,
)
from lal_wishart.models.latent_wishart_attention_nhp import (
    LatentWishartAttentionNHP,
)
from lal_wishart.models.reference_bos_lal import (
    ReferenceCOTICBOSLaL,
    ReferenceTHPBOSLaL,
    split_reference_bos_lal_component,
)
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
    split_reference_nhp_k1,
)
from lal_wishart.models.reference_output_mixtures import (
    ReferenceCOTICOutputMixture,
    ReferenceNHPOutputMixture,
    ReferenceTHPOutputMixture,
)
from lal_wishart.models.reference_history_mixtures import (
    ReferenceHistoryMixture,
)
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    squared_correlation_attention,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_direct_nhp_mixture,
    evaluate_latent_wishart_nhp,
    fit_direct_nhp_mixture,
    fit_latent_wishart_attention_nhp,
)

def sequences(split):
    """Compatibility helper kept local for the readable runner."""

    return split.sequences


_AUDIT_SEQUENCE = MarkedSequence(
    times=np.array([0.2, 0.7]),
    marks=np.array([0, 1], dtype=np.int64),
    horizon=9.4,
)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _environment_snapshot(device: torch.device) -> dict[str, object]:
    def version(distribution: str) -> str:
        return importlib.metadata.version(distribution)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "easy_tpp_distribution": version("easy-tpp"),
        "omegaconf": version("omegaconf"),
        "pyyaml": version("PyYAML"),
        "packaging": version("packaging"),
        "cotic_commit": "362b8ab1f3cbb9e9dced2518e9daacf68235e77a",
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _output_dimension(model) -> int:
    if isinstance(model, ReferenceNHPOutputMixture):
        return model.intensity_linear.out_features
    if isinstance(model, ReferenceTHPOutputMixture):
        return model.layer_intensity_hidden.out_features
    if isinstance(model, ReferenceCOTICOutputMixture):
        return model.intensity_head.layer.out_features
    return model.n_marks


def _architecture_audit(model, parameterization: str) -> dict[str, object]:
    module_names = tuple(name for name, _ in model.named_modules())
    has_independent_component_list = any(
        name.startswith("components.") or name.startswith("encoders.")
        for name in module_names
    )
    if has_independent_component_list:
        raise RuntimeError("corrected model contains independent encoders")
    if parameterization == "output_split":
        times, types, valid = model._prepare_histories(
            (
                # The values matter only for checking shared encoder states.
                _AUDIT_SEQUENCE,
            )
        )
        model.eval()
        with torch.no_grad():
            states = model.encode_histories(times, types, valid)
        maximum_state_difference = float(
            (states - states[0:1]).abs().max().cpu()
        )
        if maximum_state_difference != 0.0:
            raise RuntimeError("output-split encoder states differ by cluster")
    else:
        maximum_state_difference = float("nan")
    return {
        "parameterization": parameterization,
        "encoder_instances": 1,
        "has_independent_component_module_list": False,
        "final_output_dimension": _output_dimension(model),
        "expected_output_dimension": (
            model.n_components * model.n_marks
            if parameterization == "output_split"
            else model.n_marks
        ),
        "maximum_pre_head_state_difference": maximum_state_difference,
        "cluster_specific_parameters": (
            "K*C final output rows only"
            if parameterization == "output_split"
            else "LaL initial state/BOS code; shared C-output head"
        ),
        "uses_random_walk": False,
    }


def _initial_backbone(
    architecture: str,
    parameterization: str,
    config: dict[str, object],
    *,
    initial_total_rate: float,
    initialization_seed: int,
    device: torch.device,
):
    common = {
        "n_marks": int(config["n_marks"]),
        "horizon": float(config["horizon"]),
        "quadrature_order": int(config["quadrature_order"]),
        "initialization_seed": initialization_seed,
    }
    n_components = int(config["n_components"])
    if parameterization == "output_split":
        if architecture == "nhp":
            model = ReferenceNHPOutputMixture(
                n_components,
                **common,
                hidden_size=int(config["nhp_hidden_size"]),
                initial_total_rate=initial_total_rate,
            )
        elif architecture == "thp":
            model = ReferenceTHPOutputMixture(
                n_components,
                **common,
                hidden_size=int(config["thp_hidden_size"]),
                num_layers=int(config["thp_num_layers"]),
                num_heads=int(config["thp_num_heads"]),
                dropout=float(config["dropout"]),
            )
        elif architecture == "cotic":
            model = ReferenceCOTICOutputMixture(
                n_components,
                **common,
                input_channels=int(config["cotic_input_channels"]),
                hidden_size=int(config["cotic_hidden_size"]),
                num_layers=int(config["cotic_num_layers"]),
                kernel_size=int(config["cotic_kernel_size"]),
                dropout=float(config["dropout"]),
                dilation_factor=float(config["cotic_dilation_factor"]),
            )
        else:
            raise ValueError(f"unknown architecture: {architecture}")
        return model.to(device)

    if parameterization != "lal_fixed":
        raise ValueError(f"unknown parameterization: {parameterization}")
    if architecture == "nhp":
        k1 = ReferenceNeuralHawkesMixture(
            1,
            **common,
            hidden_size=int(config["nhp_hidden_size"]),
            initial_total_rate=initial_total_rate,
            dtype=torch.float32,
        ).to(device)
        return split_reference_nhp_k1(
            k1,
            n_components,
            initialization_seed=initialization_seed + 101,
        )
    if architecture == "thp":
        current = ReferenceTHPBOSLaL(
            1,
            **common,
            hidden_size=int(config["thp_hidden_size"]),
            num_layers=int(config["thp_num_layers"]),
            num_heads=int(config["thp_num_heads"]),
            dropout=float(config["dropout"]),
        ).to(device)
    elif architecture == "cotic":
        current = ReferenceCOTICBOSLaL(
            1,
            **common,
            input_channels=int(config["cotic_input_channels"]),
            hidden_size=int(config["cotic_hidden_size"]),
            num_layers=int(config["cotic_num_layers"]),
            kernel_size=int(config["cotic_kernel_size"]),
            dropout=float(config["dropout"]),
            dilation_factor=float(config["cotic_dilation_factor"]),
        ).to(device)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    current = split_reference_bos_lal_component(
        current,
        0,
        initialization_seed=initialization_seed + 101,
        beta=0.35,
    )
    current = split_reference_bos_lal_component(
        current,
        1,
        initialization_seed=initialization_seed + 102,
        beta=0.40,
    )
    return current.to(device)


def _wishart_wrapper(backbone, *, degrees_of_freedom: int):
    if isinstance(backbone, ReferenceHistoryMixture):
        return LatentWishartAttentionHistory(
            backbone,
            degrees_of_freedom=degrees_of_freedom,
        )
    return LatentWishartAttentionNHP(
        backbone,
        degrees_of_freedom=degrees_of_freedom,
    )


def _nll_and_cluster_rows(
    *,
    architecture: str,
    parameterization: str,
    model_name: str,
    evaluation,
    labels: np.ndarray,
    suffix_exposure: float,
    full_exposure: float,
    parameter_count: int,
    best_epoch: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    nll = {
        "architecture": architecture,
        "parameterization": parameterization,
        "model": model_name,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "suffix_nll_per_exposure": float(
            -evaluation.conditional_suffix_scores.sum() / suffix_exposure
        ),
        "full_marginal_nll_per_exposure": float(
            -evaluation.full_marginal_scores.sum() / full_exposure
        ),
    }
    cluster = []
    for representation, probabilities in (
        ("prefix_half", evaluation.prefix_cluster_probabilities),
        ("full_sequence", evaluation.full_cluster_probabilities),
    ):
        row, _ = clustering_row(
            labels,
            probabilities,
            model=model_name,
            degrees_of_freedom=None,
            representation=representation,
        )
        row["architecture"] = architecture
        row["parameterization"] = parameterization
        cluster.append(row)
    return nll, cluster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("nhp", "thp", "cotic"),
        default=["nhp", "thp", "cotic"],
    )
    parser.add_argument(
        "--parameterizations",
        nargs="+",
        choices=("output_split", "lal_fixed"),
        default=["output_split", "lal_fixed"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/"
            "corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0"
        ),
    )
    args = parser.parse_args()
    config: dict[str, object] = {
        "parameter_seed": 20260746,
        "simulation_seed": 20260747,
        "split_seed": 20260748,
        "init_seed": 2026080701,
        "sample_seed": 2026080702,
        "architectures": args.architectures,
        "parameterizations": args.parameterizations,
        "n_components": 3,
        "n_marks": 5,
        "wishart_dimension": 15,
        "degrees_of_freedom": 20,
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
        "evaluation_batch_size": 16,
        "neural_learning_rate": 0.001,
        "distribution_learning_rate": 0.01,
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
            "evaluation_batch_size": 4,
        })
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
    train, validation, test, data_audit = make_splits(
        parameter_seed=int(config["parameter_seed"]),
        simulation_seed=int(config["simulation_seed"]),
        split_seed=int(config["split_seed"]),
        horizon=float(config["horizon"]),
        generated_per_class=int(config["generated_per_class"]),
        train_per_class=int(config["train_per_class"]),
        validation_per_class=int(config["validation_per_class"]),
        test_per_class=int(config["test_per_class"]),
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

    total = len(args.architectures) * len(args.parameterizations)
    run_index = 0
    for architecture_index, architecture in enumerate(args.architectures):
        for parameterization_index, parameterization in enumerate(
            args.parameterizations
        ):
            run_index += 1
            print(
                f"[{run_index}/{total}] {architecture} {parameterization}",
                flush=True,
            )
            init_seed = (
                int(config["init_seed"])
                + 10_000 * architecture_index
                + 1_000 * parameterization_index
            )
            initial = _initial_backbone(
                architecture,
                parameterization,
                config,
                initial_total_rate=initial_total_rate,
                initialization_seed=init_seed,
                device=device,
            )
            audit = _architecture_audit(initial, parameterization)
            audit["architecture"] = architecture
            audit["backbone_parameter_count"] = _parameter_count(initial)
            architecture_rows.append(audit)

            print("  no-W", flush=True)
            _set_seed(init_seed + 30_000)
            no_w_fit = fit_direct_nhp_mixture(
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
            stem = f"{architecture}_{parameterization}"
            pd.DataFrame(no_w_fit.history).to_csv(
                histories / f"{stem}_no_w.csv",
                index=False,
            )
            torch.save(
                no_w_fit.model.state_dict(),
                checkpoints / f"{stem}_no_w.pt",
            )
            no_w_eval = evaluate_direct_nhp_mixture(
                no_w_fit.model,
                test_sequences,
                cutoff=cutoff,
                batch_size=int(config["evaluation_batch_size"]),
            )
            no_w_nll, no_w_cluster = _nll_and_cluster_rows(
                architecture=architecture,
                parameterization=parameterization,
                model_name="no_wishart",
                evaluation=no_w_eval,
                labels=test.labels,
                suffix_exposure=suffix_exposure,
                full_exposure=full_exposure,
                parameter_count=_parameter_count(no_w_fit.model),
                best_epoch=no_w_fit.best_epoch,
            )
            nll_rows.append(no_w_nll)
            cluster_rows.extend(no_w_cluster)
            save_probabilities(
                predictions / f"{stem}_no_w.csv",
                labels=test.labels,
                prefix=no_w_eval.prefix_cluster_probabilities,
                full=no_w_eval.full_cluster_probabilities,
            )

            print("  latent Wishart", flush=True)
            _set_seed(init_seed + 30_000)
            wishart = _wishart_wrapper(
                copy.deepcopy(initial),
                degrees_of_freedom=int(config["degrees_of_freedom"]),
            ).to(device)
            w_fit = fit_latent_wishart_attention_nhp(
                wishart,
                train_sequences,
                validation_sequences,
                max_epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                train_samples=int(config["train_samples"]),
                validation_samples=int(config["validation_samples"]),
                neural_learning_rate=float(config["neural_learning_rate"]),
                distribution_learning_rate=float(
                    config["distribution_learning_rate"]
                ),
                neural_weight_decay=float(config["neural_weight_decay"]),
                mean_hyperprior_strength=float(
                    config["mean_hyperprior_strength"]
                ),
                evaluation_interval=int(config["evaluation_interval"]),
                validation_cutoff=cutoff,
                evaluation_batch_size=int(config["evaluation_batch_size"]),
                gradient_clip=float(config["gradient_clip"]),
                batch_seed=init_seed + 300,
                sample_seed=(
                    int(config["sample_seed"])
                    + 1_000_000 * run_index
                ),
            )
            pd.DataFrame(w_fit.history).to_csv(
                histories / f"{stem}_wishart.csv",
                index=False,
            )
            torch.save(
                w_fit.model.state_dict(),
                checkpoints / f"{stem}_wishart.pt",
            )
            save_mean_matrix(
                means / f"{stem}_wishart_mean.csv",
                w_fit.model.mean_matrix(),
            )
            w_eval = evaluate_latent_wishart_nhp(
                w_fit.model,
                test_sequences,
                cutoff=cutoff,
                n_samples=int(config["test_samples"]),
                sample_seed=(
                    int(config["sample_seed"])
                    + 500_000
                    + 1_000_000 * run_index
                ),
                batch_size=int(config["evaluation_batch_size"]),
            )
            w_nll, w_cluster = _nll_and_cluster_rows(
                architecture=architecture,
                parameterization=parameterization,
                model_name="latent_wishart_attention",
                evaluation=w_eval,
                labels=test.labels,
                suffix_exposure=suffix_exposure,
                full_exposure=full_exposure,
                parameter_count=_parameter_count(w_fit.model),
                best_epoch=w_fit.best_epoch,
            )
            w_nll["delta_suffix_vs_no_w"] = (
                w_nll["suffix_nll_per_exposure"]
                - no_w_nll["suffix_nll_per_exposure"]
            )
            w_nll["delta_full_vs_no_w"] = (
                w_nll["full_marginal_nll_per_exposure"]
                - no_w_nll["full_marginal_nll_per_exposure"]
            )
            no_w_nll["delta_suffix_vs_no_w"] = 0.0
            no_w_nll["delta_full_vs_no_w"] = 0.0
            cluster_rows.extend(w_cluster)
            nll_rows.append(w_nll)
            save_probabilities(
                predictions / f"{stem}_wishart.csv",
                labels=test.labels,
                prefix=w_eval.prefix_cluster_probabilities,
                full=w_eval.full_cluster_probabilities,
            )
            with torch.no_grad():
                mean = w_fit.model.mean_matrix()
                eigenvalues = torch.linalg.eigvalsh(mean)
                mean_gate = cluster_log_weights_from_matrices(
                    mean[None],
                    n_components=3,
                    n_marks=5,
                ).exp()[0]
            distribution_rows.append({
                "architecture": architecture,
                "parameterization": parameterization,
                "best_epoch": w_fit.best_epoch,
                "minimum_mean_eigenvalue": float(
                    eigenvalues.min().cpu()
                ),
                "maximum_mean_eigenvalue": float(
                    eigenvalues.max().cpu()
                ),
                "mean_matrix_gate": mean_gate.cpu().tolist(),
                "mean_attention_diagonal": float(
                    squared_correlation_attention(mean[None])
                    .diagonal(dim1=-2, dim2=-1)
                    .mean()
                    .cpu()
                ),
                "normalized_prefix_joint_ess": float(
                    w_eval.prefix_joint_effective_sample_size.mean()
                    / (int(config["test_samples"]) * 3)
                ),
                "normalized_full_joint_ess": float(
                    w_eval.full_joint_effective_sample_size.mean()
                    / (int(config["test_samples"]) * 3)
                ),
            })

    nll = pd.DataFrame(nll_rows)
    cluster = pd.DataFrame(cluster_rows)
    architecture_audit = pd.DataFrame(architecture_rows)
    distribution = pd.DataFrame(distribution_rows)
    nll.to_csv(outdir / "nll.csv", index=False)
    cluster.to_csv(outdir / "clustering.csv", index=False)
    architecture_audit.to_csv(
        outdir / "architecture_audit.csv",
        index=False,
    )
    distribution.to_csv(
        outdir / "distribution_diagnostics.csv",
        index=False,
    )
    nll_table = markdown_table(
        nll,
        (
            "architecture",
            "parameterization",
            "model",
            "parameter_count",
            "best_epoch",
            "suffix_nll_per_exposure",
            "delta_suffix_vs_no_w",
            "full_marginal_nll_per_exposure",
        ),
        digits=6,
    )
    cluster_table = markdown_table(
        cluster,
        (
            "architecture",
            "parameterization",
            "model",
            "representation",
            "purity",
            "ari",
            "active_k",
            "cluster_sizes",
            "mean_entropy",
        ),
        digits=4,
    )
    report = f"""# Corrected shared-encoder Wishart comparison

Every architecture now has exactly one encoder.

`output_split` expands only the final intensity head from C=5 to K*C=15.
The pre-head history representation is exactly identical for all three
mixture components; component k uses output rows `5k:5(k+1)`.

`lal_fixed` keeps a shared five-output head and introduces only the
LaL-style cluster state: CT-LSTM initial state for NHP and BOS embedding for
THP/COTIC.  It is initialized by two LaL splits to K=3 and then trained at
fixed K.  There are no split/merge/delete walks in either parameterization.

Both branches are evaluated with and without the same latent law
`W ~ Wishart_15(20, M/20)`.  The likelihood is the classical continuous-time
TPP likelihood with a terminal compensator; W is integrated by Monte Carlo.

## Likelihood

{nll_table}

## Clustering

{cluster_table}
"""
    (outdir / "REPORT.md").write_text(report, encoding="utf-8")
    result = {
        "status": "pass",
        "runtime_seconds": time.time() - started,
        "architectures": args.architectures,
        "parameterizations": args.parameterizations,
        "encoder_instances_per_model": 1,
        "output_split_expands_only_final_head": True,
        "lal_fixed_uses_random_walk": False,
        "uses_local_w_parameters": False,
        "uses_test_time_w_optimization": False,
        "uses_elbo": False,
    }
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(outdir)
    print(nll_table, flush=True)
    print(cluster_table, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
