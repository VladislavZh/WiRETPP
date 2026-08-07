#!/usr/bin/env python3
"""Sequential CoTIC no-W/Wishart/LaL comparison on a matched Wishart DGP."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from lal_wishart.experiment import clustering_row
from lal_wishart.models.reference_bos_lal import (
    ReferenceCOTICBOSLaL,
    split_reference_bos_lal_component,
)
from lal_wishart.models.reference_output_mixtures import (
    ReferenceCOTICOutputMixture,
)
from lal_wishart.models.signed_wishart import SignedWishartTPP
from lal_wishart.reproduction.wishart_re_sin_k5c5 import (
    dataset_audit,
    generate_wishart_re_sin_k5c5,
    load_saved_dataset,
    save_dataset,
    stratified_split,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_direct_nhp_mixture,
    evaluate_latent_wishart_nhp,
    fit_direct_nhp_mixture,
    fit_latent_wishart_attention_nhp,
)
from run_corrected_shared_wishart_architectures import _set_seed
from run_sin_k5_thp_frozen_backbone_lr_warmup_sweep import (
    _gpu_sample,
    _run_with_vram_guard,
)


VARIANTS = ("no_wishart", "signed_wishart", "lal_bos")
DEFAULT_CONFIG = Path(
    "configs/experiments/wishart_re_sin_k5_cotic_comparison.json"
)
DEFAULT_OUTDIR = Path(
    "artifacts/wishart_re_sin_k5_cotic_comparison_20260807"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_config(
    source: dict,
    *,
    quick: bool,
    within_gain: float | None = None,
    between_gain: float | None = None,
    degrees_of_freedom_mode: str | None = None,
    model_degrees_of_freedom: float | None = None,
    degrees_of_freedom_learning_rate: float | None = None,
    train_samples: int | None = None,
    validation_samples: int | None = None,
) -> dict:
    config = copy.deepcopy(source)
    if within_gain is not None:
        config["data"]["within_wishart_interaction_gain"] = within_gain
        config["training"]["within_wishart_interaction_gain"] = within_gain
    if between_gain is not None:
        config["data"]["between_wishart_interaction_gain"] = between_gain
        config["training"]["between_wishart_interaction_gain"] = between_gain
    if degrees_of_freedom_mode is not None:
        config["training"]["degrees_of_freedom_mode"] = (
            degrees_of_freedom_mode
        )
    if model_degrees_of_freedom is not None:
        config["training"]["initial_degrees_of_freedom"] = (
            model_degrees_of_freedom
        )
    if degrees_of_freedom_learning_rate is not None:
        config["training"]["degrees_of_freedom_learning_rate"] = (
            degrees_of_freedom_learning_rate
        )
    if train_samples is not None:
        config["training"]["train_samples"] = train_samples
    if validation_samples is not None:
        config["training"]["validation_samples"] = validation_samples
    if quick:
        config["data"].update({
            "n_per_cluster": 4,
            "horizon": 2.0,
            "wishart_degrees_of_freedom": 26,
        })
        config["training"].update({
            "model_seeds": [0],
            "optimizer_steps": 2,
            "k1_pretrain_steps": 2,
            "evaluation_interval": 1,
            "physical_batch_size": 2,
            "evaluation_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 2,
            "train_samples": 1,
            "validation_samples": 2,
            "test_samples": 2,
            "cotic_input_channels": 4,
            "cotic_hidden_size": 8,
            "cotic_num_layers": 2,
            "dropout": 0.0,
            "quadrature_order": 4,
        })
        config["analysis"].update({
            "minimum_observable_validation_ridge_lda_accuracy": 0.0,
            "minimum_observable_all_kmeans_ari": -1.0,
        })
    return config


def _validate_config(config: dict) -> None:
    data = config["data"]
    training = config["training"]
    n_components = int(data["n_components"])
    n_marks = int(data["n_marks"])
    if (n_components, n_marks) != (5, 5):
        raise ValueError("the registered experiment requires K=5,C=5")
    if int(data["wishart_degrees_of_freedom"]) < n_components * n_marks:
        raise ValueError("DGP Wishart nu must be at least K*C")
    if training["architecture"] != "cotic":
        raise ValueError("this comparison is CoTIC-only")
    variants = tuple(training["variants"])
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"unknown variant list: {variants}")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    seeds = tuple(int(seed) for seed in training["model_seeds"])
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be unique and non-negative")
    physical = int(training["physical_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if physical * accumulation != int(training["effective_batch_size"]):
        raise ValueError("physical batch times accumulation must equal effective batch")
    if float(training["alpha_learning_rate"]) != 1e-4:
        raise ValueError("the registered alpha learning rate is fixed at 1e-4")
    nu_mode = str(training["degrees_of_freedom_mode"])
    initial_nu = float(training["initial_degrees_of_freedom"])
    maximum_nu = float(training["maximum_degrees_of_freedom"])
    if nu_mode not in {"fixed", "learned"}:
        raise ValueError("degrees_of_freedom_mode must be fixed or learned")
    if initial_nu < n_components * n_marks:
        raise ValueError("initial model nu must be at least K*C")
    if nu_mode == "fixed" and not initial_nu.is_integer():
        raise ValueError("fixed model nu must be an integer")
    if nu_mode == "learned" and not (
        n_components * n_marks < initial_nu < maximum_nu
    ):
        raise ValueError("learned model nu must start strictly within bounds")
    if (
        maximum_nu <= initial_nu
        or float(training["degrees_of_freedom_learning_rate"]) <= 0.0
        or float(training["degrees_of_freedom_prior_strength"]) < 0.0
        or float(training["degrees_of_freedom_prior_center"])
        <= n_components * n_marks
    ):
        raise ValueError("invalid learnable-nu optimizer or prior settings")
    if (
        int(training["k1_pretrain_steps"]) <= 0
        or float(training["component_clone_noise_scale"]) < 0.0
        or float(training["maximum_initial_component_score_spread"]) <= 0.0
        or float(training["assignment_rho"]) <= 0.0
        or float(training["dual_learning_rate"]) <= 0.0
        or int(training["train_samples"]) <= 0
        or int(training["validation_samples"]) <= 0
        or int(training["test_samples"]) <= 0
    ):
        raise ValueError("invalid pretraining, expansion, or OT settings")
    totals = np.asarray(data["cluster_baseline_totals"], dtype=float)
    if totals.shape != (n_components,) or np.any(totals <= 0.0):
        raise ValueError("cluster baseline totals must be positive K-vector")
    if bool(data["block_local_activity"]) != bool(
        training["block_local_activity"]
    ):
        raise ValueError("DGP and model block-local settings must match")
    for key in (
        "within_wishart_interaction_gain",
        "between_wishart_interaction_gain",
    ):
        data_value = float(data[key])
        training_value = float(training[key])
        if (
            not np.isfinite(data_value)
            or data_value < 0.0
            or data_value != training_value
        ):
            raise ValueError(f"invalid or mismatched DGP/model {key}")
    if not bool(data["block_local_activity"]) and any(
        float(data[key]) != 1.0
        for key in (
            "within_wishart_interaction_gain",
            "between_wishart_interaction_gain",
        )
    ):
        raise ValueError("custom block gains require block-local activity")
    if bool(training["test_used_for_selection"]):
        raise ValueError("test data must not be used for checkpoint selection")


def _generate_dataset(config: dict, cache_path: Path) -> dict[str, object]:
    if cache_path.is_file():
        cached = load_saved_dataset(cache_path)
        data = config["data"]
        expected = (
            bool(data["block_local_activity"]),
            float(data["within_wishart_interaction_gain"]),
            float(data["between_wishart_interaction_gain"]),
        )
        observed = (
            cached.block_local_activity,
            cached.within_block_gain,
            cached.between_block_gain,
        )
        if observed != expected:
            raise RuntimeError(
                "cached DGP interaction settings do not match protocol: "
                f"cached={observed}, expected={expected}"
            )
        return dataset_audit(cached)
    data = config["data"]
    dataset = generate_wishart_re_sin_k5c5(
        parameter_seed=int(data["parameter_seed"]),
        simulation_seed=int(data["simulation_seed"]),
        n_per_cluster=int(data["n_per_cluster"]),
        n_components=int(data["n_components"]),
        n_marks=int(data["n_marks"]),
        horizon=float(data["horizon"]),
        degrees_of_freedom=int(data["wishart_degrees_of_freedom"]),
        within_cluster_base_strength=float(
            data["within_cluster_base_strength"]
        ),
        true_cluster_block_boost=float(data["true_cluster_block_boost"]),
        between_cluster_strength=float(data["between_cluster_strength"]),
        true_alpha=float(data["true_interaction_alpha"]),
        block_local_activity=bool(data["block_local_activity"]),
        within_block_gain=float(data["within_wishart_interaction_gain"]),
        between_block_gain=float(data["between_wishart_interaction_gain"]),
        cluster_baseline_totals=np.asarray(
            data["cluster_baseline_totals"], dtype=float
        ),
        max_jumps=int(data["max_jumps"]),
    )
    save_dataset(cache_path, dataset)
    return dataset_audit(dataset)


def _cotic_common(config: dict, *, initialization_seed: int) -> dict:
    data = config["data"]
    training = config["training"]
    return {
        "n_marks": int(data["n_marks"]),
        "horizon": float(data["horizon"]),
        "input_channels": int(training["cotic_input_channels"]),
        "hidden_size": int(training["cotic_hidden_size"]),
        "num_layers": int(training["cotic_num_layers"]),
        "kernel_size": int(training["cotic_kernel_size"]),
        "dropout": float(training["dropout"]),
        "dilation_factor": float(training["cotic_dilation_factor"]),
        "quadrature_order": int(training["quadrature_order"]),
        "initialization_seed": initialization_seed,
    }


def _k1_model(config: dict, *, initialization_seed: int, device):
    return ReferenceCOTICOutputMixture(
        1,
        **_cotic_common(config, initialization_seed=initialization_seed),
    ).to(device)


def _load_or_fit_k1_model(
    config: dict,
    split,
    *,
    seed_root: Path,
    initialization_seed: int,
    device,
):
    """Create one shared K=1 COTIC checkpoint per model seed."""

    checkpoint = seed_root / "checkpoint_cotic_k1.pt"
    metadata_path = seed_root / "pretrain_result.json"
    model = _k1_model(
        config,
        initialization_seed=initialization_seed,
        device=device,
    )
    if checkpoint.is_file() and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        expected = {
            "initialization_seed": initialization_seed,
            "pretrain_steps": int(config["training"]["k1_pretrain_steps"]),
            "n_marks": int(config["data"]["n_marks"]),
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True)
            )
            return model, metadata

    training = config["training"]
    started = time.time()
    fit = fit_direct_nhp_mixture(
        model,
        split.train,
        split.validation,
        max_epochs=int(training["k1_pretrain_steps"]),
        batch_size=int(training["physical_batch_size"]),
        learning_rate=float(training["neural_learning_rate"]),
        neural_weight_decay=float(training["weight_decay"]),
        evaluation_interval=int(training["evaluation_interval"]),
        validation_cutoff=float(config["data"]["horizon"]) / 2.0,
        evaluation_batch_size=int(training["evaluation_batch_size"]),
        gradient_clip=float(training["gradient_clip"]),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        batch_seed=initialization_seed + 200,
        validation_labels=None,
        selection_metric="validation_nll",
        show_progress=True,
        progress_description="CoTIC shared K=1 pretrain",
        assignment_ot_marginal_penalty=0.0,
        optimize_mixture_logits=False,
    )
    seed_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit.history).to_csv(
        seed_root / "pretrain_history.csv", index=False
    )
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(fit.model.state_dict(), temporary)
    temporary.replace(checkpoint)
    metadata = {
        "status": "pass",
        "initialization_seed": initialization_seed,
        "pretrain_steps": int(training["k1_pretrain_steps"]),
        "n_marks": int(config["data"]["n_marks"]),
        "best_step": int(fit.best_epoch),
        "best_validation_nll_per_exposure": float(
            fit.best_validation_suffix_nll_per_exposure
        ),
        "runtime_seconds": time.time() - started,
    }
    _write_json(metadata_path, metadata)
    return fit.model, metadata


def _output_model(
    config: dict,
    pretrained_k1,
    *,
    initialization_seed: int,
    device,
):
    model = _k1_model(
        config,
        initialization_seed=initialization_seed,
        device=device,
    )
    model.load_state_dict(pretrained_k1.state_dict())
    model.expand_components(
        int(config["data"]["n_components"]),
        noise_scale=float(config["training"]["component_clone_noise_scale"]),
        initialization_seed=initialization_seed + 10_000,
    )
    return model


def _lal_model(
    config: dict,
    pretrained_k1,
    *,
    initialization_seed: int,
    device,
):
    current = ReferenceCOTICBOSLaL(
        1,
        **_cotic_common(config, initialization_seed=initialization_seed),
    ).to(device)
    current.encoder.load_state_dict(pretrained_k1.encoder.state_dict())
    current.intensity_head.load_state_dict(
        pretrained_k1.intensity_head.state_dict()
    )
    with torch.no_grad():
        current.bos_embeddings.copy_(
            current.encoder.event_emb.weight[current._bos_type()][None]
        )
    split_plan = ((0, 0.35), (1, 0.40), (0, 0.45), (2, 0.50))
    for split_index, (component, beta) in enumerate(split_plan, start=1):
        current = split_reference_bos_lal_component(
            current,
            component,
            initialization_seed=initialization_seed + 100 + split_index,
            beta=beta,
        )
    if current.n_components != int(config["data"]["n_components"]):
        raise RuntimeError("LaL split plan did not reach target K")
    return current.to(device)


def _result_metrics(labels: np.ndarray, probabilities: torch.Tensor, variant: str):
    row, predictions = clustering_row(
        labels,
        probabilities,
        model=variant,
        degrees_of_freedom=None,
        representation="full_sequence",
    )
    return row, predictions


def _alpha_diagnostics(history: tuple[dict, ...]) -> dict[str, object]:
    values = np.asarray([
        float(row["interaction_strength"])
        for row in history
        if np.isfinite(float(row.get("interaction_strength", np.nan)))
    ])
    gradient = np.asarray([
        float(row["gradient_norm"])
        for row in history
        if np.isfinite(float(row.get("gradient_norm", np.nan)))
    ])
    clipped = np.asarray([
        bool(row.get("gradient_was_clipped", False)) for row in history
    ])
    return {
        "alpha_min": float(values.min()) if values.size else None,
        "alpha_max": float(values.max()) if values.size else None,
        "gradient_norm_median": float(np.median(gradient)) if gradient.size else None,
        "gradient_norm_maximum": float(gradient.max()) if gradient.size else None,
        "gradient_clipped_step_fraction": float(clipped.mean()) if clipped.size else None,
    }


def _nu_diagnostics(history: tuple[dict, ...]) -> dict[str, object]:
    values = np.asarray([
        float(row["degrees_of_freedom"])
        for row in history
        if np.isfinite(float(row.get("degrees_of_freedom", np.nan)))
    ])
    gradients = np.asarray([
        float(row["degrees_of_freedom_gradient"])
        for row in history
        if np.isfinite(
            float(row.get("degrees_of_freedom_gradient", np.nan))
        )
    ])
    return {
        "degrees_of_freedom_min": (
            float(values.min()) if values.size else None
        ),
        "degrees_of_freedom_max": (
            float(values.max()) if values.size else None
        ),
        "degrees_of_freedom_gradient_median": (
            float(np.median(gradients)) if gradients.size else None
        ),
        "degrees_of_freedom_gradient_abs_maximum": (
            float(np.abs(gradients).max()) if gradients.size else None
        ),
    }


def _run_worker(
    *,
    config: dict,
    dataset_cache: Path,
    outdir: Path,
    seed: int,
    variant: str,
    device_name: str,
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_saved_dataset(dataset_cache)
    data = config["data"]
    training = config["training"]
    split = stratified_split(
        dataset,
        seed=int(data["split_seed"]),
        train_fraction=float(data["train_fraction"]),
        validation_fraction=float(data["validation_fraction"]),
    )
    device = torch.device(device_name)
    initialization_seed = 2026087100 + seed * 100_000
    _set_seed(initialization_seed)
    pretrained_k1, pretrain_metadata = _load_or_fit_k1_model(
        config,
        split,
        seed_root=outdir.parent,
        initialization_seed=initialization_seed,
        device=device,
    )
    if variant == "lal_bos":
        initial = _lal_model(
            config,
            pretrained_k1,
            initialization_seed=initialization_seed,
            device=device,
        )
    else:
        initial = _output_model(
            config,
            pretrained_k1,
            initialization_seed=initialization_seed,
            device=device,
        )
    was_training = initial.training
    initial.eval()
    with torch.no_grad():
        initial_scores = initial.component_scores_for_indices(
            split.validation[: min(8, len(split.validation))],
            tuple(range(initial.n_components)),
        )
    initial.train(was_training)
    initial_spreads = (
        initial_scores.max(dim=1).values
        - initial_scores.min(dim=1).values
    )
    initial_score_spread_mean = float(initial_spreads.mean().cpu())
    initial_score_spread_max = float(initial_spreads.max().cpu())
    if initial_score_spread_max > float(
        training["maximum_initial_component_score_spread"]
    ):
        raise RuntimeError(
            "asymmetric COTIC expansion failed preflight: "
            f"component-score spread={initial_score_spread_max:.3f}"
        )
    started = time.time()
    cutoff = float(data["horizon"]) / 2.0
    common = {
        "max_epochs": int(training["optimizer_steps"]),
        "batch_size": int(training["physical_batch_size"]),
        "neural_weight_decay": float(training["weight_decay"]),
        "evaluation_interval": int(training["evaluation_interval"]),
        "validation_cutoff": cutoff,
        "evaluation_batch_size": int(training["evaluation_batch_size"]),
        "gradient_clip": float(training["gradient_clip"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "batch_seed": initialization_seed + 300,
        "validation_labels": split.validation_labels,
        "selection_metric": str(training["selection_metric"]),
        "show_progress": True,
        "progress_description": f"CoTIC {variant} seed={seed}",
        "assignment_ot_marginal_penalty": float(
            training["assignment_rho"]
        ),
        "assignment_ot_temperature": 1.0,
        "assignment_ot_dual_learning_rate": float(
            training["dual_learning_rate"]
        ),
    }
    if variant == "signed_wishart":
        dimension = int(data["n_components"]) * int(data["n_marks"])
        nu_mode = str(training["degrees_of_freedom_mode"])
        initial_nu = float(training["initial_degrees_of_freedom"])
        model = SignedWishartTPP(
            initial,
            degrees_of_freedom=(
                initial_nu if nu_mode == "learned" else int(initial_nu)
            ),
            learnable_degrees_of_freedom=nu_mode == "learned",
            maximum_degrees_of_freedom=float(
                training["maximum_degrees_of_freedom"]
            ),
            initial_alpha=float(training["initial_alpha"]),
            alpha_parameterization=str(
                training["alpha_parameterization"]
            ),
            interaction_mode=str(training["interaction_mode"]),
            block_local_activity=bool(training["block_local_activity"]),
            within_block_gain=float(
                training["within_wishart_interaction_gain"]
            ),
            between_block_gain=float(
                training["between_wishart_interaction_gain"]
            ),
            exploration_beta=float(training["initial_beta"]),
            exploration_degrees_of_freedom=int(
                training["exploration_degrees_of_freedom"]
            ),
            exploration_mode="convex",
        ).to(device)
        fit = fit_latent_wishart_attention_nhp(
            model,
            split.train,
            split.validation,
            train_samples=int(training["train_samples"]),
            validation_samples=int(training["validation_samples"]),
            neural_learning_rate=float(training["neural_learning_rate"]),
            distribution_learning_rate=float(
                training["omega_learning_rate"]
            ),
            interaction_learning_rate=float(
                training["alpha_learning_rate"]
            ),
            degrees_of_freedom_learning_rate=float(
                training["degrees_of_freedom_learning_rate"]
            ),
            degrees_of_freedom_prior_strength=float(
                training["degrees_of_freedom_prior_strength"]
            ),
            degrees_of_freedom_prior_center=float(
                training["degrees_of_freedom_prior_center"]
            ),
            mean_hyperprior_strength=float(
                training["mean_hyperprior_strength"]
            ),
            sample_seed=initialization_seed + 500,
            exploration_beta_anneal_epochs=int(
                training["optimizer_steps"]
            ),
            exploration_beta_schedule="exponential",
            exploration_beta_decay_rate=float(
                training["beta_decay_rate"]
            ),
            validation_exploration_beta=0.0,
            **common,
        )
        evaluation = evaluate_latent_wishart_nhp(
            fit.model,
            split.test,
            cutoff=cutoff,
            n_samples=int(training["test_samples"]),
            sample_seed=initialization_seed + 900_000,
            batch_size=int(training["evaluation_batch_size"]),
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        best_validation_nll = fit.best_validation_suffix_nll_per_exposure
        best_validation_purity = fit.best_validation_purity
        best_step = fit.best_epoch
        alpha = float(fit.model.interaction_strength().detach().cpu())
        np.savetxt(
            outdir / "learned_wishart_mean.csv",
            fit.model.mean_matrix().detach().cpu().numpy(),
            delimiter=",",
        )
    else:
        fit = fit_direct_nhp_mixture(
            initial,
            split.train,
            split.validation,
            learning_rate=float(training["neural_learning_rate"]),
            optimize_mixture_logits=bool(
                training["optimize_mixture_logits"]
            ),
            **common,
        )
        evaluation = evaluate_direct_nhp_mixture(
            fit.model,
            split.test,
            cutoff=cutoff,
            batch_size=int(training["evaluation_batch_size"]),
            assignment_dual=fit.assignment_dual,
            assignment_temperature=fit.assignment_temperature,
        )
        best_validation_nll = fit.best_validation_suffix_nll_per_exposure
        best_validation_purity = fit.best_validation_purity
        best_step = fit.best_epoch
        alpha = None
    if fit.assignment_dual is not None:
        np.save(outdir / "assignment_dual.npy", fit.assignment_dual.numpy())
    history = tuple(fit.history)
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    torch.save(fit.model.state_dict(), outdir / "checkpoint.pt")
    cluster, predictions = _result_metrics(
        split.test_labels,
        evaluation.full_cluster_probabilities,
        variant,
    )
    probabilities = evaluation.full_cluster_probabilities.numpy()
    prediction_frame = pd.DataFrame({
        "dataset_index": split.test_indices,
        "label": split.test_labels,
        "prediction": predictions,
    })
    for component in range(probabilities.shape[1]):
        prediction_frame[f"p_{component}"] = probabilities[:, component]
    prediction_frame.to_csv(outdir / "test_predictions.csv", index=False)
    suffix_exposure = len(split.test) * (float(data["horizon"]) - cutoff)
    full_exposure = len(split.test) * float(data["horizon"])
    diagnostics = _alpha_diagnostics(history)
    nu_diagnostics = _nu_diagnostics(history)
    result = {
        "status": "pass",
        "seed": seed,
        "variant": variant,
        "architecture": "cotic",
        "best_validation_purity": float(best_validation_purity),
        "best_validation_nll_per_exposure": float(best_validation_nll),
        "best_step": int(best_step),
        "test_purity_descriptive_only": float(cluster["purity"]),
        "test_ari_descriptive_only": float(cluster["ari"]),
        "test_nmi_descriptive_only": float(cluster["nmi"]),
        "test_active_k": int(cluster["active_k"]),
        "test_cluster_sizes": cluster["cluster_sizes"],
        "test_suffix_nll_per_exposure": float(
            -evaluation.conditional_suffix_scores.sum() / suffix_exposure
        ),
        "test_full_nll_per_exposure": float(
            -evaluation.full_marginal_scores.sum() / full_exposure
        ),
        "selected_alpha": alpha,
        "degrees_of_freedom_mode": (
            training["degrees_of_freedom_mode"]
            if variant == "signed_wishart"
            else None
        ),
        "initial_degrees_of_freedom": (
            float(training["initial_degrees_of_freedom"])
            if variant == "signed_wishart"
            else None
        ),
        "selected_degrees_of_freedom": (
            float(fit.model.degrees_of_freedom)
            if variant == "signed_wishart"
            else None
        ),
        "block_local_activity": bool(training["block_local_activity"]),
        "within_wishart_interaction_gain": float(
            training["within_wishart_interaction_gain"]
        ),
        "between_wishart_interaction_gain": float(
            training["between_wishart_interaction_gain"]
        ),
        **diagnostics,
        **nu_diagnostics,
        "runtime_seconds": time.time() - started,
        "shared_k1_pretrain_runtime_seconds": float(
            pretrain_metadata["runtime_seconds"]
        ),
        "shared_k1_pretrain_best_step": int(pretrain_metadata["best_step"]),
        "initial_component_score_spread_mean": initial_score_spread_mean,
        "initial_component_score_spread_max": initial_score_spread_max,
        "selection_metric": training["selection_metric"],
        "test_used_for_selection": False,
        "effective_batch_size": int(training["effective_batch_size"]),
        "physical_batch_size": int(training["physical_batch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "train_samples": int(training["train_samples"]),
        "validation_samples": int(training["validation_samples"]),
        "test_samples": int(training["test_samples"]),
        "lal_scope": (
            training["lal_scope"] if variant == "lal_bos" else None
        ),
    }
    _write_json(outdir / "result.json", result)


def _summary_rows(outdir: Path) -> list[dict]:
    rows = []
    for path in outdir.glob("seed_*/*/result.json"):
        result = _read_json(path)
        if result.get("status") == "pass":
            rows.append(result)
    return sorted(rows, key=lambda row: (row["seed"], row["variant"]))


def _write_summaries(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "summary.partial.csv", index=False)
    aggregate = (
        frame.groupby("variant", as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            validation_purity_mean=("best_validation_purity", "mean"),
            test_purity_mean=("test_purity_descriptive_only", "mean"),
            test_purity_sd=("test_purity_descriptive_only", "std"),
            test_ari_mean=("test_ari_descriptive_only", "mean"),
            test_ari_sd=("test_ari_descriptive_only", "std"),
            test_nmi_mean=("test_nmi_descriptive_only", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
    )
    aggregate.to_csv(outdir / "aggregate.partial.csv", index=False)
    pivot = frame.pivot(
        index="seed",
        columns="variant",
        values=["test_purity_descriptive_only", "test_ari_descriptive_only"],
    )
    required = {
        ("test_purity_descriptive_only", variant) for variant in VARIANTS
    }
    if required.issubset(set(pivot.columns)):
        paired = pd.DataFrame(index=pivot.index)
        paired["wishart_minus_no_w_purity"] = (
            pivot[("test_purity_descriptive_only", "signed_wishart")]
            - pivot[("test_purity_descriptive_only", "no_wishart")]
        )
        paired["wishart_minus_lal_purity"] = (
            pivot[("test_purity_descriptive_only", "signed_wishart")]
            - pivot[("test_purity_descriptive_only", "lal_bos")]
        )
        paired["wishart_minus_no_w_ari"] = (
            pivot[("test_ari_descriptive_only", "signed_wishart")]
            - pivot[("test_ari_descriptive_only", "no_wishart")]
        )
        paired["wishart_minus_lal_ari"] = (
            pivot[("test_ari_descriptive_only", "signed_wishart")]
            - pivot[("test_ari_descriptive_only", "lal_bos")]
        )
        paired.reset_index().to_csv(
            outdir / "paired_effects.partial.csv", index=False
        )


def _worker_command(
    root: Path,
    *,
    config_path: Path,
    dataset_cache: Path,
    run_dir: Path,
    seed: int,
    variant: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config", str(config_path),
        "--dataset-cache", str(dataset_cache),
        "--outdir", str(run_dir),
        "--seed", str(seed),
        "--variant", variant,
        "--device", device,
    ]


def _run_suite(args, config: dict, root: Path) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    effective_path = outdir / "protocol.json"
    _write_json(effective_path, config)
    cache = outdir / "generated_data" / "wishart_re_sin_k5c5.npz"
    audit = _generate_dataset(config, cache)
    _write_json(outdir / "dgp_audit.json", audit)
    analysis = config["analysis"]
    observable_accuracy = float(
        audit["observable_validation_ridge_lda_accuracy"]
    )
    observable_kmeans_ari = float(audit["observable_all_kmeans_ari"])
    minimum_accuracy = float(
        analysis["minimum_observable_validation_ridge_lda_accuracy"]
    )
    minimum_kmeans_ari = float(
        analysis["minimum_observable_all_kmeans_ari"]
    )
    if (
        observable_accuracy < minimum_accuracy
        or observable_kmeans_ari < minimum_kmeans_ari
    ):
        raise RuntimeError(
            "DGP observable preflight failed: "
            f"ridge-LDA={observable_accuracy:.3f} < {minimum_accuracy:.3f} "
            f"or k-means ARI={observable_kmeans_ari:.3f} "
            f"< {minimum_kmeans_ari:.3f}"
        )
    training = config["training"]
    seeds = (
        list(args.seeds)
        if args.seeds is not None
        else [int(seed) for seed in training["model_seeds"]]
    )
    variants = (
        list(args.variants)
        if args.variants is not None
        else list(training["variants"])
    )
    specifications = [
        (seed, variant) for seed in seeds for variant in variants
    ]
    commands = [
        _worker_command(
            root,
            config_path=effective_path,
            dataset_cache=cache,
            run_dir=outdir / f"seed_{seed}" / variant,
            seed=seed,
            variant=variant,
            device=args.device,
        )
        for seed, variant in specifications
    ]
    _write_json(outdir / "commands.json", commands)
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if args.device == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        gpu = _gpu_sample()
        if gpu is None:
            raise RuntimeError("could not inspect NVIDIA GPU")
        if int(gpu["memory_used_mib"]) >= int(
            config["runtime"]["vram_limit_mib"]
        ):
            raise RuntimeError(f"GPU is already above the safety limit: {gpu}")
    failures = []
    _write_json(outdir / "failures.json", failures)
    for queue_index, ((seed, variant), command) in enumerate(
        zip(specifications, commands, strict=True), start=1
    ):
        run_dir = outdir / f"seed_{seed}" / variant
        result_path = run_dir / "result.json"
        if result_path.is_file() and _read_json(result_path).get("status") == "pass":
            rows = _summary_rows(outdir)
            _write_summaries(outdir, rows)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = _summary_rows(outdir)
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": queue_index,
            "current": {
                "seed": seed,
                "variant": variant,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
        })
        started = time.time()
        if args.device == "cuda":
            returncode, samples, exceeded = _run_with_vram_guard(
                command,
                root=root,
                run_dir=run_dir,
                environment=environment,
                vram_limit_mib=int(config["runtime"]["vram_limit_mib"]),
                poll_seconds=float(config["runtime"]["vram_poll_seconds"]),
            )
            peak = max(
                (sample["memory_used_mib"] for sample in samples),
                default=None,
            )
        else:
            with (run_dir / "worker.stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout, (run_dir / "worker.stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            returncode = process.returncode
            exceeded = False
            peak = None
        if exceeded or returncode != 0 or not result_path.is_file():
            failure = {
                "seed": seed,
                "variant": variant,
                "reason": (
                    "dedicated_vram_safety_limit_exceeded"
                    if exceeded
                    else "worker_failed"
                ),
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "peak_sampled_dedicated_vram_mib": peak,
                "run_dir": str(run_dir),
            }
            failures.append(failure)
            _write_json(outdir / "failures.json", failures)
            _write_json(outdir / "suite_progress.json", {
                "status": "stopped_vram_limit" if exceeded else "failed",
                "completed": len(_summary_rows(outdir)),
                "failed": len(failures),
                "total": len(specifications),
                "current": failure,
                "test_used_for_selection": False,
            })
            return
        result = _read_json(result_path)
        result["peak_sampled_dedicated_vram_mib"] = peak
        _write_json(result_path, result)
        rows = _summary_rows(outdir)
        _write_summaries(outdir, rows)
    rows = _summary_rows(outdir)
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)
    for partial, final in (
        ("aggregate.partial.csv", "aggregate.csv"),
        ("paired_effects.partial.csv", "paired_effects.csv"),
    ):
        source = outdir / partial
        if source.is_file():
            source.replace(outdir / final)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass",
        "completed": len(rows),
        "failed": 0,
        "total": len(specifications),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--within-gain", type=float)
    parser.add_argument("--between-gain", type=float)
    parser.add_argument(
        "--degrees-of-freedom-mode", choices=("fixed", "learned")
    )
    parser.add_argument("--model-degrees-of-freedom", type=float)
    parser.add_argument("--degrees-of-freedom-learning-rate", type=float)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config if args.config.is_absolute() else root / args.config
    ).resolve()
    config = _effective_config(
        _read_json(config_path),
        quick=args.quick,
        within_gain=args.within_gain,
        between_gain=args.between_gain,
        degrees_of_freedom_mode=args.degrees_of_freedom_mode,
        model_degrees_of_freedom=args.model_degrees_of_freedom,
        degrees_of_freedom_learning_rate=(
            args.degrees_of_freedom_learning_rate
        ),
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
    )
    _validate_config(config)
    if args.worker:
        if args.dataset_cache is None or args.seed is None or args.variant is None:
            parser.error("worker mode requires dataset-cache, seed, and variant")
        _run_worker(
            config=config,
            dataset_cache=args.dataset_cache.resolve(),
            outdir=args.outdir.resolve(),
            seed=args.seed,
            variant=args.variant,
            device_name=args.device,
        )
        return
    _run_suite(args, config, root)


if __name__ == "__main__":
    main()
