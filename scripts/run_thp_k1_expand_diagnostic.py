#!/usr/bin/env python3
"""Diagnose K=1 pretraining followed by direct K expansion for THP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from lal_wishart.metrics import (
    adjusted_rand_index,
    cluster_purity,
    normalized_mutual_information,
)
from lal_wishart.models.reference_output_mixtures import (
    ReferenceTHPOutputMixture,
)
from lal_wishart.reproduction.dan_synthetic import load_dataset, shuffled_split
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_direct_nhp_mixture,
    fit_direct_nhp_mixture,
)
from run_thp_overcomplete_backward_elimination import _pretrain_k1, _set_seed


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="K2_C5")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument(
        "--head-initialization",
        choices=("clone_noise", "random_base"),
        default="clone_noise",
    )
    parser.add_argument("--head-initialization-noise", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.pretrain_steps < 0 or args.train_steps <= 0:
        raise ValueError("training step counts must be non-negative/positive")
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_root, args.dataset)
    if dataset.n_components != 2:
        raise ValueError("this diagnostic currently requires target K=2")
    split = shuffled_split(dataset, seed=args.split_seed)
    device = torch.device(args.device)
    initialization_seed = 2026121100 + args.seed * 1_000_000
    _set_seed(initialization_seed)
    model = ReferenceTHPOutputMixture(
        1 if args.pretrain_steps > 0 else dataset.n_components,
        dataset.n_marks,
        horizon=max(sequence.horizon for sequence in dataset.sequences),
        quadrature_order=4,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        initialization_seed=initialization_seed,
    ).to(device)
    if args.pretrain_steps > 0:
        _pretrain_k1(
            model,
            split.train,
            split.validation,
            steps=args.pretrain_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            rng=np.random.default_rng(initialization_seed + 101),
            outdir=args.outdir,
        )
        if args.head_initialization == "clone_noise":
            model.expand_components(
                dataset.n_components,
                noise_scale=args.head_initialization_noise,
                initialization_seed=initialization_seed + 17,
            )
        else:
            # Preserve the fitted shared representation while replacing the
            # marginal K=1 output with a fresh K-way output.  This isolates
            # whether the fitted K=1 head itself anchors the mixture in a bad
            # basin after expansion.
            pretrained_backbone = model.backbone.state_dict()
            fresh_model = ReferenceTHPOutputMixture(
                dataset.n_components,
                dataset.n_marks,
                horizon=max(
                    sequence.horizon for sequence in dataset.sequences
                ),
                quadrature_order=4,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
                initialization_seed=initialization_seed + 10_003,
                output_noise_scale=args.head_initialization_noise,
            ).to(device)
            fresh_model.backbone.load_state_dict(pretrained_backbone)
            model = fresh_model
    fit = fit_direct_nhp_mixture(
        model,
        split.train,
        split.validation,
        max_epochs=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        neural_weight_decay=args.weight_decay,
        evaluation_interval=1,
        validation_cutoff=None,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_clip=args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_seed=initialization_seed + 301,
        validation_labels=split.validation_labels,
        selection_metric="validation_purity",
        show_progress=True,
        progress_description="THP K1->K2 direct",
    )
    model = fit.model
    evaluation = evaluate_direct_nhp_mixture(
        model,
        split.test,
        cutoff=None,
        batch_size=args.evaluation_batch_size,
    )
    probabilities = evaluation.full_cluster_probabilities.cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    purity = cluster_purity(split.test_labels, predictions)
    exposure = sum(sequence.horizon for sequence in split.test)
    test_nll = float(-evaluation.full_marginal_scores.sum() / exposure)
    pd.DataFrame(fit.history).to_csv(args.outdir / "history.csv", index=False)
    prediction_frame = pd.DataFrame({
        "source_id": split.test_ids,
        "true_cluster": split.test_labels,
        "predicted_cluster": predictions,
    })
    for component in range(dataset.n_components):
        prediction_frame[f"probability_{component}"] = probabilities[:, component]
    prediction_frame.to_csv(args.outdir / "predictions.csv", index=False)
    torch.save(model.state_dict(), args.outdir / "checkpoint.pt")
    result = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.name,
        "architecture": "thp",
        "variant": (
            "no_wishart_k1_expand_direct"
            if args.pretrain_steps > 0
            else "no_wishart_direct_same_seed"
        ),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "initialization_seed": initialization_seed,
        "pretrain_steps": args.pretrain_steps,
        "train_steps": args.train_steps,
        "target_components": dataset.n_components,
        "head_initialization": args.head_initialization,
        "head_initialization_noise": args.head_initialization_noise,
        "assignment_objective": "ordinary_mixture_nll",
        "dpp_strength": 0.0,
        "pruning_steps": 0,
        "best_step": fit.best_epoch,
        "best_validation_purity": fit.best_validation_purity,
        "best_validation_nll_per_exposure": (
            fit.best_validation_suffix_nll_per_exposure
        ),
        "test_purity": purity,
        "test_ari": adjusted_rand_index(split.test_labels, predictions),
        "test_nmi": normalized_mutual_information(
            split.test_labels, predictions
        ),
        "test_cluster_sizes": np.bincount(
            predictions, minlength=dataset.n_components
        ).tolist(),
        "test_nll_per_exposure": test_nll,
        "runtime_seconds": time.time() - started,
    }
    _write_json(args.outdir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
