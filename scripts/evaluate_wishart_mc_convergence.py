#!/usr/bin/env python3
"""Measure clustering-metric convergence versus Wishart MC sample count."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
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
)
from run_corrected_shared_wishart_architectures import (
    _initial_backbone,
    _set_seed,
)
from run_dan_synthetic_benchmark import _evaluation_batch_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="sin_K5_C5")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--candidate", default="baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        any(seed < 0 for seed in args.seeds)
        or any(samples <= 0 for samples in args.samples)
        or args.repeats <= 0
    ):
        raise ValueError("seeds must be non-negative; samples/repeats positive")

    names = dataset_names(args.data_root)
    if args.dataset not in names:
        raise ValueError(f"unknown dataset: {args.dataset}")
    dataset_index = names.index(args.dataset)
    dataset = load_dataset(args.data_root, args.dataset)
    device = torch.device(args.device)
    evaluation_batch_size = _evaluation_batch_size(
        max(sequence.count for sequence in dataset.sequences)
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
    rows: list[dict[str, float | int | str]] = []
    for model_seed in args.seeds:
        job_dir = (
            args.checkpoint_root
            / args.candidate
            / f"seed_{model_seed}"
        )
        result = json.loads(
            (job_dir / "result.json").read_text(encoding="utf-8")
        )
        split_seed = int(result["split_seed"])
        split = shuffled_split(dataset, seed=split_seed)
        train = split.train
        initial_total_rate = sum(
            sequence.count for sequence in train
        ) / sum(sequence.horizon for sequence in train)
        init_seed = (
            2026090100
            + model_seed * 1_000_000
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
        parameters = result["parameters"]
        model = SignedWishartTPP(
            copy.deepcopy(initial),
            degrees_of_freedom=int(parameters["wishart_nu"]),
            alpha_max=1.0,
            initial_alpha=0.5,
            interaction_temperature=float(parameters["alpha_temperature"]),
        ).to(device)
        model.load_state_dict(torch.load(
            job_dir / "checkpoint.pt",
            map_location=device,
            weights_only=True,
        ))
        for split_name, sequences, labels, offset in (
            (
                "validation",
                split.validation,
                split.validation_labels,
                100_000,
            ),
            ("test", split.test, split.test_labels, 900_000),
        ):
            exposure = sum(sequence.horizon for sequence in sequences)
            for samples in args.samples:
                sample_batch_size = min(
                    evaluation_batch_size,
                    max(1, evaluation_batch_size * 8 // samples),
                )
                for repeat in range(args.repeats):
                    evaluation = evaluate_latent_wishart_nhp(
                        model,
                        sequences,
                        cutoff=None,
                        n_samples=samples,
                        sample_seed=(
                            sample_seed + offset + repeat * 1_000_003
                        ),
                        batch_size=sample_batch_size,
                    )
                    predictions = (
                        evaluation.full_cluster_probabilities
                        .argmax(dim=1)
                        .numpy()
                    )
                    rows.append({
                        "model_seed": model_seed,
                        "split_seed": split_seed,
                        "split": split_name,
                        "samples": samples,
                        "repeat": repeat,
                        "purity": cluster_purity(labels, predictions),
                        "ari": adjusted_rand_index(labels, predictions),
                        "nmi": normalized_mutual_information(
                            labels,
                            predictions,
                        ),
                        "nll_per_time": float(
                            -evaluation.full_marginal_scores.sum() / exposure
                        ),
                        "active_clusters": int(
                            np.unique(predictions).size
                        ),
                    })

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "MC_CONVERGENCE_RUNS.csv", index=False)
    aggregate = (
        frame.groupby(["split", "samples"], as_index=False)
        .agg(
            purity_mean=("purity", "mean"),
            purity_sd=("purity", "std"),
            purity_min=("purity", "min"),
            purity_max=("purity", "max"),
            ari_mean=("ari", "mean"),
            nll_mean=("nll_per_time", "mean"),
            nll_sd=("nll_per_time", "std"),
        )
        .sort_values(["split", "samples"])
    )
    aggregate.to_csv(args.outdir / "MC_CONVERGENCE_AGGREGATE.csv", index=False)
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
