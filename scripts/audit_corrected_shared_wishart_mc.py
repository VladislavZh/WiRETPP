#!/usr/bin/env python3
"""Repeat final corrected-Wishart evaluation with fresh 256-sample draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lal_wishart.metrics import adjusted_rand_index, cluster_purity
from lal_wishart.experiment import make_splits, write_manifest
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_latent_wishart_nhp,
)

from run_corrected_shared_wishart_architectures import (
    _initial_backbone,
    _wishart_wrapper,
)


def sequences(split):
    return split.sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/"
            "corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0"
        ),
    )
    args = parser.parse_args()
    config = json.loads((args.artifact / "config.json").read_text())
    device = torch.device(args.device)
    train, _, test, _ = make_splits(
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
    test_sequences = sequences(test)
    initial_total_rate = float(
        np.mean([sequence.count for sequence in train_sequences])
        / float(config["horizon"])
    )
    cutoff = float(config["cutoff"])
    suffix_exposure = len(test_sequences) * (
        float(config["horizon"]) - cutoff
    )
    full_exposure = len(test_sequences) * float(config["horizon"])
    rows = []
    for architecture_index, architecture in enumerate(
        config["architectures"]
    ):
        for parameterization_index, parameterization in enumerate(
            config["parameterizations"]
        ):
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
            model = _wishart_wrapper(
                initial,
                degrees_of_freedom=int(config["degrees_of_freedom"]),
            ).to(device)
            checkpoint = (
                args.artifact
                / "checkpoints"
                / f"{architecture}_{parameterization}_wishart.pt"
            )
            model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location=device,
                    weights_only=True,
                )
            )
            for repeat in range(args.repeats):
                evaluation = evaluate_latent_wishart_nhp(
                    model,
                    test_sequences,
                    cutoff=cutoff,
                    n_samples=args.samples,
                    sample_seed=(
                        2026090701
                        + repeat * 1_000_000
                        + architecture_index * 100_000
                        + parameterization_index * 10_000
                    ),
                    batch_size=int(config["evaluation_batch_size"]),
                )
                prefix = (
                    evaluation.prefix_cluster_probabilities.argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                full = (
                    evaluation.full_cluster_probabilities.argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                rows.append({
                    "architecture": architecture,
                    "parameterization": parameterization,
                    "repeat": repeat,
                    "samples": args.samples,
                    "suffix_nll_per_exposure": float(
                        -evaluation.conditional_suffix_scores.sum()
                        / suffix_exposure
                    ),
                    "full_nll_per_exposure": float(
                        -evaluation.full_marginal_scores.sum()
                        / full_exposure
                    ),
                    "prefix_purity": cluster_purity(test.labels, prefix),
                    "prefix_ari": adjusted_rand_index(
                        test.labels,
                        prefix,
                    ),
                    "full_purity": cluster_purity(test.labels, full),
                    "full_ari": adjusted_rand_index(test.labels, full),
                    "prefix_active_k": int(np.unique(prefix).size),
                    "full_active_k": int(np.unique(full).size),
                })
            print(
                f"audited {architecture} {parameterization}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.artifact / "mc_stability_256x3.csv", index=False)
    summary = (
        frame.groupby(["architecture", "parameterization"], as_index=False)
        .agg(
            suffix_nll_mean=("suffix_nll_per_exposure", "mean"),
            suffix_nll_sd=("suffix_nll_per_exposure", "std"),
            full_nll_mean=("full_nll_per_exposure", "mean"),
            prefix_purity_mean=("prefix_purity", "mean"),
            prefix_ari_mean=("prefix_ari", "mean"),
            full_purity_mean=("full_purity", "mean"),
            full_purity_sd=("full_purity", "std"),
            full_ari_mean=("full_ari", "mean"),
            full_ari_sd=("full_ari", "std"),
            minimum_full_active_k=("full_active_k", "min"),
        )
    )
    summary.to_csv(
        args.artifact / "mc_stability_256x3_summary.csv",
        index=False,
    )
    nll = pd.read_csv(args.artifact / "nll.csv")
    clustering = pd.read_csv(args.artifact / "clustering.csv")
    comparison_rows = []
    for architecture in config["architectures"]:
        for parameterization in config["parameterizations"]:
            values = summary.loc[
                (summary["architecture"] == architecture)
                & (summary["parameterization"] == parameterization)
            ].iloc[0]
            selection = (
                (nll["architecture"] == architecture)
                & (nll["parameterization"] == parameterization)
                & (nll["model"] == "no_wishart")
            )
            direct_nll = nll.loc[selection].iloc[0]
            cluster_selection = (
                (clustering["architecture"] == architecture)
                & (clustering["parameterization"] == parameterization)
                & (clustering["model"] == "no_wishart")
                & (clustering["representation"] == "full_sequence")
            )
            direct_cluster = clustering.loc[cluster_selection].iloc[0]
            comparison_rows.append({
                "architecture": architecture,
                "parameterization": parameterization,
                "no_w_suffix_nll": direct_nll[
                    "suffix_nll_per_exposure"
                ],
                "no_w_full_purity": direct_cluster["purity"],
                "no_w_full_ari": direct_cluster["ari"],
                "wishart_suffix_nll_256_mean": values[
                    "suffix_nll_mean"
                ],
                "wishart_suffix_nll_256_sd": values["suffix_nll_sd"],
                "wishart_full_purity_256_mean": values[
                    "full_purity_mean"
                ],
                "wishart_full_ari_256_mean": values["full_ari_mean"],
                "wishart_minimum_active_k": values[
                    "minimum_full_active_k"
                ],
            })
    pd.DataFrame(comparison_rows).to_csv(
        args.artifact / "FINAL_COMPARISON.csv",
        index=False,
    )
    write_manifest(args.artifact)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
