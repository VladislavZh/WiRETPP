#!/usr/bin/env python3
"""Verify hashes, architecture construction, and all final checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from run_corrected_shared_wishart_architectures import (
    _initial_backbone,
    _wishart_wrapper,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(artifact: Path) -> int:
    manifest = artifact / "MANIFEST.sha256"
    failures = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = artifact / Path(relative)
        if not path.is_file() or _sha256(path) != expected:
            print(f"HASH MISMATCH: {relative}")
            failures += 1
    return failures


def _verify_tables(artifact: Path) -> int:
    """Check the structural claims used by the final report."""

    failures = 0

    def require(condition: bool, message: str) -> None:
        nonlocal failures
        if not condition:
            print(f"TABLE CHECK FAILED: {message}")
            failures += 1

    final = pd.read_csv(artifact / "FINAL_COMPARISON.csv")
    nll = pd.read_csv(artifact / "nll.csv")
    clustering = pd.read_csv(artifact / "clustering.csv")
    architecture = pd.read_csv(artifact / "architecture_audit.csv")
    stability = pd.read_csv(artifact / "mc_stability_256x3_summary.csv")
    require(len(final) == 6, "FINAL_COMPARISON must have six paired rows")
    require(len(nll) == 12, "nll.csv must have twelve model rows")
    require(len(clustering) == 24, "clustering.csv must have 24 rows")
    require(len(architecture) == 6, "architecture audit must have six rows")
    require(
        bool((architecture["encoder_instances"] == 1).all()),
        "every backbone must have exactly one encoder",
    )
    require(
        bool((~architecture["has_independent_component_module_list"]).all()),
        "no architecture may contain independent component encoders",
    )
    require(
        bool(
            (
                architecture["final_output_dimension"]
                == architecture["expected_output_dimension"]
            ).all()
        ),
        "final output dimensions must match the parameterization",
    )
    require(len(stability) == 6, "stability summary must have six rows")
    for final_row in final.itertuples(index=False):
        direct_nll = nll.loc[
            (nll["architecture"] == final_row.architecture)
            & (nll["parameterization"] == final_row.parameterization)
            & (nll["model"] == "no_wishart")
        ].iloc[0]
        direct_cluster = clustering.loc[
            (clustering["architecture"] == final_row.architecture)
            & (clustering["parameterization"] == final_row.parameterization)
            & (clustering["model"] == "no_wishart")
            & (clustering["representation"] == "full_sequence")
        ].iloc[0]
        wishart = stability.loc[
            (stability["architecture"] == final_row.architecture)
            & (stability["parameterization"] == final_row.parameterization)
        ].iloc[0]
        pairs = (
            (final_row.no_w_suffix_nll, direct_nll["suffix_nll_per_exposure"]),
            (final_row.no_w_full_purity, direct_cluster["purity"]),
            (final_row.no_w_full_ari, direct_cluster["ari"]),
            (final_row.wishart_suffix_nll_256_mean, wishart["suffix_nll_mean"]),
            (final_row.wishart_suffix_nll_256_sd, wishart["suffix_nll_sd"]),
            (final_row.wishart_full_purity_256_mean, wishart["full_purity_mean"]),
            (final_row.wishart_full_ari_256_mean, wishart["full_ari_mean"]),
            (final_row.wishart_minimum_active_k, wishart["minimum_full_active_k"]),
        )
        require(
            all(abs(float(left) - float(right)) < 1e-12 for left, right in pairs),
            "FINAL_COMPARISON must be exactly derivable from raw result tables",
        )
    require(
        bool((stability["minimum_full_active_k"] == 3).all()),
        "every Wishart stability repeat must use all three clusters",
    )
    require(
        bool((stability["suffix_nll_sd"] < 0.001).all()),
        "every Wishart suffix-NLL SD must be below 0.001",
    )
    mean_files = sorted((artifact / "learned_means").glob("*.csv"))
    require(len(mean_files) == 6, "six learned Wishart means are required")
    for path in mean_files:
        require(
            len(pd.read_csv(path)) == 225,
            f"{path.name} must contain a 15x15 matrix",
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/"
            "corrected_shared_wishart_output_vs_lal_fixed_300ep_seed0"
        ),
    )
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    failures = _verify_manifest(artifact) + _verify_tables(artifact)
    loaded = 0
    for architecture_index, architecture in enumerate(config["architectures"]):
        for parameterization_index, parameterization in enumerate(
            config["parameterizations"]
        ):
            init_seed = (
                int(config["init_seed"])
                + 10_000 * architecture_index
                + 1_000 * parameterization_index
            )
            kwargs = dict(
                architecture=architecture,
                parameterization=parameterization,
                config=config,
                initial_total_rate=5.0,
                initialization_seed=init_seed,
                device=torch.device("cpu"),
            )
            stem = f"{architecture}_{parameterization}"
            direct = _initial_backbone(**kwargs)
            direct.load_state_dict(
                torch.load(
                    artifact / "checkpoints" / f"{stem}_no_w.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            wishart = _wishart_wrapper(
                _initial_backbone(**kwargs),
                degrees_of_freedom=int(config["degrees_of_freedom"]),
            )
            wishart.load_state_dict(
                torch.load(
                    artifact / "checkpoints" / f"{stem}_wishart.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            print(
                f"OK {stem}: no-W={sum(p.numel() for p in direct.parameters())}, "
                f"W={sum(p.numel() for p in wishart.parameters())}"
            )
            loaded += 2
    if failures:
        raise SystemExit(f"verification failed: {failures} hash mismatches")
    print(
        "PASS: manifest and table invariants valid; "
        f"{loaded} checkpoints loaded strictly"
    )


if __name__ == "__main__":
    main()
