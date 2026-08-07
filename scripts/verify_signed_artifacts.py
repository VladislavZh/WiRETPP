#!/usr/bin/env python3
"""Verify signed WiRE-TPP manifests, tables, and strict checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from run_corrected_shared_wishart_architectures import _initial_backbone
from run_signed_wishart_experiment import _wrapper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/signed_wishart_pilot_seed0"),
    )
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for line in (artifact / "MANIFEST.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = artifact / relative
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    architectures = list(config["architectures"])
    variants = [
        variant
        for variant in config["variants"]
        if variant != "no_wishart"
    ]
    nll = pd.read_csv(artifact / "nll.csv")
    clustering = pd.read_csv(artifact / "clustering.csv")
    architecture = pd.read_csv(artifact / "architecture_audit.csv")
    stability = pd.read_csv(artifact / "mc_stability.csv")
    summary = pd.read_csv(artifact / "mc_stability_summary.csv")
    final = pd.read_csv(artifact / "FINAL_COMPARISON.csv")
    expected_models = len(architectures) * (1 + len(variants))
    checks = {
        "one nll row per trained model": len(nll) == expected_models,
        "two clustering rows per trained model": (
            len(clustering) == 2 * expected_models
        ),
        "one architecture audit per backbone": (
            len(architecture) == len(architectures)
        ),
        "one stability summary per random-effect model": (
            len(summary) == len(architectures) * len(variants)
        ),
        "one final row per random-effect model": (
            len(final) == len(architectures) * len(variants)
        ),
        "all architecture audits have one encoder": bool(
            (architecture["encoder_instances"] == 1).all()
        ),
        "alpha-zero rows are exactly zero": bool(
            (
                final.loc[
                    final["variant"] == "signed_alpha_zero", "alpha"
                ]
                == 0.0
            ).all()
        ),
        "learned alpha rows remain bounded": bool(
            final.loc[
                final["variant"].isin(
                    ["signed", "signed_deterministic"]
                ),
                "alpha",
            ].between(0.0, float(config["alpha_max"])).all()
        ),
        "stability table has every repeat": (
            len(stability)
            == len(architectures)
            * len(variants)
            * int(config["audit_repeats"])
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)

    loaded = 0
    device = torch.device("cpu")
    for architecture_index, architecture_name in enumerate(architectures):
        init_seed = int(config["init_seed"]) + 10_000 * architecture_index
        initial = _initial_backbone(
            architecture_name,
            "output_split",
            config,
            initial_total_rate=5.0,
            initialization_seed=init_seed,
            device=device,
        )
        stem = f"{architecture_name}_output_split"
        direct_state = torch.load(
            artifact / "checkpoints" / f"{stem}_no_wishart.pt",
            map_location="cpu",
            weights_only=True,
        )
        initial.load_state_dict(direct_state, strict=True)
        loaded += 1
        for variant in variants:
            model = _wrapper(variant, initial, config).to(device)
            state = torch.load(
                artifact / "checkpoints" / f"{stem}_{variant}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            loaded += 1

    if failures:
        raise SystemExit("verification failed:\n- " + "\n- ".join(failures))
    print(
        "PASS: signed manifest and table invariants valid; "
        f"{loaded} checkpoints loaded strictly"
    )


if __name__ == "__main__":
    main()
