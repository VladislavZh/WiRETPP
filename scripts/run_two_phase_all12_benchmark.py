#!/usr/bin/env python3
"""Resumable best-elimination THP/COTIC benchmark on all DAN datasets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


DATASETS = (
    "sin_K5_C5",
    "sin_K4_C5",
    "sin_K3_C5",
    "sin_K2_C5",
    "K5_C5",
    "K4_C5",
    "K3_C5",
    "K2_C5",
    "trunc_K5_C5",
    "trunc_K4_C5",
    "trunc_K3_C5",
    "trunc_K2_C5",
)
ARCHITECTURES = ("thp", "cotic")
VARIANTS = ("no_wishart", "wishart")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _reuse_paired_k1_pretrain(run_dir: Path) -> bool:
    """Seed a Wishart run with the identical completed no-W K=1 fit."""

    paired = run_dir.parent / "no_wishart"
    source_checkpoint = paired / "checkpoint_k1.pt"
    source_history = paired / "pretrain_history.csv"
    target_checkpoint = run_dir / "checkpoint_cotic_k1.pt"
    target_history = run_dir / "pretrain_history.csv"
    if (
        target_checkpoint.is_file()
        and target_history.is_file()
    ):
        return True
    if not source_checkpoint.is_file() or not source_history.is_file():
        return False
    shutil.copy2(source_checkpoint, target_checkpoint)
    shutil.copy2(source_history, target_history)
    return True


def _target_components(dataset: str) -> int:
    match = re.search(r"K(\d+)_C", dataset)
    if match is None:
        raise ValueError(f"cannot read K from {dataset}")
    return int(match.group(1))


def _published_lal(root: Path) -> dict[str, tuple[float, float]]:
    path = (
        root
        / "artifacts"
        / "dan_all_arch_newwishart_beta_cotic7_seed0_20260802"
        / "PUBLISHED_DAN_PURITY.csv"
    )
    values = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values[row["dataset"]] = (
                float(row["dan_moitpp_mean"]),
                float(row["dan_moitpp_sd"]),
            )
    return values


def _command(
    root: Path,
    *,
    seed: int,
    dataset: str,
    architecture: str,
    variant: str,
    outdir: Path,
    wishart_interaction_mode: str = "convex",
    wishart_initial_alpha: float = 0.1,
    wishart_alpha_parameterization: str = "projected",
    wishart_alpha_temperature: float = 1.0,
) -> list[str]:
    target = _target_components(dataset)
    initial = target + 5
    shared = [
        "--architecture", architecture,
        "--data-root", str(root / "data"),
        "--dataset", dataset,
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--split-seed", "42",
        "--initial-components", str(initial),
        "--target-components", str(target),
        "--warmup-steps", "300",
        "--prune-interval", "100",
        "--post-pruning-steps", "600",
        # Preserve the effective batch of 128 paths for both variants.
        "--batch-size", "16" if variant == "wishart" else "64",
        "--evaluation-batch-size", "16",
        "--gradient-accumulation-steps", "8" if variant == "wishart" else "2",
        "--gradient-clip", "20",
        "--initial-rho", "5",
        "--final-rho", "5",
        "--nu-multiplier", "1.5",
        "--pretrain-steps", "10",
        "--backbone-ramp-steps", "200",
        "--head-initialization-noise", "0.01",
        "--dpp-strength", "0",
        "--dpp-bandwidth", "1.0",
        "--dpp-jitter", "1e-4",
        "--dpp-decay-steps", "200",
        "--device", "cuda",
    ]
    if variant == "no_wishart":
        return [
            sys.executable,
            str(root / "scripts" / "run_thp_overcomplete_backward_elimination.py"),
            *shared,
            "--dual-learning-rate", "0.05",
            "--backbone-learning-rate", "1e-3",
            "--learning-rate", "1e-3",
            "--cotic-calibration-steps", "0",
        ]
    return [
        sys.executable,
        str(root / "scripts" / "run_wishart_overcomplete_backward_elimination.py"),
        *shared,
        # These two rates are part of the selected sin_K5_C5 protocol.
        "--dual-learning-rate", "0.01",
        "--backbone-learning-rate", "1e-5",
        "--neural-learning-rate", "1e-3",
        "--omega-learning-rate", "0.01",
        "--alpha-learning-rate", "0.0001",
        "--train-samples", "2",
        "--validation-samples", "8",
        "--test-samples", "16",
        "--pruning-samples", "8",
        "--alpha-temperature", str(wishart_alpha_temperature),
        "--initial-alpha", str(wishart_initial_alpha),
        "--alpha-parameterization", wishart_alpha_parameterization,
        "--interaction-mode", wishart_interaction_mode,
        "--mean-hyperprior-strength", "1.566",
        "--initial-beta", "0.9",
        "--beta-decay-rate", "5",
        "--exploration-nu", "200",
    ]


def _summary_row(
    result: dict,
    *,
    variant: str,
    lal: tuple[float, float],
) -> dict:
    return {
        "dataset": result["dataset"],
        "architecture": result["architecture"],
        "variant": variant,
        "seed": result["seed"],
        "best_validation_purity": result["best_validation_purity"],
        "test_purity": result["test_purity"],
        "test_ari": result["test_ari"],
        "test_nmi": result["test_nmi"],
        "test_nll_per_exposure": result["test_nll_per_exposure"],
        "best_step": result["best_step"],
        "selected_alpha": result.get("best_alpha"),
        "alpha_parameterization": result.get("alpha_parameterization"),
        "interaction_mode": result.get("interaction_mode"),
        "lal_moitpp_mean": lal[0],
        "lal_moitpp_sd": lal[1],
        "purity_minus_lal_mean": result["test_purity"] - lal[0],
        "runtime_seconds": result["runtime_seconds"],
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/dan_all12_best_elimination_corrected_seed0_20260805"),
    )
    parser.add_argument(
        "--wishart-interaction-mode",
        choices=("residual", "convex"),
        default="convex",
    )
    parser.add_argument("--wishart-initial-alpha", type=float, default=0.1)
    parser.add_argument(
        "--wishart-alpha-parameterization",
        choices=("sigmoid", "projected"),
        default="projected",
    )
    parser.add_argument(
        "--wishart-alpha-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Run all variants by default; accepts wishart-only correction runs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Model seeds to run sequentially; the data split seed remains 42.",
    )
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    root = Path(__file__).resolve().parents[1]
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    lal = _published_lal(root)
    specifications = [
        (seed, dataset, architecture, variant)
        for seed in args.seeds
        for dataset in DATASETS
        for architecture in ARCHITECTURES
        for variant in args.variants
    ]
    rows = []
    failures = []
    _write_json(outdir / "failures.json", failures)
    for index, (seed, dataset, architecture, variant) in enumerate(
        specifications, start=1
    ):
        run_dir = (
            outdir
            / f"seed_{seed}"
            / dataset
            / architecture
            / variant
        )
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                rows.append(
                    _summary_row(result, variant=variant, lal=lal[dataset])
                )
                _write_summary(outdir / "summary.partial.csv", rows)
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        if variant == "wishart":
            _reuse_paired_k1_pretrain(run_dir)
        command = _command(
            root,
            seed=seed,
            dataset=dataset,
            architecture=architecture,
            variant=variant,
            outdir=run_dir,
            wishart_interaction_mode=args.wishart_interaction_mode,
            wishart_initial_alpha=args.wishart_initial_alpha,
            wishart_alpha_parameterization=(
                args.wishart_alpha_parameterization
            ),
            wishart_alpha_temperature=args.wishart_alpha_temperature,
        )
        started = time.time()
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": index,
            "current": {
                "seed": seed,
                "dataset": dataset,
                "architecture": architecture,
                "variant": variant,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
        })
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0 or not result_path.is_file():
            failures.append({
                "dataset": dataset,
                "architecture": architecture,
                "variant": variant,
                "returncode": process.returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_summary_row(result, variant=variant, lal=lal[dataset]))
        _write_summary(outdir / "summary.partial.csv", rows)
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": index,
            "current": None,
            "last_completed": rows[-1],
        })
    _write_summary(outdir / "summary.csv", rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(specifications),
        "current": None,
        "last_completed": rows[-1] if rows else None,
    })
    print(json.dumps({
        "completed": len(rows),
        "failed": len(failures),
        "summary": str(outdir / "summary.csv"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
