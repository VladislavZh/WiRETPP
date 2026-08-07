#!/usr/bin/env python3
"""Resumable lr_alpha sweep for beta_0=1 THP Wishart on sin_K5_C5."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from run_two_phase_all12_benchmark import _command, _write_json


DATASET = "sin_K5_C5"
ARCHITECTURE = "thp"


def _tag(value: float) -> str:
    return f"lr_alpha_{value:.0e}".replace("e-", "em").replace("e+", "ep")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _replace_argument(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def _source_pretrain(root: Path, seed: int) -> Path:
    if seed == 0:
        return (
            root
            / "artifacts"
            / "dan_all12_best_elimination_seed0_20260804"
            / DATASET
            / ARCHITECTURE
            / "no_wishart"
        )
    return (
        root
        / "artifacts"
        / "dan_all12_best_elimination_seeds1_2_20260805"
        / f"seed_{seed}"
        / DATASET
        / ARCHITECTURE
        / "no_wishart"
    )


def _copy_pretrain(source: Path, target: Path) -> None:
    checkpoint = source / "checkpoint_k1.pt"
    history = source / "pretrain_history.csv"
    if not checkpoint.is_file() or not history.is_file():
        raise FileNotFoundError(f"missing completed no-W pretrain in {source}")
    target.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target / "checkpoint_cotic_k1.pt"
    target_history = target / "pretrain_history.csv"
    if not target_checkpoint.is_file():
        shutil.copy2(checkpoint, target_checkpoint)
    if not target_history.is_file():
        shutil.copy2(history, target_history)


def _read_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(result: dict, lr_alpha: float) -> dict:
    return {
        "lr_alpha": lr_alpha,
        "initial_beta": 1.0,
        "seed": result["seed"],
        "split_seed": result["split_seed"],
        "best_validation_purity": result["best_validation_purity"],
        "test_purity": result["test_purity"],
        "test_ari": result["test_ari"],
        "test_nmi": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "alpha_min": result["alpha_min_during_training"],
        "alpha_max": result["alpha_max_during_training"],
        "alpha_gradient_norm_median": result["alpha_gradient_norm_median"],
        "alpha_gradient_norm_maximum": result["alpha_gradient_norm_maximum"],
        "runtime_seconds": result["runtime_seconds"],
    }


def _write_lr_summary(
    outdir: Path,
    rows: list[dict],
    learning_rates: list[float],
    seeds: list[int],
) -> None:
    aggregate_rows: list[dict] = []
    for learning_rate in learning_rates:
        group = [
            row for row in rows
            if float(row["lr_alpha"]) == learning_rate
        ]
        if len(group) != len(seeds):
            continue
        validation = [float(row["best_validation_purity"]) for row in group]
        tests = [float(row["test_purity"]) for row in group]
        aris = [float(row["test_ari"]) for row in group]
        aggregate_rows.append({
            "lr_alpha": learning_rate,
            "initial_beta": 1.0,
            "n_seeds": len(group),
            "mean_best_validation_purity": statistics.mean(validation),
            "sample_std_best_validation_purity": statistics.stdev(validation),
            "mean_test_purity_descriptive_only": statistics.mean(tests),
            "sample_std_test_purity_descriptive_only": statistics.stdev(tests),
            "mean_test_ari_descriptive_only": statistics.mean(aris),
        })
    if not aggregate_rows:
        return
    ranked = sorted(
        aggregate_rows,
        key=lambda row: (
            -float(row["mean_best_validation_purity"]),
            float(row["sample_std_best_validation_purity"]),
            float(row["lr_alpha"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["validation_selection_rank"] = rank
        row["selected_by_validation"] = rank == 1
    _write_csv(outdir / "lr_summary.csv", ranked)
    _write_json(outdir / "selection.json", {
        "selection_policy": (
            "maximize mean best_validation_purity across seeds; "
            "tie-break by lower sample validation std, then lower lr_alpha"
        ),
        "test_used_for_selection": False,
        "selected_lr_alpha": ranked[0]["lr_alpha"],
        "selected_validation_mean": ranked[0][
            "mean_best_validation_purity"
        ],
        "selected_validation_sample_std": ranked[0][
            "sample_std_best_validation_purity"
        ],
        "selected_test_purity_mean_descriptive_only": ranked[0][
            "mean_test_purity_descriptive_only"
        ],
        "selected_test_purity_sample_std_descriptive_only": ranked[0][
            "sample_std_test_purity_descriptive_only"
        ],
        "ranking": ranked,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_sin_k5_thp_beta1_lr_alpha_sweep_seeds0_1_2_20260806"
        ),
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.learning_rates):
        parser.error("all --learning-rates must be positive")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("--learning-rates must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        parser.error("all --seeds must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    root = Path(__file__).resolve().parents[1]
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (learning_rate, seed)
        for learning_rate in args.learning_rates
        for seed in args.seeds
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for index, (learning_rate, seed) in enumerate(specifications, start=1):
        run_dir = outdir / _tag(learning_rate) / f"seed_{seed}"
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = _read_result(result_path)
            if result.get("status") == "pass":
                rows.append(_row(result, learning_rate))
                _write_csv(outdir / "summary.partial.csv", rows)
                _write_lr_summary(
                    outdir,
                    rows,
                    list(args.learning_rates),
                    list(args.seeds),
                )
                continue

        _copy_pretrain(_source_pretrain(root, seed), run_dir)
        command = _command(
            root,
            seed=seed,
            dataset=DATASET,
            architecture=ARCHITECTURE,
            variant="wishart",
            outdir=run_dir,
            wishart_interaction_mode="convex",
            wishart_initial_alpha=0.1,
            wishart_alpha_parameterization="projected",
            wishart_alpha_temperature=1.0,
        )
        _replace_argument(
            command,
            "--alpha-learning-rate",
            str(learning_rate),
        )
        _replace_argument(command, "--initial-beta", "1.0")

        started = time.time()
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": index,
            "current": {
                "seed": seed,
                "dataset": DATASET,
                "architecture": ARCHITECTURE,
                "variant": "wishart",
                "initial_beta": 1.0,
                "lr_alpha": learning_rate,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
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
                "lr_alpha": learning_rate,
                "seed": seed,
                "returncode": process.returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = _read_result(result_path)
        rows.append(_row(result, learning_rate))
        _write_csv(outdir / "summary.partial.csv", rows)
        _write_lr_summary(
            outdir,
            rows,
            list(args.learning_rates),
            list(args.seeds),
        )

    _write_csv(outdir / "summary.csv", rows)
    _write_lr_summary(
        outdir,
        rows,
        list(args.learning_rates),
        list(args.seeds),
    )
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(specifications),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
