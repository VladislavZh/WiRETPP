#!/usr/bin/env python3
"""Wishart convex-interaction ablation on the two completed sine datasets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_two_phase_all12_benchmark import (
    _command,
    _published_lal,
    _write_json,
)


DATASETS = ("sin_K5_C5", "sin_K4_C5")
ARCHITECTURES = ("thp", "cotic")


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _copy_pretrain(source_run: Path, target_run: Path) -> None:
    source_checkpoint = source_run / "checkpoint_k1.pt"
    source_history = source_run / "pretrain_history.csv"
    if not source_checkpoint.is_file() or not source_history.is_file():
        raise FileNotFoundError(
            f"paired no-W K=1 pretrain is incomplete: {source_run}"
        )
    target_run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, target_run / "checkpoint_cotic_k1.pt")
    shutil.copy2(source_history, target_run / "pretrain_history.csv")


def _baseline(path: Path) -> dict:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/dan_two_phase_all12_seed0_20260803"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_two_phase_convex_interaction_sin_seed0_20260803"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_root = args.baseline_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    lal = _published_lal(root)
    specifications = [
        (dataset, architecture)
        for dataset in DATASETS
        for architecture in ARCHITECTURES
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)
    for index, (dataset, architecture) in enumerate(specifications, start=1):
        run_dir = outdir / dataset / architecture / "wishart_convex"
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                residual = _baseline(
                    baseline_root / dataset / architecture / "wishart"
                )
                no_w = _baseline(
                    baseline_root / dataset / architecture / "no_wishart"
                )
                rows.append(_row(result, residual, no_w, lal[dataset]))
                _write_summary(outdir / "summary.partial.csv", rows)
                continue
        _copy_pretrain(
            baseline_root / dataset / architecture / "no_wishart",
            run_dir,
        )
        command = _command(
            root,
            dataset=dataset,
            architecture=architecture,
            variant="wishart",
            outdir=run_dir,
        )
        command.extend(("--interaction-mode", "convex"))
        started = time.time()
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": index,
            "current": {
                "dataset": dataset,
                "architecture": architecture,
                "variant": "wishart_convex",
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
                "returncode": process.returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        residual = _baseline(
            baseline_root / dataset / architecture / "wishart"
        )
        no_w = _baseline(
            baseline_root / dataset / architecture / "no_wishart"
        )
        rows.append(_row(result, residual, no_w, lal[dataset]))
        _write_summary(outdir / "summary.partial.csv", rows)
    _write_summary(outdir / "summary.csv", rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(specifications),
        "current": None,
        "last_completed": rows[-1] if rows else None,
    })


def _row(
    result: dict,
    residual: dict,
    no_w: dict,
    lal: tuple[float, float],
) -> dict:
    purity = float(result["test_purity"])
    residual_purity = float(residual["test_purity"])
    no_w_purity = float(no_w["test_purity"])
    return {
        "dataset": result["dataset"],
        "architecture": result["architecture"],
        "variant": "wishart_convex_interaction",
        "best_validation_purity": result["best_validation_purity"],
        "test_purity": purity,
        "test_ari": result["test_ari"],
        "test_nmi": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "residual_wishart_test_purity": residual_purity,
        "purity_minus_residual_wishart": purity - residual_purity,
        "no_w_test_purity": no_w_purity,
        "purity_minus_no_w": purity - no_w_purity,
        "lal_moitpp_mean": lal[0],
        "lal_moitpp_sd": lal[1],
        "purity_minus_lal_mean": purity - lal[0],
        "runtime_seconds": result["runtime_seconds"],
    }


if __name__ == "__main__":
    main()
