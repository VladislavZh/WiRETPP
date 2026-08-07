#!/usr/bin/env python3
"""Temperature sweep for convex-interaction THP Wishart on sin_K5_C5."""

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

from run_two_phase_all12_benchmark import _command, _write_json


def _tag(value: float) -> str:
    return f"T{value:g}".replace(".", "p")


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _copy_pretrain(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "checkpoint_k1.pt", target / "checkpoint_cotic_k1.pt")
    shutil.copy2(source / "pretrain_history.csv", target / "pretrain_history.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/dan_two_phase_all12_seed0_20260803"),
    )
    parser.add_argument(
        "--convex-root",
        type=Path,
        default=Path(
            "artifacts/dan_two_phase_convex_interaction_sin_seed0_20260803"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_convex_temperature_sweep_sin_k5_thp_seed0_20260803"
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=(0.5, 1.943, 4.0),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_root = args.baseline_root.resolve()
    convex_root = args.convex_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    source_pretrain = baseline_root / "sin_K5_C5" / "thp" / "no_wishart"
    residual = json.loads(
        (
            baseline_root
            / "sin_K5_C5"
            / "thp"
            / "wishart"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    convex_t1 = json.loads(
        (
            convex_root
            / "sin_K5_C5"
            / "thp"
            / "wishart_convex"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)
    for index, temperature in enumerate(args.temperatures, start=1):
        run_dir = outdir / _tag(temperature)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                rows.append(_row(result, temperature, convex_t1, residual))
                _write_summary(outdir / "summary.partial.csv", rows)
                continue
        _copy_pretrain(source_pretrain, run_dir)
        command = _command(
            root,
            dataset="sin_K5_C5",
            architecture="thp",
            variant="wishart",
            outdir=run_dir,
        )
        command.extend((
            "--interaction-mode",
            "convex",
            "--alpha-temperature",
            str(temperature),
        ))
        started = time.time()
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(args.temperatures),
            "queue_index": index,
            "current_temperature": temperature,
            "current_run_dir": str(run_dir),
            "started_utc": datetime.now(timezone.utc).isoformat(),
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
                "temperature": temperature,
                "returncode": process.returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_row(result, temperature, convex_t1, residual))
        _write_summary(outdir / "summary.partial.csv", rows)
    _write_summary(outdir / "summary.csv", rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(args.temperatures),
        "current_temperature": None,
        "last_completed": rows[-1] if rows else None,
    })


def _row(
    result: dict,
    temperature: float,
    convex_t1: dict,
    residual: dict,
) -> dict:
    purity = float(result["test_purity"])
    return {
        "alpha_temperature": temperature,
        "best_validation_purity": result["best_validation_purity"],
        "test_purity": purity,
        "test_ari": result["test_ari"],
        "test_nmi": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "purity_minus_convex_T1": purity - float(convex_t1["test_purity"]),
        "purity_minus_residual": purity - float(residual["test_purity"]),
        "runtime_seconds": result["runtime_seconds"],
    }


if __name__ == "__main__":
    main()
