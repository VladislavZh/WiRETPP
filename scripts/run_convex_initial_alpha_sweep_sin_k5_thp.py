#!/usr/bin/env python3
"""Initial-alpha sweep for convex-interaction THP Wishart on sin_K5_C5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from run_convex_temperature_sweep_sin_k5_thp import (
    _copy_pretrain,
    _write_summary,
)
from run_two_phase_all12_benchmark import _command, _write_json


def _tag(value: float) -> str:
    return f"alpha0_{value:g}".replace(".", "p")


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
            "artifacts/dan_convex_initial_alpha_sweep_sin_k5_thp_seed0_20260803"
        ),
    )
    parser.add_argument(
        "--initial-alphas",
        type=float,
        nargs="+",
        default=(0.1, 0.25),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_root = args.baseline_root.resolve()
    convex_root = args.convex_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    source_pretrain = baseline_root / "sin_K5_C5" / "thp" / "no_wishart"
    residual = _read(
        baseline_root / "sin_K5_C5" / "thp" / "wishart" / "result.json"
    )
    convex_alpha_half = _read(
        convex_root
        / "sin_K5_C5"
        / "thp"
        / "wishart_convex"
        / "result.json"
    )
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)
    for index, initial_alpha in enumerate(args.initial_alphas, start=1):
        run_dir = outdir / _tag(initial_alpha)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = _read(result_path)
            if result.get("status") == "pass":
                rows.append(
                    _row(result, initial_alpha, convex_alpha_half, residual)
                )
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
            "1.0",
            "--initial-alpha",
            str(initial_alpha),
        ))
        started = time.time()
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(args.initial_alphas),
            "queue_index": index,
            "current_initial_alpha": initial_alpha,
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
                "initial_alpha": initial_alpha,
                "returncode": process.returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = _read(result_path)
        rows.append(_row(result, initial_alpha, convex_alpha_half, residual))
        _write_summary(outdir / "summary.partial.csv", rows)
    _write_summary(outdir / "summary.csv", rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(args.initial_alphas),
        "current_initial_alpha": None,
        "last_completed": rows[-1] if rows else None,
    })


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    result: dict,
    initial_alpha: float,
    convex_alpha_half: dict,
    residual: dict,
) -> dict:
    purity = float(result["test_purity"])
    return {
        "initial_alpha": initial_alpha,
        "best_validation_purity": result["best_validation_purity"],
        "test_purity": purity,
        "test_ari": result["test_ari"],
        "test_nmi": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "purity_minus_convex_alpha0p5": (
            purity - float(convex_alpha_half["test_purity"])
        ),
        "purity_minus_residual": purity - float(residual["test_purity"]),
        "runtime_seconds": result["runtime_seconds"],
    }


if __name__ == "__main__":
    main()
