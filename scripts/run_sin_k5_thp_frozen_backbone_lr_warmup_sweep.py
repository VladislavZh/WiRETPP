#!/usr/bin/env python3
"""Resumable frozen-encoder backbone-LR/warmup sweep on sin_K5_C5.

The THP encoder is frozen throughout backward elimination and is unfrozen only
after the target component count is reached.  CUDA is restricted to GPU 0 and
the runner stops the suite if sampled dedicated VRAM usage crosses the
configured safety limit, avoiding WDDM paging into shared system memory.
"""

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


DATASET = "sin_K5_C5"
ARCHITECTURE = "thp"
SEED = 0


def _lr_tag(value: float) -> str:
    return f"lr_{value:.0e}".replace("e-", "em").replace("e+", "ep")


def _run_tag(learning_rate: float, warmup_steps: int) -> str:
    return f"warmup_{warmup_steps}/{_lr_tag(learning_rate)}"


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


def _prepare_pretrain(root: Path, run_dir: Path) -> None:
    source = (
        root
        / "artifacts"
        / "dan_convex_projected_alpha0p1_sin_k5_thp_seed0_20260803"
    )
    mappings = (
        (source / "checkpoint_cotic_k1.pt", run_dir / "checkpoint_cotic_k1.pt"),
        (source / "pretrain_history.csv", run_dir / "pretrain_history.csv"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for source_path, target_path in mappings:
        if not source_path.is_file():
            raise FileNotFoundError(f"missing frozen-protocol pretrain: {source_path}")
        if not target_path.is_file():
            shutil.copy2(source_path, target_path)


def _gpu_sample() -> dict | None:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one NVIDIA GPU, found {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 7:
        return None
    return {
        "timestamp": fields[0],
        "gpu_index": int(fields[1]),
        "gpu_name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "memory_free_mib": int(fields[5]),
        "utilization_gpu_percent": int(fields[6]),
    }


def _run_with_vram_guard(
    command: list[str],
    *,
    root: Path,
    run_dir: Path,
    environment: dict[str, str],
    vram_limit_mib: int,
    poll_seconds: float,
) -> tuple[int, list[dict], bool]:
    samples: list[dict] = []
    exceeded = False
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            sample = _gpu_sample()
            if sample is not None:
                samples.append(sample)
                _write_csv(run_dir / "vram_monitor.csv", samples)
                if sample["memory_used_mib"] >= vram_limit_mib:
                    exceeded = True
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            time.sleep(poll_seconds)
        returncode = process.wait()
    final_sample = _gpu_sample()
    if final_sample is not None:
        samples.append(final_sample)
        _write_csv(run_dir / "vram_monitor.csv", samples)
    return returncode, samples, exceeded


def _row(result: dict, *, learning_rate: float, warmup_steps: int,
         vram_samples: list[dict]) -> dict:
    return {
        "seed": result["seed"],
        "split_seed": result["split_seed"],
        "backbone_learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "best_validation_purity": result["best_validation_purity"],
        "best_validation_nll_per_exposure": result[
            "best_validation_nll_per_exposure"
        ],
        "test_purity_descriptive_only": result["test_purity"],
        "test_ari_descriptive_only": result["test_ari"],
        "test_nmi_descriptive_only": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "runtime_seconds": result["runtime_seconds"],
        "peak_sampled_dedicated_vram_mib": max(
            (sample["memory_used_mib"] for sample in vram_samples),
            default=None,
        ),
        "batch_size": result["batch_size"],
        "gradient_accumulation_steps": result[
            "gradient_accumulation_steps"
        ],
        "effective_batch_size": result["effective_batch_size"],
        "encoder_frozen_until_target": result[
            "encoder_frozen_until_target"
        ],
    }


def _write_selection(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["best_validation_purity"]),
            float(row["best_validation_nll_per_exposure"]),
            int(row["warmup_steps"]),
            float(row["backbone_learning_rate"]),
        ),
    )
    ranking = []
    for rank, row in enumerate(ranked, start=1):
        ranked_row = dict(row)
        ranked_row["validation_selection_rank"] = rank
        ranked_row["selected_by_validation"] = rank == 1
        ranking.append(ranked_row)
    _write_csv(outdir / "sweep_summary.csv", ranking)
    _write_json(outdir / "selection.json", {
        "selection_policy": (
            "maximize best_validation_purity; tie-break by lower validation "
            "NLL, fewer warmup steps, then lower backbone learning rate"
        ),
        "test_used_for_selection": False,
        "selected_backbone_learning_rate": ranking[0][
            "backbone_learning_rate"
        ],
        "selected_warmup_steps": ranking[0]["warmup_steps"],
        "selected_best_validation_purity": ranking[0][
            "best_validation_purity"
        ],
        "selected_test_purity_descriptive_only": ranking[0][
            "test_purity_descriptive_only"
        ],
        "ranking": ranking,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_sin_k5_thp_frozen_backbone_lr_warmup_seed0_20260806"
        ),
    )
    parser.add_argument(
        "--backbone-learning-rates",
        type=float,
        nargs="+",
        default=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    )
    parser.add_argument(
        "--warmup-steps-grid",
        type=int,
        nargs="+",
        default=(300, 100, 50, 25, 10),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--pruning-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--vram-limit-mib", type=int, default=7000)
    parser.add_argument("--vram-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.backbone_learning_rates):
        parser.error("all backbone learning rates must be positive")
    if any(value <= 0 for value in args.warmup_steps_grid):
        parser.error("all warmup values must be positive")
    if any(value <= 0 for value in (
        args.batch_size,
        args.evaluation_batch_size,
        args.pruning_batch_size,
        args.gradient_accumulation_steps,
        args.vram_limit_mib,
    )):
        parser.error("batch, accumulation, and VRAM values must be positive")
    if args.vram_poll_seconds <= 0.0:
        parser.error("VRAM polling interval must be positive")

    root = Path(__file__).resolve().parents[1]
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    initial_gpu = _gpu_sample()
    if initial_gpu is None:
        raise RuntimeError("could not inspect NVIDIA GPU")
    if initial_gpu["gpu_index"] != 0 or "RTX 4060" not in initial_gpu["gpu_name"]:
        raise RuntimeError(f"unexpected CUDA device: {initial_gpu}")
    if initial_gpu["memory_total_mib"] < args.vram_limit_mib:
        raise RuntimeError("VRAM safety limit exceeds physical dedicated VRAM")

    _write_json(outdir / "gpu_policy.json", {
        "cuda_visible_devices": "0",
        "physical_gpu": initial_gpu,
        "vram_limit_mib": args.vram_limit_mib,
        "stop_before_wddm_shared_memory_paging": True,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "pruning_batch_size": args.pruning_batch_size,
        "pruning_trace_padded_event_budget": 512,
        "pruning_rng_group_preserved": True,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
    })

    specifications = [
        (learning_rate, warmup_steps)
        for warmup_steps in args.warmup_steps_grid
        for learning_rate in args.backbone_learning_rates
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for index, (learning_rate, warmup_steps) in enumerate(
        specifications, start=1
    ):
        run_dir = outdir / _run_tag(learning_rate, warmup_steps)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                vram_path = run_dir / "vram_monitor.csv"
                vram_samples = (
                    list(csv.DictReader(vram_path.open(encoding="utf-8")))
                    if vram_path.is_file()
                    else []
                )
                for sample in vram_samples:
                    sample["memory_used_mib"] = int(sample["memory_used_mib"])
                rows.append(_row(
                    result,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    vram_samples=vram_samples,
                ))
                _write_csv(outdir / "summary.partial.csv", rows)
                _write_selection(outdir, rows)
                continue

        _prepare_pretrain(root, run_dir)
        command = _command(
            root,
            seed=SEED,
            dataset=DATASET,
            architecture=ARCHITECTURE,
            variant="wishart",
            outdir=run_dir,
            wishart_interaction_mode="convex",
            wishart_initial_alpha=0.1,
            wishart_alpha_parameterization="projected",
            wishart_alpha_temperature=1.0,
        )
        replacements = {
            "--warmup-steps": str(warmup_steps),
            "--batch-size": str(args.batch_size),
            "--evaluation-batch-size": str(args.evaluation_batch_size),
            "--gradient-accumulation-steps": str(
                args.gradient_accumulation_steps
            ),
            "--dual-learning-rate": "0.05",
            "--backbone-learning-rate": str(learning_rate),
            "--omega-learning-rate": "0.010096",
            "--alpha-learning-rate": "0.00059",
            "--pretrain-steps": "300",
            "--final-rho": "50",
            "--dpp-strength": "0.1",
            "--initial-beta": "0.9",
        }
        for flag, value in replacements.items():
            _replace_argument(command, flag, value)
        command.extend(("--pruning-batch-size", str(args.pruning_batch_size)))
        command.append("--freeze-encoder-until-target")

        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": index,
            "current": {
                "seed": SEED,
                "dataset": DATASET,
                "architecture": ARCHITECTURE,
                "variant": "wishart_frozen_encoder_until_target",
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "batch_size": args.batch_size,
                "evaluation_batch_size": args.evaluation_batch_size,
                "pruning_batch_size": args.pruning_batch_size,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
            "vram_limit_mib": args.vram_limit_mib,
        })
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        started = time.time()
        returncode, vram_samples, exceeded = _run_with_vram_guard(
            command,
            root=root,
            run_dir=run_dir,
            environment=environment,
            vram_limit_mib=args.vram_limit_mib,
            poll_seconds=args.vram_poll_seconds,
        )
        if exceeded:
            failure = {
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "reason": "dedicated_vram_safety_limit_exceeded",
                "vram_limit_mib": args.vram_limit_mib,
                "peak_sampled_dedicated_vram_mib": max(
                    sample["memory_used_mib"] for sample in vram_samples
                ),
                "run_dir": str(run_dir),
            }
            failures.append(failure)
            _write_json(outdir / "failures.json", failures)
            _write_json(outdir / "suite_progress.json", {
                "status": "stopped_vram_limit",
                "completed": len(rows),
                "failed": len(failures),
                "total": len(specifications),
                "current": failure,
                "last_completed": rows[-1] if rows else None,
                "test_used_for_selection": False,
            })
            return
        if returncode != 0 or not result_path.is_file():
            failures.append({
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_row(
            result,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            vram_samples=vram_samples,
        ))
        _write_csv(outdir / "summary.partial.csv", rows)
        _write_selection(outdir, rows)

    _write_csv(outdir / "summary.csv", rows)
    _write_selection(outdir, rows)
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
