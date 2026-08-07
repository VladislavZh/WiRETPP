#!/usr/bin/env python3
"""Audit COTIC parameter-group gradients at a saved overcomplete checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lal_wishart.metrics import cluster_purity
from lal_wishart.models.reference_output_mixtures import (
    ReferenceCOTICOutputMixture,
)
from lal_wishart.reproduction.dan_synthetic import load_dataset, shuffled_split
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    assignment_probabilities_from_dual,
    unbalanced_ot_dual_free_energy,
)


def _group(name: str) -> str:
    if name == "mixture_logits":
        return "mixture_logits"
    if name.startswith("encoder.") and "kernel_network" in name:
        return "encoder_continuous_affine_kernels"
    if name.startswith("encoder."):
        return "encoder_classical_parameters"
    if name.startswith("intensity_head.convolution.kernel_network"):
        return "intensity_continuous_affine_kernel"
    if name.startswith("intensity_head.layer"):
        return "intensity_output_projection"
    if name == "intensity_head.softplus_params":
        return "intensity_softplus_scale"
    raise ValueError(f"unclassified trainable COTIC parameter: {name}")


def _l2(values) -> float:
    tensors = [value.detach().square().sum() for value in values]
    return float(torch.stack(tensors).sum().sqrt().cpu()) if tensors else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="K5_C5")
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument("--dual-refit-steps", type=int, default=1000)
    parser.add_argument(
        "--dual-refit-learning-rates",
        nargs="+",
        type=float,
        default=(0.05, 0.2),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = json.loads((args.artifact / "result.json").read_text())
    dataset = load_dataset(args.data_root, args.dataset)
    split = shuffled_split(dataset, seed=int(result["split_seed"]))
    device = torch.device(args.device)
    model = ReferenceCOTICOutputMixture(
        int(result["target_components"]),
        dataset.n_marks,
        horizon=max(sequence.horizon for sequence in dataset.sequences),
        input_channels=32,
        hidden_size=64,
        num_layers=7,
        kernel_size=3,
        dropout=0.1,
        dilation_factor=1.29,
        quadrature_order=4,
        initialization_seed=int(result["initialization_seed"]),
    ).to(device)
    model.load_state_dict(
        torch.load(
            args.artifact / "checkpoint_best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    dual = torch.as_tensor(
        np.load(args.artifact / "assignment_dual.npy"),
        device=device,
        dtype=model.dtype,
    )
    rho = float(result["best_rho"])
    grouped = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped.setdefault(_group(name), []).append((name, parameter))
    parameter_rows = []
    for group_name, named in grouped.items():
        parameters = [parameter for _, parameter in named]
        parameter_rows.append({
            "group": group_name,
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(p.numel() for p in parameters),
            "parameter_l2": _l2(parameters),
            "parameter_names": ";".join(name for name, _ in named),
        })
    parameter_frame = pd.DataFrame(parameter_rows).set_index("group")

    rng = np.random.default_rng(int(result["initialization_seed"]) + 9901)
    rows = []
    score_rows = []
    torch.manual_seed(int(result["initialization_seed"]) + 9902)
    model.train()
    for batch_index in range(args.batches):
        indices = rng.choice(
            len(split.train), size=args.batch_size, replace=False
        )
        batch = tuple(split.train[int(index)] for index in indices)
        model.zero_grad(set_to_none=True)
        component = model.component_scores(batch)
        scores = component + model.mixture_log_weights()[None, :]
        objective, assignments, reference = unbalanced_ot_dual_free_energy(
            scores,
            dual,
            temperature=1.0,
            marginal_penalty=rho,
        )
        objective.backward()
        all_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        global_norm = _l2(parameter.grad for parameter in all_parameters)
        clip_coefficient = min(
            1.0, args.gradient_clip / max(global_norm, 1e-30)
        )
        for group_name, named in grouped.items():
            gradients = [
                parameter.grad
                for _, parameter in named
                if parameter.grad is not None
            ]
            gradient_l2 = _l2(gradients)
            elements = int(parameter_frame.loc[group_name, "parameter_elements"])
            parameter_l2 = float(
                parameter_frame.loc[group_name, "parameter_l2"]
            )
            rows.append({
                "batch": batch_index,
                "group": group_name,
                "global_gradient_l2": global_norm,
                "clip_coefficient": clip_coefficient,
                "gradient_l2": gradient_l2,
                "gradient_rms": gradient_l2 / math.sqrt(elements),
                "unclipped_relative_step": (
                    args.learning_rate * gradient_l2
                    / max(parameter_l2, 1e-30)
                ),
                "clipped_relative_step": (
                    args.learning_rate * clip_coefficient * gradient_l2
                    / max(parameter_l2, 1e-30)
                ),
            })
        score_rows.append({
            "batch": batch_index,
            "objective": float(objective.detach().cpu()),
            "component_score_mean": float(component.mean().detach().cpu()),
            "component_score_std": float(component.std().detach().cpu()),
            "component_score_min": float(component.min().detach().cpu()),
            "component_score_max": float(component.max().detach().cpu()),
            "assignment_stationarity_max_abs": float(
                (assignments.mean(0) - reference).abs().max().detach().cpu()
            ),
        })

    frame = pd.DataFrame(rows)
    summary = frame.groupby("group", as_index=False).agg(
        gradient_l2_median=("gradient_l2", "median"),
        gradient_l2_p90=("gradient_l2", lambda x: x.quantile(0.9)),
        gradient_rms_median=("gradient_rms", "median"),
        unclipped_relative_step_median=("unclipped_relative_step", "median"),
        clipped_relative_step_median=("clipped_relative_step", "median"),
    )
    summary = summary.merge(
        parameter_frame.reset_index(), on="group", how="left"
    )
    score_frame = pd.DataFrame(score_rows)

    @torch.no_grad()
    def full_scores(sequences) -> torch.Tensor:
        model.eval()
        values = []
        for start in range(0, len(sequences), args.batch_size):
            component = model.component_scores(
                sequences[start : start + args.batch_size]
            )
            values.append(
                component + model.mixture_log_weights()[None, :]
            )
        return torch.cat(values).detach()

    train_scores = full_scores(split.train)
    validation_scores = full_scores(split.validation)
    dual_rows = []
    checkpoints = {
        0,
        1,
        5,
        10,
        25,
        50,
        100,
        200,
        500,
        args.dual_refit_steps,
    }
    for dual_lr in args.dual_refit_learning_rates:
        fitted_dual = torch.nn.Parameter(dual.detach().clone())
        optimizer = torch.optim.Adam(
            (fitted_dual,), lr=dual_lr, maximize=True
        )
        for dual_step in range(args.dual_refit_steps + 1):
            objective, assignments, reference = (
                unbalanced_ot_dual_free_energy(
                    train_scores,
                    fitted_dual,
                    temperature=1.0,
                    marginal_penalty=rho,
                )
            )
            if dual_step in checkpoints:
                validation_probabilities = (
                    assignment_probabilities_from_dual(
                        validation_scores,
                        assignment_dual=fitted_dual.detach(),
                        temperature=1.0,
                    )
                )
                dual_rows.append({
                    "dual_learning_rate": dual_lr,
                    "dual_step": dual_step,
                    "dual_l2": float(
                        torch.linalg.vector_norm(fitted_dual).detach().cpu()
                    ),
                    "stationarity_max_abs": float(
                        (
                            assignments.mean(0) - reference
                        ).abs().max().detach().cpu()
                    ),
                    "validation_purity": cluster_purity(
                        split.validation_labels,
                        validation_probabilities.argmax(1).cpu().numpy(),
                    ),
                })
            if dual_step == args.dual_refit_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                fitted_dual.sub_(fitted_dual.mean())
    dual_frame = pd.DataFrame(dual_rows)
    metadata = {
        "artifact": str(args.artifact),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "rho": rho,
        "dual_l2": float(torch.linalg.vector_norm(dual).cpu()),
        "global_gradient_l2_median": float(
            frame.groupby("batch")["global_gradient_l2"].first().median()
        ),
        "clipping_fraction": float(
            (
                frame.groupby("batch")["clip_coefficient"].first() < 1.0
            ).mean()
        ),
        "component_score_std_median": float(
            score_frame["component_score_std"].median()
        ),
        "assignment_stationarity_max_abs_median": float(
            score_frame["assignment_stationarity_max_abs"].median()
        ),
    }
    frame.to_csv(args.artifact / "gradient_audit_batches.csv", index=False)
    summary.to_csv(args.artifact / "gradient_audit_summary.csv", index=False)
    score_frame.to_csv(args.artifact / "gradient_audit_scores.csv", index=False)
    dual_frame.to_csv(args.artifact / "dual_refit_audit.csv", index=False)
    (args.artifact / "gradient_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(dual_frame.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
