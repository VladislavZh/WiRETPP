"""Training and evaluation for distributional latent-Wishart NHP attention."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    cluster_probabilities_from_sampled_posterior,
    conditional_suffix_scores_from_samples,
    monte_carlo_marginal_scores,
    sampled_posterior_log_weights,
    squared_correlation_attention,
)
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
    build_nhp_mixture_batch_trace,
    LatentWishartAttentionNHP,
    sampled_nhp_component_scores,
)
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
)


@dataclass(frozen=True)
class DirectNHPMixtureFitResult:
    model: ReferenceNeuralHawkesMixture
    history: tuple[dict[str, float | int | bool], ...]
    best_epoch: int
    best_validation_suffix_nll_per_exposure: float
    final_gradient_norm: float


@dataclass(frozen=True)
class DirectNHPMixtureEvaluation:
    conditional_suffix_scores: Tensor
    full_marginal_scores: Tensor
    prefix_cluster_probabilities: Tensor
    full_cluster_probabilities: Tensor


@dataclass(frozen=True)
class LatentWishartNHPFitResult:
    model: LatentWishartAttentionNHP
    history: tuple[dict[str, float | int | bool], ...]
    best_epoch: int
    best_validation_suffix_nll_per_exposure: float
    final_gradient_norm: float


@dataclass(frozen=True)
class LatentWishartNHPEvaluation:
    conditional_suffix_scores: Tensor
    full_marginal_scores: Tensor
    prefix_cluster_probabilities: Tensor
    full_cluster_probabilities: Tensor
    prior_cluster_probabilities: Tensor
    prefix_posterior_attention_diagonal: Tensor
    full_posterior_attention_diagonal: Tensor
    prior_attention_diagonal: Tensor
    prefix_joint_effective_sample_size: Tensor
    full_joint_effective_sample_size: Tensor


def _build_mixture_trace(
    backbone,
    sequences: Iterable[MarkedSequence],
    *,
    boundary: float | None = None,
):
    """Use a backbone-native trace builder when one is available."""

    if hasattr(backbone, "build_trace"):
        return backbone.build_trace(sequences, boundary=boundary)
    return build_nhp_mixture_batch_trace(
        backbone,
        sequences,
        boundary=boundary,
    )


def evaluate_direct_nhp_mixture(
    model: ReferenceNeuralHawkesMixture,
    sequences: Iterable[MarkedSequence],
    *,
    cutoff: float,
    batch_size: int = 64,
) -> DirectNHPMixtureEvaluation:
    """Evaluate predictive and full likelihoods for a conventional mixture."""

    sequence_list = tuple(sequences)
    if not sequence_list:
        raise ValueError("evaluation sequences must be non-empty")
    if not 0.0 < cutoff < model.horizon or batch_size <= 0:
        raise ValueError("invalid cutoff or batch size")
    was_training = model.training
    model.eval()
    suffix_rows: list[Tensor] = []
    full_rows: list[Tensor] = []
    prefix_probability_rows: list[Tensor] = []
    full_probability_rows: list[Tensor] = []
    with torch.no_grad():
        for first in range(0, len(sequence_list), batch_size):
            batch = sequence_list[first : first + batch_size]
            trace = _build_mixture_trace(
                model,
                batch,
                boundary=cutoff,
            )
            prefix = base_nhp_component_scores_from_trace(
                trace,
                end_time=cutoff,
            )
            suffix = base_nhp_component_scores_from_trace(
                trace,
                start_time=cutoff,
            )
            log_weights = model.mixture_log_weights()[None].expand_as(prefix)
            prefix_posterior = torch.log_softmax(
                prefix + log_weights,
                dim=1,
            )
            full_component = prefix + suffix
            full_joint = full_component + log_weights
            suffix_rows.append(
                torch.logsumexp(prefix_posterior + suffix, dim=1).cpu()
            )
            full_rows.append(torch.logsumexp(full_joint, dim=1).cpu())
            prefix_probability_rows.append(prefix_posterior.exp().cpu())
            full_probability_rows.append(
                torch.softmax(full_joint, dim=1).cpu()
            )
    model.train(was_training)
    return DirectNHPMixtureEvaluation(
        conditional_suffix_scores=torch.cat(suffix_rows),
        full_marginal_scores=torch.cat(full_rows),
        prefix_cluster_probabilities=torch.cat(prefix_probability_rows),
        full_cluster_probabilities=torch.cat(full_probability_rows),
    )


def evaluate_latent_wishart_nhp(
    model: LatentWishartAttentionNHP,
    sequences: Iterable[MarkedSequence],
    *,
    cutoff: float,
    n_samples: int,
    sample_seed: int,
    batch_size: int = 16,
) -> LatentWishartNHPEvaluation:
    """Evaluate fresh-W posterior prediction without test-time optimization."""

    sequence_list = tuple(sequences)
    if not sequence_list:
        raise ValueError("evaluation sequences must be non-empty")
    if (
        not 0.0 < cutoff < model.backbone.horizon
        or n_samples <= 0
        or batch_size <= 0
    ):
        raise ValueError("invalid evaluation settings")
    was_training = model.training
    model.eval()
    suffix_rows: list[Tensor] = []
    full_rows: list[Tensor] = []
    prefix_cluster_rows: list[Tensor] = []
    full_cluster_rows: list[Tensor] = []
    prior_cluster_rows: list[Tensor] = []
    prefix_attention_rows: list[Tensor] = []
    full_attention_rows: list[Tensor] = []
    prior_attention_rows: list[Tensor] = []
    prefix_ess_rows: list[Tensor] = []
    full_ess_rows: list[Tensor] = []
    with torch.no_grad():
        for first in range(0, len(sequence_list), batch_size):
            batch = sequence_list[first : first + batch_size]
            trace = _build_mixture_trace(
                model.backbone,
                batch,
                boundary=cutoff,
            )
            matrices = model.sample_matrices(
                len(batch),
                n_samples,
                sample_seed=sample_seed + first,
            )
            gates = cluster_log_weights_from_matrices(
                matrices,
                n_components=model.n_components,
                n_marks=model.n_marks,
            )
            prefix = sampled_nhp_component_scores(
                trace,
                matrices,
                end_time=cutoff,
            )
            suffix = sampled_nhp_component_scores(
                trace,
                matrices,
                start_time=cutoff,
            )
            conditional, prefix_posterior = (
                conditional_suffix_scores_from_samples(
                    prefix,
                    suffix,
                    gates,
                )
            )
            full_component = prefix + suffix
            full_posterior = sampled_posterior_log_weights(
                full_component,
                gates,
            )
            attention_diagonal = squared_correlation_attention(
                matrices
            ).diagonal(dim1=-2, dim2=-1).mean(dim=2)
            prefix_sample_probability = prefix_posterior.exp().sum(dim=2)
            full_sample_probability = full_posterior.exp().sum(dim=2)
            prefix_joint_probability = prefix_posterior.exp()
            full_joint_probability = full_posterior.exp()
            suffix_rows.append(conditional.cpu())
            full_rows.append(
                monte_carlo_marginal_scores(
                    full_component,
                    gates,
                ).cpu()
            )
            prefix_cluster_rows.append(
                cluster_probabilities_from_sampled_posterior(
                    prefix_posterior
                ).cpu()
            )
            full_cluster_rows.append(
                cluster_probabilities_from_sampled_posterior(
                    full_posterior
                ).cpu()
            )
            prior_cluster_rows.append(gates.exp().mean(dim=1).cpu())
            prefix_attention_rows.append(
                (
                    prefix_sample_probability * attention_diagonal
                ).sum(dim=1).cpu()
            )
            full_attention_rows.append(
                (
                    full_sample_probability * attention_diagonal
                ).sum(dim=1).cpu()
            )
            prior_attention_rows.append(attention_diagonal.mean(dim=1).cpu())
            prefix_ess_rows.append(
                (
                    1.0
                    / prefix_joint_probability.square().sum(dim=(1, 2))
                ).cpu()
            )
            full_ess_rows.append(
                (
                    1.0
                    / full_joint_probability.square().sum(dim=(1, 2))
                ).cpu()
            )
    model.train(was_training)
    return LatentWishartNHPEvaluation(
        conditional_suffix_scores=torch.cat(suffix_rows),
        full_marginal_scores=torch.cat(full_rows),
        prefix_cluster_probabilities=torch.cat(prefix_cluster_rows),
        full_cluster_probabilities=torch.cat(full_cluster_rows),
        prior_cluster_probabilities=torch.cat(prior_cluster_rows),
        prefix_posterior_attention_diagonal=torch.cat(
            prefix_attention_rows
        ),
        full_posterior_attention_diagonal=torch.cat(full_attention_rows),
        prior_attention_diagonal=torch.cat(prior_attention_rows),
        prefix_joint_effective_sample_size=torch.cat(prefix_ess_rows),
        full_joint_effective_sample_size=torch.cat(full_ess_rows),
    )


def _selected_sequences(
    sequences: tuple[MarkedSequence, ...],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[MarkedSequence, ...]:
    if batch_size >= len(sequences):
        return sequences
    indices = rng.choice(len(sequences), size=batch_size, replace=False)
    return tuple(sequences[int(index)] for index in indices)


def fit_direct_nhp_mixture(
    model: ReferenceNeuralHawkesMixture,
    train_sequences: Iterable[MarkedSequence],
    validation_sequences: Iterable[MarkedSequence],
    *,
    max_epochs: int = 50,
    batch_size: int = 90,
    learning_rate: float = 0.001,
    neural_weight_decay: float = 1e-5,
    evaluation_interval: int = 5,
    validation_cutoff: float,
    evaluation_batch_size: int = 32,
    gradient_clip: float = 20.0,
    batch_seed: int = 0,
) -> DirectNHPMixtureFitResult:
    """Fit a conventional differentiable finite NHP mixture."""

    train = tuple(train_sequences)
    validation = tuple(validation_sequences)
    if model.n_components <= 1:
        raise ValueError("direct mixture fit requires K > 1")
    if not train or not validation:
        raise ValueError("train and validation sequences must be non-empty")
    if (
        max_epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0.0
        or evaluation_interval <= 0
        or gradient_clip <= 0.0
    ):
        raise ValueError("optimization settings must be positive")
    neural_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "mixture_logits"
    ]
    optimizer = torch.optim.Adam(
        (
            {
                "params": neural_parameters,
                "lr": learning_rate,
                "weight_decay": neural_weight_decay,
            },
            {
                "params": (model.mixture_logits,),
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
        )
    )
    optimized = neural_parameters + [model.mixture_logits]
    rng = np.random.default_rng(batch_seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = -1
    final_gradient_norm = math.nan
    history: list[dict[str, float | int | bool]] = []
    indices = tuple(range(model.n_components))
    for epoch in range(1, max_epochs + 1):
        batch = _selected_sequences(train, rng, batch_size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        component = model.component_scores_for_indices(batch, indices)
        log_weights = model.mixture_log_weights()[None].expand_as(component)
        loss = -torch.logsumexp(component + log_weights, dim=1).mean()
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("non-finite direct NHP mixture loss")
        loss.backward()
        final_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(optimized, gradient_clip)
            .detach()
            .cpu()
        )
        optimizer.step()
        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        validation_evaluation = evaluate_direct_nhp_mixture(
            model,
            validation,
            cutoff=validation_cutoff,
            batch_size=evaluation_batch_size,
        )
        exposure = len(validation) * (model.horizon - validation_cutoff)
        validation_nll = float(
            -validation_evaluation.conditional_suffix_scores.sum()
            / exposure
        )
        probabilities = (
            validation_evaluation.prefix_cluster_probabilities
        )
        history.append({
            "epoch": epoch,
            "optimizer_steps": epoch,
            "train_negative_log_likelihood_per_path": float(
                loss.detach().cpu()
            ),
            "validation_suffix_nll_per_exposure": validation_nll,
            "validation_prefix_posterior_entropy": float(
                -(
                    probabilities * torch.log(probabilities + 1e-12)
                ).sum(dim=1).mean()
            ),
            "gradient_norm": final_gradient_norm,
            "uses_elbo": False,
            "uses_lal_logic": False,
        })
        if validation_nll < best_validation:
            best_validation = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return DirectNHPMixtureFitResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_suffix_nll_per_exposure=best_validation,
        final_gradient_norm=final_gradient_norm,
    )


def fit_latent_wishart_attention_nhp(
    model: LatentWishartAttentionNHP,
    train_sequences: Iterable[MarkedSequence],
    validation_sequences: Iterable[MarkedSequence],
    *,
    max_epochs: int = 50,
    batch_size: int = 90,
    train_samples: int = 4,
    validation_samples: int = 16,
    neural_learning_rate: float = 0.001,
    distribution_learning_rate: float = 0.01,
    neural_weight_decay: float = 1e-5,
    mean_hyperprior_strength: float = 1.0,
    evaluation_interval: int = 5,
    validation_cutoff: float,
    evaluation_batch_size: int = 16,
    gradient_clip: float = 20.0,
    batch_seed: int = 0,
    sample_seed: int = 0,
) -> LatentWishartNHPFitResult:
    """Optimize NHP and Wishart-law parameters by MC marginal likelihood."""

    train = tuple(train_sequences)
    validation = tuple(validation_sequences)
    if not train or not validation:
        raise ValueError("train and validation sequences must be non-empty")
    if (
        max_epochs <= 0
        or batch_size <= 0
        or train_samples <= 0
        or validation_samples <= 0
        or neural_learning_rate <= 0.0
        or distribution_learning_rate <= 0.0
        or evaluation_interval <= 0
        or gradient_clip <= 0.0
        or mean_hyperprior_strength < 0.0
    ):
        raise ValueError("optimization settings are invalid")
    neural_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
        and name != "backbone.mixture_logits"
    ]
    optimizer = torch.optim.Adam(
        (
            {
                "params": neural_parameters,
                "lr": neural_learning_rate,
                "weight_decay": neural_weight_decay,
            },
            {
                "params": (model.raw_mean_cholesky,),
                "lr": distribution_learning_rate,
                "weight_decay": 0.0,
            },
        )
    )
    optimized = neural_parameters + [model.raw_mean_cholesky]
    rng = np.random.default_rng(batch_seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = -1
    final_gradient_norm = math.nan
    history: list[dict[str, float | int | bool]] = []
    for epoch in range(1, max_epochs + 1):
        batch = _selected_sequences(train, rng, batch_size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        marginal = model.marginal_scores(
            batch,
            n_samples=train_samples,
            sample_seed=sample_seed + epoch,
        )
        hyperprior = model.mean_matrix_hyperprior_penalty(
            strength=mean_hyperprior_strength,
        )
        loss = -marginal.mean() + hyperprior / len(train)
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError("non-finite latent Wishart NHP loss")
        loss.backward()
        final_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(optimized, gradient_clip)
            .detach()
            .cpu()
        )
        optimizer.step()
        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        validation_evaluation = evaluate_latent_wishart_nhp(
            model,
            validation,
            cutoff=validation_cutoff,
            n_samples=validation_samples,
            sample_seed=sample_seed + 100_000,
            batch_size=evaluation_batch_size,
        )
        exposure = (
            len(validation)
            * (model.backbone.horizon - validation_cutoff)
        )
        validation_nll = float(
            -validation_evaluation.conditional_suffix_scores.sum()
            / exposure
        )
        with torch.no_grad():
            mean = model.mean_matrix()
            identity = torch.eye(
                model.dimension,
                dtype=mean.dtype,
                device=mean.device,
            )
            eigenvalues = torch.linalg.eigvalsh(mean)
            probabilities = (
                validation_evaluation.prefix_cluster_probabilities
            )
        history.append({
            "epoch": epoch,
            "optimizer_steps": epoch,
            "train_negative_mc_log_posterior_per_path": float(
                loss.detach().cpu()
            ),
            "validation_suffix_nll_per_exposure": validation_nll,
            "validation_prefix_cluster_entropy": float(
                -(
                    probabilities * torch.log(probabilities + 1e-12)
                ).sum(dim=1).mean()
            ),
            "validation_prior_attention_diagonal": float(
                validation_evaluation.prior_attention_diagonal.mean()
            ),
            "validation_prefix_posterior_attention_diagonal": float(
                validation_evaluation
                .prefix_posterior_attention_diagonal.mean()
            ),
            "mean_matrix_hyperprior_penalty": float(
                model.mean_matrix_hyperprior_penalty(
                    strength=mean_hyperprior_strength
                ).detach().cpu()
            ),
            "mean_abs_mean_matrix_minus_identity": float(
                (mean - identity).abs().mean().cpu()
            ),
            "minimum_mean_matrix_eigenvalue": float(
                eigenvalues.min().cpu()
            ),
            "maximum_mean_matrix_eigenvalue": float(
                eigenvalues.max().cpu()
            ),
            "gradient_norm": final_gradient_norm,
            "degrees_of_freedom": model.degrees_of_freedom,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "uses_local_w_parameters": False,
            "uses_elbo": False,
            "uses_lal_logic": False,
            "w_is_integrated_by_monte_carlo": True,
        })
        if validation_nll < best_validation:
            best_validation = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return LatentWishartNHPFitResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_suffix_nll_per_exposure=best_validation,
        final_gradient_norm=final_gradient_norm,
    )
