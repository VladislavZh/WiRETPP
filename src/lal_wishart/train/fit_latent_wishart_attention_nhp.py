"""Training and evaluation for distributional latent-Wishart NHP attention."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from lal_wishart.data.hawkes_branching import MarkedSequence
from lal_wishart.metrics import cluster_purity
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    cluster_probabilities_from_sampled_posterior,
    conditional_suffix_scores_from_samples,
    monte_carlo_marginal_scores,
    sampled_posterior_log_weights,
)
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
    build_nhp_mixture_batch_trace,
    LatentWishartAttentionNHP,
)
from lal_wishart.models.reference_neural_lal import (
    ReferenceNeuralHawkesMixture,
)


@dataclass(frozen=True)
class DirectNHPMixtureFitResult:
    model: ReferenceNeuralHawkesMixture
    history: tuple[dict[str, float | int | bool | str], ...]
    best_epoch: int
    best_validation_suffix_nll_per_exposure: float
    best_validation_purity: float
    selection_metric: str
    final_gradient_norm: float
    assignment_dual: Tensor | None
    assignment_temperature: float
    assignment_objective: str


@dataclass(frozen=True)
class DirectNHPMixtureEvaluation:
    conditional_suffix_scores: Tensor
    full_marginal_scores: Tensor
    prefix_cluster_probabilities: Tensor
    full_cluster_probabilities: Tensor


@dataclass(frozen=True)
class LatentWishartNHPFitResult:
    model: LatentWishartAttentionNHP
    history: tuple[dict[str, float | int | bool | str], ...]
    best_epoch: int
    best_validation_suffix_nll_per_exposure: float
    best_validation_purity: float
    selection_metric: str
    final_gradient_norm: float
    assignment_dual: Tensor | None
    assignment_temperature: float
    assignment_objective: str


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


def _gradient_l2_norm(parameters: Iterable[Tensor]) -> float:
    """Return the pre-clipping L2 norm for a parameter group."""

    squared_norms = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        return 0.0
    return float(torch.stack(squared_norms).sum().sqrt().cpu())


def component_removal_marginal_nll_deltas(
    component_scores: Tensor,
    mixture_logits: Tensor,
) -> tuple[Tensor, Tensor]:
    """Full-mixture NLL and increases after renormalized component removal."""

    if component_scores.ndim != 2:
        raise ValueError("component scores must have shape (paths, K)")
    if mixture_logits.shape != (component_scores.shape[1],):
        raise ValueError("mixture logits must have shape (K,)")
    if component_scores.shape[1] <= 1:
        raise ValueError("component removal requires K > 1")
    log_weights = torch.log_softmax(
        mixture_logits.to(
            device=component_scores.device,
            dtype=component_scores.dtype,
        ),
        dim=0,
    )
    full_nll = -torch.logsumexp(
        component_scores + log_weights[None, :],
        dim=1,
    ).mean()
    deltas = []
    for removed in range(component_scores.shape[1]):
        kept = [
            index
            for index in range(component_scores.shape[1])
            if index != removed
        ]
        kept_log_weights = log_weights[kept]
        kept_log_weights = (
            kept_log_weights - torch.logsumexp(kept_log_weights, dim=0)
        )
        removed_nll = -torch.logsumexp(
            component_scores[:, kept] + kept_log_weights[None, :],
            dim=1,
        ).mean()
        deltas.append(removed_nll - full_nll)
    return full_nll, torch.stack(deltas)


class _LaLAdjacentLossDecay:
    """Reproduce the learning-rate callback shipped with sequence_clusterers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        factor: float,
        tolerance: int,
        min_lr: float | None,
        updated_lr: float | None,
    ) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.tolerance = tolerance
        self.min_lr = min_lr
        self.updated_lr = updated_lr
        self.old_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        self.previous_loss = math.inf
        self.checker = 0

    def step(self, loss: float) -> bool:
        if loss >= self.previous_loss:
            self.checker += 1
        self.previous_loss = loss
        if self.checker < self.tolerance:
            return False
        self.checker = 0
        new_lrs: list[float] = []
        for old_lr, group in zip(
            self.old_lrs,
            self.optimizer.param_groups,
            strict=True,
        ):
            new_lr = old_lr * self.factor
            if self.min_lr is not None and new_lr < self.min_lr:
                if self.updated_lr is None:
                    raise RuntimeError(
                        "LaL LR decay requires updated_lr below min_lr"
                    )
                new_lr = self.updated_lr
            group["lr"] = new_lr
            new_lrs.append(new_lr)
        self.old_lrs = new_lrs
        return True


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


SelectionMetric = Literal["validation_nll", "validation_purity"]


def _selection_is_better(
    selection_metric: SelectionMetric,
    *,
    validation_nll: float,
    validation_purity: float,
    best_validation_nll: float,
    best_validation_purity: float,
) -> bool:
    """Compare checkpoints, breaking equal purity by lower validation NLL."""

    if selection_metric == "validation_nll":
        return validation_nll < best_validation_nll
    if selection_metric != "validation_purity":
        raise ValueError(f"unknown selection metric: {selection_metric}")
    if validation_purity > best_validation_purity + 1e-12:
        return True
    return (
        abs(validation_purity - best_validation_purity) <= 1e-12
        and validation_nll < best_validation_nll
    )


def _validated_selection_labels(
    validation_labels: np.ndarray | None,
    *,
    validation_size: int,
    selection_metric: SelectionMetric,
) -> np.ndarray | None:
    if validation_labels is None:
        if selection_metric == "validation_purity":
            raise ValueError(
                "validation labels are required for purity selection"
            )
        return None
    labels = np.asarray(validation_labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != validation_size:
        raise ValueError("validation labels and sequences must align")
    return labels


def integrated_wishart_component_scores(
    component_scores: Tensor,
    cluster_log_weights: Tensor,
) -> Tensor:
    """Monte-Carlo integrate ``W`` while retaining the cluster index.

    The result is the score matrix ``S[b, k]`` whose log-sum-exp over
    clusters is exactly the ordinary Monte-Carlo mixture likelihood.
    """

    if (
        component_scores.ndim != 3
        or component_scores.shape != cluster_log_weights.shape
    ):
        raise ValueError("scores and gates must have shape (batch, S, K)")
    return (
        torch.logsumexp(component_scores + cluster_log_weights, dim=1)
        - math.log(component_scores.shape[1])
    )


def unbalanced_ot_dual_free_energy(
    component_scores: Tensor,
    assignment_dual: Tensor,
    *,
    temperature: float = 1.0,
    marginal_penalty: float,
    target_marginal: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the stochastic dual of global unbalanced entropic OT.

    The model minimizes the returned free energy while ``assignment_dual``
    maximizes it.  Thus random batches estimate one global train marginal;
    no individual batch is constrained to contain fixed cluster counts.
    """

    if component_scores.ndim != 2:
        raise ValueError("component scores must have shape (batch, K)")
    if assignment_dual.shape != (component_scores.shape[1],):
        raise ValueError("assignment dual must have shape (K,)")
    if temperature <= 0.0 or marginal_penalty <= 0.0:
        raise ValueError("OT temperature and marginal penalty must be positive")
    if target_marginal is None:
        target = component_scores.new_full(
            (component_scores.shape[1],),
            1.0 / component_scores.shape[1],
        )
    else:
        target = target_marginal.to(
            device=component_scores.device,
            dtype=component_scores.dtype,
        )
        if target.shape != (component_scores.shape[1],):
            raise ValueError("target marginal must have shape (K,)")
        if bool(torch.any(target <= 0.0)):
            raise ValueError("target marginal must be strictly positive")
        target = target / target.sum()
    dual = assignment_dual.to(
        device=component_scores.device,
        dtype=component_scores.dtype,
    )
    adjusted = (component_scores - dual[None, :]) / temperature
    assignments = torch.softmax(adjusted, dim=1)
    dual_reference_marginal = torch.softmax(
        torch.log(target) + dual / marginal_penalty,
        dim=0,
    )
    free_energy = (
        -temperature * torch.logsumexp(adjusted, dim=1).mean()
        - marginal_penalty
        * torch.logsumexp(
            torch.log(target) + dual / marginal_penalty,
            dim=0,
        )
    )
    return free_energy, assignments, dual_reference_marginal


def assignment_probabilities_from_dual(
    log_probabilities: Tensor,
    *,
    assignment_dual: Tensor | None,
    temperature: float,
) -> Tensor:
    """Apply a train-fitted global OT dual to posterior log-probabilities."""

    if temperature <= 0.0:
        raise ValueError("assignment temperature must be positive")
    if assignment_dual is None:
        return torch.softmax(log_probabilities, dim=1)
    if assignment_dual.shape != (log_probabilities.shape[1],):
        raise ValueError("assignment dual must have shape (K,)")
    dual = assignment_dual.to(
        device=log_probabilities.device,
        dtype=log_probabilities.dtype,
    )
    return torch.softmax(
        (log_probabilities - dual[None, :]) / temperature,
        dim=1,
    )


def evaluate_direct_nhp_mixture(
    model: ReferenceNeuralHawkesMixture,
    sequences: Iterable[MarkedSequence],
    *,
    cutoff: float | None,
    batch_size: int = 64,
    assignment_dual: Tensor | None = None,
    assignment_temperature: float = 1.0,
) -> DirectNHPMixtureEvaluation:
    """Evaluate predictive and full likelihoods for a conventional mixture."""

    sequence_list = tuple(sequences)
    if not sequence_list:
        raise ValueError("evaluation sequences must be non-empty")
    if (
        batch_size <= 0
        or assignment_temperature <= 0.0
        or (
            cutoff is not None
            and not 0.0 < cutoff < max(
                sequence.horizon for sequence in sequence_list
            )
        )
    ):
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
            if cutoff is None:
                full_component = base_nhp_component_scores_from_trace(trace)
                suffix = full_component
                prefix = torch.zeros_like(full_component)
            else:
                prefix = base_nhp_component_scores_from_trace(
                    trace,
                    end_time=cutoff,
                )
                suffix = base_nhp_component_scores_from_trace(
                    trace,
                    start_time=cutoff,
                )
                full_component = prefix + suffix
            log_weights = model.mixture_log_weights()[None].expand_as(prefix)
            prefix_posterior = torch.log_softmax(
                prefix + log_weights,
                dim=1,
            )
            full_joint = full_component + log_weights
            suffix_rows.append(
                torch.logsumexp(prefix_posterior + suffix, dim=1).cpu()
            )
            full_rows.append(torch.logsumexp(full_joint, dim=1).cpu())
            prefix_probability_rows.append(
                assignment_probabilities_from_dual(
                    prefix_posterior,
                    assignment_dual=assignment_dual,
                    temperature=assignment_temperature,
                ).cpu()
            )
            full_probability_rows.append(
                assignment_probabilities_from_dual(
                    torch.log_softmax(full_joint, dim=1),
                    assignment_dual=assignment_dual,
                    temperature=assignment_temperature,
                ).cpu()
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
    cutoff: float | None,
    n_samples: int,
    sample_seed: int,
    batch_size: int = 16,
    assignment_dual: Tensor | None = None,
    assignment_temperature: float = 1.0,
) -> LatentWishartNHPEvaluation:
    """Evaluate fresh-W posterior prediction without test-time optimization."""

    sequence_list = tuple(sequences)
    if not sequence_list:
        raise ValueError("evaluation sequences must be non-empty")
    if (
        (
            cutoff is not None
            and not 0.0 < cutoff < max(
                sequence.horizon for sequence in sequence_list
            )
        )
        or n_samples <= 0
        or batch_size <= 0
        or assignment_temperature <= 0.0
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
            if cutoff is None:
                full_component = model.component_scores_from_trace(
                    trace,
                    matrices,
                )
                suffix = full_component
                prefix = torch.zeros_like(full_component)
            else:
                prefix = model.component_scores_from_trace(
                    trace,
                    matrices,
                    end_time=cutoff,
                )
                suffix = model.component_scores_from_trace(
                    trace,
                    matrices,
                    start_time=cutoff,
                )
                full_component = prefix + suffix
            conditional, prefix_posterior = (
                conditional_suffix_scores_from_samples(
                    prefix,
                    suffix,
                    gates,
                )
            )
            full_posterior = sampled_posterior_log_weights(
                full_component,
                gates,
            )
            attention_diagonal = model.transformation_diagnostic(
                matrices
            ).mean(dim=2)
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
                assignment_probabilities_from_dual(
                    torch.log(
                        cluster_probabilities_from_sampled_posterior(
                            prefix_posterior
                        ).clamp_min(1e-12)
                    ),
                    assignment_dual=assignment_dual,
                    temperature=assignment_temperature,
                ).cpu()
            )
            full_cluster_rows.append(
                assignment_probabilities_from_dual(
                    torch.log(
                        cluster_probabilities_from_sampled_posterior(
                            full_posterior
                        ).clamp_min(1e-12)
                    ),
                    assignment_dual=assignment_dual,
                    temperature=assignment_temperature,
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


def balanced_sinkhorn_assignments(
    log_probabilities: Tensor,
    *,
    temperature: float = 0.1,
    iterations: int = 200,
) -> Tensor:
    """Project assignments onto uniform cluster marginals with entropic OT.

    The returned rows sum to one and the columns sum to ``batch_size / K``.
    The transport target is intentionally detached: it is a balanced
    pseudo-label, not a second differentiable route through the model.
    """

    if log_probabilities.ndim != 2:
        raise ValueError("assignment logits must have shape (batch, K)")
    batch_size, n_components = log_probabilities.shape
    if batch_size <= 0 or n_components <= 1:
        raise ValueError("balanced OT requires a non-empty K > 1 batch")
    if temperature <= 0.0 or iterations <= 0:
        raise ValueError("Sinkhorn temperature and iterations must be positive")
    with torch.no_grad():
        log_kernel = log_probabilities.detach() / temperature
        log_row_mass = -math.log(batch_size)
        log_column_mass = -math.log(n_components)
        log_u = torch.zeros(
            batch_size,
            dtype=log_kernel.dtype,
            device=log_kernel.device,
        )
        log_v = torch.zeros(
            n_components,
            dtype=log_kernel.dtype,
            device=log_kernel.device,
        )
        for _ in range(iterations):
            log_u = log_row_mass - torch.logsumexp(
                log_kernel + log_v[None, :], dim=1
            )
            log_v = log_column_mass - torch.logsumexp(
                log_kernel + log_u[:, None], dim=0
            )
        log_plan = log_kernel + log_u[:, None] + log_v[None, :]
        assignments = log_plan.exp() * batch_size
        assignments = assignments / assignments.sum(
            dim=1, keepdim=True
        ).clamp_min(torch.finfo(assignments.dtype).tiny)
    return assignments


def balanced_assignment_kl(
    log_probabilities: Tensor,
    *,
    temperature: float = 0.1,
    iterations: int = 200,
) -> tuple[Tensor, Tensor]:
    """Return ``KL(stopgrad(Q_OT) || P_model)`` and the OT target."""

    normalized = torch.log_softmax(log_probabilities, dim=1)
    target = balanced_sinkhorn_assignments(
        normalized,
        temperature=temperature,
        iterations=iterations,
    )
    divergence = (
        target
        * (
            torch.log(target.clamp_min(torch.finfo(target.dtype).tiny))
            - normalized
        )
    ).sum(dim=1).mean()
    return divergence, target


def fit_direct_nhp_mixture(
    model: ReferenceNeuralHawkesMixture,
    train_sequences: Iterable[MarkedSequence],
    validation_sequences: Iterable[MarkedSequence],
    *,
    max_epochs: int = 50,
    batch_size: int = 90,
    learning_rate: float = 0.001,
    neural_weight_decay: float = 1e-5,
    evaluation_interval: int = 1,
    validation_cutoff: float | None,
    evaluation_batch_size: int = 32,
    gradient_clip: float = 20.0,
    optimization_objective_scale: float = 1.0,
    gradient_accumulation_steps: int = 1,
    batch_seed: int = 0,
    validation_labels: np.ndarray | None = None,
    selection_metric: SelectionMetric = "validation_nll",
    show_progress: bool = False,
    progress_description: str | None = None,
    assignment_balance_strength: float = 0.0,
    assignment_balance_temperature: float = 0.1,
    assignment_balance_iterations: int = 200,
    assignment_ot_marginal_penalty: float = 0.0,
    assignment_ot_temperature: float = 1.0,
    assignment_ot_dual_learning_rate: float = 0.05,
    optimize_mixture_logits: bool = True,
) -> DirectNHPMixtureFitResult:
    """Fit a conventional differentiable finite NHP mixture."""

    train = tuple(train_sequences)
    validation = tuple(validation_sequences)
    if model.n_components <= 0:
        raise ValueError("direct mixture fit requires K >= 1")
    if not train or not validation:
        raise ValueError("train and validation sequences must be non-empty")
    labels = _validated_selection_labels(
        validation_labels,
        validation_size=len(validation),
        selection_metric=selection_metric,
    )
    if (
        max_epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0.0
        or evaluation_interval <= 0
        or gradient_clip <= 0.0
        or optimization_objective_scale <= 0.0
        or gradient_accumulation_steps <= 0
        or assignment_balance_strength < 0.0
        or assignment_balance_temperature <= 0.0
        or assignment_balance_iterations <= 0
        or assignment_ot_marginal_penalty < 0.0
        or assignment_ot_temperature <= 0.0
        or assignment_ot_dual_learning_rate <= 0.0
        or (
            assignment_balance_strength > 0.0
            and assignment_ot_marginal_penalty > 0.0
        )
        or (
            model.n_components == 1
            and (
                assignment_balance_strength > 0.0
                or assignment_ot_marginal_penalty > 0.0
            )
        )
    ):
        raise ValueError("optimization settings must be positive")
    named_neural_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name != "mixture_logits"
    ]
    neural_parameters = [
        parameter for _, parameter in named_neural_parameters
    ]
    backbone_parameters = [
        parameter
        for name, parameter in named_neural_parameters
        if name.startswith("backbone.")
    ]
    non_backbone_parameters = [
        parameter
        for name, parameter in named_neural_parameters
        if not name.startswith("backbone.")
    ]
    if not optimize_mixture_logits:
        model.mixture_logits.requires_grad_(False)
    optimizer_groups = [{
        "params": neural_parameters,
        "lr": learning_rate,
        "weight_decay": neural_weight_decay,
    }]
    if optimize_mixture_logits:
        optimizer_groups.append({
            "params": (model.mixture_logits,),
            "lr": learning_rate,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.Adam(optimizer_groups)
    assignment_dual = (
        torch.nn.Parameter(
            torch.zeros(
                model.n_components,
                dtype=model.mixture_logits.dtype,
                device=model.mixture_logits.device,
            )
        )
        if assignment_ot_marginal_penalty > 0.0
        else None
    )
    dual_optimizer = (
        torch.optim.Adam(
            (assignment_dual,),
            lr=assignment_ot_dual_learning_rate,
            maximize=True,
        )
        if assignment_dual is not None
        else None
    )
    optimized = neural_parameters + (
        [model.mixture_logits] if optimize_mixture_logits else []
    )
    rng = np.random.default_rng(batch_seed)
    best_state = copy.deepcopy(model.state_dict())
    best_assignment_dual = (
        assignment_dual.detach().clone()
        if assignment_dual is not None
        else None
    )
    best_validation = math.inf
    best_validation_purity = -math.inf
    best_epoch = -1
    final_gradient_norm = math.nan
    history: list[dict[str, float | int | bool | str]] = []
    indices = tuple(range(model.n_components))
    progress = tqdm(
        range(1, max_epochs + 1),
        desc=progress_description or "direct NHP mixture",
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=2.0,
    )
    progress_metrics: dict[str, str] = {}
    for epoch in progress:
        batches = tuple(
            _selected_sequences(train, rng, batch_size)
            for _ in range(gradient_accumulation_steps)
        )
        batch_sequences = tuple(
            sequence for batch in batches for sequence in batch
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if dual_optimizer is not None:
            dual_optimizer.zero_grad(set_to_none=True)
        detached_losses: list[Tensor] = []
        detached_negative_log_likelihoods: list[Tensor] = []
        detached_balance_divergences: list[Tensor] = []
        detached_ot_assignments: list[Tensor] = []
        ot_reference_marginal = None
        for batch in batches:
            component = model.component_scores_for_indices(batch, indices)
            log_weights = model.mixture_log_weights()[None].expand_as(
                component
            )
            posterior_logits = component + log_weights
            negative_log_likelihood = -torch.logsumexp(
                posterior_logits, dim=1
            ).mean()
            balance_divergence = None
            ot_free_energy = ot_assignments = None
            if assignment_balance_strength > 0.0:
                balance_divergence, _ = balanced_assignment_kl(
                    posterior_logits,
                    temperature=assignment_balance_temperature,
                    iterations=assignment_balance_iterations,
                )
            if assignment_dual is not None:
                (
                    ot_free_energy,
                    ot_assignments,
                    ot_reference_marginal,
                ) = unbalanced_ot_dual_free_energy(
                    posterior_logits,
                    assignment_dual,
                    temperature=assignment_ot_temperature,
                    marginal_penalty=assignment_ot_marginal_penalty,
                )
            microbatch_loss = (
                ot_free_energy
                if ot_free_energy is not None
                else negative_log_likelihood
            )
            if balance_divergence is not None:
                microbatch_loss = (
                    microbatch_loss
                    + assignment_balance_strength * balance_divergence
                )
            optimization_loss = (
                microbatch_loss
                / optimization_objective_scale
                / gradient_accumulation_steps
            )
            if not bool(torch.isfinite(optimization_loss).detach().cpu()):
                raise RuntimeError("non-finite direct NHP mixture loss")
            optimization_loss.backward()
            detached_losses.append(microbatch_loss.detach())
            detached_negative_log_likelihoods.append(
                negative_log_likelihood.detach()
            )
            if balance_divergence is not None:
                detached_balance_divergences.append(
                    balance_divergence.detach()
                )
            if ot_assignments is not None:
                detached_ot_assignments.append(ot_assignments.detach())
        loss = torch.stack(detached_losses).mean()
        negative_log_likelihood = torch.stack(
            detached_negative_log_likelihoods
        ).mean()
        optimization_loss = loss / optimization_objective_scale
        balance_divergence = (
            torch.stack(detached_balance_divergences).mean()
            if detached_balance_divergences
            else None
        )
        ot_assignments = (
            torch.cat(detached_ot_assignments)
            if detached_ot_assignments
            else None
        )
        neural_gradient_norm = _gradient_l2_norm(neural_parameters)
        backbone_gradient_norm = _gradient_l2_norm(backbone_parameters)
        non_backbone_gradient_norm = _gradient_l2_norm(
            non_backbone_parameters
        )
        mixture_gradient_norm = _gradient_l2_norm(
            (model.mixture_logits,)
        )
        final_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(optimized, gradient_clip)
            .detach()
            .cpu()
        )
        optimizer.step()
        if dual_optimizer is not None:
            dual_optimizer.step()
            with torch.no_grad():
                assignment_dual.sub_(assignment_dual.mean())
        progress_metrics["train_loss"] = f"{float(loss.detach().cpu()):.4f}"
        progress.set_postfix(progress_metrics, refresh=False)
        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        validation_evaluation = evaluate_direct_nhp_mixture(
            model,
            validation,
            cutoff=validation_cutoff,
            batch_size=evaluation_batch_size,
            assignment_dual=assignment_dual,
            assignment_temperature=assignment_ot_temperature,
        )
        exposure = (
            sum(sequence.horizon for sequence in validation)
            if validation_cutoff is None
            else sum(
                max(sequence.horizon - validation_cutoff, 0.0)
                for sequence in validation
            )
        )
        validation_nll = float(
            -validation_evaluation.conditional_suffix_scores.sum()
            / exposure
        )
        validation_purity = (
            cluster_purity(
                labels,
                validation_evaluation.full_cluster_probabilities
                .argmax(dim=1)
                .numpy(),
            )
            if labels is not None
            else float("nan")
        )
        probabilities = (
            validation_evaluation.prefix_cluster_probabilities
        )
        history.append({
            "epoch": epoch,
            "optimizer_steps": epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": len(batch_sequences),
            "train_negative_log_likelihood_per_path": float(
                negative_log_likelihood.detach().cpu()
            ),
            "train_objective_per_path": float(loss.detach().cpu()),
            "train_scaled_optimization_objective": float(
                optimization_loss.detach().cpu()
            ),
            "optimization_objective_scale": optimization_objective_scale,
            "assignment_balance_kl": (
                float(balance_divergence.detach().cpu())
                if balance_divergence is not None
                else 0.0
            ),
            "assignment_balance_strength": assignment_balance_strength,
            "assignment_objective": (
                "global_unbalanced_ot"
                if assignment_dual is not None
                else (
                    "minibatch_sinkhorn_kl"
                    if balance_divergence is not None
                    else "mixture_nll"
                )
            ),
            "assignment_ot_temperature": assignment_ot_temperature,
            "assignment_ot_marginal_penalty": (
                assignment_ot_marginal_penalty
            ),
            "assignment_ot_dual_learning_rate": (
                assignment_ot_dual_learning_rate
            ),
            "assignment_ot_batch_marginal_max_error": (
                float(
                    (
                        ot_assignments.mean(dim=0)
                        - 1.0 / model.n_components
                    ).abs().max().detach().cpu()
                )
                if ot_assignments is not None
                else float("nan")
            ),
            "assignment_ot_dual_stationarity_error": (
                float(
                    (
                        ot_assignments.mean(dim=0)
                        - ot_reference_marginal
                    ).abs().max().detach().cpu()
                )
                if ot_assignments is not None
                and ot_reference_marginal is not None
                else float("nan")
            ),
            "assignment_ot_dual_max_abs": (
                float(assignment_dual.abs().max().detach().cpu())
                if assignment_dual is not None
                else float("nan")
            ),
            "validation_suffix_nll_per_exposure": validation_nll,
            "validation_purity": validation_purity,
            "validation_prefix_posterior_entropy": float(
                -(
                    probabilities * torch.log(probabilities + 1e-12)
                ).sum(dim=1).mean()
            ),
            "gradient_norm": final_gradient_norm,
            "gradient_norm_unscaled_equivalent": (
                final_gradient_norm * optimization_objective_scale
            ),
            "gradient_was_clipped": final_gradient_norm > gradient_clip,
            "gradient_clip_coefficient": min(
                1.0,
                gradient_clip / max(final_gradient_norm, 1e-30),
            ),
            "gradient_norm_neural": neural_gradient_norm,
            "gradient_norm_backbone": backbone_gradient_norm,
            "gradient_norm_non_backbone": non_backbone_gradient_norm,
            "gradient_norm_mixture_logits": mixture_gradient_norm,
            "mixture_logits_optimized": optimize_mixture_logits,
            "gradient_clip_threshold": gradient_clip,
            "batch_mean_events": float(
                np.mean([sequence.count for sequence in batch_sequences])
            ),
            "batch_max_events": max(
                sequence.count for sequence in batch_sequences
            ),
            "batch_total_events": sum(
                sequence.count for sequence in batch_sequences
            ),
            "batch_mean_horizon": float(
                np.mean(
                    [sequence.horizon for sequence in batch_sequences]
                )
            ),
            "batch_max_horizon": max(
                sequence.horizon for sequence in batch_sequences
            ),
            "uses_elbo": False,
            "uses_lal_logic": False,
        })
        if _selection_is_better(
            selection_metric,
            validation_nll=validation_nll,
            validation_purity=validation_purity,
            best_validation_nll=best_validation,
            best_validation_purity=best_validation_purity,
        ):
            best_validation = validation_nll
            best_validation_purity = validation_purity
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_assignment_dual = (
                assignment_dual.detach().clone()
                if assignment_dual is not None
                else None
            )
        progress_metrics.update({
            "val_purity": f"{validation_purity:.4f}",
            "best_purity": f"{best_validation_purity:.4f}",
            "val_nll": f"{validation_nll:.4f}",
            "best_epoch": str(best_epoch),
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        progress.set_postfix(progress_metrics, refresh=False)
    model.load_state_dict(best_state)
    if assignment_dual is not None and best_assignment_dual is not None:
        with torch.no_grad():
            assignment_dual.copy_(best_assignment_dual)
    model.eval()
    return DirectNHPMixtureFitResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_suffix_nll_per_exposure=best_validation,
        best_validation_purity=best_validation_purity,
        selection_metric=selection_metric,
        final_gradient_norm=final_gradient_norm,
        assignment_dual=(
            assignment_dual.detach().cpu().clone()
            if assignment_dual is not None
            else None
        ),
        assignment_temperature=assignment_ot_temperature,
        assignment_objective=(
            "global_unbalanced_ot"
            if assignment_dual is not None
            else (
                "minibatch_sinkhorn_kl"
                if assignment_balance_strength > 0.0
                else "mixture_nll"
            )
        ),
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
    interaction_learning_rate: float | None = None,
    degrees_of_freedom_learning_rate: float | None = None,
    neural_weight_decay: float = 1e-5,
    mean_hyperprior_strength: float = 1.0,
    degrees_of_freedom_prior_strength: float = 0.0,
    degrees_of_freedom_prior_center: float | None = None,
    evaluation_interval: int = 1,
    validation_cutoff: float | None,
    evaluation_batch_size: int = 16,
    gradient_clip: float = 20.0,
    optimization_objective_scale: float = 1.0,
    gradient_accumulation_steps: int = 1,
    batch_seed: int = 0,
    sample_seed: int = 0,
    validation_labels: np.ndarray | None = None,
    selection_metric: SelectionMetric = "validation_nll",
    show_progress: bool = False,
    progress_description: str | None = None,
    exploration_beta_anneal_epochs: int | None = None,
    exploration_beta_schedule: Literal["linear", "exponential"] = "linear",
    exploration_beta_decay_rate: float = 5.0,
    validation_exploration_beta: float | None = None,
    lr_plateau_factor: float | None = None,
    lr_plateau_patience: int = 25,
    lr_plateau_min_lr: float = 1e-6,
    lal_lr_decay_factor: float | None = None,
    lal_lr_decay_tolerance: int = 25,
    lal_lr_min: float | None = 0.001,
    lal_lr_updated: float | None = 0.001,
    assignment_information_strength: float = 0.0,
    assignment_balance_strength: float = 0.0,
    assignment_balance_temperature: float = 0.1,
    assignment_balance_iterations: int = 200,
    assignment_ot_marginal_penalty: float = 0.0,
    assignment_ot_temperature: float = 1.0,
    assignment_ot_dual_learning_rate: float = 0.05,
) -> LatentWishartNHPFitResult:
    """Optimize NHP and Wishart-law parameters by MC marginal likelihood."""

    train = tuple(train_sequences)
    validation = tuple(validation_sequences)
    if not train or not validation:
        raise ValueError("train and validation sequences must be non-empty")
    labels = _validated_selection_labels(
        validation_labels,
        validation_size=len(validation),
        selection_metric=selection_metric,
    )
    if (
        max_epochs <= 0
        or batch_size <= 0
        or train_samples <= 0
        or validation_samples <= 0
        or neural_learning_rate <= 0.0
        or distribution_learning_rate <= 0.0
        or (
            interaction_learning_rate is not None
            and interaction_learning_rate <= 0.0
        )
        or (
            degrees_of_freedom_learning_rate is not None
            and degrees_of_freedom_learning_rate <= 0.0
        )
        or evaluation_interval <= 0
        or gradient_clip <= 0.0
        or optimization_objective_scale <= 0.0
        or gradient_accumulation_steps <= 0
        or mean_hyperprior_strength < 0.0
        or degrees_of_freedom_prior_strength < 0.0
        or (
            degrees_of_freedom_prior_center is not None
            and degrees_of_freedom_prior_center <= model.dimension
        )
        or (
            exploration_beta_anneal_epochs is not None
            and exploration_beta_anneal_epochs <= 1
        )
        or exploration_beta_schedule not in {"linear", "exponential"}
        or exploration_beta_decay_rate <= 0.0
        or (
            validation_exploration_beta is not None
            and validation_exploration_beta < 0.0
        )
        or (
            lr_plateau_factor is not None
            and not 0.0 < lr_plateau_factor < 1.0
        )
        or lr_plateau_patience < 0
        or lr_plateau_min_lr < 0.0
        or (
            lal_lr_decay_factor is not None
            and not 0.0 < lal_lr_decay_factor < 1.0
        )
        or lal_lr_decay_tolerance <= 0
        or (lal_lr_min is not None and lal_lr_min < 0.0)
        or (lal_lr_updated is not None and lal_lr_updated < 0.0)
        or assignment_information_strength < 0.0
        or assignment_balance_strength < 0.0
        or assignment_balance_temperature <= 0.0
        or assignment_balance_iterations <= 0
        or assignment_ot_marginal_penalty < 0.0
        or assignment_ot_temperature <= 0.0
        or assignment_ot_dual_learning_rate <= 0.0
        or (
            assignment_ot_marginal_penalty > 0.0
            and (
                assignment_balance_strength > 0.0
                or assignment_information_strength > 0.0
            )
        )
        or (
            lal_lr_decay_factor is not None
            and lr_plateau_factor is not None
        )
        or (
            lal_lr_decay_factor is not None
            and lal_lr_min is not None
            and lal_lr_updated is None
        )
    ):
        raise ValueError("optimization settings are invalid")
    initial_exploration_beta = float(
        getattr(model, "exploration_beta", 0.0)
    )
    if (
        exploration_beta_anneal_epochs is not None
        and not hasattr(model, "exploration_beta")
    ):
        raise ValueError("model does not support Wishart exploration")
    neural_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
        and name != "backbone.mixture_logits"
    ]
    interaction_parameters = list(model.interaction_parameters())
    interaction_parameter_ids = {
        id(parameter) for parameter in interaction_parameters
    }
    degrees_of_freedom_parameters = list(
        model.degrees_of_freedom_parameters()
    )
    degrees_of_freedom_parameter_ids = {
        id(parameter) for parameter in degrees_of_freedom_parameters
    }
    distribution_parameters = [
        parameter
        for parameter in model.distribution_parameters()
        if id(parameter) not in interaction_parameter_ids
        and id(parameter) not in degrees_of_freedom_parameter_ids
    ]
    optimizer_groups = [
        {
            "params": neural_parameters,
            "lr": neural_learning_rate,
            "weight_decay": neural_weight_decay,
        },
        {
            "params": distribution_parameters,
            "lr": distribution_learning_rate,
            "weight_decay": 0.0,
        },
    ]
    interaction_group_index = None
    if interaction_parameters:
        interaction_group_index = len(optimizer_groups)
        optimizer_groups.append({
            "params": interaction_parameters,
            "lr": (
                distribution_learning_rate
                if interaction_learning_rate is None
                else interaction_learning_rate
            ),
            "weight_decay": 0.0,
        })
    degrees_of_freedom_group_index = None
    if degrees_of_freedom_parameters:
        degrees_of_freedom_group_index = len(optimizer_groups)
        optimizer_groups.append({
            "params": degrees_of_freedom_parameters,
            "lr": (
                distribution_learning_rate
                if degrees_of_freedom_learning_rate is None
                else degrees_of_freedom_learning_rate
            ),
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.Adam(optimizer_groups)
    nu_prior_center = float(
        model.degrees_of_freedom
        if degrees_of_freedom_prior_center is None
        else degrees_of_freedom_prior_center
    )
    assignment_dual = (
        torch.nn.Parameter(
            torch.zeros(
                model.n_components,
                dtype=model.dtype,
                device=model.device,
            )
        )
        if assignment_ot_marginal_penalty > 0.0
        else None
    )
    dual_optimizer = (
        torch.optim.Adam(
            (assignment_dual,),
            lr=assignment_ot_dual_learning_rate,
            maximize=True,
        )
        if assignment_dual is not None
        else None
    )
    scheduler = (
        None
        if lr_plateau_factor is None
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=lr_plateau_min_lr,
        )
    )
    lal_lr_decay = (
        None
        if lal_lr_decay_factor is None
        else _LaLAdjacentLossDecay(
            optimizer,
            factor=lal_lr_decay_factor,
            tolerance=lal_lr_decay_tolerance,
            min_lr=lal_lr_min,
            updated_lr=lal_lr_updated,
        )
    )
    optimized = (
        neural_parameters
        + distribution_parameters
        + interaction_parameters
        + degrees_of_freedom_parameters
    )
    rng = np.random.default_rng(batch_seed)
    best_state = copy.deepcopy(model.state_dict())
    best_assignment_dual = (
        assignment_dual.detach().clone()
        if assignment_dual is not None
        else None
    )
    best_validation = math.inf
    best_validation_purity = -math.inf
    best_epoch = -1
    final_gradient_norm = math.nan
    history: list[dict[str, float | int | bool | str]] = []
    progress = tqdm(
        range(1, max_epochs + 1),
        desc=progress_description or "latent Wishart NHP",
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=2.0,
    )
    progress_metrics: dict[str, str] = {}
    for epoch in progress:
        if exploration_beta_anneal_epochs is not None:
            anneal_fraction = min(
                (epoch - 1) / (exploration_beta_anneal_epochs - 1),
                1.0,
            )
            if exploration_beta_schedule == "linear":
                beta_fraction = 1.0 - anneal_fraction
            else:
                endpoint = math.exp(-exploration_beta_decay_rate)
                beta_fraction = (
                    math.exp(
                        -exploration_beta_decay_rate * anneal_fraction
                    )
                    - endpoint
                ) / (1.0 - endpoint)
            model.exploration_beta = initial_exploration_beta * beta_fraction
        training_exploration_beta = float(
            getattr(model, "exploration_beta", 0.0)
        )
        batches = tuple(
            _selected_sequences(train, rng, batch_size)
            for _ in range(gradient_accumulation_steps)
        )
        batch_sequences = tuple(
            sequence for batch in batches for sequence in batch
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if dual_optimizer is not None:
            dual_optimizer.zero_grad(set_to_none=True)
        detached_losses: list[Tensor] = []
        detached_negative_log_likelihoods: list[Tensor] = []
        detached_assignment_mutual_information: list[Tensor] = []
        detached_marginal_entropy: list[Tensor] = []
        detached_conditional_entropy: list[Tensor] = []
        detached_balance_divergences: list[Tensor] = []
        detached_ot_assignments: list[Tensor] = []
        ot_reference_marginal = None
        for microbatch, batch in enumerate(batches):
            assignment_mutual_information = loss_marginal_entropy = (
                loss_conditional_entropy
            ) = balance_divergence = None
            ot_free_energy = ot_assignments = None
            current_sample_seed = (
                sample_seed + epoch * 10_007 + microbatch * 1_009
            )
            if (
                assignment_information_strength > 0.0
                or assignment_balance_strength > 0.0
                or assignment_dual is not None
            ):
                component, gates, _ = model.sampled_component_scores(
                    batch,
                    n_samples=train_samples,
                    sample_seed=current_sample_seed,
                )
                marginal = monte_carlo_marginal_scores(component, gates)
                assignment_probabilities = (
                    cluster_probabilities_from_sampled_posterior(
                        sampled_posterior_log_weights(component, gates)
                    )
                )
                if assignment_dual is not None:
                    (
                        ot_free_energy,
                        ot_assignments,
                        ot_reference_marginal,
                    ) = unbalanced_ot_dual_free_energy(
                        integrated_wishart_component_scores(
                            component, gates
                        ),
                        assignment_dual,
                        temperature=assignment_ot_temperature,
                        marginal_penalty=assignment_ot_marginal_penalty,
                    )
                mean_assignment = assignment_probabilities.mean(dim=0)
                if assignment_information_strength > 0.0:
                    loss_marginal_entropy = -(
                        mean_assignment
                        * torch.log(mean_assignment + 1e-12)
                    ).sum()
                    loss_conditional_entropy = -(
                        assignment_probabilities
                        * torch.log(assignment_probabilities + 1e-12)
                    ).sum(dim=1).mean()
                    assignment_mutual_information = (
                        loss_marginal_entropy - loss_conditional_entropy
                    )
                if assignment_balance_strength > 0.0:
                    balance_divergence, _ = balanced_assignment_kl(
                        torch.log(assignment_probabilities + 1e-12),
                        temperature=assignment_balance_temperature,
                        iterations=assignment_balance_iterations,
                    )
            else:
                marginal = model.marginal_scores(
                    batch,
                    n_samples=train_samples,
                    sample_seed=current_sample_seed,
                )
            hyperprior = model.mean_matrix_hyperprior_penalty(
                strength=mean_hyperprior_strength,
            )
            nu_prior = model.degrees_of_freedom_prior_penalty(
                strength=degrees_of_freedom_prior_strength,
                center=nu_prior_center,
            )
            negative_mc_log_likelihood = -marginal.mean()
            microbatch_loss = (
                ot_free_energy
                if ot_free_energy is not None
                else negative_mc_log_likelihood
            ) + (hyperprior + nu_prior) / len(train)
            if assignment_mutual_information is not None:
                microbatch_loss = (
                    microbatch_loss
                    - assignment_information_strength
                    * assignment_mutual_information
                )
            if balance_divergence is not None:
                microbatch_loss = (
                    microbatch_loss
                    + assignment_balance_strength * balance_divergence
                )
            microbatch_optimization_loss = (
                microbatch_loss
                / optimization_objective_scale
                / gradient_accumulation_steps
            )
            if not bool(
                torch.isfinite(microbatch_optimization_loss).detach().cpu()
            ):
                raise RuntimeError("non-finite latent Wishart NHP loss")
            microbatch_optimization_loss.backward()
            detached_losses.append(microbatch_loss.detach())
            detached_negative_log_likelihoods.append(
                negative_mc_log_likelihood.detach()
            )
            if assignment_mutual_information is not None:
                detached_assignment_mutual_information.append(
                    assignment_mutual_information.detach()
                )
                detached_marginal_entropy.append(
                    loss_marginal_entropy.detach()
                )
                detached_conditional_entropy.append(
                    loss_conditional_entropy.detach()
                )
            if balance_divergence is not None:
                detached_balance_divergences.append(
                    balance_divergence.detach()
                )
            if ot_assignments is not None:
                detached_ot_assignments.append(ot_assignments.detach())
        loss = torch.stack(detached_losses).mean()
        negative_mc_log_likelihood = torch.stack(
            detached_negative_log_likelihoods
        ).mean()
        optimization_loss = loss / optimization_objective_scale
        assignment_mutual_information = (
            torch.stack(detached_assignment_mutual_information).mean()
            if detached_assignment_mutual_information
            else None
        )
        loss_marginal_entropy = (
            torch.stack(detached_marginal_entropy).mean()
            if detached_marginal_entropy
            else None
        )
        loss_conditional_entropy = (
            torch.stack(detached_conditional_entropy).mean()
            if detached_conditional_entropy
            else None
        )
        balance_divergence = (
            torch.stack(detached_balance_divergences).mean()
            if detached_balance_divergences
            else None
        )
        ot_assignments = (
            torch.cat(detached_ot_assignments)
            if detached_ot_assignments
            else None
        )
        degrees_of_freedom_gradient = (
            float(
                degrees_of_freedom_parameters[0].grad.detach().cpu()
            )
            if degrees_of_freedom_parameters
            and degrees_of_freedom_parameters[0].grad is not None
            else float("nan")
        )
        final_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(optimized, gradient_clip)
            .detach()
            .cpu()
        )
        optimizer.step()
        if hasattr(model, "project_interaction_strength_"):
            model.project_interaction_strength_()
        if dual_optimizer is not None:
            dual_optimizer.step()
            with torch.no_grad():
                assignment_dual.sub_(assignment_dual.mean())
        if scheduler is not None:
            scheduler.step(float(loss.detach().cpu()))
        lal_lr_decay_applied = (
            False
            if lal_lr_decay is None
            else lal_lr_decay.step(float(loss.detach().cpu()))
        )
        progress_metrics["train_loss"] = f"{float(loss.detach().cpu()):.4f}"
        progress.set_postfix(progress_metrics, refresh=False)
        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        if validation_exploration_beta is not None:
            model.exploration_beta = validation_exploration_beta
        try:
            validation_evaluation = evaluate_latent_wishart_nhp(
                model,
                validation,
                cutoff=validation_cutoff,
                n_samples=validation_samples,
                sample_seed=sample_seed + 100_000,
                batch_size=evaluation_batch_size,
                assignment_dual=assignment_dual,
                assignment_temperature=assignment_ot_temperature,
            )
        finally:
            if validation_exploration_beta is not None:
                model.exploration_beta = training_exploration_beta
        evaluated_exploration_beta = (
            training_exploration_beta
            if validation_exploration_beta is None
            else validation_exploration_beta
        )
        exposure = (
            sum(sequence.horizon for sequence in validation)
            if validation_cutoff is None
            else sum(
                max(sequence.horizon - validation_cutoff, 0.0)
                for sequence in validation
            )
        )
        validation_nll = float(
            -validation_evaluation.conditional_suffix_scores.sum()
            / exposure
        )
        validation_purity = (
            cluster_purity(
                labels,
                validation_evaluation.full_cluster_probabilities
                .argmax(dim=1)
                .numpy(),
            )
            if labels is not None
            else float("nan")
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
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": len(batch_sequences),
            "train_negative_mc_log_posterior_per_path": float(
                loss.detach().cpu()
            ),
            "train_scaled_optimization_objective": float(
                optimization_loss.detach().cpu()
            ),
            "optimization_objective_scale": optimization_objective_scale,
            "train_negative_mc_log_likelihood_per_path": float(
                negative_mc_log_likelihood.detach().cpu()
            ),
            "validation_suffix_nll_per_exposure": validation_nll,
            "validation_purity": validation_purity,
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
            "gradient_norm_unscaled_equivalent": (
                final_gradient_norm * optimization_objective_scale
            ),
            "degrees_of_freedom_prior_penalty": float(
                model.degrees_of_freedom_prior_penalty(
                    strength=degrees_of_freedom_prior_strength,
                    center=nu_prior_center,
                ).detach().cpu()
            ),
            "gradient_was_clipped": final_gradient_norm > gradient_clip,
            "gradient_clip_coefficient": min(
                1.0,
                gradient_clip / max(final_gradient_norm, 1e-30),
            ),
            "gradient_clip_threshold": gradient_clip,
            "batch_mean_events": float(
                np.mean([sequence.count for sequence in batch_sequences])
            ),
            "batch_max_events": max(
                sequence.count for sequence in batch_sequences
            ),
            "batch_total_events": sum(
                sequence.count for sequence in batch_sequences
            ),
            "degrees_of_freedom": model.degrees_of_freedom,
            "learnable_degrees_of_freedom": bool(
                model.learns_degrees_of_freedom
            ),
            "degrees_of_freedom_gradient": degrees_of_freedom_gradient,
            "maximum_degrees_of_freedom": float(
                model.maximum_degrees_of_freedom
            ),
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "transformation": model.transformation_name,
            "interaction_strength": (
                float(model.interaction_strength().detach().cpu())
                if hasattr(model, "interaction_strength")
                else float("nan")
            ),
            "interaction_temperature": float(
                getattr(model, "interaction_temperature", float("nan"))
            ),
            "training_exploration_beta": training_exploration_beta,
            "validation_exploration_beta": evaluated_exploration_beta,
            "exploration_degrees_of_freedom": int(
                getattr(model, "exploration_degrees_of_freedom", 0)
            ),
            "lr_plateau_factor": (
                float(lr_plateau_factor)
                if lr_plateau_factor is not None
                else float("nan")
            ),
            "lr_plateau_patience": lr_plateau_patience,
            "lal_lr_decay_factor": (
                float(lal_lr_decay_factor)
                if lal_lr_decay_factor is not None
                else float("nan")
            ),
            "lal_lr_decay_tolerance": lal_lr_decay_tolerance,
            "lal_lr_decay_counter": (
                lal_lr_decay.checker if lal_lr_decay is not None else 0
            ),
            "lal_lr_decay_applied": lal_lr_decay_applied,
            "neural_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "distribution_learning_rate": float(
                optimizer.param_groups[1]["lr"]
            ),
            "interaction_learning_rate": (
                float(optimizer.param_groups[interaction_group_index]["lr"])
                if interaction_group_index is not None
                else float("nan")
            ),
            "degrees_of_freedom_learning_rate": (
                float(
                    optimizer.param_groups[
                        degrees_of_freedom_group_index
                    ]["lr"]
                )
                if degrees_of_freedom_group_index is not None
                else float("nan")
            ),
            "assignment_information_strength": (
                assignment_information_strength
            ),
            "assignment_mutual_information": (
                float(assignment_mutual_information.detach().cpu())
                if assignment_mutual_information is not None
                else float("nan")
            ),
            "assignment_marginal_entropy": (
                float(loss_marginal_entropy.detach().cpu())
                if loss_marginal_entropy is not None
                else float("nan")
            ),
            "assignment_conditional_entropy": (
                float(loss_conditional_entropy.detach().cpu())
                if loss_conditional_entropy is not None
                else float("nan")
            ),
            "assignment_balance_strength": assignment_balance_strength,
            "assignment_balance_temperature": (
                assignment_balance_temperature
            ),
            "assignment_balance_iterations": assignment_balance_iterations,
            "assignment_balance_kl": (
                float(balance_divergence.detach().cpu())
                if balance_divergence is not None
                else 0.0
            ),
            "assignment_objective": (
                "global_unbalanced_ot"
                if assignment_dual is not None
                else (
                    "minibatch_sinkhorn_kl"
                    if balance_divergence is not None
                    else (
                        "infomax"
                        if assignment_mutual_information is not None
                        else "mixture_nll"
                    )
                )
            ),
            "assignment_ot_temperature": assignment_ot_temperature,
            "assignment_ot_marginal_penalty": (
                assignment_ot_marginal_penalty
            ),
            "assignment_ot_dual_learning_rate": (
                assignment_ot_dual_learning_rate
            ),
            "assignment_ot_batch_marginal_max_error": (
                float(
                    (
                        ot_assignments.mean(dim=0)
                        - 1.0 / model.n_components
                    ).abs().max().detach().cpu()
                )
                if ot_assignments is not None
                else float("nan")
            ),
            "assignment_ot_dual_stationarity_error": (
                float(
                    (
                        ot_assignments.mean(dim=0)
                        - ot_reference_marginal
                    ).abs().max().detach().cpu()
                )
                if ot_assignments is not None
                and ot_reference_marginal is not None
                else float("nan")
            ),
            "assignment_ot_dual_max_abs": (
                float(assignment_dual.abs().max().detach().cpu())
                if assignment_dual is not None
                else float("nan")
            ),
            "uses_deterministic_matrices": (
                model.uses_deterministic_matrices
            ),
            "uses_local_w_parameters": False,
            "uses_elbo": False,
            "uses_lal_logic": False,
            "w_is_integrated_by_monte_carlo": (
                not model.uses_deterministic_matrices
            ),
        })
        if _selection_is_better(
            selection_metric,
            validation_nll=validation_nll,
            validation_purity=validation_purity,
            best_validation_nll=best_validation,
            best_validation_purity=best_validation_purity,
        ):
            best_validation = validation_nll
            best_validation_purity = validation_purity
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_assignment_dual = (
                assignment_dual.detach().clone()
                if assignment_dual is not None
                else None
            )
        progress_metrics.update({
            "val_purity": f"{validation_purity:.4f}",
            "best_purity": f"{best_validation_purity:.4f}",
            "val_nll": f"{validation_nll:.4f}",
            "best_epoch": str(best_epoch),
            "lr_neural": f"{optimizer.param_groups[0]['lr']:.2e}",
            "lr_omega": f"{optimizer.param_groups[1]['lr']:.2e}",
            "beta": f"{training_exploration_beta:.3f}",
        })
        if interaction_parameters:
            progress_metrics["lr_alpha"] = (
                f"{optimizer.param_groups[-1]['lr']:.2e}"
            )
        progress.set_postfix(progress_metrics, refresh=False)
    model.load_state_dict(best_state)
    if assignment_dual is not None and best_assignment_dual is not None:
        with torch.no_grad():
            assignment_dual.copy_(best_assignment_dual)
    if validation_exploration_beta is not None:
        model.exploration_beta = validation_exploration_beta
    model.eval()
    return LatentWishartNHPFitResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_suffix_nll_per_exposure=best_validation,
        best_validation_purity=best_validation_purity,
        selection_metric=selection_metric,
        final_gradient_norm=final_gradient_norm,
        assignment_dual=(
            assignment_dual.detach().cpu().clone()
            if assignment_dual is not None
            else None
        ),
        assignment_temperature=assignment_ot_temperature,
        assignment_objective=(
            "global_unbalanced_ot"
            if assignment_dual is not None
            else (
                "minibatch_sinkhorn_kl"
                if assignment_balance_strength > 0.0
                else "mixture_nll"
            )
        ),
    )
