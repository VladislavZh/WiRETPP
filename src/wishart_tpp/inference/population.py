"""Exact trace-constrained population M-step for Omega."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class PopulationMstep:
    """Minimize log|Omega| + tr(Omega^-1 S) with tr(Omega)=C."""

    def __init__(self, eigenvalue_floor: float = 1e-4) -> None:
        self.eigenvalue_floor = eigenvalue_floor

    @staticmethod
    def sufficient_mean(means: Tensor, responsibilities: Tensor) -> Tensor:
        mass = responsibilities.sum(0, keepdim=True).clamp_min(1e-12)
        weights = responsibilities / mass
        result = torch.einsum("nk,nkij->kij", weights, means)
        return 0.5 * (result + result.transpose(-1, -2))

    @staticmethod
    def objective(omega: Tensor, means: Tensor, responsibilities: Tensor) -> Tensor:
        mass = responsibilities.sum(0)
        weighted = torch.einsum("nk,nkij->kij", responsibilities, means)
        cholesky = torch.linalg.cholesky(omega)
        logdet = 2.0 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        solved = torch.cholesky_solve(weighted, cholesky)
        trace = solved.diagonal(dim1=-2, dim2=-1).sum(-1)
        return (mass * logdet + trace).sum() / responsibilities.sum().clamp_min(1e-12)

    @staticmethod
    def _small_roots(eigenvalues: Tensor, multiplier: float) -> Tensor:
        discriminant = (1.0 + 4.0 * multiplier * eigenvalues).clamp_min(0.0)
        return 2.0 * eigenvalues / (1.0 + discriminant.sqrt())

    @staticmethod
    def _eigen_objective(candidate: Tensor, sufficient: Tensor) -> float:
        return float((candidate.log() + sufficient / candidate).sum())

    def _positive_multiplier_solution(
        self, sufficient: Tensor, trace_target: float
    ) -> Tensor:
        low, high = 0.0, 1.0
        while float(self._small_roots(sufficient, high).sum()) > trace_target:
            high *= 2.0
        for _ in range(100):
            middle = 0.5 * (low + high)
            if float(self._small_roots(sufficient, middle).sum()) > trace_target:
                low = middle
            else:
                high = middle
        return self._small_roots(sufficient, 0.5 * (low + high))

    def _bisect_branch(
        self,
        sufficient: Tensor,
        trace_target: float,
        large_index: int | None,
        left: float,
        right: float,
    ) -> Tensor:
        left_values = self._branch(sufficient, left, large_index)
        left_residual = float(left_values.sum()) - trace_target
        for _ in range(80):
            middle = 0.5 * (left + right)
            middle_values = self._branch(sufficient, middle, large_index)
            middle_residual = float(middle_values.sum()) - trace_target
            if left_residual * middle_residual <= 0.0:
                right = middle
            else:
                left, left_residual = middle, middle_residual
        return self._branch(sufficient, 0.5 * (left + right), large_index)

    def _negative_multiplier_solutions(
        self, sufficient: Tensor, trace_target: float
    ) -> list[Tensor]:
        lower = -1.0 / (4.0 * float(sufficient.max()))
        grid = -torch.logspace(
            math.log10(-lower * (1.0 - 1e-12)),
            -12.0,
            512,
            dtype=torch.float64,
        )
        solutions = []

        # A negative multiplier admits one optional large root.
        for large_index in [None, *range(len(sufficient))]:
            previous = float(grid[0])
            previous_values = self._branch(sufficient, previous, large_index)
            previous_residual = float(previous_values.sum()) - trace_target
            for point in grid[1:]:
                multiplier = float(point)
                values = self._branch(sufficient, multiplier, large_index)
                residual = float(values.sum()) - trace_target
                if previous_residual * residual <= 0.0:
                    solutions.append(
                        self._bisect_branch(
                            sufficient, trace_target, large_index, previous, multiplier
                        )
                    )
                previous, previous_residual = multiplier, residual
        return solutions

    def _solve_eigenvalues(self, sufficient: Tensor, trace_target: float) -> Tensor:
        raw_trace = float(sufficient.sum())
        candidates = []

        # The KKT multiplier sign is determined by the unconstrained trace.
        if abs(raw_trace - trace_target) <= 1e-7 * max(1.0, trace_target):
            # Posterior means are trace-normalized in float32.  Their float64
            # sufficient mean can therefore miss the target by a few ulps;
            # treating that as a genuinely negative KKT branch is unstable
            # because the required multiplier is indistinguishable from zero.
            candidates.append(sufficient * (trace_target / raw_trace))
        elif raw_trace > trace_target:
            candidates.append(
                self._positive_multiplier_solution(sufficient, trace_target)
            )
        else:
            candidates.extend(
                self._negative_multiplier_solutions(sufficient, trace_target)
            )
        candidates = [
            values.clone()
            for values in candidates
            if bool(torch.all(values > 0.0))
            and abs(float(values.sum()) - trace_target) < 1e-7
        ]
        if not candidates:
            raise RuntimeError(
                "trace-constrained Omega solver found no feasible branch: "
                f"raw_trace={raw_trace:.17g} target={trace_target:.17g} "
                f"min={float(sufficient.min()):.17g} "
                f"max={float(sufficient.max()):.17g}"
            )
        return min(
            candidates,
            key=lambda value: self._eigen_objective(value, sufficient),
        )

    def _branch(
        self, sufficient: Tensor, multiplier: float, large: int | None
    ) -> Tensor:
        values = self._small_roots(sufficient, multiplier)
        if large is not None:
            discriminant = max(0.0, 1.0 + 4.0 * multiplier * float(sufficient[large]))
            values[large] = (-1.0 - math.sqrt(discriminant)) / (2.0 * multiplier)
        return values

    @torch.no_grad()
    def update(self, means: Tensor, responsibilities: Tensor) -> Tensor:
        sufficient = self.sufficient_mean(means, responsibilities)
        dimension = sufficient.shape[-1]
        outputs = []
        for matrix in sufficient.detach().double().cpu():
            values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
            values = values.clamp_min(1e-12)
            optimum = self._solve_eigenvalues(values, float(dimension))
            reconstructed = vectors @ torch.diag(optimum) @ vectors.T
            eigenvalues, eigenvectors = torch.linalg.eigh(reconstructed)
            eigenvalues = eigenvalues.clamp_min(self.eigenvalue_floor)
            stable = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            stable = stable * (dimension / torch.trace(stable))
            outputs.append(0.5 * (stable + stable.T))
        return torch.stack(outputs).to(device=means.device, dtype=means.dtype)
