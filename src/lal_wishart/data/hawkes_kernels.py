"""Exact COTIC Hawkes kernels, compensators, and offspring-lag samplers.

The time-scale convention is

    phi_s(u) = s * phi(s * u),

which changes the temporal shape without changing the branching ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]


class KernelFamily(str, Enum):
    EXPONENTIAL = "exponential"
    SINUS = "sinus"
    RAYLEIGH = "rayleigh"


@dataclass(frozen=True)
class HawkesKernel:
    """A non-negative causal Hawkes kernel with exact integrated mass."""

    family: KernelFamily
    alpha: float
    beta: float
    rho: float = 0.0
    omega: float = 0.0
    amplitude_scale: float = 1.0
    time_scale: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if self.amplitude_scale <= 0 or self.time_scale <= 0:
            raise ValueError("amplitude_scale and time_scale must be positive")
        if self.family is KernelFamily.SINUS:
            if not 0 <= self.rho < 1:
                raise ValueError("sinus rho must satisfy 0 <= rho < 1")
            if self.omega <= 0:
                raise ValueError("sinus omega must be positive")

    @property
    def branching_ratio(self) -> float:
        if self.family is KernelFamily.EXPONENTIAL:
            base = self.alpha
        elif self.family is KernelFamily.SINUS:
            correction = self.rho * self.beta * self.omega
            correction /= self.beta**2 + self.omega**2
            base = self.alpha * (1.0 + correction)
        else:
            base = self.alpha / self.beta
        return self.amplitude_scale * base

    def rescaled(self, time_scale: float, *, name: str | None = None) -> "HawkesKernel":
        """Return ``s * phi(s u)``; the integrated mass is unchanged."""

        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        return replace(
            self,
            time_scale=self.time_scale * float(time_scale),
            name=self.name if name is None else name,
        )

    def value(self, lag: ArrayLike) -> float | FloatArray:
        raw = np.asarray(lag, dtype=float)
        u = self.time_scale * raw
        positive = u > 0

        if self.family is KernelFamily.EXPONENTIAL:
            base = self.alpha * self.beta * np.exp(-self.beta * np.maximum(u, 0))
        elif self.family is KernelFamily.SINUS:
            base = self.alpha * self.beta * np.exp(-self.beta * np.maximum(u, 0))
            base *= 1.0 + self.rho * np.sin(self.omega * u)
        else:
            base = self.alpha * np.maximum(u, 0) * np.exp(
                -0.5 * self.beta * np.maximum(u, 0) ** 2
            )

        result = np.where(
            positive,
            self.amplitude_scale * self.time_scale * base,
            0.0,
        )
        return float(result) if result.ndim == 0 else result

    def cumulative(self, upper: ArrayLike) -> float | FloatArray:
        """Return ``Phi(L) = integral_0^L phi(u) du`` exactly."""

        raw = np.asarray(upper, dtype=float)
        u = self.time_scale * np.maximum(raw, 0)

        if self.family is KernelFamily.EXPONENTIAL:
            base = self.alpha * (1.0 - np.exp(-self.beta * u))
        elif self.family is KernelFamily.SINUS:
            decay = np.exp(-self.beta * u)
            oscillatory = self.omega - decay * (
                self.beta * np.sin(self.omega * u)
                + self.omega * np.cos(self.omega * u)
            )
            base = self.alpha * (
                1.0
                - decay
                + self.rho
                * self.beta
                * oscillatory
                / (self.beta**2 + self.omega**2)
            )
        else:
            base = (self.alpha / self.beta) * (
                1.0 - np.exp(-0.5 * self.beta * u**2)
            )

        result = np.where(raw > 0, self.amplitude_scale * base, 0.0)
        return float(result) if result.ndim == 0 else result

    def sample_lags(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> FloatArray:
        """Sample offspring lags from ``phi(u) / integral(phi)``."""

        if size < 0:
            raise ValueError("size must be non-negative")
        if size == 0:
            return np.empty(0, dtype=float)

        if self.family is KernelFamily.EXPONENTIAL:
            base = rng.exponential(scale=1.0 / self.beta, size=size)
        elif self.family is KernelFamily.RAYLEIGH:
            base = rng.rayleigh(scale=1.0 / math.sqrt(self.beta), size=size)
        else:
            accepted: list[FloatArray] = []
            remaining = size
            expected_acceptance = (
                1.0
                + self.rho
                * self.beta
                * self.omega
                / (self.beta**2 + self.omega**2)
            ) / (1.0 + self.rho)
            while remaining:
                batch = max(16, int(math.ceil(1.15 * remaining / expected_acceptance)))
                proposals = rng.exponential(scale=1.0 / self.beta, size=batch)
                probability = (
                    1.0 + self.rho * np.sin(self.omega * proposals)
                ) / (1.0 + self.rho)
                keep = proposals[rng.random(batch) < probability]
                if keep.size:
                    take = keep[:remaining]
                    accepted.append(take)
                    remaining -= take.size
            base = np.concatenate(accepted)

        return np.asarray(base / self.time_scale, dtype=float)


def cotic_exponential(*, time_scale: float = 1.0) -> HawkesKernel:
    return HawkesKernel(
        family=KernelFamily.EXPONENTIAL,
        alpha=0.35,
        beta=0.7,
        time_scale=time_scale,
        name="cotic_exponential",
    )


def cotic_sinus(
    *,
    variant: str = "cotic_original",
    time_scale: float = 1.0,
) -> HawkesKernel:
    original = HawkesKernel(
        family=KernelFamily.SINUS,
        alpha=0.35,
        beta=0.7,
        rho=0.4,
        omega=2.0 * math.pi,
        time_scale=time_scale,
        name="cotic_sinus_original",
    )
    if variant == "cotic_original":
        return original
    if variant == "branching_matched":
        return replace(
            original,
            amplitude_scale=0.35 / original.branching_ratio,
            name="cotic_sinus_branching_matched",
        )
    raise ValueError(f"unknown sinus variant: {variant!r}")


def cotic_rayleigh(*, time_scale: float = 1.0) -> HawkesKernel:
    return HawkesKernel(
        family=KernelFamily.RAYLEIGH,
        alpha=0.035,
        beta=0.1,
        time_scale=time_scale,
        name="cotic_rayleigh",
    )


def integrated_hawkes_intensity(
    baseline: ArrayLike,
    coupling: ArrayLike,
    kernel: HawkesKernel,
    horizon: float,
    event_times: ArrayLike,
    event_marks: ArrayLike,
    *,
    start: float = 0.0,
) -> FloatArray:
    """Integrate every mark intensity over ``[start, horizon]``.

    ``event_times`` may include pre-window history. Events at or beyond the
    horizon are ignored.
    """

    mu = np.asarray(baseline, dtype=float)
    matrix = np.asarray(coupling, dtype=float)
    times = np.asarray(event_times, dtype=float)
    marks = np.asarray(event_marks, dtype=int)
    if horizon <= start:
        raise ValueError("horizon must be greater than start")
    if times.shape != marks.shape:
        raise ValueError("event_times and event_marks must have the same shape")
    if matrix.shape != (mu.size, mu.size):
        raise ValueError("coupling must be square with one row per mark")

    result = mu * (horizon - start)
    for time, mark in zip(times, marks, strict=True):
        if time >= horizon:
            continue
        upper = kernel.cumulative(horizon - time)
        lower = kernel.cumulative(max(start - time, 0.0))
        result += matrix[:, mark] * (upper - lower)
    return result

