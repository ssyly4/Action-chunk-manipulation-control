from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
import toppra as ta
from toppra.interpolator import AbstractGeometricPath


class RestrictedPath(AbstractGeometricPath):
    """A view of an existing path over a smaller phase interval."""

    def __init__(self, path: AbstractGeometricPath, start: float, end: float) -> None:
        lo, hi = np.asarray(path.path_interval, dtype=np.float64)
        if not lo <= start < end <= hi:
            raise ValueError("restricted path interval is outside the reference path")
        self._path = path
        self._interval = np.asarray((start, end), dtype=np.float64)

    def __call__(self, path_positions, order: int = 0) -> np.ndarray:
        return self._path(path_positions, order)

    @property
    def dof(self) -> int:
        return self._path.dof

    @property
    def path_interval(self) -> np.ndarray:
        return self._interval.copy()


@dataclass(frozen=True)
class ProjectionResult:
    phase: float
    weighted_error: float
    max_joint_error: float


class ReferencePath:
    """Smooth q_ref(s) passing through every pi0.5 action waypoint."""

    def __init__(self, arm_waypoints: np.ndarray, *, bc_type: str = "natural") -> None:
        values = np.asarray(arm_waypoints, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("arm_waypoints must be a finite HxD matrix")
        self.waypoints = values.copy()
        self.phases = np.arange(len(values), dtype=np.float64)
        self.path = ta.SplineInterpolator(self.phases, values, bc_type=bc_type)

    @property
    def horizon(self) -> int:
        return len(self.waypoints)

    @property
    def dofs(self) -> int:
        return self.waypoints.shape[1]

    def evaluate(self, phase: np.ndarray | float, order: int = 0) -> np.ndarray:
        return np.asarray(self.path(phase, order), dtype=np.float64)

    def linear_reference(self, phase: np.ndarray) -> np.ndarray:
        samples = np.asarray(phase, dtype=np.float64)
        left = np.minimum(np.floor(samples).astype(np.int64), self.horizon - 2)
        alpha = samples - left
        return self.waypoints[left] + alpha[:, None] * (
            self.waypoints[left + 1] - self.waypoints[left]
        )

    def spline_deviation_from_polyline(self, *, samples_per_segment: int = 20) -> float:
        dense = np.linspace(0.0, self.horizon - 1, (self.horizon - 1) * samples_per_segment + 1)
        return float(np.max(np.abs(self.evaluate(dense) - self.linear_reference(dense))))

    def project(
        self,
        measured: np.ndarray,
        *,
        lower: float = 0.0,
        upper: float | None = None,
        weights: np.ndarray | None = None,
        coarse_samples_per_step: int = 16,
    ) -> ProjectionResult:
        q = np.asarray(measured, dtype=np.float64)
        if q.shape != (self.dofs,) or not np.isfinite(q).all():
            raise ValueError(f"measured must have shape ({self.dofs},)")
        hi = float(self.horizon - 1 if upper is None else upper)
        if not 0.0 <= lower <= hi <= self.horizon - 1:
            raise ValueError("invalid projection interval")
        w = np.ones(self.dofs) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (self.dofs,) or np.any(w <= 0.0) or not np.isfinite(w).all():
            raise ValueError("projection weights must be a finite positive vector")

        count = max(3, int(np.ceil((hi - lower) * coarse_samples_per_step)) + 1)
        coarse_phase = np.linspace(lower, hi, count)
        coarse_q = self.evaluate(coarse_phase)
        cost = np.sum(np.square((coarse_q - q) * w[None, :]), axis=1)
        best = int(np.argmin(cost))
        local_lo = coarse_phase[max(0, best - 1)]
        local_hi = coarse_phase[min(count - 1, best + 1)]

        def objective(s: float) -> float:
            return float(np.sum(np.square((self.evaluate(s) - q) * w)))

        if local_hi - local_lo <= 1e-12:
            phase = float(coarse_phase[best])
        else:
            solved = minimize_scalar(objective, bounds=(local_lo, local_hi), method="bounded")
            phase = float(solved.x)
        error = self.evaluate(phase) - q
        return ProjectionResult(
            phase=phase,
            weighted_error=float(np.sqrt(np.sum(np.square(error * w)))),
            max_joint_error=float(np.max(np.abs(error))),
        )

