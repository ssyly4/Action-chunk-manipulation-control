from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import osqp
from scipy import sparse


def _as_limit(value: np.ndarray | float, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{label} must be positive and have shape ({size},)")
    return array


def _difference_matrix(horizon: int, order: int, dt: float) -> sparse.csc_matrix:
    if order < 1 or horizon <= order:
        raise ValueError("difference order must be smaller than the horizon")
    matrix = np.diff(np.eye(horizon, dtype=np.float64), n=order, axis=0)
    return sparse.csc_matrix(matrix / dt**order)


def _finite_differences(
    positions: np.ndarray, action_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.diff(positions, axis=0) * action_hz
    acceleration = np.diff(velocity, axis=0) * action_hz
    jerk = np.diff(acceleration, axis=0) * action_hz
    return velocity, acceleration, jerk


def _maximum_ratio(values: np.ndarray, limits: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values) / limits))


@dataclass(frozen=True)
class WaypointSmootherConfig:
    action_hz: float
    arm_columns: tuple[int, ...]
    trust_region: np.ndarray | float
    max_velocity: np.ndarray | float
    max_acceleration: np.ndarray | float
    max_jerk: np.ndarray | float
    lower_position: np.ndarray | None = None
    upper_position: np.ndarray | None = None
    tracking_weight: float = 20.0
    velocity_tracking_weight: float = 0.25
    acceleration_weight: float = 0.10
    jerk_weight: float = 1.0
    terminal_velocity_weight: float = 1.0
    solver_eps_abs: float = 1e-5
    solver_eps_rel: float = 1e-5
    solver_max_iterations: int = 4000
    solver_time_limit_sec: float = 0.02
    verification_tolerance: float = 2e-5
    constraint_ratio_tolerance: float = 2e-4

    def __post_init__(self) -> None:
        columns = tuple(int(column) for column in self.arm_columns)
        dofs = len(columns)
        if self.action_hz <= 0.0 or dofs == 0:
            raise ValueError("action_hz and arm_columns must be non-empty and positive")
        if len(set(columns)) != dofs or min(columns) < 0:
            raise ValueError("arm_columns must be unique and non-negative")
        trust = _as_limit(self.trust_region, dofs, "trust_region")
        velocity = _as_limit(self.max_velocity, dofs, "max_velocity")
        acceleration = _as_limit(self.max_acceleration, dofs, "max_acceleration")
        jerk = _as_limit(self.max_jerk, dofs, "max_jerk")
        lower = None if self.lower_position is None else np.asarray(
            self.lower_position, dtype=np.float64
        )
        upper = None if self.upper_position is None else np.asarray(
            self.upper_position, dtype=np.float64
        )
        if (lower is None) != (upper is None):
            raise ValueError("lower_position and upper_position must be supplied together")
        if lower is not None:
            if lower.shape != (dofs,) or upper.shape != (dofs,):
                raise ValueError("position limits must match arm_columns")
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise ValueError("position limits must be finite")
            if np.any(lower >= upper):
                raise ValueError("lower_position must be below upper_position")
        weights = (
            self.tracking_weight,
            self.velocity_tracking_weight,
            self.acceleration_weight,
            self.jerk_weight,
            self.terminal_velocity_weight,
        )
        if any(weight < 0.0 for weight in weights) or self.tracking_weight == 0.0:
            raise ValueError("objective weights must be non-negative with positive tracking")
        if self.solver_max_iterations < 1 or self.solver_time_limit_sec <= 0.0:
            raise ValueError("solver limits must be positive")
        if self.solver_eps_abs <= 0.0 or self.solver_eps_rel <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.verification_tolerance < 0.0 or self.constraint_ratio_tolerance < 0.0:
            raise ValueError("verification tolerances must be non-negative")
        object.__setattr__(self, "arm_columns", columns)
        object.__setattr__(self, "trust_region", trust)
        object.__setattr__(self, "max_velocity", velocity)
        object.__setattr__(self, "max_acceleration", acceleration)
        object.__setattr__(self, "max_jerk", jerk)
        object.__setattr__(self, "lower_position", lower)
        object.__setattr__(self, "upper_position", upper)


@dataclass(frozen=True)
class WaypointSmoothingResult:
    status: str
    feasible: bool
    fallback: bool
    reason: str | None
    commands: np.ndarray
    solve_ms: float
    iterations: int
    metrics: dict[str, Any]


@dataclass
class _Problem:
    solver: osqp.OSQP
    first_difference: sparse.csc_matrix
    second_difference: sparse.csc_matrix
    third_difference: sparse.csc_matrix
    previous_solution: np.ndarray | None = None


class OsqpWaypointSmoother:
    """Bounded fixed-horizon joint waypoint denoiser.

    The solver may move arm waypoints only inside the configured trust region.
    It never changes the number of actions, the 30 Hz timeline, or gripper data.
    """

    def __init__(self, config: WaypointSmootherConfig) -> None:
        self.config = config
        self._problems: dict[tuple[int, int], _Problem] = {}

    def smooth(
        self,
        actions: np.ndarray,
        *,
        start_index: int = 0,
        boundary_position: np.ndarray | None = None,
        boundary_velocity: np.ndarray | None = None,
    ) -> WaypointSmoothingResult:
        started = time.perf_counter()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError("actions must be a finite HxD matrix with H >= 4")
        if max(self.config.arm_columns) >= values.shape[1]:
            raise ValueError("arm_columns exceed the action width")
        if not 0 <= start_index <= len(values) - 4:
            raise ValueError("start_index must leave at least four actions")
        reference = values[start_index:, self.config.arm_columns].copy()
        horizon, dofs = reference.shape
        boundary_q = reference[0] if boundary_position is None else np.asarray(
            boundary_position, dtype=np.float64
        )
        if boundary_q.shape != (dofs,) or not np.isfinite(boundary_q).all():
            raise ValueError(f"boundary_position must have shape ({dofs},)")
        boundary_v = None if boundary_velocity is None else np.asarray(
            boundary_velocity, dtype=np.float64
        )
        if boundary_v is not None and (
            boundary_v.shape != (dofs,) or not np.isfinite(boundary_v).all()
        ):
            raise ValueError(f"boundary_velocity must have shape ({dofs},)")

        raw_velocity, raw_acceleration, raw_jerk = _finite_differences(
            reference, self.config.action_hz
        )
        problem = self._problem(horizon, dofs)
        linear_cost = self._linear_cost(problem, reference)
        lower, upper = self._bounds(
            problem,
            reference,
            boundary_position=boundary_q,
            boundary_velocity=boundary_v,
        )
        problem.solver.update(q=linear_cost, l=lower, u=upper)
        if problem.previous_solution is not None:
            problem.solver.warm_start(x=problem.previous_solution)
        solution = problem.solver.solve(raise_error=False)
        solve_ms = (time.perf_counter() - started) * 1000.0
        status = str(solution.info.status).lower()
        iterations = int(solution.info.iter)
        accepted_status = status in {"solved", "solved inaccurate"}
        if not accepted_status or solution.x is None:
            return self._fallback(
                values,
                reference,
                raw_velocity,
                raw_acceleration,
                raw_jerk,
                status=status,
                reason=f"OSQP did not solve the QP: {solution.info.status}",
                solve_ms=solve_ms,
                iterations=iterations,
            )

        smoothed = np.asarray(solution.x, dtype=np.float64).reshape(horizon, dofs)
        verification_error = self._verification_error(
            reference,
            smoothed,
            boundary_position=boundary_q,
            boundary_velocity=boundary_v,
        )
        if verification_error is not None:
            return self._fallback(
                values,
                reference,
                raw_velocity,
                raw_acceleration,
                raw_jerk,
                status="verification_failed",
                reason=verification_error,
                solve_ms=solve_ms,
                iterations=iterations,
            )

        commands = values.copy()
        commands[start_index:, self.config.arm_columns] = smoothed
        problem.previous_solution = solution.x.copy()
        metrics = self._metrics(reference, smoothed)
        metrics.update(
            {
                "solver_run_time_ms": float(solution.info.run_time * 1000.0),
                "solver_prim_res": float(solution.info.prim_res),
                "solver_dual_res": float(solution.info.dual_res),
            }
        )
        return WaypointSmoothingResult(
            status=status,
            feasible=True,
            fallback=False,
            reason=None,
            commands=commands,
            solve_ms=solve_ms,
            iterations=iterations,
            metrics=metrics,
        )

    def _problem(self, horizon: int, dofs: int) -> _Problem:
        key = (horizon, dofs)
        cached = self._problems.get(key)
        if cached is not None:
            return cached
        dt = 1.0 / self.config.action_hz
        time_d1 = _difference_matrix(horizon, 1, dt)
        time_d2 = _difference_matrix(horizon, 2, dt)
        time_d3 = _difference_matrix(horizon, 3, dt)
        identity_dof = sparse.eye(dofs, format="csc")
        d1 = sparse.kron(time_d1, identity_dof, format="csc")
        d2 = sparse.kron(time_d2, identity_dof, format="csc")
        d3 = sparse.kron(time_d3, identity_dof, format="csc")
        identity = sparse.eye(horizon * dofs, format="csc")

        trust_scale = np.tile(1.0 / self.config.trust_region, horizon)
        velocity_scale = np.tile(1.0 / self.config.max_velocity, horizon - 1)
        acceleration_scale = np.tile(1.0 / self.config.max_acceleration, horizon - 2)
        jerk_scale = np.tile(1.0 / self.config.max_jerk, horizon - 3)
        track = sparse.diags(trust_scale, format="csc")
        scaled_d1 = sparse.diags(velocity_scale) @ d1
        scaled_d2 = sparse.diags(acceleration_scale) @ d2
        scaled_d3 = sparse.diags(jerk_scale) @ d3
        terminal_d1 = scaled_d1[-dofs:]

        hessian = self.config.tracking_weight * (track.T @ track)
        hessian += self.config.velocity_tracking_weight * (scaled_d1.T @ scaled_d1)
        hessian += self.config.acceleration_weight * (scaled_d2.T @ scaled_d2)
        hessian += self.config.jerk_weight * (scaled_d3.T @ scaled_d3)
        hessian += self.config.terminal_velocity_weight * (terminal_d1.T @ terminal_d1)
        hessian = sparse.triu(2.0 * hessian, format="csc")
        constraints = sparse.vstack((identity, d1, d2, d3), format="csc")

        solver = osqp.OSQP()
        variable_count = horizon * dofs
        constraint_count = constraints.shape[0]
        solver.setup(
            P=hessian,
            q=np.zeros(variable_count, dtype=np.float64),
            A=constraints,
            l=np.full(constraint_count, -np.inf, dtype=np.float64),
            u=np.full(constraint_count, np.inf, dtype=np.float64),
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=self.config.solver_eps_abs,
            eps_rel=self.config.solver_eps_rel,
            max_iter=self.config.solver_max_iterations,
            time_limit=self.config.solver_time_limit_sec,
            adaptive_rho=True,
        )
        problem = _Problem(solver, d1, d2, d3)
        self._problems[key] = problem
        return problem

    def _linear_cost(self, problem: _Problem, reference: np.ndarray) -> np.ndarray:
        horizon, dofs = reference.shape
        flat = reference.reshape(-1)
        trust_scale = np.tile(1.0 / self.config.trust_region, horizon)
        velocity_scale = np.tile(1.0 / self.config.max_velocity, horizon - 1)
        track = sparse.diags(trust_scale, format="csc")
        scaled_d1 = sparse.diags(velocity_scale) @ problem.first_difference
        raw_velocity = problem.first_difference @ flat
        scaled_raw_velocity = velocity_scale * raw_velocity
        terminal_d1 = scaled_d1[-dofs:]
        terminal_target = scaled_raw_velocity[-dofs:]
        linear = -2.0 * self.config.tracking_weight * (track.T @ (track @ flat))
        linear += -2.0 * self.config.velocity_tracking_weight * (
            scaled_d1.T @ scaled_raw_velocity
        )
        linear += -2.0 * self.config.terminal_velocity_weight * (
            terminal_d1.T @ terminal_target
        )
        return np.asarray(linear, dtype=np.float64).reshape(-1)

    def _bounds(
        self,
        problem: _Problem,
        reference: np.ndarray,
        *,
        boundary_position: np.ndarray,
        boundary_velocity: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        horizon, dofs = reference.shape
        trust = np.tile(self.config.trust_region, horizon).reshape(horizon, dofs)
        position_lower = reference - trust
        position_upper = reference + trust
        if self.config.lower_position is not None:
            position_lower = np.maximum(position_lower, self.config.lower_position)
            position_upper = np.minimum(position_upper, self.config.upper_position)
        position_lower[0] = boundary_position
        position_upper[0] = boundary_position

        velocity_lower = np.tile(-self.config.max_velocity, horizon - 1)
        velocity_upper = np.tile(self.config.max_velocity, horizon - 1)
        if boundary_velocity is not None:
            velocity_lower[:dofs] = boundary_velocity
            velocity_upper[:dofs] = boundary_velocity
        acceleration_lower = np.tile(-self.config.max_acceleration, horizon - 2)
        acceleration_upper = np.tile(self.config.max_acceleration, horizon - 2)
        jerk_lower = np.tile(-self.config.max_jerk, horizon - 3)
        jerk_upper = np.tile(self.config.max_jerk, horizon - 3)
        return (
            np.concatenate(
                (position_lower.reshape(-1), velocity_lower, acceleration_lower, jerk_lower)
            ),
            np.concatenate(
                (position_upper.reshape(-1), velocity_upper, acceleration_upper, jerk_upper)
            ),
        )

    def _verification_error(
        self,
        reference: np.ndarray,
        smoothed: np.ndarray,
        *,
        boundary_position: np.ndarray,
        boundary_velocity: np.ndarray | None,
    ) -> str | None:
        tolerance = self.config.verification_tolerance
        ratio_tolerance = self.config.constraint_ratio_tolerance
        deviation = np.abs(smoothed - reference)
        if np.any(deviation > self.config.trust_region + tolerance):
            return "smoothed waypoint exceeded the trust region"
        if np.max(np.abs(smoothed[0] - boundary_position)) > tolerance:
            return "smoothed start position does not match the boundary"
        velocity, acceleration, jerk = _finite_differences(
            smoothed, self.config.action_hz
        )
        if boundary_velocity is not None and np.max(
            np.abs(velocity[0] - boundary_velocity)
        ) > tolerance:
            return "smoothed start velocity does not match the boundary"
        if np.any(
            np.abs(velocity)
            > self.config.max_velocity * (1.0 + ratio_tolerance) + tolerance
        ):
            return "smoothed velocity exceeds the configured limit"
        if np.any(
            np.abs(acceleration)
            > self.config.max_acceleration * (1.0 + ratio_tolerance) + tolerance
        ):
            return "smoothed acceleration exceeds the configured limit"
        if np.any(
            np.abs(jerk)
            > self.config.max_jerk * (1.0 + ratio_tolerance) + tolerance
        ):
            return "smoothed jerk exceeds the configured limit"
        if self.config.lower_position is not None:
            if np.any(smoothed < self.config.lower_position - tolerance) or np.any(
                smoothed > self.config.upper_position + tolerance
            ):
                return "smoothed waypoint exceeds joint position limits"
        return None

    def _metrics(self, reference: np.ndarray, smoothed: np.ndarray) -> dict[str, Any]:
        raw_v, raw_a, raw_j = _finite_differences(reference, self.config.action_hz)
        smooth_v, smooth_a, smooth_j = _finite_differences(
            smoothed, self.config.action_hz
        )
        deviation = smoothed - reference
        return {
            "max_waypoint_deviation_rad": float(np.max(np.abs(deviation))),
            "rms_waypoint_deviation_rad": float(np.sqrt(np.mean(deviation**2))),
            "raw_velocity_ratio": _maximum_ratio(raw_v, self.config.max_velocity),
            "raw_acceleration_ratio": _maximum_ratio(raw_a, self.config.max_acceleration),
            "raw_jerk_ratio": _maximum_ratio(raw_j, self.config.max_jerk),
            "smoothed_velocity_ratio": _maximum_ratio(smooth_v, self.config.max_velocity),
            "smoothed_acceleration_ratio": _maximum_ratio(
                smooth_a, self.config.max_acceleration
            ),
            "smoothed_jerk_ratio": _maximum_ratio(smooth_j, self.config.max_jerk),
            "terminal_velocity_change_rad_s": float(
                np.max(np.abs(smooth_v[-1] - raw_v[-1]))
            ),
        }

    def _fallback(
        self,
        values: np.ndarray,
        reference: np.ndarray,
        raw_velocity: np.ndarray,
        raw_acceleration: np.ndarray,
        raw_jerk: np.ndarray,
        *,
        status: str,
        reason: str,
        solve_ms: float,
        iterations: int,
    ) -> WaypointSmoothingResult:
        metrics = {
            "max_waypoint_deviation_rad": 0.0,
            "rms_waypoint_deviation_rad": 0.0,
            "raw_velocity_ratio": _maximum_ratio(
                raw_velocity, self.config.max_velocity
            ),
            "raw_acceleration_ratio": _maximum_ratio(
                raw_acceleration, self.config.max_acceleration
            ),
            "raw_jerk_ratio": _maximum_ratio(raw_jerk, self.config.max_jerk),
            "smoothed_velocity_ratio": _maximum_ratio(
                raw_velocity, self.config.max_velocity
            ),
            "smoothed_acceleration_ratio": _maximum_ratio(
                raw_acceleration, self.config.max_acceleration
            ),
            "smoothed_jerk_ratio": _maximum_ratio(raw_jerk, self.config.max_jerk),
            "terminal_velocity_change_rad_s": 0.0,
        }
        return WaypointSmoothingResult(
            status=status,
            feasible=False,
            fallback=True,
            reason=reason,
            commands=values.copy(),
            solve_ms=solve_ms,
            iterations=iterations,
            metrics=metrics,
        )
