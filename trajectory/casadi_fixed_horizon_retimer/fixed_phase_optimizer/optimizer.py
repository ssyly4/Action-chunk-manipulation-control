from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline


def _finite_differences(
    commands: np.ndarray, action_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.diff(commands, axis=0) * action_hz
    acceleration = np.diff(velocity, axis=0) * action_hz
    jerk = np.diff(acceleration, axis=0) * action_hz
    return velocity, acceleration, jerk


@dataclass(frozen=True)
class PhaseOptimizerConfig:
    action_hz: float
    max_velocity: np.ndarray
    max_acceleration: np.ndarray
    max_jerk: np.ndarray | None = None
    arm_columns: tuple[int, ...] | None = None
    minimum_terminal_phase_speed: float = 1.0
    phase_distortion_weight: float = 200.0
    phase_speed_weight: float = 1.0
    phase_acceleration_weight: float = 0.05
    phase_jerk_weight: float = 0.002
    boundary_speed_weight: float = 5.0
    constraint_tolerance: float = 2e-4
    solver_max_iterations: int = 300
    solver_max_cpu_sec: float = 0.2
    solver_tolerance: float = 1e-6
    solver_acceptable_tolerance: float = 5e-6
    solver_acceptable_iterations: int = 15
    solver_warm_start_duals: bool = False

    def __post_init__(self) -> None:
        velocity = np.asarray(self.max_velocity, dtype=np.float64)
        acceleration = np.asarray(self.max_acceleration, dtype=np.float64)
        jerk = None if self.max_jerk is None else np.asarray(self.max_jerk, dtype=np.float64)
        if self.action_hz <= 0.0:
            raise ValueError("action_hz must be positive")
        if velocity.ndim != 1 or acceleration.shape != velocity.shape:
            raise ValueError("velocity and acceleration limits must have equal shape")
        if jerk is not None and jerk.shape != velocity.shape:
            raise ValueError("jerk limits must have the same shape as velocity limits")
        if np.any(velocity <= 0.0) or np.any(acceleration <= 0.0):
            raise ValueError("velocity and acceleration limits must be positive")
        if jerk is not None and np.any(jerk <= 0.0):
            raise ValueError("jerk limits must be positive")
        if self.minimum_terminal_phase_speed < 0.0:
            raise ValueError("minimum terminal phase speed must be non-negative")
        if self.solver_max_cpu_sec <= 0.0:
            raise ValueError("solver_max_cpu_sec must be positive")
        if not 0.0 < self.solver_tolerance <= self.solver_acceptable_tolerance:
            raise ValueError("solver tolerances must be positive and ordered")
        if self.solver_acceptable_iterations < 1:
            raise ValueError("solver_acceptable_iterations must be positive")
        object.__setattr__(self, "max_velocity", velocity)
        object.__setattr__(self, "max_acceleration", acceleration)
        object.__setattr__(self, "max_jerk", jerk)


@dataclass(frozen=True)
class PhaseOptimizerResult:
    status: str
    feasible: bool
    fallback: bool
    reason: str | None
    commands: np.ndarray
    phase_samples: np.ndarray
    phase_speed_samples: np.ndarray
    phase_acceleration_samples: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    solve_ms: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _SolverBundle:
    solver: ca.Function
    lower_constraints: np.ndarray
    upper_constraints: np.ndarray
    lower_phase: np.ndarray
    upper_phase: np.ndarray
    nominal_phase: np.ndarray


class NaturalCubicPath:
    """The same natural cubic q_ref(s) represented numerically and symbolically."""

    def __init__(self, waypoints: np.ndarray) -> None:
        values = np.asarray(waypoints, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("waypoints must be a finite HxD matrix")
        self.waypoints = values.copy()
        self.phases = np.arange(len(values), dtype=np.float64)
        self.spline = CubicSpline(self.phases, values, axis=0, bc_type="natural")

    def evaluate(self, phase: np.ndarray | float, order: int = 0) -> np.ndarray:
        return np.asarray(self.spline(phase, nu=order), dtype=np.float64)

    def symbolic(self, phase: ca.MX) -> ca.MX:
        coefficients = self.spline.c
        segment_count = coefficients.shape[1]
        expressions: list[ca.MX] = []
        for joint in range(self.waypoints.shape[1]):
            selected: ca.MX | None = None
            for segment in range(segment_count - 1, -1, -1):
                local = phase - float(segment)
                c = coefficients[:, segment, joint]
                polynomial = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
                if selected is None:
                    selected = polynomial
                else:
                    selected = ca.if_else(phase <= segment + 1.0, polynomial, selected)
            assert selected is not None
            expressions.append(selected)
        return ca.vertcat(*expressions)


class CasadiPhaseOptimizer:
    """Fixed-time phase optimizer; never replans the spatial joint path."""

    def __init__(self, config: PhaseOptimizerConfig) -> None:
        self.config = config
        self._solver_cache: dict[tuple[int, int, float, int], _SolverBundle] = {}
        self._warm_phase: dict[tuple[int, int], np.ndarray] = {}
        self._warm_duals: dict[
            tuple[int, int, float, int], tuple[np.ndarray, np.ndarray]
        ] = {}

    def _arm_columns(self, width: int) -> tuple[int, ...]:
        columns = self.config.arm_columns or tuple(range(width))
        if len(columns) != len(self.config.max_velocity):
            raise ValueError("arm_columns count must match the constraint vectors")
        if len(set(columns)) != len(columns) or min(columns) < 0 or max(columns) >= width:
            raise ValueError("arm_columns are invalid")
        return columns

    def optimize(
        self,
        actions: np.ndarray,
        *,
        start_phase: float = 0.0,
        output_ticks: int | None = None,
        start_phase_speed: float | None = None,
        end_phase_speed: float | None = None,
        fallback_start_index: int = 0,
    ) -> PhaseOptimizerResult:
        started = time.perf_counter()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("actions must be a finite HxD matrix")
        arm_columns = self._arm_columns(values.shape[1])
        ticks = len(values) if output_ticks is None else int(output_ticks)
        if ticks < 4 or not 0.0 <= start_phase < len(values) - 1.0:
            raise ValueError("optimization requires four ticks and a non-terminal start phase")
        if not 0 <= fallback_start_index < len(values) - 1:
            raise ValueError("fallback_start_index must retain at least two rows")

        end_phase = float(len(values) - 1)
        raw_phase = np.linspace(float(fallback_start_index), end_phase, ticks)
        fallback_arm = np.column_stack(
            [
                np.interp(raw_phase, np.arange(len(values)), values[:, column])
                for column in arm_columns
            ]
        )
        fallback = self._compose(
            values,
            arm_phase=raw_phase,
            non_arm_phase=raw_phase,
            arm_values=fallback_arm,
            arm_columns=arm_columns,
        )
        path = NaturalCubicPath(values[:, arm_columns])
        nominal_phase = np.linspace(start_phase, end_phase, ticks)
        dt = 1.0 / self.config.action_hz
        nominal_speed = (end_phase - start_phase) / ((ticks - 1) * dt)
        start_speed = nominal_speed if start_phase_speed is None else max(0.0, start_phase_speed)
        terminal_speed = nominal_speed if end_phase_speed is None else max(
            self.config.minimum_terminal_phase_speed, end_phase_speed
        )

        try:
            bundle = self._solver_bundle(
                horizon=len(values),
                ticks=ticks,
                start_phase=start_phase,
                dofs=len(arm_columns),
            )
            parameters = np.r_[
                values[:, arm_columns].flatten(order="F"),
                start_speed,
                terminal_speed,
            ]
            warm_key = (len(values), ticks)
            x0 = self._warm_start(
                bundle.nominal_phase,
                start_phase=start_phase,
                end_phase=end_phase,
                key=warm_key,
            )
            solver_key = (len(values), ticks, float(start_phase), len(arm_columns))
            solver_arguments: dict[str, np.ndarray] = {
                "x0": x0,
                "p": parameters,
                "lbx": bundle.lower_phase,
                "ubx": bundle.upper_phase,
                "lbg": bundle.lower_constraints,
                "ubg": bundle.upper_constraints,
            }
            warm_duals = (
                self._warm_duals.get(solver_key)
                if self.config.solver_warm_start_duals
                else None
            )
            if warm_duals is not None:
                solver_arguments["lam_x0"] = warm_duals[0]
                solver_arguments["lam_g0"] = warm_duals[1]
            solution = bundle.solver(
                **solver_arguments,
            )
            stats = bundle.solver.stats()
            if not stats.get("success", False):
                raise RuntimeError(str(stats.get("return_status", "IPOPT failed")))
            solved_phase = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
            solved_phase[0] = start_phase
            solved_phase[-1] = end_phase
            self._warm_phase[warm_key] = solved_phase.copy()
            if self.config.solver_warm_start_duals:
                self._warm_duals[solver_key] = (
                    np.asarray(solution["lam_x"], dtype=np.float64).reshape(-1),
                    np.asarray(solution["lam_g"], dtype=np.float64).reshape(-1),
                )
            arm_commands = path.evaluate(solved_phase)
            commands = self._compose(
                values,
                arm_phase=solved_phase,
                non_arm_phase=raw_phase,
                arm_values=arm_commands,
                arm_columns=arm_columns,
            )
            result = self._evaluate(
                commands,
                solved_phase,
                nominal_phase,
                arm_columns,
                path,
                started,
                stats,
            )
            if result.feasible:
                return result
            return self._fallback(
                fallback,
                raw_phase,
                arm_columns,
                started,
                result.reason or "discrete verification failed",
                extra=result.metrics,
            )

        except (RuntimeError, ValueError, ArithmeticError) as exc:
            return self._fallback(
                fallback,
                raw_phase,
                arm_columns,
                started,
                f"{type(exc).__name__}: {exc}",
            )

    def _warm_start(
        self,
        nominal_phase: np.ndarray,
        *,
        start_phase: float,
        end_phase: float,
        key: tuple[int, int],
    ) -> np.ndarray:
        previous = self._warm_phase.get(key)
        if previous is None or previous.shape != nominal_phase.shape:
            return nominal_phase.copy()
        previous_span = float(previous[-1] - previous[0])
        if previous_span <= 1e-9:
            return nominal_phase.copy()
        normalized = (previous - previous[0]) / previous_span
        warm = start_phase + normalized * (end_phase - start_phase)
        warm[0] = start_phase
        warm[-1] = end_phase
        return np.maximum.accumulate(warm)

    def _solver_bundle(
        self,
        *,
        horizon: int,
        ticks: int,
        start_phase: float,
        dofs: int,
    ) -> _SolverBundle:
        key = (horizon, ticks, float(start_phase), dofs)
        cached = self._solver_cache.get(key)
        if cached is not None:
            return cached

        end_phase = float(horizon - 1)
        dt = 1.0 / self.config.action_hz
        nominal_phase = np.linspace(start_phase, end_phase, ticks)
        nominal_speed = (end_phase - start_phase) / ((ticks - 1) * dt)
        phase = ca.MX.sym("phase", ticks)
        parameters = ca.MX.sym("parameters", horizon * dofs + 2)
        waypoints = ca.reshape(parameters[: horizon * dofs], horizon, dofs)
        start_speed = parameters[-2]
        terminal_speed = parameters[-1]
        basis_spline = make_interp_spline(
            np.arange(horizon, dtype=np.float64),
            np.eye(horizon, dtype=np.float64),
            axis=0,
            bc_type="natural",
        )
        coefficient_map = np.asarray(basis_spline.c, dtype=np.float64)
        q_rows = [
            self._symbolic_parameterized_path(
                phase[index],
                waypoints,
                coefficient_map,
                np.asarray(basis_spline.t, dtype=np.float64),
                int(basis_spline.k),
            )
            for index in range(ticks)
        ]
        velocity = [(q_rows[index + 1] - q_rows[index]) / dt for index in range(ticks - 1)]
        acceleration = [
            (velocity[index + 1] - velocity[index]) / dt for index in range(ticks - 2)
        ]
        jerk = [
            (acceleration[index + 1] - acceleration[index]) / dt
            for index in range(ticks - 3)
        ]
        phase_speed = [(phase[index + 1] - phase[index]) / dt for index in range(ticks - 1)]
        phase_acceleration = [
            (phase_speed[index + 1] - phase_speed[index]) / dt
            for index in range(ticks - 2)
        ]

        scale = max(1.0, end_phase - start_phase)
        objective = self.config.phase_distortion_weight * ca.sumsqr(
            (phase - ca.DM(nominal_phase)) / scale
        )
        objective += self.config.phase_speed_weight * ca.sumsqr(
            ca.vertcat(*phase_speed) / max(1.0, nominal_speed) - 1.0
        )
        objective += self.config.phase_acceleration_weight * ca.sumsqr(
            ca.vertcat(*phase_acceleration) / max(1.0, nominal_speed * self.config.action_hz)
        )
        if len(phase_acceleration) > 1:
            phase_jerk = [
                (phase_acceleration[index + 1] - phase_acceleration[index]) / dt
                for index in range(len(phase_acceleration) - 1)
            ]
            objective += self.config.phase_jerk_weight * ca.sumsqr(
                ca.vertcat(*phase_jerk) / max(1.0, nominal_speed * self.config.action_hz**2)
            )
        objective += self.config.boundary_speed_weight * (
            ((phase_speed[0] - start_speed) / max(1.0, nominal_speed)) ** 2
            + ((phase_speed[-1] - terminal_speed) / max(1.0, nominal_speed)) ** 2
        )

        constraints: list[ca.MX] = [phase[0], phase[-1]]
        lower = [start_phase, end_phase]
        upper = [start_phase, end_phase]
        for index in range(ticks - 1):
            constraints.append(phase[index + 1] - phase[index])
            lower.append(0.0)
            upper.append(np.inf)
        constraints.append(phase[-1] - phase[-2])
        lower.append(self.config.minimum_terminal_phase_speed * dt)
        upper.append(np.inf)
        for row in velocity:
            constraints.append(row)
            lower.extend((-self.config.max_velocity).tolist())
            upper.extend(self.config.max_velocity.tolist())
        for row in acceleration:
            constraints.append(row)
            lower.extend((-self.config.max_acceleration).tolist())
            upper.extend(self.config.max_acceleration.tolist())
        if self.config.max_jerk is not None:
            for row in jerk:
                constraints.append(row)
                lower.extend((-self.config.max_jerk).tolist())
                upper.extend(self.config.max_jerk.tolist())

        solver_options: dict[str, Any] = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.config.solver_max_iterations,
            "ipopt.max_cpu_time": self.config.solver_max_cpu_sec,
            "ipopt.tol": self.config.solver_tolerance,
            "ipopt.acceptable_tol": self.config.solver_acceptable_tolerance,
            "ipopt.acceptable_iter": self.config.solver_acceptable_iterations,
            "ipopt.mu_strategy": "adaptive",
        }
        if self.config.solver_warm_start_duals:
            solver_options.update(
                {
                    "ipopt.warm_start_init_point": "yes",
                    "ipopt.warm_start_bound_push": 1e-6,
                    "ipopt.warm_start_mult_bound_push": 1e-6,
                    "ipopt.warm_start_slack_bound_push": 1e-6,
                }
            )
        solver = ca.nlpsol(
            f"fixed_phase_{id(self)}_{horizon}_{ticks}_{int(round(start_phase * 1000))}",
            "ipopt",
            {"x": phase, "p": parameters, "f": objective, "g": ca.vertcat(*constraints)},
            solver_options,
        )
        bundle = _SolverBundle(
            solver=solver,
            lower_constraints=np.asarray(lower, dtype=np.float64),
            upper_constraints=np.asarray(upper, dtype=np.float64),
            lower_phase=np.full(ticks, start_phase),
            upper_phase=np.full(ticks, end_phase),
            nominal_phase=nominal_phase,
        )
        self._solver_cache[key] = bundle
        return bundle

    @staticmethod
    def _symbolic_parameterized_path(
        phase: ca.MX,
        waypoints: ca.MX,
        coefficient_map: np.ndarray,
        knots: np.ndarray,
        degree: int,
    ) -> ca.MX:
        coefficients = ca.mtimes(ca.DM(coefficient_map), waypoints)
        # CasADi's multi-output bspline expects row-major [coefficient, dof]
        # ordering, which is column-major flattening of the transpose.
        flattened = ca.reshape(coefficients.T, coefficients.numel(), 1)
        return ca.bspline(
            phase,
            flattened,
            [knots.tolist()],
            [degree],
            int(waypoints.shape[1]),
            {},
        )

    def _compose(
        self,
        actions: np.ndarray,
        *,
        arm_phase: np.ndarray,
        non_arm_phase: np.ndarray,
        arm_values: np.ndarray,
        arm_columns: Sequence[int],
    ) -> np.ndarray:
        if len(arm_phase) != len(non_arm_phase):
            raise ValueError("arm and non-arm phases must have equal lengths")
        result = np.empty((len(arm_phase), actions.shape[1]), dtype=np.float64)
        arm_set = set(arm_columns)
        arm = np.asarray(arm_values, dtype=np.float64)
        if arm.shape != (len(arm_phase), len(arm_columns)):
            raise ValueError("arm_values do not match phase samples and arm columns")
        result[:, arm_columns] = arm
        for column in sorted(set(range(actions.shape[1])) - arm_set):
            # Retiming is defined only for the arm path. Preserve gripper/event
            # channels on their original fixed-rate action timeline.
            result[:, column] = np.interp(
                non_arm_phase, np.arange(len(actions)), actions[:, column]
            )
        return result

    def _evaluate(
        self,
        commands: np.ndarray,
        phase: np.ndarray,
        nominal_phase: np.ndarray,
        arm_columns: Sequence[int],
        path: NaturalCubicPath,
        started: float,
        stats: dict[str, Any],
    ) -> PhaseOptimizerResult:
        arm = commands[:, arm_columns]
        velocity, acceleration, jerk = _finite_differences(arm, self.config.action_hz)
        tol = self.config.constraint_tolerance
        velocity_ratio = float(np.max(np.abs(velocity) / self.config.max_velocity))
        acceleration_ratio = float(np.max(np.abs(acceleration) / self.config.max_acceleration))
        jerk_ratio = None
        if self.config.max_jerk is not None and len(jerk):
            jerk_ratio = float(np.max(np.abs(jerk) / self.config.max_jerk))
        phase_speed = np.diff(phase) * self.config.action_hz
        phase_acceleration = np.diff(phase_speed) * self.config.action_hz
        path_error = float(np.max(np.abs(arm - path.evaluate(phase))))
        phase_rms = float(np.sqrt(np.mean(np.square(phase - nominal_phase))))
        feasible = (
            np.all(np.diff(phase) >= -tol)
            and velocity_ratio <= 1.0 + tol
            and acceleration_ratio <= 1.0 + tol
            and (jerk_ratio is None or jerk_ratio <= 1.0 + tol)
            and path_error <= 1e-9
        )
        reason = None if feasible else (
            f"discrete constraints failed: v={velocity_ratio:.4f} "
            f"a={acceleration_ratio:.4f} j={jerk_ratio}"
        )
        solve_ms = (time.perf_counter() - started) * 1000.0
        return PhaseOptimizerResult(
            status="optimized" if feasible else "infeasible",
            feasible=feasible,
            fallback=False,
            reason=reason,
            commands=commands,
            phase_samples=phase,
            phase_speed_samples=np.r_[phase_speed, phase_speed[-1]],
            phase_acceleration_samples=np.r_[phase_acceleration, phase_acceleration[-1], phase_acceleration[-1]],
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            solve_ms=solve_ms,
            metrics={
                "sample_span_sec": (len(commands) - 1) / self.config.action_hz,
                "boundary_duration_sec": len(commands) / self.config.action_hz,
                "phase_distortion_rms_steps": phase_rms,
                "max_path_error": path_error,
                "velocity_ratio": velocity_ratio,
                "acceleration_ratio": acceleration_ratio,
                "jerk_ratio": jerk_ratio,
                "terminal_phase_speed": float(phase_speed[-1]),
                "solver_iterations": int(stats.get("iter_count", -1)),
            },
        )

    def _fallback(
        self,
        commands: np.ndarray,
        phase: np.ndarray,
        arm_columns: Sequence[int],
        started: float,
        reason: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> PhaseOptimizerResult:
        velocity, acceleration, jerk = _finite_differences(
            commands[:, arm_columns], self.config.action_hz
        )
        phase_speed = np.diff(phase) * self.config.action_hz
        phase_acceleration = np.diff(phase_speed) * self.config.action_hz
        metrics = {
            "sample_span_sec": (len(commands) - 1) / self.config.action_hz,
            "boundary_duration_sec": len(commands) / self.config.action_hz,
        }
        if extra:
            metrics.update(extra)
        return PhaseOptimizerResult(
            status="fallback_original",
            feasible=False,
            fallback=True,
            reason=reason,
            commands=commands,
            phase_samples=phase,
            phase_speed_samples=np.r_[phase_speed, phase_speed[-1]],
            phase_acceleration_samples=np.r_[phase_acceleration, phase_acceleration[-1], phase_acceleration[-1]],
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            solve_ms=(time.perf_counter() - started) * 1000.0,
            metrics=metrics,
        )
