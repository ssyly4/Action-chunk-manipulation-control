from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import toppra.constraint as constraint
from toppra.algorithm import TOPPRAsd

from .reference_path import ProjectionResult, ReferencePath, RestrictedPath


@dataclass(frozen=True)
class RetimeConfig:
    action_hz: float
    max_velocity: np.ndarray
    max_acceleration: np.ndarray
    arm_columns: tuple[int, ...] | None = None
    gridpoints_per_step: int = 8
    duration_tolerance_sec: float = 2e-4
    constraint_tolerance: float = 1e-3
    max_spline_path_deviation: float = np.deg2rad(0.25)
    max_phase_distortion_rms: float | None = None
    solver_wrapper: str = "seidel"

    def __post_init__(self) -> None:
        velocity = np.asarray(self.max_velocity, dtype=np.float64)
        acceleration = np.asarray(self.max_acceleration, dtype=np.float64)
        if self.action_hz <= 0 or self.gridpoints_per_step < 2:
            raise ValueError("action_hz and gridpoints_per_step must be positive")
        if velocity.ndim != 1 or acceleration.shape != velocity.shape:
            raise ValueError("velocity and acceleration limits must be equal-length vectors")
        if np.any(velocity <= 0) or np.any(acceleration <= 0):
            raise ValueError("kinematic limits must be positive")
        object.__setattr__(self, "max_velocity", velocity)
        object.__setattr__(self, "max_acceleration", acceleration)


@dataclass(frozen=True)
class RetimeResult:
    status: str
    feasible: bool
    fallback: bool
    reason: str | None
    commands: np.ndarray
    phase_samples: np.ndarray
    raw_phase_samples: np.ndarray
    phase_speed_samples: np.ndarray
    phase_acceleration_samples: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RtcRetimeResult:
    retiming: RetimeResult
    projection: ProjectionResult
    consumed_steps: int
    remaining_ticks: int
    request_boundary_tick: int
    replacement_start_tick: int
    replacement_end_tick: int


def _finite_differences(commands: np.ndarray, hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.diff(commands, axis=0) * hz
    acceleration = np.diff(velocity, axis=0) * hz
    jerk = np.diff(acceleration, axis=0) * hz
    return velocity, acceleration, jerk


def _phase_at_times(
    grid: np.ndarray, speeds: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = np.diff(grid)
    denom = speeds[:-1] + speeds[1:]
    if np.any(denom <= 1e-12):
        raise ValueError("TOPPRA returned a zero-speed interior segment")
    segment_dt = 2.0 * ds / denom
    cumulative = np.concatenate(([0.0], np.cumsum(segment_dt)))
    phase = np.empty_like(query)
    phase_speed = np.empty_like(query)
    phase_acceleration = np.empty_like(query)
    for i, t in enumerate(query):
        if t >= cumulative[-1] - 1e-10:
            phase[i] = grid[-1]
            phase_speed[i] = speeds[-1]
            phase_acceleration[i] = 0.0
            continue
        index = min(int(np.searchsorted(cumulative, t, side="right") - 1), len(ds) - 1)
        tau = max(0.0, t - cumulative[index])
        accel = (speeds[index + 1] ** 2 - speeds[index] ** 2) / (2.0 * ds[index])
        phase[i] = grid[index] + speeds[index] * tau + 0.5 * accel * tau * tau
        phase_speed[i] = max(0.0, speeds[index] + accel * tau)
        phase_acceleration[i] = accel
    phase[0] = grid[0]
    phase[-1] = grid[-1]
    return phase, phase_speed, phase_acceleration


class FixedPathRetimer:
    """Retimes q_ref(s); it never creates a current-state-to-endpoint path."""

    def __init__(self, config: RetimeConfig) -> None:
        self.config = config

    def _arm_columns(self, width: int) -> tuple[int, ...]:
        columns = self.config.arm_columns or tuple(range(width))
        if len(columns) != len(self.config.max_velocity):
            raise ValueError("arm_columns count must equal the limit vector length")
        if len(set(columns)) != len(columns) or min(columns) < 0 or max(columns) >= width:
            raise ValueError("arm_columns are invalid for the action width")
        return columns

    def build_reference(self, actions: np.ndarray) -> tuple[ReferencePath, tuple[int, ...]]:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("actions must be a finite HxD matrix")
        columns = self._arm_columns(values.shape[1])
        return ReferencePath(values[:, columns]), columns

    def project_state(
        self,
        actions: np.ndarray,
        measured_arm: np.ndarray,
        *,
        lower_phase: float,
        upper_phase: float,
        weights: np.ndarray | None = None,
    ) -> ProjectionResult:
        reference, _ = self.build_reference(actions)
        return reference.project(
            measured_arm,
            lower=lower_phase,
            upper=upper_phase,
            weights=weights,
        )

    def retime(
        self,
        actions: np.ndarray,
        *,
        start_phase: float = 0.0,
        output_ticks: int | None = None,
        start_phase_speed: float | None = None,
        end_phase_speed: float | None = None,
        fallback_start_index: int = 0,
    ) -> RetimeResult:
        values = np.asarray(actions, dtype=np.float64)
        reference, arm_columns = self.build_reference(values)
        output_ticks = len(values) if output_ticks is None else int(output_ticks)
        if output_ticks < 2 or not 0.0 <= start_phase < len(values) - 1:
            raise ValueError("retiming requires at least two ticks and a non-terminal start phase")

        # H rows occupy H wall-clock slots, but row H-1 is emitted at (H-1)/f.
        # The remaining 1/f interval is an endpoint hold before the RTC boundary.
        sample_span = (output_ticks - 1) / self.config.action_hz
        boundary_duration = output_ticks / self.config.action_hz
        end_phase = float(len(values) - 1)
        # TOPPRA boundary path speeds are explicit inputs.  Zero is the only
        # assumption that does not invent cross-chunk derivative continuity.
        # An eventual RTC integration must pass speeds estimated from the
        # previous accepted q_ref(s), rather than silently using H Hz here.
        sd_start = 0.0 if start_phase_speed is None else float(start_phase_speed)
        sd_end = 0.0 if end_phase_speed is None else float(end_phase_speed)
        fallback = self._fallback_commands(values, output_ticks, fallback_start_index)
        # Raw RTC rows are indexed by elapsed wall-clock slots. Projection is
        # allowed to move the retimed phase, but must not rewrite that baseline.
        raw_phase = np.linspace(fallback_start_index, end_phase, output_ticks)

        spline_deviation = reference.spline_deviation_from_polyline()
        if spline_deviation > self.config.max_spline_path_deviation:
            return self._fallback_result(
                fallback,
                raw_phase,
                arm_columns,
                boundary_duration,
                f"spline path deviation {spline_deviation:.6g} exceeds limit",
                spline_deviation=spline_deviation,
            )

        restricted = RestrictedPath(reference.path, start_phase, end_phase)
        grid_count = max(3, int(np.ceil((end_phase - start_phase) * self.config.gridpoints_per_step)) + 1)
        grid = np.linspace(start_phase, end_phase, grid_count)
        velocity_bounds = np.column_stack((-self.config.max_velocity, self.config.max_velocity))
        acceleration_bounds = np.column_stack((-self.config.max_acceleration, self.config.max_acceleration))
        algorithm = TOPPRAsd(
            [
                constraint.JointVelocityConstraint(velocity_bounds),
                constraint.JointAccelerationConstraint(acceleration_bounds),
            ],
            restricted,
            gridpoints=grid,
            solver_wrapper=self.config.solver_wrapper,
        )
        algorithm.set_desired_duration(sample_span)
        try:
            _, phase_speeds, _ = algorithm.compute_parameterization(sd_start, sd_end)
            if phase_speeds is None:
                raise ValueError("TOPPRAsd found no controllable parameterization")
            if not np.isfinite(phase_speeds).all():
                raise ValueError("TOPPRAsd returned a non-finite parameterization")
            segment_duration = 2.0 * np.diff(grid) / (phase_speeds[:-1] + phase_speeds[1:])
            actual_duration = float(np.sum(segment_duration))
            if abs(actual_duration - sample_span) > self.config.duration_tolerance_sec:
                raise ValueError(
                    f"TOPPRAsd returned {actual_duration:.6f}s for requested {sample_span:.6f}s"
                )
            tick_times = np.arange(output_ticks, dtype=np.float64) / self.config.action_hz
            phase, sampled_phase_speed, sampled_phase_acceleration = _phase_at_times(
                grid, phase_speeds, tick_times
            )
            arm_commands = reference.evaluate(phase)
            commands = self._compose_commands(
                values,
                arm_phase=phase,
                non_arm_phase=raw_phase,
                arm_commands=arm_commands,
                arm_columns=arm_columns,
            )
            result = self._evaluate(
                commands,
                phase,
                raw_phase,
                arm_columns,
                boundary_duration,
                spline_deviation=spline_deviation,
                actual_duration=actual_duration,
                phase_speed=sampled_phase_speed,
                phase_acceleration=sampled_phase_acceleration,
                raw_arm=reference.linear_reference(raw_phase),
            )
            if not result.feasible:
                return self._fallback_result(
                    fallback,
                    raw_phase,
                    arm_columns,
                    boundary_duration,
                    result.reason or "discrete constraint verification failed",
                    spline_deviation=spline_deviation,
                )
            return result
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            return self._fallback_result(
                fallback,
                raw_phase,
                arm_columns,
                boundary_duration,
                f"{type(exc).__name__}: {exc}",
                spline_deviation=spline_deviation,
            )

    def retime_rtc_replacement(
        self,
        actions: np.ndarray,
        measured_arm: np.ndarray,
        *,
        consumed_steps: int,
        emitted_at_request: int,
        projection_backtrack_steps: float = 1.0,
        projection_forward_steps: float = 2.0,
        projection_weights: np.ndarray | None = None,
    ) -> RtcRetimeResult:
        values = np.asarray(actions, dtype=np.float64)
        if not 0 <= consumed_steps < len(values) - 1:
            raise ValueError("consumed_steps must leave at least two RTC slots")
        lower = max(0.0, consumed_steps - projection_backtrack_steps)
        upper = min(len(values) - 1.0, consumed_steps + projection_forward_steps)
        projection = self.project_state(
            values,
            measured_arm,
            lower_phase=lower,
            upper_phase=upper,
            weights=projection_weights,
        )
        remaining = len(values) - consumed_steps
        retiming = self.retime(
            values,
            start_phase=projection.phase,
            output_ticks=remaining,
            fallback_start_index=consumed_steps,
        )
        boundary = emitted_at_request + len(values)
        start_tick = emitted_at_request + consumed_steps
        return RtcRetimeResult(
            retiming=retiming,
            projection=projection,
            consumed_steps=consumed_steps,
            remaining_ticks=remaining,
            request_boundary_tick=boundary,
            replacement_start_tick=start_tick,
            replacement_end_tick=start_tick + remaining,
        )

    def _compose_commands(
        self,
        actions: np.ndarray,
        *,
        arm_phase: np.ndarray,
        non_arm_phase: np.ndarray,
        arm_commands: np.ndarray,
        arm_columns: Sequence[int],
    ) -> np.ndarray:
        if len(arm_phase) != len(non_arm_phase):
            raise ValueError("arm and non-arm phases must have equal lengths")
        result = np.empty((len(arm_phase), actions.shape[1]), dtype=np.float64)
        all_columns = set(range(actions.shape[1]))
        arm_set = set(arm_columns)
        result[:, arm_columns] = arm_commands
        for column in sorted(all_columns - arm_set):
            # Grippers and other non-arm action channels are events on the
            # original 30 Hz policy timeline. Arm retiming must not move them.
            result[:, column] = np.interp(
                non_arm_phase, np.arange(len(actions)), actions[:, column]
            )
        return result

    @staticmethod
    def _fallback_commands(actions: np.ndarray, output_ticks: int, start_index: int) -> np.ndarray:
        tail = actions[start_index:].copy()
        if len(tail) == output_ticks:
            return tail
        phase = np.linspace(start_index, len(actions) - 1, output_ticks)
        result = np.empty((output_ticks, actions.shape[1]), dtype=np.float64)
        for column in range(actions.shape[1]):
            result[:, column] = np.interp(phase, np.arange(len(actions)), actions[:, column])
        return result

    def _evaluate(
        self,
        commands: np.ndarray,
        phase: np.ndarray,
        raw_phase: np.ndarray,
        arm_columns: Sequence[int],
        boundary_duration: float,
        **extra: Any,
    ) -> RetimeResult:
        arm = commands[:, arm_columns]
        velocity, acceleration, jerk = _finite_differences(arm, self.config.action_hz)
        velocity_ratio = float(np.max(np.abs(velocity) / self.config.max_velocity))
        acceleration_ratio = float(np.max(np.abs(acceleration) / self.config.max_acceleration))
        phase_rms = float(np.sqrt(np.mean(np.square(phase - raw_phase))))
        feasible = (
            velocity_ratio <= 1.0 + self.config.constraint_tolerance
            and acceleration_ratio <= 1.0 + self.config.constraint_tolerance
            and np.all(np.diff(phase) >= -1e-9)
            and (
                self.config.max_phase_distortion_rms is None
                or phase_rms <= self.config.max_phase_distortion_rms
            )
        )
        reason = (
            None
            if feasible
            else (
                "30 Hz sampled trajectory violates configured contract: "
                f"velocity_ratio={velocity_ratio:.6f} "
                f"acceleration_ratio={acceleration_ratio:.6f} "
                f"phase_rms={phase_rms:.6f}"
            )
        )
        metrics = {
            "boundary_duration_sec": boundary_duration,
            "sample_span_sec": (len(commands) - 1) / self.config.action_hz,
            "phase_distortion_rms_steps": phase_rms,
            "max_temporal_joint_deviation_rad": float(
                np.max(np.abs(commands[:, arm_columns] - np.asarray(extra.pop("raw_arm", commands[:, arm_columns]))))
            ),
            "max_velocity_rad_s": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
            "max_acceleration_rad_s2": float(np.max(np.abs(acceleration))) if acceleration.size else 0.0,
            "max_jerk_rad_s3": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
            "max_velocity_ratio": velocity_ratio,
            "max_acceleration_ratio": acceleration_ratio,
            **extra,
        }
        return RetimeResult(
            status="retimed" if feasible else "infeasible",
            feasible=feasible,
            fallback=False,
            reason=reason,
            commands=commands,
            phase_samples=phase,
            raw_phase_samples=raw_phase,
            phase_speed_samples=np.asarray(extra["phase_speed"], dtype=np.float64),
            phase_acceleration_samples=np.asarray(
                extra["phase_acceleration"], dtype=np.float64
            ),
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            metrics=metrics,
        )

    def _fallback_result(
        self,
        commands: np.ndarray,
        phase: np.ndarray,
        arm_columns: Sequence[int],
        boundary_duration: float,
        reason: str,
        **extra: Any,
    ) -> RetimeResult:
        arm = commands[:, arm_columns]
        velocity, acceleration, jerk = _finite_differences(arm, self.config.action_hz)
        phase_speed = np.gradient(phase) * self.config.action_hz
        phase_acceleration = np.gradient(phase_speed) * self.config.action_hz
        metrics = {
            "boundary_duration_sec": boundary_duration,
            "sample_span_sec": (len(commands) - 1) / self.config.action_hz,
            "max_velocity_rad_s": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
            "max_acceleration_rad_s2": float(np.max(np.abs(acceleration))) if acceleration.size else 0.0,
            "max_jerk_rad_s3": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
            "fallback_velocity_ratio": float(np.max(np.abs(velocity) / self.config.max_velocity)),
            "fallback_acceleration_ratio": float(np.max(np.abs(acceleration) / self.config.max_acceleration)),
            **extra,
        }
        return RetimeResult(
            status="fallback_original",
            feasible=False,
            fallback=True,
            reason=reason,
            commands=commands,
            phase_samples=phase.copy(),
            raw_phase_samples=phase.copy(),
            phase_speed_samples=phase_speed,
            phase_acceleration_samples=phase_acceleration,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            metrics=metrics,
        )
