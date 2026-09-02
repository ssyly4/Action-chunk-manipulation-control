from __future__ import annotations

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOPPRA_ROOT = ROOT.parent / "toppra_fixed_horizon_retimer"
TOPPRA_RUNTIME = TOPPRA_ROOT / "runtime"
for path in (ROOT / "vendor", ROOT, TOPPRA_ROOT / "vendor", TOPPRA_ROOT, TOPPRA_RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from fixed_phase_optimizer import CasadiPhaseOptimizer, PhaseOptimizerConfig
from fixed_path_retimer import RecedingPlan, RetimeResult
from fixed_path_retimer.reference_path import ReferencePath
from receding_toppra_queue import (
    ARM_COLUMNS,
    ActionGainResult,
    QuinticHandoffCorrection,
    ReadyPlan,
    RecedingToppraRtcQueue,
    scale_actions_to_envelope,
)


class RecedingCasadiRtcQueue(RecedingToppraRtcQueue):
    """CasADi phase optimization using the shared TOPPRA queue contract."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_jerk = np.deg2rad(
            float(os.environ.get("NERO_CASADI_MAX_JERK_DEG_S3", "8000"))
        )
        self.max_solve_sec = float(os.environ.get("NERO_CASADI_MAX_SOLVE_SEC", "0.20"))
        if self.max_jerk <= 0.0 or self.max_solve_sec <= 0.0:
            raise ValueError("CasADi jerk and solve-time limits must be positive")
        self._casadi_optimizer = CasadiPhaseOptimizer(
            PhaseOptimizerConfig(
                action_hz=self.action_hz,
                max_velocity=np.full(14, self.max_velocity),
                max_acceleration=np.full(14, self.max_acceleration),
                max_jerk=np.full(14, self.max_jerk),
                arm_columns=ARM_COLUMNS,
                solver_max_cpu_sec=self.max_solve_sec,
                solver_warm_start_duals=True,
            )
        )
        print(
            "[CASADI] fixed-horizon phase optimizer: "
            f"jerk={np.rad2deg(self.max_jerk):.0f}deg/s3 "
            f"solve_timeout={self.max_solve_sec * 1000:.0f}ms "
            "warm_start=primal+dual"
        )

    def _select_bounded_blend(
        self,
        candidate: RecedingPlan,
        *,
        local: int,
        position_offset: np.ndarray,
        velocity_offset: np.ndarray,
        minimum_ticks: int,
        match_velocity: bool,
    ) -> tuple[int, float, float] | None:
        effective_velocity_offset = (
            velocity_offset if match_velocity else np.zeros_like(velocity_offset)
        )
        _, boundary_velocity, boundary_source = self._boundary_state()
        if match_velocity:
            boundary_acceleration = (
                self._command_acceleration
                if boundary_source == "follower_command"
                else self._feedback_acceleration
            )
        else:
            boundary_velocity = np.zeros(14, dtype=np.float64)
            boundary_acceleration = np.zeros(14, dtype=np.float64)

        for ticks in range(minimum_ticks, self.max_blend_ticks + 1):
            # Include the first sample after correction reaches zero so the
            # blend exit is part of the acceleration and jerk contract.
            if local + ticks >= len(candidate.retiming.commands):
                break
            blend = QuinticHandoffCorrection(
                takeover_wall_tick=self._emitted,
                ticks=ticks,
                action_hz=self.action_hz,
                position_offset=position_offset,
                velocity_offset=effective_velocity_offset,
            )
            positions = []
            corrections = []
            for offset in range(ticks + 1):
                command = candidate.retiming.commands[local + offset]
                correction = blend.evaluate(self._emitted + offset)
                positions.append(command[list(ARM_COLUMNS)] + correction)
                corrections.append(correction)
            checked_positions = np.asarray(
                positions if match_velocity else corrections,
                dtype=np.float64,
            )
            velocities = np.diff(checked_positions, axis=0) * self.action_hz
            velocity_history = np.vstack((boundary_velocity, velocities))
            accelerations = np.diff(velocity_history, axis=0) * self.action_hz
            acceleration_history = np.vstack(
                (boundary_acceleration, accelerations)
            )
            jerk = np.diff(acceleration_history, axis=0) * self.action_hz
            max_velocity = float(np.max(np.abs(velocity_history)))
            max_acceleration = float(np.max(np.abs(acceleration_history)))
            max_jerk = float(np.max(np.abs(jerk)))
            if (
                max_velocity <= self.max_velocity + 1e-9
                and max_acceleration <= self.max_acceleration + 1e-9
                and max_jerk <= self.max_jerk + 1e-9
            ):
                return ticks, max_velocity, max_acceleration
        return None

    def _retime_job(
        self,
        *,
        values: np.ndarray,
        generation: int,
        raw_start_tick: int,
        loaded_at_tick: int,
        skip_steps: int,
        started: float,
        anchor: np.ndarray,
        initial_velocity: np.ndarray,
        previous_gain: float,
        boundary_state_source: str,
        boundary_path_mode: str,
    ) -> ReadyPlan:
        gain = scale_actions_to_envelope(
            values,
            anchor=anchor,
            initial_velocity=initial_velocity,
            action_hz=self.action_hz,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
            minimum_gain=self.minimum_action_gain,
            gain_step=self.action_gain_step,
            previous_gain=previous_gain,
            maximum_gain_rise=self.maximum_action_gain_rise,
            start_index=skip_steps,
            anchor_start_waypoint=boundary_path_mode == "hard_anchor",
        )
        reference = ReferencePath(gain.actions[:, ARM_COLUMNS])
        start_speed = self._inherited_phase_speed(
            reference,
            float(skip_steps),
            initial_velocity,
            use_curvature=boundary_path_mode == "path_curvature",
        )
        raw_phase = np.linspace(
            float(skip_steps), len(gain.actions) - 1.0, len(gain.actions) - skip_steps
        )
        if generation == 0 and loaded_at_tick == 0:
            # The production queue loads generation zero synchronously before
            # the first 30 Hz tick. IPOPT compilation/timeout must never delay
            # that scheduler boundary. The accepted baseline already permits
            # explicit fallback_original, so start immediately and let later
            # generations use the asynchronous worker.
            retiming = self._initial_fallback(gain.actions, skip_steps, raw_phase, gain)
        else:
            optimized = self._casadi_optimizer.optimize(
                gain.actions,
                start_phase=float(skip_steps),
                output_ticks=len(gain.actions) - skip_steps,
                start_phase_speed=start_speed,
                fallback_start_index=skip_steps,
            )
            metrics = dict(optimized.metrics)
            if optimized.fallback:
                metrics.setdefault("fallback_velocity_ratio", gain.scaled_velocity_ratio)
                metrics.setdefault("fallback_acceleration_ratio", gain.scaled_acceleration_ratio)
            retiming = RetimeResult(
                status="casadi_optimized" if optimized.feasible else "fallback_original",
                feasible=optimized.feasible,
                fallback=optimized.fallback,
                reason=optimized.reason,
                commands=optimized.commands,
                phase_samples=optimized.phase_samples,
                raw_phase_samples=raw_phase,
                phase_speed_samples=optimized.phase_speed_samples,
                phase_acceleration_samples=optimized.phase_acceleration_samples,
                velocity=optimized.velocity,
                acceleration=optimized.acceleration,
                jerk=optimized.jerk,
                metrics=metrics,
            )
        candidate = RecedingPlan(
            start_wall_tick=loaded_at_tick,
            actions=gain.actions.copy(),
            reference=reference,
            retiming=retiming,
            requested_start_phase_speed=start_speed,
            requested_end_phase_speed=float(retiming.phase_speed_samples[-1]),
        )
        return ReadyPlan(
            plan=candidate,
            raw_actions=values.copy(),
            generation=generation,
            raw_start_tick=raw_start_tick,
            loaded_at_tick=loaded_at_tick,
            skip_steps=skip_steps,
            retime_ms=(time.perf_counter() - started) * 1000.0,
            ready_started_at=started,
            action_gain=gain,
            boundary_state_source=boundary_state_source,
            boundary_path_mode=boundary_path_mode,
        )

    def _initial_fallback(
        self,
        actions: np.ndarray,
        skip_steps: int,
        phase: np.ndarray,
        gain: ActionGainResult,
    ) -> RetimeResult:
        commands = actions[skip_steps:].copy()
        arms = commands[:, ARM_COLUMNS]
        velocity = np.diff(arms, axis=0) * self.action_hz
        acceleration = np.diff(velocity, axis=0) * self.action_hz
        jerk = np.diff(acceleration, axis=0) * self.action_hz
        phase_speed = np.diff(phase) * self.action_hz
        phase_acceleration = np.diff(phase_speed) * self.action_hz
        return RetimeResult(
            status="fallback_original",
            feasible=False,
            fallback=True,
            reason="initial chunk bypasses synchronous CasADi optimization",
            commands=commands,
            phase_samples=phase.copy(),
            raw_phase_samples=phase.copy(),
            phase_speed_samples=np.r_[phase_speed, phase_speed[-1]],
            phase_acceleration_samples=np.r_[
                phase_acceleration,
                phase_acceleration[-1],
                phase_acceleration[-1],
            ],
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            metrics={
                "sample_span_sec": (len(commands) - 1) / self.action_hz,
                "boundary_duration_sec": len(commands) / self.action_hz,
                "fallback_velocity_ratio": gain.scaled_velocity_ratio,
                "fallback_acceleration_ratio": gain.scaled_acceleration_ratio,
                "initial_scheduler_safe_fallback": True,
            },
        )

    def _inherited_phase_speed(
        self,
        reference: ReferencePath,
        phase: float,
        arm_velocity: np.ndarray,
        *,
        use_curvature: bool,
    ) -> float:
        tangent = reference.evaluate(phase, 1)
        norm_squared = float(np.dot(tangent, tangent))
        if norm_squared <= 1e-12:
            return 0.0
        speed = max(0.0, float(np.dot(tangent, arm_velocity) / norm_squared))
        moving = np.abs(tangent) > 1e-10
        speed = min(
            speed,
            0.95 * float(np.min(self.max_velocity / np.abs(tangent[moving]))),
        )
        if use_curvature:
            curvature = np.abs(reference.evaluate(phase, 2))
            curved = curvature > 1e-10
            if np.any(curved):
                speed = min(
                    speed,
                    self.curvature_speed_margin
                    * float(np.min(np.sqrt(self.max_acceleration / curvature[curved]))),
                )
        return speed
