from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CASADI_ROOT = PROJECT_ROOT / "casadi_fixed_horizon_retimer"
TOPPRA_ROOT = PROJECT_ROOT / "toppra_fixed_horizon_retimer"
for path in reversed(
    (
        ROOT / "vendor",
        ROOT,
        CASADI_ROOT / "vendor",
        CASADI_ROOT,
        CASADI_ROOT / "runtime",
        TOPPRA_ROOT / "vendor",
        TOPPRA_ROOT,
        TOPPRA_ROOT / "runtime",
    )
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from casadi_rtc_queue import RecedingCasadiRtcQueue
from fixed_phase_optimizer import CasadiPhaseOptimizer
from fixed_path_retimer import RecedingPlan, RetimeResult
from fixed_path_retimer.reference_path import ReferencePath
from receding_toppra_queue import (
    ARM_COLUMNS,
    ActionGainResult,
    ReadyPlan,
    scale_actions_to_envelope,
)
from waypoint_smoother import OsqpWaypointSmoother, WaypointSmootherConfig


@dataclass
class DualCandidateReadyPlan(ReadyPlan):
    recovery_plan: RecedingPlan | None = None
    recovery_action_gain: ActionGainResult | None = None
    recovery_retime_ms: float | None = None
    recovery_skip_steps: int | None = None
    recovery_prediction_tick: int | None = None
    recovery_prediction_source: str | None = None
    recovery_selected: bool = False


@dataclass(frozen=True)
class PredictedTakeoverBoundary:
    wall_tick: int
    action_index: int
    position: np.ndarray
    velocity: np.ndarray
    source: str


class RecedingOsqpCasadiRtcQueue(RecedingCasadiRtcQueue):
    """Raw actions -> bounded OSQP smoothing -> CasADi phase retiming.

    Action Gain is retained only as a recovery step when the bounded QP cannot
    make the unscaled policy chunk feasible.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.osqp_trust_region = np.deg2rad(
            float(os.environ.get("NERO_OSQP_TRUST_REGION_DEG", "0.3"))
        )
        self.osqp_enforce_boundary = (
            os.environ.get("NERO_OSQP_ENFORCE_BOUNDARY", "0") == "1"
        )
        self.osqp_fast_path = os.environ.get("NERO_OSQP_FAST_PATH", "0") == "1"
        self.speculative_recovery = (
            os.environ.get("NERO_OSQP_SPECULATIVE_RECOVERY", "1") == "1"
        )
        self._recovery_casadi_optimizer = CasadiPhaseOptimizer(
            self._casadi_optimizer.config
        )
        self._dual_optimize_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="osqp-dual-candidate"
        )
        self._predicted_recovery_boundaries: dict[
            int, PredictedTakeoverBoundary
        ] = {}
        self._waypoint_smoother = OsqpWaypointSmoother(
            WaypointSmootherConfig(
                action_hz=self.action_hz,
                arm_columns=ARM_COLUMNS,
                trust_region=np.full(14, self.osqp_trust_region),
                max_velocity=np.full(14, self.max_velocity),
                max_acceleration=np.full(14, self.max_acceleration),
                max_jerk=np.full(14, self.max_jerk),
                tracking_weight=float(
                    os.environ.get("NERO_OSQP_TRACKING_WEIGHT", "20")
                ),
                velocity_tracking_weight=float(
                    os.environ.get("NERO_OSQP_VELOCITY_WEIGHT", "0.25")
                ),
                acceleration_weight=float(
                    os.environ.get("NERO_OSQP_ACCELERATION_WEIGHT", "0.10")
                ),
                jerk_weight=float(os.environ.get("NERO_OSQP_JERK_WEIGHT", "1.0")),
                terminal_velocity_weight=float(
                    os.environ.get("NERO_OSQP_TERMINAL_VELOCITY_WEIGHT", "1.0")
                ),
                solver_time_limit_sec=float(
                    os.environ.get("NERO_OSQP_MAX_SOLVE_SEC", "0.02")
                ),
            )
        )
        print(
            "[OSQP+CASADI] bounded waypoint smoother: "
            f"trust={np.rad2deg(self.osqp_trust_region):.3f}deg "
            f"boundary={'q/v' if self.osqp_enforce_boundary else 'handoff-owned'} "
            f"fast_path={self.osqp_fast_path} "
            f"speculative_recovery={self.speculative_recovery}"
        )

    def load(self, actions: np.ndarray, *, skip_steps: int = 0) -> None:
        generation = self._generation
        self._refresh_command_state()
        prediction = self._predict_takeover_boundary(
            action_count=len(actions),
            skip_steps=skip_steps,
        )
        if prediction is not None:
            self._predicted_recovery_boundaries[generation] = prediction
        try:
            super().load(actions, skip_steps=skip_steps)
        except BaseException:
            self._predicted_recovery_boundaries.pop(generation, None)
            raise

    def _predict_takeover_boundary(
        self,
        *,
        action_count: int,
        skip_steps: int,
    ) -> PredictedTakeoverBoundary | None:
        active = self._active
        if active is None or action_count < 4:
            return None

        active_takeover_tick = self._active_takeover_tick
        if active_takeover_tick is None:
            active_takeover_tick = self._emitted
        commit_tick = active_takeover_tick + self.minimum_commit_ticks
        reserve_margin = (
            self.emergency_reserve_ticks
            if self.allow_reserve_follower_handoff
            else self.replan_reserve_ticks
        )
        reserve_tick = active.optimization_end_wall_tick - reserve_margin
        predicted_tick = max(self._emitted, min(commit_tick, reserve_tick))
        active_local = predicted_tick - active.start_wall_tick
        if not 0 <= active_local < len(active.retiming.commands):
            return None

        raw_start_tick = self._emitted - skip_steps
        action_index = predicted_tick - raw_start_tick
        # OSQP needs at least four future rows. If the predicted boundary is
        # already in the chunk tail, a separately retimed recovery trajectory
        # cannot be constructed without changing the RTC horizon.
        if not skip_steps <= action_index <= action_count - 4:
            return None

        if self._command_position is None:
            state = active.motion_state_at(active_local)
            position = state.q.copy()
            velocity = state.qd.copy()
            if self._blend is not None:
                position += self._blend.evaluate(predicted_tick)
                velocity += self._blend.evaluate_velocity(predicted_tick)
            source = "predicted_active_future"
        else:
            position, velocity = self._rollout_follower_command(predicted_tick)
            source = "predicted_follower_rollout"
        return PredictedTakeoverBoundary(
            wall_tick=predicted_tick,
            action_index=action_index,
            position=position,
            velocity=velocity,
            source=source,
        )

    def _rollout_follower_command(
        self, predicted_tick: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._active is not None
        assert self._command_position is not None
        dt = 1.0 / self.action_hz
        position = self._command_position.copy()
        velocity = self._command_velocity.copy()
        acceleration = self._command_acceleration.copy()
        position_gain = float(
            os.environ.get("NERO_CASADI_STREAMING_POSITION_GAIN_S", "4.0")
        )
        for wall_tick in range(self._emitted, predicted_tick):
            local = wall_tick - self._active.start_wall_tick
            if not 0 <= local < len(self._active.retiming.commands):
                break
            target = self._active.motion_state_at(local)
            target_position = target.q.copy()
            target_velocity = target.qd.copy()
            if self._blend is not None:
                target_position += self._blend.evaluate(wall_tick)
                target_velocity += self._blend.evaluate_velocity(wall_tick)
            desired_velocity = np.clip(
                target_velocity + position_gain * (target_position - position),
                -self.max_velocity,
                self.max_velocity,
            )
            desired_acceleration = np.clip(
                (desired_velocity - velocity) / dt,
                -self.max_acceleration,
                self.max_acceleration,
            )
            acceleration_delta = np.clip(
                desired_acceleration - acceleration,
                -self.max_jerk * dt,
                self.max_jerk * dt,
            )
            acceleration += acceleration_delta
            velocity = np.clip(
                velocity + acceleration * dt,
                -self.max_velocity,
                self.max_velocity,
            )
            position += velocity * dt
        return position, velocity

    def _identity_gain_result(
        self,
        values: np.ndarray,
        *,
        anchor: np.ndarray,
        initial_velocity: np.ndarray,
        skip_steps: int,
    ):
        # Reuse the established envelope accounting without changing a single
        # policy waypoint. This keeps downstream diagnostics/API compatibility.
        return scale_actions_to_envelope(
            values,
            anchor=anchor,
            initial_velocity=initial_velocity,
            action_hz=self.action_hz,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
            minimum_gain=1.0,
            gain_step=1.0,
            previous_gain=1.0,
            maximum_gain_rise=1.0,
            start_index=skip_steps,
            anchor_start_waypoint=False,
        )

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
        predicted_boundary = self._predicted_recovery_boundaries.pop(
            generation, None
        )
        boundary_position = anchor if self.osqp_enforce_boundary else None
        boundary_velocity = initial_velocity if self.osqp_enforce_boundary else None
        primary_gain = self._identity_gain_result(
            values,
            anchor=anchor,
            initial_velocity=initial_velocity,
            skip_steps=skip_steps,
        )
        primary_smoothed = self._waypoint_smoother.smooth(
            values,
            start_index=skip_steps,
            boundary_position=boundary_position,
            boundary_velocity=boundary_velocity,
        )
        raw_primary_smoothed = primary_smoothed
        recovery_osqp_ms = 0.0
        action_gain_fallback = False
        if not primary_smoothed.feasible:
            action_gain_fallback = True
            primary_gain = scale_actions_to_envelope(
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
            recovered_primary = self._waypoint_smoother.smooth(
                primary_gain.actions,
                start_index=skip_steps,
                boundary_position=boundary_position,
                boundary_velocity=boundary_velocity,
            )
            recovery_osqp_ms = recovered_primary.solve_ms
            if not recovered_primary.feasible:
                raise RuntimeError(
                    "OSQP rejected both the raw and Action-Gain recovery chunk: "
                    f"raw={primary_smoothed.reason!r} "
                    f"recovery={recovered_primary.reason!r}"
                )
            primary_smoothed = recovered_primary

        recovery_gain = None
        recovery_smoothed = None
        recovery_skip_steps = skip_steps
        recovery_start_tick = loaded_at_tick
        recovery_anchor = anchor
        recovery_velocity = initial_velocity
        recovery_prediction_source = "request_boundary"
        if predicted_boundary is not None:
            recovery_skip_steps = predicted_boundary.action_index
            recovery_start_tick = predicted_boundary.wall_tick
            recovery_anchor = predicted_boundary.position
            recovery_velocity = predicted_boundary.velocity
            recovery_prediction_source = predicted_boundary.source
        if (
            self.speculative_recovery
            and not self.osqp_fast_path
            and not action_gain_fallback
            and not (generation == 0 and loaded_at_tick == 0)
        ):
            candidate_gain = scale_actions_to_envelope(
                values,
                anchor=recovery_anchor,
                initial_velocity=recovery_velocity,
                action_hz=self.action_hz,
                max_velocity=self.max_velocity,
                max_acceleration=self.max_acceleration,
                minimum_gain=self.minimum_action_gain,
                gain_step=self.action_gain_step,
                previous_gain=previous_gain,
                maximum_gain_rise=self.maximum_action_gain_rise,
                start_index=recovery_skip_steps,
                anchor_start_waypoint=boundary_path_mode == "hard_anchor",
            )
            if candidate_gain.gain < 1.0 - 1e-9:
                candidate_smoothed = self._waypoint_smoother.smooth(
                    candidate_gain.actions,
                    start_index=recovery_skip_steps,
                    boundary_position=None,
                    boundary_velocity=None,
                )
                if candidate_smoothed.feasible:
                    recovery_gain = candidate_gain
                    recovery_smoothed = candidate_smoothed

        primary_actions = primary_smoothed.commands
        primary_reference = ReferencePath(primary_actions[:, ARM_COLUMNS])
        primary_speed = self._inherited_phase_speed(
            primary_reference,
            float(skip_steps),
            initial_velocity,
            use_curvature=boundary_path_mode == "path_curvature",
        )

        casadi_primary_ms = 0.0
        casadi_recovery_ms = 0.0
        if self.osqp_fast_path:
            retiming = self._fixed_timeline_result(
                primary_smoothed,
                skip_steps=skip_steps,
                primary_smoothed=raw_primary_smoothed,
                action_gain_fallback=action_gain_fallback,
            )
            recovery_plan = None
            recovery_retime_ms = None
        elif generation == 0 and loaded_at_tick == 0:
            raw_phase = self._raw_phase(primary_actions, skip_steps)
            retiming = self._initial_fallback(
                primary_actions, skip_steps, raw_phase, primary_gain
            )
            metrics = dict(retiming.metrics)
            metrics.update(
                self._osqp_metrics(
                    primary_smoothed,
                    primary_smoothed=primary_smoothed,
                    action_gain_fallback=action_gain_fallback,
                )
            )
            metrics["fallback_velocity_ratio"] = primary_smoothed.metrics[
                "smoothed_velocity_ratio"
            ]
            metrics["fallback_acceleration_ratio"] = primary_smoothed.metrics[
                "smoothed_acceleration_ratio"
            ]
            retiming = RetimeResult(
                status=retiming.status,
                feasible=retiming.feasible,
                fallback=retiming.fallback,
                reason=retiming.reason,
                commands=retiming.commands,
                phase_samples=retiming.phase_samples,
                raw_phase_samples=retiming.raw_phase_samples,
                phase_speed_samples=retiming.phase_speed_samples,
                phase_acceleration_samples=retiming.phase_acceleration_samples,
                velocity=retiming.velocity,
                acceleration=retiming.acceleration,
                jerk=retiming.jerk,
                metrics=metrics,
            )
            recovery_plan = None
            recovery_retime_ms = None
        else:
            primary_future = self._dual_optimize_pool.submit(
                self._casadi_optimizer.optimize,
                primary_actions,
                start_phase=float(skip_steps),
                output_ticks=len(primary_actions) - skip_steps,
                start_phase_speed=primary_speed,
                fallback_start_index=skip_steps,
            )
            recovery_future = None
            recovery_reference = None
            recovery_speed = None
            if recovery_gain is not None and recovery_smoothed is not None:
                recovery_actions = recovery_smoothed.commands
                recovery_reference = ReferencePath(
                    recovery_actions[:, ARM_COLUMNS]
                )
                recovery_speed = self._inherited_phase_speed(
                    recovery_reference,
                    float(recovery_skip_steps),
                    recovery_velocity,
                    use_curvature=boundary_path_mode == "path_curvature",
                )
                recovery_future = self._dual_optimize_pool.submit(
                    self._recovery_casadi_optimizer.optimize,
                    recovery_actions,
                    start_phase=float(recovery_skip_steps),
                    output_ticks=len(recovery_actions) - recovery_skip_steps,
                    start_phase_speed=recovery_speed,
                    fallback_start_index=recovery_skip_steps,
                )
            primary_optimized = primary_future.result()
            casadi_primary_ms = primary_optimized.solve_ms
            retiming = self._retiming_result(
                primary_optimized,
                primary_smoothed,
                skip_steps=skip_steps,
                status_prefix="osqp",
                primary_smoothed=primary_smoothed,
                action_gain_fallback=action_gain_fallback,
            )
            recovery_plan = None
            recovery_retime_ms = None
            if recovery_future is not None:
                recovery_optimized = recovery_future.result()
                casadi_recovery_ms = recovery_optimized.solve_ms
                recovery_retiming = self._retiming_result(
                    recovery_optimized,
                    recovery_smoothed,
                    skip_steps=recovery_skip_steps,
                    status_prefix="osqp_gain_recovery",
                    primary_smoothed=primary_smoothed,
                    action_gain_fallback=True,
                )
                recovery_plan = RecedingPlan(
                    start_wall_tick=recovery_start_tick,
                    actions=recovery_smoothed.commands.copy(),
                    reference=recovery_reference,
                    retiming=recovery_retiming,
                    requested_start_phase_speed=recovery_speed,
                    requested_end_phase_speed=float(
                        recovery_retiming.phase_speed_samples[-1]
                    ),
                )
                recovery_retime_ms = (
                    recovery_smoothed.solve_ms + recovery_optimized.solve_ms
                )

        recovery_text = "none"
        if recovery_gain is not None:
            recovery_text = f"{recovery_gain.gain:.3f}"
            if predicted_boundary is not None:
                recovery_text += (
                    f"@tick{recovery_start_tick}/action{recovery_skip_steps}"
                )

        total_retime_ms = (time.perf_counter() - started) * 1000.0
        retime_breakdown = {
            "fast_path": self.osqp_fast_path,
            "fast_path_used": retiming.status == "osqp_fixed_timeline",
            "osqp_primary_ms": raw_primary_smoothed.solve_ms,
            "osqp_recovery_ms": recovery_osqp_ms,
            "casadi_primary_ms": casadi_primary_ms,
            "casadi_recovery_ms": casadi_recovery_ms,
            "casadi_solver_cache_hit": retiming.metrics.get("solver_cache_hit"),
            "casadi_solver_build_ms": retiming.metrics.get("solver_build_ms"),
            "casadi_solver_call_ms": retiming.metrics.get("solver_call_ms"),
            "casadi_solver_iterations": retiming.metrics.get("solver_iterations"),
            "total_retime_ms": total_retime_ms,
        }
        print(
            f"[OSQP+CASADI] generation={generation} "
            f"osqp={primary_smoothed.status} {primary_smoothed.solve_ms:.2f}ms "
            f"gain={'fallback:' + format(primary_gain.gain, '.3f') if action_gain_fallback else 'bypassed'} "
            f"recovery={recovery_text} "
            f"deviation={np.rad2deg(primary_smoothed.metrics['max_waypoint_deviation_rad']):.3f}deg "
            f"jerk={primary_smoothed.metrics['raw_jerk_ratio']:.2f}x->"
            f"{primary_smoothed.metrics['smoothed_jerk_ratio']:.2f}x "
            f"retime={retiming.status} total={total_retime_ms:.1f}ms"
        )
        candidate = RecedingPlan(
            start_wall_tick=loaded_at_tick,
            actions=primary_actions.copy(),
            reference=primary_reference,
            retiming=retiming,
            requested_start_phase_speed=primary_speed,
            requested_end_phase_speed=float(retiming.phase_speed_samples[-1]),
        )
        return DualCandidateReadyPlan(
            plan=candidate,
            raw_actions=values.copy(),
            generation=generation,
            raw_start_tick=raw_start_tick,
            loaded_at_tick=loaded_at_tick,
            skip_steps=skip_steps,
            retime_ms=total_retime_ms,
            ready_started_at=started,
            action_gain=primary_gain,
            boundary_state_source=boundary_state_source,
            boundary_path_mode=boundary_path_mode,
            recovery_plan=recovery_plan,
            recovery_action_gain=recovery_gain,
            recovery_retime_ms=recovery_retime_ms,
            recovery_skip_steps=(
                recovery_skip_steps if recovery_plan is not None else None
            ),
            recovery_prediction_tick=(
                recovery_start_tick if recovery_plan is not None else None
            ),
            recovery_prediction_source=(
                recovery_prediction_source if recovery_plan is not None else None
            ),
            retime_breakdown=retime_breakdown,
        )

    def _raw_phase(self, actions: np.ndarray, skip_steps: int) -> np.ndarray:
        return np.linspace(
            float(skip_steps),
            len(actions) - 1.0,
            len(actions) - skip_steps,
        )

    def _osqp_metrics(
        self,
        smoothed,
        *,
        primary_smoothed,
        action_gain_fallback: bool,
    ) -> dict[str, object]:
        metrics = {f"osqp_{key}": value for key, value in smoothed.metrics.items()}
        metrics.update(
            {
                "osqp_status": smoothed.status,
                "osqp_feasible": smoothed.feasible,
                "osqp_fallback": smoothed.fallback,
                "osqp_reason": smoothed.reason,
                "osqp_solve_ms": smoothed.solve_ms,
                "osqp_iterations": smoothed.iterations,
                "osqp_action_gain_fallback": action_gain_fallback,
                "osqp_primary_status": primary_smoothed.status,
                "osqp_primary_reason": primary_smoothed.reason,
            }
        )
        return metrics

    def _fixed_timeline_result(
        self,
        smoothed,
        *,
        skip_steps: int,
        primary_smoothed,
        action_gain_fallback: bool,
    ) -> RetimeResult:
        """Use an already feasible OSQP path without nonlinear phase retiming."""
        commands = smoothed.commands[skip_steps:].copy()
        arm = commands[:, ARM_COLUMNS]
        velocity = np.diff(arm, axis=0) * self.action_hz
        acceleration = np.diff(velocity, axis=0) * self.action_hz
        jerk = np.diff(acceleration, axis=0) * self.action_hz
        phase = self._raw_phase(smoothed.commands, skip_steps)
        phase_speed = np.full(len(phase), self.action_hz, dtype=np.float64)
        phase_acceleration = np.zeros(len(phase), dtype=np.float64)
        metrics = self._osqp_metrics(
            smoothed,
            primary_smoothed=primary_smoothed,
            action_gain_fallback=action_gain_fallback,
        )
        metrics.update(
            {
                "sample_span_sec": (len(commands) - 1) / self.action_hz,
                "boundary_duration_sec": len(commands) / self.action_hz,
                "phase_distortion_rms_steps": 0.0,
                "solver_cache_hit": None,
                "solver_build_ms": 0.0,
                "solver_call_ms": 0.0,
                "solver_iterations": 0,
                "casadi_skipped": True,
                "casadi_skip_reason": (
                    "OSQP fixed-rate result satisfies v/a/jerk limits"
                ),
            }
        )
        return RetimeResult(
            status="osqp_fixed_timeline",
            feasible=True,
            fallback=False,
            reason=None,
            commands=commands,
            phase_samples=phase,
            raw_phase_samples=phase.copy(),
            phase_speed_samples=phase_speed,
            phase_acceleration_samples=phase_acceleration,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            metrics=metrics,
        )

    def _retiming_result(
        self,
        optimized,
        smoothed,
        *,
        skip_steps: int,
        status_prefix: str,
        primary_smoothed,
        action_gain_fallback: bool,
    ) -> RetimeResult:
        metrics = dict(optimized.metrics)
        metrics.update(
            self._osqp_metrics(
                smoothed,
                primary_smoothed=primary_smoothed,
                action_gain_fallback=action_gain_fallback,
            )
        )
        if optimized.fallback:
            metrics["fallback_velocity_ratio"] = smoothed.metrics[
                "smoothed_velocity_ratio"
            ]
            metrics["fallback_acceleration_ratio"] = smoothed.metrics[
                "smoothed_acceleration_ratio"
            ]
        return RetimeResult(
            status=(
                f"{status_prefix}_casadi_optimized"
                if optimized.feasible
                else f"{status_prefix}_then_fallback_original"
            ),
            feasible=optimized.feasible,
            fallback=optimized.fallback,
            reason=optimized.reason,
            commands=optimized.commands,
            phase_samples=optimized.phase_samples,
            raw_phase_samples=self._raw_phase(smoothed.commands, skip_steps),
            phase_speed_samples=optimized.phase_speed_samples,
            phase_acceleration_samples=optimized.phase_acceleration_samples,
            velocity=optimized.velocity,
            acceleration=optimized.acceleration,
            jerk=optimized.jerk,
            metrics=metrics,
        )

    def _takeover_ready_plan(self) -> None:
        ready = self._ready
        recovery_evaluation_due = True
        if self._active is not None and self.minimum_commit_ticks > 0:
            takeover_tick = self._active_takeover_tick
            if takeover_tick is None:
                takeover_tick = self._emitted
            committed_ticks = self._emitted - takeover_tick
            reserve_ticks = max(
                0, self._active.optimization_end_wall_tick - self._emitted
            )
            commit_bypass_reserve = (
                self.emergency_reserve_ticks
                if self.allow_reserve_follower_handoff
                else self.replan_reserve_ticks
            )
            recovery_evaluation_due = not (
                committed_ticks < self.minimum_commit_ticks
                and reserve_ticks > commit_bypass_reserve
            )
        if (
            recovery_evaluation_due
            and isinstance(ready, DualCandidateReadyPlan)
            and not ready.recovery_selected
            and ready.recovery_plan is not None
            and ready.recovery_action_gain is not None
        ):
            self._refresh_command_state()
            boundary_position, boundary_velocity, _ = self._boundary_state()
            primary_local = self._emitted - ready.plan.start_wall_tick
            recovery_local = self._emitted - ready.recovery_plan.start_wall_tick
            if (
                boundary_position is not None
                and 0 <= primary_local < len(ready.plan.retiming.commands)
                and 0 <= recovery_local < len(ready.recovery_plan.retiming.commands)
            ):
                primary_state = ready.plan.motion_state_at(primary_local)
                recovery_state = ready.recovery_plan.motion_state_at(recovery_local)
                primary_q = float(
                    np.max(np.abs(boundary_position - primary_state.q))
                )
                primary_v = float(
                    np.max(np.abs(boundary_velocity - primary_state.qd))
                )
                recovery_q = float(
                    np.max(np.abs(boundary_position - recovery_state.q))
                )
                recovery_v = float(
                    np.max(np.abs(boundary_velocity - recovery_state.qd))
                )
                primary_hard = (
                    primary_q > self.hard_position_error
                    or primary_v > self.hard_velocity_error
                )
                recovery_safe = (
                    recovery_q <= self.hard_position_error
                    and recovery_v <= self.hard_velocity_error
                )
                primary_blendable = self._bounded_handoff_possible(
                    ready.plan,
                    local=primary_local,
                    boundary_position=boundary_position,
                    boundary_velocity=boundary_velocity,
                )
                recovery_blendable = self._bounded_handoff_possible(
                    ready.recovery_plan,
                    local=recovery_local,
                    boundary_position=boundary_position,
                    boundary_velocity=boundary_velocity,
                )
                recovery_reason = (
                    "hard_mismatch" if primary_hard else "primary_unblendable"
                )
                if (
                    (primary_hard or not primary_blendable)
                    and recovery_safe
                    and recovery_blendable
                ):
                    print(
                        "[OSQP+CASADI] "
                        f"generation={ready.generation} handoff_candidate=recovery "
                        f"reason={recovery_reason} "
                        f"gain={ready.recovery_action_gain.gain:.3f} "
                        f"prediction={ready.recovery_prediction_source}@"
                        f"{ready.recovery_prediction_tick} "
                        f"q={np.rad2deg(primary_q):.2f}->{np.rad2deg(recovery_q):.2f}deg "
                        f"v={np.rad2deg(primary_v):.1f}->{np.rad2deg(recovery_v):.1f}deg/s"
                    )
                    ready.plan = ready.recovery_plan
                    ready.action_gain = ready.recovery_action_gain
                    if ready.recovery_skip_steps is not None:
                        ready.skip_steps = ready.recovery_skip_steps
                    ready.recovery_selected = True
        super()._takeover_ready_plan()

    def _bounded_handoff_possible(
        self,
        plan: RecedingPlan,
        *,
        local: int,
        boundary_position: np.ndarray,
        boundary_velocity: np.ndarray,
    ) -> bool:
        state = plan.motion_state_at(local)
        position_error = float(np.max(np.abs(boundary_position - state.q)))
        velocity_error = float(np.max(np.abs(boundary_velocity - state.qd)))
        if (
            position_error <= self.direct_position_error
            and velocity_error <= self.direct_velocity_error
        ):
            return True
        if (
            position_error > self.hard_position_error
            or velocity_error > self.hard_velocity_error
        ):
            return False
        severity = max(
            position_error / self.hard_position_error,
            velocity_error / self.hard_velocity_error,
        )
        requested_ticks = int(
            np.clip(
                np.ceil(
                    self.min_blend_ticks
                    + severity * (self.max_blend_ticks - self.min_blend_ticks)
                ),
                self.min_blend_ticks,
                self.max_blend_ticks,
            )
        )
        return self._select_bounded_blend(
            plan,
            local=local,
            position_offset=boundary_position - state.q,
            velocity_offset=boundary_velocity - state.qd,
            minimum_ticks=requested_ticks,
            match_velocity=plan.retiming.feasible,
        ) is not None

    def close(self) -> None:
        self._dual_optimize_pool.shutdown(wait=True, cancel_futures=True)
        super().close()

    def __del__(self) -> None:
        pool = getattr(self, "_dual_optimize_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        super().__del__()
