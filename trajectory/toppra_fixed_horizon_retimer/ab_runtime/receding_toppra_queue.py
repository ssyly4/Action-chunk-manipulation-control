"""Experimental receding TOPPRAsd queue for process-local production A/B.

The queue keeps the production RTC public interface, but all implementation
and dependencies remain outside nero_ws.  It never accesses CAN directly.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT.parents[1]
for path in (ROOT / "vendor", ROOT, CONTROL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-toppra-retimer")

import numpy as np

from fixed_path_retimer import (
    FixedPathRetimer,
    RecedingConfig,
    RecedingFixedPathRetimer,
    RecedingPlan,
    RetimeConfig,
)
from nero_vla.bimanual_chunk_executor import (
    ACTION_DIM,
    BimanualProgressSample,
    BimanualRtcRequest,
    arm_matrix,
)
from follower_state_bridge import DESIRED_VELOCITY_REGISTRY, FOLLOWER_STATE_REGISTRY


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


@dataclass(frozen=True)
class ActionGainResult:
    actions: np.ndarray
    gain: float
    envelope_feasible: bool
    raw_velocity_ratio: float
    raw_acceleration_ratio: float
    scaled_velocity_ratio: float
    scaled_acceleration_ratio: float


def _discrete_envelope_ratios(
    arm_actions: np.ndarray,
    *,
    anchor: np.ndarray,
    initial_velocity: np.ndarray,
    action_hz: float,
    max_velocity: float,
    max_acceleration: float,
) -> tuple[float, float]:
    positions = np.vstack((anchor, arm_actions))
    velocities = np.diff(positions, axis=0) * action_hz
    velocity_history = np.vstack((initial_velocity, velocities))
    accelerations = np.diff(velocity_history, axis=0) * action_hz
    return (
        float(np.max(np.abs(velocity_history)) / max_velocity),
        float(np.max(np.abs(accelerations)) / max_acceleration),
    )


def scale_actions_to_envelope(
    actions: np.ndarray,
    *,
    anchor: np.ndarray,
    initial_velocity: np.ndarray,
    action_hz: float,
    max_velocity: float,
    max_acceleration: float,
    minimum_gain: float,
    gain_step: float,
    previous_gain: float,
    maximum_gain_rise: float,
    start_index: int = 0,
    anchor_start_waypoint: bool = True,
) -> ActionGainResult:
    values = np.asarray(actions, dtype=np.float64)
    arms = arm_matrix(values)
    anchor = np.asarray(anchor, dtype=np.float64)
    initial_velocity = np.asarray(initial_velocity, dtype=np.float64)
    if anchor.shape != (14,) or initial_velocity.shape != (14,):
        raise ValueError("action gain anchor and velocity must have shape (14,)")
    if not 0.0 < minimum_gain <= 1.0 or not 0.0 < gain_step <= 1.0:
        raise ValueError("action gain bounds must be in (0, 1]")
    if maximum_gain_rise <= 0.0:
        raise ValueError("maximum gain rise must be positive")
    if not 0 <= start_index < len(values) - 1:
        raise ValueError("action gain start_index must retain at least two rows")

    raw_velocity_ratio, raw_acceleration_ratio = _discrete_envelope_ratios(
        arms[start_index:],
        anchor=anchor,
        initial_velocity=initial_velocity,
        action_hz=action_hz,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )
    candidates = list(np.arange(1.0, minimum_gain - 1e-9, -gain_step))
    if not candidates or abs(candidates[-1] - minimum_gain) > 1e-9:
        candidates.append(minimum_gain)
    feasible: list[tuple[float, float, float]] = []
    for gain in candidates:
        scaled_arms = anchor + float(gain) * (arms[start_index:] - anchor)
        ratios = _discrete_envelope_ratios(
            scaled_arms,
            anchor=anchor,
            initial_velocity=initial_velocity,
            action_hz=action_hz,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        if max(ratios) <= 1.0 + 1e-9:
            feasible.append((float(gain), *ratios))

    rise_cap = min(1.0, float(previous_gain) + maximum_gain_rise)
    permitted = [item for item in feasible if item[0] <= rise_cap + 1e-9]
    if permitted:
        gain, velocity_ratio, acceleration_ratio = max(permitted, key=lambda item: item[0])
        envelope_feasible = True
    elif feasible:
        gain, velocity_ratio, acceleration_ratio = min(feasible, key=lambda item: item[0])
        envelope_feasible = True
    else:
        gain = float(minimum_gain)
        minimum_arms = anchor + gain * (arms[start_index:] - anchor)
        velocity_ratio, acceleration_ratio = _discrete_envelope_ratios(
            minimum_arms,
            anchor=anchor,
            initial_velocity=initial_velocity,
            action_hz=action_hz,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        envelope_feasible = False

    scaled = values.copy()
    scaled[:, ARM_COLUMNS] = anchor + gain * (arms - anchor)
    if anchor_start_waypoint:
        # A mode forces position continuity into q_ref(s). B mode keeps the
        # policy path intact and leaves the bounded handoff to bridge the gap.
        scaled[start_index, ARM_COLUMNS] = anchor
    return ActionGainResult(
        actions=scaled,
        gain=gain,
        envelope_feasible=envelope_feasible,
        raw_velocity_ratio=raw_velocity_ratio,
        raw_acceleration_ratio=raw_acceleration_ratio,
        scaled_velocity_ratio=velocity_ratio,
        scaled_acceleration_ratio=acceleration_ratio,
    )


def _arm_state_at_action(
    actions: np.ndarray, *, index: int, action_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return one action-row position and a local finite-difference velocity."""
    arms = arm_matrix(actions)
    if not 0 <= index < len(arms):
        raise ValueError("action index lies outside the chunk")
    if index == 0:
        velocity = (arms[1] - arms[0]) * action_hz
    elif index == len(arms) - 1:
        velocity = (arms[-1] - arms[-2]) * action_hz
    else:
        velocity = (arms[index + 1] - arms[index - 1]) * (0.5 * action_hz)
    return arms[index].copy(), velocity


def _mismatch_summary(
    boundary_position: np.ndarray,
    boundary_velocity: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
) -> dict[str, float]:
    """Summarize a bimanual q/v mismatch in degrees for runtime diagnosis."""
    q_error = np.abs(boundary_position - reference_position)
    velocity_error = np.abs(boundary_velocity - reference_velocity)
    return {
        "q_max_deg": float(np.rad2deg(np.max(q_error))),
        "v_max_deg_s": float(np.rad2deg(np.max(velocity_error))),
        "left_q_max_deg": float(np.rad2deg(np.max(q_error[:7]))),
        "right_q_max_deg": float(np.rad2deg(np.max(q_error[7:]))),
        "left_v_max_deg_s": float(np.rad2deg(np.max(velocity_error[:7]))),
        "right_v_max_deg_s": float(np.rad2deg(np.max(velocity_error[7:]))),
    }


@dataclass(frozen=True)
class RuntimeDecision:
    generation: int
    raw_start_tick: int
    loaded_at_tick: int
    skip_steps: int
    retime_status: str
    retime_reason: str | None
    retime_ms: float
    fixed_horizon_feasible: bool
    fallback_velocity_ratio: float | None
    fallback_acceleration_ratio: float | None
    action_gain: float
    action_gain_feasible: bool
    raw_action_velocity_ratio: float
    raw_action_acceleration_ratio: float
    scaled_action_velocity_ratio: float
    scaled_action_acceleration_ratio: float
    projection_phase: float | None
    projection_error_deg: float | None
    handoff_type: str
    handoff_position_error_deg: float | None
    handoff_velocity_error_deg_s: float | None
    blend_ticks: int
    blend_max_velocity_deg_s: float | None
    blend_max_acceleration_deg_s2: float | None
    blend_within_retimer_limits: bool | None
    active_reserve_ticks: int
    ready_to_takeover_ms: float
    evaluation_attempt: int
    terminal_decision: bool
    boundary_state_source: str
    boundary_velocity_max_deg_s: float
    boundary_acceleration_max_deg_s2: float
    requested_start_phase_speed: float
    boundary_path_mode: str
    action: str
    committed_ticks: int | None = None
    minimum_commit_ticks: int = 0
    layer_mismatch: dict[str, dict[str, float]] | None = None


@dataclass
class ReadyPlan:
    plan: RecedingPlan
    raw_actions: np.ndarray
    generation: int
    raw_start_tick: int
    loaded_at_tick: int
    skip_steps: int
    retime_ms: float
    ready_started_at: float
    action_gain: ActionGainResult
    boundary_state_source: str
    boundary_path_mode: str
    evaluation_attempts: int = 0
    rejection_logged: bool = False
    commit_wait_logged: bool = False


@dataclass(frozen=True)
class QuinticHandoffCorrection:
    takeover_wall_tick: int
    ticks: int
    action_hz: float
    position_offset: np.ndarray
    velocity_offset: np.ndarray

    def _normalized_time(self, wall_tick: int) -> tuple[float, float] | None:
        local = wall_tick - self.takeover_wall_tick
        if local < 0 or local >= self.ticks:
            return None
        if self.ticks == 1:
            return 1.0, 0.0
        return local / (self.ticks - 1), (self.ticks - 1) / self.action_hz

    def evaluate(self, wall_tick: int) -> np.ndarray:
        normalized = self._normalized_time(wall_tick)
        if normalized is None:
            return np.zeros_like(self.position_offset)
        x, duration = normalized
        position_basis = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
        velocity_basis = x - 6.0 * x**3 + 8.0 * x**4 - 3.0 * x**5
        return (
            position_basis * self.position_offset
            + duration * velocity_basis * self.velocity_offset
        )

    def evaluate_velocity(self, wall_tick: int) -> np.ndarray:
        normalized = self._normalized_time(wall_tick)
        if normalized is None:
            return np.zeros_like(self.velocity_offset)
        x, duration = normalized
        if duration == 0.0:
            return np.zeros_like(self.velocity_offset)
        position_basis_dx = -30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4
        velocity_basis_dx = 1.0 - 18.0 * x**2 + 32.0 * x**3 - 15.0 * x**4
        return (
            position_basis_dx / duration * self.position_offset
            + velocity_basis_dx * self.velocity_offset
        )


class RecedingToppraRtcQueue:
    """Process-local fixed-time TOPPRA receding-horizon RTC queue."""

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        handoff_decay_steps: int = 0,
        max_handoff_error_rad: float | None = None,
    ) -> None:
        del handoff_decay_steps
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        self.action_hz = float(action_hz)
        self.max_handoff_error_rad = max_handoff_error_rad
        self.boundary_path_mode = os.environ.get(
            "NERO_TOPPRA_BOUNDARY_PATH_MODE", "hard_anchor"
        )
        if self.boundary_path_mode not in {"hard_anchor", "path_curvature"}:
            raise ValueError(
                "NERO_TOPPRA_BOUNDARY_PATH_MODE must be hard_anchor or path_curvature"
            )
        self.curvature_speed_margin = float(
            os.environ.get("NERO_TOPPRA_CURVATURE_SPEED_MARGIN", "0.80")
        )
        if not 0.0 < self.curvature_speed_margin <= 1.0:
            raise ValueError("TOPPRA curvature speed margin must be in (0, 1]")
        self.max_velocity = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_MAX_VELOCITY_DEG_S", "25"))
        )
        self.max_acceleration = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_MAX_ACCELERATION_DEG_S2", "220"))
        )
        self.feedback_velocity_tau_sec = float(
            os.environ.get("NERO_TOPPRA_FEEDBACK_VELOCITY_TAU_SEC", "0.08")
        )
        if self.feedback_velocity_tau_sec <= 0.0:
            raise ValueError("feedback velocity filter time constant must be positive")
        self.minimum_action_gain = float(
            os.environ.get("NERO_TOPPRA_MIN_ACTION_GAIN", "0.50")
        )
        self.action_gain_step = float(
            os.environ.get("NERO_TOPPRA_ACTION_GAIN_STEP", "0.025")
        )
        self.maximum_action_gain_rise = float(
            os.environ.get("NERO_TOPPRA_MAX_ACTION_GAIN_RISE", "0.10")
        )
        if not 0.0 < self.minimum_action_gain <= 1.0:
            raise ValueError("minimum action gain must be in (0, 1]")
        self.direct_position_error = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_DIRECT_POSITION_DEG", "0.20"))
        )
        self.direct_velocity_error = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_DIRECT_VELOCITY_DEG_S", "6"))
        )
        self.hard_position_error = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_HARD_POSITION_DEG", "2.5"))
        )
        self.hard_velocity_error = np.deg2rad(
            float(os.environ.get("NERO_TOPPRA_HARD_VELOCITY_DEG_S", "50"))
        )
        self.min_blend_ticks = int(os.environ.get("NERO_TOPPRA_MIN_BLEND_TICKS", "2"))
        self.max_blend_ticks = int(os.environ.get("NERO_TOPPRA_MAX_BLEND_TICKS", "12"))
        if not 2 <= self.min_blend_ticks <= self.max_blend_ticks:
            raise ValueError("TOPPRA blend ticks must satisfy 2 <= min <= max")
        self.minimum_commit_ticks = int(
            os.environ.get("NERO_TOPPRA_MIN_COMMIT_TICKS", "0")
        )
        self.pending_request_block_steps = int(
            os.environ.get("NERO_TOPPRA_PENDING_REQUEST_BLOCK_STEPS", "23")
        )
        self.emergency_reserve_ticks = int(
            os.environ.get("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", "0")
        )
        self.allow_reserve_follower_handoff = (
            os.environ.get("NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF", "1")
            == "1"
        )
        self.replan_reserve_ticks = int(
            os.environ.get(
                "NERO_TOPPRA_REPLAN_RESERVE_TICKS",
                str(self.emergency_reserve_ticks),
            )
        )
        if self.minimum_commit_ticks < 0:
            raise ValueError("TOPPRA minimum commit ticks must be non-negative")
        if self.pending_request_block_steps < 1:
            raise ValueError("TOPPRA pending request block steps must be positive")
        if self.emergency_reserve_ticks < 0:
            raise ValueError("TOPPRA emergency reserve ticks must be non-negative")
        if self.replan_reserve_ticks < self.emergency_reserve_ticks:
            raise ValueError(
                "TOPPRA replan reserve must be no smaller than emergency reserve"
            )
        self._engine = RecedingFixedPathRetimer(
            FixedPathRetimer(
                RetimeConfig(
                    action_hz=self.action_hz,
                    max_velocity=np.full(14, self.max_velocity),
                    max_acceleration=np.full(14, self.max_acceleration),
                    arm_columns=ARM_COLUMNS,
                )
            ),
            RecedingConfig(),
        )
        self._active: RecedingPlan | None = None
        self._ready: ReadyPlan | None = None
        self._retime_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="toppra-retime")
        self._retime_future: Future[ReadyPlan] | None = None
        self._latest_generation = -1
        self._blend: QuinticHandoffCorrection | None = None
        self._hold_action: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._last_sample_at: float | None = None
        self._next_action_at: float | None = None
        self._emitted = 0
        self._generation = 0
        self._active_takeover_tick: int | None = None
        self._last_action_gain = 1.0
        self._last_handoff_error_rad = 0.0
        self._feedback_position: np.ndarray | None = None
        self._feedback_velocity = np.zeros(14, dtype=np.float64)
        self._feedback_acceleration = np.zeros(14, dtype=np.float64)
        self._feedback_at: float | None = None
        self._command_position: np.ndarray | None = None
        self._command_velocity = np.zeros(14, dtype=np.float64)
        self._command_acceleration = np.zeros(14, dtype=np.float64)
        self._command_at: float | None = None

        default_log = ROOT / "outputs" / f"receding_runtime_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_path = Path(os.environ.get("NERO_TOPPRA_RUNTIME_LOG", str(default_log)))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        print(
            "[TOPPRA AB] receding queue: "
            f"rate={self.action_hz:.1f}Hz "
            f"retimer_limits={np.rad2deg(self.max_velocity):.1f}deg/s/"
            f"{np.rad2deg(self.max_acceleration):.1f}deg/s2 "
            f"gain={self.minimum_action_gain:.2f}-1.00 "
            f"boundary_mode={self.boundary_path_mode} "
            f"min_commit={self.minimum_commit_ticks}tick "
            f"emergency_reserve={self.emergency_reserve_ticks}tick "
            f"replan_reserve={self.replan_reserve_ticks}tick "
            f"reserve_follower={self.allow_reserve_follower_handoff} "
            f"rise<={self.maximum_action_gain_rise:.2f}/chunk log={self.log_path}"
        )

    @property
    def emitted_steps(self) -> int:
        return self._emitted

    @property
    def last_handoff_error_rad(self) -> float:
        return self._last_handoff_error_rad

    @property
    def remaining_steps(self) -> int:
        plan = self._active
        if plan is None:
            remaining = 2 if self._hold_action is not None else 0
        else:
            remaining = max(0, plan.optimization_end_wall_tick - self._emitted)
        candidate_pending = self._ready is not None or self._retime_future is not None
        if self.minimum_commit_ticks > 0 and candidate_pending:
            # The production loop uses remaining_steps to decide when to request
            # another policy chunk. While one candidate is being retimed or is
            # waiting for its commit boundary, report a full queue so a third
            # chunk cannot replace the pending candidate.
            return max(remaining, self.pending_request_block_steps)
        return remaining

    def handoff_errors_rad(self, actions: np.ndarray, *, skip_steps: int = 0) -> np.ndarray:
        # The production stream asks this before the candidate has gone through
        # TOPPRA. In this isolated A/B runtime the raw chunk is diagnostic only;
        # the single authoritative guard runs after retiming against live q/qd.
        self._raw_handoff_errors_rad(actions, skip_steps=skip_steps)
        return np.zeros(2, dtype=np.float64)

    def _raw_handoff_errors_rad(
        self, actions: np.ndarray, *, skip_steps: int = 0
    ) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        arms = arm_matrix(values)
        if not 0 <= skip_steps < len(values):
            raise ValueError("RTC skip_steps must retain at least one action")
        if self._last_action is None:
            return np.zeros(2, dtype=np.float64)
        previous = np.concatenate((self._last_action[:7], self._last_action[8:15]))
        error = np.abs(arms[skip_steps] - previous)
        return np.asarray((np.max(error[:7]), np.max(error[7:])), dtype=np.float64)

    def handoff_error_rad(self, actions: np.ndarray, *, skip_steps: int = 0) -> float:
        return float(np.max(self.handoff_errors_rad(actions, skip_steps=skip_steps)))

    def load(self, actions: np.ndarray, *, skip_steps: int = 0) -> None:
        values = np.asarray(actions, dtype=np.float64)
        arm_matrix(values)
        if not 0 <= skip_steps < len(values) - 1:
            raise ValueError("receding RTC load must retain at least two actions")
        self._last_handoff_error_rad = float(
            np.max(self._raw_handoff_errors_rad(values, skip_steps=skip_steps))
        )
        raw_start_tick = self._emitted - skip_steps
        generation = self._generation
        self._generation += 1
        self._latest_generation = generation
        started = time.perf_counter()
        self._refresh_command_state()
        boundary_position, boundary_velocity, boundary_source = self._boundary_state()
        anchor = (
            arm_matrix(values)[0].copy()
            if boundary_position is None
            else boundary_position.copy()
        )
        initial_velocity = (
            np.zeros(14, dtype=np.float64)
            if boundary_position is None
            else boundary_velocity.copy()
        )
        job = dict(
            values=values.copy(),
            generation=generation,
            raw_start_tick=raw_start_tick,
            loaded_at_tick=self._emitted,
            skip_steps=skip_steps,
            started=started,
            anchor=anchor,
            initial_velocity=initial_velocity,
            previous_gain=self._last_action_gain,
            boundary_state_source=boundary_source,
            boundary_path_mode=self.boundary_path_mode,
        )
        if self._active is None and self._feedback_position is None:
            # Initial planning happens before the live timeline exists. Keeping
            # this synchronous avoids an empty queue that would trigger an
            # unnecessary second policy request during startup.
            self._ready = self._retime_job(**job)
            return
        if self._retime_future is not None and not self._retime_future.done():
            raise RuntimeError("new policy chunk arrived before prior TOPPRA retiming completed")
        self._retime_future = self._retime_pool.submit(self._retime_job, **job)

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
        candidate = self._engine.plan(
            gain.actions,
            start_wall_tick=loaded_at_tick,
            consumed_steps=skip_steps,
            start_arm_velocity=initial_velocity,
            start_speed_curvature_margin=(
                self.curvature_speed_margin
                if boundary_path_mode == "path_curvature"
                else None
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ReadyPlan(
            plan=candidate,
            raw_actions=values.copy(),
            generation=generation,
            raw_start_tick=raw_start_tick,
            loaded_at_tick=loaded_at_tick,
            skip_steps=skip_steps,
            retime_ms=elapsed_ms,
            ready_started_at=started,
            action_gain=gain,
            boundary_state_source=boundary_state_source,
            boundary_path_mode=boundary_path_mode,
        )

    def _refresh_command_state(self) -> None:
        state = FOLLOWER_STATE_REGISTRY.combined()
        if state is None:
            return
        self._command_position = state.position.copy()
        self._command_velocity = state.velocity.copy()
        self._command_acceleration = state.acceleration.copy()
        self._command_at = state.updated_at

    def _boundary_state(self) -> tuple[np.ndarray | None, np.ndarray, str]:
        if (
            self._command_position is not None
            and self._command_at is not None
            and self._feedback_at is not None
            and abs(self._feedback_at - self._command_at) <= 0.2
        ):
            return self._command_position, self._command_velocity, "follower_command"
        return self._feedback_position, self._feedback_velocity, "measured_feedback"

    def _poll_retime(self) -> None:
        future = self._retime_future
        if future is None or not future.done():
            return
        self._retime_future = None
        ready = future.result()
        if ready.generation == self._latest_generation:
            self._ready = ready

    def reanchor_hold(self, action: np.ndarray, *, now: float) -> None:
        anchor = np.asarray(action, dtype=np.float64)
        if anchor.shape != (ACTION_DIM,) or not np.isfinite(anchor).all():
            raise ValueError(f"RTC reanchor action must have shape ({ACTION_DIM},)")
        self._active = None
        self._ready = None
        self._blend = None
        self._active_takeover_tick = None
        self._latest_generation = self._generation
        self._generation += 1
        self._hold_action = anchor.copy()
        self._last_action = anchor.copy()
        self._last_sample_at = None
        self._next_action_at = float(now)
        self._last_handoff_error_rad = 0.0
        self._last_action_gain = 1.0

    def make_request(
        self,
        *,
        now: float,
        execution_horizon: int,
        predicted_delay_steps: int,
    ) -> BimanualRtcRequest:
        if execution_horizon < 2 or predicted_delay_steps < 0:
            raise ValueError("invalid RTC horizon or delay")
        rows = [self._command_at_tick(self._emitted + offset) for offset in range(execution_horizon)]
        if any(row is None for row in rows):
            if self._last_action is None:
                raise RuntimeError("RTC request requires queued or emitted actions")
            rows = [self._last_action.copy() if row is None else row for row in rows]
        return BimanualRtcRequest(
            previous_actions=np.asarray(rows, dtype=np.float64),
            valid_previous_steps=execution_horizon,
            emitted_at_request=self._emitted,
            requested_at=float(now),
            predicted_delay_steps=int(predicted_delay_steps),
        )

    def consumed_since(self, request: BimanualRtcRequest) -> int:
        return max(0, self._emitted - request.emitted_at_request)

    def _observe_feedback(self, feedback: np.ndarray, now: float) -> None:
        position = np.asarray(feedback, dtype=np.float64)
        if position.shape != (14,) or not np.isfinite(position).all():
            raise ValueError("combined feedback must be finite and have shape (14,)")
        if self._feedback_position is not None and self._feedback_at is not None:
            dt = now - self._feedback_at
            if 1e-4 < dt <= 0.2:
                measured_velocity = (position - self._feedback_position) / dt
                measured_velocity = np.clip(
                    measured_velocity, -2.0 * self.max_velocity, 2.0 * self.max_velocity
                )
                alpha = 1.0 - np.exp(-dt / self.feedback_velocity_tau_sec)
                old_velocity = self._feedback_velocity.copy()
                velocity = old_velocity + alpha * (measured_velocity - old_velocity)
                measured_acceleration = (velocity - old_velocity) / dt
                self._feedback_acceleration += alpha * (
                    measured_acceleration - self._feedback_acceleration
                )
                self._feedback_velocity = velocity
        self._feedback_position = position.copy()
        self._feedback_at = float(now)

    def _takeover_ready_plan(self) -> None:
        ready = self._ready
        if ready is None or self._feedback_position is None:
            return
        if self._active is not None and self.minimum_commit_ticks > 0:
            takeover_tick = self._active_takeover_tick
            if takeover_tick is None:
                takeover_tick = self._emitted
                self._active_takeover_tick = takeover_tick
            committed_ticks = self._emitted - takeover_tick
            reserve_ticks = max(
                0, self._active.optimization_end_wall_tick - self._emitted
            )
            # The commit window prevents rapid plan churn only while the
            # active plan can still provide a real command reserve.  Holding a
            # ready candidate after that reserve is exhausted creates a
            # circular wait: the scheduler cannot execute the old plan, but
            # the minimum-commit gate also forbids evaluating the new one.
            # In fail-closed mode, evaluate by the earlier replan boundary so
            # an unblendable candidate can be discarded while useful reserve
            # remains.  The legacy follower mode keeps its emergency boundary.
            commit_bypass_reserve = (
                self.emergency_reserve_ticks
                if self.allow_reserve_follower_handoff
                else self.replan_reserve_ticks
            )
            wait_for_commit = (
                committed_ticks < self.minimum_commit_ticks
                and reserve_ticks > commit_bypass_reserve
            )
            if wait_for_commit:
                if not ready.commit_wait_logged:
                    print(
                        "[TOPPRA AB] "
                        f"generation={ready.generation} ready_waiting_minimum_commit "
                        f"committed={committed_ticks}/{self.minimum_commit_ticks}tick"
                    )
                    ready.commit_wait_logged = True
                return
            if (
                not ready.commit_wait_logged
                and committed_ticks < self.minimum_commit_ticks
            ):
                print(
                    "[TOPPRA AB] "
                    f"generation={ready.generation} commit_gate_bypassed_low_reserve "
                    f"committed={committed_ticks}/{self.minimum_commit_ticks}tick "
                    f"reserve={reserve_ticks}"
                )
                ready.commit_wait_logged = True
        self._refresh_command_state()
        boundary_position, boundary_velocity, boundary_source = self._boundary_state()
        if boundary_position is None:
            return
        candidate = ready.plan
        local = self._emitted - candidate.start_wall_tick
        if not 0 <= local < len(candidate.retiming.commands):
            self._write_expired_decision(ready)
            self._ready = None
            return
        ready.evaluation_attempts += 1
        state = candidate.motion_state_at(local)
        raw_index = min(
            len(candidate.actions) - 1,
            max(0, ready.skip_steps + local),
        )
        raw_position, raw_velocity = _arm_state_at_action(
            ready.raw_actions,
            index=raw_index,
            action_hz=self.action_hz,
        )
        gain_position, gain_velocity = _arm_state_at_action(
            ready.action_gain.actions,
            index=raw_index,
            action_hz=self.action_hz,
        )
        layer_mismatch = {
            # Same live boundary and same elapsed raw action index. This makes
            # it possible to attribute a discontinuity to raw policy output,
            # action scaling, retiming, or the final handoff separately.
            "rtc_raw": _mismatch_summary(
                boundary_position, boundary_velocity, raw_position, raw_velocity
            ),
            "action_gain": _mismatch_summary(
                boundary_position, boundary_velocity, gain_position, gain_velocity
            ),
            "casadi_retimed": _mismatch_summary(
                boundary_position, boundary_velocity, state.q, state.qd
            ),
        }
        projection = candidate.reference.project(
            boundary_position,
            lower=max(0.0, local - 1.0),
            upper=min(len(candidate.actions) - 1.0, local + 2.0),
        )
        position_error = float(np.max(np.abs(boundary_position - state.q)))
        velocity_error = float(np.max(np.abs(boundary_velocity - state.qd)))
        reserve_ticks = (
            0
            if self._active is None
            else max(0, self._active.optimization_end_wall_tick - self._emitted)
        )
        committed_ticks = (
            None
            if self._active is None or self._active_takeover_tick is None
            else self._emitted - self._active_takeover_tick
        )
        handoff_type = "reject"
        blend_ticks = 0
        blend_max_velocity = None
        blend_max_acceleration = None
        blend_within_limits = None
        action = "rejected_hard_measured_mismatch"
        accepted = False
        discarded_for_replan = False
        if (
            position_error <= self.direct_position_error
            and velocity_error <= self.direct_velocity_error
        ):
            handoff_type = "direct"
            action = (
                "initial_plan"
                if self._active is None
                else "recovered_direct_handoff"
                if ready.rejection_logged
                else "immediate_direct_handoff"
            )
            accepted = True
        elif (
            position_error <= self.hard_position_error
            and velocity_error <= self.hard_velocity_error
        ):
            severity = max(
                position_error / self.hard_position_error,
                velocity_error / self.hard_velocity_error,
            )
            small_handoff = (
                position_error <= self.direct_position_error
                and velocity_error <= self.direct_velocity_error
            )
            requested_ticks = (
                self.min_blend_ticks
                if small_handoff
                else int(
                    np.clip(
                        np.ceil(
                            self.min_blend_ticks
                            + severity
                            * (self.max_blend_ticks - self.min_blend_ticks)
                        ),
                        self.min_blend_ticks,
                        self.max_blend_ticks,
                    )
                )
            )
            blend_result = self._select_bounded_blend(
                candidate,
                local=local,
                position_offset=boundary_position - state.q,
                velocity_offset=boundary_velocity - state.qd,
                minimum_ticks=requested_ticks,
                match_velocity=candidate.retiming.feasible,
            )
            if blend_result is not None:
                blend_ticks, blend_max_velocity, blend_max_acceleration = blend_result
                blend_within_limits = True
                handoff_type = "blend"
                if candidate.retiming.feasible:
                    action = (
                        "recovered_quintic_handoff"
                        if ready.rejection_logged
                        else "immediate_quintic_handoff"
                    )
                else:
                    action = (
                        "recovered_fallback_position_handoff"
                        if ready.rejection_logged
                        else "immediate_fallback_position_handoff"
                    )
                accepted = True
            elif candidate.retiming.fallback:
                # The downstream RateLimitedJointFollower owns shaping for a
                # raw fallback path. Do not force an out-of-envelope local
                # correction merely to preserve derivative offsets.
                handoff_type = "follower"
                action = (
                    "recovered_fallback_follower_handoff"
                    if ready.rejection_logged
                    else "immediate_fallback_follower_handoff"
                )
                accepted = True
            elif (
                not self.allow_reserve_follower_handoff
                and reserve_ticks <= self.replan_reserve_ticks
            ):
                # Experimental fail-closed mode: release the pending slot while
                # the active plan still has enough reserve for another RTC
                # request. Never hide an infeasible handoff in the downstream
                # follower merely because the old horizon is nearly empty.
                action = "discarded_unblendable_for_early_replan"
                discarded_for_replan = True
            elif reserve_ticks <= self.emergency_reserve_ticks:
                handoff_type = "follower"
                action = "reserve_exhaustion_follower_handoff"
                accepted = True
            else:
                action = "deferred_unbounded_quintic_handoff"

        if (
            not accepted
            and action == "rejected_hard_measured_mismatch"
            and reserve_ticks <= self.emergency_reserve_ticks
        ):
            # A hard-unsafe candidate must never be forced through. Dropping it
            # releases remaining_steps suppression so the production RTC loop
            # requests a fresh chunk from the current state instead of freezing
            # forever on a candidate whose local phase cannot advance in hold.
            action = "discarded_hard_mismatch_for_replan"
            discarded_for_replan = True

        next_blend = None
        if accepted and handoff_type == "blend":
            next_blend = QuinticHandoffCorrection(
                takeover_wall_tick=self._emitted,
                ticks=blend_ticks,
                action_hz=self.action_hz,
                position_offset=(boundary_position - state.q).copy(),
                velocity_offset=(
                    (boundary_velocity - state.qd).copy()
                    if candidate.retiming.feasible
                    else np.zeros(14, dtype=np.float64)
                ),
            )

        handoff_position = state.q.copy()
        handoff_velocity = state.qd.copy()
        if next_blend is not None:
            handoff_position += next_blend.evaluate(self._emitted)
            handoff_velocity += next_blend.evaluate_velocity(self._emitted)
        layer_mismatch["handoff_target"] = _mismatch_summary(
            boundary_position,
            boundary_velocity,
            handoff_position,
            handoff_velocity,
        )
        layer_mismatch["meta"] = {
            "raw_action_index": float(raw_index),
            "retimed_local_index": float(local),
            "retimed_phase": float(candidate.retiming.phase_samples[local]),
            "handoff_accepted": float(accepted),
        }

        if accepted:
            self._active = candidate
            self._active_takeover_tick = self._emitted
            self._last_action_gain = ready.action_gain.gain
            self._hold_action = None
            self._blend = next_blend

        takeover_ms = (time.perf_counter() - ready.ready_started_at) * 1000.0
        decision = RuntimeDecision(
            generation=ready.generation,
            raw_start_tick=ready.raw_start_tick,
            loaded_at_tick=ready.loaded_at_tick,
            skip_steps=ready.skip_steps,
            retime_status=candidate.retiming.status,
            retime_reason=candidate.retiming.reason,
            retime_ms=ready.retime_ms,
            fixed_horizon_feasible=candidate.retiming.feasible,
            fallback_velocity_ratio=candidate.retiming.metrics.get(
                "fallback_velocity_ratio"
            ),
            fallback_acceleration_ratio=candidate.retiming.metrics.get(
                "fallback_acceleration_ratio"
            ),
            action_gain=ready.action_gain.gain,
            action_gain_feasible=ready.action_gain.envelope_feasible,
            raw_action_velocity_ratio=ready.action_gain.raw_velocity_ratio,
            raw_action_acceleration_ratio=ready.action_gain.raw_acceleration_ratio,
            scaled_action_velocity_ratio=ready.action_gain.scaled_velocity_ratio,
            scaled_action_acceleration_ratio=ready.action_gain.scaled_acceleration_ratio,
            projection_phase=projection.phase,
            projection_error_deg=float(np.rad2deg(projection.max_joint_error)),
            handoff_type=handoff_type,
            handoff_position_error_deg=float(np.rad2deg(position_error)),
            handoff_velocity_error_deg_s=float(np.rad2deg(velocity_error)),
            blend_ticks=blend_ticks,
            blend_max_velocity_deg_s=(
                None if blend_max_velocity is None else float(np.rad2deg(blend_max_velocity))
            ),
            blend_max_acceleration_deg_s2=(
                None
                if blend_max_acceleration is None
                else float(np.rad2deg(blend_max_acceleration))
            ),
            blend_within_retimer_limits=blend_within_limits,
            active_reserve_ticks=reserve_ticks,
            ready_to_takeover_ms=takeover_ms,
            evaluation_attempt=ready.evaluation_attempts,
            terminal_decision=accepted or discarded_for_replan,
            boundary_state_source=boundary_source,
            boundary_velocity_max_deg_s=float(
                np.rad2deg(np.max(np.abs(boundary_velocity)))
            ),
            boundary_acceleration_max_deg_s2=float(
                np.rad2deg(np.max(np.abs(self._command_acceleration)))
                if boundary_source == "follower_command"
                else np.rad2deg(np.max(np.abs(self._feedback_acceleration)))
            ),
            requested_start_phase_speed=float(candidate.requested_start_phase_speed),
            boundary_path_mode=ready.boundary_path_mode,
            action=action,
            committed_ticks=committed_ticks,
            minimum_commit_ticks=self.minimum_commit_ticks,
            layer_mismatch=layer_mismatch,
        )
        if accepted:
            self._ready = None
            self._write_decision(decision)
        elif discarded_for_replan:
            self._ready = None
            self._write_decision(decision)
        elif not ready.rejection_logged:
            ready.rejection_logged = True
            self._write_decision(decision)

    def _write_decision(self, decision: RuntimeDecision) -> None:
        self._log.write(json.dumps(asdict(decision), sort_keys=True) + "\n")
        self._log.flush()
        qerr = (
            "None"
            if decision.handoff_position_error_deg is None
            else f"{decision.handoff_position_error_deg:.2f}deg"
        )
        verr = (
            "None"
            if decision.handoff_velocity_error_deg_s is None
            else f"{decision.handoff_velocity_error_deg_s:.1f}deg/s"
        )
        contract = "feasible"
        if not decision.fixed_horizon_feasible:
            velocity_ratio = decision.fallback_velocity_ratio
            acceleration_ratio = decision.fallback_acceleration_ratio
            velocity_text = (
                "?" if velocity_ratio is None else f"{velocity_ratio:.2f}x"
            )
            acceleration_text = (
                "?" if acceleration_ratio is None else f"{acceleration_ratio:.2f}x"
            )
            contract = f"fallback(v={velocity_text} a={acceleration_text})"
        print(
            f"[TOPPRA AB] generation={decision.generation} {decision.action} "
            f"attempt={decision.evaluation_attempt} retime={decision.retime_status} "
            f"contract={contract} {decision.retime_ms:.1f}ms "
            f"gain={decision.action_gain:.3f} "
            f"env={decision.scaled_action_velocity_ratio:.2f}x/"
            f"{decision.scaled_action_acceleration_ratio:.2f}x "
            f"handoff={decision.handoff_type} "
            f"qerr={qerr} verr={verr} blend={decision.blend_ticks} "
            f"boundary={decision.boundary_state_source} "
            f"path={decision.boundary_path_mode} "
            f"commit={decision.committed_ticks}/{decision.minimum_commit_ticks}tick "
            f"reserve={decision.active_reserve_ticks} "
            f"takeover={decision.ready_to_takeover_ms:.1f}ms"
        )
        layers = decision.layer_mismatch
        if layers is not None:
            def text(name: str) -> str:
                value = layers[name]
                return f"{name}=q{value['q_max_deg']:.2f}deg/v{value['v_max_deg_s']:.1f}deg/s"

            print(
                "[TOPPRA AB] handoff layers "
                f"{text('rtc_raw')} {text('action_gain')} "
                f"{text('casadi_retimed')} {text('handoff_target')} "
                f"raw_index={layers['meta']['raw_action_index']:.0f} "
                f"phase={layers['meta']['retimed_phase']:.2f} "
                f"accepted={bool(layers['meta']['handoff_accepted'])}"
            )

    def _write_expired_decision(self, ready: ReadyPlan) -> None:
        reserve_ticks = (
            0
            if self._active is None
            else max(0, self._active.optimization_end_wall_tick - self._emitted)
        )
        self._write_decision(
            RuntimeDecision(
                generation=ready.generation,
                raw_start_tick=ready.raw_start_tick,
                loaded_at_tick=ready.loaded_at_tick,
                skip_steps=ready.skip_steps,
                retime_status=ready.plan.retiming.status,
                retime_reason=ready.plan.retiming.reason,
                retime_ms=ready.retime_ms,
                fixed_horizon_feasible=ready.plan.retiming.feasible,
                fallback_velocity_ratio=ready.plan.retiming.metrics.get(
                    "fallback_velocity_ratio"
                ),
                fallback_acceleration_ratio=ready.plan.retiming.metrics.get(
                    "fallback_acceleration_ratio"
                ),
                action_gain=ready.action_gain.gain,
                action_gain_feasible=ready.action_gain.envelope_feasible,
                raw_action_velocity_ratio=ready.action_gain.raw_velocity_ratio,
                raw_action_acceleration_ratio=ready.action_gain.raw_acceleration_ratio,
                scaled_action_velocity_ratio=ready.action_gain.scaled_velocity_ratio,
                scaled_action_acceleration_ratio=ready.action_gain.scaled_acceleration_ratio,
                projection_phase=None,
                projection_error_deg=None,
                handoff_type="reject",
                handoff_position_error_deg=None,
                handoff_velocity_error_deg_s=None,
                blend_ticks=0,
                blend_max_velocity_deg_s=None,
                blend_max_acceleration_deg_s2=None,
                blend_within_retimer_limits=None,
                active_reserve_ticks=reserve_ticks,
                ready_to_takeover_ms=(
                    time.perf_counter() - ready.ready_started_at
                )
                * 1000.0,
                evaluation_attempt=ready.evaluation_attempts,
                terminal_decision=True,
                boundary_state_source=ready.boundary_state_source,
                boundary_velocity_max_deg_s=0.0,
                boundary_acceleration_max_deg_s2=0.0,
                requested_start_phase_speed=float(
                    ready.plan.requested_start_phase_speed
                ),
                boundary_path_mode=ready.boundary_path_mode,
                action="candidate_expired_without_safe_handoff",
            )
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
        for ticks in range(minimum_ticks, self.max_blend_ticks + 1):
            if local + ticks > len(candidate.retiming.commands):
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
            for offset in range(ticks):
                command = candidate.retiming.commands[local + offset]
                arms = command[list(ARM_COLUMNS)].copy()
                correction = blend.evaluate(self._emitted + offset)
                arms += correction
                positions.append(arms)
                corrections.append(correction)
            checked_positions = np.asarray(
                positions if match_velocity else corrections
            )
            if len(checked_positions) < 2:
                continue
            velocities = np.diff(checked_positions, axis=0) * self.action_hz
            initial_velocity = (
                self._boundary_state()[1]
                if match_velocity
                else np.zeros(14, dtype=np.float64)
            )
            velocity_history = np.vstack((initial_velocity, velocities))
            accelerations = np.diff(velocity_history, axis=0) * self.action_hz
            max_velocity = float(np.max(np.abs(velocity_history)))
            max_acceleration = float(np.max(np.abs(accelerations)))
            if (
                max_velocity <= self.max_velocity + 1e-9
                and max_acceleration <= self.max_acceleration + 1e-9
            ):
                return ticks, max_velocity, max_acceleration
        return None

    def _command_from_plan(self, plan: RecedingPlan, wall_tick: int) -> np.ndarray | None:
        local = wall_tick - plan.start_wall_tick
        if not 0 <= local < len(plan.retiming.commands):
            return None
        command = plan.retiming.commands[local].copy()
        if plan is self._active and self._blend is not None:
            command[list(ARM_COLUMNS)] += self._blend.evaluate(wall_tick)
            if wall_tick >= self._blend.takeover_wall_tick + self._blend.ticks - 1:
                self._blend = None
        return command

    def _velocity_from_plan(self, plan: RecedingPlan, wall_tick: int) -> np.ndarray | None:
        local = wall_tick - plan.start_wall_tick
        if not 0 <= local < len(plan.retiming.commands):
            return None
        velocity = plan.motion_state_at(local).qd.copy()
        if plan is self._active and self._blend is not None:
            velocity += self._blend.evaluate_velocity(wall_tick)
        return velocity

    def _scheduled_velocity_at_tick(self, wall_tick: int) -> np.ndarray | None:
        if self._active is not None:
            velocity = self._velocity_from_plan(self._active, wall_tick)
            if velocity is not None:
                return velocity
        if self._hold_action is not None:
            return np.zeros(14, dtype=np.float64)
        return None

    def _scheduled_command_at_tick(self, wall_tick: int) -> np.ndarray | None:
        if self._active is not None:
            command = self._command_from_plan(self._active, wall_tick)
            if command is not None:
                return command
        if self._hold_action is not None:
            return self._hold_action.copy()
        return None

    def _command_at_tick(self, wall_tick: int) -> np.ndarray | None:
        scheduled = self._scheduled_command_at_tick(wall_tick)
        if scheduled is not None:
            return scheduled
        return None if self._last_action is None else self._last_action.copy()

    def sample(self, now: float, *, feedback: np.ndarray) -> BimanualProgressSample:
        self._observe_feedback(feedback, float(now))
        self._refresh_command_state()
        self._poll_retime()
        self._takeover_ready_plan()
        same_tick = self._last_sample_at is not None and now == self._last_sample_at
        action_due = self._next_action_at is None or now + 1e-9 >= self._next_action_at
        scheduled = False
        if not same_tick and action_due:
            wall_tick = self._emitted
            command = self._scheduled_command_at_tick(self._emitted)
            if command is not None:
                scheduled = True
                self._last_action = command
                DESIRED_VELOCITY_REGISTRY.publish(
                    self._scheduled_velocity_at_tick(wall_tick)
                )
                self._emitted += 1
            else:
                DESIRED_VELOCITY_REGISTRY.publish(np.zeros(14, dtype=np.float64))
            self._last_sample_at = float(now)
            self._next_action_at = (
                float(now) + 1.0 / self.action_hz
                if self._next_action_at is None
                else self._next_action_at + 1.0 / self.action_hz
            )
        if self._last_action is None:
            return BimanualProgressSample(
                None, None, None, None, "no_chunk_hold", None, None, None, None, None
            )
        status = (
            "rtc_toppra_tracking"
            if scheduled
            else "rtc_queue_hold"
            if action_due
            else "rtc_rate_hold"
        )
        return BimanualProgressSample(
            left_target=self._last_action[:7].copy(),
            right_target=self._last_action[8:15].copy(),
            left_gripper_target=float(self._last_action[7]),
            right_gripper_target=float(self._last_action[15]),
            status=status,
            phase_steps=float(self._emitted - 1),
            target_phase_steps=float(self._emitted - 1),
            projection_error_rad=None,
            source_age_sec=None,
            remaining_sec=self.remaining_steps / self.action_hz,
        )

    def close(self) -> None:
        self._retime_pool.shutdown(wait=True, cancel_futures=True)
        if not self._log.closed:
            self._log.close()

    def __del__(self) -> None:
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()
        pool = getattr(self, "_retime_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


# Keep the historical import working for existing local test commands. New code
# must use RecedingToppraRtcQueue; the runtime no longer uses rolling overlap.
RollingToppraRtcQueue = RecedingToppraRtcQueue
