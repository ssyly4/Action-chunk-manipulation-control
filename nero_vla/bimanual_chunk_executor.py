"""Shared-phase action-chunk alignment for two seven-joint NERO arms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ARM_JOINTS = 7
BIMANUAL_JOINTS = 14
ACTION_DIM = 16


def _vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape ({size},) and contain finite values")
    return result


def arm_matrix(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or len(values) < 2:
        raise ValueError(f"actions must have shape (horizon, {ACTION_DIM})")
    if not np.isfinite(values).all():
        raise ValueError("actions must contain finite values")
    return np.concatenate((values[:, :7], values[:, 8:15]), axis=1)


def _interpolate(values: np.ndarray, phase_steps: float) -> np.ndarray:
    phase = float(np.clip(phase_steps, 0.0, len(values) - 1))
    left = min(int(np.floor(phase)), len(values) - 2)
    alpha = phase - left
    return values[left] + alpha * (values[left + 1] - values[left])


def _gripper_event_phase(
    values: np.ndarray,
    *,
    base_phase_steps: float,
    lookahead_steps: float,
    activation_delta: float,
) -> float:
    """Lead to the deepest explicit close event without advancing an idle gripper."""
    base_phase = float(np.clip(base_phase_steps, 0.0, len(values) - 1))
    if lookahead_steps <= 0 or activation_delta <= 0:
        return base_phase
    base_value = float(_interpolate(values[:, None], base_phase)[0])
    end_phase = min(float(len(values) - 1), base_phase + lookahead_steps)
    threshold = base_value - activation_delta
    first_index = int(np.ceil(base_phase))
    last_index = int(np.floor(end_phase))
    if last_index < first_index:
        return base_phase
    window = values[first_index : last_index + 1]
    relative_index = int(np.argmin(window))
    index = first_index + relative_index
    return float(index) if float(values[index]) <= threshold else base_phase


def _project_forward(
    positions: np.ndarray,
    feedback: np.ndarray,
    *,
    phase_steps: float,
    max_advance_steps: float,
) -> tuple[float, float]:
    start_phase = float(np.clip(phase_steps, 0.0, len(positions) - 1))
    end_phase = min(float(len(positions) - 1), start_phase + max_advance_steps)
    first_segment = min(int(np.floor(start_phase)), len(positions) - 2)
    last_segment = min(int(np.floor(end_phase)), len(positions) - 2)
    best_phase = start_phase
    best_error_sq = float("inf")
    for segment in range(first_segment, last_segment + 1):
        start = positions[segment]
        delta = positions[segment + 1] - start
        alpha_min = float(np.clip(start_phase - segment, 0.0, 1.0))
        alpha_max = float(np.clip(end_phase - segment, 0.0, 1.0))
        if alpha_max < alpha_min:
            continue
        denominator = float(delta @ delta)
        alpha = (
            alpha_min
            if denominator <= 1e-15
            else float(np.clip(delta @ (feedback - start) / denominator, alpha_min, alpha_max))
        )
        candidate = start + alpha * delta
        error_sq = float(np.mean(np.square(candidate - feedback)))
        if error_sq < best_error_sq:
            best_phase = float(segment + alpha)
            best_error_sq = error_sq
    return best_phase, float(np.sqrt(best_error_sq))


@dataclass(frozen=True)
class BimanualAlignment:
    offset_steps: float
    action: np.ndarray
    latency_steps: float
    max_feedback_error_rad: float
    rms_feedback_error_rad: float
    max_command_error_rad: float
    score: float
    search_max_steps: float


def align_bimanual_chunk_to_state(
    actions: np.ndarray,
    *,
    feedback: np.ndarray,
    command: np.ndarray,
    latency_steps: float,
    max_alignment_error_rad: float,
    search_margin_steps: float = 2.0,
    command_weight: float = 0.5,
    latency_weight: float = 0.05,
) -> BimanualAlignment:
    positions = arm_matrix(actions)
    live_feedback = _vector(feedback, BIMANUAL_JOINTS, "feedback")
    live_command = _vector(command, BIMANUAL_JOINTS, "command")
    if not np.isfinite(latency_steps) or latency_steps < 0:
        raise ValueError("latency_steps must be finite and non-negative")
    if max_alignment_error_rad <= 0 or search_margin_steps < 0:
        raise ValueError("alignment limits must be positive/non-negative")

    search_max = min(float(len(positions) - 2), latency_steps + search_margin_steps)
    error_scale_sq = float(max_alignment_error_rad) ** 2
    latency_scale_sq = max(1.0, latency_steps) ** 2
    best: BimanualAlignment | None = None
    segment_count = max(1, min(len(positions) - 1, int(np.ceil(search_max))))
    for segment in range(segment_count):
        start = positions[segment]
        delta = positions[segment + 1] - start
        alpha_max = float(np.clip(search_max - segment, 0.0, 1.0))
        quadratic = (
            (1.0 + command_weight) * float(delta @ delta) / error_scale_sq
            + latency_weight / latency_scale_sq
        )
        linear = (
            float(delta @ (start - live_feedback))
            + command_weight * float(delta @ (start - live_command))
        ) / error_scale_sq
        linear += latency_weight * (segment - latency_steps) / latency_scale_sq
        alpha = 0.0 if quadratic <= 1e-15 else float(np.clip(-linear / quadratic, 0.0, alpha_max))
        candidate = start + alpha * delta
        feedback_error = candidate - live_feedback
        command_error = candidate - live_command
        offset = float(segment + alpha)
        score = float(np.mean(np.square(feedback_error)) / error_scale_sq)
        score += command_weight * float(np.mean(np.square(command_error)) / error_scale_sq)
        score += latency_weight * ((offset - latency_steps) ** 2) / latency_scale_sq
        current = BimanualAlignment(
            offset_steps=offset,
            action=candidate.copy(),
            latency_steps=float(latency_steps),
            max_feedback_error_rad=float(np.max(np.abs(feedback_error))),
            rms_feedback_error_rad=float(np.sqrt(np.mean(np.square(feedback_error)))),
            max_command_error_rad=float(np.max(np.abs(command_error))),
            score=score,
            search_max_steps=search_max,
        )
        if best is None or current.score < best.score:
            best = current
    assert best is not None
    if best.max_feedback_error_rad > max_alignment_error_rad:
        raise ValueError(
            "no safely aligned bimanual action in search window: "
            f"offset={best.offset_steps:.2f} "
            f"max_error_deg={np.rad2deg(best.max_feedback_error_rad):.3f}"
        )
    return best


@dataclass(frozen=True)
class BimanualProgressSample:
    left_target: np.ndarray | None
    right_target: np.ndarray | None
    left_gripper_target: float | None
    right_gripper_target: float | None
    status: str
    phase_steps: float | None
    target_phase_steps: float | None
    projection_error_rad: float | None
    source_age_sec: float | None
    remaining_sec: float | None


@dataclass(frozen=True)
class _Chunk:
    actions: np.ndarray
    observed_at: float
    received_at: float
    stale_at: float


class BimanualFeedbackProgressActionChunk:
    """Advance one shared phase from the measured progress of both arms."""

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        arm_lead_steps: float = 1.0,
        gripper_lead_steps: float = 0.0,
        gripper_event_lookahead_steps: float = 0.0,
        gripper_event_activation_delta: float = 0.03,
        max_progress_steps_per_tick: float = 1.0,
        blend_duration_sec: float = 0.067,
        stale_after_sec: float = 0.15,
    ) -> None:
        if action_hz <= 0 or arm_lead_steps <= 0 or max_progress_steps_per_tick <= 0:
            raise ValueError("action rate, lead, and progress must be positive")
        if (
            gripper_lead_steps < 0
            or gripper_event_lookahead_steps < 0
            or gripper_event_activation_delta <= 0
            or blend_duration_sec < 0
            or stale_after_sec < 0
        ):
            raise ValueError("gripper lead, blend, and stale duration must be non-negative")
        self.action_hz = float(action_hz)
        self.arm_lead_steps = float(arm_lead_steps)
        self.gripper_lead_steps = float(gripper_lead_steps)
        self.gripper_event_lookahead_steps = float(gripper_event_lookahead_steps)
        self.gripper_event_activation_delta = float(gripper_event_activation_delta)
        self.max_progress_steps_per_tick = float(max_progress_steps_per_tick)
        self.blend_duration_sec = float(blend_duration_sec)
        self.stale_after_sec = float(stale_after_sec)
        self._chunk: _Chunk | None = None
        self._phase = 0.0
        self._transition_at: float | None = None
        self._transition_origin: np.ndarray | None = None
        self._transition_offset: np.ndarray | None = None
        self._last_target: np.ndarray | None = None

    def push(
        self,
        actions: np.ndarray,
        *,
        initial_phase_steps: float,
        observed_at: float,
        received_at: float,
    ) -> None:
        values = np.asarray(actions, dtype=np.float64)
        arm_matrix(values)
        if not 0.0 <= initial_phase_steps < len(values) - 1:
            raise ValueError("initial phase must leave at least one future action")
        remaining = (len(values) - 1 - initial_phase_steps) / self.action_hz
        self._transition_origin = (
            None if self._last_target is None else self._last_target.copy()
        )
        self._transition_offset = None
        self._transition_at = float(received_at)
        self._chunk = _Chunk(
            values.copy(),
            float(observed_at),
            float(received_at),
            float(received_at) + remaining + self.stale_after_sec,
        )
        self._phase = float(initial_phase_steps)

    def sample(self, now: float, *, feedback: np.ndarray) -> BimanualProgressSample:
        if self._chunk is None:
            return BimanualProgressSample(
                None, None, None, None, "no_chunk_hold", None, None, None, None, None
            )
        chunk = self._chunk
        source_age = float(now - chunk.observed_at)
        remaining = float(len(chunk.actions) - 1 - self._phase) / self.action_hz
        if now > chunk.stale_at:
            return BimanualProgressSample(
                None,
                None,
                None,
                None,
                "stale_chunk_hold",
                self._phase,
                None,
                None,
                source_age,
                remaining,
            )

        live = _vector(feedback, BIMANUAL_JOINTS, "feedback")
        phase, error = _project_forward(
            arm_matrix(chunk.actions),
            live,
            phase_steps=self._phase,
            max_advance_steps=self.max_progress_steps_per_tick,
        )
        self._phase = max(self._phase, phase)
        target_phase = min(float(len(chunk.actions) - 1), self._phase + self.arm_lead_steps)
        target = _interpolate(chunk.actions, target_phase)
        base_gripper_phase = min(
            float(len(chunk.actions) - 1), self._phase + self.gripper_lead_steps
        )
        left_gripper_phase = _gripper_event_phase(
            chunk.actions[:, 7],
            base_phase_steps=base_gripper_phase,
            lookahead_steps=self.gripper_event_lookahead_steps,
            activation_delta=self.gripper_event_activation_delta,
        )
        right_gripper_phase = _gripper_event_phase(
            chunk.actions[:, 15],
            base_phase_steps=base_gripper_phase,
            lookahead_steps=self.gripper_event_lookahead_steps,
            activation_delta=self.gripper_event_activation_delta,
        )
        grippers = np.array(
            (
                _interpolate(chunk.actions[:, 7, None], left_gripper_phase)[0],
                _interpolate(chunk.actions[:, 15, None], right_gripper_phase)[0],
            ),
            dtype=np.float64,
        )
        status = "tracking" if self._phase < len(chunk.actions) - 1 else "chunk_tail_hold"

        if self._transition_at is not None and self.blend_duration_sec > 0:
            alpha = (now - self._transition_at) / self.blend_duration_sec
            if 0.0 <= alpha < 1.0 and self._transition_origin is not None:
                if self._transition_offset is None:
                    self._transition_offset = self._transition_origin - target
                # Keep the new chunk's complete per-tick motion and only decay
                # the handoff position offset. The former smoothstep blend
                # multiplied the new motion by an alpha starting at zero, so
                # every inference response repeatedly stopped and restarted
                # the commanded trajectory.
                target = target + (1.0 - alpha) * self._transition_offset
                status = "blending"
            elif alpha >= 1.0:
                self._transition_at = None
                self._transition_origin = None
                self._transition_offset = None

        # Joint trajectories retain one shared measured phase. Each gripper may
        # independently lead to a nearby explicit close event in the same chunk.
        target = target.copy()
        target[7] = grippers[0]
        target[15] = grippers[1]
        self._last_target = target.copy()
        remaining = float(len(chunk.actions) - 1 - self._phase) / self.action_hz
        return BimanualProgressSample(
            left_target=target[:7].copy(),
            right_target=target[8:15].copy(),
            left_gripper_target=float(target[7]),
            right_gripper_target=float(target[15]),
            status=status,
            phase_steps=self._phase,
            target_phase_steps=target_phase,
            projection_error_rad=error,
            source_age_sec=source_age,
            remaining_sec=remaining,
        )


class BimanualFixedHorizonActionChunk:
    """Execute a fixed prefix of one bimanual action chunk at its native rate.

    This is deliberately separate from feedback-phase alignment.  It is an
    Fixed-horizon executor for policy evaluation: all arm and gripper targets use the
    same integer action index. Phase advances once per control-loop sample,
    rather than from a floating wall-clock estimate. A short arm-only blend
    may smooth chunk handoffs; grippers always retain the exact action index.
    """

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        execution_horizon_steps: int = 8,
        blend_duration_sec: float = 0.067,
        stale_after_sec: float = 0.15,
    ) -> None:
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if execution_horizon_steps < 2:
            raise ValueError("execution_horizon_steps must be at least two")
        if blend_duration_sec < 0 or stale_after_sec < 0:
            raise ValueError("blend_duration_sec and stale_after_sec must be non-negative")
        self.action_hz = float(action_hz)
        self.execution_horizon_steps = int(execution_horizon_steps)
        self.blend_duration_sec = float(blend_duration_sec)
        self.stale_after_sec = float(stale_after_sec)
        self._chunk: _Chunk | None = None
        self._last_arm_target: np.ndarray | None = None
        self._transition_arm: np.ndarray | None = None
        self._transition_offset: np.ndarray | None = None
        self._transition_at: float | None = None
        self._samples_emitted = 0
        self._start_index = 0
        self._last_sample_at: float | None = None

    def push(
        self,
        actions: np.ndarray,
        *,
        initial_phase_steps: float = 0.0,
        observed_at: float,
        received_at: float,
    ) -> None:
        values = np.asarray(actions, dtype=np.float64)
        arm_matrix(values)
        if self.execution_horizon_steps > len(values):
            raise ValueError("execution horizon exceeds action chunk length")
        if not 0.0 <= initial_phase_steps < self.execution_horizon_steps:
            raise ValueError("initial phase must lie inside the fixed execution horizon")
        self._start_index = int(np.ceil(initial_phase_steps - 1e-9))
        sample_count = self.execution_horizon_steps - self._start_index
        self._chunk = _Chunk(
            values.copy(),
            float(observed_at),
            float(received_at),
            float(received_at) + sample_count / self.action_hz + self.stale_after_sec,
        )
        self._transition_arm = (
            None if self._last_arm_target is None else self._last_arm_target.copy()
        )
        self._transition_offset = None
        self._transition_at = float(received_at)
        self._samples_emitted = 0
        self._last_sample_at = None

    def sample(self, now: float, *, feedback: np.ndarray) -> BimanualProgressSample:
        del feedback  # Fixed-horizon phase intentionally does not project feedback.
        if self._chunk is None:
            return BimanualProgressSample(
                None, None, None, None, "no_chunk_hold", None, None, None, None, None
            )
        chunk = self._chunk
        source_age = float(now - chunk.observed_at)
        if now > chunk.stale_at:
            return BimanualProgressSample(
                None,
                None,
                None,
                None,
                "stale_chunk_hold",
                float(max(0, self._samples_emitted - 1)),
                float(max(0, self._samples_emitted - 1)),
                None,
                source_age,
                0.0,
            )
        # This function is called exactly once per scheduler tick. Advancing
        # with a counter avoids float rounding such as 0 -> 0 -> 2 at 30 Hz.
        same_tick = self._last_sample_at is not None and now == self._last_sample_at
        sample_count = self.execution_horizon_steps - self._start_index
        if self._samples_emitted >= sample_count and not same_tick:
            index = self.execution_horizon_steps - 1
            return BimanualProgressSample(
                left_target=chunk.actions[index, :7].copy(),
                right_target=chunk.actions[index, 8:15].copy(),
                left_gripper_target=float(chunk.actions[index, 7]),
                right_gripper_target=float(chunk.actions[index, 15]),
                status="fixed_horizon_complete_hold",
                phase_steps=float(index),
                target_phase_steps=float(index),
                projection_error_rad=None,
                source_age_sec=source_age,
                remaining_sec=0.0,
            )
        if same_tick:
            index = self._start_index + max(0, self._samples_emitted - 1)
        else:
            index = self._start_index + self._samples_emitted
            self._samples_emitted += 1
            self._last_sample_at = now
        target = chunk.actions[index].copy()
        status = "fixed_horizon_tracking"
        if self._transition_at is not None and self._transition_arm is not None:
            alpha = (now - self._transition_at) / self.blend_duration_sec if self.blend_duration_sec else 1.0
            if 0.0 <= alpha < 1.0:
                arm_target = np.concatenate((target[:7], target[8:15]))
                if self._transition_offset is None:
                    self._transition_offset = self._transition_arm - arm_target
                # Preserve the new chunk's per-row motion and decay only the
                # handoff position offset. Scaling the complete target delta
                # restarts velocity from zero at every chunk boundary.
                arm_target = arm_target + (1.0 - alpha) * self._transition_offset
                target[:7] = arm_target[:7]
                target[8:15] = arm_target[7:]
                status = "fixed_horizon_blending"
            else:
                self._transition_at = None
                self._transition_arm = None
                self._transition_offset = None
        self._last_arm_target = np.concatenate((target[:7], target[8:15]))
        return BimanualProgressSample(
            left_target=target[:7].copy(),
            right_target=target[8:15].copy(),
            left_gripper_target=float(target[7]),
            right_gripper_target=float(target[15]),
            status=status,
            phase_steps=float(index),
            target_phase_steps=float(index),
            projection_error_rad=None,
            source_age_sec=source_age,
            remaining_sec=max(0.0, (sample_count - self._samples_emitted) / self.action_hz),
        )


@dataclass(frozen=True)
class BimanualRtcRequest:
    previous_actions: np.ndarray
    valid_previous_steps: int
    emitted_at_request: int
    requested_at: float
    predicted_delay_steps: int


class BimanualFeedbackRtcActionQueue(BimanualFeedbackProgressActionChunk):
    """Prefetch RTC chunks while advancing execution from measured arm progress."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cumulative_progress = 0.0
        self._progress_rate_hz = 0.0
        self._last_progress_sample_at: float | None = None

    @property
    def remaining_steps(self) -> int:
        if self._chunk is None:
            return 0
        return max(0, int(np.ceil(len(self._chunk.actions) - 1 - self._phase)))

    @property
    def progress_rate_hz(self) -> float:
        return self._progress_rate_hz

    @property
    def cumulative_progress_steps(self) -> float:
        return self._cumulative_progress

    def push(
        self,
        actions: np.ndarray,
        *,
        initial_phase_steps: float,
        observed_at: float,
        received_at: float,
    ) -> None:
        super().push(
            actions,
            initial_phase_steps=initial_phase_steps,
            observed_at=observed_at,
            received_at=received_at,
        )
        # Feedback progress, not the nominal action rate, owns this chunk's
        # lifetime. The caller separately times out a queue that has reached
        # its tail without a safe replacement.
        assert self._chunk is not None
        self._chunk = _Chunk(
            self._chunk.actions,
            self._chunk.observed_at,
            self._chunk.received_at,
            float("inf"),
        )
        self._last_progress_sample_at = None

    def make_request(
        self,
        *,
        now: float,
        execution_horizon: int,
        predicted_delay_steps: int,
    ) -> BimanualRtcRequest:
        if self._chunk is None:
            raise RuntimeError("RTC request requires a current feedback-progress chunk")
        if execution_horizon < 2 or predicted_delay_steps < 0:
            raise ValueError("invalid RTC horizon or delay")
        start = min(
            len(self._chunk.actions) - 1,
            int(np.ceil(self._phase + self.arm_lead_steps)),
        )
        remaining = self._chunk.actions[start:]
        valid_steps = min(len(remaining), execution_horizon)
        prefix = remaining[:valid_steps].copy()
        if valid_steps < execution_horizon:
            padding = np.repeat(prefix[-1:], execution_horizon - valid_steps, axis=0)
            prefix = np.concatenate((prefix, padding), axis=0)
        return BimanualRtcRequest(
            previous_actions=prefix,
            valid_previous_steps=valid_steps,
            emitted_at_request=int(np.floor(self._cumulative_progress)),
            requested_at=float(now),
            predicted_delay_steps=int(predicted_delay_steps),
        )

    def consumed_since(self, request: BimanualRtcRequest) -> int:
        return max(
            0,
            int(np.floor(self._cumulative_progress)) - request.emitted_at_request,
        )

    def sample(self, now: float, *, feedback: np.ndarray) -> BimanualProgressSample:
        previous_phase = self._phase
        sample = super().sample(now, feedback=feedback)
        progress = max(0.0, self._phase - previous_phase)
        self._cumulative_progress += progress
        if self._last_progress_sample_at is not None:
            dt = float(now - self._last_progress_sample_at)
            if dt > 0:
                instantaneous_rate = progress / dt
                self._progress_rate_hz = (
                    0.8 * self._progress_rate_hz + 0.2 * instantaneous_rate
                )
        self._last_progress_sample_at = float(now)
        status = {
            "tracking": "rtc_feedback_tracking",
            "blending": "rtc_feedback_blending",
            "chunk_tail_hold": "rtc_feedback_tail_hold",
            "stale_chunk_hold": "rtc_feedback_stale_hold",
            "no_chunk_hold": "rtc_feedback_no_chunk_hold",
        }.get(sample.status, sample.status)
        return BimanualProgressSample(
            left_target=sample.left_target,
            right_target=sample.right_target,
            left_gripper_target=sample.left_gripper_target,
            right_gripper_target=sample.right_gripper_target,
            status=status,
            phase_steps=sample.phase_steps,
            target_phase_steps=sample.target_phase_steps,
            projection_error_rad=sample.projection_error_rad,
            source_age_sec=sample.source_age_sec,
            remaining_sec=sample.remaining_sec,
        )


class BimanualRtcActionQueue:
    """FIFO action queue with atomic RTC chunk replacement."""

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        handoff_decay_steps: int = 0,
        max_handoff_error_rad: float | None = None,
    ) -> None:
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        self.action_hz = float(action_hz)
        if handoff_decay_steps < 0:
            raise ValueError("handoff_decay_steps must be non-negative")
        if max_handoff_error_rad is not None and max_handoff_error_rad <= 0:
            raise ValueError("max_handoff_error_rad must be positive when provided")
        self.handoff_decay_steps = int(handoff_decay_steps)
        self.max_handoff_error_rad = max_handoff_error_rad
        self._actions: np.ndarray | None = None
        self._index = 0
        self._emitted = 0
        self._last_action: np.ndarray | None = None
        self._last_sample_at: float | None = None
        self._next_action_at: float | None = None
        self._handoff_offset: np.ndarray | None = None
        self._loaded_emissions = 0
        self._last_handoff_error_rad = 0.0

    @property
    def remaining_steps(self) -> int:
        return 0 if self._actions is None else max(0, len(self._actions) - self._index)

    @property
    def emitted_steps(self) -> int:
        return self._emitted

    @property
    def last_handoff_error_rad(self) -> float:
        return self._last_handoff_error_rad

    def handoff_error_rad(self, actions: np.ndarray, *, skip_steps: int = 0) -> float:
        return float(np.max(self.handoff_errors_rad(actions, skip_steps=skip_steps)))

    def handoff_errors_rad(self, actions: np.ndarray, *, skip_steps: int = 0) -> np.ndarray:
        """Return independent maximum joint handoff errors for left and right."""
        values = np.asarray(actions, dtype=np.float64)
        arm_matrix(values)
        if not 0 <= skip_steps < len(values):
            raise ValueError("RTC skip_steps must retain at least one action")
        if self._last_action is None:
            return np.zeros(2, dtype=np.float64)
        previous_arm = np.concatenate((self._last_action[:7], self._last_action[8:15]))
        next_arm = arm_matrix(values)[skip_steps]
        error = np.abs(previous_arm - next_arm)
        return np.asarray(
            (np.max(error[:7]), np.max(error[7:])), dtype=np.float64
        )

    def reanchor_hold(self, action: np.ndarray, *, now: float) -> None:
        """Discard stale policy rows and restart RTC from a commanded hold.

        Cartesian safety assists can deliberately move one arm away from the
        learned action stream.  An RTC replacement conditioned on the old tail
        is no longer a meaningful continuation in that case.  Keep two copies
        of the live command so the next request has a valid, explicit hold
        prefix while a fresh inference request is started.
        """
        anchor = np.asarray(action, dtype=np.float64)
        if anchor.shape != (16,):
            raise ValueError("RTC reanchor action must have shape (16,)")
        hold_actions = np.repeat(anchor[None, :], 2, axis=0)
        arm_matrix(hold_actions)
        self._actions = hold_actions
        self._index = 0
        self._last_action = anchor.copy()
        self._last_sample_at = None
        self._next_action_at = float(now)
        self._handoff_offset = None
        self._loaded_emissions = 0
        self._last_handoff_error_rad = 0.0

    def load(self, actions: np.ndarray, *, skip_steps: int = 0) -> None:
        values = np.asarray(actions, dtype=np.float64)
        arm_matrix(values)
        if not 0 <= skip_steps < len(values):
            raise ValueError("RTC skip_steps must retain at least one action")
        handoff_error = self.handoff_error_rad(values, skip_steps=skip_steps)
        if (
            self.max_handoff_error_rad is not None
            and handoff_error > self.max_handoff_error_rad
        ):
            raise ValueError(
                "RTC handoff exceeds configured limit: "
                f"error_deg={np.rad2deg(handoff_error):.3f} "
                f"limit_deg={np.rad2deg(self.max_handoff_error_rad):.3f}"
            )
        self._actions = values[skip_steps:].copy()
        self._index = 0
        self._last_sample_at = None
        self._loaded_emissions = 0
        self._last_handoff_error_rad = handoff_error
        if self._last_action is None or self.handoff_decay_steps == 0:
            self._handoff_offset = None
        else:
            previous_arm = np.concatenate((self._last_action[:7], self._last_action[8:15]))
            next_arm = arm_matrix(values)[skip_steps]
            self._handoff_offset = previous_arm - next_arm

    def _apply_handoff_offset(self, action: np.ndarray, local_index: int) -> np.ndarray:
        target = action.copy()
        if self._handoff_offset is None or self.handoff_decay_steps == 0:
            return target
        weight = max(0.0, 1.0 - local_index / self.handoff_decay_steps)
        target[:7] += weight * self._handoff_offset[:7]
        target[8:15] += weight * self._handoff_offset[7:]
        return target

    def make_request(
        self,
        *,
        now: float,
        execution_horizon: int,
        predicted_delay_steps: int,
    ) -> BimanualRtcRequest:
        if execution_horizon < 2 or predicted_delay_steps < 0:
            raise ValueError("invalid RTC horizon or delay")
        if self.remaining_steps > 0:
            assert self._actions is not None
            remaining = self._actions[self._index :].copy()
            for offset in range(len(remaining)):
                remaining[offset] = self._apply_handoff_offset(
                    remaining[offset], self._loaded_emissions + offset
                )
            valid_steps = min(len(remaining), execution_horizon)
            prefix = remaining[:valid_steps].copy()
            if valid_steps < execution_horizon:
                padding = np.repeat(prefix[-1:], execution_horizon - valid_steps, axis=0)
                prefix = np.concatenate((prefix, padding), axis=0)
        elif self._last_action is not None:
            # A rejected replacement can exhaust the active queue. Continue RTC
            # from an explicit safe hold trajectory so inference can refill it.
            prefix = np.repeat(self._last_action[None, :], execution_horizon, axis=0)
            valid_steps = execution_horizon
        else:
            raise RuntimeError("RTC request requires a queued or previously emitted action")
        return BimanualRtcRequest(
            previous_actions=prefix,
            valid_previous_steps=valid_steps,
            emitted_at_request=self._emitted,
            requested_at=float(now),
            predicted_delay_steps=int(predicted_delay_steps),
        )

    def consumed_since(self, request: BimanualRtcRequest) -> int:
        return max(0, self._emitted - request.emitted_at_request)

    def sample(self, now: float, *, feedback: np.ndarray) -> BimanualProgressSample:
        del feedback
        same_tick = self._last_sample_at is not None and now == self._last_sample_at
        action_due = self._next_action_at is None or now + 1e-9 >= self._next_action_at
        if self._actions is not None and self._index < len(self._actions):
            if not same_tick and action_due:
                self._last_action = self._apply_handoff_offset(
                    self._actions[self._index], self._loaded_emissions
                )
                self._index += 1
                self._emitted += 1
                self._loaded_emissions += 1
                self._last_sample_at = float(now)
                self._next_action_at = (
                    float(now) + 1.0 / self.action_hz
                    if self._next_action_at is None
                    else self._next_action_at + 1.0 / self.action_hz
                )
            assert self._last_action is not None
            target = self._last_action
            status = "rtc_tracking" if action_due else "rtc_rate_hold"
        elif self._last_action is not None:
            target = self._last_action
            status = "rtc_queue_hold"
        else:
            return BimanualProgressSample(
                None, None, None, None, "no_chunk_hold", None, None, None, None, None
            )
        return BimanualProgressSample(
            left_target=target[:7].copy(),
            right_target=target[8:15].copy(),
            left_gripper_target=float(target[7]),
            right_gripper_target=float(target[15]),
            status=status,
            phase_steps=float(self._emitted - 1),
            target_phase_steps=float(self._emitted - 1),
            projection_error_rad=None,
            source_age_sec=None,
            remaining_sec=self.remaining_steps / self.action_hz,
        )
