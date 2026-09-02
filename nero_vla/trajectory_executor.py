"""Model-independent action-chunk buffering and joint trajectory limiting.

This module deliberately has no robot SDK dependency.  A hardware backend may
send the returned commands only after it has independently validated feedback
and the selected continuous-control API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_COUNT = 7


def _joint_vector(value: float | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(JOINT_COUNT, float(result), dtype=np.float64)
    if result.shape != (JOINT_COUNT,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and have shape ({JOINT_COUNT},)")
    return result


@dataclass(frozen=True)
class TrajectorySample:
    target: np.ndarray | None
    status: str
    source_age_sec: float | None
    remaining_sec: float | None
    target_velocity: np.ndarray | None = None


@dataclass(frozen=True)
class CommandSample:
    command: np.ndarray
    velocity: np.ndarray
    desired: np.ndarray | None
    status: str
    feedback_governed: bool = False
    joint_limit_clamped: bool = False


@dataclass(frozen=True)
class ChunkAlignment:
    offset_steps: float
    action: np.ndarray
    latency_steps: float
    max_feedback_error_rad: float
    rms_feedback_error_rad: float
    max_command_error_rad: float
    score: float
    segment_index: int
    segment_alpha: float
    search_max_steps: float


@dataclass(frozen=True)
class ProgressTrajectorySample:
    arm_target: np.ndarray | None
    gripper_target: float | None
    status: str
    source_age_sec: float | None
    remaining_sec: float | None
    phase_steps: float | None
    arm_target_phase_steps: float | None
    phase_feedback_error_rad: float | None


def align_action_chunk_to_state(
    actions: np.ndarray,
    *,
    feedback: np.ndarray,
    command: np.ndarray,
    latency_steps: float,
    max_alignment_error_rad: float,
    search_margin_steps: float = 2.0,
    command_weight: float = 0.5,
    latency_weight: float = 0.05,
    command_velocity: np.ndarray | None = None,
    direction_weight: float = 0.05,
) -> ChunkAlignment:
    """Project live arm state onto a bounded prefix of an action trajectory."""
    positions = np.asarray(actions, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != JOINT_COUNT
        or len(positions) < 2
        or not np.isfinite(positions).all()
    ):
        raise ValueError(f"actions must have shape (horizon, {JOINT_COUNT})")
    live_feedback = _joint_vector(feedback, "feedback")
    live_command = _joint_vector(command, "command")
    if not np.isfinite(latency_steps) or latency_steps < 0:
        raise ValueError("latency_steps must be finite and non-negative")
    if not np.isfinite(max_alignment_error_rad) or max_alignment_error_rad <= 0:
        raise ValueError("max_alignment_error_rad must be finite and positive")
    if not np.isfinite(search_margin_steps) or search_margin_steps < 0:
        raise ValueError("search_margin_steps must be finite and non-negative")
    if command_weight < 0 or latency_weight < 0 or direction_weight < 0:
        raise ValueError("alignment weights must be non-negative")

    velocity = (
        np.zeros(JOINT_COUNT, dtype=np.float64)
        if command_velocity is None
        else _joint_vector(command_velocity, "command_velocity")
    )
    search_max = min(
        float(len(positions) - 2),
        float(latency_steps) + float(search_margin_steps),
    )
    error_scale_sq = float(max_alignment_error_rad) ** 2
    latency_scale_sq = max(1.0, float(latency_steps)) ** 2
    velocity_norm = float(np.linalg.norm(velocity))
    best: ChunkAlignment | None = None

    segment_count = max(1, min(len(positions) - 1, int(np.ceil(search_max))))
    for segment_index in range(segment_count):
        start = positions[segment_index]
        delta = positions[segment_index + 1] - start
        alpha_max = float(np.clip(search_max - segment_index, 0.0, 1.0))

        quadratic = (
            (1.0 + command_weight) * float(delta @ delta) / error_scale_sq
            + latency_weight / latency_scale_sq
        )
        linear = (
            float(delta @ (start - live_feedback))
            + command_weight * float(delta @ (start - live_command))
        ) / error_scale_sq
        linear += latency_weight * (segment_index - latency_steps) / latency_scale_sq
        alpha = 0.0 if quadratic <= 1e-15 else float(np.clip(-linear / quadratic, 0.0, alpha_max))

        candidate = start + alpha * delta
        feedback_error = candidate - live_feedback
        command_error = candidate - live_command
        offset_steps = float(segment_index + alpha)
        score = float(np.mean(np.square(feedback_error)) / error_scale_sq)
        score += command_weight * float(np.mean(np.square(command_error)) / error_scale_sq)
        score += latency_weight * ((offset_steps - latency_steps) ** 2) / latency_scale_sq

        delta_norm = float(np.linalg.norm(delta))
        if velocity_norm > 1e-9 and delta_norm > 1e-9:
            direction_cosine = float(delta @ velocity) / (delta_norm * velocity_norm)
            if direction_cosine < 0:
                score += direction_weight * direction_cosine**2

        alignment = ChunkAlignment(
            offset_steps=offset_steps,
            action=candidate.copy(),
            latency_steps=float(latency_steps),
            max_feedback_error_rad=float(np.max(np.abs(feedback_error))),
            rms_feedback_error_rad=float(np.sqrt(np.mean(np.square(feedback_error)))),
            max_command_error_rad=float(np.max(np.abs(command_error))),
            score=score,
            segment_index=segment_index,
            segment_alpha=alpha,
            search_max_steps=search_max,
        )
        if best is None or alignment.score < best.score:
            best = alignment

    assert best is not None
    if best.max_feedback_error_rad > max_alignment_error_rad:
        raise ValueError(
            "no safely aligned action in bounded search window: "
            f"offset={best.offset_steps:.2f} "
            f"max_error_deg={np.rad2deg(best.max_feedback_error_rad):.3f} "
            f"limit_deg={np.rad2deg(max_alignment_error_rad):.3f}"
        )
    return best


def _interpolate_steps(positions: np.ndarray, phase_steps: float) -> np.ndarray:
    phase = float(np.clip(phase_steps, 0.0, len(positions) - 1))
    left = min(int(np.floor(phase)), len(positions) - 2)
    alpha = phase - left
    return positions[left] + alpha * (positions[left + 1] - positions[left])


def _project_feedback_forward(
    positions: np.ndarray,
    feedback: np.ndarray,
    *,
    phase_steps: float,
    max_advance_steps: float,
) -> tuple[float, float]:
    """Project feedback onto a short, forward-only section of a joint path."""
    start_phase = float(np.clip(phase_steps, 0.0, len(positions) - 1))
    end_phase = min(float(len(positions) - 1), start_phase + max_advance_steps)
    best_phase = start_phase
    best_error_sq = float("inf")

    first_segment = min(int(np.floor(start_phase)), len(positions) - 2)
    last_segment = min(int(np.floor(end_phase)), len(positions) - 2)
    for segment_index in range(first_segment, last_segment + 1):
        start = positions[segment_index]
        delta = positions[segment_index + 1] - start
        alpha_min = float(np.clip(start_phase - segment_index, 0.0, 1.0))
        alpha_max = float(np.clip(end_phase - segment_index, 0.0, 1.0))
        if alpha_max < alpha_min:
            continue
        denominator = float(delta @ delta)
        alpha = (
            alpha_min
            if denominator <= 1e-15
            else float(
                np.clip(
                    delta @ (feedback - start) / denominator,
                    alpha_min,
                    alpha_max,
                )
            )
        )
        candidate = start + alpha * delta
        error_sq = float(np.mean(np.square(candidate - feedback)))
        if error_sq < best_error_sq:
            best_phase = float(segment_index + alpha)
            best_error_sq = error_sq

    return best_phase, float(np.sqrt(best_error_sq))


@dataclass(frozen=True)
class _ProgressActionChunk:
    actions: np.ndarray
    observed_at: float
    received_at: float
    stale_at: float


class FeedbackProgressActionChunk:
    """Advance a shared arm/gripper chunk phase from measured arm progress."""

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        arm_lead_steps: float = 1.0,
        gripper_lead_steps: float = 0.0,
        max_progress_steps_per_tick: float = 1.0,
        blend_duration_sec: float = 0.10,
        stale_after_sec: float = 0.15,
    ) -> None:
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if arm_lead_steps <= 0:
            raise ValueError("arm_lead_steps must be positive")
        if gripper_lead_steps < 0:
            raise ValueError("gripper_lead_steps must be non-negative")
        if max_progress_steps_per_tick <= 0:
            raise ValueError("max_progress_steps_per_tick must be positive")
        if blend_duration_sec < 0 or stale_after_sec < 0:
            raise ValueError("blend and stale durations must be non-negative")
        self.action_hz = float(action_hz)
        self.arm_lead_steps = float(arm_lead_steps)
        self.gripper_lead_steps = float(gripper_lead_steps)
        self.max_progress_steps_per_tick = float(max_progress_steps_per_tick)
        self.blend_duration_sec = float(blend_duration_sec)
        self.stale_after_sec = float(stale_after_sec)
        self._current: _ProgressActionChunk | None = None
        self._phase_steps = 0.0
        self._transition_at: float | None = None
        self._transition_arm: np.ndarray | None = None
        self._transition_gripper: float | None = None
        self._last_arm_target: np.ndarray | None = None
        self._last_gripper_target: float | None = None

    def push(
        self,
        actions: np.ndarray,
        *,
        initial_phase_steps: float,
        observed_at: float,
        received_at: float,
    ) -> None:
        values = np.asarray(actions, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] < JOINT_COUNT + 1
            or len(values) < 2
            or not np.isfinite(values).all()
        ):
            raise ValueError("actions must contain at least two finite 8D samples")
        if not np.isfinite([initial_phase_steps, observed_at, received_at]).all():
            raise ValueError("chunk phase and timestamps must be finite")
        if received_at + 0.1 < observed_at:
            raise ValueError("observation timestamp is unexpectedly in the future")
        if not 0.0 <= initial_phase_steps < len(values) - 1:
            raise ValueError("initial_phase_steps must leave at least one future action")

        nominal_remaining_sec = (
            len(values) - 1 - float(initial_phase_steps)
        ) / self.action_hz
        self._transition_arm = (
            None if self._last_arm_target is None else self._last_arm_target.copy()
        )
        self._transition_gripper = self._last_gripper_target
        self._transition_at = float(received_at)
        self._current = _ProgressActionChunk(
            actions=values.copy(),
            observed_at=float(observed_at),
            received_at=float(received_at),
            stale_at=float(received_at) + nominal_remaining_sec + self.stale_after_sec,
        )
        self._phase_steps = float(initial_phase_steps)

    def sample(
        self,
        now: float,
        *,
        arm_feedback: np.ndarray,
    ) -> ProgressTrajectorySample:
        if self._current is None:
            return ProgressTrajectorySample(
                None, None, "no_chunk_hold", None, None, None, None, None
            )
        current = self._current
        source_age = float(now - current.observed_at)
        remaining_steps = float(len(current.actions) - 1 - self._phase_steps)
        remaining_sec = remaining_steps / self.action_hz
        if now > current.stale_at:
            return ProgressTrajectorySample(
                None,
                None,
                "stale_chunk_hold",
                source_age,
                remaining_sec,
                self._phase_steps,
                None,
                None,
            )

        feedback = _joint_vector(arm_feedback, "arm_feedback")
        phase, projection_error = _project_feedback_forward(
            current.actions[:, :JOINT_COUNT],
            feedback,
            phase_steps=self._phase_steps,
            max_advance_steps=self.max_progress_steps_per_tick,
        )
        self._phase_steps = max(self._phase_steps, phase)
        remaining_steps = float(len(current.actions) - 1 - self._phase_steps)
        remaining_sec = remaining_steps / self.action_hz
        arm_phase = min(
            float(len(current.actions) - 1),
            self._phase_steps + self.arm_lead_steps,
        )
        arm_target = _interpolate_steps(
            current.actions[:, :JOINT_COUNT],
            arm_phase,
        )
        gripper_phase = min(
            float(len(current.actions) - 1),
            self._phase_steps + self.gripper_lead_steps,
        )
        gripper_target = float(
            _interpolate_steps(
                current.actions[:, JOINT_COUNT : JOINT_COUNT + 1],
                gripper_phase,
            )[0]
        )
        status = "tracking" if remaining_steps > 0 else "chunk_tail_hold"

        if self._transition_at is not None and self.blend_duration_sec > 0:
            alpha = (now - self._transition_at) / self.blend_duration_sec
            if 0.0 <= alpha < 1.0:
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                if self._transition_arm is not None:
                    arm_target = self._transition_arm + smooth_alpha * (
                        arm_target - self._transition_arm
                    )
                if self._transition_gripper is not None:
                    gripper_target = self._transition_gripper + smooth_alpha * (
                        gripper_target - self._transition_gripper
                    )
                status = "blending"
            elif alpha >= 1.0:
                self._transition_at = None
                self._transition_arm = None
                self._transition_gripper = None

        self._last_arm_target = arm_target.copy()
        self._last_gripper_target = gripper_target
        return ProgressTrajectorySample(
            arm_target=arm_target,
            gripper_target=gripper_target,
            status=status,
            source_age_sec=source_age,
            remaining_sec=remaining_sec,
            phase_steps=self._phase_steps,
            arm_target_phase_steps=arm_phase,
            phase_feedback_error_rad=projection_error,
        )


@dataclass(frozen=True)
class _ActionChunk:
    positions: np.ndarray
    sample_times: np.ndarray
    observed_at: float
    received_at: float


class ActionChunkBuffer:
    """Put timestamped policy chunks on one continuous execution timeline."""

    def __init__(
        self,
        *,
        action_hz: float = 30.0,
        action_dim: int = JOINT_COUNT,
        blend_duration_sec: float = 0.10,
        stale_after_sec: float = 0.10,
        first_action_offset_steps: int = 1,
    ) -> None:
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if blend_duration_sec < 0 or stale_after_sec < 0:
            raise ValueError("blend and stale durations must be non-negative")
        if first_action_offset_steps < 0:
            raise ValueError("first_action_offset_steps must be non-negative")
        self.action_hz = float(action_hz)
        self.action_dim = int(action_dim)
        self.blend_duration_sec = float(blend_duration_sec)
        self.stale_after_sec = float(stale_after_sec)
        self.first_action_offset_steps = int(first_action_offset_steps)
        self._current: _ActionChunk | None = None
        self._previous: _ActionChunk | None = None
        self._transition_at: float | None = None

    def push(self, actions: np.ndarray, *, observed_at: float, received_at: float) -> None:
        positions = np.asarray(actions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape (horizon, {self.action_dim})")
        if len(positions) < 2 or not np.isfinite(positions).all():
            raise ValueError("actions must contain at least two finite samples")
        if not np.isfinite([observed_at, received_at]).all():
            raise ValueError("chunk timestamps must be finite")
        if received_at + 0.1 < observed_at:
            raise ValueError("observation timestamp is unexpectedly in the future")

        offsets = self.first_action_offset_steps + np.arange(len(positions))
        sample_times = float(observed_at) + offsets / self.action_hz
        if sample_times[-1] <= received_at:
            raise ValueError("action chunk has no future samples when received")

        previous = self._current
        if previous is not None and received_at > previous.sample_times[-1] + self.stale_after_sec:
            previous = None
        self._previous = previous
        self._current = _ActionChunk(
            positions=positions.copy(),
            sample_times=sample_times,
            observed_at=float(observed_at),
            received_at=float(received_at),
        )
        self._transition_at = float(received_at)

    def sample(self, now: float) -> TrajectorySample:
        if self._current is None:
            return TrajectorySample(None, "no_chunk_hold", None, None)
        current = self._current
        source_age = float(now - current.observed_at)
        remaining = float(current.sample_times[-1] - now)
        if now > current.sample_times[-1] + self.stale_after_sec:
            return TrajectorySample(None, "stale_chunk_hold", source_age, remaining)

        target = self._interpolate(current, now)
        status = "tracking" if remaining >= 0 else "chunk_tail_hold"
        if (
            self._previous is not None
            and self._transition_at is not None
            and self.blend_duration_sec > 0
        ):
            alpha = (now - self._transition_at) / self.blend_duration_sec
            if 0.0 <= alpha < 1.0:
                old_target = self._interpolate(self._previous, now)
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                target = old_target + smooth_alpha * (target - old_target)
                status = "blending"
            elif alpha >= 1.0:
                self._previous = None
        return TrajectorySample(target, status, source_age, remaining)

    @staticmethod
    def _interpolate(chunk: _ActionChunk, now: float) -> np.ndarray:
        if now <= chunk.sample_times[0]:
            return chunk.positions[0].copy()
        if now >= chunk.sample_times[-1]:
            return chunk.positions[-1].copy()
        right = int(np.searchsorted(chunk.sample_times, now, side="right"))
        left = right - 1
        span = chunk.sample_times[right] - chunk.sample_times[left]
        alpha = (now - chunk.sample_times[left]) / span
        return chunk.positions[left] + alpha * (chunk.positions[right] - chunk.positions[left])


class RateLimitedJointFollower:
    """Turn desired samples into bounded commands without ever inventing zeros."""

    def __init__(
        self,
        *,
        max_velocity_rad_s: float | np.ndarray,
        max_acceleration_rad_s2: float | np.ndarray,
        max_jerk_rad_s3: float | np.ndarray | None = None,
        max_target_feedback_error_rad: float | np.ndarray,
        max_command_feedback_error_rad: float | np.ndarray,
        feedback_governor_error_rad: float | np.ndarray | None = None,
        max_tick_interval_sec: float = 0.10,
        joint_limits_rad: np.ndarray | None = None,
        tracking_mode: str = "point_to_point",
        streaming_position_gain_s: float = 6.0,
    ) -> None:
        self.max_velocity = _joint_vector(max_velocity_rad_s, "max_velocity_rad_s")
        self.max_acceleration = _joint_vector(
            max_acceleration_rad_s2, "max_acceleration_rad_s2"
        )
        self.max_jerk = (
            None
            if max_jerk_rad_s3 is None
            else _joint_vector(max_jerk_rad_s3, "max_jerk_rad_s3")
        )
        self.max_target_feedback_error = _joint_vector(
            max_target_feedback_error_rad, "max_target_feedback_error_rad"
        )
        self.max_command_feedback_error = _joint_vector(
            max_command_feedback_error_rad, "max_command_feedback_error_rad"
        )
        self.feedback_governor_error = _joint_vector(
            max_command_feedback_error_rad
            if feedback_governor_error_rad is None
            else feedback_governor_error_rad,
            "feedback_governor_error_rad",
        )
        if np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("velocity and acceleration limits must be positive")
        if self.max_jerk is not None and np.any(self.max_jerk <= 0):
            raise ValueError("jerk limits must be positive")
        if np.any(self.max_target_feedback_error <= 0) or np.any(
            self.max_command_feedback_error <= 0
        ):
            raise ValueError("feedback error limits must be positive")
        if np.any(self.feedback_governor_error <= 0) or np.any(
            self.feedback_governor_error > self.max_command_feedback_error
        ):
            raise ValueError("feedback governor error must be positive and no greater than hard error")
        if max_tick_interval_sec <= 0:
            raise ValueError("max_tick_interval_sec must be positive")
        self.max_tick_interval_sec = float(max_tick_interval_sec)
        if tracking_mode not in {"point_to_point", "streaming_trajectory"}:
            raise ValueError("tracking_mode must be point_to_point or streaming_trajectory")
        if not np.isfinite(streaming_position_gain_s) or streaming_position_gain_s <= 0:
            raise ValueError("streaming_position_gain_s must be finite and positive")
        self.tracking_mode = tracking_mode
        self.streaming_position_gain_s = float(streaming_position_gain_s)
        if joint_limits_rad is None:
            self.joint_limits = None
        else:
            limits = np.asarray(joint_limits_rad, dtype=np.float64)
            if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
                raise ValueError(f"joint_limits_rad must have shape ({JOINT_COUNT}, 2)")
            if np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError("joint lower limits must be below upper limits")
            self.joint_limits = limits
        self._command: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._acceleration: np.ndarray | None = None
        self._last_tick: float | None = None
        self._previous_target: np.ndarray | None = None

    def initialize(self, measured: np.ndarray, *, now: float) -> CommandSample:
        position = _joint_vector(measured, "measured")
        self._validate_position_limits(position, "initial feedback")
        self._command = position.copy()
        self._velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
        self._acceleration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self._last_tick = float(now)
        self._previous_target = None
        return CommandSample(position.copy(), self._velocity.copy(), None, "initialized_hold")

    def step(
        self,
        sample: TrajectorySample,
        *,
        measured: np.ndarray,
        now: float,
    ) -> CommandSample:
        if (
            self._command is None
            or self._velocity is None
            or self._acceleration is None
            or self._last_tick is None
        ):
            raise RuntimeError("follower must be initialized from complete feedback")
        feedback = _joint_vector(measured, "measured")
        dt = float(now - self._last_tick)
        if dt <= 0:
            raise ValueError("control timestamps must be strictly increasing")
        self._last_tick = float(now)
        if dt > self.max_tick_interval_sec:
            return self._hold("scheduler_gap_hold", sample.target)
        if sample.target is None:
            return self._hold(sample.status, None)

        target = _joint_vector(sample.target, "trajectory target")
        try:
            self._validate_position_limits(target, "trajectory target")
        except ValueError:
            return self._hold("joint_limit_rejected", target)
        if np.any(np.abs(target - feedback) > self.max_target_feedback_error):
            return self._hold("target_feedback_rejected", target)

        error = target - self._command
        streaming = self.tracking_mode == "streaming_trajectory" and (
            sample.status.endswith("tracking")
            or sample.status.endswith("blending")
            or sample.status == "rtc_rate_hold"
        )
        if streaming:
            if sample.target_velocity is None:
                target_velocity = (
                    np.zeros(JOINT_COUNT, dtype=np.float64)
                    if self._previous_target is None
                    else (target - self._previous_target) / dt
                )
            else:
                target_velocity = _joint_vector(
                    sample.target_velocity, "trajectory target velocity"
                )
            desired_velocity = np.clip(
                target_velocity + self.streaming_position_gain_s * error,
                -self.max_velocity,
                self.max_velocity,
            )
            direction = np.sign(desired_velocity)
            unconstrained_speed = np.abs(desired_velocity)
        else:
            direction = np.sign(error)
            braking_speed = np.sqrt(2.0 * self.max_acceleration * np.abs(error))
            unconstrained_speed = np.minimum(self.max_velocity, braking_speed)

        # Slow down before the command-to-feedback error reaches the hard guard.
        # Without this governor, a lagging joint repeatedly alternates between a
        # full-speed command and _hold(), which creates a visible stop/start jerk.
        tracking_error = self._command - feedback
        feedback_headroom = np.maximum(
            0.0,
            self.feedback_governor_error - direction * tracking_error,
        )
        feedback_braking_speed = np.sqrt(
            2.0 * self.max_acceleration * feedback_headroom
        )
        feedback_governed = bool(
            np.any((direction != 0.0) & (feedback_braking_speed < unconstrained_speed))
        )
        desired_velocity = direction * np.minimum(
            unconstrained_speed, feedback_braking_speed
        )
        previous_velocity = self._velocity.copy()
        desired_acceleration = np.clip(
            (desired_velocity - previous_velocity) / dt,
            -self.max_acceleration,
            self.max_acceleration,
        )
        if self.max_jerk is None:
            acceleration = desired_acceleration
        else:
            acceleration_delta = np.clip(
                desired_acceleration - self._acceleration,
                -self.max_jerk * dt,
                self.max_jerk * dt,
            )
            acceleration = self._acceleration + acceleration_delta
        velocity = np.clip(
            previous_velocity + acceleration * dt,
            -self.max_velocity,
            self.max_velocity,
        )
        command_step = velocity * dt
        command = self._command + command_step

        # A legal target can still be crossed by one discrete integration step
        # when the follower carries velocity toward a nearby joint limit. Clamp
        # that numerical overshoot and remove only the outward velocity so the
        # next tick starts from a physically valid state.
        joint_limit_clamped = False
        if self.joint_limits is not None:
            lower = self.joint_limits[:, 0]
            upper = self.joint_limits[:, 1]
            below = command < lower
            above = command > upper
            clamped = below | above
            if np.any(clamped):
                command = np.clip(command, lower, upper)
                outward = (below & (velocity < 0.0)) | (above & (velocity > 0.0))
                velocity = velocity.copy()
                velocity[outward] = 0.0
                joint_limit_clamped = True

        # Stop exactly at a nearby target. Discrete velocity integration can
        # otherwise cross it and command a visible correction in the opposite
        # direction on the next tick, especially while holding a chunk tail.
        crossed_target = (direction != 0.0) & (direction * (target - command) <= 0.0)
        if not streaming and np.any(crossed_target):
            command = command.copy()
            velocity = velocity.copy()
            command[crossed_target] = target[crossed_target]
            velocity[crossed_target] = 0.0

        if np.any(np.abs(command - feedback) > self.max_command_feedback_error):
            return self._hold("command_feedback_guard_hold", target)
        self._validate_position_limits(command, "limited command")
        self._command = command
        self._velocity = velocity
        self._acceleration = (velocity - previous_velocity) / dt
        self._previous_target = target.copy()
        return CommandSample(
            command.copy(),
            velocity.copy(),
            target.copy(),
            sample.status,
            feedback_governed,
            joint_limit_clamped,
        )

    def _hold(self, status: str, desired: np.ndarray | None) -> CommandSample:
        assert self._command is not None
        self._velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
        self._acceleration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self._previous_target = None
        desired_copy = None if desired is None else np.asarray(desired, dtype=np.float64).copy()
        return CommandSample(
            self._command.copy(), self._velocity.copy(), desired_copy, status
        )

    def _validate_position_limits(self, position: np.ndarray, label: str) -> None:
        if self.joint_limits is None:
            return
        if np.any(position < self.joint_limits[:, 0]) or np.any(
            position > self.joint_limits[:, 1]
        ):
            raise ValueError(f"{label} lies outside configured joint limits")
