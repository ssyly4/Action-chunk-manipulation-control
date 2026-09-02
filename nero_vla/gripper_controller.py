"""Rate-limited normalized-position controller for the NERO AGX gripper."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GripperCommand:
    normalized: float
    width_m: float
    status: str


class ConfirmedCloseLatch:
    """Require repeated close intent, then prevent reopening during a pick trial."""

    def __init__(self, *, close_threshold: float = 0.30, confirmations: int = 2) -> None:
        if not 0.0 <= close_threshold <= 1.0:
            raise ValueError("close_threshold must be in [0, 1]")
        if confirmations < 1:
            raise ValueError("confirmations must be positive")
        self.close_threshold = float(close_threshold)
        self.confirmations = int(confirmations)
        self._close_count = 0
        self._pending_target: float | None = None
        self._output_target: float | None = None
        self._latched = False
        self._last_confirmation_token: object | None = None

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def close_count(self) -> int:
        return self._close_count

    def update(
        self,
        policy_target: float,
        *,
        confirmation_token: object | None = None,
    ) -> float:
        target = float(policy_target)
        if not np.isfinite(target) or not -0.1 <= target <= 1.1:
            raise ValueError(f"invalid gripper policy target: {policy_target!r}")
        target = float(np.clip(target, 0.0, 1.0))

        if self._latched:
            assert self._output_target is not None
            self._output_target = min(self._output_target, target)
            return self._output_target

        if self._output_target is None:
            self._output_target = target

        if target <= self.close_threshold:
            is_new_confirmation = (
                confirmation_token is None
                or confirmation_token != self._last_confirmation_token
            )
            if is_new_confirmation:
                self._close_count += 1
                self._last_confirmation_token = confirmation_token
                self._pending_target = (
                    target if self._pending_target is None else min(self._pending_target, target)
                )
            if self._close_count >= self.confirmations:
                self._latched = True
                self._output_target = self._pending_target
            return self._output_target

        self._close_count = 0
        self._pending_target = None
        self._output_target = target
        return self._output_target


class ConfirmedGraspState:
    """Map confirmed physical contact to the policy's trained grasp state.

    The real gripper can stop at a wider opening than demonstrations when it
    contacts a different part of the object. Safety control must keep using the
    measured width, but the policy needs the semantic "grasped" state in order
    to advance to its post-grasp trajectory.
    """

    def __init__(self, *, grasp_state: float = 0.23, contact_confirmations: int = 3) -> None:
        if not 0.0 <= grasp_state <= 1.0:
            raise ValueError("grasp_state must be in [0, 1]")
        if contact_confirmations < 1:
            raise ValueError("contact_confirmations must be positive")
        self.grasp_state = float(grasp_state)
        self.contact_confirmations = int(contact_confirmations)
        self._contact_count = 0
        self._latched = False

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def contact_count(self) -> int:
        return self._contact_count

    def update(self, *, close_latched: bool, contact_detected: bool) -> None:
        if self._latched:
            return
        if close_latched and contact_detected:
            self._contact_count += 1
            if self._contact_count >= self.contact_confirmations:
                self._latched = True
            return
        self._contact_count = 0

    def policy_state(self, measured_state: float) -> float:
        measured = float(measured_state)
        if not np.isfinite(measured) or not -0.1 <= measured <= 1.1:
            raise ValueError(f"invalid measured gripper state: {measured_state!r}")
        measured = float(np.clip(measured, 0.0, 1.0))
        return min(measured, self.grasp_state) if self._latched else measured


class RateLimitedGripperFollower:
    """Map policy open fraction to a bounded width command.

    Policy convention is closed=0 and open=1. A detected closing contact holds
    the measured width, preventing a persistent policy target from winding up
    against an object.
    """

    def __init__(
        self,
        *,
        closed_m: float,
        open_m: float,
        max_speed_m_s: float,
        contact_force_n: float,
        contact_preload_m: float = 0.0,
        contact_confirmations: int = 3,
        contact_window_ticks: int = 10,
        max_tick_interval_sec: float = 0.15,
        force_hold_enabled: bool = True,
        feedback_tolerance_m: float = 0.003,
    ) -> None:
        if not np.isfinite(
            [
                closed_m,
                open_m,
                max_speed_m_s,
                contact_force_n,
                contact_preload_m,
                feedback_tolerance_m,
            ]
        ).all():
            raise ValueError("gripper parameters must be finite")
        if open_m - closed_m < 0.005:
            raise ValueError("gripper calibration span must be at least 5 mm")
        if max_speed_m_s <= 0 or contact_force_n <= 0 or max_tick_interval_sec <= 0:
            raise ValueError("gripper speed, contact force, and tick interval must be positive")
        if not 0.0 <= contact_preload_m <= 0.005:
            raise ValueError("contact preload must be in [0, 5] mm")
        if not 0.0 <= feedback_tolerance_m <= 0.01:
            raise ValueError("feedback tolerance must be in [0, 10] mm")
        if contact_confirmations < 1 or contact_window_ticks < contact_confirmations:
            raise ValueError("contact window must contain the required confirmations")
        self.closed_m = float(closed_m)
        self.open_m = float(open_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.contact_force_n = float(contact_force_n)
        self.contact_preload_m = float(contact_preload_m)
        self.contact_confirmations = int(contact_confirmations)
        self.contact_window_ticks = int(contact_window_ticks)
        self.max_tick_interval_sec = float(max_tick_interval_sec)
        self.force_hold_enabled = bool(force_hold_enabled)
        self.feedback_tolerance_m = float(feedback_tolerance_m)
        self._command_m: float | None = None
        self._last_tick: float | None = None
        self._contact_history: deque[bool] = deque(maxlen=self.contact_window_ticks)
        self._contact_latched = False
        self._contact_width_m: float | None = None

    @property
    def contact_latched(self) -> bool:
        return self._contact_latched

    def initialize(self, measured_width_m: float, *, now: float) -> GripperCommand:
        measured = self._validate_width(measured_width_m)
        self._command_m = measured
        self._last_tick = float(now)
        return GripperCommand(self._normalize(measured), measured, "initialized_hold")

    def step(
        self,
        policy_target: float,
        *,
        measured_width_m: float,
        measured_force_n: float,
        now: float,
        contact_latch_enabled: bool = False,
    ) -> GripperCommand:
        if self._command_m is None or self._last_tick is None:
            raise RuntimeError("gripper follower must be initialized from feedback")
        measured = self._validate_width(measured_width_m)
        if not np.isfinite([policy_target, measured_force_n, now]).all():
            return self._hold("invalid_gripper_input_hold")
        dt = float(now - self._last_tick)
        if dt <= 0:
            raise ValueError("gripper timestamps must be strictly increasing")
        self._last_tick = float(now)
        if dt > self.max_tick_interval_sec:
            return self._hold("gripper_scheduler_gap_hold")
        if not -0.1 <= policy_target <= 1.1:
            return self._hold("gripper_policy_rejected")

        normalized_target = float(np.clip(policy_target, 0.0, 1.0))
        target_m = self.closed_m + normalized_target * (self.open_m - self.closed_m)
        closing = target_m < measured - 0.0005

        if self._contact_latched:
            assert self._contact_width_m is not None
            max_step = self.max_speed_m_s * dt
            delta = float(
                np.clip(self._contact_width_m - self._command_m, -max_step, max_step)
            )
            self._command_m = float(
                np.clip(self._command_m + delta, self.closed_m, self.open_m)
            )
            status = (
                "gripper_contact_hold"
                if abs(self._contact_width_m - self._command_m) <= 1e-6
                else "gripper_contact_preload"
            )
            return GripperCommand(
                self._normalize(self._command_m),
                self._command_m,
                status,
            )

        if contact_latch_enabled and closing:
            self._contact_history.append(abs(measured_force_n) >= self.contact_force_n)
            if sum(self._contact_history) >= self.contact_confirmations:
                self._contact_latched = True
                self._contact_width_m = max(
                    self.closed_m,
                    measured - self.contact_preload_m,
                )
                status = (
                    "gripper_contact_hold"
                    if abs(self._contact_width_m - self._command_m) <= 1e-6
                    else "gripper_contact_preload"
                )
                return GripperCommand(
                    self._normalize(self._command_m),
                    self._command_m,
                    status,
                )
        else:
            self._contact_history.clear()

        if (
            self.force_hold_enabled
            and closing
            and abs(measured_force_n) >= self.contact_force_n
        ):
            self._command_m = measured
            return GripperCommand(self._normalize(measured), measured, "gripper_contact_hold")

        max_step = self.max_speed_m_s * dt
        delta = float(np.clip(target_m - self._command_m, -max_step, max_step))
        command = float(np.clip(self._command_m + delta, self.closed_m, self.open_m))
        self._command_m = command
        status = "gripper_tracking" if abs(target_m - command) > 0.0005 else "gripper_at_target"
        return GripperCommand(self._normalize(command), command, status)

    def _hold(self, status: str) -> GripperCommand:
        assert self._command_m is not None
        return GripperCommand(self._normalize(self._command_m), self._command_m, status)

    def _normalize(self, width_m: float) -> float:
        return float(np.clip((width_m - self.closed_m) / (self.open_m - self.closed_m), 0.0, 1.0))

    def _validate_width(self, width_m: float) -> float:
        width = float(width_m)
        if not np.isfinite(width) or not (
            self.closed_m - self.feedback_tolerance_m
            <= width
            <= self.open_m + self.feedback_tolerance_m
        ):
            raise ValueError(f"gripper feedback is outside calibrated range: {width_m!r}")
        return float(np.clip(width, self.closed_m, self.open_m))
