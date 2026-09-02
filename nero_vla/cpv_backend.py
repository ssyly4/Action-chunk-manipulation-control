"""Thin, testable adapter for NERO CPV position streaming.

Connection, enable, health checks, and physical emergency-stop policy belong to
the supervising runner.  This adapter only guarantees a stable CPV command
sequence and refuses discontinuous writes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


JOINT_COUNT = 7


def _positions(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (JOINT_COUNT,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite and have shape ({JOINT_COUNT},)")
    return result


class NeroCpvPositionBackend:
    """Send seven position registers without repeating motion-mode commands."""

    def __init__(self, robot: Any, *, max_command_step_rad: float | np.ndarray) -> None:
        self.robot = robot
        limit = np.asarray(max_command_step_rad, dtype=np.float64)
        if limit.ndim == 0:
            limit = np.full(JOINT_COUNT, float(limit), dtype=np.float64)
        if limit.shape != (JOINT_COUNT,) or not np.isfinite(limit).all() or np.any(limit <= 0):
            raise ValueError("max_command_step_rad must contain seven positive finite limits")
        self.max_command_step = limit
        self._last_command: np.ndarray | None = None

    @property
    def prepared(self) -> bool:
        return self._last_command is not None

    def prepare_hold(self, complete_feedback: np.ndarray) -> None:
        """Preload a hold target, enter CPV once, then refresh that hold target."""
        if self.prepared:
            raise RuntimeError("CPV backend is already prepared")
        current = _positions(complete_feedback, "complete_feedback")

        # With automatic mode switching disabled, register writes do not emit a
        # mode frame on every joint command.  Preloading before the one explicit
        # transition prevents CPV from starting with implicit zero registers.
        self.robot.set_auto_set_motion_mode_enabled(False)
        self._send_all(current)
        self.robot.set_motion_mode("cpv")
        self._send_all(current)
        self._last_command = current.copy()

    def send(self, command: np.ndarray) -> None:
        if self._last_command is None:
            raise RuntimeError("prepare_hold must succeed before sending CPV commands")
        target = _positions(command, "command")
        step = np.abs(target - self._last_command)
        if np.any(step > self.max_command_step):
            raise ValueError(
                "CPV command step exceeds backend limit: "
                f"step_deg={np.rad2deg(step).tolist()}"
            )
        self._send_all(target)
        self._last_command = target.copy()

    def hold(self) -> None:
        if self._last_command is None:
            raise RuntimeError("prepare_hold must succeed before holding")
        self._send_all(self._last_command)

    def _send_all(self, positions: np.ndarray) -> None:
        for joint_index, position in enumerate(positions, start=1):
            self.robot.move_cpv_pos(joint_index, float(position))
