"""Task-terminal arm target smoothing for close-range manipulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_COUNT = 7


@dataclass(frozen=True)
class TerminalFilterSample:
    target: np.ndarray | None
    status: str
    active: bool
    alpha: float | None
    stage: str


class TerminalArmTargetFilter:
    """Low-pass arm targets only after a confirmed close-range grasp intent."""

    def __init__(
        self,
        *,
        enabled: bool,
        approach_flange_height_m: float,
        approach_confirm_sec: float,
        approach_time_constant_sec: float,
        gripper_threshold: float,
        flange_height_m: float,
        confirm_sec: float,
        time_constant_sec: float,
        gripper_hysteresis: float = 0.55,
        approach_height_hysteresis_m: float = 0.04,
        height_hysteresis_m: float = 0.08,
    ) -> None:
        if not 0.0 <= gripper_threshold <= 1.0:
            raise ValueError("gripper_threshold must be in [0, 1]")
        if approach_flange_height_m <= 0 or flange_height_m <= 0:
            raise ValueError("filter flange heights must be positive")
        if approach_flange_height_m > flange_height_m:
            raise ValueError("approach flange height must not exceed terminal flange height")
        if (
            approach_confirm_sec < 0
            or confirm_sec < 0
            or approach_time_constant_sec <= 0
            or time_constant_sec <= 0
        ):
            raise ValueError("confirmation must be non-negative and time constant positive")
        if (
            gripper_hysteresis < 0
            or approach_height_hysteresis_m < 0
            or height_hysteresis_m < 0
        ):
            raise ValueError("terminal filter hysteresis values must be non-negative")
        self.enabled = bool(enabled)
        self.approach_flange_height_m = float(approach_flange_height_m)
        self.approach_confirm_sec = float(approach_confirm_sec)
        self.approach_time_constant_sec = float(approach_time_constant_sec)
        self.gripper_threshold = float(gripper_threshold)
        self.flange_height_m = float(flange_height_m)
        self.confirm_sec = float(confirm_sec)
        self.time_constant_sec = float(time_constant_sec)
        self.gripper_hysteresis = float(gripper_hysteresis)
        self.approach_height_hysteresis_m = float(approach_height_hysteresis_m)
        self.height_hysteresis_m = float(height_hysteresis_m)
        self._approach_candidate_since: float | None = None
        self._terminal_candidate_since: float | None = None
        self._stage = "inactive"
        self._filtered: np.ndarray | None = None
        self._last_update: float | None = None
        self.activation_count = 0
        self.approach_activation_count = 0
        self.terminal_activation_count = 0

    @property
    def active(self) -> bool:
        return self._stage != "inactive"

    @property
    def stage(self) -> str:
        return self._stage

    def update(
        self,
        target: np.ndarray | None,
        *,
        gripper_target: float,
        measured_flange_height_m: float,
        now: float,
        bypass: bool = False,
    ) -> TerminalFilterSample:
        if target is None:
            self._reset()
            return TerminalFilterSample(None, "no_target", False, None, "inactive")
        raw = np.asarray(target, dtype=np.float64)
        if raw.shape != (JOINT_COUNT,) or not np.isfinite(raw).all():
            raise ValueError("terminal arm target must contain seven finite joints")
        if not np.isfinite([gripper_target, measured_flange_height_m, now]).all():
            raise ValueError("terminal filter inputs must be finite")

        if not self.enabled or bypass:
            self._track_raw(raw, now)
            self._clear_candidates()
            self._stage = "inactive"
            status = "bypassed" if bypass else "disabled"
            return TerminalFilterSample(raw.copy(), status, False, None, "inactive")

        approach_enter = measured_flange_height_m <= self.approach_flange_height_m
        approach_remain = (
            measured_flange_height_m
            <= self.approach_flange_height_m + self.approach_height_hysteresis_m
        )
        terminal_enter = (
            gripper_target <= self.gripper_threshold
            and measured_flange_height_m <= self.flange_height_m
        )
        terminal_remain = (
            gripper_target <= self.gripper_threshold + self.gripper_hysteresis
            and measured_flange_height_m
            <= self.flange_height_m + self.height_hysteresis_m
        )

        self._approach_candidate_since = self._update_candidate(
            self._approach_candidate_since,
            approach_enter,
            now,
        )
        self._terminal_candidate_since = self._update_candidate(
            self._terminal_candidate_since,
            terminal_enter,
            now,
        )
        approach_confirmed = self._confirmed(
            self._approach_candidate_since,
            self.approach_confirm_sec,
            now,
        )
        terminal_confirmed = self._confirmed(
            self._terminal_candidate_since,
            self.confirm_sec,
            now,
        )

        previous_stage = self._stage
        if previous_stage == "terminal" and terminal_remain:
            next_stage = "terminal"
        elif terminal_confirmed:
            next_stage = "terminal"
        elif previous_stage in {"approach", "terminal"} and approach_remain:
            next_stage = "approach"
        elif approach_confirmed:
            next_stage = "approach"
        else:
            next_stage = "inactive"
        self._stage = next_stage

        if next_stage != previous_stage and next_stage != "inactive":
            self.activation_count += 1
            if next_stage == "approach":
                self.approach_activation_count += 1
            else:
                self.terminal_activation_count += 1

        if next_stage == "inactive":
            self._track_raw(raw, now)
            if self._approach_candidate_since is not None:
                status = "approach_arming"
            elif self._terminal_candidate_since is not None:
                status = "terminal_arming"
            else:
                status = "inactive"
            return TerminalFilterSample(raw.copy(), status, False, None, "inactive")

        if self._filtered is None or self._last_update is None:
            self._track_raw(raw, now)
            return TerminalFilterSample(
                raw.copy(),
                f"{next_stage}_filtering",
                True,
                1.0,
                next_stage,
            )
        dt = float(now - self._last_update)
        if dt <= 0:
            raise ValueError("terminal filter timestamps must be strictly increasing")
        time_constant = (
            self.time_constant_sec
            if next_stage == "terminal"
            else self.approach_time_constant_sec
        )
        alpha = float(np.clip(dt / (time_constant + dt), 0.0, 1.0))
        self._filtered = self._filtered + alpha * (raw - self._filtered)
        self._last_update = float(now)
        return TerminalFilterSample(
            self._filtered.copy(),
            f"{next_stage}_filtering",
            True,
            alpha,
            next_stage,
        )

    @staticmethod
    def _update_candidate(
        candidate_since: float | None,
        condition: bool,
        now: float,
    ) -> float | None:
        if not condition:
            return None
        return float(now) if candidate_since is None else candidate_since

    @staticmethod
    def _confirmed(candidate_since: float | None, confirm_sec: float, now: float) -> bool:
        return candidate_since is not None and now - candidate_since >= confirm_sec

    def _track_raw(self, target: np.ndarray, now: float) -> None:
        self._filtered = target.copy()
        self._last_update = float(now)

    def _reset(self) -> None:
        self._clear_candidates()
        self._stage = "inactive"
        self._filtered = None
        self._last_update = None

    def _clear_candidates(self) -> None:
        self._approach_candidate_since = None
        self._terminal_candidate_since = None
