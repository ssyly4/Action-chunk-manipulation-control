from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_path import ReferencePath
from .retimer import FixedPathRetimer, RetimeResult


@dataclass(frozen=True)
class RecedingConfig:
    minimum_terminal_phase_speed: float = 1.0
    unobserved_boundary_speed_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.minimum_terminal_phase_speed < 0.0:
            raise ValueError("minimum terminal phase speed must be non-negative")
        if not 0.0 < self.unobserved_boundary_speed_fraction <= 1.0:
            raise ValueError("unobserved boundary speed fraction must be in (0, 1]")


@dataclass(frozen=True)
class RecedingMotionState:
    wall_tick: int
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    phase: float
    phase_speed: float
    phase_acceleration: float


@dataclass(frozen=True)
class RecedingPlan:
    start_wall_tick: int
    actions: np.ndarray
    reference: ReferencePath
    retiming: RetimeResult
    requested_start_phase_speed: float
    requested_end_phase_speed: float

    @property
    def optimization_end_wall_tick(self) -> int:
        return self.start_wall_tick + len(self.retiming.commands)

    def motion_state_at(self, local_tick: int) -> RecedingMotionState:
        if not 0 <= local_tick < len(self.retiming.commands):
            raise IndexError("local tick is outside receding plan")
        phase = float(self.retiming.phase_samples[local_tick])
        phase_speed = float(self.retiming.phase_speed_samples[local_tick])
        phase_acceleration = float(self.retiming.phase_acceleration_samples[local_tick])
        q = self.reference.evaluate(phase)
        qs = self.reference.evaluate(phase, 1)
        qss = self.reference.evaluate(phase, 2)
        qd = qs * phase_speed
        qdd = qss * phase_speed**2 + qs * phase_acceleration
        return RecedingMotionState(
            wall_tick=self.start_wall_tick + local_tick,
            q=q,
            qd=qd,
            qdd=qdd,
            phase=phase,
            phase_speed=phase_speed,
            phase_acceleration=phase_acceleration,
        )


class RecedingFixedPathRetimer:
    """Fixed-boundary TOPPRA planning without old/new trajectory splicing."""

    def __init__(self, retimer: FixedPathRetimer, config: RecedingConfig) -> None:
        self.retimer = retimer
        self.config = config

    def plan(
        self,
        actions: np.ndarray,
        *,
        start_wall_tick: int,
        consumed_steps: int = 0,
        start_arm_velocity: np.ndarray | None = None,
        start_speed_curvature_margin: float | None = None,
    ) -> RecedingPlan:
        values = np.asarray(actions, dtype=np.float64)
        reference, _ = self.retimer.build_reference(values)
        if not 0 <= consumed_steps < len(values) - 1:
            raise ValueError("consumed_steps must leave at least two trajectory rows")

        start_phase = float(consumed_steps)
        start_speed = (
            self._nominal_feasible_phase_speed(reference, start_phase)
            if start_arm_velocity is None
            else self._phase_speed_from_arm_velocity(
                reference,
                start_phase,
                start_arm_velocity,
                curvature_margin=start_speed_curvature_margin,
            )
        )
        output_ticks = len(values) - consumed_steps
        average_speed = (len(values) - 1.0 - start_phase) / (
            (output_ticks - 1) / self.retimer.config.action_hz
        )
        terminal_speeds = self._terminal_speed_candidates(start_speed, average_speed)
        selected: RetimeResult | None = None
        selected_end_speed = terminal_speeds[-1]
        for end_speed in terminal_speeds:
            candidate = self.retimer.retime(
                values,
                start_phase=start_phase,
                output_ticks=output_ticks,
                start_phase_speed=start_speed,
                end_phase_speed=end_speed,
                fallback_start_index=consumed_steps,
            )
            selected = candidate
            selected_end_speed = end_speed
            if candidate.status == "retimed":
                break
        assert selected is not None
        return RecedingPlan(
            start_wall_tick=int(start_wall_tick),
            actions=values.copy(),
            reference=reference,
            retiming=selected,
            requested_start_phase_speed=start_speed,
            requested_end_phase_speed=selected_end_speed,
        )

    def _phase_speed_from_arm_velocity(
        self,
        reference: ReferencePath,
        phase: float,
        arm_velocity: np.ndarray,
        *,
        curvature_margin: float | None = None,
    ) -> float:
        velocity = np.asarray(arm_velocity, dtype=np.float64)
        if velocity.shape != self.retimer.config.max_velocity.shape:
            raise ValueError("start_arm_velocity has the wrong shape")
        tangent = reference.evaluate(phase, 1)
        norm_squared = float(np.dot(tangent, tangent))
        if norm_squared <= 1e-12:
            return 0.0
        projected = max(0.0, float(np.dot(tangent, velocity) / norm_squared))
        nonzero = np.abs(tangent) > 1e-10
        feasible_limit = float(
            np.min(self.retimer.config.max_velocity[nonzero] / np.abs(tangent[nonzero]))
        )
        bounded = min(projected, 0.95 * feasible_limit)
        if curvature_margin is not None:
            if not 0.0 < curvature_margin <= 1.0:
                raise ValueError("curvature_margin must be in (0, 1]")
            curvature = np.abs(reference.evaluate(phase, 2))
            curved = curvature > 1e-10
            if np.any(curved):
                curvature_limit = float(
                    np.min(
                        np.sqrt(
                            self.retimer.config.max_acceleration[curved]
                            / curvature[curved]
                        )
                    )
                )
                bounded = min(bounded, curvature_margin * curvature_limit)
        return bounded

    def _nominal_feasible_phase_speed(self, reference: ReferencePath, phase: float) -> float:
        tangent = np.abs(reference.evaluate(phase, 1))
        nonzero = tangent > 1e-10
        if not np.any(nonzero):
            return self.config.minimum_terminal_phase_speed
        limit = float(np.min(self.retimer.config.max_velocity[nonzero] / tangent[nonzero]))
        return max(
            self.config.minimum_terminal_phase_speed,
            min(
                self.config.unobserved_boundary_speed_fraction
                * self.retimer.config.action_hz,
                0.8 * limit,
            ),
        )

    def _terminal_speed_candidates(
        self, start_speed: float, average_speed: float
    ) -> tuple[float, ...]:
        floor = self.config.minimum_terminal_phase_speed
        values = (
            max(floor, average_speed),
            max(floor, 0.75 * average_speed + 0.25 * start_speed),
            max(floor, 0.5 * average_speed + 0.5 * start_speed),
            max(floor, start_speed),
            max(floor, 0.5 * average_speed),
        )
        unique: list[float] = []
        for value in values:
            if not any(abs(value - old) < 1e-9 for old in unique):
                unique.append(float(value))
        return tuple(unique)
