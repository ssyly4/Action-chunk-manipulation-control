from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_path import ReferencePath
from .retimer import FixedPathRetimer, RetimeResult


@dataclass(frozen=True)
class MotionState:
    wall_tick: int
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    phase: float
    phase_speed: float
    phase_acceleration: float


@dataclass(frozen=True)
class OverlapAlignment:
    phase: float
    phase_speed: float
    phase_acceleration: float
    position_error: float
    velocity_error: float
    acceleration_error: float
    score: float


@dataclass(frozen=True)
class RollingConfig:
    commit_ticks: int = 12
    overlap_back_steps: float = 2.0
    overlap_forward_steps: float = 5.0
    overlap_samples_per_step: int = 24
    position_scale: float = np.deg2rad(1.0)
    velocity_scale: float = np.deg2rad(8.0)
    acceleration_scale: float = np.deg2rad(80.0)
    minimum_terminal_phase_speed: float = 1.0
    unobserved_boundary_speed_fraction: float = 0.5
    max_splice_position_error: float = np.deg2rad(2.5)
    max_splice_velocity_error: float = np.deg2rad(32.0)
    max_splice_acceleration_error: float = np.deg2rad(400.0)

    def __post_init__(self) -> None:
        if self.commit_ticks < 1:
            raise ValueError("commit_ticks must be positive")
        if self.overlap_samples_per_step < 4:
            raise ValueError("overlap sampling is too sparse")
        if min(self.position_scale, self.velocity_scale, self.acceleration_scale) <= 0:
            raise ValueError("overlap score scales must be positive")
        if not 0.0 < self.unobserved_boundary_speed_fraction <= 1.0:
            raise ValueError("unobserved boundary speed fraction must be in (0, 1]")


@dataclass(frozen=True)
class SpliceCandidate:
    wall_tick: int
    old_local_tick: int
    new_local_tick: int
    position_error: float
    velocity_error: float
    acceleration_error: float
    score: float
    accepted: bool


@dataclass(frozen=True)
class RollingPlan:
    start_wall_tick: int
    commit_ticks: int
    actions: np.ndarray
    reference: ReferencePath
    retiming: RetimeResult
    alignment: OverlapAlignment | None
    requested_start_phase_speed: float
    requested_end_phase_speed: float
    inherited_phase_acceleration: float

    @property
    def committed_commands(self) -> np.ndarray:
        return self.retiming.commands[: self.commit_ticks].copy()

    @property
    def lookahead_commands(self) -> np.ndarray:
        return self.retiming.commands[self.commit_ticks :].copy()

    @property
    def handoff_wall_tick(self) -> int:
        return self.start_wall_tick + self.commit_ticks

    @property
    def optimization_end_wall_tick(self) -> int:
        return self.start_wall_tick + len(self.retiming.commands)

    def motion_state_at(self, local_tick: int) -> MotionState:
        if not 0 <= local_tick < len(self.retiming.commands):
            raise IndexError("local tick is outside rolling plan")
        phase = float(self.retiming.phase_samples[local_tick])
        phase_speed = float(self.retiming.phase_speed_samples[local_tick])
        phase_acceleration = float(self.retiming.phase_acceleration_samples[local_tick])
        q = self.reference.evaluate(phase)
        qs = self.reference.evaluate(phase, 1)
        qss = self.reference.evaluate(phase, 2)
        qd = qs * phase_speed
        qdd = qss * phase_speed**2 + qs * phase_acceleration
        return MotionState(
            wall_tick=self.start_wall_tick + local_tick,
            q=q,
            qd=qd,
            qdd=qdd,
            phase=phase,
            phase_speed=phase_speed,
            phase_acceleration=phase_acceleration,
        )

    def handoff_state(self) -> MotionState:
        if self.commit_ticks >= len(self.retiming.commands):
            raise RuntimeError("rolling plan has no lookahead state at its handoff tick")
        return self.motion_state_at(self.commit_ticks)

    def commands_from_splice(self, splice: SpliceCandidate, commit_ticks: int) -> np.ndarray:
        if not splice.accepted:
            raise RuntimeError("cannot commit a rejected overlap splice")
        if splice.wall_tick != self.start_wall_tick + splice.new_local_tick:
            raise ValueError("splice does not belong to this rolling plan")
        end = min(len(self.retiming.commands), splice.new_local_tick + commit_ticks)
        return self.retiming.commands[splice.new_local_tick:end].copy()


class RollingFixedPathRetimer:
    """Rolling TOPPRAsd planning where optimization ends but motion does not."""

    def __init__(self, retimer: FixedPathRetimer, config: RollingConfig) -> None:
        self.retimer = retimer
        self.config = config

    def align_overlap(
        self,
        reference: ReferencePath,
        inherited: MotionState,
        *,
        expected_phase: float,
    ) -> OverlapAlignment:
        lower = max(0.0, expected_phase - self.config.overlap_back_steps)
        upper = min(reference.horizon - 1.0, expected_phase + self.config.overlap_forward_steps)
        count = max(
            5,
            int(np.ceil((upper - lower) * self.config.overlap_samples_per_step)) + 1,
        )
        best: OverlapAlignment | None = None
        for phase in np.linspace(lower, upper, count):
            q = reference.evaluate(phase)
            qs = reference.evaluate(phase, 1)
            qss = reference.evaluate(phase, 2)
            tangent_norm = float(np.dot(qs, qs))
            if tangent_norm <= 1e-14:
                continue
            phase_speed = max(0.0, float(np.dot(qs, inherited.qd) / tangent_norm))
            residual_acceleration = inherited.qdd - qss * phase_speed**2
            phase_acceleration = float(np.dot(qs, residual_acceleration) / tangent_norm)
            qd = qs * phase_speed
            qdd = qss * phase_speed**2 + qs * phase_acceleration
            position_error = float(np.linalg.norm(q - inherited.q))
            velocity_error = float(np.linalg.norm(qd - inherited.qd))
            acceleration_error = float(np.linalg.norm(qdd - inherited.qdd))
            score = (
                (position_error / self.config.position_scale) ** 2
                + (velocity_error / self.config.velocity_scale) ** 2
                + (acceleration_error / self.config.acceleration_scale) ** 2
            )
            candidate = OverlapAlignment(
                phase=float(phase),
                phase_speed=phase_speed,
                phase_acceleration=phase_acceleration,
                position_error=position_error,
                velocity_error=velocity_error,
                acceleration_error=acceleration_error,
                score=score,
            )
            if best is None or candidate.score < best.score:
                best = candidate
        if best is None:
            raise RuntimeError("new reference has no usable tangent in the overlap window")
        return best

    def plan(
        self,
        actions: np.ndarray,
        *,
        start_wall_tick: int,
        inherited: MotionState | None = None,
        expected_phase: float = 0.0,
        consumed_steps: int = 0,
    ) -> RollingPlan:
        values = np.asarray(actions, dtype=np.float64)
        reference, _ = self.retimer.build_reference(values)
        if not 0 <= consumed_steps < len(values) - 1:
            raise ValueError("consumed_steps must leave rolling lookahead")

        if inherited is None:
            alignment = None
            start_phase = float(consumed_steps)
            start_speed = self._nominal_feasible_phase_speed(reference, start_phase)
            inherited_phase_acceleration = 0.0
        else:
            if inherited.wall_tick != start_wall_tick:
                raise ValueError("inherited motion state must belong to the handoff wall tick")
            alignment = self.align_overlap(
                reference,
                inherited,
                expected_phase=expected_phase,
            )
            start_phase = alignment.phase
            start_speed = alignment.phase_speed
            inherited_phase_acceleration = alignment.phase_acceleration

        output_ticks = len(values) - consumed_steps
        if self.config.commit_ticks >= output_ticks:
            raise ValueError("commit horizon must leave at least one lookahead tick")
        average_speed = (len(values) - 1.0 - start_phase) / (
            (output_ticks - 1) / self.retimer.config.action_hz
        )
        candidates = self._terminal_speed_candidates(start_speed, average_speed)
        selected: RetimeResult | None = None
        selected_end_speed = candidates[-1]
        for end_speed in candidates:
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
        return RollingPlan(
            start_wall_tick=int(start_wall_tick),
            commit_ticks=self.config.commit_ticks,
            actions=values.copy(),
            reference=reference,
            retiming=selected,
            alignment=alignment,
            requested_start_phase_speed=start_speed,
            requested_end_phase_speed=selected_end_speed,
            inherited_phase_acceleration=inherited_phase_acceleration,
        )

    def find_overlap_splice(
        self,
        old_plan: RollingPlan,
        new_plan: RollingPlan,
        *,
        earliest_wall_tick: int | None = None,
        latest_wall_tick: int | None = None,
    ) -> SpliceCandidate:
        """Find a future wall tick where q, qd and qdd agree.

        The caller keeps executing ``old_plan`` until this tick.  A rejected
        result means that the old lookahead remains authoritative; no terminal
        hold or forced switch is introduced.
        """
        first = max(old_plan.start_wall_tick, new_plan.start_wall_tick)
        last = min(
            old_plan.optimization_end_wall_tick - 1,
            new_plan.optimization_end_wall_tick - 1,
        )
        if earliest_wall_tick is not None:
            first = max(first, int(earliest_wall_tick))
        if latest_wall_tick is not None:
            last = min(last, int(latest_wall_tick))
        if first > last:
            raise RuntimeError("old and new rolling plans have no wall-clock overlap")

        best: SpliceCandidate | None = None
        for wall_tick in range(first, last + 1):
            old_index = wall_tick - old_plan.start_wall_tick
            new_index = wall_tick - new_plan.start_wall_tick
            old_state = old_plan.motion_state_at(old_index)
            new_state = new_plan.motion_state_at(new_index)
            position_error = float(np.max(np.abs(new_state.q - old_state.q)))
            velocity_error = float(np.max(np.abs(new_state.qd - old_state.qd)))
            acceleration_error = float(np.max(np.abs(new_state.qdd - old_state.qdd)))
            score = (
                (position_error / self.config.position_scale) ** 2
                + (velocity_error / self.config.velocity_scale) ** 2
                + (acceleration_error / self.config.acceleration_scale) ** 2
            )
            candidate = SpliceCandidate(
                wall_tick=wall_tick,
                old_local_tick=old_index,
                new_local_tick=new_index,
                position_error=position_error,
                velocity_error=velocity_error,
                acceleration_error=acceleration_error,
                score=float(score),
                accepted=bool(
                    position_error <= self.config.max_splice_position_error
                    and velocity_error <= self.config.max_splice_velocity_error
                    and acceleration_error <= self.config.max_splice_acceleration_error
                ),
            )
            if best is None or (candidate.accepted, -candidate.score) > (
                best.accepted,
                -best.score,
            ):
                best = candidate
        assert best is not None
        return best

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

    def _terminal_speed_candidates(self, start_speed: float, average_speed: float) -> tuple[float, ...]:
        floor = self.config.minimum_terminal_phase_speed
        values = (
            max(floor, average_speed),
            max(floor, 0.75 * average_speed + 0.25 * start_speed),
            max(floor, 0.5 * average_speed + 0.5 * start_speed),
            max(floor, start_speed),
            max(floor, 0.5 * average_speed),
        )
        unique = []
        for value in values:
            if not any(abs(value - old) < 1e-9 for old in unique):
                unique.append(float(value))
        return tuple(unique)
