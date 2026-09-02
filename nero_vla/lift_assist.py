"""Strictly gated Cartesian lift assistance for a confirmed physical grasp."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh
from pyAgxArm.utiles.mdh_kinematics import get_mdh


JOINT_COUNT = 7
NERO_MDH = list(get_mdh("nero"))


def rigid_contact_is_stable(
    *,
    measured_width_m: float,
    commanded_width_m: float,
    measured_force_n: float,
    force_threshold_n: float,
    minimum_width_gap_m: float,
) -> bool:
    values = [
        measured_width_m,
        commanded_width_m,
        measured_force_n,
        force_threshold_n,
        minimum_width_gap_m,
    ]
    if not np.isfinite(values).all():
        return False
    if force_threshold_n <= 0 or minimum_width_gap_m <= 0:
        raise ValueError("rigid contact thresholds must be positive")
    return (
        abs(measured_force_n) >= force_threshold_n
        and measured_width_m - commanded_width_m >= minimum_width_gap_m
    )


def flange_position_m(joint_rad: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_rad, dtype=np.float64)
    if joints.shape != (JOINT_COUNT,) or not np.isfinite(joints).all():
        raise ValueError("joint_rad must contain seven finite values")
    return np.asarray(fk_from_mdh(NERO_MDH, joints.tolist())[:3], dtype=np.float64)


def flange_pose(joint_rad: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_rad, dtype=np.float64)
    if joints.shape != (JOINT_COUNT,) or not np.isfinite(joints).all():
        raise ValueError("joint_rad must contain seven finite values")
    return np.asarray(fk_from_mdh(NERO_MDH, joints.tolist()), dtype=np.float64)


def _wrap_angles(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def numerical_pose_jacobian(
    joint_rad: np.ndarray,
    *,
    epsilon_rad: float = 1e-5,
) -> np.ndarray:
    joints = np.asarray(joint_rad, dtype=np.float64)
    if epsilon_rad <= 0:
        raise ValueError("epsilon_rad must be positive")
    jacobian = np.empty((6, JOINT_COUNT), dtype=np.float64)
    for index in range(JOINT_COUNT):
        offset = np.zeros(JOINT_COUNT, dtype=np.float64)
        offset[index] = epsilon_rad
        difference = flange_pose(joints + offset) - flange_pose(joints - offset)
        difference[3:] = _wrap_angles(difference[3:])
        jacobian[:, index] = difference / (2.0 * epsilon_rad)
    return jacobian


def damped_pose_step(
    joint_rad: np.ndarray,
    goal_pose: np.ndarray,
    *,
    damping: float = 0.01,
    max_cartesian_step_m: float = 0.0015,
    max_joint_step_rad: float = np.deg2rad(0.5),
) -> np.ndarray:
    """Return a nearby joint target for a bounded Cartesian position correction."""
    joints = np.asarray(joint_rad, dtype=np.float64)
    goal = np.asarray(goal_pose, dtype=np.float64)
    if goal.shape != (6,) or not np.isfinite(goal).all():
        raise ValueError("goal_pose must contain six finite values")
    if damping <= 0 or max_cartesian_step_m <= 0 or max_joint_step_rad <= 0:
        raise ValueError("DLS limits must be positive")

    error = goal - flange_pose(joints)
    error[3:] = _wrap_angles(error[3:])
    position_norm = float(np.linalg.norm(error[:3]))
    if position_norm > max_cartesian_step_m:
        error[:3] *= max_cartesian_step_m / position_norm
    orientation_limit = np.deg2rad(0.5)
    orientation_norm = float(np.linalg.norm(error[3:]))
    if orientation_norm > orientation_limit:
        error[3:] *= orientation_limit / orientation_norm

    jacobian = numerical_pose_jacobian(joints)
    weights = np.diag([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
    weighted_jacobian = weights @ jacobian
    weighted_error = weights @ error
    regularized = (
        weighted_jacobian @ weighted_jacobian.T + damping**2 * np.eye(6)
    )
    delta = weighted_jacobian.T @ np.linalg.solve(regularized, weighted_error)
    max_delta = float(np.max(np.abs(delta)))
    if max_delta > max_joint_step_rad:
        delta = delta * (max_joint_step_rad / max_delta)
    result = joints + delta
    if not np.isfinite(result).all():
        raise RuntimeError("Cartesian lift IK produced a non-finite joint target")
    return result


def bounded_pose_target(
    joint_rad: np.ndarray,
    goal_pose: np.ndarray,
    *,
    iterations: int = 4,
    joint_limits_rad: np.ndarray | None = None,
    joint_limit_margin_rad: float = np.deg2rad(0.25),
) -> np.ndarray:
    if iterations < 1 or iterations > 10:
        raise ValueError("iterations must be in [1, 10]")
    limits = None
    if joint_limits_rad is not None:
        limits = np.asarray(joint_limits_rad, dtype=np.float64)
        if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
            raise ValueError("joint_limits_rad must have shape (7, 2)")
        if joint_limit_margin_rad < 0:
            raise ValueError("joint_limit_margin_rad must be non-negative")
        if np.any(limits[:, 0] + joint_limit_margin_rad >= limits[:, 1] - joint_limit_margin_rad):
            raise ValueError("joint limit margin leaves no valid range")
    target = np.asarray(joint_rad, dtype=np.float64).copy()
    for _ in range(iterations):
        updated = damped_pose_step(target, goal_pose)
        if limits is not None:
            updated = np.clip(
                updated,
                limits[:, 0] + joint_limit_margin_rad,
                limits[:, 1] - joint_limit_margin_rad,
            )
        if float(np.max(np.abs(updated - target))) < 1e-8:
            break
        target = updated
    return target


@dataclass(frozen=True)
class LiftAssistStatus:
    state: str
    contact_count: int
    nominal_lift_m: float
    measured_lift_m: float


@dataclass(frozen=True)
class PreGraspDescentStatus:
    state: str
    confirmation_count: int
    measured_descent_m: float


@dataclass(frozen=True)
class PostReleaseHeightStatus:
    state: str
    close_count: int
    open_count: int
    measured_height_m: float
    overriding_policy: bool
    forward_extension_m: float


class PostReleaseHeightGuard:
    """Recover and preserve right-TCP height after a grasp/release cycle."""

    def __init__(
        self,
        *,
        enabled: bool,
        floor_height_m: float = 0.195,
        recovery_height_m: float = 0.210,
        close_threshold: float = 0.25,
        open_threshold: float = 0.75,
        confirmations: int = 3,
        tolerance_m: float = 0.003,
        timeout_sec: float = 6.0,
        joint_limits_rad: np.ndarray | None = None,
        forward_extension_m: float = 0.0,
        extension_ramp_sec: float = 1.0,
        lock_height: bool = False,
    ) -> None:
        if floor_height_m <= 0 or recovery_height_m <= floor_height_m:
            raise ValueError("recovery height must be above a positive floor height")
        if recovery_height_m - floor_height_m > 0.05:
            raise ValueError("post-release recovery band must not exceed 50 mm")
        if not 0.0 < close_threshold < open_threshold < 1.0:
            raise ValueError("post-release gripper thresholds are invalid")
        if confirmations < 1 or tolerance_m <= 0 or timeout_sec <= 0:
            raise ValueError("post-release confirmations, tolerance and timeout must be positive")
        if not 0.0 <= forward_extension_m <= 0.04 or extension_ramp_sec <= 0:
            raise ValueError("post-release forward extension or ramp is invalid")
        if not lock_height and floor_height_m < 0.145:
            raise ValueError(
                "unlocked post-release descent requires a TCP floor of at least 145 mm"
            )
        self.enabled = bool(enabled)
        self.floor_height_m = float(floor_height_m)
        self.recovery_height_m = float(recovery_height_m)
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.confirmations = int(confirmations)
        self.tolerance_m = float(tolerance_m)
        self.timeout_sec = float(timeout_sec)
        self.forward_extension_m = float(forward_extension_m)
        self.extension_ramp_sec = float(extension_ramp_sec)
        self.lock_height = bool(lock_height)
        self.joint_limits_rad = (
            None
            if joint_limits_rad is None
            else np.asarray(joint_limits_rad, dtype=np.float64).copy()
        )
        if self.joint_limits_rad is not None and self.joint_limits_rad.shape != (7, 2):
            raise ValueError("joint_limits_rad must have shape (7, 2)")
        self._state = "waiting_for_close" if self.enabled else "disabled"
        self._close_count = 0
        self._open_count = 0
        self._started_at: float | None = None
        self._recovery_goal_pose: np.ndarray | None = None
        self._overriding_policy = False
        self._guarding_started_at: float | None = None
        self._applied_forward_extension_m = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def overriding_policy(self) -> bool:
        return self._overriding_policy

    def observe(
        self,
        *,
        now: float,
        joint_rad: np.ndarray,
        measured_gripper_state: float,
    ) -> bool:
        """Return True only when a confirmed reopen activates the guard."""
        if not self.enabled or self._state in {"recovering", "guarding"}:
            return False
        if not np.isfinite([now, measured_gripper_state]).all():
            self._close_count = 0
            self._open_count = 0
            return False
        if self._state == "waiting_for_close":
            self._close_count = (
                self._close_count + 1
                if measured_gripper_state <= self.close_threshold
                else 0
            )
            if self._close_count >= self.confirmations:
                self._state = "waiting_for_open"
            return False

        self._open_count = (
            self._open_count + 1
            if measured_gripper_state >= self.open_threshold
            else 0
        )
        if self._open_count < self.confirmations:
            return False
        pose = flange_pose(joint_rad)
        self._started_at = float(now)
        # Keep a reachable pose anchor for both recovery and the subsequent
        # finishing sweep.  Policy XY/orientation can become unreachable near
        # a joint limit, whereas this pose is measured from the live arm.
        self._recovery_goal_pose = pose.copy()
        self._recovery_goal_pose[2] = self.recovery_height_m
        # The finishing sweep is defined from the recovered working height,
        # not merely from the collision floor.  Even a release between the
        # floor and recovery height must first return to the working plane.
        if pose[2] < self.recovery_height_m:
            self._state = "recovering"
        else:
            self._state = "guarding"
            self._guarding_started_at = float(now)
        return True

    def arm_target(
        self,
        *,
        now: float,
        measured_joint_rad: np.ndarray,
        policy_joint_rad: np.ndarray,
    ) -> np.ndarray | None:
        self._overriding_policy = False
        if self._state in {"disabled", "waiting_for_close", "waiting_for_open"}:
            return None
        measured_pose = flange_pose(measured_joint_rad)
        if (
            self._state == "guarding"
            and measured_pose[2] < self.floor_height_m - self.tolerance_m
        ):
            # A policy-derived IK target may become unreachable after the
            # sweep changes the arm configuration.  Never keep forwarding a
            # low target: return to the measured, reachable recovery anchor.
            self._state = "recovering"
            self._started_at = float(now)
            self._applied_forward_extension_m = 0.0
        if self._state == "recovering":
            assert self._started_at is not None
            assert self._recovery_goal_pose is not None
            if measured_pose[2] >= self.recovery_height_m - self.tolerance_m:
                self._state = "guarding"
                self._guarding_started_at = float(now)
                self._overriding_policy = True
                return np.asarray(measured_joint_rad, dtype=np.float64).copy()
            elif now - self._started_at > self.timeout_sec:
                raise RuntimeError("Post-release TCP height recovery timed out")
            else:
                self._overriding_policy = True
                return bounded_pose_target(
                    measured_joint_rad,
                    self._recovery_goal_pose,
                    iterations=6,
                    joint_limits_rad=self.joint_limits_rad,
                )

        policy_pose = flange_pose(policy_joint_rad)
        guarded_goal = policy_pose.copy()
        assert self._guarding_started_at is not None
        ramp = float(np.clip((now - self._guarding_started_at) / self.extension_ramp_sec, 0.0, 1.0))
        self._applied_forward_extension_m = self.forward_extension_m * ramp
        guarded_goal[0] -= self._applied_forward_extension_m
        guarded_goal[2] = (
            self.recovery_height_m
            if self.lock_height
            else max(float(guarded_goal[2]), self.floor_height_m)
        )
        if (
            self._applied_forward_extension_m <= 1e-9
            and not self.lock_height
            and policy_pose[2] >= self.floor_height_m
        ):
            return np.asarray(policy_joint_rad, dtype=np.float64).copy()
        self._overriding_policy = True
        target = bounded_pose_target(
            measured_joint_rad,
            guarded_goal,
            iterations=10,
            joint_limits_rad=self.joint_limits_rad,
        )
        if flange_pose(target)[2] < self.floor_height_m - self.tolerance_m:
            # The guarded policy pose cannot satisfy the height constraint
            # within joint limits.  Restart a recovery step from the safe
            # anchor rather than emitting a command that descends further.
            self._state = "recovering"
            self._started_at = float(now)
            self._applied_forward_extension_m = 0.0
            assert self._recovery_goal_pose is not None
            return bounded_pose_target(
                measured_joint_rad,
                self._recovery_goal_pose,
                iterations=6,
                joint_limits_rad=self.joint_limits_rad,
            )
        return target

    def status(self, measured_joint_rad: np.ndarray) -> PostReleaseHeightStatus:
        return PostReleaseHeightStatus(
            state=self._state,
            close_count=self._close_count,
            open_count=self._open_count,
            measured_height_m=float(flange_position_m(measured_joint_rad)[2]),
            overriding_policy=self._overriding_policy,
            forward_extension_m=self._applied_forward_extension_m,
        )


class PreGraspDescentAssist:
    """Hold an open gripper while making one bounded downward correction."""

    def __init__(
        self,
        *,
        enabled: bool,
        descent_distance_m: float = 0.005,
        close_threshold: float = 0.5,
        confirmations: int = 3,
        release_gripper_threshold: float = 0.25,
        timeout_sec: float = 2.0,
        tolerance_m: float = 0.001,
    ) -> None:
        if descent_distance_m <= 0 or descent_distance_m > 0.01:
            raise ValueError("descent_distance_m must be in (0, 0.01]")
        if not 0.0 < close_threshold < 1.0:
            raise ValueError("close_threshold must be in (0, 1)")
        if not 0.0 < release_gripper_threshold < close_threshold:
            raise ValueError("release gripper threshold must be below close threshold")
        if confirmations < 1 or timeout_sec <= 0 or tolerance_m <= 0:
            raise ValueError("confirmations, timeout and tolerance must be positive")
        self.enabled = bool(enabled)
        self.descent_distance_m = float(descent_distance_m)
        self.close_threshold = float(close_threshold)
        self.confirmations = int(confirmations)
        self.release_gripper_threshold = float(release_gripper_threshold)
        self.timeout_sec = float(timeout_sec)
        self.tolerance_m = float(tolerance_m)
        self._state = "idle" if self.enabled else "disabled"
        self._confirmation_count = 0
        self._started_at: float | None = None
        self._anchor_pose: np.ndarray | None = None
        self._goal_pose: np.ndarray | None = None
        self._hold_gripper_target: float | None = None

    @property
    def active(self) -> bool:
        return self._state in {"descending", "closing_hold"}

    @property
    def state(self) -> str:
        return self._state

    def observe(
        self,
        *,
        now: float,
        joint_rad: np.ndarray,
        measured_gripper_state: float,
        policy_gripper_target: float,
    ) -> bool:
        """Return True only when a confirmed close intent starts the correction."""
        if not self.enabled or self._state == "completed":
            return False
        values = [now, measured_gripper_state, policy_gripper_target]
        if not np.isfinite(values).all():
            self._confirmation_count = 0
            return False
        if self._state == "closing_hold":
            if measured_gripper_state <= self.release_gripper_threshold:
                self._state = "completed"
            return False
        if self._state == "descending":
            return False
        qualified = (
            measured_gripper_state >= 0.25
            and policy_gripper_target <= self.close_threshold
        )
        self._confirmation_count = self._confirmation_count + 1 if qualified else 0
        if self._confirmation_count < self.confirmations:
            return False

        self._started_at = float(now)
        self._anchor_pose = flange_pose(joint_rad)
        self._goal_pose = self._anchor_pose.copy()
        self._goal_pose[2] -= self.descent_distance_m
        self._hold_gripper_target = float(measured_gripper_state)
        self._state = "descending"
        return True

    def gripper_target(self, raw_policy_target: float) -> float:
        if self._state != "descending":
            return float(raw_policy_target)
        assert self._hold_gripper_target is not None
        return self._hold_gripper_target

    def arm_target(self, *, now: float, measured_joint_rad: np.ndarray) -> np.ndarray | None:
        if not self.active:
            return None
        assert self._started_at is not None
        assert self._goal_pose is not None
        if now - self._started_at > self.timeout_sec:
            if self._state == "descending":
                raise RuntimeError("Pre-grasp descent did not reach its bounded target")
            self._state = "completed"
            return None
        current_pose = flange_pose(measured_joint_rad)
        if (
            self._state == "descending"
            and float(np.linalg.norm(current_pose[:3] - self._goal_pose[:3]))
            <= self.tolerance_m
        ):
            self._state = "closing_hold"
        target = bounded_pose_target(measured_joint_rad, self._goal_pose, iterations=4)
        if (
            self._state == "descending"
            and flange_pose(target)[2] > current_pose[2] + 1e-5
        ):
            raise RuntimeError("Pre-grasp descent IK moved upward")
        return target

    def status(self, measured_joint_rad: np.ndarray) -> PreGraspDescentStatus:
        measured_descent = 0.0
        if self._anchor_pose is not None:
            measured_descent = float(
                self._anchor_pose[2] - flange_position_m(measured_joint_rad)[2]
            )
        return PreGraspDescentStatus(
            state=self._state,
            confirmation_count=self._confirmation_count,
            measured_descent_m=measured_descent,
        )


class PostGraspLiftAssist:
    """Latch only a force-confirmed close, then hold XY and raise the flange."""

    def __init__(
        self,
        *,
        enabled: bool,
        gripper_state_threshold: float = 0.32,
        contact_force_threshold_n: float = 0.55,
        contact_confirmations: int = 3,
        settle_sec: float = 0.25,
        lift_distance_m: float = 0.05,
        lift_speed_m_s: float = 0.02,
    ) -> None:
        if not 0.0 < gripper_state_threshold < 1.0:
            raise ValueError("gripper_state_threshold must be in (0, 1)")
        if contact_force_threshold_n <= 0:
            raise ValueError("contact_force_threshold_n must be positive")
        if contact_confirmations < 1:
            raise ValueError("contact_confirmations must be positive")
        if settle_sec < 0 or lift_distance_m <= 0 or lift_speed_m_s <= 0:
            raise ValueError("lift timing and distance must be positive")
        self.enabled = bool(enabled)
        self.gripper_state_threshold = float(gripper_state_threshold)
        self.contact_force_threshold_n = float(contact_force_threshold_n)
        self.contact_confirmations = int(contact_confirmations)
        self.settle_sec = float(settle_sec)
        self.lift_distance_m = float(lift_distance_m)
        self.lift_speed_m_s = float(lift_speed_m_s)
        self._state = "idle" if self.enabled else "disabled"
        self._contact_count = 0
        self._triggered_at: float | None = None
        self._anchor_pose: np.ndarray | None = None
        self._hold_gripper_target: float | None = None

    @property
    def active(self) -> bool:
        return self._triggered_at is not None

    @property
    def state(self) -> str:
        return self._state

    def observe(
        self,
        *,
        now: float,
        joint_rad: np.ndarray,
        measured_gripper_state: float,
        policy_gripper_target: float,
        measured_force_n: float,
    ) -> bool:
        """Return True only on the tick where a confirmed grasp is latched."""
        if not self.enabled or self.active:
            return False
        values = [now, measured_gripper_state, policy_gripper_target, measured_force_n]
        if not np.isfinite(values).all():
            self._contact_count = 0
            return False
        qualified = (
            measured_gripper_state <= self.gripper_state_threshold
            and policy_gripper_target <= self.gripper_state_threshold
            and abs(measured_force_n) >= self.contact_force_threshold_n
        )
        self._contact_count = self._contact_count + 1 if qualified else 0
        if self._contact_count < self.contact_confirmations:
            return False

        self._triggered_at = float(now)
        self._anchor_pose = flange_pose(joint_rad)
        self._hold_gripper_target = float(
            min(measured_gripper_state, policy_gripper_target)
        )
        self._state = "settling"
        return True

    def gripper_target(self, raw_policy_target: float) -> float:
        if not self.active:
            return float(raw_policy_target)
        assert self._hold_gripper_target is not None
        return float(min(raw_policy_target, self._hold_gripper_target))

    def arm_target(self, *, now: float, measured_joint_rad: np.ndarray) -> np.ndarray | None:
        if not self.active:
            return None
        assert self._triggered_at is not None
        assert self._anchor_pose is not None
        elapsed = max(0.0, float(now - self._triggered_at))
        lift_elapsed = max(0.0, elapsed - self.settle_sec)
        nominal_lift = min(self.lift_distance_m, lift_elapsed * self.lift_speed_m_s)
        self._state = (
            "settling"
            if elapsed < self.settle_sec
            else "lifting"
            if nominal_lift < self.lift_distance_m
            else "holding"
        )
        goal = self._anchor_pose.copy()
        goal[2] += nominal_lift
        return bounded_pose_target(measured_joint_rad, goal)

    def status(self, measured_joint_rad: np.ndarray) -> LiftAssistStatus:
        measured_lift = 0.0
        nominal_lift = 0.0
        if self.active:
            assert self._anchor_pose is not None
            measured_lift = float(
                flange_position_m(measured_joint_rad)[2] - self._anchor_pose[2]
            )
            if self._state == "holding":
                nominal_lift = self.lift_distance_m
        return LiftAssistStatus(
            state=self._state,
            contact_count=self._contact_count,
            nominal_lift_m=nominal_lift,
            measured_lift_m=measured_lift,
        )

    def reached_target(
        self,
        measured_joint_rad: np.ndarray,
        *,
        tolerance_m: float = 0.003,
    ) -> bool:
        if tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive")
        if self._state != "holding":
            return False
        status = self.status(measured_joint_rad)
        return status.measured_lift_m >= self.lift_distance_m - tolerance_m
