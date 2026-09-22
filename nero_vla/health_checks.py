"""NERO 实机控制启动与运行期间的反馈健康检查。"""

from __future__ import annotations

import time

import numpy as np


def wait_complete_joint_feedback(robot, timeout_sec: float = 5.0) -> np.ndarray:
    """运动前要求连续、完整且稳定的 J1-J7 反馈。"""
    deadline = time.monotonic() + timeout_sec
    samples: list[np.ndarray] = []
    timestamps: set[float] = set()
    while time.monotonic() < deadline:
        feedback = robot.get_joint_angles()
        if feedback is not None and feedback.timestamp not in timestamps:
            value = np.asarray(feedback.msg, dtype=np.float64)
            if value.shape == (7,) and np.isfinite(value).all():
                timestamps.add(feedback.timestamp)
                samples.append(value.copy())
                if len(samples) >= 8:
                    stacked = np.stack(samples[-8:])
                    if float(np.max(np.ptp(np.rad2deg(stacked), axis=0))) <= 0.25:
                        return stacked[-1]
        time.sleep(0.01)
    raise RuntimeError("No stable, complete seven-joint CAN feedback snapshot")


def wait_enabled(robot, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if robot.enable():
            return
        time.sleep(0.03)
    raise RuntimeError("All seven joints did not report enabled")


def wait_cpv_mode(robot, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if (
            status is not None
            and int(status.msg.ctrl_mode) == 0x01
            and int(status.msg.mode_feedback) == 0x05
            and int(status.msg.arm_status) == 0x00
        ):
            return
        time.sleep(0.01)
    raise RuntimeError("CAN/CPV mode was not confirmed by arm status feedback")


def check_driver_health(robot) -> None:
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "driver_error_status",
        "stall_status",
    )
    for joint_index in range(1, 8):
        state = robot.get_driver_states(joint_index)
        if state is None:
            raise RuntimeError(f"Missing driver state for joint {joint_index}")
        flags = state.msg.foc_status
        active = [name for name in fault_names if bool(getattr(flags, name, False))]
        if active:
            raise RuntimeError(f"Joint {joint_index} driver fault: {active}")


def check_gripper_health(gripper_driver) -> None:
    state = gripper_driver.get_gripper_status()
    if state is None:
        raise RuntimeError("Missing gripper status")
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "sensor_status",
        "driver_error_status",
    )
    flags = state.msg.foc_status
    active = [name for name in fault_names if bool(getattr(flags, name, False))]
    if active:
        raise RuntimeError(f"Gripper health fault: {active}")
