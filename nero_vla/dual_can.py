"""NERO dual-CAN roles, validation, and bridge control helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import struct
import time
from typing import Iterable


LEADER_CAN_PORT = os.environ.get("NERO_LEADER_CAN", "can0")
FOLLOWER_CAN_PORT = os.environ.get("NERO_FOLLOWER_CAN", "can1")
BRIDGE_SOCKET_PATH = os.environ.get(
    "NERO_CAN_BRIDGE_SOCKET", "/tmp/nero_leader_follower_bridge.sock"
)

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_FRAME = struct.Struct("=IB3x8s")

LEADER_JOINT_IDS = frozenset({0x155, 0x156, 0x157, 0x170})
LEADER_FORWARD_IDS = frozenset({0x151, 0x155, 0x156, 0x157, 0x159, 0x170})
FOLLOWER_JOINT_IDS = frozenset(range(0x251, 0x258))


class CanRoleError(RuntimeError):
    """Raised when a CAN interface does not have the required NERO role."""


@dataclass(frozen=True)
class CanTraffic:
    interface: str
    ids: frozenset[int]
    frames: int
    duration_sec: float

    @property
    def leader_joint_ids(self) -> frozenset[int]:
        return self.ids & LEADER_JOINT_IDS

    @property
    def follower_joint_ids(self) -> frozenset[int]:
        return self.ids & FOLLOWER_JOINT_IDS


def arbitration_id(can_id: int) -> int:
    if can_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG):
        return -1
    return can_id & CAN_SFF_MASK


def sample_can_traffic(interface: str, duration_sec: float = 0.25) -> CanTraffic:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    ids: set[int] = set()
    frames = 0
    deadline = time.monotonic() + duration_sec
    bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        bus.bind((interface,))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            bus.settimeout(remaining)
            try:
                frame = bus.recv(CAN_FRAME.size)
            except TimeoutError:
                break
            if len(frame) != CAN_FRAME.size:
                continue
            can_id, _, _ = CAN_FRAME.unpack(frame)
            frames += 1
            frame_id = arbitration_id(can_id)
            if frame_id >= 0:
                ids.add(frame_id)
    finally:
        bus.close()
    return CanTraffic(interface, frozenset(ids), frames, duration_sec)


def classify_can_traffic(ids: Iterable[int]) -> str:
    observed = frozenset(ids)
    has_leader = LEADER_JOINT_IDS.issubset(observed)
    has_follower = FOLLOWER_JOINT_IDS.issubset(observed)
    if has_leader and not has_follower:
        return "leader"
    if has_follower and not has_leader:
        return "follower"
    if has_leader and has_follower:
        return "mixed"
    return "unknown"


def require_can_role(
    interface: str,
    role: str,
    duration_sec: float = 0.25,
    recovery_timeout_sec: float = 3.0,
) -> CanTraffic:
    if role not in {"leader", "follower"}:
        raise ValueError(f"unsupported CAN role: {role}")
    if recovery_timeout_sec < duration_sec:
        raise ValueError("recovery_timeout_sec must be >= duration_sec")

    deadline = time.monotonic() + recovery_timeout_sec
    observed_ids: set[int] = set()
    observed_frames = 0
    while True:
        traffic = sample_can_traffic(interface, duration_sec)
        observed_ids.update(traffic.ids)
        observed_frames += traffic.frames
        actual = classify_can_traffic(observed_ids)
        if actual == role:
            return CanTraffic(
                interface,
                frozenset(observed_ids),
                observed_frames,
                recovery_timeout_sec - max(0.0, deadline - time.monotonic()),
            )
        # A confirmed opposite or mixed role is a wiring/configuration fault,
        # not a transient post-SDK handoff. Never retry through it.
        if actual in {"leader", "follower", "mixed"} or time.monotonic() >= deadline:
            observed = ",".join(
                f"0x{value:03X}" for value in sorted(observed_ids)
            ) or "none"
            raise CanRoleError(
                f"{interface} is not a clean {role} interface: classified={actual} "
                f"frames={observed_frames} ids={observed}"
            )
        time.sleep(0.1)


def bridge_command(command: str, socket_path: str = BRIDGE_SOCKET_PATH) -> dict:
    if command not in {"status", "check", "pause", "resume", "stop"}:
        raise ValueError(f"unsupported bridge command: {command}")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2.0)
    try:
        client.connect(socket_path)
        client.sendall((command + "\n").encode("ascii"))
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    finally:
        client.close()
    if not response:
        raise RuntimeError("CAN bridge returned an empty response")
    result = json.loads(response)
    if not result.get("ok", False):
        raise RuntimeError(str(result.get("error", "CAN bridge command failed")))
    return result


def bridge_status_if_running(socket_path: str = BRIDGE_SOCKET_PATH) -> dict | None:
    try:
        return bridge_command("status", socket_path)
    except OSError:
        return None


def require_bridge_not_forwarding(socket_path: str = BRIDGE_SOCKET_PATH) -> dict | None:
    status = bridge_status_if_running(socket_path)
    if status is None:
        return None
    if not status.get("paused", False) and not status.get("dry_run", False):
        raise RuntimeError(
            "Leader/follower CAN bridge is forwarding commands. Pause it before direct "
            "follower control: nero_can_bridge.py pause"
        )
    return status


def wait_for_leader_pose(
    target_deg: Iterable[float],
    *,
    interface: str = LEADER_CAN_PORT,
    tolerance_deg: float = 1.0,
    timeout_sec: float = 120.0,
) -> list[float]:
    import math

    import numpy as np
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    target = np.asarray(list(target_deg), dtype=np.float64)
    if target.shape != (7,):
        raise ValueError("leader target must contain seven joints")
    require_can_role(interface, "leader")
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=interface,
    )
    robot = AgxArmFactory.create_arm(config)
    robot.connect()
    try:
        warmup_deadline = time.monotonic() + 0.25
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            feedback = robot.get_leader_joint_angles()
            if feedback is not None:
                current = np.asarray(
                    [math.degrees(value) for value in feedback.msg], dtype=np.float64
                )
                if current.shape == (7,) and np.isfinite(current).all():
                    if (
                        time.monotonic() >= warmup_deadline
                        and float(np.max(np.abs(target - current))) <= tolerance_deg
                    ):
                        return current.tolist()
            time.sleep(0.05)
    finally:
        robot.disconnect()
    raise TimeoutError(
        f"leader did not reach the capture pose within {timeout_sec:.1f}s on {interface}"
    )


def read_leader_pose(
    interface: str = LEADER_CAN_PORT,
    timeout_sec: float = 2.0,
) -> list[float]:
    import math

    import numpy as np
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    require_can_role(interface, "leader")
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=interface,
    )
    robot = AgxArmFactory.create_arm(config)
    robot.connect()
    try:
        warmup_deadline = time.monotonic() + 0.25
        deadline = time.monotonic() + timeout_sec
        latest: list[float] | None = None
        while time.monotonic() < deadline:
            feedback = robot.get_leader_joint_angles()
            if feedback is not None:
                current = np.asarray(
                    [math.degrees(value) for value in feedback.msg], dtype=np.float64
                )
                if current.shape == (7,) and np.isfinite(current).all():
                    latest = current.tolist()
                    if time.monotonic() >= warmup_deadline:
                        return latest
            time.sleep(0.02)
    finally:
        robot.disconnect()
    raise TimeoutError(f"no complete leader pose received on {interface}")


def read_follower_pose(
    interface: str = FOLLOWER_CAN_PORT,
    timeout_sec: float = 2.0,
) -> list[float]:
    import math

    import numpy as np
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    require_can_role(interface, "follower")
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=interface,
    )
    robot = AgxArmFactory.create_arm(config)
    robot.connect()
    try:
        warmup_deadline = time.monotonic() + 0.25
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            feedback = robot.get_joint_angles()
            if feedback is not None and time.monotonic() >= warmup_deadline:
                current = np.asarray(
                    [math.degrees(value) for value in feedback.msg], dtype=np.float64
                )
                if current.shape == (7,) and np.isfinite(current).all():
                    return current.tolist()
            time.sleep(0.02)
    finally:
        robot.disconnect()
    raise TimeoutError(f"no complete follower pose received on {interface}")


def measure_role_pose_mismatch(
    leader_can: str = LEADER_CAN_PORT,
    follower_can: str = FOLLOWER_CAN_PORT,
) -> dict:
    leader = read_leader_pose(leader_can)
    follower = read_follower_pose(follower_can)
    errors = [leader_value - follower_value for leader_value, follower_value in zip(leader, follower)]
    return {
        "leader_deg": leader,
        "follower_deg": follower,
        "error_deg": errors,
        "max_abs_error_deg": max(abs(value) for value in errors),
    }
