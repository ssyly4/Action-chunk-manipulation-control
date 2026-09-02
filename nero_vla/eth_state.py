"""Passive NERO state reader for the controller's Ethernet WebSocket."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import threading
import time
from typing import Any

from websockets.sync.client import ClientConnection, connect


@dataclass(frozen=True)
class TimedMessage:
    uri: str
    data: dict[str, Any]
    monotonic_ns: int
    unix_ns: int


@dataclass(frozen=True)
class NeroEthSnapshot:
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_monotonic_ns: int
    joint_unix_ns: int
    gripper_stroke_mm: float | None
    gripper_position_rad: float | None
    gripper_monotonic_ns: int | None
    gripper_unix_ns: int | None
    flange_pose_m_rad: tuple[float, ...] | None
    pose_monotonic_ns: int | None
    pose_unix_ns: int | None


class NeroEthStateReader:
    """Receive broadcast state without sending commands or changing control mode."""

    def __init__(self, host: str = "10.90.0.150", port: int = 9090) -> None:
        self.url = f"ws://{host}:{port}"
        self._condition = threading.Condition()
        self._latest: dict[str, TimedMessage] = {}
        self._counts: Counter[str] = Counter()
        self._started_ns: int | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection: ClientConnection | None = None

    def start(self) -> "NeroEthStateReader":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._started_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run, name="nero-eth", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._connection is not None:
            self._connection.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def __enter__(self) -> "NeroEthStateReader":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def wait_ready(self, timeout_sec: float = 3.0, require_gripper: bool = True) -> None:
        required = {"/jointStates"}
        if require_gripper:
            required.add("/gripperFeedback")
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while not required.issubset(self._latest):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    suffix = f" ({self._error})" if self._error else ""
                    missing = sorted(required - self._latest.keys())
                    raise TimeoutError(f"Missing NERO ETH streams {missing} from {self.url}{suffix}")
                self._condition.wait(remaining)

    def snapshot(self, max_age_sec: float = 0.2) -> NeroEthSnapshot:
        with self._condition:
            joint = self._latest.get("/jointStates")
            gripper = self._latest.get("/gripperFeedback")
            pose = self._latest.get("/poseStates")
        if joint is None:
            raise RuntimeError("No /jointStates message received")
        age_sec = (time.monotonic_ns() - joint.monotonic_ns) / 1e9
        if age_sec > max_age_sec:
            raise RuntimeError(f"NERO ETH joint state is stale: {age_sec:.3f}s")

        positions = tuple(float(x) for x in joint.data.get("position", ()))
        velocities = tuple(float(x) for x in joint.data.get("velocity", ()))
        if len(positions) != 7:
            raise RuntimeError(f"Expected 7 joints, received {len(positions)}")
        gripper_data = gripper.data if gripper else {}
        pose_data = pose.data if pose else {}
        flange = None
        if pose is not None:
            flange = tuple(float(pose_data[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw"))
        return NeroEthSnapshot(
            joint_position_rad=positions,
            joint_velocity_rad_s=velocities,
            joint_monotonic_ns=joint.monotonic_ns,
            joint_unix_ns=joint.unix_ns,
            gripper_stroke_mm=_optional_float(gripper_data.get("stroke_mm")),
            gripper_position_rad=_optional_float(gripper_data.get("position_rad")),
            gripper_monotonic_ns=gripper.monotonic_ns if gripper else None,
            gripper_unix_ns=gripper.unix_ns if gripper else None,
            flange_pose_m_rad=flange,
            pose_monotonic_ns=pose.monotonic_ns if pose else None,
            pose_unix_ns=pose.unix_ns if pose else None,
        )

    def rates_hz(self) -> dict[str, float]:
        if self._started_ns is None:
            return {}
        elapsed = (time.monotonic_ns() - self._started_ns) / 1e9
        with self._condition:
            return {key: value / elapsed for key, value in self._counts.items()}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=3, close_timeout=1) as connection:
                    self._connection = connection
                    with self._condition:
                        self._error = None
                    while not self._stop.is_set():
                        try:
                            raw = connection.recv(timeout=0.5)
                        except TimeoutError:
                            continue
                        received_mono_ns = time.monotonic_ns()
                        received_unix_ns = time.time_ns()
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        message = json.loads(raw)
                        uri, data = message.get("uri"), message.get("data")
                        if not isinstance(uri, str) or not isinstance(data, dict):
                            continue
                        sample = TimedMessage(uri, data, received_mono_ns, received_unix_ns)
                        with self._condition:
                            self._latest[uri] = sample
                            self._counts[uri] += 1
                            self._condition.notify_all()
            except Exception as exc:
                if self._stop.is_set():
                    break
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                self._stop.wait(1.0)
            finally:
                self._connection = None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
