#!/usr/bin/env python3

import argparse
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pyAgxArm import AgxArmFactory
from pyAgxArm import ArmModel
from pyAgxArm import NeroFW
from pyAgxArm import create_agx_arm_config
from nero_vla.dual_can import FOLLOWER_CAN_PORT


SCHEMA_VERSION = "1.0"
FAST_STALE_SEC = 0.100
SLOW_STALE_SEC = 0.200
LOST_FEEDBACK_EXIT_SEC = 1.0

DRIVER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)

GRIPPER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)


def enum_code(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raw = getattr(value, "value", value)
        return int(raw)


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def firmware_driver(software_version: str) -> str:
    try:
        major, minor = (int(part) for part in software_version.split(".", maxsplit=1))
    except ValueError:
        return NeroFW.DEFAULT

    version = (major, minor)
    if version >= (1, 20):
        return NeroFW.V120
    if version >= (1, 12):
        return NeroFW.V112
    if version >= (1, 11):
        return NeroFW.V111
    return NeroFW.DEFAULT


def make_robot(can_port: str, firmware: str):
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=firmware,
        interface="socketcan",
        channel=can_port,
    )
    return AgxArmFactory.create_arm(config)


def wait_for(getter, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = getter()
        if value is not None:
            return value
        time.sleep(0.05)
    return None


def detect_firmware(can_port: str, timeout: float = 5.0) -> tuple[str, str]:
    detector = make_robot(can_port, NeroFW.DEFAULT)
    detector.connect()
    try:
        feedback = wait_for(detector.get_firmware, timeout)
    finally:
        detector.disconnect()
    if feedback is None:
        raise RuntimeError("No firmware response received from NERO")
    software_version = str(feedback["software_version"])
    return software_version, firmware_driver(software_version)


def flag_dict(flags: Any, fields: tuple[str, ...], extra: tuple[str, ...] = ()) -> dict[str, bool]:
    return {
        field: bool(getattr(flags, field, False))
        for field in (*fields, *extra)
    }


def source_metadata(feedback: Any, now_wall_sec: float) -> dict[str, float | None]:
    if feedback is None:
        return {"timestamp": None, "hz": None, "age_ms": None}
    timestamp = float(feedback.timestamp)
    age_sec = max(0.0, now_wall_sec - timestamp) if timestamp > 0.0 else math.inf
    return {
        "timestamp": timestamp,
        "hz": float(feedback.hz),
        "age_ms": None if not math.isfinite(age_sec) else age_sec * 1000.0,
    }


def collect_feedback(robot: Any, gripper: Any) -> dict[str, Any]:
    return {
        "arm": robot.get_arm_status(),
        "joints": robot.get_joint_angles(),
        "flange": robot.get_flange_pose(),
        "motors": [robot.get_motor_states(index) for index in range(1, 8)],
        "drivers": [robot.get_driver_states(index) for index in range(1, 8)],
        "gripper": gripper.get_gripper_status(),
    }


def missing_sources(feedback: dict[str, Any]) -> list[str]:
    missing = [
        name
        for name in ("arm", "joints", "flange", "gripper")
        if feedback[name] is None
    ]
    missing.extend(
        f"motor_{index}"
        for index, value in enumerate(feedback["motors"], start=1)
        if value is None
    )
    missing.extend(
        f"driver_{index}"
        for index, value in enumerate(feedback["drivers"], start=1)
        if value is None
    )
    return missing


def source_map(feedback: dict[str, Any], now_wall_sec: float) -> dict[str, dict[str, float | None]]:
    sources = {
        name: source_metadata(feedback[name], now_wall_sec)
        for name in ("arm", "joints", "flange", "gripper")
    }
    sources.update(
        {
            f"motor_{index}": source_metadata(value, now_wall_sec)
            for index, value in enumerate(feedback["motors"], start=1)
        }
    )
    sources.update(
        {
            f"driver_{index}": source_metadata(value, now_wall_sec)
            for index, value in enumerate(feedback["drivers"], start=1)
        }
    )
    return sources


def stale_sources(sources: dict[str, dict[str, float | None]]) -> list[str]:
    stale = []
    for name, metadata in sources.items():
        age_ms = metadata["age_ms"]
        if age_ms is None:
            continue
        threshold_ms = (
            FAST_STALE_SEC * 1000.0
            if name in ("arm", "joints") or name.startswith("motor_")
            else SLOW_STALE_SEC * 1000.0
        )
        if age_ms > threshold_ms:
            stale.append(name)
    return stale


def find_nonfinite(value: Any, path: str = "") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [] if math.isfinite(float(value)) else [path or "<root>"]
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.extend(find_nonfinite(child, child_path))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            result.extend(find_nonfinite(child, child_path))
        return result
    return []


def build_snapshot(
    robot: Any,
    gripper: Any,
    *,
    firmware: str,
    seq: int,
    wall_time_ns: int,
    monotonic_time_ns: int,
    scheduled_time_ns: int,
    period_ns: int,
    missed_deadlines: int,
) -> dict[str, Any]:
    feedback = collect_feedback(robot, gripper)
    now_wall_sec = wall_time_ns / 1_000_000_000
    missing = missing_sources(feedback)
    sources = source_map(feedback, now_wall_sec)
    stale = stale_sources(sources)

    arm_feedback = feedback["arm"]
    joint_feedback = feedback["joints"]
    flange_feedback = feedback["flange"]
    gripper_feedback = feedback["gripper"]

    arm = None
    arm_faults: list[str] = []
    if arm_feedback is not None:
        arm_msg = arm_feedback.msg
        arm_state_code = enum_code(arm_msg.arm_status)
        arm = {
            "ctrl_mode": {
                "code": enum_code(arm_msg.ctrl_mode),
                "name": enum_name(arm_msg.ctrl_mode),
            },
            "arm_status": {
                "code": arm_state_code,
                "name": enum_name(arm_msg.arm_status),
            },
            "motion_mode": {
                "code": enum_code(arm_msg.mode_feedback),
                "name": enum_name(arm_msg.mode_feedback),
            },
            "motion_status": {
                "code": enum_code(arm_msg.motion_status),
                "name": enum_name(arm_msg.motion_status),
            },
        }
        if arm_state_code != 0:
            arm_faults.append(f"arm_status:{arm_state_code}:{enum_name(arm_msg.arm_status)}")

    motors = []
    for index, item in enumerate(feedback["motors"], start=1):
        if item is None:
            motors.append(None)
            continue
        msg = item.msg
        motors.append(
            {
                "joint": index,
                "position_rad": float(msg.position),
                "velocity_rad_s": float(msg.velocity),
                "current_a": float(msg.current),
                "torque_nm": float(msg.torque),
            }
        )

    joint_positions = (
        [float(value) for value in joint_feedback.msg]
        if joint_feedback is not None
        else None
    )
    joints = {
        "position_rad": joint_positions,
        "velocity_rad_s": [
            None if motor is None else motor["velocity_rad_s"]
            for motor in motors
        ],
        "torque_nm": [
            None if motor is None else motor["torque_nm"]
            for motor in motors
        ],
        "motor_current_a": [
            None if motor is None else motor["current_a"]
            for motor in motors
        ],
    }

    driver_faults: list[str] = []
    drivers = []
    for index, item in enumerate(feedback["drivers"], start=1):
        if item is None:
            drivers.append(None)
            continue
        msg = item.msg
        flags = flag_dict(
            msg.foc_status,
            DRIVER_FAULT_FIELDS,
            extra=("driver_enable_status",),
        )
        for name in DRIVER_FAULT_FIELDS:
            if flags[name]:
                driver_faults.append(f"driver_{index}:{name}")
        drivers.append(
            {
                "joint": index,
                "voltage_v": float(msg.vol),
                "driver_temp_c": float(msg.foc_temp),
                "motor_temp_c": float(msg.motor_temp),
                "bus_current_a": float(msg.bus_current),
                "flags": flags,
            }
        )

    gripper_data = None
    gripper_faults: list[str] = []
    if gripper_feedback is not None:
        msg = gripper_feedback.msg
        flags = flag_dict(
            msg.foc_status,
            GRIPPER_FAULT_FIELDS,
            extra=("driver_enable_status", "homing_status"),
        )
        for name in GRIPPER_FAULT_FIELDS:
            if flags[name]:
                gripper_faults.append(f"gripper:{name}")
        gripper_data = {
            "mode": str(msg.mode),
            "value": float(msg.value),
            "force_n": float(msg.force),
            "flags": flags,
        }

    comm_error = bool(robot.has_comm_error())
    comm_error_text = None
    if comm_error:
        comm_error_text = str(robot.get_comm_error())

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "seq": seq,
        "time": {
            "wall_time_ns": wall_time_ns,
            "monotonic_time_ns": monotonic_time_ns,
            "scheduled_time_ns": scheduled_time_ns,
            "period_ms": period_ns / 1_000_000,
            "lateness_ms": max(0, monotonic_time_ns - scheduled_time_ns) / 1_000_000,
        },
        "firmware": firmware,
        "arm": arm,
        "joints": joints,
        "flange_pose_m_rad": (
            [float(value) for value in flange_feedback.msg]
            if flange_feedback is not None
            else None
        ),
        "motors": motors,
        "drivers": drivers,
        "gripper": gripper_data,
        "sources": sources,
    }

    nonfinite = find_nonfinite(snapshot)
    faults = [*arm_faults, *driver_faults, *gripper_faults]
    snapshot["health"] = {
        "ok": not (missing or stale or faults or comm_error or nonfinite),
        "missing_sources": missing,
        "stale_sources": stale,
        "faults": faults,
        "comm_error": comm_error,
        "comm_error_text": comm_error_text,
        "nonfinite_fields": nonfinite,
        "missed_deadlines": missed_deadlines,
    }
    return snapshot


def advance_deadline(now_ns: int, deadline_ns: int, period_ns: int) -> tuple[int, int]:
    if now_ns < deadline_ns + period_ns:
        return deadline_ns + period_ns, 0
    skipped = (now_ns - deadline_ns) // period_ns
    return deadline_ns + (skipped + 1) * period_ns, int(skipped)


def console_summary(snapshot: dict[str, Any], actual_hz: float) -> str:
    joints = snapshot["joints"]["position_rad"]
    joint_deg = (
        [round(math.degrees(value), 2) for value in joints]
        if joints is not None
        else None
    )
    flange = snapshot["flange_pose_m_rad"]
    flange_xyz = [round(value, 4) for value in flange[:3]] if flange else None
    gripper = snapshot["gripper"]
    gripper_text = "none"
    if gripper is not None:
        unit_value = gripper["value"] * 1000.0 if gripper["mode"] == "width" else gripper["value"]
        unit = "mm" if gripper["mode"] == "width" else "deg"
        gripper_text = f"{unit_value:.2f}{unit}"
    temperatures = [
        driver["motor_temp_c"]
        for driver in snapshot["drivers"]
        if driver is not None
    ]
    max_temp = max(temperatures) if temperatures else math.nan
    arm = snapshot["arm"]
    ctrl = arm["ctrl_mode"]["name"] if arm else "none"
    state = arm["arm_status"]["name"] if arm else "none"
    health = snapshot["health"]
    return (
        f"seq={snapshot['seq']} hz={actual_hz:.2f} ok={health['ok']} "
        f"ctrl={ctrl} state={state} q_deg={joint_deg} "
        f"flange_xyz={flange_xyz} gripper={gripper_text} "
        f"max_motor_temp={max_temp:.1f}C stale={health['stale_sources']} "
        f"missed={health['missed_deadlines']}"
    )


class StateMonitor:
    def __init__(
        self,
        *,
        can_port: str,
        hz: float,
        print_hz: float,
        duration: float,
        log_dir: Path,
        logging_enabled: bool,
    ) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be positive")
        if print_hz <= 0.0:
            raise ValueError("print-hz must be positive")
        if duration < 0.0:
            raise ValueError("duration must be >= 0")

        self.can_port = can_port
        self.hz = hz
        self.print_hz = print_hz
        self.duration = duration
        self.log_dir = log_dir
        self.logging_enabled = logging_enabled
        self.stop_event = threading.Event()

    def stop(self, *_args) -> None:
        self.stop_event.set()

    def run(self) -> int:
        firmware, driver = detect_firmware(self.can_port)
        robot = make_robot(self.can_port, driver)
        robot.connect()
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

        log_file = None
        log_path = None
        period_ns = max(1, round(1_000_000_000 / self.hz))
        print_period_ns = max(1, round(1_000_000_000 / self.print_hz))
        missed_deadlines = 0
        seq = 0
        bad_feedback_since_ns = None

        try:
            initial_deadline = time.monotonic() + 5.0
            while time.monotonic() < initial_deadline:
                if not missing_sources(collect_feedback(robot, gripper)):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("Timed out waiting for complete NERO feedback")

            start_mono_ns = time.monotonic_ns()
            next_deadline_ns = start_mono_ns
            next_print_ns = start_mono_ns + print_period_ns
            last_flush_ns = start_mono_ns

            if self.logging_enabled:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                log_path = self.log_dir / f"nero_state_{stamp}.jsonl"
                log_file = log_path.open("x", encoding="utf-8")

            print(
                f"NERO state monitor ready firmware={firmware} driver={driver} "
                f"can={self.can_port} target_hz={self.hz:.2f} "
                f"log={log_path if log_path else 'disabled'}"
            )

            while not self.stop_event.is_set():
                now_ns = time.monotonic_ns()
                if self.duration > 0.0 and (now_ns - start_mono_ns) / 1e9 >= self.duration:
                    break
                if now_ns < next_deadline_ns:
                    time.sleep((next_deadline_ns - now_ns) / 1e9)

                sample_mono_ns = time.monotonic_ns()
                sample_wall_ns = time.time_ns()
                snapshot = build_snapshot(
                    robot,
                    gripper,
                    firmware=firmware,
                    seq=seq,
                    wall_time_ns=sample_wall_ns,
                    monotonic_time_ns=sample_mono_ns,
                    scheduled_time_ns=next_deadline_ns,
                    period_ns=period_ns,
                    missed_deadlines=missed_deadlines,
                )

                if log_file is not None:
                    log_file.write(json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + "\n")

                feedback_bad = bool(
                    snapshot["health"]["missing_sources"]
                    or snapshot["health"]["stale_sources"]
                )
                if feedback_bad:
                    if bad_feedback_since_ns is None:
                        bad_feedback_since_ns = sample_mono_ns
                    elif (sample_mono_ns - bad_feedback_since_ns) / 1e9 >= LOST_FEEDBACK_EXIT_SEC:
                        raise RuntimeError("Critical NERO feedback missing or stale for more than 1 second")
                else:
                    bad_feedback_since_ns = None

                seq += 1
                elapsed_sec = max((sample_mono_ns - start_mono_ns) / 1e9, 1e-9)
                if sample_mono_ns >= next_print_ns:
                    print(console_summary(snapshot, max(seq - 1, 0) / elapsed_sec))
                    next_print_ns = sample_mono_ns + print_period_ns
                if log_file is not None and sample_mono_ns - last_flush_ns >= 1_000_000_000:
                    log_file.flush()
                    last_flush_ns = sample_mono_ns

                next_deadline_ns, skipped = advance_deadline(
                    sample_mono_ns,
                    next_deadline_ns,
                    period_ns,
                )
                missed_deadlines += skipped

            elapsed_sec = max((time.monotonic_ns() - start_mono_ns) / 1e9, 1e-9)
            print(
                f"NERO state monitor stopped samples={seq} "
                f"actual_hz={max(seq - 1, 0) / elapsed_sec:.3f} missed_deadlines={missed_deadlines} "
                f"log={log_path if log_path else 'disabled'}"
            )
            return 0
        finally:
            if log_file is not None:
                log_file.flush()
                log_file.close()
            robot.disconnect()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-rate read-only NERO state monitor")
    parser.add_argument("--can-port", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--print-hz", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/logs/state",
    )
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    monitor = StateMonitor(
        can_port=args.can_port,
        hz=args.hz,
        print_hz=args.print_hz,
        duration=args.duration,
        log_dir=args.log_dir,
        logging_enabled=not args.no_log,
    )
    signal.signal(signal.SIGINT, monitor.stop)
    signal.signal(signal.SIGTERM, monitor.stop)
    try:
        return monitor.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
