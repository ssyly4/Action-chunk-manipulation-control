#!/usr/bin/env python3

import argparse
import json
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from nero_vla.dual_can import FOLLOWER_CAN_PORT

from nero_vla.policy_probe import make_test_images
from nero_vla.policy_probe import normalize_gripper
from nero_vla.state_monitor import detect_firmware
from nero_vla.state_monitor import make_robot


@dataclass(frozen=True)
class TimedValue:
    wall_time_ns: int
    value: np.ndarray


class SignalBuffer:
    def __init__(self, maxlen: int = 1000) -> None:
        self._values: deque[TimedValue] = deque(maxlen=maxlen)
        self._condition = threading.Condition()
        self.duplicate_count = 0
        self.backward_count = 0

    def append(self, sample: TimedValue) -> None:
        with self._condition:
            if self._values and sample.wall_time_ns == self._values[-1].wall_time_ns:
                self.duplicate_count += 1
                return
            if self._values and sample.wall_time_ns < self._values[-1].wall_time_ns:
                self.backward_count += 1
                return
            self._values.append(sample)
            self._condition.notify_all()

    def wait_until_bracketed(self, target_ns: int, timeout: float) -> tuple[TimedValue, TimedValue, float]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._values or self._values[-1].wall_time_ns < target_ns:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"No feedback after camera timestamp within {timeout:.3f}s")
                self._condition.wait(remaining)

            if self._values[0].wall_time_ns > target_ns:
                raise RuntimeError("State buffer does not contain feedback before camera timestamp")

            values = list(self._values)
            for index in range(1, len(values)):
                before = values[index - 1]
                after = values[index]
                if after.wall_time_ns >= target_ns:
                    span_ns = after.wall_time_ns - before.wall_time_ns
                    alpha = 0.0 if span_ns == 0 else (target_ns - before.wall_time_ns) / span_ns
                    return before, after, float(alpha)
        raise RuntimeError("Could not bracket camera timestamp")


def interpolate(buffer: SignalBuffer, target_ns: int, timeout: float) -> tuple[np.ndarray, dict[str, Any]]:
    before, after, alpha = buffer.wait_until_bracketed(target_ns, timeout)
    value = before.value + alpha * (after.value - before.value)
    timing = {
        "before_wall_time_ns": before.wall_time_ns,
        "after_wall_time_ns": after.wall_time_ns,
        "before_delta_ms": (target_ns - before.wall_time_ns) / 1e6,
        "after_delta_ms": (after.wall_time_ns - target_ns) / 1e6,
        "bracket_span_ms": (after.wall_time_ns - before.wall_time_ns) / 1e6,
        "alpha": alpha,
    }
    return value.astype(np.float32), timing


class NeroStateSampler:
    def __init__(self, robot: Any, gripper: Any, hz: float, gripper_open_width: float) -> None:
        self.robot = robot
        self.gripper = gripper
        self.period_ns = round(1e9 / hz)
        self.gripper_open_width = gripper_open_width
        self.joints = SignalBuffer()
        self.gripper_position = SignalBuffer()
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.poll_count = 0
        self.thread = threading.Thread(target=self._run, name="nero-state-sampler", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        next_deadline_ns = time.monotonic_ns()
        try:
            while not self.stop_event.is_set():
                now_ns = time.monotonic_ns()
                if now_ns < next_deadline_ns:
                    time.sleep((next_deadline_ns - now_ns) / 1e9)
                joints = self.robot.get_joint_angles()
                gripper = self.gripper.get_gripper_status()
                self.poll_count += 1
                if joints is not None:
                    value = np.asarray(joints.msg, dtype=np.float64)
                    if value.shape == (7,) and np.isfinite(value).all():
                        self.joints.append(TimedValue(round(float(joints.timestamp) * 1e9), value))
                if gripper is not None:
                    msg = gripper.msg
                    normalized = normalize_gripper(
                        float(msg.value), str(msg.mode), self.gripper_open_width
                    )
                    self.gripper_position.append(
                        TimedValue(round(float(gripper.timestamp) * 1e9), np.asarray([normalized]))
                    )
                next_deadline_ns += self.period_ns
                now_ns = time.monotonic_ns()
                if next_deadline_ns <= now_ns:
                    skipped = (now_ns - next_deadline_ns) // self.period_ns + 1
                    next_deadline_ns += skipped * self.period_ns
        except Exception as exc:
            self.error = exc
            self.stop_event.set()


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def run(args: argparse.Namespace) -> Path:
    if args.duration <= 0 or args.camera_hz <= 0 or args.state_hz <= 0:
        raise ValueError("duration and frequencies must be positive")
    if args.alignment_timeout <= 0:
        raise ValueError("alignment timeout must be positive")

    run_dir = args.output_dir / datetime.now().strftime("synthetic_observation_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    exterior, wrist = make_test_images()
    np.save(run_dir / "exterior_test_image.npy", exterior)
    np.save(run_dir / "wrist_test_image.npy", wrist)

    firmware, driver = detect_firmware(args.can_port)
    robot = make_robot(args.can_port, driver)
    robot.connect()
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    sampler = NeroStateSampler(robot, gripper, args.state_hz, args.gripper_open_width)
    stop_event = threading.Event()

    def stop(*_args) -> None:
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, stop)
    previous_sigterm = signal.signal(signal.SIGTERM, stop)
    frame_records: list[dict[str, Any]] = []
    aligned_joints: list[np.ndarray] = []
    aligned_grippers: list[np.ndarray] = []
    camera_period_ns = round(1e9 / args.camera_hz)
    try:
        sampler.start()
        time.sleep(0.1)
        start_mono_ns = time.monotonic_ns()
        next_frame_mono_ns = start_mono_ns
        frame_index = 0
        while not stop_event.is_set():
            now_mono_ns = time.monotonic_ns()
            if (now_mono_ns - start_mono_ns) / 1e9 >= args.duration:
                break
            if now_mono_ns < next_frame_mono_ns:
                time.sleep((next_frame_mono_ns - now_mono_ns) / 1e9)

            camera_wall_ns = time.time_ns()
            camera_mono_ns = time.monotonic_ns()
            joint_position, joint_timing = interpolate(
                sampler.joints, camera_wall_ns, args.alignment_timeout
            )
            gripper_position, gripper_timing = interpolate(
                sampler.gripper_position, camera_wall_ns, args.alignment_timeout
            )
            built_wall_ns = time.time_ns()
            frame_records.append({
                "frame": frame_index,
                "prompt": args.prompt,
                "camera_wall_time_ns": camera_wall_ns,
                "camera_monotonic_time_ns": camera_mono_ns,
                "observation_built_wall_time_ns": built_wall_ns,
                "build_latency_ms": (built_wall_ns - camera_wall_ns) / 1e6,
                "joint_alignment": joint_timing,
                "gripper_alignment": gripper_timing,
            })
            aligned_joints.append(joint_position)
            aligned_grippers.append(gripper_position)
            frame_index += 1
            next_frame_mono_ns += camera_period_ns
            if next_frame_mono_ns <= time.monotonic_ns():
                next_frame_mono_ns = time.monotonic_ns() + camera_period_ns
            if sampler.error is not None:
                raise RuntimeError(f"NERO state sampler failed: {sampler.error}")
    finally:
        sampler.stop()
        robot.disconnect()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    if not frame_records:
        raise RuntimeError("No synthetic camera observations were produced")
    elapsed_sec = (time.monotonic_ns() - start_mono_ns) / 1e9
    joints_array = np.stack(aligned_joints)
    grippers_array = np.stack(aligned_grippers)
    np.savez_compressed(
        run_dir / "aligned_observations.npz",
        joint_position=joints_array,
        gripper_position=grippers_array,
        camera_wall_time_ns=np.asarray([r["camera_wall_time_ns"] for r in frame_records], dtype=np.int64),
        camera_monotonic_time_ns=np.asarray(
            [r["camera_monotonic_time_ns"] for r in frame_records], dtype=np.int64
        ),
    )
    with (run_dir / "frames.jsonl").open("w", encoding="utf-8") as output:
        for record in frame_records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")

    summary = {
        "mode": "synthetic_camera_read_only",
        "firmware": firmware,
        "driver": driver,
        "prompt": args.prompt,
        "duration_requested_sec": args.duration,
        "elapsed_sec": elapsed_sec,
        "state_target_hz": args.state_hz,
        "camera_target_hz": args.camera_hz,
        "gripper_open_width_m": args.gripper_open_width,
        "camera_actual_hz": len(frame_records) / elapsed_sec,
        "frames": len(frame_records),
        "state_polls": sampler.poll_count,
        "joint_duplicate_timestamps": sampler.joints.duplicate_count,
        "gripper_duplicate_timestamps": sampler.gripper_position.duplicate_count,
        "joint_backward_timestamps": sampler.joints.backward_count,
        "gripper_backward_timestamps": sampler.gripper_position.backward_count,
        "joint_before_delta_ms": numeric_summary(
            [r["joint_alignment"]["before_delta_ms"] for r in frame_records]
        ),
        "joint_after_delta_ms": numeric_summary(
            [r["joint_alignment"]["after_delta_ms"] for r in frame_records]
        ),
        "joint_bracket_span_ms": numeric_summary(
            [r["joint_alignment"]["bracket_span_ms"] for r in frame_records]
        ),
        "joint_nearest_delta_ms": numeric_summary(
            [
                min(
                    r["joint_alignment"]["before_delta_ms"],
                    r["joint_alignment"]["after_delta_ms"],
                )
                for r in frame_records
            ]
        ),
        "gripper_before_delta_ms": numeric_summary(
            [r["gripper_alignment"]["before_delta_ms"] for r in frame_records]
        ),
        "gripper_after_delta_ms": numeric_summary(
            [r["gripper_alignment"]["after_delta_ms"] for r in frame_records]
        ),
        "gripper_nearest_delta_ms": numeric_summary(
            [
                min(
                    r["gripper_alignment"]["before_delta_ms"],
                    r["gripper_alignment"]["after_delta_ms"],
                )
                for r in frame_records
            ]
        ),
        "build_latency_ms": numeric_summary([r["build_latency_ms"] for r in frame_records]),
        "joint_shape": list(joints_array.shape),
        "gripper_shape": list(grippers_array.shape),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"synthetic observation complete frames={len(frame_records)} "
        f"camera_hz={summary['camera_actual_hz']:.2f} "
        f"joint_after_p95={summary['joint_after_delta_ms']['p95']:.2f}ms "
        f"build_p95={summary['build_latency_ms']['p95']:.2f}ms"
    )
    print(f"saved: {run_dir}")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only NERO + synthetic camera observation pipeline")
    parser.add_argument("--can-port", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--state-hz", type=float, default=100.0)
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument("--alignment-timeout", type=float, default=0.2)
    parser.add_argument("--gripper-open-width", type=float, default=0.07)
    parser.add_argument("--prompt", default="pick up the red cube")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/logs/synthetic_observation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
