#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from nero_vla.dual_can import FOLLOWER_CAN_PORT

from nero_vla.policy_client import OpenPiPolicyClient
from nero_vla.policy_client import port_open
from nero_vla.state_monitor import detect_firmware
from nero_vla.state_monitor import make_robot
from nero_vla.state_monitor import wait_for


IMAGE_SIZE = 224


def make_test_images() -> tuple[np.ndarray, np.ndarray]:
    exterior = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    exterior[:16, :, :] = np.array([40, 80, 180], dtype=np.uint8)
    exterior[-16:, :, :] = np.array([40, 160, 80], dtype=np.uint8)
    exterior[72:152, 72:152, :] = np.array([220, 30, 30], dtype=np.uint8)

    wrist = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 96, dtype=np.uint8)
    wrist[:, 108:116, :] = 220
    wrist[108:116, :, :] = 220
    return np.ascontiguousarray(exterior), np.ascontiguousarray(wrist)


def normalize_gripper(value: float, mode: str, open_width_m: float) -> float:
    if mode != "width":
        raise RuntimeError(f"Expected gripper width mode, got {mode!r}")
    if open_width_m <= 0.0:
        raise ValueError("gripper open width must be positive")
    return float(np.clip(1.0 - value / open_width_m, 0.0, 1.0))


def read_observation(
    robot: Any,
    gripper: Any,
    exterior: np.ndarray,
    wrist: np.ndarray,
    prompt: str,
    open_width_m: float,
    timeout: float,
    max_feedback_age_sec: float,
    max_feedback_skew_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    joints = wait_for(robot.get_joint_angles, timeout)
    gripper_status = wait_for(gripper.get_gripper_status, timeout)
    if joints is None or gripper_status is None:
        raise RuntimeError("Timed out waiting for NERO joint/gripper feedback")

    joint_position = np.asarray(joints.msg, dtype=np.float32)
    if joint_position.shape != (7,) or not np.isfinite(joint_position).all():
        raise RuntimeError(f"Invalid NERO joint feedback: shape={joint_position.shape}")
    gripper_msg = gripper_status.msg
    gripper_position = normalize_gripper(
        float(gripper_msg.value),
        str(gripper_msg.mode),
        open_width_m,
    )
    assembled_wall_ns = time.time_ns()
    assembled_mono_ns = time.monotonic_ns()
    joint_timestamp_s = float(joints.timestamp)
    gripper_timestamp_s = float(gripper_status.timestamp)
    joint_age_ms = max(0.0, assembled_wall_ns / 1e9 - joint_timestamp_s) * 1000.0
    gripper_age_ms = max(0.0, assembled_wall_ns / 1e9 - gripper_timestamp_s) * 1000.0
    feedback_skew_ms = abs(joint_timestamp_s - gripper_timestamp_s) * 1000.0
    max_age_ms = max_feedback_age_sec * 1000.0
    max_skew_ms = max_feedback_skew_sec * 1000.0
    if joint_age_ms > max_age_ms or gripper_age_ms > max_age_ms:
        raise RuntimeError(
            f"Stale NERO feedback: joint_age={joint_age_ms:.1f}ms "
            f"gripper_age={gripper_age_ms:.1f}ms max={max_age_ms:.1f}ms"
        )
    if feedback_skew_ms > max_skew_ms:
        raise RuntimeError(
            f"NERO feedback skew too large: skew={feedback_skew_ms:.1f}ms "
            f"max={max_skew_ms:.1f}ms"
        )
    observation = {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_position,
        "observation/gripper_position": np.asarray([gripper_position], dtype=np.float32),
        "prompt": prompt,
    }
    state = {
        "joint_position_rad": joint_position.tolist(),
        "gripper_width_m": float(gripper_msg.value),
        "gripper_position_droid": gripper_position,
        "timing": {
            "observation_assembled_wall_time_ns": assembled_wall_ns,
            "observation_assembled_monotonic_time_ns": assembled_mono_ns,
            "joint_feedback_wall_time_ns": round(joint_timestamp_s * 1e9),
            "gripper_feedback_wall_time_ns": round(gripper_timestamp_s * 1e9),
            "joint_feedback_hz": float(joints.hz),
            "gripper_feedback_hz": float(gripper_status.hz),
            "joint_age_at_assembly_ms": joint_age_ms,
            "gripper_age_at_assembly_ms": gripper_age_ms,
            "joint_gripper_skew_ms": feedback_skew_ms,
        },
    }
    return observation, state


def action_stats(actions: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "std": float(actions.std()),
        "first_action": actions[0].tolist(),
        "first_action_norm": float(np.linalg.norm(actions[0])),
    }


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def run(args: argparse.Namespace) -> Path:
    if args.requests <= 0:
        raise ValueError("requests must be positive")
    if args.interval < 0.0:
        raise ValueError("interval must be non-negative")
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")
    if args.max_feedback_age <= 0.0 or args.max_feedback_skew <= 0.0:
        raise ValueError("feedback age and skew limits must be positive")
    if not port_open(args.host, args.port, args.connect_timeout):
        raise RuntimeError(f"Policy server is not reachable at {args.host}:{args.port}")

    run_dir = args.output_dir / datetime.now().strftime("nero_policy_probe_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    exterior, wrist = make_test_images()
    np.save(run_dir / "exterior_test_image.npy", exterior)
    np.save(run_dir / "wrist_test_image.npy", wrist)

    firmware, driver = detect_firmware(args.can_port)
    robot = make_robot(args.can_port, driver)
    robot.connect()
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    client = None
    actions_list: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    try:
        client = OpenPiPolicyClient(
            args.host,
            args.port,
            api_key=args.api_key or None,
            open_timeout=args.connect_timeout,
        )
        print(f"connected policy={args.host}:{args.port} metadata={client.metadata}")
        print(f"NERO firmware={firmware} driver={driver}; DRY RUN: actions will not be executed")

        for index in range(args.requests):
            observation, state = read_observation(
                robot,
                gripper,
                exterior,
                wrist,
                args.prompt,
                args.gripper_open_width,
                args.feedback_timeout,
                args.max_feedback_age,
                args.max_feedback_skew,
            )
            result, transport_timing = client.infer_timed(observation)
            client_ms = float(transport_timing["total_ms"])
            state_timing = state["timing"]
            send_wall_ns = int(transport_timing["request_send_start_wall_time_ns"])
            response_wall_ns = int(transport_timing["response_received_wall_time_ns"])
            state_timing["joint_age_at_send_ms"] = (
                send_wall_ns - state_timing["joint_feedback_wall_time_ns"]
            ) / 1e6
            state_timing["gripper_age_at_send_ms"] = (
                send_wall_ns - state_timing["gripper_feedback_wall_time_ns"]
            ) / 1e6
            state_timing["joint_age_at_response_ms"] = (
                response_wall_ns - state_timing["joint_feedback_wall_time_ns"]
            ) / 1e6
            state_timing["gripper_age_at_response_ms"] = (
                response_wall_ns - state_timing["gripper_feedback_wall_time_ns"]
            ) / 1e6
            actions = result.get("actions")
            if not isinstance(actions, np.ndarray) or actions.ndim != 2:
                raise RuntimeError(f"Expected 2D action ndarray, got {type(actions)!r} {getattr(actions, 'shape', None)}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Policy returned non-finite actions")
            actions = np.asarray(actions, dtype=np.float32)
            stats = action_stats(actions)
            actions_list.append(actions)
            records.append({
                "request": index,
                "client_ms": client_ms,
                "transport_timing": transport_timing,
                "state": state,
                "actions": stats,
                "server_timing": result.get("server_timing", {}),
                "policy_timing": result.get("policy_timing", {}),
            })
            print(
                f"request={index + 1}/{args.requests} latency={client_ms:.1f}ms "
                f"state_age={state_timing['joint_age_at_send_ms']:.1f}/"
                f"{state_timing['gripper_age_at_send_ms']:.1f}ms "
                f"shape={tuple(actions.shape)} range=[{stats['min']:.3f},{stats['max']:.3f}] "
                f"first={np.round(actions[0], 4).tolist()}"
            )
            if index + 1 < args.requests and args.interval:
                time.sleep(args.interval)
    finally:
        if client is not None:
            client.close()
        robot.disconnect()

    stacked = np.stack(actions_list)
    np.save(run_dir / "actions.npy", stacked)
    summary = {
        "mode": "dry_run_no_action_execution",
        "host": args.host,
        "port": args.port,
        "prompt": args.prompt,
        "firmware": firmware,
        "driver": driver,
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "image_dtype": "uint8",
        "gripper_open_width_m": args.gripper_open_width,
        "max_feedback_age_sec": args.max_feedback_age,
        "max_feedback_skew_sec": args.max_feedback_skew,
        "actions_file": str(run_dir / "actions.npy"),
        "actions_shape": list(stacked.shape),
        "timing_summary": {
            "client_total_ms": distribution([record["client_ms"] for record in records]),
            "pack_ms": distribution([record["transport_timing"]["pack_ms"] for record in records]),
            "send_ms": distribution([record["transport_timing"]["send_ms"] for record in records]),
            "wait_response_ms": distribution(
                [record["transport_timing"]["wait_response_ms"] for record in records]
            ),
            "unpack_ms": distribution([record["transport_timing"]["unpack_ms"] for record in records]),
            "joint_age_at_send_ms": distribution(
                [record["state"]["timing"]["joint_age_at_send_ms"] for record in records]
            ),
            "gripper_age_at_send_ms": distribution(
                [record["state"]["timing"]["gripper_age_at_send_ms"] for record in records]
            ),
            "joint_gripper_skew_ms": distribution(
                [record["state"]["timing"]["joint_gripper_skew_ms"] for record in records]
            ),
        },
        "records": records,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved dry-run result: {run_dir}")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only NERO -> pi0.5 DROID inference probe")
    parser.add_argument("--host", default="172.24.1.174")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--can-port", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--prompt", default="pick up the red cube")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--feedback-timeout", type=float, default=3.0)
    parser.add_argument("--max-feedback-age", type=float, default=0.2)
    parser.add_argument("--max-feedback-skew", type=float, default=0.1)
    parser.add_argument("--gripper-open-width", type=float, default=0.07)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/logs/policy_probe",
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
