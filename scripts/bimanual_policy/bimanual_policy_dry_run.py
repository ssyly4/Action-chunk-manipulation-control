#!/usr/bin/env python3
"""Validate the bimanual towel policy with live inputs without commanding either arm."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


CONTROL_ROOT = Path(__file__).resolve().parents[2]
PYAGXARM_SOURCE = Path(
    os.environ.get("NERO_ARM_SDK_ROOT", "/home/dev/nero_ws/src/pyAgxArm")
)
SCRIPT_DIR = Path(__file__).resolve().parent
TELEOP_SRC = Path(os.environ.get("NERO_TELEOP_SRC", "/home/dev/nero_neo_teleop/src"))
for source in (CONTROL_ROOT, PYAGXARM_SOURCE, SCRIPT_DIR, TELEOP_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from nero_neo_teleop.recording.bimanual_lerobot_recorder import NeroCanStateSource
from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.image_tools import resize_with_pad
from nero_vla.policy_client import OpenPiPolicyClient, port_open


def save_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="172.24.1.154")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--left-can", default="can_left")
    parser.add_argument("--right-can", default="can_right")
    parser.add_argument("--world-camera", required=True)
    parser.add_argument("--left-wrist-camera", required=True)
    parser.add_argument("--right-wrist-camera", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-input-age", type=float, default=0.25)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONTROL_ROOT / "artifacts/logs/bimanual_policy_dry_run",
    )
    args = parser.parse_args()

    if args.requests <= 0 or args.interval < 0 or args.max_input_age <= 0:
        parser.error("requests/max-input-age must be positive and interval non-negative")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"Policy server is not reachable at {args.policy_host}:{args.policy_port}")

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    fixed_noise = None
    if args.noise_seed is not None:
        fixed_noise = np.random.default_rng(args.noise_seed).standard_normal(
            (24, 32), dtype=np.float32
        )

    left = NeroCanStateSource(args.left_can, "left")
    right = NeroCanStateSource(args.right_can, "right")
    world = V4L2CameraReader(
        args.world_camera, width=args.width, height=args.height, fps=args.fps, name="world"
    )
    left_wrist = V4L2CameraReader(
        args.left_wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="left_wrist",
    )
    right_wrist = V4L2CameraReader(
        args.right_wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="right_wrist",
    )
    sources = (left, right, world, left_wrist, right_wrist)
    client: OpenPiPolicyClient | None = None
    records: list[dict] = []
    chunks: list[np.ndarray] = []

    print("DRY RUN ONLY: no SDK or robot command API is used", flush=True)
    for source in sources:
        source.start()
    try:
        left.wait_ready(8.0)
        right.wait_ready(8.0)
        world.wait_ready(8.0)
        left_wrist.wait_ready(8.0)
        right_wrist.wait_ready(8.0)
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
        print(f"inputs ready; policy metadata={client.metadata}", flush=True)

        for index in range(args.requests):
            assembled_ns = time.monotonic_ns()
            left_state = left.snapshot(max_age_sec=args.max_input_age)
            right_state = right.snapshot(max_age_sec=args.max_input_age)
            state = np.concatenate((left_state.vector, right_state.vector)).astype(np.float32)
            world_frame = world.latest(max_age_sec=args.max_input_age)
            left_frame = left_wrist.latest(max_age_sec=args.max_input_age)
            right_frame = right_wrist.latest(max_age_sec=args.max_input_age)
            world_model = resize_with_pad(world_frame.image_rgb)
            left_model = resize_with_pad(left_frame.image_rgb)
            right_model = resize_with_pad(right_frame.image_rgb)
            observation = {
                "observation/world_image": world_model,
                "observation/left_wrist_image": left_model,
                "observation/right_wrist_image": right_model,
                "observation/state": state,
                "prompt": args.prompt,
            }
            if fixed_noise is not None:
                observation["__openpi_noise"] = fixed_noise

            started = time.perf_counter()
            result, transport = client.infer_timed(observation)
            inference_ms = (time.perf_counter() - started) * 1000.0
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.shape != (24, 16):
                raise RuntimeError(f"Expected bimanual action shape (24, 16), got {actions.shape}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Policy returned NaN or infinity")

            joint_indices = np.asarray([*range(7), *range(8, 15)])
            first_delta_deg = np.rad2deg(actions[0, joint_indices] - state[joint_indices])
            chunk_delta_deg = np.rad2deg(
                actions[:, joint_indices] - state[None, joint_indices]
            )
            consecutive_deg = np.rad2deg(np.diff(actions[:, joint_indices], axis=0))
            record = {
                "request": index,
                "prompt": args.prompt,
                "inference_ms": inference_ms,
                "transport": transport,
                "state": state.tolist(),
                "action_shape": list(actions.shape),
                "source_image_shape": list(world_frame.image_rgb.shape),
                "policy_image_shape": list(world_model.shape),
                "policy_image_bytes": int(
                    world_model.nbytes + left_model.nbytes + right_model.nbytes
                ),
                "first_action": actions[0].tolist(),
                "first_joint_delta_deg": first_delta_deg.tolist(),
                "max_abs_chunk_delta_deg": float(np.max(np.abs(chunk_delta_deg))),
                "max_abs_consecutive_delta_deg": float(np.max(np.abs(consecutive_deg))),
                "left_gripper_range": [float(actions[:, 7].min()), float(actions[:, 7].max())],
                "right_gripper_range": [float(actions[:, 15].min()), float(actions[:, 15].max())],
                "input_age_ms": {
                    "left_can": (assembled_ns - left_state.monotonic_ns) / 1e6,
                    "right_can": (assembled_ns - right_state.monotonic_ns) / 1e6,
                    "world": (assembled_ns - world_frame.monotonic_ns) / 1e6,
                    "left_wrist": (assembled_ns - left_frame.monotonic_ns) / 1e6,
                    "right_wrist": (assembled_ns - right_frame.monotonic_ns) / 1e6,
                },
            }
            records.append(record)
            chunks.append(actions)
            if index == 0:
                save_rgb(run_dir / "world.png", world_frame.image_rgb)
                save_rgb(run_dir / "left_wrist.png", left_frame.image_rgb)
                save_rgb(run_dir / "right_wrist.png", right_frame.image_rgb)
                save_rgb(run_dir / "world_model_224.png", world_model)
                save_rgb(run_dir / "left_wrist_model_224.png", left_model)
                save_rgb(run_dir / "right_wrist_model_224.png", right_model)
            print(
                f"request={index + 1}/{args.requests} infer={inference_ms:.1f}ms "
                f"send={transport['send_ms']:.1f}ms "
                f"chunk_max={record['max_abs_chunk_delta_deg']:.2f}deg "
                f"consecutive_max={record['max_abs_consecutive_delta_deg']:.2f}deg "
                f"gripL={record['left_gripper_range']} gripR={record['right_gripper_range']}",
                flush=True,
            )
            if index + 1 < args.requests and args.interval:
                time.sleep(args.interval)
    finally:
        if client is not None:
            client.close()
        for source in reversed(sources):
            source.stop()

    np.save(run_dir / "actions.npy", np.stack(chunks))
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "mode": "bimanual_live_dry_run_no_robot_commands",
        "requests": len(records),
        "prompt": args.prompt,
        "mean_inference_ms": float(np.mean([row["inference_ms"] for row in records])),
        "max_chunk_delta_deg": float(max(row["max_abs_chunk_delta_deg"] for row in records)),
        "max_consecutive_delta_deg": float(
            max(row["max_abs_consecutive_delta_deg"] for row in records)
        ),
        "camera_rates_hz": {
            "world": world.measured_hz,
            "left_wrist": left_wrist.measured_hz,
            "right_wrist": right_wrist.measured_hz,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PASS: bimanual policy dry run complete; log={run_dir}", flush=True)


if __name__ == "__main__":
    main()
