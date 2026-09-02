#!/usr/bin/env python3

"""Run the trained NERO policy against live observations without commanding motion."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.eth_state import NeroEthStateReader
from nero_vla.image_tools import prepare_external_model_image, resize_with_pad
from nero_vla.policy_client import OpenPiPolicyClient, port_open


DEFAULT_EXTERNAL = "/dev/v4l/by-path/pci-0000:07:00.4-usb-0:1.1:1.0-video-index0"
DEFAULT_WRIST = "/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2:1.0-video-index0"


def load_gripper_calibration(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    closed_mm = float(payload["closed_mm"])
    open_mm = float(payload["open_mm"])
    if abs(open_mm - closed_mm) < 5.0:
        raise ValueError("Gripper calibration span must be at least 5 mm")
    return closed_mm, open_mm


def normalize_gripper(stroke_mm: float | None, closed_mm: float, open_mm: float) -> float:
    if stroke_mm is None:
        raise RuntimeError("No gripper stroke received from NERO ETH")
    # This is intentionally identical to the data recorder: closed=0, open=1.
    return float(np.clip((stroke_mm - closed_mm) / (open_mm - closed_mm), 0.0, 1.0))


def save_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live NERO policy dry run; sends no robot commands")
    parser.add_argument("--robot-host", default="10.90.0.150")
    parser.add_argument("--policy-host", default="172.24.1.174")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--external-camera", default=DEFAULT_EXTERNAL)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--prompt", default="pick up the water bottle")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--max-input-age", type=float, default=0.2)
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "nero_ws/logs/real_policy_dry_run",
    )
    args = parser.parse_args()

    if args.requests <= 0 or args.interval < 0 or args.max_input_age <= 0:
        parser.error("requests/max-input-age must be positive and interval must be non-negative")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"Policy server is not reachable at {args.policy_host}:{args.policy_port}")

    closed_mm, open_mm = load_gripper_calibration(args.gripper_calibration)
    fixed_noise = None
    if args.noise_seed is not None:
        fixed_noise = np.random.default_rng(args.noise_seed).standard_normal(
            (16, 32), dtype=np.float32
        )
    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    external = V4L2CameraReader(
        args.external_camera, width=args.width, height=args.height, fps=args.fps, name="external"
    )
    wrist = V4L2CameraReader(
        args.wrist_camera, width=args.width, height=args.height, fps=args.fps, name="wrist"
    )
    eth = NeroEthStateReader(args.robot_host)
    client: OpenPiPolicyClient | None = None
    records: list[dict] = []
    action_chunks: list[np.ndarray] = []

    print("DRY RUN ONLY: this process contains no robot control API calls", flush=True)
    external.start()
    wrist.start()
    eth.start()
    try:
        external.wait_ready(8.0)
        wrist.wait_ready(8.0)
        eth.wait_ready(5.0, require_gripper=True)
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=5.0)
        print(f"inputs ready; policy metadata={client.metadata}", flush=True)

        for index in range(args.requests):
            assembled_ns = time.monotonic_ns()
            state = eth.snapshot(max_age_sec=args.max_input_age)
            external_frame = external.latest(max_age_sec=args.max_input_age)
            wrist_frame = wrist.latest(max_age_sec=args.max_input_age)
            gripper = normalize_gripper(state.gripper_stroke_mm, closed_mm, open_mm)
            state_vector = np.asarray([*state.joint_position_rad, gripper], dtype=np.float32)
            external_model = prepare_external_model_image(external_frame.image_rgb)
            wrist_model = resize_with_pad(wrist_frame.image_rgb)
            observation = {
                "observation/external_image": external_model,
                "observation/wrist_image": wrist_model,
                "observation/state": state_vector,
                "prompt": args.prompt,
            }
            if fixed_noise is not None:
                observation["__openpi_noise"] = fixed_noise

            start = time.perf_counter()
            result, transport = client.infer_timed(observation)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.shape != (16, 8):
                raise RuntimeError(f"Expected action shape (16, 8), got {actions.shape}")
            if not np.isfinite(actions).all():
                raise RuntimeError("Policy returned NaN or infinity")

            first_delta_deg = np.rad2deg(actions[0, :7] - state_vector[:7])
            chunk_delta_deg = np.rad2deg(actions[:, :7] - state_vector[None, :7])
            consecutive_deg = np.rad2deg(np.diff(actions[:, :7], axis=0))
            record = {
                "request": index,
                "prompt": args.prompt,
                "inference_ms": elapsed_ms,
                "transport": transport,
                "state": state_vector.tolist(),
                "gripper_stroke_mm": state.gripper_stroke_mm,
                "action_shape": list(actions.shape),
                "first_action": actions[0].tolist(),
                "first_delta_deg": first_delta_deg.tolist(),
                "max_abs_chunk_delta_deg": float(np.max(np.abs(chunk_delta_deg))),
                "max_abs_consecutive_delta_deg": float(np.max(np.abs(consecutive_deg))),
                "gripper_action_min": float(np.min(actions[:, 7])),
                "gripper_action_max": float(np.max(actions[:, 7])),
                "input_age_ms": {
                    "joint": (assembled_ns - state.joint_monotonic_ns) / 1e6,
                    "gripper": None
                    if state.gripper_monotonic_ns is None
                    else (assembled_ns - state.gripper_monotonic_ns) / 1e6,
                    "external": (assembled_ns - external_frame.monotonic_ns) / 1e6,
                    "wrist": (assembled_ns - wrist_frame.monotonic_ns) / 1e6,
                },
            }
            records.append(record)
            action_chunks.append(actions)
            if index == 0:
                save_rgb(run_dir / "external.png", external_frame.image_rgb)
                save_rgb(run_dir / "wrist.png", wrist_frame.image_rgb)
                save_rgb(run_dir / "external_model_224.png", external_model)
                save_rgb(run_dir / "wrist_model_224.png", wrist_model)
            print(
                f"request={index + 1}/{args.requests} infer={elapsed_ms:.1f}ms "
                f"first_delta_deg={np.round(first_delta_deg, 2).tolist()} "
                f"chunk_max={record['max_abs_chunk_delta_deg']:.2f}deg "
                f"gripper=[{record['gripper_action_min']:.3f},{record['gripper_action_max']:.3f}]",
                flush=True,
            )
            if index + 1 < args.requests and args.interval:
                time.sleep(args.interval)
    finally:
        if client is not None:
            client.close()
        eth.stop()
        external.stop()
        wrist.stop()

    np.save(run_dir / "actions.npy", np.stack(action_chunks))
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "mode": "live_observation_dry_run_no_robot_commands",
        "requests": len(records),
        "prompt": args.prompt,
        "noise_seed": args.noise_seed,
        "mean_inference_ms": float(np.mean([x["inference_ms"] for x in records])),
        "max_chunk_delta_deg": float(max(x["max_abs_chunk_delta_deg"] for x in records)),
        "max_consecutive_delta_deg": float(
            max(x["max_abs_consecutive_delta_deg"] for x in records)
        ),
        "external_hz": external.measured_hz,
        "wrist_hz": wrist.measured_hz,
        "eth_rates_hz": eth.rates_hz(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: dry run complete; results={run_dir}", flush=True)


if __name__ == "__main__":
    main()
