#!/usr/bin/env python3

"""Compare live NERO policy predictions with future leader-follower motion."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.eth_state import NeroEthStateReader
from nero_vla.policy_client import OpenPiPolicyClient, port_open
from nero_vla.real_policy_dry_run import DEFAULT_EXTERNAL
from nero_vla.real_policy_dry_run import DEFAULT_WRIST
from nero_vla.real_policy_dry_run import load_gripper_calibration
from nero_vla.real_policy_dry_run import normalize_gripper
from nero_vla.image_tools import prepare_external_model_image, resize_with_pad


@dataclass(frozen=True)
class StateSample:
    monotonic_ns: int
    state: np.ndarray


class StateHistory:
    """Continuously sample passive ETH feedback while inference is in flight."""

    def __init__(
        self,
        eth: NeroEthStateReader,
        closed_mm: float,
        open_mm: float,
        fps: int,
    ) -> None:
        self.eth = eth
        self.closed_mm = closed_mm
        self.open_mm = open_mm
        self.period_ns = round(1e9 / fps)
        self.samples: list[StateSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nero-shadow-state", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> StateSample:
        with self._lock:
            if not self.samples:
                raise RuntimeError("No shadow state samples have been recorded")
            return self.samples[-1]

    def snapshot(self) -> list[StateSample]:
        with self._lock:
            return list(self.samples)

    def _run(self) -> None:
        next_tick = time.monotonic_ns()
        try:
            while not self._stop.is_set():
                remaining_ns = next_tick - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
                snapshot = self.eth.snapshot(max_age_sec=0.2)
                gripper = normalize_gripper(
                    snapshot.gripper_stroke_mm, self.closed_mm, self.open_mm
                )
                state = np.asarray([*snapshot.joint_position_rad, gripper], dtype=np.float32)
                sample = StateSample(time.monotonic_ns(), state)
                with self._lock:
                    self.samples.append(sample)
                next_tick += self.period_ns
                if next_tick < time.monotonic_ns() - self.period_ns:
                    next_tick = time.monotonic_ns() + self.period_ns
        except Exception as exc:
            self.error = exc
            self._stop.set()


def interpolate_states(samples: list[StateSample], target_ns: np.ndarray) -> np.ndarray:
    if len(samples) < 2:
        raise RuntimeError("At least two state samples are required")
    sample_ns = np.asarray([sample.monotonic_ns for sample in samples], dtype=np.int64)
    states = np.stack([sample.state for sample in samples])
    if target_ns.min() < sample_ns[0] or target_ns.max() > sample_ns[-1]:
        raise RuntimeError("Prediction target lies outside recorded state history")
    result = np.empty((*target_ns.shape, states.shape[1]), dtype=np.float32)
    flat_targets = target_ns.reshape(-1)
    flat_result = result.reshape(-1, states.shape[1])
    for index, target in enumerate(flat_targets):
        right = bisect.bisect_left(sample_ns, int(target))
        if right == 0:
            flat_result[index] = states[0]
        elif right == len(samples):
            flat_result[index] = states[-1]
        else:
            left = right - 1
            span = int(sample_ns[right] - sample_ns[left])
            alpha = 0.0 if span == 0 else (int(target) - int(sample_ns[left])) / span
            flat_result[index] = states[left] + alpha * (states[right] - states[left])
    return result


def save_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow-test NERO policy against leader-follower motion; sends no commands"
    )
    parser.add_argument("--robot-host", default="10.90.0.150")
    parser.add_argument("--policy-host", default="172.24.1.174")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--external-camera", default=DEFAULT_EXTERNAL)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--state-fps", type=int, default=30)
    parser.add_argument("--action-fps", type=int, default=30)
    parser.add_argument("--inference-hz", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--prompt", default="pick up the water bottle")
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "nero_ws/logs/policy_shadow",
    )
    args = parser.parse_args()

    if min(args.camera_fps, args.state_fps, args.action_fps) <= 0:
        parser.error("all FPS values must be positive")
    if args.inference_hz <= 0 or args.duration <= 0:
        parser.error("inference-hz and duration must be positive")
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
        args.external_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="external",
    )
    wrist = V4L2CameraReader(
        args.wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="wrist",
    )
    eth = NeroEthStateReader(args.robot_host)
    history = StateHistory(eth, closed_mm, open_mm, args.state_fps)
    client: OpenPiPolicyClient | None = None
    prediction_chunks: list[np.ndarray] = []
    prediction_times_ns: list[int] = []
    prediction_rows: list[dict] = []

    print("SHADOW ONLY: no CAN connection and no robot control API calls", flush=True)
    external.start()
    wrist.start()
    eth.start()
    try:
        external.wait_ready(8.0)
        wrist.wait_ready(8.0)
        eth.wait_ready(5.0, require_gripper=True)
        history.start()
        deadline = time.monotonic() + 3.0
        while True:
            try:
                history.latest()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("State history did not start")
                time.sleep(0.02)
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=5.0)
        print("inputs and policy ready", flush=True)
        if not args.auto_start:
            input("Place the robot at the start pose. Press Enter, then perform one fresh demonstration: ")

        start_ns = time.monotonic_ns()
        end_ns = start_ns + round(args.duration * 1e9)
        inference_period_ns = round(1e9 / args.inference_hz)
        next_inference_ns = start_ns
        request = 0
        while next_inference_ns < end_ns:
            remaining_ns = next_inference_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1e9)
            if time.monotonic_ns() >= end_ns:
                break
            state_sample = history.latest()
            external_frame = external.latest(max_age_sec=0.2)
            wrist_frame = wrist.latest(max_age_sec=0.2)
            external_model = prepare_external_model_image(external_frame.image_rgb)
            wrist_model = resize_with_pad(wrist_frame.image_rgb)
            observation = {
                "observation/external_image": external_model,
                "observation/wrist_image": wrist_model,
                "observation/state": state_sample.state,
                "prompt": args.prompt,
            }
            if fixed_noise is not None:
                observation["__openpi_noise"] = fixed_noise
            infer_start = time.perf_counter()
            result, transport = client.infer_timed(observation)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.shape != (16, 8) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid action chunk: shape={actions.shape}")
            prediction_chunks.append(actions)
            prediction_times_ns.append(state_sample.monotonic_ns)
            prediction_rows.append(
                {
                    "request": request,
                    "observation_monotonic_ns": state_sample.monotonic_ns,
                    "response_monotonic_ns": time.monotonic_ns(),
                    "inference_ms": infer_ms,
                    "transport": transport,
                    "state": state_sample.state.tolist(),
                    "external_age_ms": (
                        state_sample.monotonic_ns - external_frame.monotonic_ns
                    )
                    / 1e6,
                    "wrist_age_ms": (state_sample.monotonic_ns - wrist_frame.monotonic_ns) / 1e6,
                }
            )
            if request == 0:
                save_rgb(run_dir / "external_model_224.png", external_model)
                save_rgb(run_dir / "wrist_model_224.png", wrist_model)
            request += 1
            print(
                f"shadow request={request} elapsed={(time.monotonic_ns() - start_ns) / 1e9:.1f}s "
                f"infer={infer_ms:.1f}ms",
                flush=True,
            )
            next_inference_ns += inference_period_ns

        # Capture the future corresponding to the tail of the final action chunk.
        settle_until_ns = prediction_times_ns[-1] + round(16 / args.action_fps * 1e9)
        while time.monotonic_ns() < settle_until_ns:
            time.sleep(0.02)
    finally:
        if client is not None:
            client.close()
        history.stop()
        eth.stop()
        external.stop()
        wrist.stop()

    if history.error is not None:
        raise RuntimeError(f"State sampler failed: {history.error}")
    if not prediction_chunks:
        raise RuntimeError("No policy predictions were collected")

    samples = history.snapshot()
    predictions = np.stack(prediction_chunks)
    observation_ns = np.asarray(prediction_times_ns, dtype=np.int64)
    horizon_offsets_ns = np.rint(np.arange(1, 17) * 1e9 / args.action_fps).astype(np.int64)
    target_ns = observation_ns[:, None] + horizon_offsets_ns[None, :]
    actual = interpolate_states(samples, target_ns)
    errors = predictions - actual
    joint_errors_deg = np.rad2deg(errors[..., :7])
    prediction_delta_deg = np.rad2deg(
        predictions[..., :7] - np.asarray([row["state"][:7] for row in prediction_rows])[:, None, :]
    )
    actual_delta_deg = np.rad2deg(
        actual[..., :7] - np.asarray([row["state"][:7] for row in prediction_rows])[:, None, :]
    )
    moving = np.abs(actual_delta_deg) >= 0.2
    direction_matches = np.sign(prediction_delta_deg[moving]) == np.sign(actual_delta_deg[moving])
    latencies = np.asarray([row["inference_ms"] for row in prediction_rows])

    sample_ns = np.asarray([sample.monotonic_ns for sample in samples], dtype=np.int64)
    sample_states = np.stack([sample.state for sample in samples])
    np.savez_compressed(run_dir / "actual_states.npz", monotonic_ns=sample_ns, states=sample_states)
    np.save(run_dir / "predictions.npy", predictions)
    np.save(run_dir / "aligned_actual.npy", actual)
    np.save(run_dir / "errors.npy", errors)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in prediction_rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "mode": "leader_follower_shadow_no_robot_commands",
        "prompt": args.prompt,
        "noise_seed": args.noise_seed,
        "duration_sec": args.duration,
        "predictions": len(prediction_rows),
        "state_samples": len(samples),
        "action_horizon": 16,
        "action_fps": args.action_fps,
        "inference_ms": {
            "mean": float(latencies.mean()),
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
            "max": float(latencies.max()),
        },
        "joint_error_deg": {
            "mae_all": float(np.mean(np.abs(joint_errors_deg))),
            "p95_all": float(np.quantile(np.abs(joint_errors_deg), 0.95)),
            "mae_by_joint": np.mean(np.abs(joint_errors_deg), axis=(0, 1)).tolist(),
            "mae_by_horizon": np.mean(np.abs(joint_errors_deg), axis=(0, 2)).tolist(),
            "first_step_mae": float(np.mean(np.abs(joint_errors_deg[:, 0]))),
        },
        "gripper_error": {
            "mae_all": float(np.mean(np.abs(errors[..., 7]))),
            "first_step_mae": float(np.mean(np.abs(errors[:, 0, 7]))),
        },
        "direction_match_rate_when_actual_moves_0.2deg": None
        if not direction_matches.size
        else float(np.mean(direction_matches)),
        "external_hz": external.measured_hz,
        "wrist_hz": wrist.measured_hz,
        "eth_rates_hz": eth.rates_hz(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"PASS: shadow comparison saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
