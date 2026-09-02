"""Replay a recorded NERO episode through the offline trajectory executor."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from nero_vla.trajectory_executor import ActionChunkBuffer
from nero_vla.trajectory_executor import RateLimitedJointFollower


def load_episode(parquet_path: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["episode_index", "timestamp", "action"])
    data = table.to_pydict()
    selected = [index for index, value in enumerate(data["episode_index"]) if value == episode]
    if len(selected) < 20:
        raise RuntimeError(f"episode {episode} has only {len(selected)} frames")
    timestamps = np.asarray([data["timestamp"][index] for index in selected], dtype=np.float64)
    actions = np.asarray([data["action"][index][:7] for index in selected], dtype=np.float64)
    order = np.argsort(timestamps)
    return timestamps[order], actions[order]


def run_replay(
    timestamps: np.ndarray,
    actions: np.ndarray,
    *,
    duration_sec: float,
    control_hz: float,
    inference_interval_sec: float,
    inference_latency_sec: float,
    horizon: int,
    max_velocity_deg_s: float,
    max_acceleration_deg_s2: float,
) -> dict:
    source_period = float(np.median(np.diff(timestamps)))
    source_hz = 1.0 / source_period
    end_time = min(float(timestamps[-1]), float(duration_sec))
    if end_time <= inference_latency_sec + source_period * horizon:
        raise ValueError("duration is too short for the requested latency and horizon")

    observation_indices = []
    next_observation = float(timestamps[0])
    for index, timestamp in enumerate(timestamps):
        if timestamp + 1e-9 >= next_observation and index + horizon + 1 <= len(actions):
            observation_indices.append(index)
            next_observation += inference_interval_sec
        if timestamp >= end_time:
            break
    arrivals = [
        (
            float(timestamps[index] + inference_latency_sec),
            float(timestamps[index]),
            actions[index + 1:index + 1 + horizon],
        )
        for index in observation_indices
    ]

    buffer = ActionChunkBuffer(
        action_hz=source_hz,
        blend_duration_sec=0.10,
        stale_after_sec=0.10,
        first_action_offset_steps=1,
    )
    follower = RateLimitedJointFollower(
        max_velocity_rad_s=np.deg2rad(max_velocity_deg_s),
        max_acceleration_rad_s2=np.deg2rad(max_acceleration_deg_s2),
        # Offline replay measures lag instead of rejecting fast demonstrations.
        # A real backend must use a separately qualified, much tighter guard.
        max_target_feedback_error_rad=np.deg2rad(180.0),
        max_command_feedback_error_rad=np.deg2rad(2.0),
        max_tick_interval_sec=2.5 / control_hz,
    )
    initial = actions[0].copy()
    follower.initialize(initial, now=0.0)
    feedback = initial.copy()
    commands = [initial.copy()]
    velocities = [np.zeros(7, dtype=np.float64)]
    tracking_errors_deg = []
    statuses: Counter[str] = Counter()
    arrival_index = 0
    period = 1.0 / control_hz
    tick = period
    while tick <= end_time + 1e-9:
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= tick + 1e-9:
            received_at, observed_at, chunk = arrivals[arrival_index]
            buffer.push(chunk, observed_at=observed_at, received_at=received_at)
            arrival_index += 1
        trajectory_sample = buffer.sample(tick)
        result = follower.step(trajectory_sample, measured=feedback, now=tick)
        feedback = result.command.copy()  # Ideal mock backend; no hardware is contacted.
        commands.append(result.command)
        velocities.append(result.velocity)
        if result.desired is not None:
            tracking_errors_deg.append(
                float(np.max(np.abs(np.rad2deg(result.desired - result.command))))
            )
        statuses[result.status] += 1
        tick += period

    commands_array = np.stack(commands)
    velocities_array = np.stack(velocities)
    acceleration = np.diff(velocities_array, axis=0) * control_hz
    legacy_targets = []
    lead_index = min(horizon - 1, int(math.ceil(inference_latency_sec * source_hz)))
    for _, _, chunk in arrivals:
        legacy_targets.append(chunk[lead_index])
    legacy_jump = np.diff(np.stack(legacy_targets), axis=0) if len(legacy_targets) > 1 else np.zeros((0, 7))

    return {
        "hardware_commands_sent": 0,
        "source_hz": source_hz,
        "control_hz": control_hz,
        "duration_sec": end_time,
        "chunks": len(arrivals),
        "status_counts": dict(statuses),
        "max_command_step_deg": float(np.max(np.abs(np.rad2deg(np.diff(commands_array, axis=0))))),
        "max_velocity_deg_s": float(np.max(np.abs(np.rad2deg(velocities_array)))),
        "max_acceleration_deg_s2": float(np.max(np.abs(np.rad2deg(acceleration)))),
        "max_tracking_error_deg": float(max(tracking_errors_deg, default=0.0)),
        "legacy_replan_target_jump_deg": float(
            0.0 if legacy_jump.size == 0 else np.max(np.abs(np.rad2deg(legacy_jump)))
        ),
        "final_command_rad": commands_array[-1].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline NERO trajectory executor replay")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path.home()
        / "nero_ws/data/nero_pick_water_bottle_50_v2/data/chunk-000/file-000.parquet",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--inference-interval", type=float, default=0.22)
    parser.add_argument("--inference-latency", type=float, default=0.22)
    parser.add_argument("--horizon", type=int, default=16)
    # Replay defaults cover the recorded dataset distribution.  They are not
    # approved real-robot limits; hardware qualification must choose those.
    parser.add_argument("--max-velocity-deg-s", type=float, default=90.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.duration, args.control_hz, args.inference_interval, args.horizon) <= 0:
        parser.error("duration, rates, and horizon must be positive")
    if args.inference_latency < 0:
        parser.error("inference latency must be non-negative")

    timestamps, actions = load_episode(args.parquet, args.episode)
    summary = run_replay(
        timestamps,
        actions,
        duration_sec=args.duration,
        control_hz=args.control_hz,
        inference_interval_sec=args.inference_interval,
        inference_latency_sec=args.inference_latency,
        horizon=args.horizon,
        max_velocity_deg_s=args.max_velocity_deg_s,
        max_acceleration_deg_s2=args.max_acceleration_deg_s2,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
