#!/usr/bin/env python3
"""Compare live policy chunks against recorded bimanual LeRobot demonstrations.

This is intentionally offline: it reads a local LeRobot v3 dataset and uses
the policy websocket only.  It never opens CAN or instantiates an SDK robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


NERO_WS = Path("/home/dev/nero_ws")
if str(NERO_WS) not in sys.path:
    sys.path.insert(0, str(NERO_WS))

from nero_vla.image_tools import resize_with_pad
from nero_vla.policy_client import OpenPiPolicyClient, port_open


ACTION_HORIZON = 24
ACTION_DIM = 16
JOINT_INDICES = np.r_[0:7, 8:15]


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("fractions must be comma-separated values in [0, 1]")
    return values


def episode_samples(episodes: list[dict], count: int, fractions: tuple[float, ...]) -> list[tuple[dict, int]]:
    if count <= 0:
        raise ValueError("episode count must be positive")
    selected = episodes[:count]
    result: list[tuple[dict, int]] = []
    for episode in selected:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        latest = end - ACTION_HORIZON
        if latest < start:
            continue
        for fraction in fractions:
            result.append((episode, start + round((latest - start) * fraction)))
    return result


def image_to_rgb_uint8(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"expected image with three dimensions, got {array.shape}")
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {array.shape}")
    if array.dtype.kind == "f":
        array = np.clip(array * 255.0, 0.0, 255.0)
    return np.asarray(array, dtype=np.uint8)


def tensor_to_numpy(value) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def first_close_step(actions: np.ndarray, gripper_index: int, threshold: float) -> int | None:
    indices = np.flatnonzero(actions[:, gripper_index] <= threshold)
    return None if len(indices) == 0 else int(indices[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dev/nero_data/raw/towel_fold/nero_towel_50_20260817_v2"),
    )
    parser.add_argument("--repo-id", default="local/nero_towel_50_20260817_v2")
    parser.add_argument("--policy-host", default="172.24.1.154")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fractions", type=parse_csv_floats, default=(0.1, 0.35, 0.6))
    parser.add_argument("--noise-seed", type=int, default=3)
    parser.add_argument("--close-threshold", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dev/nero_ws/logs/bimanual_policy_offline_regression"),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or not 0.0 <= args.close_threshold <= 1.0:
        parser.error("episodes must be positive and close-threshold must be in [0, 1]")
    if not args.root.is_dir():
        raise RuntimeError(f"dataset root does not exist: {args.root}")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"policy server is not reachable at {args.policy_host}:{args.policy_port}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    episode_files = sorted((args.root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise RuntimeError("dataset has no episode metadata")
    episodes = []
    for path in episode_files:
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: int(row["episode_index"]))
    samples = episode_samples(episodes, args.episodes, args.fractions)
    if not samples:
        raise RuntimeError("no episodes are long enough for a 16-step action comparison")

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    run_dir = args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    fixed_noise = np.random.default_rng(args.noise_seed).standard_normal(
        (ACTION_HORIZON, 32), dtype=np.float32
    )
    records: list[dict] = []
    client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
    try:
        print(
            f"OFFLINE REGRESSION: dataset={args.root} samples={len(samples)} "
            "CAN/SDK robot commands are disabled",
            flush=True,
        )
        for sample_index, (episode, index) in enumerate(samples, start=1):
            item = dataset[index]
            state = tensor_to_numpy(item["observation.state"]).astype(np.float32)
            recorded = np.stack(
                [tensor_to_numpy(dataset[index + offset]["action"]) for offset in range(ACTION_HORIZON)]
            ).astype(np.float32)
            observation = {
                "observation/world_image": resize_with_pad(
                    image_to_rgb_uint8(item["observation.images.world"])
                ),
                "observation/left_wrist_image": resize_with_pad(
                    image_to_rgb_uint8(item["observation.images.left_wrist"])
                ),
                "observation/right_wrist_image": resize_with_pad(
                    image_to_rgb_uint8(item["observation.images.right_wrist"])
                ),
                "observation/state": state,
                "prompt": args.prompt,
                "__openpi_noise": fixed_noise,
            }
            predicted, timing = client.infer_timed(observation)
            actions = np.asarray(predicted.get("actions"), dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid policy action chunk: {actions.shape}")
            joint_error_deg = np.abs(np.rad2deg(actions[:, JOINT_INDICES] - recorded[:, JOINT_INDICES]))
            left_grip_error = np.abs(actions[:, 7] - recorded[:, 7])
            right_grip_error = np.abs(actions[:, 15] - recorded[:, 15])
            record = {
                "sample": sample_index,
                "episode_index": int(episode["episode_index"]),
                "dataset_index": index,
                "frame_index": int(tensor_to_numpy(item["frame_index"])),
                "inference_ms": float(timing["total_ms"]),
                "joint_mae_deg": float(np.mean(joint_error_deg)),
                "joint_p95_deg": float(np.quantile(joint_error_deg, 0.95)),
                "joint_max_deg": float(np.max(joint_error_deg)),
                "left_gripper_mae": float(np.mean(left_grip_error)),
                "right_gripper_mae": float(np.mean(right_grip_error)),
                "left_close_step_predicted": first_close_step(actions, 7, args.close_threshold),
                "left_close_step_recorded": first_close_step(recorded, 7, args.close_threshold),
                "right_close_step_predicted": first_close_step(actions, 15, args.close_threshold),
                "right_close_step_recorded": first_close_step(recorded, 15, args.close_threshold),
            }
            records.append(record)
            print(
                f"sample={sample_index}/{len(samples)} ep={record['episode_index']} "
                f"frame={record['frame_index']} infer={record['inference_ms']:.0f}ms "
                f"joint_mae={record['joint_mae_deg']:.2f}deg "
                f"gripR={record['right_gripper_mae']:.3f} "
                f"closeR={record['right_close_step_predicted']}/{record['right_close_step_recorded']}",
                flush=True,
            )
    finally:
        client.close()

    summary = {
        "dataset": str(args.root),
        "repo_id": args.repo_id,
        "samples": len(records),
        "episodes_requested": args.episodes,
        "fractions": list(args.fractions),
        "noise_seed": args.noise_seed,
        "joint_mae_deg": float(np.mean([row["joint_mae_deg"] for row in records])),
        "joint_p95_deg": float(np.quantile([row["joint_p95_deg"] for row in records], 0.95)),
        "right_gripper_mae": float(np.mean([row["right_gripper_mae"] for row in records])),
        "right_close_event_matches": int(
            sum(
                row["right_close_step_predicted"] == row["right_close_step_recorded"]
                for row in records
            )
        ),
    }
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PASS: offline regression complete; log={run_dir}", flush=True)


if __name__ == "__main__":
    main()
