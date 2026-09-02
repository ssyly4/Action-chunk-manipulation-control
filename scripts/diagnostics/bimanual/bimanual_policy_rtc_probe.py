#!/usr/bin/env python3
"""Offline probe for OpenPI Real-Time Chunking. No robot APIs are used."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np


CONTROL_ROOT = Path(__file__).resolve().parents[3]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from nero_vla.image_tools import resize_with_pad
from nero_vla.policy_client import OpenPiPolicyClient, port_open

from bimanual_policy_offline_regression import image_to_rgb_uint8, tensor_to_numpy


ACTION_HZ = 30.0
ACTION_HORIZON = 24
ACTION_DIM = 16
JOINT_INDICES = np.r_[0:7, 8:15]


def make_observation(dataset, index: int, prompt: str, noise: np.ndarray, num_steps: int) -> dict:
    item = dataset[index]
    return {
        "observation/world_image": resize_with_pad(image_to_rgb_uint8(item["observation.images.world"])),
        "observation/left_wrist_image": resize_with_pad(
            image_to_rgb_uint8(item["observation.images.left_wrist"])
        ),
        "observation/right_wrist_image": resize_with_pad(
            image_to_rgb_uint8(item["observation.images.right_wrist"])
        ),
        "observation/state": tensor_to_numpy(item["observation.state"]).astype(np.float32),
        "prompt": prompt,
        "__openpi_noise": noise,
        "__openpi_num_steps": num_steps,
    }


def joint_prefix_error_deg(candidate: np.ndarray, previous: np.ndarray, start: int, end: int) -> float:
    delta = candidate[start:end, JOINT_INDICES] - previous[start:end, JOINT_INDICES]
    return float(np.mean(np.abs(np.rad2deg(delta))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dev/nero_data/raw/towel_fold/nero_towel_50_20260817_v2"),
    )
    parser.add_argument("--repo-id", default="local/nero_towel_50_20260817_v2")
    parser.add_argument("--policy-host", default="172.24.1.154")
    parser.add_argument("--policy-port", type=int, default=8001)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--noise-seed", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()
    if not 2 <= args.action_horizon <= 64:
        parser.error("action-horizon must be in [2, 64]")
    if not 2 <= args.execution_horizon <= args.action_horizon:
        parser.error(f"execution-horizon must be in [2, {args.action_horizon}]")
    if args.num_steps <= 0:
        parser.error("num-steps must be positive")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"policy server is not reachable at {args.policy_host}:{args.policy_port}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    noise = np.random.default_rng(args.noise_seed).standard_normal(
        (args.action_horizon, 32), dtype=np.float32
    )
    client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
    try:
        first_observation = make_observation(dataset, args.dataset_index, args.prompt, noise, args.num_steps)
        first, first_timing = client.infer_timed(first_observation)
        first_actions = np.asarray(first.get("actions"), dtype=np.float32)
        if first_actions.shape != (args.action_horizon, ACTION_DIM):
            raise RuntimeError(f"invalid first action chunk: {first_actions.shape}")

        delay_steps = min(
            args.action_horizon - 1,
            math.ceil(first_timing["total_ms"] / 1000.0 * ACTION_HZ),
        )
        delayed_index = args.dataset_index + delay_steps
        vanilla, vanilla_timing = client.infer_timed(
            make_observation(dataset, delayed_index, args.prompt, noise, args.num_steps)
        )
        rtc_observation = make_observation(dataset, delayed_index, args.prompt, noise, args.num_steps)
        rtc_observation["__openpi_rtc"] = {
            "prev_chunk_left_over": first_actions[: args.execution_horizon],
            "inference_delay": delay_steps,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
        }
        rtc_result, rtc_timing = client.infer_timed(rtc_observation)
        vanilla_actions = np.asarray(vanilla.get("actions"), dtype=np.float32)
        rtc_actions = np.asarray(rtc_result.get("actions"), dtype=np.float32)
        if vanilla_actions.shape != first_actions.shape or rtc_actions.shape != first_actions.shape:
            raise RuntimeError("policy returned an invalid processed action shape")

        overlap_end = max(delay_steps + 1, args.execution_horizon)
        vanilla_error = joint_prefix_error_deg(vanilla_actions, first_actions, delay_steps, overlap_end)
        rtc_error = joint_prefix_error_deg(rtc_actions, first_actions, delay_steps, overlap_end)
        recorded_actions = np.stack(
            [
                tensor_to_numpy(dataset[delayed_index + offset]["action"])
                for offset in range(args.action_horizon)
            ]
        ).astype(np.float32)
        vanilla_demo_mae = float(
            np.mean(np.abs(np.rad2deg(vanilla_actions[:, JOINT_INDICES] - recorded_actions[:, JOINT_INDICES])))
        )
        rtc_demo_mae = float(
            np.mean(np.abs(np.rad2deg(rtc_actions[:, JOINT_INDICES] - recorded_actions[:, JOINT_INDICES])))
        )
        print("OPENPI RTC OFFLINE PROBE PASSED", flush=True)
        print(
            f"latency_ms first={first_timing['total_ms']:.1f} "
            f"vanilla={vanilla_timing['total_ms']:.1f} rtc={rtc_timing['total_ms']:.1f}",
            flush=True,
        )
        print(
            f"delay_steps={delay_steps} execution_horizon={args.execution_horizon} "
            f"remaining_after_delay={args.action_horizon - delay_steps}",
            flush=True,
        )
        print(
            f"overlap_joint_mae_deg vanilla={vanilla_error:.4f} rtc={rtc_error:.4f} "
            f"improvement={vanilla_error - rtc_error:.4f}",
            flush=True,
        )
        print(
            f"demonstration_joint_mae_deg vanilla={vanilla_demo_mae:.4f} rtc={rtc_demo_mae:.4f}",
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
