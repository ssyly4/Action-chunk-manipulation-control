#!/usr/bin/env python3
"""Warm the normal and RTC paths of an 8D right-arm OpenPI policy."""

from __future__ import annotations

import argparse

import numpy as np

from nero_vla.policy_client import OpenPiPolicyClient, port_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="172.24.1.154")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--inference-delay", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--max-guidance-weight", type=float, default=2.0)
    parser.add_argument(
        "--prompt", default="pick up the bottle and place it into the box"
    )
    args = parser.parse_args()
    if not 2 <= args.execution_horizon <= args.action_horizon:
        parser.error("execution horizon must be within the action horizon")
    if not 1 <= args.inference_delay < args.action_horizon:
        parser.error("inference delay must be within the action horizon")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError("policy server is not reachable")

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    noise = np.random.default_rng(3).standard_normal(
        (args.action_horizon, 32), dtype=np.float32
    )
    observation = {
        "observation/external_image": image,
        "observation/wrist_image": image.copy(),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": args.prompt,
        "__openpi_noise": noise,
        "__openpi_num_steps": args.num_steps,
    }
    client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
    try:
        first, first_timing = client.infer_timed(observation)
        actions = np.asarray(first.get("actions"), dtype=np.float32)
        if actions.shape != (args.action_horizon, 8):
            raise RuntimeError(f"invalid right-arm action chunk: {actions.shape}")
        rtc_observation = dict(observation)
        rtc_observation["__openpi_rtc"] = {
            "prev_chunk_left_over": actions[: args.execution_horizon],
            "inference_delay": args.inference_delay,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
        }
        rtc, rtc_timing = client.infer_timed(rtc_observation)
        rtc_actions = np.asarray(rtc.get("actions"), dtype=np.float32)
        if rtc_actions.shape != actions.shape:
            raise RuntimeError(f"invalid RTC right-arm action chunk: {rtc_actions.shape}")
        print(
            "RIGHT POLICY WARMUP PASSED "
            f"normal={first_timing['total_ms']:.1f}ms "
            f"rtc={rtc_timing['total_ms']:.1f}ms",
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
