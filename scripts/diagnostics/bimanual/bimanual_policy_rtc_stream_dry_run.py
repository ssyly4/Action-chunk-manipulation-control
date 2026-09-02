#!/usr/bin/env python3
"""Exercise the RTC queue against the live policy server without robot APIs."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
from pathlib import Path
import sys
import time

import numpy as np


NERO_WS = Path("/home/dev/nero_ws")
if str(NERO_WS) not in sys.path:
    sys.path.insert(0, str(NERO_WS))

from nero_vla.bimanual_chunk_executor import BimanualRtcActionQueue
from nero_vla.policy_client import OpenPiPolicyClient, port_open

from bimanual_policy_rtc_probe import ACTION_HORIZON, ACTION_HZ, make_observation


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
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dataset-index", type=int, default=800)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--queue-threshold", type=int, default=22)
    parser.add_argument("--action-hz", type=float, default=25.0)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--max-guidance-weight", type=float, default=1.0)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--noise-seed", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.execution_horizon <= ACTION_HORIZON:
        parser.error("execution-horizon must be in [2, 16]")
    if not 1 <= args.queue_threshold < ACTION_HORIZON:
        parser.error("queue-threshold must be in [1, 15]")
    if not 20.0 <= args.action_hz <= ACTION_HZ:
        parser.error("action-hz must be in [20, 30]")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"policy server is not reachable at {args.policy_host}:{args.policy_port}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    noise = np.random.default_rng(args.noise_seed).standard_normal((ACTION_HORIZON, 32), dtype=np.float32)
    base_observation = make_observation(dataset, args.dataset_index, args.prompt, noise, args.num_steps)
    client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
    queue = BimanualRtcActionQueue(action_hz=args.action_hz)
    latency_history: deque[float] = deque(maxlen=8)
    latency_ms: list[float] = []
    skip_history: list[int] = []
    status_counts: Counter[str] = Counter()
    min_remaining = ACTION_HORIZON
    request_count = 0
    future: Future | None = None
    request = None

    try:
        first, timing = client.infer_timed(dict(base_observation))
        first_actions = np.asarray(first.get("actions"), dtype=np.float64)
        queue.load(first_actions)
        latency_history.append(timing["total_ms"] / 1000.0)
        latency_ms.append(float(timing["total_ms"]))
        started = time.monotonic()
        next_tick = started
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-dry-infer") as pool:
            while time.monotonic() - started < args.duration:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                next_tick += 1.0 / ACTION_HZ

                if future is not None and future.done():
                    result, timing = future.result()
                    future = None
                    if request is None:
                        raise RuntimeError("RTC dry-run response has no request")
                    consumed = queue.consumed_since(request)
                    elapsed = math.ceil((now - request.requested_at) * args.action_hz)
                    skip_steps = max(consumed, elapsed)
                    if skip_steps >= ACTION_HORIZON:
                        raise RuntimeError(f"RTC queue exhausted: elapsed={elapsed} consumed={consumed}")
                    queue.load(np.asarray(result["actions"], dtype=np.float64), skip_steps=skip_steps)
                    latency_history.append(timing["total_ms"] / 1000.0)
                    latency_ms.append(float(timing["total_ms"]))
                    skip_history.append(skip_steps)
                    request = None

                sample = queue.sample(now, feedback=np.zeros(14))
                status_counts[sample.status] += 1
                min_remaining = min(min_remaining, queue.remaining_steps)

                if future is None and queue.remaining_steps <= args.queue_threshold:
                    predicted_delay = min(
                        ACTION_HORIZON - 1,
                        max(1, math.ceil(max(latency_history) * args.action_hz)),
                    )
                    request = queue.make_request(
                        now=now,
                        execution_horizon=args.execution_horizon,
                        predicted_delay_steps=predicted_delay,
                    )
                    observation = dict(base_observation)
                    observation["__openpi_rtc"] = {
                        "prev_chunk_left_over": request.previous_actions,
                        "inference_delay": request.predicted_delay_steps,
                        "execution_horizon": request.valid_previous_steps,
                        "max_guidance_weight": args.max_guidance_weight,
                    }
                    future = pool.submit(client.infer_timed, observation)
                    request_count += 1

            if future is not None:
                future.result(timeout=10.0)
    finally:
        client.close()

    queue_holds = status_counts["rtc_queue_hold"]
    print("RTC STREAM DRY RUN COMPLETE: no CAN/SDK/robot command was used", flush=True)
    print(
        f"duration={args.duration:.1f}s requests={request_count} "
        f"latency_ms median={np.median(latency_ms):.1f} max={np.max(latency_ms):.1f}",
        flush=True,
    )
    print(
        f"skip_steps={skip_history} min_remaining={min_remaining} "
        f"queue_holds={queue_holds} status_counts={dict(status_counts)}",
        flush=True,
    )
    if queue_holds:
        raise RuntimeError(f"RTC queue starved for {queue_holds} control ticks")
    print("PASS: RTC queue sustained the 30 Hz action consumer", flush=True)


if __name__ == "__main__":
    main()
