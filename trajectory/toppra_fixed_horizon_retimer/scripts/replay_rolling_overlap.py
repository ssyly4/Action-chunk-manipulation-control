#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-toppra-retimer")

import numpy as np

from fixed_path_retimer import (
    FixedPathRetimer,
    RetimeConfig,
    RollingConfig,
    RollingFixedPathRetimer,
)


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def default_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise RTC overlap/splice replay")
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("--commit-ticks", type=int, default=12)
    parser.add_argument("--max-velocity-deg-s", type=float, default=32.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=400.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rolling_overlap.json")
    args = parser.parse_args()

    rows = []
    with args.chunks_jsonl.open() as stream:
        for line in stream:
            row = json.loads(line)
            if "actions" in row and row.get("accepted", True):
                rows.append(row)
    if len(rows) < 2:
        raise RuntimeError("at least two accepted chunks are required")
    base_ns = int(rows[0]["observation_ns"])
    hz = 30.0
    engine = RollingFixedPathRetimer(
        FixedPathRetimer(
            RetimeConfig(
                action_hz=hz,
                max_velocity=np.full(14, np.deg2rad(args.max_velocity_deg_s)),
                max_acceleration=np.full(14, np.deg2rad(args.max_acceleration_deg_s2)),
                arm_columns=ARM_COLUMNS,
            )
        ),
        RollingConfig(commit_ticks=args.commit_ticks),
    )

    plans = []
    for row in rows:
        start_tick = round((int(row["observation_ns"]) - base_ns) * hz / 1e9)
        plans.append(
            engine.plan(
                np.asarray(row["actions"], dtype=np.float64),
                start_wall_tick=start_tick,
            )
        )

    pairs = []
    for old_row, new_row, old_plan, new_plan in zip(rows, rows[1:], plans, plans[1:]):
        response_tick = new_plan.start_wall_tick + math.ceil(
            float(new_row["observation_to_response_ms"]) * hz / 1000.0
        )
        try:
            splice = engine.find_overlap_splice(
                old_plan,
                new_plan,
                earliest_wall_tick=response_tick,
                latest_wall_tick=(
                    new_plan.optimization_end_wall_tick - args.commit_ticks
                ),
            )
            payload = {
                "old_chunk": old_row.get("chunk"),
                "new_chunk": new_row.get("chunk"),
                "response_tick": response_tick,
                **splice.__dict__,
            }
        except RuntimeError as exc:
            payload = {
                "old_chunk": old_row.get("chunk"),
                "new_chunk": new_row.get("chunk"),
                "response_tick": response_tick,
                "accepted": False,
                "reason": str(exc),
            }
        pairs.append(payload)

    output = {
        "source": str(args.chunks_jsonl),
        "chunks": len(rows),
        "retimed_chunks": sum(plan.retiming.status == "retimed" for plan in plans),
        "fallback_chunks": sum(plan.retiming.fallback for plan in plans),
        "overlap_pairs": len(pairs),
        "accepted_splices": sum(bool(pair.get("accepted")) for pair in pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=default_json) + "\n")
    print(
        json.dumps(
            {key: output[key] for key in ("chunks", "retimed_chunks", "fallback_chunks", "overlap_pairs", "accepted_splices")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
