#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-toppra-retimer")

import numpy as np

from fixed_path_retimer import FixedPathRetimer, RetimeConfig


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only TOPPRAsd replay over policy chunks")
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-velocity-deg-s", type=float, default=8.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=24.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "replay_summary.json")
    args = parser.parse_args()

    retimer = FixedPathRetimer(
        RetimeConfig(
            action_hz=30.0,
            max_velocity=np.full(14, np.deg2rad(args.max_velocity_deg_s)),
            max_acceleration=np.full(14, np.deg2rad(args.max_acceleration_deg_s2)),
            arm_columns=ARM_COLUMNS,
        )
    )
    rows = []
    with args.chunks_jsonl.open() as stream:
        for line in stream:
            source = json.loads(line)
            if "actions" not in source or not source.get("accepted", True):
                continue
            result = retimer.retime(np.asarray(source["actions"], dtype=np.float64))
            rows.append(
                {
                    "chunk": source.get("chunk"),
                    "status": result.status,
                    "reason": result.reason,
                    "metrics": result.metrics,
                }
            )
            if len(rows) >= args.limit:
                break
    summary = {
        "source": str(args.chunks_jsonl),
        "chunks": len(rows),
        "retimed": sum(row["status"] == "retimed" for row in rows),
        "fallback": sum(row["status"] == "fallback_original" for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("source", "chunks", "retimed", "fallback")}, indent=2))


if __name__ == "__main__":
    main()
