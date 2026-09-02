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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline TOPPRAsd retiming of one policy chunk")
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("--chunk", type=int, default=0, help="zero-based accepted record index")
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--max-velocity-deg-s", type=float, default=8.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=24.0)
    parser.add_argument("--consumed-steps", type=int, default=0)
    parser.add_argument("--measured-arm", type=Path, help="JSON array with 14 measured joints in radians")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retimed_chunk.json")
    return parser.parse_args()


def load_actions(path: Path, selected: int) -> np.ndarray:
    accepted = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if "actions" in row and row.get("accepted", True):
                accepted.append(row)
    if not 0 <= selected < len(accepted):
        raise IndexError(f"chunk {selected} is outside {len(accepted)} accepted records")
    return np.asarray(accepted[selected]["actions"], dtype=np.float64)


def serializable(result, *, source: Path) -> dict:
    retiming = result.retiming if hasattr(result, "retiming") else result
    payload = {
        "source": str(source),
        "status": retiming.status,
        "feasible": retiming.feasible,
        "fallback": retiming.fallback,
        "reason": retiming.reason,
        "metrics": retiming.metrics,
        "raw_phase": retiming.raw_phase_samples.tolist(),
        "retimed_phase": retiming.phase_samples.tolist(),
        "phase_speed": retiming.phase_speed_samples.tolist(),
        "phase_acceleration": retiming.phase_acceleration_samples.tolist(),
        "commands": retiming.commands.tolist(),
        "velocity": retiming.velocity.tolist(),
        "acceleration": retiming.acceleration.tolist(),
        "jerk": retiming.jerk.tolist(),
    }
    if hasattr(result, "projection"):
        payload["rtc"] = {
            "consumed_steps": result.consumed_steps,
            "remaining_ticks": result.remaining_ticks,
            "request_boundary_tick": result.request_boundary_tick,
            "replacement_start_tick": result.replacement_start_tick,
            "replacement_end_tick": result.replacement_end_tick,
            "projection_phase": result.projection.phase,
            "projection_max_joint_error": result.projection.max_joint_error,
        }
    return payload


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    actions = load_actions(args.chunks_jsonl, args.chunk)
    if actions.shape[1] != 16:
        raise ValueError(f"expected 16-column bimanual actions, got {actions.shape}")
    config = RetimeConfig(
        action_hz=args.action_hz,
        max_velocity=np.full(14, np.deg2rad(args.max_velocity_deg_s)),
        max_acceleration=np.full(14, np.deg2rad(args.max_acceleration_deg_s2)),
        arm_columns=ARM_COLUMNS,
    )
    retimer = FixedPathRetimer(config)
    if args.consumed_steps:
        measured = (
            np.asarray(json.loads(args.measured_arm.read_text()), dtype=np.float64)
            if args.measured_arm
            else actions[args.consumed_steps, ARM_COLUMNS]
        )
        result = retimer.retime_rtc_replacement(
            actions,
            measured,
            consumed_steps=args.consumed_steps,
            emitted_at_request=1000,
        )
    else:
        result = retimer.retime(actions)
    payload = serializable(result, source=args.chunks_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({"output": str(args.output), **{k: payload[k] for k in ("status", "feasible", "fallback", "reason")}, "metrics": payload["metrics"]}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
