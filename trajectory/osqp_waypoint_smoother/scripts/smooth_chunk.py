#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from waypoint_smoother import OsqpWaypointSmoother, WaypointSmootherConfig


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OSQP waypoint smoothing probe")
    parser.add_argument("input", type=Path, help="Hx16 JSON array or NumPy .npy chunk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--trust-deg", type=float, default=0.3)
    parser.add_argument("--max-velocity-deg-s", type=float, default=28.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=280.0)
    parser.add_argument("--max-jerk-deg-s3", type=float, default=8000.0)
    return parser.parse_args()


def load_actions(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    return np.asarray(json.loads(path.read_text()), dtype=np.float64)


def main() -> None:
    args = parse_args()
    actions = load_actions(args.input)
    smoother = OsqpWaypointSmoother(
        WaypointSmootherConfig(
            action_hz=args.action_hz,
            arm_columns=ARM_COLUMNS,
            trust_region=np.deg2rad(args.trust_deg),
            max_velocity=np.deg2rad(args.max_velocity_deg_s),
            max_acceleration=np.deg2rad(args.max_acceleration_deg_s2),
            max_jerk=np.deg2rad(args.max_jerk_deg_s3),
        )
    )
    result = smoother.smooth(actions)
    report = {
        "status": result.status,
        "feasible": result.feasible,
        "fallback": result.fallback,
        "reason": result.reason,
        "solve_ms": result.solve_ms,
        "iterations": result.iterations,
        "metrics": result.metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, result.commands)


if __name__ == "__main__":
    main()
