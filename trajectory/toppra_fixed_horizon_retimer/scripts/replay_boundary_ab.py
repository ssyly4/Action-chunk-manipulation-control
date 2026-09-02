#!/usr/bin/env python3
"""Replay hard-anchor A versus path-preserving B on one production run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT.parents[1]
for path in (ROOT / "vendor", ROOT, ROOT / "ab_runtime", CONTROL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixed_path_retimer import FixedPathRetimer, RecedingConfig, RecedingFixedPathRetimer, RetimeConfig
from nero_vla.bimanual_chunk_executor import arm_matrix
from receding_toppra_queue import ARM_COLUMNS, scale_actions_to_envelope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_run", type=Path)
    parser.add_argument("decision_log", type=Path)
    parser.add_argument("--max-velocity-deg-s", type=float, default=25.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=220.0)
    parser.add_argument("--curvature-margin", type=float, default=0.8)
    return parser.parse_args()


def classify(plan) -> str:
    if plan.retiming.status == "retimed":
        return "retimed"
    reason = plan.retiming.reason or ""
    if "no controllable parameterization" in reason:
        return "no_controllable_start"
    if "returned" in reason and "requested" in reason:
        return "duration_mismatch"
    return "other_fallback"


def main() -> None:
    args = parse_args()
    chunks = [json.loads(line) for line in (args.production_run / "chunks.jsonl").read_text().splitlines()]
    ticks = [json.loads(line) for line in (args.production_run / "ticks.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in args.decision_log.read_text().splitlines()]
    if len(chunks) != len(decisions):
        raise RuntimeError(f"chunk/decision count mismatch: {len(chunks)} != {len(decisions)}")
    ticks_by_phase = {
        int(round(row["phase_steps"])): row
        for row in ticks
        if row.get("phase_steps") is not None
    }
    velocity_limit = np.deg2rad(args.max_velocity_deg_s)
    acceleration_limit = np.deg2rad(args.max_acceleration_deg_s2)
    engine = RecedingFixedPathRetimer(
        FixedPathRetimer(
            RetimeConfig(
                action_hz=30.0,
                max_velocity=np.full(14, velocity_limit),
                max_acceleration=np.full(14, acceleration_limit),
                arm_columns=ARM_COLUMNS,
            )
        ),
        RecedingConfig(),
    )
    counts = {mode: Counter() for mode in ("A_hard_anchor", "B_path_curvature")}
    b_start_errors = []
    for decision, chunk in zip(decisions, chunks, strict=True):
        skip = int(decision["skip_steps"])
        loaded = int(decision["loaded_at_tick"])
        tick = ticks_by_phase.get(max(0, loaded - 1), ticks[0])
        anchor = np.asarray(tick["left_command"] + tick["right_command"], dtype=np.float64)
        velocity = np.deg2rad(
            np.asarray(
                tick["left_command_velocity_deg_s"] + tick["right_command_velocity_deg_s"],
                dtype=np.float64,
            )
        )
        raw = np.asarray(chunk["actions"], dtype=np.float64)
        for mode, hard_anchor in (("A_hard_anchor", True), ("B_path_curvature", False)):
            gain = scale_actions_to_envelope(
                raw,
                anchor=anchor,
                initial_velocity=velocity,
                action_hz=30.0,
                max_velocity=velocity_limit,
                max_acceleration=acceleration_limit,
                minimum_gain=0.5,
                gain_step=0.025,
                previous_gain=float(decision["action_gain"]),
                maximum_gain_rise=0.1,
                start_index=skip,
                anchor_start_waypoint=hard_anchor,
            )
            if not hard_anchor:
                b_start_errors.append(
                    float(
                        np.rad2deg(
                            np.max(np.abs(arm_matrix(gain.actions)[skip] - anchor))
                        )
                    )
                )
            plan = engine.plan(
                gain.actions,
                start_wall_tick=loaded,
                consumed_steps=skip,
                start_arm_velocity=velocity,
                start_speed_curvature_margin=(
                    None if hard_anchor else args.curvature_margin
                ),
            )
            counts[mode][classify(plan)] += 1
    print(f"run={args.production_run}")
    print(f"decisions={len(decisions)} limits={args.max_velocity_deg_s:g}/{args.max_acceleration_deg_s2:g}")
    for mode, result in counts.items():
        print(f"{mode}: {dict(result)}")
    print(
        "B_start_error_deg: "
        f"median={np.percentile(b_start_errors, 50):.3f} "
        f"p95={np.percentile(b_start_errors, 95):.3f} "
        f"max={np.max(b_start_errors):.3f}"
    )


if __name__ == "__main__":
    main()
