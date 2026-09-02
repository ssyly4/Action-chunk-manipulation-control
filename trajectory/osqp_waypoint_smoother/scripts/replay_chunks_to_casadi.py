#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CASADI_ROOT = PROJECT_ROOT / "casadi_fixed_horizon_retimer"
for path in (
    ROOT / "vendor",
    ROOT,
    CASADI_ROOT / "vendor",
    CASADI_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from fixed_phase_optimizer import CasadiPhaseOptimizer, PhaseOptimizerConfig
from waypoint_smoother import OsqpWaypointSmoother, WaypointSmootherConfig


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw->CasADi with OSQP->CasADi on recorded policy chunks"
    )
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--trust-deg", type=float, default=0.3)
    parser.add_argument("--max-velocity-deg-s", type=float, default=28.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=280.0)
    parser.add_argument("--max-jerk-deg-s3", type=float, default=8000.0)
    parser.add_argument("--casadi-timeout-sec", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.floor(quantile * len(ordered))))
    return float(ordered[index])


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    smooth_ok = [row for row in rows if row["osqp_feasible"]]
    baseline_ok = [row for row in rows if row["baseline_casadi_feasible"]]
    pipeline_ok = [row for row in rows if row["pipeline_casadi_feasible"]]
    solve_ms = [float(row["osqp_solve_ms"]) for row in rows]
    deviations = [float(row["max_deviation_deg"]) for row in smooth_ok]
    return {
        "chunks": len(rows),
        "osqp_feasible": len(smooth_ok),
        "baseline_casadi_feasible": len(baseline_ok),
        "pipeline_casadi_feasible": len(pipeline_ok),
        "osqp_solve_ms_mean": statistics.fmean(solve_ms) if solve_ms else None,
        "osqp_solve_ms_p90": percentile(solve_ms, 0.9),
        "max_deviation_deg_mean": (
            statistics.fmean(deviations) if deviations else None
        ),
        "max_deviation_deg_max": max(deviations, default=None),
        "raw_jerk_ratio_mean": statistics.fmean(
            float(row["raw_jerk_ratio"]) for row in rows
        ),
        "osqp_jerk_ratio_mean": statistics.fmean(
            float(row["osqp_jerk_ratio"]) for row in rows
        ),
        "baseline_casadi_jerk_ratio_mean": statistics.fmean(
            float(row["baseline_casadi_jerk_ratio"]) for row in rows
        ),
        "pipeline_casadi_jerk_ratio_mean": statistics.fmean(
            float(row["pipeline_casadi_jerk_ratio"]) for row in rows
        ),
    }


def jerk_ratio(result: object, limits: np.ndarray) -> float:
    jerk = np.asarray(result.jerk, dtype=np.float64)
    if jerk.size == 0:
        return 0.0
    return float(np.max(np.abs(jerk) / limits))


def main() -> None:
    args = parse_args()
    vmax = np.full(14, np.deg2rad(args.max_velocity_deg_s))
    amax = np.full(14, np.deg2rad(args.max_acceleration_deg_s2))
    jmax = np.full(14, np.deg2rad(args.max_jerk_deg_s3))
    smoother = OsqpWaypointSmoother(
        WaypointSmootherConfig(
            action_hz=args.action_hz,
            arm_columns=ARM_COLUMNS,
            trust_region=np.deg2rad(args.trust_deg),
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
        )
    )
    phase_config = PhaseOptimizerConfig(
        action_hz=args.action_hz,
        max_velocity=vmax,
        max_acceleration=amax,
        max_jerk=jmax,
        arm_columns=ARM_COLUMNS,
        solver_max_cpu_sec=args.casadi_timeout_sec,
        solver_warm_start_duals=True,
    )
    baseline_optimizer = CasadiPhaseOptimizer(phase_config)
    pipeline_optimizer = CasadiPhaseOptimizer(phase_config)
    rows: list[dict[str, object]] = []

    with args.chunks.open() as stream:
        for line in stream:
            if args.max_chunks and len(rows) >= args.max_chunks:
                break
            record = json.loads(line)
            actions = np.asarray(record["actions"], dtype=np.float64)
            if actions.shape[1] != 16 or len(actions) < 4:
                continue
            smoothed = smoother.smooth(actions)
            baseline = baseline_optimizer.optimize(actions)
            pipeline = pipeline_optimizer.optimize(smoothed.commands)
            rows.append(
                {
                    "chunk": int(record.get("chunk", len(rows))),
                    "osqp_status": smoothed.status,
                    "osqp_feasible": smoothed.feasible,
                    "osqp_reason": smoothed.reason,
                    "osqp_solve_ms": smoothed.solve_ms,
                    "max_deviation_deg": float(
                        np.rad2deg(smoothed.metrics["max_waypoint_deviation_rad"])
                    ),
                    "raw_jerk_ratio": smoothed.metrics["raw_jerk_ratio"],
                    "osqp_jerk_ratio": smoothed.metrics["smoothed_jerk_ratio"],
                    "baseline_casadi_feasible": baseline.feasible,
                    "baseline_casadi_jerk_ratio": jerk_ratio(baseline, jmax),
                    "pipeline_casadi_feasible": pipeline.feasible,
                    "pipeline_casadi_jerk_ratio": jerk_ratio(pipeline, jmax),
                }
            )

    report = {"summary": summary(rows), "chunks": rows}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
