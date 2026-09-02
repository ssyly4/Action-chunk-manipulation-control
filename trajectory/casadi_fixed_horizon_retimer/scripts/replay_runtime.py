#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from fixed_phase_optimizer import CasadiPhaseOptimizer, PhaseOptimizerConfig
from fixed_phase_optimizer.optimizer import NaturalCubicPath


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arm_matrix(actions: np.ndarray) -> np.ndarray:
    return np.concatenate((actions[:, :7], actions[:, 8:15]), axis=1)


def combined_tick_vector(tick: dict, suffix: str) -> np.ndarray:
    return np.asarray(tick[f"left_{suffix}"] + tick[f"right_{suffix}"], dtype=np.float64)


def inherited_phase_speed(
    actions: np.ndarray,
    phase: float,
    arm_velocity: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    curvature_margin: float,
) -> float:
    path = NaturalCubicPath(arm_matrix(actions))
    tangent = path.evaluate(phase, 1)
    norm_squared = float(np.dot(tangent, tangent))
    if norm_squared <= 1e-12:
        return 0.0
    speed = max(0.0, float(np.dot(tangent, arm_velocity) / norm_squared))
    moving = np.abs(tangent) > 1e-10
    speed = min(speed, 0.95 * float(np.min(max_velocity[moving] / np.abs(tangent[moving]))))
    curvature = np.abs(path.evaluate(phase, 2))
    curved = curvature > 1e-10
    if np.any(curved):
        speed = min(
            speed,
            curvature_margin
            * float(np.min(np.sqrt(max_acceleration[curved] / curvature[curved]))),
        )
    return speed


def percentile(values: list[float], quantile: float) -> float | None:
    return None if not values else float(np.percentile(values, quantile))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay real RTC chunks through CasADi")
    parser.add_argument("policy_run", type=Path)
    parser.add_argument("retimer_log", type=Path)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-velocity-deg-s", type=float, default=25.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=220.0)
    parser.add_argument("--max-jerk-deg-s3", type=float, default=12000.0)
    parser.add_argument("--max-solve-sec", type=float, default=0.2)
    parser.add_argument("--warm-start-duals", action="store_true")
    parser.add_argument("--curvature-margin", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    chunks = load_jsonl(args.policy_run / "chunks.jsonl")
    ticks = load_jsonl(args.policy_run / "ticks.jsonl")
    decisions = load_jsonl(args.retimer_log)
    if args.max_chunks is not None:
        decisions = decisions[: args.max_chunks]

    vmax = np.full(14, np.deg2rad(args.max_velocity_deg_s))
    amax = np.full(14, np.deg2rad(args.max_acceleration_deg_s2))
    jmax = np.full(14, np.deg2rad(args.max_jerk_deg_s3))
    optimizer = CasadiPhaseOptimizer(
        PhaseOptimizerConfig(
            action_hz=30.0,
            max_velocity=vmax,
            max_acceleration=amax,
            max_jerk=jmax,
            arm_columns=ARM_COLUMNS,
            solver_max_cpu_sec=args.max_solve_sec,
            solver_warm_start_duals=args.warm_start_duals,
        )
    )

    output = args.output or ROOT / "outputs" / f"runtime_replay_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with output.open("w") as stream:
        for decision in decisions:
            generation = int(decision["generation"])
            if generation >= len(chunks):
                raise RuntimeError(f"missing policy chunk {generation}")
            tick_index = min(int(decision["loaded_at_tick"]), len(ticks) - 1)
            tick = ticks[tick_index]
            raw = np.asarray(chunks[generation]["actions"], dtype=np.float64)
            anchor = combined_tick_vector(tick, "command")
            initial_velocity = np.deg2rad(combined_tick_vector(tick, "command_velocity_deg_s"))
            gain = float(decision["action_gain"])
            scaled = raw.copy()
            scaled[:, ARM_COLUMNS] = anchor + gain * (arm_matrix(raw) - anchor)
            consumed = int(decision["skip_steps"])
            start_speed = inherited_phase_speed(
                scaled,
                float(consumed),
                initial_velocity,
                vmax,
                amax,
                args.curvature_margin,
            )
            result = optimizer.optimize(
                scaled,
                start_phase=float(consumed),
                output_ticks=len(scaled) - consumed,
                start_phase_speed=start_speed,
                fallback_start_index=consumed,
            )
            row = {
                "generation": generation,
                "skip_steps": consumed,
                "action_gain": gain,
                "toppra_status": decision["retime_status"],
                "casadi_status": result.status,
                "casadi_reason": result.reason,
                "solve_ms": result.solve_ms,
                **result.metrics,
            }
            results.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"generation={generation:02d} TOPPRA={decision['retime_status']:<17} "
                f"CasADi={result.status:<17} solve={result.solve_ms:7.1f}ms"
            )

    optimized = [row for row in results if row["casadi_status"] == "optimized"]
    solve_times = [float(row["solve_ms"]) for row in results]
    summary = {
        "chunks": len(results),
        "optimized": len(optimized),
        "fallback": len(results) - len(optimized),
        "toppra_retimed": sum(row["toppra_status"] == "retimed" for row in results),
        "casadi_recovered_toppra_fallback": sum(
            row["toppra_status"] != "retimed" and row["casadi_status"] == "optimized"
            for row in results
        ),
        "solve_ms_median": percentile(solve_times, 50),
        "solve_ms_p95": percentile(solve_times, 95),
        "phase_distortion_rms_steps_p95": percentile(
            [float(row["phase_distortion_rms_steps"]) for row in optimized], 95
        ),
        "velocity_ratio_max": max(
            (float(row["velocity_ratio"]) for row in optimized), default=None
        ),
        "acceleration_ratio_max": max(
            (float(row["acceleration_ratio"]) for row in optimized), default=None
        ),
        "jerk_ratio_max": max(
            (float(row["jerk_ratio"]) for row in optimized), default=None
        ),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
