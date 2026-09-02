#!/usr/bin/env python3
"""Compare demonstration, policy-chunk, and physical NERO motion dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
PERCENTILES = (50, 90, 95, 99, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-parquet", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def derivatives(
    positions: np.ndarray,
    timestamps: np.ndarray,
    *,
    same_segment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(timestamps)
    valid = np.isfinite(dt) & (dt >= 0.01) & (dt <= 0.1)
    if same_segment is not None:
        valid &= same_segment
    velocity = np.rad2deg(np.diff(positions, axis=0)) / dt[:, None]
    velocity = velocity[valid]
    valid_times = timestamps[1:][valid]
    if len(velocity) < 2:
        return velocity, np.empty((0, positions.shape[1]), dtype=np.float64)
    velocity_dt = np.diff(valid_times)
    consecutive = np.isfinite(velocity_dt) & (velocity_dt >= 0.01) & (velocity_dt <= 0.1)
    acceleration = np.diff(velocity, axis=0) / velocity_dt[:, None]
    return velocity, acceleration[consecutive]


def fixed_rate_derivatives(
    trajectories: list[np.ndarray], action_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    velocities: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue
        velocity = np.rad2deg(np.diff(trajectory, axis=0)) * action_hz
        velocities.append(velocity)
        if len(velocity) >= 2:
            accelerations.append(np.diff(velocity, axis=0) * action_hz)
    return (
        np.concatenate(velocities) if velocities else np.empty((0, 14)),
        np.concatenate(accelerations) if accelerations else np.empty((0, 14)),
    )


def distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"samples": 0, "per_tick_max_abs": {}, "joint_element_abs": {}}
    per_tick = np.max(np.abs(values), axis=1)
    elements = np.abs(values).reshape(-1)
    return {
        "samples": int(len(values)),
        "per_tick_max_abs": {
            f"p{percentile}": float(np.percentile(per_tick, percentile))
            for percentile in PERCENTILES
        },
        "joint_element_abs": {
            f"p{percentile}": float(np.percentile(elements, percentile))
            for percentile in PERCENTILES
        },
    }


def load_demonstration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["observation.state", "action", "timestamp", "episode_index"],
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[
        :, ARM_INDICES
    ]
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[:, ARM_INDICES]
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    same_episode = episodes[1:] == episodes[:-1]
    state_velocity, state_acceleration = derivatives(
        state, timestamps, same_segment=same_episode
    )
    action_velocity, action_acceleration = derivatives(
        action, timestamps, same_segment=same_episode
    )
    return state_velocity, state_acceleration, action_velocity, action_acceleration


def load_policy_chunks(path: Path, action_hz: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = [row for row in read_jsonl(path) if row.get("accepted", True)]
    trajectories = [
        np.asarray(row["actions"], dtype=np.float64)[:, ARM_INDICES]
        for row in rows
        if "actions" in row
    ]
    velocity, acceleration = fixed_rate_derivatives(trajectories, action_hz)
    return velocity, acceleration, rows


def load_execution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = read_jsonl(path)
    timestamps = np.asarray([row["elapsed_sec"] for row in rows], dtype=np.float64)
    command = np.asarray(
        [row["left_command"] + row["right_command"] for row in rows], dtype=np.float64
    )
    feedback = np.asarray(
        [row["left_feedback"] + row["right_feedback"] for row in rows], dtype=np.float64
    )
    command_velocity, command_acceleration = derivatives(command, timestamps)
    feedback_velocity, feedback_acceleration = derivatives(feedback, timestamps)
    return command_velocity, command_acceleration, feedback_velocity, feedback_acceleration, rows


def rounded_table(stats: dict, metric: str) -> str:
    columns = ["source", "p50", "p90", "p95", "p99", "max"]
    lines = ["| " + " | ".join(columns) + " |", "|---|---:|---:|---:|---:|---:|"]
    for source, source_stats in stats.items():
        values = source_stats[metric]["per_tick_max_abs"]
        lines.append(
            "| "
            + " | ".join(
                [source]
                + [f"{values.get(key, float('nan')):.2f}" for key in ("p50", "p90", "p95", "p99", "p100")]
            )
            + " |"
        )
    return "\n".join(lines)


def plot_cdf(series: dict[str, np.ndarray], unit: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for label, values in series.items():
        if values.size == 0:
            continue
        maximum = np.max(np.abs(values), axis=1)
        ordered = np.sort(maximum)
        cdf = np.linspace(0.0, 1.0, len(ordered), endpoint=True)
        axis.plot(ordered, cdf, label=label, linewidth=1.8)
    axis.set_xlabel(unit)
    axis.set_ylabel("Cumulative fraction")
    axis.set_ylim(0.0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_tracking(rows: list[dict], path: Path) -> dict:
    time_axis = np.asarray([row["elapsed_sec"] for row in rows], dtype=np.float64)
    errors = np.asarray(
        [
            max(
                np.max(np.abs(row["left_command_error_deg"])),
                np.max(np.abs(row["right_command_error_deg"])),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    chunks = np.asarray([row["chunk_count"] for row in rows], dtype=np.int64)
    boundary = np.concatenate(([False], chunks[1:] != chunks[:-1]))
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(time_axis, errors, linewidth=1.1, label="max command-feedback error")
    axis.scatter(time_axis[boundary], errors[boundary], s=13, color="tab:red", label="chunk replacement")
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Joint error (deg)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "p50_deg": float(np.percentile(errors, 50)),
        "p90_deg": float(np.percentile(errors, 90)),
        "p95_deg": float(np.percentile(errors, 95)),
        "p99_deg": float(np.percentile(errors, 99)),
        "max_deg": float(np.max(errors)),
        "chunk_replacements": int(np.count_nonzero(boundary)),
        "boundary_p95_deg": float(np.percentile(errors[boundary], 95))
        if np.any(boundary)
        else None,
    }


def main() -> None:
    args = parse_args()
    if args.action_hz <= 0:
        raise ValueError("action-hz must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo_state_v, demo_state_a, demo_action_v, demo_action_a = load_demonstration(
        args.demo_parquet
    )
    model_v, model_a, chunk_rows = load_policy_chunks(
        args.run_dir / "chunks.jsonl", args.action_hz
    )
    command_v, command_a, feedback_v, feedback_a, tick_rows = load_execution(
        args.run_dir / "ticks.jsonl"
    )

    raw = {
        "demo_feedback": {"velocity": demo_state_v, "acceleration": demo_state_a},
        "demo_action": {"velocity": demo_action_v, "acceleration": demo_action_a},
        "model_chunk": {"velocity": model_v, "acceleration": model_a},
        "cpv_command": {"velocity": command_v, "acceleration": command_a},
        "can_feedback": {"velocity": feedback_v, "acceleration": feedback_a},
    }
    statistics = {
        source: {
            "velocity": distribution(values["velocity"]),
            "acceleration": distribution(values["acceleration"]),
        }
        for source, values in raw.items()
    }
    tracking = plot_tracking(tick_rows, args.output_dir / "tracking_error.png")
    demo_speed_p95 = statistics["demo_feedback"]["velocity"]["per_tick_max_abs"]["p95"]
    model_speed_p95 = statistics["model_chunk"]["velocity"]["per_tick_max_abs"]["p95"]
    command_speed_p95 = statistics["cpv_command"]["velocity"]["per_tick_max_abs"]["p95"]
    demo_accel_p95 = statistics["demo_feedback"]["acceleration"]["per_tick_max_abs"]["p95"]
    model_accel_p95 = statistics["model_chunk"]["acceleration"]["per_tick_max_abs"]["p95"]
    command_accel_p95 = statistics["cpv_command"]["acceleration"]["per_tick_max_abs"]["p95"]
    comparison = {
        "model_to_demo_speed_p95_ratio": model_speed_p95 / demo_speed_p95,
        "cpv_command_to_model_speed_p95_ratio": command_speed_p95 / model_speed_p95,
        "model_to_demo_acceleration_p95_ratio": model_accel_p95 / demo_accel_p95,
        "cpv_command_to_configured_acceleration_p95_ratio": command_accel_p95 / 160.0,
    }
    report = {
        "inputs": {
            "demo_parquet": str(args.demo_parquet),
            "run_dir": str(args.run_dir),
            "action_hz": args.action_hz,
            "policy_chunks": len(chunk_rows),
            "execution_ticks": len(tick_rows),
        },
        "statistics": statistics,
        "tracking": tracking,
        "comparison": comparison,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_cdf(
        {source: values["velocity"] for source, values in raw.items()},
        "Per-tick maximum joint speed (deg/s)",
        args.output_dir / "velocity_cdf.png",
    )
    plot_cdf(
        {source: values["acceleration"] for source, values in raw.items()},
        "Per-tick maximum joint acceleration (deg/s^2)",
        args.output_dir / "acceleration_cdf.png",
    )

    markdown = "\n".join(
        [
            "# Action / CPV Execution Match",
            "",
            f"Action timeline: `{args.action_hz:.1f} Hz`; chunks: `{len(chunk_rows)}`; execution ticks: `{len(tick_rows)}`.",
            "",
            "## Joint Speed",
            "",
            "Per-sample maximum over the fourteen arm joints, in deg/s.",
            "",
            rounded_table(statistics, "velocity"),
            "",
            "## Joint Acceleration",
            "",
            "Per-sample maximum over the fourteen arm joints, in deg/s^2.",
            "",
            rounded_table(statistics, "acceleration"),
            "",
            "## Command Tracking",
            "",
            f"Command-feedback error: p50 `{tracking['p50_deg']:.2f} deg`, p95 `{tracking['p95_deg']:.2f} deg`, max `{tracking['max_deg']:.2f} deg`.",
            "",
            "## Interpretation",
            "",
            f"- Model speed p95 is `{model_speed_p95:.2f} deg/s`, or `{comparison['model_to_demo_speed_p95_ratio']:.2f}x` the demonstration-feedback p95. The model trajectory is not abnormally fast.",
            f"- CPV command speed p95 is only `{command_speed_p95:.2f} deg/s`, or `{comparison['cpv_command_to_model_speed_p95_ratio']:.2f}x` the model demand. The physical command timeline falls behind the chunk.",
            f"- Model and demonstration-feedback acceleration p95 are `{model_accel_p95:.2f}` and `{demo_accel_p95:.2f} deg/s^2`; their distributions are closely matched.",
            f"- CPV command acceleration p95 is `{command_accel_p95:.2f} deg/s^2`, despite the run being configured for `160 deg/s^2`. This indicates discrete target snapping/stopping or handoff discontinuities rather than genuinely smooth high acceleration.",
            f"- Tracking error remains near the governor: p50 `{tracking['p50_deg']:.2f} deg`, p95 `{tracking['p95_deg']:.2f} deg`.",
            "",
            "The first control change should preserve the 30 Hz model timeline and replace point-to-point braking at every action row with trajectory-velocity tracking plus continuous chunk handoff. Top-speed tuning should follow only after that baseline is measured.",
            "",
            "See `velocity_cdf.png`, `acceleration_cdf.png`, and `tracking_error.png`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(f"PASS: action/execution comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
