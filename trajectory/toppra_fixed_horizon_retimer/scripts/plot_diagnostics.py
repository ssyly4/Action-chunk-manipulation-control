#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-toppra-retimer")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one retime_chunk.py diagnostic JSON")
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    output = args.output or args.result.with_suffix(".png")
    raw_phase = np.asarray(data["raw_phase"])
    phase = np.asarray(data["retimed_phase"])
    command = np.asarray(data["commands"])
    velocity = np.asarray(data["velocity"])
    acceleration = np.asarray(data["acceleration"])
    jerk = np.asarray(data["jerk"])
    hz = 30.0

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    tick = np.arange(len(phase))
    axes[0, 0].plot(tick, raw_phase, label="raw phase", linewidth=2)
    axes[0, 0].plot(tick, phase, label="retimed phase", linewidth=2)
    axes[0, 0].set(title="Action phase at fixed 30 Hz ticks", xlabel="tick", ylabel="phase")
    axes[0, 0].legend()
    axes[0, 1].plot(tick, phase - raw_phase)
    axes[0, 1].set(title="Local phase redistribution", xlabel="tick", ylabel="phase delta")
    axes[1, 0].plot(tick / hz, command)
    axes[1, 0].set(title="Retimed joint commands", xlabel="time (s)", ylabel="position (rad)")
    axes[1, 1].plot(np.arange(len(velocity)) / hz, velocity)
    axes[1, 1].set(title="Discrete velocity", xlabel="time (s)", ylabel="rad/s")
    axes[2, 0].plot(np.arange(len(acceleration)) / hz, acceleration)
    axes[2, 0].set(title="Discrete acceleration", xlabel="time (s)", ylabel="rad/s2")
    axes[2, 1].plot(np.arange(len(jerk)) / hz, jerk)
    axes[2, 1].set(title="Measured jerk (not constrained by TOPPRAsd)", xlabel="time (s)", ylabel="rad/s3")
    fig.suptitle(f"Fixed-path TOPPRAsd: {data['status']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(output)


if __name__ == "__main__":
    main()

