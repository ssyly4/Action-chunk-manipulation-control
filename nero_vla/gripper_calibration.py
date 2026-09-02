"""Interactive, read-only NERO gripper range calibration over Ethernet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from nero_vla.eth_state import NeroEthStateReader


def sample_stroke(reader: NeroEthStateReader, duration_sec: float, hz: float = 30) -> float:
    values: list[float] = []
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        stroke = reader.snapshot().gripper_stroke_mm
        if stroke is not None:
            values.append(stroke)
        time.sleep(1 / hz)
    if len(values) < 5:
        raise RuntimeError(f"Only {len(values)} valid gripper samples received")
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate NERO gripper stroke using ETH feedback")
    parser.add_argument("--host", default="10.90.0.150")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    args = parser.parse_args()
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")

    print("This tool is read-only. Move the gripper with the native leader/follower system.")
    with NeroEthStateReader(args.host) as reader:
        reader.wait_ready()
        input("Move the gripper to FULLY CLOSED, keep it still, then press Enter: ")
        closed_mm = sample_stroke(reader, args.sample_seconds)
        print(f"closed_mm={closed_mm:.4f}")
        input("Move the gripper to FULLY OPEN, keep it still, then press Enter: ")
        open_mm = sample_stroke(reader, args.sample_seconds)
        print(f"open_mm={open_mm:.4f}")

    span_mm = abs(open_mm - closed_mm)
    if span_mm < 5:
        raise RuntimeError(f"Calibration span is too small: {span_mm:.3f} mm")
    calibration = {
        "schema_version": 1,
        "host": args.host,
        "closed_mm": closed_mm,
        "open_mm": open_mm,
        "span_mm": span_mm,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
