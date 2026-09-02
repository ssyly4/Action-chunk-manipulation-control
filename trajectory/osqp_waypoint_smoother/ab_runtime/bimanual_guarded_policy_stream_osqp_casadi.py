#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = ROOT.parent
CONTROL_ROOT = TRAJECTORY_ROOT.parent
CASADI_ROOT = TRAJECTORY_ROOT / "casadi_fixed_horizon_retimer"
TOPPRA_RUNTIME = TRAJECTORY_ROOT / "toppra_fixed_horizon_retimer" / "ab_runtime"
ARM_SDK_ROOT = Path(os.environ.get("NERO_ARM_SDK_ROOT", "/home/dev/nero_ws/src/pyAgxArm"))
PRODUCTION_STREAM = CONTROL_ROOT / "scripts/bimanual_policy/bimanual_guarded_policy_stream.py"
for path in reversed(
    (
        ROOT / "vendor",
        ROOT,
        Path(__file__).resolve().parent,
        CASADI_ROOT / "vendor",
        CASADI_ROOT,
        CASADI_ROOT / "ab_runtime",
        TOPPRA_RUNTIME.parent / "vendor",
        TOPPRA_RUNTIME.parent,
        TOPPRA_RUNTIME,
        CONTROL_ROOT,
        ARM_SDK_ROOT,
    )
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from follower_state_bridge import (
    DESIRED_VELOCITY_REGISTRY,
    FOLLOWER_STATE_REGISTRY,
    make_bridged_follower_class,
)
from osqp_casadi_rtc_queue import RecedingOsqpCasadiRtcQueue


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "nero_production_stream_osqp_casadi_ab", PRODUCTION_STREAM
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load production stream: {PRODUCTION_STREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FOLLOWER_STATE_REGISTRY.reset()
    DESIRED_VELOCITY_REGISTRY.reset()
    module.BimanualRtcActionQueue = RecedingOsqpCasadiRtcQueue
    bridged_follower = make_bridged_follower_class(module.RateLimitedJointFollower)

    class OsqpCasadiJerkLimitedFollower(bridged_follower):
        def __init__(self, *args, **kwargs) -> None:
            kwargs.setdefault(
                "max_jerk_rad_s3",
                np.deg2rad(
                    float(os.environ.get("NERO_CASADI_MAX_JERK_DEG_S3", "8000"))
                ),
            )
            kwargs.setdefault(
                "streaming_position_gain_s",
                float(os.environ.get("NERO_CASADI_STREAMING_POSITION_GAIN_S", "4.0")),
            )
            super().__init__(*args, **kwargs)

        def step(self, sample, *, measured: np.ndarray, now: float):
            desired_velocity = DESIRED_VELOCITY_REGISTRY.arm(self._toppra_bridge_slot)
            if desired_velocity is not None and sample.target is not None:
                sample = module.TrajectorySample(
                    sample.target,
                    sample.status,
                    sample.source_age_sec,
                    sample.remaining_sec,
                    desired_velocity,
                )
            return super().step(sample, measured=measured, now=now)

    module.RateLimitedJointFollower = OsqpCasadiJerkLimitedFollower
    print(
        "[OSQP+CASADI AB] production source unchanged; "
        "queue and follower replaced in this process only"
    )
    module.main()


if __name__ == "__main__":
    main()
