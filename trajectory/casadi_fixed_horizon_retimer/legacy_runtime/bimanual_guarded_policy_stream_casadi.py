#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NERO_WS = Path("/home/dev/nero_ws")
TOPPRA_RUNTIME = ROOT.parent / "toppra_fixed_horizon_retimer" / "ab_runtime"
PRODUCTION_STREAM = NERO_WS / "scripts/bimanual_policy/bimanual_guarded_policy_stream.py"
for path in (ROOT / "vendor", ROOT, Path(__file__).resolve().parent, TOPPRA_RUNTIME, NERO_WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from casadi_rtc_queue import RecedingCasadiRtcQueue
from follower_state_bridge import (
    DESIRED_VELOCITY_REGISTRY,
    FOLLOWER_STATE_REGISTRY,
    make_bridged_follower_class,
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("nero_production_stream_casadi_ab", PRODUCTION_STREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load production stream: {PRODUCTION_STREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FOLLOWER_STATE_REGISTRY.reset()
    DESIRED_VELOCITY_REGISTRY.reset()
    module.BimanualRtcActionQueue = RecedingCasadiRtcQueue
    bridged_follower = make_bridged_follower_class(module.RateLimitedJointFollower)

    class CasadiJerkLimitedFollower(bridged_follower):
        def __init__(self, *args, **kwargs) -> None:
            kwargs.setdefault(
                "max_jerk_rad_s3",
                np.deg2rad(float(os.environ.get("NERO_CASADI_MAX_JERK_DEG_S3", "8000"))),
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

    module.RateLimitedJointFollower = CasadiJerkLimitedFollower
    print(
        "[CASADI AB] production source unchanged; "
        "RTC queue and velocity/jerk follower bridge replaced in this process only; "
        f"position_gain={os.environ.get('NERO_CASADI_STREAMING_POSITION_GAIN_S', '4.0')}s^-1"
    )
    module.main()


if __name__ == "__main__":
    main()
