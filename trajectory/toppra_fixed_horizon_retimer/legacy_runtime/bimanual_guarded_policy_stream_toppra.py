#!/usr/bin/env python3
"""Load the production stream and replace only its RTC queue in this process."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


NERO_WS = Path("/home/dev/nero_ws")
PRODUCTION_STREAM = NERO_WS / "scripts/bimanual_policy/bimanual_guarded_policy_stream.py"
for path in (NERO_WS, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from receding_toppra_queue import RecedingToppraRtcQueue
from follower_state_bridge import FOLLOWER_STATE_REGISTRY, make_bridged_follower_class


def main() -> None:
    spec = importlib.util.spec_from_file_location("nero_production_stream_toppra_ab", PRODUCTION_STREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load production stream: {PRODUCTION_STREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FOLLOWER_STATE_REGISTRY.reset()
    module.BimanualRtcActionQueue = RecedingToppraRtcQueue
    module.RateLimitedJointFollower = make_bridged_follower_class(
        module.RateLimitedJointFollower
    )
    print(
        "[TOPPRA AB] production source unchanged; "
        "RTC queue and follower-state bridge replaced in this process only"
    )
    module.main()


if __name__ == "__main__":
    main()
