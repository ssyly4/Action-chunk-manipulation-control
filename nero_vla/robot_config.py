"""NERO command limits that differ by controller mode."""

from __future__ import annotations

import math


CAPTURE_HOME_PROFILE = "capture_fixed_v1"
CAPTURE_HOME_DEG = (-0.075, -28.838, 4.080, 103.610, -0.030, 2.905, 53.238)
CAPTURE_HOME_GRIPPER_M = 0.09056


# MOVE_J firmware stops J4 at 123 degrees. CPV and leader/follower mode permit
# the demonstrated 125.62-degree range, so only CPV receives this override.
NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD = {
    "joint4": [math.radians(-58.0), math.radians(127.0)],
}
