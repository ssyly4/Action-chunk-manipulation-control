import math

from nero_vla.robot_config import CAPTURE_HOME_DEG
from nero_vla.robot_config import NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD


def test_nero_j4_override_keeps_both_limits_enabled() -> None:
    lower, upper = NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD["joint4"]

    assert math.degrees(lower) == -58.0
    assert math.degrees(upper) == 127.0


def test_capture_home_has_seven_joints() -> None:
    assert len(CAPTURE_HOME_DEG) == 7
