#!/usr/bin/env python3
"""Run an 8D right-arm policy through the production RTC/OSQP/CasADi stack."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = ROOT.parent
CONTROL_ROOT = TRAJECTORY_ROOT.parent
CASADI_ROOT = TRAJECTORY_ROOT / "casadi_fixed_horizon_retimer"
TOPPRA_RUNTIME = TRAJECTORY_ROOT / "toppra_fixed_horizon_retimer" / "runtime"
ARM_SDK_ROOT = Path(os.environ.get("NERO_ARM_SDK_ROOT", "/home/dev/nero_ws/src/pyAgxArm"))
PRODUCTION_STREAM = CONTROL_ROOT / "scripts/bimanual_policy/bimanual_guarded_policy_stream.py"
for path in reversed(
    (
        ROOT / "vendor",
        ROOT,
        Path(__file__).resolve().parent,
        CASADI_ROOT / "vendor",
        CASADI_ROOT,
        CASADI_ROOT / "runtime",
        TOPPRA_RUNTIME.parent / "vendor",
        TOPPRA_RUNTIME.parent,
        TOPPRA_RUNTIME,
        CONTROL_ROOT,
        ARM_SDK_ROOT,
    )
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from follower_state_bridge import (  # noqa: E402
    DESIRED_VELOCITY_REGISTRY,
    FOLLOWER_STATE_REGISTRY,
    make_bridged_follower_class,
)
from osqp_casadi_rtc_queue import RecedingOsqpCasadiRtcQueue  # noqa: E402
from right_policy_adapter import RightOnlyPolicyClient  # noqa: E402


VIRTUAL_LEFT = "virtual_left"


def _foc_status() -> SimpleNamespace:
    names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "sensor_status",
        "driver_error_status",
        "stall_status",
    )
    return SimpleNamespace(**{name: False for name in names})


class VirtualGripper:
    """In-process left-gripper hold; it never owns a hardware interface."""

    def __init__(self, width_m: float = 0.09) -> None:
        self.width_m = float(width_m)

    def get_gripper_status(self):
        return SimpleNamespace(
            timestamp=time.monotonic(),
            msg=SimpleNamespace(
                mode="width", value=self.width_m, force=0.0, foc_status=_foc_status()
            ),
        )

    def move_gripper_m(self, *, value: float, force: float) -> None:
        del force
        self.width_m = float(value)


class VirtualArm:
    """Perfectly tracking virtual left arm used only by the 14-joint optimizer."""

    _nero_virtual_arm = True

    def __init__(self) -> None:
        self.joints = np.zeros(7, dtype=np.float64)

    def get_joint_angles(self):
        return SimpleNamespace(timestamp=time.monotonic_ns(), msg=self.joints.copy())

    def get_driver_states(self, joint_index: int):
        if not 1 <= joint_index <= 7:
            return None
        return SimpleNamespace(msg=SimpleNamespace(foc_status=_foc_status()))

    def get_arm_status(self):
        return SimpleNamespace(
            msg=SimpleNamespace(ctrl_mode=0x01, mode_feedback=0x05, arm_status=0x00)
        )

    def enable(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def set_auto_set_motion_mode_enabled(self, enabled: bool) -> None:
        del enabled

    def set_motion_mode(self, mode: str) -> None:
        del mode

    def move_cpv_pos(self, joint_index: int, position: float) -> None:
        self.joints[joint_index - 1] = float(position)

    def disconnect(self) -> None:
        pass


class VirtualCamera:
    """Placeholder for the unused left wrist image."""

    def __init__(self) -> None:
        self._image = np.zeros((224, 224, 3), dtype=np.uint8)

    def start(self) -> None:
        pass

    def wait_ready(self, timeout_sec: float) -> None:
        del timeout_sec

    def latest(self, *, max_age_sec: float):
        del max_age_sec
        return SimpleNamespace(image_rgb=self._image)

    def stop(self) -> None:
        pass


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "nero_production_stream_osqp_casadi_right", PRODUCTION_STREAM
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load production stream: {PRODUCTION_STREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    real_camera = module.V4L2CameraReader
    real_client = module.OpenPiPolicyClient
    real_create_robot = module.create_robot
    real_require_can_role = module.require_can_role

    def camera_factory(*args: Any, **kwargs: Any):
        if kwargs.get("name") == "left_wrist":
            return VirtualCamera()
        return real_camera(*args, **kwargs)

    def client_factory(*args: Any, **kwargs: Any):
        return RightOnlyPolicyClient(real_client(*args, **kwargs))

    def create_robot(interface: str):
        if interface != VIRTUAL_LEFT:
            return real_create_robot(interface)
        config = module.create_agx_arm_config(
            robot=module.ArmModel.NERO,
            firmeware_version=module.NeroFW.V120,
            interface="socketcan",
            channel="can_right",
            joint_limits=module.NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD,
        )
        return VirtualArm(), VirtualGripper(), config

    def require_can_role(interface: str, role: str, **kwargs: Any) -> None:
        if interface != VIRTUAL_LEFT:
            real_require_can_role(interface, role, **kwargs)

    FOLLOWER_STATE_REGISTRY.reset()
    DESIRED_VELOCITY_REGISTRY.reset()
    module.V4L2CameraReader = camera_factory
    module.OpenPiPolicyClient = client_factory
    module.create_robot = create_robot
    module.require_can_role = require_can_role
    module.BimanualRtcActionQueue = RecedingOsqpCasadiRtcQueue

    bridged_follower = make_bridged_follower_class(module.RateLimitedJointFollower)

    class OsqpCasadiJerkLimitedFollower(bridged_follower):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
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
            velocity = DESIRED_VELOCITY_REGISTRY.arm(self._toppra_bridge_slot)
            if velocity is not None and sample.target is not None:
                sample = module.TrajectorySample(
                    sample.target,
                    sample.status,
                    sample.source_age_sec,
                    sample.remaining_sec,
                    velocity,
                )
            return super().step(sample, measured=measured, now=now)

    module.RateLimitedJointFollower = OsqpCasadiJerkLimitedFollower
    print(
        "[RIGHT-ONLY] 8D policy -> virtual-left 16D -> RTC -> OSQP -> "
        "CasADi -> jerk-limited follower -> right CPV"
    )
    module.main()


if __name__ == "__main__":
    main()
