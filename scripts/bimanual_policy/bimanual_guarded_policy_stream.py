#!/usr/bin/env python3
"""Run the three-camera pi0.5 towel policy on two guarded NERO CPV arms."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


CONTROL_ROOT = Path(__file__).resolve().parents[2]
PYAGXARM_SOURCE = Path(
    os.environ.get("NERO_ARM_SDK_ROOT", "/home/dev/nero_ws/src/pyAgxArm")
)
for source in (CONTROL_ROOT, PYAGXARM_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from nero_vla.bimanual_chunk_executor import BimanualFeedbackProgressActionChunk
from nero_vla.bimanual_chunk_executor import BimanualFeedbackRtcActionQueue
from nero_vla.bimanual_chunk_executor import BimanualFixedHorizonActionChunk
from nero_vla.bimanual_chunk_executor import BimanualRtcActionQueue
from nero_vla.bimanual_chunk_executor import align_bimanual_chunk_to_state
from nero_vla.bimanual_chunk_executor import arm_matrix
from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.cpv_backend import NeroCpvPositionBackend
from nero_vla.dual_can import require_bridge_not_forwarding, require_can_role
from nero_vla.gripper_controller import RateLimitedGripperFollower
from nero_vla.guarded_policy_stream import check_driver_health
from nero_vla.guarded_policy_stream import check_gripper_health
from nero_vla.guarded_policy_stream import wait_complete_joint_feedback
from nero_vla.guarded_policy_stream import wait_cpv_mode, wait_enabled
from nero_vla.image_tools import resize_with_pad
from nero_vla.lift_assist import PreGraspDescentAssist
from nero_vla.lift_assist import PostReleaseHeightGuard
from nero_vla.policy_client import OpenPiPolicyClient, port_open
from nero_vla.robot_config import NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD
from nero_vla.trajectory_executor import RateLimitedJointFollower, TrajectorySample


ACTION_HORIZON = 24
ACTION_DIM = 16
ACTION_HZ = 30.0
GRIPPER_OPEN_M = 0.09
CONFIRMATION = "RUN GUARDED BIMANUAL POLICY"


def create_robot(interface: str):
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=interface,
        joint_limits=NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD,
    )
    robot = AgxArmFactory.create_arm(config)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    robot.set_joint_limits_enabled(True)
    robot.connect()
    return robot, gripper, config


def joint_limits(config) -> np.ndarray:
    return np.asarray(
        [config["joint_limits"][f"joint{index}"] for index in range(1, 8)],
        dtype=np.float64,
    )


def wait_gripper(gripper, timeout_sec: float = 3.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = gripper.get_gripper_status()
        if status is not None and status.msg.mode == "width":
            return status
        time.sleep(0.01)
    raise RuntimeError("No valid width-mode gripper feedback")


def normalized_gripper(width_m: float) -> float:
    return float(np.clip(width_m / GRIPPER_OPEN_M, 0.0, 1.0))


def combined_joints(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate((left, right)).astype(np.float64)


def validate_chunk(
    packet: dict,
    *,
    feedback: np.ndarray,
    max_first_action_deg: float,
    max_consecutive_deg: float,
    max_gripper_consecutive: float | None = None,
    aligned_action: np.ndarray | None = None,
    aligned_offset_steps: float = 0.0,
    enforce_first_action: bool = True,
) -> dict[str, float]:
    actions = np.asarray(packet["actions"], dtype=np.float64)
    observed = np.asarray(packet["observation_joints"], dtype=np.float64)
    action_joints = np.concatenate((actions[:, :7], actions[:, 8:15]), axis=1)
    aligned = action_joints[0] if aligned_action is None else np.asarray(aligned_action)
    first_from_observation = np.rad2deg(action_joints[0] - observed)
    aligned_from_feedback = np.rad2deg(aligned - feedback)
    consecutive = np.rad2deg(np.diff(action_joints, axis=0))
    metrics = {
        "max_first_from_observation_deg": float(np.max(np.abs(first_from_observation))),
        "max_aligned_from_feedback_deg": float(np.max(np.abs(aligned_from_feedback))),
        "max_chunk_from_observation_deg": float(
            np.max(np.abs(np.rad2deg(action_joints - observed[None, :])))
        ),
        "max_consecutive_deg": float(np.max(np.abs(consecutive))),
        "aligned_offset_steps": float(aligned_offset_steps),
        "left_gripper_min": float(actions[:, 7].min()),
        "left_gripper_max": float(actions[:, 7].max()),
        "right_gripper_min": float(actions[:, 15].min()),
        "right_gripper_max": float(actions[:, 15].max()),
        "gripper_max_consecutive": float(
            max(np.abs(np.diff(actions[:, 7])).max(), np.abs(np.diff(actions[:, 15])).max())
        ),
    }
    if enforce_first_action and metrics["max_first_from_observation_deg"] > max_first_action_deg:
        raise RuntimeError(f"Policy first action rejected: {metrics}")
    if metrics["max_consecutive_deg"] > max_consecutive_deg:
        raise RuntimeError(f"Policy chunk has a discontinuous joint step: {metrics}")
    if min(metrics["left_gripper_min"], metrics["right_gripper_min"]) < -0.1 or max(
        metrics["left_gripper_max"], metrics["right_gripper_max"]
    ) > 1.1:
        raise RuntimeError(f"Policy chunk has an invalid gripper target: {metrics}")
    metrics["gripper_max_consecutive_limit"] = (
        None if max_gripper_consecutive is None else float(max_gripper_consecutive)
    )
    if (
        max_gripper_consecutive is not None
        and metrics["gripper_max_consecutive"] > max_gripper_consecutive
    ):
        raise RuntimeError(f"Policy chunk has a discontinuous gripper step: {metrics}")
    return metrics


def arm_status_ok(robot) -> bool:
    status = robot.get_arm_status()
    return bool(
        status is not None
        and int(status.msg.ctrl_mode) == 0x01
        and int(status.msg.mode_feedback) == 0x05
        and int(status.msg.arm_status) == 0x00
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="172.24.1.154")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--left-can", default="can_left")
    parser.add_argument("--right-can", default="can_right")
    parser.add_argument("--world-camera", required=True)
    parser.add_argument("--left-wrist-camera", required=True)
    parser.add_argument("--right-wrist-camera", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--noise-seed", type=int, default=3)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=ACTION_HORIZON,
        help="Action rows produced by the loaded policy checkpoint.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("feedback_phase", "fixed_horizon", "rtc", "rtc_time"),
        default="feedback_phase",
        help=(
            "Action chunk consumer. rtc advances phase from feedback; rtc_time "
            "uses the same guided asynchronous replacement but advances at rtc-action-hz."
        ),
    )
    parser.add_argument(
        "--fixed-horizon-steps",
        type=int,
        default=8,
        help="Number of action rows consumed per fixed_horizon chunk.",
    )
    parser.add_argument("--rtc-execution-horizon", type=int, default=12)
    parser.add_argument("--rtc-queue-threshold", type=int, default=22)
    parser.add_argument("--rtc-action-hz", type=float, default=25.0)
    parser.add_argument("--rtc-handoff-decay-steps", type=int, default=0)
    parser.add_argument("--rtc-max-handoff-error-deg", type=float, default=2.5)
    parser.add_argument(
        "--post-release-rtc-max-right-handoff-error-deg",
        type=float,
        default=2.5,
        help="Right-arm RTC handoff limit only while the post-release Cartesian guard is overriding policy.",
    )
    parser.add_argument("--rtc-num-steps", type=int, default=3)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=1.0)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--max-velocity-deg-s", type=float, default=8.0)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=24.0)
    parser.add_argument("--max-first-action-deg", type=float, default=2.0)
    parser.add_argument("--max-consecutive-action-deg", type=float, default=2.5)
    parser.add_argument("--max-alignment-error-deg", type=float, default=2.0)
    parser.add_argument("--alignment-search-margin-steps", type=float, default=2.0)
    parser.add_argument("--progress-arm-lead-steps", type=float, default=1.0)
    parser.add_argument("--progress-gripper-lead-steps", type=float, default=0.0)
    parser.add_argument("--gripper-event-lookahead-steps", type=float, default=10.0)
    parser.add_argument("--gripper-event-activation-delta", type=float, default=0.03)
    parser.add_argument(
        "--max-gripper-consecutive",
        type=float,
        default=None,
        help="Optional normalized gripper event limit. Disabled by default; physical gripper speed and force remain separately limited.",
    )
    parser.add_argument("--gripper-catchup-hold-sec", type=float, default=0.75)
    parser.add_argument(
        "--fail-on-gripper-catchup-timeout",
        action="store_true",
        help="Stop safely instead of resuming arm motion when a closing gripper remains behind.",
    )
    parser.add_argument("--max-progress-steps-per-tick", type=float, default=1.0)
    parser.add_argument("--chunk-blend-ms", type=float, default=67.0)
    parser.add_argument("--max-command-error-deg", type=float, default=1.5)
    parser.add_argument("--feedback-governor-error-deg", type=float, default=0.75)
    parser.add_argument("--policy-gap-timeout-sec", type=float, default=5.0)
    parser.add_argument("--gripper-speed-mm-s", type=float, default=200.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument(
        "--right-pregrasp-descent-mm",
        type=float,
        default=0.0,
        help="One-shot right TCP descent after confirmed close intent; zero disables it.",
    )
    parser.add_argument("--pregrasp-close-threshold", type=float, default=0.5)
    parser.add_argument("--pregrasp-release-threshold", type=float, default=0.25)
    parser.add_argument("--pregrasp-confirmations", type=int, default=3)
    parser.add_argument("--pregrasp-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--exit-on-right-gripper-cycle",
        action="store_true",
        help="Exit successfully after the right gripper closes, reopens, and both arms settle.",
    )
    parser.add_argument("--stage-close-threshold", type=float, default=0.25)
    parser.add_argument("--stage-open-threshold", type=float, default=0.75)
    parser.add_argument("--stage-settle-sec", type=float, default=0.5)
    parser.add_argument("--right-post-release-height-guard", action="store_true")
    parser.add_argument("--post-release-floor-mm", type=float, default=195.0)
    parser.add_argument("--post-release-recovery-mm", type=float, default=210.0)
    parser.add_argument("--post-release-height-timeout-sec", type=float, default=6.0)
    parser.add_argument("--post-release-forward-extension-mm", type=float, default=0.0)
    parser.add_argument("--post-release-extension-ramp-sec", type=float, default=1.0)
    parser.add_argument(
        "--post-release-lock-height",
        action="store_true",
        help="After right-gripper release, hold the configured TCP height while retaining policy XY/orientation.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONTROL_ROOT / "artifacts/logs/bimanual_policy_stream",
    )
    args = parser.parse_args()

    if args.left_can == args.right_can:
        parser.error("left and right CAN interfaces must differ")
    if not 5.0 <= args.duration <= 180.0:
        parser.error("duration must be in [5, 180]")
    if not 0.5 <= args.action_gain <= 1.5:
        parser.error("action-gain must be in [0.5, 1.5]")
    if not 0 < args.max_velocity_deg_s <= 32.0:
        parser.error("max-velocity-deg-s must be in (0, 32]")
    if not 0 < args.max_acceleration_deg_s2 <= 400.0:
        parser.error("max-acceleration-deg-s2 must be in (0, 400]")
    if not 0 < args.max_first_action_deg <= 3.0:
        parser.error("max-first-action-deg must be in (0, 3]")
    if not 0 < args.max_consecutive_action_deg <= 3.0:
        parser.error("max-consecutive-action-deg must be in (0, 3]")
    if not 0 < args.max_alignment_error_deg <= 3.0:
        parser.error("max-alignment-error-deg must be in (0, 3]")
    if not 0.25 <= args.progress_arm_lead_steps <= 2.0:
        parser.error("progress-arm-lead-steps must be in [0.25, 2]")
    if not 0 <= args.progress_gripper_lead_steps <= 4.0:
        parser.error("progress-gripper-lead-steps must be in [0, 4]")
    if not 0 <= args.gripper_event_lookahead_steps <= 15.0:
        parser.error("gripper-event-lookahead-steps must be in [0, 15]")
    if not 0.01 <= args.gripper_event_activation_delta <= 0.2:
        parser.error("gripper-event-activation-delta must be in [0.01, 0.2]")
    if args.max_gripper_consecutive is not None and not 0.01 <= args.max_gripper_consecutive <= 1.0:
        parser.error("max-gripper-consecutive must be in [0.01, 1]")
    if not 0 <= args.gripper_catchup_hold_sec <= 2.0:
        parser.error("gripper-catchup-hold-sec must be in [0, 2]")
    if not 0.25 <= args.max_progress_steps_per_tick <= 2.0:
        parser.error("max-progress-steps-per-tick must be in [0.25, 2]")
    if not 33 <= args.chunk_blend_ms <= 200:
        parser.error("chunk-blend-ms must be in [33, 200]")
    if not 0 < args.feedback_governor_error_deg <= args.max_command_error_deg:
        parser.error("feedback governor must not exceed the command error limit")
    if not 1 <= args.gripper_speed_mm_s <= 300:
        parser.error("gripper-speed-mm-s must be in [1, 300]")
    if not 0.2 <= args.gripper_force_n <= 2.0:
        parser.error("gripper-force-n must be in [0.2, 2]")
    if not 0.0 <= args.right_pregrasp_descent_mm <= 10.0:
        parser.error("right-pregrasp-descent-mm must be in [0, 10]")
    if not 0.1 <= args.pregrasp_close_threshold <= 0.8:
        parser.error("pregrasp-close-threshold must be in [0.1, 0.8]")
    if not 0.05 <= args.pregrasp_release_threshold < args.pregrasp_close_threshold:
        parser.error("pregrasp-release-threshold must be below the close threshold")
    if not 1 <= args.pregrasp_confirmations <= 10:
        parser.error("pregrasp-confirmations must be in [1, 10]")
    if not 0.5 <= args.pregrasp_timeout_sec <= 5.0:
        parser.error("pregrasp-timeout-sec must be in [0.5, 5]")
    if not 0.05 <= args.stage_close_threshold < args.stage_open_threshold <= 1.0:
        parser.error("stage gripper thresholds must satisfy 0.05 <= close < open <= 1")
    if not 0.1 <= args.stage_settle_sec <= 5.0:
        parser.error("stage-settle-sec must be in [0.1, 5]")
    if not 80.0 <= args.post_release_floor_mm <= 300.0:
        parser.error("post-release-floor-mm must be in [80, 300]")
    if not args.post_release_floor_mm + 5.0 <= args.post_release_recovery_mm <= 320.0:
        parser.error("post-release-recovery-mm must be at least 5 mm above the floor")
    if args.post_release_recovery_mm - args.post_release_floor_mm > 50.0:
        parser.error("post-release recovery band must not exceed 50 mm")
    if not 1.0 <= args.post_release_height_timeout_sec <= 10.0:
        parser.error("post-release-height-timeout-sec must be in [1, 10]")
    if not 0.0 <= args.post_release_forward_extension_mm <= 40.0:
        parser.error("post-release-forward-extension-mm must be in [0, 40]")
    if not 0.25 <= args.post_release_extension_ramp_sec <= 3.0:
        parser.error("post-release-extension-ramp-sec must be in [0.25, 3]")
    if not 2 <= args.action_horizon <= 64:
        parser.error("action-horizon must be in [2, 64]")
    if not 2 <= args.fixed_horizon_steps <= args.action_horizon:
        parser.error(f"fixed-horizon-steps must be in [2, {args.action_horizon}]")
    if args.chunk_mode in {"rtc", "rtc_time"}:
        if not 2 <= args.rtc_execution_horizon <= args.action_horizon:
            parser.error(f"rtc-execution-horizon must be in [2, {args.action_horizon}]")
        if not 1 <= args.rtc_queue_threshold < args.action_horizon:
            parser.error(f"rtc-queue-threshold must be in [1, {args.action_horizon - 1}]")
        if not 20.0 <= args.rtc_action_hz <= ACTION_HZ:
            parser.error(f"rtc-action-hz must be in [20, {ACTION_HZ:.0f}]")
        if not 0 <= args.rtc_handoff_decay_steps <= args.action_horizon:
            parser.error(
                f"rtc-handoff-decay-steps must be in [0, {args.action_horizon}]"
            )
        if not 0.25 <= args.rtc_max_handoff_error_deg <= 3.0:
            parser.error("rtc-max-handoff-error-deg must be in [0.25, 3]")
        if not (
            args.rtc_max_handoff_error_deg
            <= args.post_release_rtc_max_right_handoff_error_deg
            <= 6.0
        ):
            parser.error(
                "post-release-rtc-max-right-handoff-error-deg must be between "
                "rtc-max-handoff-error-deg and 6"
            )
        if not 1 <= args.rtc_num_steps <= 10:
            parser.error("rtc-num-steps must be in [1, 10]")
        if not 0.1 <= args.rtc_max_guidance_weight <= 10.0:
            parser.error("rtc-max-guidance-weight must be in [0.1, 10]")
    max_gripper_consecutive = (
        args.max_gripper_consecutive
        if args.max_gripper_consecutive is not None
        else None
    )

    require_can_role(args.left_can, "follower", recovery_timeout_sec=3.0)
    require_can_role(args.right_can, "follower", recovery_timeout_sec=3.0)
    require_bridge_not_forwarding()
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"Policy server is not reachable at {args.policy_host}:{args.policy_port}")

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    tick_stream = (run_dir / "ticks.jsonl").open("w", encoding="utf-8")
    chunk_stream = (run_dir / "chunks.jsonl").open("w", encoding="utf-8")
    fixed_noise = np.random.default_rng(args.noise_seed).standard_normal(
        (args.action_horizon, 32), dtype=np.float32
    )

    world = V4L2CameraReader(
        args.world_camera, width=args.width, height=args.height, fps=args.camera_fps, name="world"
    )
    left_wrist = V4L2CameraReader(
        args.left_wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="left_wrist",
    )
    right_wrist = V4L2CameraReader(
        args.right_wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="right_wrist",
    )
    cameras = (world, left_wrist, right_wrist)
    left_robot = right_robot = None
    left_backend = right_backend = None
    client: OpenPiPolicyClient | None = None
    summary = {
        "executed": False,
        "completed": False,
        "completion_reason": "preflight_only",
        "prompt": args.prompt,
        "policy_host": args.policy_host,
        "action_hz": ACTION_HZ,
        "chunk_mode": args.chunk_mode,
        "fixed_horizon_steps": args.fixed_horizon_steps
        if args.chunk_mode == "fixed_horizon"
        else None,
        "fixed_horizon_prefetch_max_sec": 0.5
        if args.chunk_mode == "fixed_horizon"
        else None,
        "rtc_execution_horizon": args.rtc_execution_horizon
        if args.chunk_mode in {"rtc", "rtc_time"}
        else None,
        "rtc_queue_threshold": (
            args.rtc_queue_threshold if args.chunk_mode in {"rtc", "rtc_time"} else None
        ),
        "rtc_action_hz": (
            args.rtc_action_hz if args.chunk_mode in {"rtc", "rtc_time"} else None
        ),
        "rtc_handoff_decay_steps": (
            args.rtc_handoff_decay_steps if args.chunk_mode == "rtc_time" else None
        ),
        "rtc_max_handoff_error_deg": (
            args.rtc_max_handoff_error_deg if args.chunk_mode == "rtc_time" else None
        ),
        "post_release_rtc_max_right_handoff_error_deg": (
            args.post_release_rtc_max_right_handoff_error_deg
            if args.chunk_mode == "rtc_time"
            else None
        ),
        "rtc_delay_prediction_clock": (
            "measured_feedback_progress"
            if args.chunk_mode == "rtc"
            else "emitted_action_count"
            if args.chunk_mode == "rtc_time"
            else None
        ),
        "rtc_num_steps": (
            args.rtc_num_steps if args.chunk_mode in {"rtc", "rtc_time"} else None
        ),
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight
        if args.chunk_mode in {"rtc", "rtc_time"}
        else None,
        "rtc_execution_clock": (
            "feedback_progress"
            if args.chunk_mode == "rtc"
            else "fixed_time"
            if args.chunk_mode == "rtc_time"
            else None
        ),
        "shared_bimanual_phase": True,
        "client_image_size": 224,
        "noise_seed": args.noise_seed,
        "max_velocity_deg_s": args.max_velocity_deg_s,
        "max_acceleration_deg_s2": args.max_acceleration_deg_s2,
        "joint_follower_mode": (
            "streaming_trajectory"
            if args.chunk_mode == "rtc_time"
            else "point_to_point"
        ),
        "gripper_speed_mm_s": args.gripper_speed_mm_s,
        "gripper_event_lookahead_steps": (
            0.0
            if args.chunk_mode == "fixed_horizon"
            else args.gripper_event_lookahead_steps
        ),
        "max_gripper_consecutive": max_gripper_consecutive,
        "gripper_catchup_hold_sec": args.gripper_catchup_hold_sec,
        "fail_on_gripper_catchup_timeout": args.fail_on_gripper_catchup_timeout,
        "right_pregrasp_descent_mm": args.right_pregrasp_descent_mm,
        "pregrasp_close_threshold": args.pregrasp_close_threshold,
        "pregrasp_release_threshold": args.pregrasp_release_threshold,
        "pregrasp_confirmations": args.pregrasp_confirmations,
        "pregrasp_triggered": False,
        "pregrasp_completed": False,
        "right_post_release_height_guard": args.right_post_release_height_guard,
        "post_release_floor_mm": args.post_release_floor_mm,
        "post_release_recovery_mm": args.post_release_recovery_mm,
        "post_release_forward_extension_mm": args.post_release_forward_extension_mm,
        "post_release_extension_ramp_sec": args.post_release_extension_ramp_sec,
        "post_release_height_triggered": False,
    }

    for camera in cameras:
        camera.start()
    try:
        for camera in cameras:
            camera.wait_ready(8.0)
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=10.0)
        left_robot, left_gripper, left_config = create_robot(args.left_can)
        right_robot, right_gripper, right_config = create_robot(args.right_can)
        left_initial = wait_complete_joint_feedback(left_robot)
        right_initial = wait_complete_joint_feedback(right_robot)
        left_gripper_initial = wait_gripper(left_gripper)
        right_gripper_initial = wait_gripper(right_gripper)
        for robot, gripper in (
            (left_robot, left_gripper),
            (right_robot, right_gripper),
        ):
            check_driver_health(robot)
            check_gripper_health(gripper)

        def capture_observation() -> tuple[dict, int, np.ndarray, tuple[float, float]]:
            left_feedback = left_robot.get_joint_angles()
            right_feedback = right_robot.get_joint_angles()
            left_grip = left_gripper.get_gripper_status()
            right_grip = right_gripper.get_gripper_status()
            if any(
                value is None
                for value in (left_feedback, right_feedback, left_grip, right_grip)
            ):
                raise RuntimeError("Incomplete dual-arm CAN feedback while assembling observation")
            if left_grip.msg.mode != "width" or right_grip.msg.mode != "width":
                raise RuntimeError("A gripper is not reporting width feedback")
            left_q = np.asarray(left_feedback.msg, dtype=np.float64)
            right_q = np.asarray(right_feedback.msg, dtype=np.float64)
            joints = combined_joints(left_q, right_q)
            grippers = (
                normalized_gripper(float(left_grip.msg.value)),
                normalized_gripper(float(right_grip.msg.value)),
            )
            state = np.asarray(
                [*left_q, grippers[0], *right_q, grippers[1]], dtype=np.float32
            )
            world_frame = world.latest(max_age_sec=0.2)
            left_frame = left_wrist.latest(max_age_sec=0.2)
            right_frame = right_wrist.latest(max_age_sec=0.2)
            observation_ns = time.monotonic_ns()
            observation = {
                "observation/world_image": resize_with_pad(world_frame.image_rgb),
                "observation/left_wrist_image": resize_with_pad(left_frame.image_rgb),
                "observation/right_wrist_image": resize_with_pad(right_frame.image_rgb),
                "observation/state": state,
                "prompt": args.prompt,
                "__openpi_noise": fixed_noise,
            }
            return observation, observation_ns, joints, grippers

        def infer_chunk(packet: tuple[dict, int, np.ndarray, tuple[float, float]]) -> dict:
            observation, observation_ns, observation_joints, grippers = packet
            if args.chunk_mode in {"rtc", "rtc_time"}:
                observation["__openpi_num_steps"] = args.rtc_num_steps
            started_ns = time.monotonic_ns()
            result, transport = client.infer_timed(observation)
            response_ns = time.monotonic_ns()
            actions = np.asarray(result.get("actions"), dtype=np.float64)
            if actions.shape != (args.action_horizon, ACTION_DIM) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid bimanual action chunk: {actions.shape}")
            actions = actions.copy()
            actions[:, :7] = observation_joints[None, :7] + args.action_gain * (
                actions[:, :7] - observation_joints[None, :7]
            )
            actions[:, 8:15] = observation_joints[None, 7:] + args.action_gain * (
                actions[:, 8:15] - observation_joints[None, 7:]
            )
            return {
                "actions": actions,
                "observation_ns": observation_ns,
                "observation_joints": observation_joints,
                "observation_grippers": grippers,
                "inference_ms": (response_ns - started_ns) / 1e6,
                "observation_to_response_ms": (response_ns - observation_ns) / 1e6,
                "transport": transport,
            }

        first_chunk = infer_chunk(capture_observation())
        initial_joints = combined_joints(left_initial, right_initial)
        first_metrics = validate_chunk(
            first_chunk,
            feedback=initial_joints,
            max_first_action_deg=args.max_first_action_deg,
            max_consecutive_deg=args.max_consecutive_action_deg,
            max_gripper_consecutive=max_gripper_consecutive,
        )
        print("BIMANUAL PI0.5 TOWEL POLICY PREFLIGHT PASSED", flush=True)
        print(
            f"policy={first_chunk['inference_ms']:.1f}ms "
            f"actions={args.action_horizon}x{ACTION_DIM} "
            f"speed={args.max_velocity_deg_s:.1f}deg/s "
            f"accel={args.max_acceleration_deg_s2:.1f}deg/s^2 "
            f"first_guard={first_metrics}",
            flush=True,
        )
        print(
            f"left_deg={np.round(np.rad2deg(left_initial), 2).tolist()} "
            f"right_deg={np.round(np.rad2deg(right_initial), 2).tolist()}",
            flush=True,
        )
        if args.preflight_only or not args.execute:
            print("PREFLIGHT ONLY: no enable, CPV, or gripper command was sent", flush=True)
            summary.update(completed=True, first_chunk=first_metrics)
            return
        if args.confirm != CONFIRMATION:
            raise RuntimeError(f"Exact confirmation required: {CONFIRMATION!r}")

        wait_enabled(left_robot)
        wait_enabled(right_robot)
        left_backend = NeroCpvPositionBackend(
            left_robot,
            max_command_step_rad=np.deg2rad(args.max_velocity_deg_s * 1.5 / ACTION_HZ),
        )
        right_backend = NeroCpvPositionBackend(
            right_robot,
            max_command_step_rad=np.deg2rad(args.max_velocity_deg_s * 1.5 / ACTION_HZ),
        )
        left_backend.prepare_hold(left_initial)
        right_backend.prepare_hold(right_initial)
        wait_cpv_mode(left_robot)
        wait_cpv_mode(right_robot)

        period = 1.0 / ACTION_HZ
        left_follower = RateLimitedJointFollower(
            max_velocity_rad_s=np.deg2rad(args.max_velocity_deg_s),
            max_acceleration_rad_s2=np.deg2rad(args.max_acceleration_deg_s2),
            max_target_feedback_error_rad=np.deg2rad(180.0),
            max_command_feedback_error_rad=np.deg2rad(args.max_command_error_deg),
            feedback_governor_error_rad=np.deg2rad(args.feedback_governor_error_deg),
            max_tick_interval_sec=3.0 * period,
            joint_limits_rad=joint_limits(left_config),
            tracking_mode="streaming_trajectory"
            if args.chunk_mode == "rtc_time"
            else "point_to_point",
        )
        right_follower = RateLimitedJointFollower(
            max_velocity_rad_s=np.deg2rad(args.max_velocity_deg_s),
            max_acceleration_rad_s2=np.deg2rad(args.max_acceleration_deg_s2),
            max_target_feedback_error_rad=np.deg2rad(180.0),
            max_command_feedback_error_rad=np.deg2rad(args.max_command_error_deg),
            feedback_governor_error_rad=np.deg2rad(args.feedback_governor_error_deg),
            max_tick_interval_sec=3.0 * period,
            joint_limits_rad=joint_limits(right_config),
            tracking_mode="streaming_trajectory"
            if args.chunk_mode == "rtc_time"
            else "point_to_point",
        )
        if args.chunk_mode == "fixed_horizon":
            progress = BimanualFixedHorizonActionChunk(
                action_hz=ACTION_HZ,
                execution_horizon_steps=args.fixed_horizon_steps,
                blend_duration_sec=args.chunk_blend_ms / 1000.0,
                # Server inference is commonly 280-330 ms while an 8-step
                # horizon lasts only 267 ms. Hold the final safe target long
                # enough for a late replacement rather than orphaning it.
                stale_after_sec=0.75,
            )
        elif args.chunk_mode == "rtc":
            progress = BimanualFeedbackRtcActionQueue(
                action_hz=ACTION_HZ,
                arm_lead_steps=args.progress_arm_lead_steps,
                gripper_lead_steps=args.progress_gripper_lead_steps,
                gripper_event_lookahead_steps=args.gripper_event_lookahead_steps,
                gripper_event_activation_delta=args.gripper_event_activation_delta,
                max_progress_steps_per_tick=args.max_progress_steps_per_tick,
                blend_duration_sec=args.chunk_blend_ms / 1000.0,
                stale_after_sec=args.policy_gap_timeout_sec,
            )
        elif args.chunk_mode == "rtc_time":
            progress = BimanualRtcActionQueue(
                action_hz=args.rtc_action_hz,
                handoff_decay_steps=args.rtc_handoff_decay_steps,
                # The stream applies the final handoff rule below because the
                # post-release Cartesian guard has an intentionally different
                # limit for the guarded right arm.  Keeping a second global
                # queue-level limit would reject that explicitly safe case.
                max_handoff_error_rad=None,
            )
        else:
            progress = BimanualFeedbackProgressActionChunk(
                action_hz=ACTION_HZ,
                arm_lead_steps=args.progress_arm_lead_steps,
                gripper_lead_steps=args.progress_gripper_lead_steps,
                gripper_event_lookahead_steps=args.gripper_event_lookahead_steps,
                gripper_event_activation_delta=args.gripper_event_activation_delta,
                max_progress_steps_per_tick=args.max_progress_steps_per_tick,
                blend_duration_sec=args.chunk_blend_ms / 1000.0,
                stale_after_sec=0.15,
            )
        started = time.monotonic()
        left_follower.initialize(left_initial, now=started - period)
        right_follower.initialize(right_initial, now=started - period)
        left_gripper_follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=GRIPPER_OPEN_M,
            max_speed_m_s=args.gripper_speed_mm_s / 1000.0,
            contact_force_n=0.5,
            max_tick_interval_sec=3.0 * period,
            force_hold_enabled=False,
            feedback_tolerance_m=0.01,
        )
        right_gripper_follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=GRIPPER_OPEN_M,
            max_speed_m_s=args.gripper_speed_mm_s / 1000.0,
            contact_force_n=0.5,
            max_tick_interval_sec=3.0 * period,
            force_hold_enabled=False,
            feedback_tolerance_m=0.01,
        )
        left_gripper_follower.initialize(
            float(left_gripper_initial.msg.value), now=started - period
        )
        right_gripper_follower.initialize(
            float(right_gripper_initial.msg.value), now=started - period
        )
        pregrasp_assist = PreGraspDescentAssist(
            enabled=args.right_pregrasp_descent_mm > 0.0,
            descent_distance_m=(
                args.right_pregrasp_descent_mm / 1000.0
                if args.right_pregrasp_descent_mm > 0.0
                else 0.005
            ),
            close_threshold=args.pregrasp_close_threshold,
            confirmations=args.pregrasp_confirmations,
            release_gripper_threshold=args.pregrasp_release_threshold,
            timeout_sec=args.pregrasp_timeout_sec,
        )
        post_release_height_guard = PostReleaseHeightGuard(
            enabled=args.right_post_release_height_guard,
            floor_height_m=args.post_release_floor_mm / 1000.0,
            recovery_height_m=args.post_release_recovery_mm / 1000.0,
            close_threshold=args.stage_close_threshold,
            open_threshold=args.stage_open_threshold,
            confirmations=3,
            timeout_sec=args.post_release_height_timeout_sec,
            joint_limits_rad=joint_limits(right_config),
            forward_extension_m=args.post_release_forward_extension_mm / 1000.0,
            extension_ramp_sec=args.post_release_extension_ramp_sec,
            lock_height=args.post_release_lock_height,
        )
        if args.chunk_mode == "rtc":
            progress.push(
                first_chunk["actions"],
                initial_phase_steps=0.0,
                observed_at=first_chunk["observation_ns"] / 1e9,
                received_at=started,
            )
        elif args.chunk_mode == "rtc_time":
            progress.load(first_chunk["actions"])
        else:
            progress.push(
                first_chunk["actions"],
                initial_phase_steps=0.0,
                observed_at=started,
                received_at=started,
            )
        chunk_stream.write(json.dumps({"chunk": 0, **first_chunk, **first_metrics}, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
        chunk_stream.flush()

        last_left_feedback = left_initial.copy()
        last_right_feedback = right_initial.copy()
        last_left_command = left_initial.copy()
        last_right_command = right_initial.copy()
        last_left_gripper_command = float(left_gripper_initial.msg.value)
        last_right_gripper_command = float(right_gripper_initial.msg.value)
        last_left_gripper_feedback = float(left_gripper_initial.msg.value)
        last_right_gripper_feedback = float(right_gripper_initial.msg.value)
        gripper_catchup_started: float | None = None
        last_gripper_send = started - 1.0
        next_tick = started
        next_health = started
        chunk_count = 1
        policy_gap_started: float | None = None
        status_counts: Counter[str] = Counter()
        completion_reason = "duration_limit"
        last_feedback_time = started
        right_gripper_cycle_closed = False
        right_gripper_cycle_opened = False
        stage_settle_started: float | None = None
        summary["executed"] = True
        summary["completion_reason"] = completion_reason

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-infer") as infer_pool, ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-cpv") as send_pool:
            # Feedback-phase replans continuously. Fixed-horizon and RTC modes
            # schedule inference later so the replacement observation is fresh
            # when the current chunk reaches its handoff point.
            future: Future | None = (
                infer_pool.submit(infer_chunk, capture_observation())
                if args.chunk_mode == "feedback_phase"
                else None
            )
            pending_fixed_chunk: dict | None = None
            fixed_horizon_wait_started: float | None = None
            fixed_rejection_started: float | None = None
            fixed_rejected_chunks = 0
            fixed_latency_history = deque(
                [first_chunk["inference_ms"] / 1000.0], maxlen=8
            )
            rtc_request = None
            rtc_latency_history = deque(
                [first_chunk["inference_ms"] / 1000.0], maxlen=8
            )
            rtc_queue_hold_started: float | None = None
            rtc_rejection_started: float | None = None
            rtc_rejected_chunks = 0
            rtc_generation = 0
            rtc_request_generation: int | None = None
            post_release_was_overriding = False
            while time.monotonic() - started < args.duration:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                if now - started >= args.duration:
                    break
                scheduler_lag = now - next_tick
                next_tick += period
                if scheduler_lag > 3.0 * period:
                    raise RuntimeError(f"Control scheduler gap: {scheduler_lag * 1000:.1f}ms")

                combined_feedback = combined_joints(last_left_feedback, last_right_feedback)
                combined_command = combined_joints(last_left_command, last_right_command)
                if future is not None and future.done():
                    completed = future.result()
                    future = None
                    if args.chunk_mode == "rtc_time":
                        if rtc_request is None:
                            raise RuntimeError("RTC response has no matching request context")
                        completed_request = rtc_request
                        completed_request_generation = rtc_request_generation
                        accepted_at = now
                        nominal_elapsed_delay_steps = math.ceil(
                            (accepted_at - completed_request.requested_at)
                            * args.rtc_action_hz
                        )
                        consumed_delay_steps = progress.consumed_since(completed_request)
                        rtc_latency_history.append(completed["inference_ms"] / 1000.0)
                        rtc_request = None
                        rtc_request_generation = None
                        # A Cartesian safety assist may have re-anchored the
                        # queue after this request was sent.  Its result was
                        # conditioned on the old policy tail, so it must never
                        # be used to restart the live command timeline.
                        if completed_request_generation != rtc_generation:
                            chunk_stream.write(
                                json.dumps(
                                    {
                                        "chunk": chunk_count,
                                        "accepted": False,
                                        "discarded": "stale_after_external_reanchor",
                                        "rtc_alignment_mode": "elapsed_time",
                                        **completed,
                                    },
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            continue
                        # This mode deliberately does not search the action
                        # chunk against robot feedback. Rows elapsed while the
                        # request was in flight are skipped by the fixed action
                        # clock, then the replacement continues immediately.
                        skip_steps = min(
                            int(consumed_delay_steps), len(completed["actions"]) - 1
                        )
                        try:
                            metrics = validate_chunk(
                                completed,
                                feedback=combined_feedback,
                                max_first_action_deg=args.max_first_action_deg,
                                max_consecutive_deg=args.max_consecutive_action_deg,
                                max_gripper_consecutive=max_gripper_consecutive,
                                aligned_action=arm_matrix(completed["actions"])[skip_steps],
                                aligned_offset_steps=float(skip_steps),
                                enforce_first_action=False,
                            )
                            handoff_errors_rad = progress.handoff_errors_rad(
                                completed["actions"], skip_steps=skip_steps
                            )
                            left_handoff_error_rad, right_handoff_error_rad = (
                                float(handoff_errors_rad[0]),
                                float(handoff_errors_rad[1]),
                            )
                            right_handoff_limit_deg = (
                                args.post_release_rtc_max_right_handoff_error_deg
                                if post_release_height_guard.overriding_policy
                                else args.rtc_max_handoff_error_deg
                            )
                            if (
                                left_handoff_error_rad
                                > np.deg2rad(args.rtc_max_handoff_error_deg)
                                or right_handoff_error_rad
                                > np.deg2rad(right_handoff_limit_deg)
                            ):
                                raise RuntimeError(
                                    "RTC handoff rejected: "
                                    f"left_error_deg={np.rad2deg(left_handoff_error_rad):.3f} "
                                    f"right_error_deg={np.rad2deg(right_handoff_error_rad):.3f} "
                                    f"right_limit_deg={right_handoff_limit_deg:.3f}"
                                )
                        except (RuntimeError, ValueError) as exc:
                            rtc_rejected_chunks += 1
                            if rtc_rejection_started is None:
                                rtc_rejection_started = now
                            chunk_stream.write(
                                json.dumps(
                                    {
                                        "chunk": chunk_count,
                                        "accepted": False,
                                        "rejection": f"{type(exc).__name__}: {exc}",
                                        "rtc_alignment_mode": "elapsed_time",
                                        "rtc_predicted_delay_steps": completed_request.predicted_delay_steps,
                                        "rtc_nominal_elapsed_delay_steps": nominal_elapsed_delay_steps,
                                        "rtc_consumed_delay_steps": consumed_delay_steps,
                                        **completed,
                                    },
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            if now - rtc_rejection_started > args.policy_gap_timeout_sec:
                                raise RuntimeError(
                                    "RTC time-aligned replacements remained unsafe"
                                ) from exc
                        else:
                            rtc_rejection_started = None
                            metrics.update(
                                accepted=True,
                                rtc_alignment_mode="elapsed_time",
                                rtc_predicted_delay_steps=completed_request.predicted_delay_steps,
                                rtc_nominal_elapsed_delay_steps=nominal_elapsed_delay_steps,
                                rtc_consumed_delay_steps=consumed_delay_steps,
                                rtc_valid_previous_steps=completed_request.valid_previous_steps,
                                rtc_remaining_actions=float(
                                    args.action_horizon - 1 - skip_steps
                                ),
                                rtc_left_handoff_error_deg=float(
                                    np.rad2deg(left_handoff_error_rad)
                                ),
                                rtc_right_handoff_error_deg=float(
                                    np.rad2deg(right_handoff_error_rad)
                                ),
                                rtc_right_handoff_limit_deg=float(right_handoff_limit_deg),
                            )
                            progress.load(completed["actions"], skip_steps=skip_steps)
                            chunk_stream.write(
                                json.dumps(
                                    {"chunk": chunk_count, **completed, **metrics},
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            chunk_count += 1
                    elif args.chunk_mode == "rtc":
                        if rtc_request is None:
                            raise RuntimeError("RTC response has no matching request context")
                        completed_request = rtc_request
                        accepted_at = now
                        nominal_elapsed_delay_steps = math.ceil(
                            (accepted_at - completed_request.requested_at)
                            * args.rtc_action_hz
                        )
                        consumed_delay_steps = progress.consumed_since(completed_request)
                        rtc_latency_history.append(completed["inference_ms"] / 1000.0)
                        rtc_request = None
                        try:
                            alignment = align_bimanual_chunk_to_state(
                                completed["actions"],
                                feedback=combined_feedback,
                                command=combined_command,
                                latency_steps=float(consumed_delay_steps),
                                max_alignment_error_rad=np.deg2rad(
                                    args.max_alignment_error_deg
                                ),
                                search_margin_steps=args.alignment_search_margin_steps,
                            )
                            metrics = validate_chunk(
                                completed,
                                feedback=combined_feedback,
                                max_first_action_deg=args.max_first_action_deg,
                                max_consecutive_deg=args.max_consecutive_action_deg,
                                max_gripper_consecutive=max_gripper_consecutive,
                                aligned_action=alignment.action,
                                aligned_offset_steps=alignment.offset_steps,
                                # The feedback-aligned action is the only RTC row
                                # that may execute; row zero belongs to the older
                                # observation captured before asynchronous inference.
                                enforce_first_action=False,
                            )
                        except (RuntimeError, ValueError) as exc:
                            rtc_rejected_chunks += 1
                            if rtc_rejection_started is None:
                                rtc_rejection_started = now
                            chunk_stream.write(
                                json.dumps(
                                    {
                                        "chunk": chunk_count,
                                        "accepted": False,
                                        "rejection": f"{type(exc).__name__}: {exc}",
                                        "rtc_predicted_delay_steps": completed_request.predicted_delay_steps,
                                        "rtc_nominal_elapsed_delay_steps": nominal_elapsed_delay_steps,
                                        "rtc_consumed_delay_steps": consumed_delay_steps,
                                        **completed,
                                    },
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            if now - rtc_rejection_started > args.policy_gap_timeout_sec:
                                raise RuntimeError(
                                    "RTC feedback-aligned replacements remained unsafe"
                                ) from exc
                        else:
                            rtc_rejection_started = None
                            metrics.update(
                                accepted=True,
                                rtc_alignment_mode="feedback_progress",
                                rtc_predicted_delay_steps=completed_request.predicted_delay_steps,
                                rtc_nominal_elapsed_delay_steps=nominal_elapsed_delay_steps,
                                rtc_consumed_delay_steps=consumed_delay_steps,
                                rtc_valid_previous_steps=completed_request.valid_previous_steps,
                                rtc_remaining_actions=float(
                                    args.action_horizon - 1 - alignment.offset_steps
                                ),
                                alignment_search_max_steps=alignment.search_max_steps,
                                alignment_rms_feedback_error_deg=float(
                                    np.rad2deg(alignment.rms_feedback_error_rad)
                                ),
                                alignment_max_command_error_deg=float(
                                    np.rad2deg(alignment.max_command_error_rad)
                                ),
                            )
                            progress.push(
                                completed["actions"],
                                initial_phase_steps=alignment.offset_steps,
                                observed_at=completed["observation_ns"] / 1e9,
                                received_at=accepted_at,
                            )
                            chunk_stream.write(
                                json.dumps(
                                    {"chunk": chunk_count, **completed, **metrics},
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            chunk_count += 1
                    elif args.chunk_mode == "fixed_horizon":
                        fixed_latency_history.append(completed["inference_ms"] / 1000.0)
                        pending_fixed_chunk = completed
                    else:
                        accepted_at = now
                        latency_steps = (
                            accepted_at - completed["observation_ns"] / 1e9
                        ) * ACTION_HZ
                        alignment = align_bimanual_chunk_to_state(
                            completed["actions"],
                            feedback=combined_feedback,
                            command=combined_command,
                            latency_steps=latency_steps,
                            max_alignment_error_rad=np.deg2rad(args.max_alignment_error_deg),
                            search_margin_steps=args.alignment_search_margin_steps,
                        )
                        metrics = validate_chunk(
                            completed,
                            feedback=combined_feedback,
                            max_first_action_deg=args.max_first_action_deg,
                            max_consecutive_deg=args.max_consecutive_action_deg,
                            max_gripper_consecutive=max_gripper_consecutive,
                            aligned_action=alignment.action,
                            aligned_offset_steps=alignment.offset_steps,
                        )
                        metrics.update(
                            alignment_latency_steps=alignment.latency_steps,
                            alignment_search_max_steps=alignment.search_max_steps,
                            alignment_rms_feedback_error_deg=float(
                                np.rad2deg(alignment.rms_feedback_error_rad)
                            ),
                        )
                        progress.push(
                            completed["actions"],
                            initial_phase_steps=alignment.offset_steps,
                            observed_at=completed["observation_ns"] / 1e9,
                            received_at=accepted_at,
                        )
                        chunk_stream.write(json.dumps({"chunk": chunk_count, **completed, **metrics}, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
                        chunk_stream.flush()
                        chunk_count += 1
                        future = infer_pool.submit(infer_chunk, capture_observation())

                target = progress.sample(now, feedback=combined_feedback)
                if (
                    args.chunk_mode == "fixed_horizon"
                    and target.status
                    in {"fixed_horizon_complete_hold", "stale_chunk_hold"}
                ):
                    if pending_fixed_chunk is None:
                        if fixed_horizon_wait_started is None:
                            fixed_horizon_wait_started = now
                        elif now - fixed_horizon_wait_started > args.policy_gap_timeout_sec:
                            raise RuntimeError("Fixed-horizon chunk replacement did not arrive")
                    else:
                        completed = pending_fixed_chunk
                        pending_fixed_chunk = None
                        fixed_horizon_wait_started = None
                        latency_steps = (
                            now - completed["observation_ns"] / 1e9
                        ) * ACTION_HZ
                        try:
                            alignment = align_bimanual_chunk_to_state(
                                completed["actions"],
                                feedback=combined_feedback,
                                command=combined_command,
                                latency_steps=latency_steps,
                                max_alignment_error_rad=np.deg2rad(
                                    args.max_alignment_error_deg
                                ),
                                search_margin_steps=args.alignment_search_margin_steps,
                            )
                            start_index = min(
                                args.fixed_horizon_steps - 1,
                                int(math.ceil(alignment.offset_steps - 1e-9)),
                            )
                            aligned_action = arm_matrix(completed["actions"])[start_index]
                            metrics = validate_chunk(
                                completed,
                                feedback=combined_feedback,
                                max_first_action_deg=args.max_first_action_deg,
                                max_consecutive_deg=args.max_consecutive_action_deg,
                                max_gripper_consecutive=max_gripper_consecutive,
                                aligned_action=aligned_action,
                                aligned_offset_steps=float(start_index),
                                enforce_first_action=False,
                            )
                            if (
                                metrics["max_aligned_from_feedback_deg"]
                                > args.max_alignment_error_deg
                            ):
                                raise RuntimeError(
                                    f"Fixed-horizon aligned action rejected: {metrics}"
                                )
                        except (RuntimeError, ValueError) as exc:
                            fixed_rejected_chunks += 1
                            if fixed_rejection_started is None:
                                fixed_rejection_started = now
                            chunk_stream.write(
                                json.dumps(
                                    {
                                        "chunk": chunk_count,
                                        "accepted": False,
                                        "rejection": f"{type(exc).__name__}: {exc}",
                                        "fixed_horizon_retry_from_live_hold": True,
                                        **completed,
                                    },
                                    default=lambda value: value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value,
                                )
                                + "\n"
                            )
                            chunk_stream.flush()
                            if now - fixed_rejection_started > args.policy_gap_timeout_sec:
                                raise RuntimeError(
                                    "Fixed-horizon replacements remained unsafe"
                                ) from exc
                        else:
                            fixed_rejection_started = None
                            metrics.update(
                                accepted=True,
                                fixed_horizon_steps=args.fixed_horizon_steps,
                                fixed_horizon_alignment_offset_steps=alignment.offset_steps,
                                fixed_horizon_start_index=start_index,
                                alignment_latency_steps=alignment.latency_steps,
                                alignment_rms_feedback_error_deg=float(
                                    np.rad2deg(alignment.rms_feedback_error_rad)
                                ),
                                strict_shared_action_index=True,
                            )
                            progress.push(
                                completed["actions"],
                                initial_phase_steps=float(start_index),
                                observed_at=completed["observation_ns"] / 1e9,
                                received_at=now,
                            )
                            chunk_stream.write(json.dumps({"chunk": chunk_count, **completed, **metrics}, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
                            chunk_stream.flush()
                            chunk_count += 1
                            target = progress.sample(now, feedback=combined_feedback)
                if (
                    args.chunk_mode == "fixed_horizon"
                    and future is None
                    and pending_fixed_chunk is None
                    and target.remaining_sec is not None
                ):
                    # Start early enough for a continuous handoff. The shared
                    # feedback alignment below skips action rows consumed while
                    # inference was in flight, so early capture does not replay
                    # stale row-zero targets.
                    prefetch_sec = min(0.5, max(fixed_latency_history) + 0.05)
                    if target.remaining_sec <= prefetch_sec:
                        future = infer_pool.submit(infer_chunk, capture_observation())
                if (
                    args.chunk_mode in {"rtc", "rtc_time"}
                    and future is None
                    and progress.remaining_steps <= args.rtc_queue_threshold
                ):
                    estimated_progress_hz = (
                        args.rtc_action_hz
                        if args.chunk_mode == "rtc_time"
                        else max(1.0, min(ACTION_HZ, progress.progress_rate_hz))
                    )
                    predicted_delay_steps = min(
                        args.action_horizon - 1,
                        max(
                            1,
                            math.ceil(
                                max(rtc_latency_history) * estimated_progress_hz
                            ),
                        ),
                    )
                    rtc_request = progress.make_request(
                        now=now,
                        execution_horizon=args.rtc_execution_horizon,
                        predicted_delay_steps=predicted_delay_steps,
                    )
                    rtc_request_generation = rtc_generation
                    packet = capture_observation()
                    packet[0]["__openpi_rtc"] = {
                        "prev_chunk_left_over": rtc_request.previous_actions,
                        "inference_delay": rtc_request.predicted_delay_steps,
                        "execution_horizon": rtc_request.valid_previous_steps,
                        "max_guidance_weight": args.rtc_max_guidance_weight,
                    }
                    future = infer_pool.submit(infer_chunk, packet)
                rtc_hold_statuses = (
                    {"rtc_queue_hold", "no_chunk_hold"}
                    if args.chunk_mode == "rtc_time"
                    else {
                        "rtc_feedback_tail_hold",
                        "rtc_feedback_stale_hold",
                        "rtc_feedback_no_chunk_hold",
                    }
                )
                if args.chunk_mode in {"rtc", "rtc_time"} and target.status in rtc_hold_statuses:
                    if rtc_queue_hold_started is None:
                        rtc_queue_hold_started = now
                    elif now - rtc_queue_hold_started > args.policy_gap_timeout_sec:
                        raise RuntimeError("RTC action queue did not recover")
                else:
                    rtc_queue_hold_started = None
                if target.left_target is None or target.right_target is None:
                    if policy_gap_started is None:
                        policy_gap_started = now
                    elif now - policy_gap_started > args.policy_gap_timeout_sec:
                        raise RuntimeError("Policy action timeline did not recover")
                else:
                    policy_gap_started = None
                raw_right_grip_target = float(
                    target.right_gripper_target
                    if target.right_gripper_target is not None
                    else normalized_gripper(last_right_gripper_command)
                )
                if post_release_height_guard.observe(
                    now=now,
                    joint_rad=last_right_feedback,
                    measured_gripper_state=normalized_gripper(
                        last_right_gripper_feedback
                    ),
                ):
                    summary["post_release_height_triggered"] = True
                    summary["post_release_height_triggered_sec"] = now - started
                triggered = pregrasp_assist.observe(
                    now=now,
                    joint_rad=last_right_feedback,
                    measured_gripper_state=normalized_gripper(
                        last_right_gripper_feedback
                    ),
                    policy_gripper_target=raw_right_grip_target,
                )
                if triggered:
                    summary["pregrasp_triggered"] = True
                    summary["pregrasp_triggered_sec"] = now - started
                assisted_right_target = pregrasp_assist.arm_target(
                    now=now,
                    measured_joint_rad=last_right_feedback,
                )
                policy_right_target = (
                    last_right_command
                    if target.right_target is None
                    else target.right_target
                )
                height_guard_target = post_release_height_guard.arm_target(
                    now=now,
                    measured_joint_rad=last_right_feedback,
                    policy_joint_rad=policy_right_target,
                )
                if pregrasp_assist.state == "completed":
                    summary["pregrasp_completed"] = True
                right_close_gap = (
                    normalized_gripper(last_right_gripper_feedback)
                    - raw_right_grip_target
                )
                closing_needs_catchup = right_close_gap >= 0.08
                if closing_needs_catchup and gripper_catchup_started is None:
                    gripper_catchup_started = now
                elif not closing_needs_catchup:
                    gripper_catchup_started = None
                gripper_catchup_hold = bool(
                    not pregrasp_assist.active
                    and gripper_catchup_started is not None
                    and now - gripper_catchup_started < args.gripper_catchup_hold_sec
                )
                if (
                    args.fail_on_gripper_catchup_timeout
                    and not pregrasp_assist.active
                    and gripper_catchup_started is not None
                    and now - gripper_catchup_started >= args.gripper_catchup_hold_sec
                ):
                    raise RuntimeError(
                        "Gripper did not catch up before timeout; refusing further arm motion"
                    )
                stage_release_hold = bool(
                    args.exit_on_right_gripper_cycle
                    and right_gripper_cycle_opened
                )
                if stage_release_hold:
                    # The measured reopen event marks the learned stage boundary.
                    # Freeze the placement pose before checking the short settle.
                    left_arm_target = last_left_command
                    right_arm_target = last_right_command
                    left_arm_status = "stage_release_hold"
                    right_arm_status = "stage_release_hold"
                elif pregrasp_assist.active:
                    left_arm_target = last_left_command
                    right_arm_target = assisted_right_target
                    left_arm_status = f"right_pregrasp_{pregrasp_assist.state}"
                    right_arm_status = left_arm_status
                elif height_guard_target is not None:
                    left_arm_target = (
                        last_left_command
                        if gripper_catchup_hold or target.left_target is None
                        else target.left_target
                    )
                    right_arm_target = height_guard_target
                    # The left arm remains on the unmodified policy stream. The
                    # right guard is also a continuously updated trajectory, so
                    # retain the streaming follower mode for both sides.
                    left_arm_status = target.status
                    right_arm_status = target.status
                else:
                    left_arm_target = (
                        last_left_command if gripper_catchup_hold else target.left_target
                    )
                    right_arm_target = (
                        last_right_command if gripper_catchup_hold else target.right_target
                    )
                    left_arm_status = (
                        "gripper_catchup_hold" if gripper_catchup_hold else target.status
                    )
                    right_arm_status = left_arm_status
                left_limited = left_follower.step(
                    TrajectorySample(
                        left_arm_target,
                        left_arm_status,
                        target.source_age_sec,
                        target.remaining_sec,
                    ),
                    measured=last_left_feedback,
                    now=now,
                )
                right_limited = right_follower.step(
                    TrajectorySample(
                        right_arm_target,
                        right_arm_status,
                        target.source_age_sec,
                        target.remaining_sec,
                    ),
                    measured=last_right_feedback,
                    now=now,
                )
                status_counts[left_limited.status] += 1
                status_counts[right_limited.status] += 1
                hard_guards = {
                    "scheduler_gap_hold",
                    "joint_limit_rejected",
                    "target_feedback_rejected",
                    "command_feedback_guard_hold",
                }
                if left_limited.status in hard_guards or right_limited.status in hard_guards:
                    raise RuntimeError(
                        "Dual executor guard: "
                        f"left={left_limited.status} right={right_limited.status}"
                    )

                sends = (
                    send_pool.submit(left_backend.send, left_limited.command),
                    send_pool.submit(right_backend.send, right_limited.command),
                )
                for sent in sends:
                    sent.result()

                left_grip_status = wait_gripper(left_gripper, timeout_sec=0.2)
                right_grip_status = wait_gripper(right_gripper, timeout_sec=0.2)
                last_left_gripper_feedback = float(left_grip_status.msg.value)
                last_right_gripper_feedback = float(right_grip_status.msg.value)
                left_grip_target = (
                    normalized_gripper(last_left_gripper_command)
                    if target.left_gripper_target is None
                    else target.left_gripper_target
                )
                right_grip_target = pregrasp_assist.gripper_target(
                    raw_right_grip_target
                )
                left_grip_command = left_gripper_follower.step(
                    left_grip_target,
                    measured_width_m=float(left_grip_status.msg.value),
                    measured_force_n=float(left_grip_status.msg.force),
                    now=now,
                )
                right_grip_command = right_gripper_follower.step(
                    right_grip_target,
                    measured_width_m=float(right_grip_status.msg.value),
                    measured_force_n=float(right_grip_status.msg.force),
                    now=now,
                )
                gripper_guards = {
                    "invalid_gripper_input_hold",
                    "gripper_scheduler_gap_hold",
                    "gripper_policy_rejected",
                }
                if left_grip_command.status in gripper_guards or right_grip_command.status in gripper_guards:
                    raise RuntimeError(
                        "Dual gripper guard: "
                        f"left={left_grip_command.status} right={right_grip_command.status}"
                    )
                if (
                    abs(left_grip_command.width_m - last_left_gripper_command) >= 0.0002
                    or abs(right_grip_command.width_m - last_right_gripper_command) >= 0.0002
                    or now - last_gripper_send >= 0.5
                ):
                    left_gripper.move_gripper_m(
                        value=left_grip_command.width_m, force=args.gripper_force_n
                    )
                    right_gripper.move_gripper_m(
                        value=right_grip_command.width_m, force=args.gripper_force_n
                    )
                    last_left_gripper_command = left_grip_command.width_m
                    last_right_gripper_command = right_grip_command.width_m
                    last_gripper_send = now

                left_feedback_msg = left_robot.get_joint_angles()
                right_feedback_msg = right_robot.get_joint_angles()
                if left_feedback_msg is None or right_feedback_msg is None:
                    raise RuntimeError("A joint feedback stream disappeared")
                left_measured = np.asarray(left_feedback_msg.msg, dtype=np.float64)
                right_measured = np.asarray(right_feedback_msg.msg, dtype=np.float64)
                left_error = np.rad2deg(left_limited.command - left_measured)
                right_error = np.rad2deg(right_limited.command - right_measured)
                if max(float(np.max(np.abs(left_error))), float(np.max(np.abs(right_error)))) > args.max_command_error_deg:
                    raise RuntimeError(
                        f"Dual command tracking error: left={left_error.tolist()} right={right_error.tolist()}"
                    )
                if not arm_status_ok(left_robot) or not arm_status_ok(right_robot):
                    raise RuntimeError("A NERO arm left healthy CAN/CPV mode")
                feedback_dt = max(now - last_feedback_time, 1e-3)
                max_feedback_velocity_deg_s = float(
                    max(
                        np.max(np.abs(np.rad2deg(left_measured - last_left_feedback))) / feedback_dt,
                        np.max(np.abs(np.rad2deg(right_measured - last_right_feedback))) / feedback_dt,
                    )
                )
                last_left_feedback = left_measured
                last_right_feedback = right_measured
                last_feedback_time = now
                last_left_command = left_limited.command.copy()
                last_right_command = right_limited.command.copy()

                post_release_overriding = height_guard_target is not None
                if (
                    args.chunk_mode == "rtc_time"
                    and post_release_overriding
                    and not post_release_was_overriding
                ):
                    # Any post-release Cartesian correction, including a
                    # Z-only height lock, intentionally diverges from the
                    # policy queue. Restart RTC from the CPV command rather
                    # than handing the stale queue tail to the new chunk.
                    progress.reanchor_hold(
                        np.concatenate(
                            (
                                last_left_command,
                                [normalized_gripper(last_left_gripper_command)],
                                last_right_command,
                                [normalized_gripper(last_right_gripper_command)],
                            )
                        ),
                        now=now,
                    )
                    rtc_generation += 1
                    summary["post_release_rtc_reanchors"] = int(
                        summary.get("post_release_rtc_reanchors", 0)
                    ) + 1
                    print(
                        "RTC re-anchored after post-release Cartesian assist",
                        flush=True,
                    )
                post_release_was_overriding = post_release_overriding

                right_gripper_normalized = normalized_gripper(
                    float(right_grip_status.msg.value)
                )
                if args.exit_on_right_gripper_cycle:
                    if right_gripper_normalized <= args.stage_close_threshold:
                        right_gripper_cycle_closed = True
                    if (
                        right_gripper_cycle_closed
                        and right_gripper_normalized >= args.stage_open_threshold
                    ):
                        right_gripper_cycle_opened = True
                    # CAN feedback is quantized, so differencing samples at 30 Hz
                    # creates false 1-5 deg/s spikes while CPV is physically holding.
                    # The release-hold command plus the existing tracking-error guard
                    # is the reliable stationary criterion.
                    settled = stage_release_hold
                    if settled and stage_settle_started is None:
                        stage_settle_started = now
                    elif not settled:
                        stage_settle_started = None
                    if (
                        stage_settle_started is not None
                        and now - stage_settle_started >= args.stage_settle_sec
                    ):
                        completion_reason = "right_gripper_cycle_settled"
                        print(
                            "PASS: stage boundary reached; right gripper cycle and arm settle confirmed",
                            flush=True,
                        )
                        break

                row = {
                    "elapsed_sec": now - started,
                    "chunk_count": chunk_count,
                    "status": target.status,
                    "phase_steps": target.phase_steps,
                    "target_phase_steps": target.target_phase_steps,
                    "rtc_cumulative_feedback_progress_steps": (
                        progress.cumulative_progress_steps
                        if args.chunk_mode == "rtc"
                        else float(progress.emitted_steps)
                        if args.chunk_mode == "rtc_time"
                        else None
                    ),
                    "rtc_feedback_progress_rate_hz": (
                        progress.progress_rate_hz
                        if args.chunk_mode == "rtc"
                        else args.rtc_action_hz
                        if args.chunk_mode == "rtc_time"
                        else None
                    ),
                    "rtc_remaining_steps": (
                        progress.remaining_steps
                        if args.chunk_mode in {"rtc", "rtc_time"}
                        else None
                    ),
                    "projection_error_deg": None
                    if target.projection_error_rad is None
                    else float(np.rad2deg(target.projection_error_rad)),
                    "left_feedback": left_measured.tolist(),
                    "right_feedback": right_measured.tolist(),
                    "left_command": left_limited.command.tolist(),
                    "right_command": right_limited.command.tolist(),
                    "left_command_velocity_deg_s": np.rad2deg(
                        left_limited.velocity
                    ).tolist(),
                    "right_command_velocity_deg_s": np.rad2deg(
                        right_limited.velocity
                    ).tolist(),
                    "left_feedback_governed": left_limited.feedback_governed,
                    "right_feedback_governed": right_limited.feedback_governed,
                    "left_command_error_deg": left_error.tolist(),
                    "right_command_error_deg": right_error.tolist(),
                    "left_gripper_feedback_m": float(left_grip_status.msg.value),
                    "right_gripper_feedback_m": float(right_grip_status.msg.value),
                    "left_gripper_target": float(left_grip_target),
                    "right_gripper_target": float(right_grip_target),
                    "gripper_catchup_hold": gripper_catchup_hold,
                    "right_gripper_close_gap": right_close_gap,
                    "max_feedback_velocity_deg_s": max_feedback_velocity_deg_s,
                    "stage_right_gripper_cycle_closed": right_gripper_cycle_closed,
                    "stage_right_gripper_cycle_opened": right_gripper_cycle_opened,
                    "stage_settle_elapsed_sec": (
                        None
                        if stage_settle_started is None
                        else now - stage_settle_started
                    ),
                    "pregrasp_assist": pregrasp_assist.status(
                        right_measured
                    ).__dict__,
                    "post_release_height_guard": post_release_height_guard.status(
                        right_measured
                    ).__dict__,
                }
                tick_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if now >= next_health:
                    check_driver_health(left_robot)
                    check_driver_health(right_robot)
                    check_gripper_health(left_gripper)
                    check_gripper_health(right_gripper)
                    rtc_progress_hz = (
                        args.rtc_action_hz
                        if args.chunk_mode == "rtc_time"
                        else progress.progress_rate_hz
                    )
                    rtc_progress_text = (
                        f"progress={rtc_progress_hz:.1f}step/s "
                        f"remain={progress.remaining_steps:d} "
                        if args.chunk_mode in {"rtc", "rtc_time"}
                        else ""
                    )
                    print(
                        f"t={now-started:5.1f}s chunks={chunk_count:3d} "
                        f"status={target.status:>12s} "
                        f"phase={target.phase_steps if target.phase_steps is not None else -1:.1f}"
                        f"->{target.target_phase_steps if target.target_phase_steps is not None else -1:.1f} "
                        f"{rtc_progress_text}"
                        f"proj={row['projection_error_deg'] if row['projection_error_deg'] is not None else -1:.2f}deg "
                        f"grip={float(left_grip_status.msg.value)*1000:.1f}/"
                        f"{float(right_grip_status.msg.value)*1000:.1f}mm "
                        f"pregrasp={pregrasp_assist.state} "
                        f"postrelease={post_release_height_guard.state}",
                        flush=True,
                    )
                    tick_stream.flush()
                    next_health = now + 1.0

        summary.update(
            completed=True,
            completion_reason=completion_reason,
            chunks=chunk_count,
            rtc_rejected_chunks=rtc_rejected_chunks
            if args.chunk_mode in {"rtc", "rtc_time"}
            else None,
            fixed_rejected_chunks=fixed_rejected_chunks
            if args.chunk_mode == "fixed_horizon"
            else None,
            status_counts=dict(status_counts),
        )
        print("PASS: guarded bimanual policy stream finished and is holding", flush=True)
    except KeyboardInterrupt:
        summary["completion_reason"] = "keyboard_interrupt"
        print("Keyboard interrupt: both CPV arms are holding", flush=True)
        raise
    except Exception as exc:
        summary.update(
            completed=False,
            completion_reason="exception",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        for backend in (left_backend, right_backend):
            if backend is not None and backend.prepared:
                try:
                    backend.hold()
                except Exception as exc:
                    print(f"WARNING: CPV hold failed: {exc}", flush=True)
        for robot in (left_robot, right_robot):
            if robot is not None:
                robot.disconnect()
        if client is not None:
            client.close()
        for camera in reversed(cameras):
            camera.stop()
        tick_stream.close()
        chunk_stream.close()
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"bimanual policy log={run_dir}", flush=True)


if __name__ == "__main__":
    main()
