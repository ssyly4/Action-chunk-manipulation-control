#!/usr/bin/env python3

"""Run a slow, guarded pi0.5 arm-only stream on a physical NERO."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh
from pyAgxArm.utiles.mdh_kinematics import get_mdh

from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.cpv_backend import NeroCpvPositionBackend
from nero_vla.dual_can import FOLLOWER_CAN_PORT
from nero_vla.dual_can import require_bridge_not_forwarding
from nero_vla.dual_can import require_can_role
from nero_vla.eth_state import NeroEthStateReader
from nero_vla.gripper_controller import RateLimitedGripperFollower
from nero_vla.lift_assist import PostGraspLiftAssist
from nero_vla.lift_assist import rigid_contact_is_stable
from nero_vla.policy_client import OpenPiPolicyClient, port_open
from nero_vla.real_policy_dry_run import DEFAULT_EXTERNAL, DEFAULT_WRIST
from nero_vla.real_policy_dry_run import load_gripper_calibration
from nero_vla.real_policy_dry_run import normalize_gripper
from nero_vla.robot_config import NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD
from nero_vla.image_tools import prepare_external_model_image, resize_with_pad
from nero_vla.terminal_target_filter import TerminalArmTargetFilter
from nero_vla.trajectory_executor import FeedbackProgressActionChunk
from nero_vla.trajectory_executor import RateLimitedJointFollower
from nero_vla.trajectory_executor import TrajectorySample
from nero_vla.trajectory_executor import align_action_chunk_to_state


ACTION_HORIZON = 16
ACTION_HZ = 30.0
CONFIRMATION = "RUN SLOW POLICY 30S"
NERO_MDH = list(get_mdh("nero"))


def flange_height_m(joint_rad: np.ndarray) -> float:
    joints = np.asarray(joint_rad, dtype=np.float64)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("flange height requires seven finite joint positions")
    return float(fk_from_mdh(NERO_MDH, joints.tolist())[2])


def wait_complete_joint_feedback(robot, timeout_sec: float = 5.0) -> np.ndarray:
    """Require several stable, complete J1-J7 snapshots before commanding motion."""
    deadline = time.monotonic() + timeout_sec
    samples: list[np.ndarray] = []
    timestamps: set[float] = set()
    while time.monotonic() < deadline:
        feedback = robot.get_joint_angles()
        if feedback is not None and feedback.timestamp not in timestamps:
            value = np.asarray(feedback.msg, dtype=np.float64)
            if value.shape == (7,) and np.isfinite(value).all():
                timestamps.add(feedback.timestamp)
                samples.append(value.copy())
                if len(samples) >= 8:
                    stacked = np.stack(samples[-8:])
                    if float(np.max(np.ptp(np.rad2deg(stacked), axis=0))) <= 0.25:
                        return stacked[-1]
        time.sleep(0.01)
    raise RuntimeError("No stable, complete seven-joint CAN feedback snapshot")


def wait_enabled(robot, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if robot.enable():
            return
        time.sleep(0.03)
    raise RuntimeError("All seven joints did not report enabled")


def wait_cpv_mode(robot, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if (
            status is not None
            and int(status.msg.ctrl_mode) == 0x01
            and int(status.msg.mode_feedback) == 0x05
            and int(status.msg.arm_status) == 0x00
        ):
            return
        time.sleep(0.01)
    raise RuntimeError("CAN/CPV mode was not confirmed by arm status feedback")


def check_driver_health(robot) -> None:
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "driver_error_status",
        "stall_status",
    )
    for joint_index in range(1, 8):
        state = robot.get_driver_states(joint_index)
        if state is None:
            raise RuntimeError(f"Missing driver state for joint {joint_index}")
        flags = state.msg.foc_status
        active = [name for name in fault_names if bool(getattr(flags, name, False))]
        if active:
            raise RuntimeError(f"Joint {joint_index} driver fault: {active}")


def check_gripper_health(gripper_driver) -> None:
    state = gripper_driver.get_gripper_status()
    if state is None:
        raise RuntimeError("Missing gripper status")
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "sensor_status",
        "driver_error_status",
    )
    flags = state.msg.foc_status
    active = [name for name in fault_names if bool(getattr(flags, name, False))]
    if active:
        raise RuntimeError(f"Gripper health fault: {active}")


def validate_chunk(
    chunk: dict,
    current_feedback: np.ndarray,
    *,
    max_first_action_deg: float,
    max_consecutive_deg: float,
    aligned_action: np.ndarray | None = None,
    aligned_action_offset_steps: float = 0.0,
) -> dict[str, float]:
    actions = chunk["actions"][:, :7]
    observed = chunk["observation_q"]
    aligned = actions[0] if aligned_action is None else np.asarray(aligned_action)[:7]
    if aligned.shape != (7,) or not np.isfinite(aligned).all():
        raise ValueError("aligned_action must contain at least seven finite values")
    from_observation = np.rad2deg(actions - observed[None, :])
    from_current = np.rad2deg(actions - current_feedback[None, :])
    aligned_from_current = np.rad2deg(aligned - current_feedback)
    consecutive = np.rad2deg(np.diff(actions, axis=0))
    metrics = {
        "max_from_observation_deg": float(np.max(np.abs(from_observation))),
        "max_from_current_deg": float(np.max(np.abs(from_current))),
        "max_consecutive_deg": float(np.max(np.abs(consecutive))),
        "max_first_from_observation_deg": float(np.max(np.abs(from_observation[0]))),
        "max_first_from_current_deg": float(np.max(np.abs(from_current[0]))),
        "aligned_action_offset_steps": float(aligned_action_offset_steps),
        "max_aligned_from_current_deg": float(np.max(np.abs(aligned_from_current))),
        "gripper_min": float(np.min(chunk["actions"][:, 7])),
        "gripper_max": float(np.max(chunk["actions"][:, 7])),
        "gripper_max_consecutive": float(np.max(np.abs(np.diff(chunk["actions"][:, 7])))),
    }
    if metrics["max_first_from_observation_deg"] > max_first_action_deg:
        raise RuntimeError(f"Policy first action rejected against observation: {metrics}")
    if metrics["max_consecutive_deg"] > max_consecutive_deg:
        raise RuntimeError(f"Policy chunk has a discontinuous action step: {metrics}")
    if metrics["gripper_min"] < -0.1 or metrics["gripper_max"] > 1.1:
        raise RuntimeError(f"Policy chunk has an invalid gripper target: {metrics}")
    if metrics["gripper_max_consecutive"] > 0.25:
        raise RuntimeError(f"Policy chunk has a discontinuous gripper step: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Slow guarded NERO pi0.5 arm-only stream")
    parser.add_argument("--robot-host", default="10.90.0.150")
    parser.add_argument("--policy-host", default="172.24.1.174")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--can-port", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--external-camera", default=DEFAULT_EXTERNAL)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--prompt", default="pick up the water bottle")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-velocity-deg-s", type=float, default=0.5)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=2.0)
    parser.add_argument("--max-first-action-deg", type=float, default=1.0)
    parser.add_argument("--max-consecutive-action-deg", type=float, default=1.0)
    parser.add_argument("--max-can-eth-mismatch-deg", type=float, default=1.0)
    parser.add_argument("--max-command-error-deg", type=float, default=1.0)
    parser.add_argument("--feedback-governor-error-deg", type=float, default=0.75)
    parser.add_argument("--max-alignment-error-deg", type=float, default=2.0)
    parser.add_argument("--alignment-search-margin-steps", type=float, default=2.0)
    parser.add_argument("--progress-arm-lead-steps", type=float, default=1.0)
    parser.add_argument("--progress-gripper-lead-steps", type=float, default=0.0)
    parser.add_argument("--max-progress-steps-per-tick", type=float, default=1.0)
    parser.add_argument("--chunk-blend-ms", type=float, default=67.0)
    parser.add_argument(
        "--terminal-target-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--approach-filter-flange-height-m", type=float, default=0.20)
    parser.add_argument("--approach-filter-confirm-sec", type=float, default=0.15)
    parser.add_argument("--approach-filter-time-constant-sec", type=float, default=0.10)
    parser.add_argument("--terminal-filter-gripper-threshold", type=float, default=0.35)
    parser.add_argument("--terminal-filter-flange-height-m", type=float, default=0.22)
    parser.add_argument("--terminal-filter-confirm-sec", type=float, default=0.15)
    parser.add_argument("--terminal-filter-time-constant-sec", type=float, default=0.12)
    parser.add_argument("--gripper-speed-mm-s", type=float, default=15.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument("--policy-gap-timeout-sec", type=float, default=2.0)
    parser.add_argument("--post-grasp-lift-assist", action="store_true")
    parser.add_argument("--lift-contact-force-n", type=float, default=0.55)
    parser.add_argument("--lift-gripper-threshold", type=float, default=0.32)
    parser.add_argument("--lift-contact-confirmations", type=int, default=3)
    parser.add_argument("--lift-settle-sec", type=float, default=0.25)
    parser.add_argument("--lift-distance-mm", type=float, default=50.0)
    parser.add_argument("--lift-speed-mm-s", type=float, default=20.0)
    parser.add_argument("--lift-gripper-preload-mm", type=float, default=0.0)
    parser.add_argument("--lift-rigid-contact-gap-mm", type=float, default=1.5)
    parser.add_argument("--lift-contact-loss-confirmations", type=int, default=3)
    parser.add_argument("--exit-after-lift-assist", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "nero_ws/logs/guarded_policy_stream",
    )
    args = parser.parse_args()

    require_can_role(args.can_port, "follower")
    require_bridge_not_forwarding()

    if not 5.0 <= args.duration <= 120.0:
        parser.error("duration must be in [5, 120] seconds")
    if not 1.0 <= args.action_gain <= 2.0:
        parser.error("action-gain must be in [1, 2]")
    if not 0 < args.max_velocity_deg_s <= 10.0:
        parser.error("max-velocity-deg-s must be in (0, 10]")
    if not 0 < args.max_acceleration_deg_s2 <= 30.0:
        parser.error("max-acceleration-deg-s2 must be in (0, 30]")
    if not 0 < args.max_first_action_deg <= 2.0:
        parser.error("max-first-action-deg must be in (0, 2]")
    if not 0 < args.max_consecutive_action_deg <= 3.0:
        parser.error("max-consecutive-action-deg must be in (0, 3]")
    if args.max_can_eth_mismatch_deg <= 0 or args.max_command_error_deg <= 0:
        parser.error("feedback guards must be positive")
    if not 0 < args.feedback_governor_error_deg <= args.max_command_error_deg:
        parser.error(
            "feedback-governor-error-deg must be positive and no greater than max-command-error-deg"
        )
    if not 0 < args.max_alignment_error_deg <= 3.0:
        parser.error("max-alignment-error-deg must be in (0, 3]")
    if not 0 <= args.alignment_search_margin_steps <= 4.0:
        parser.error("alignment-search-margin-steps must be in [0, 4]")
    if not 0.25 <= args.progress_arm_lead_steps <= 2.0:
        parser.error("progress-arm-lead-steps must be in [0.25, 2]")
    if not 0.0 <= args.progress_gripper_lead_steps <= 8.0:
        parser.error("progress-gripper-lead-steps must be in [0, 8]")
    if not 0.25 <= args.max_progress_steps_per_tick <= 2.0:
        parser.error("max-progress-steps-per-tick must be in [0.25, 2]")
    if not 33.0 <= args.chunk_blend_ms <= 200.0:
        parser.error("chunk-blend-ms must be in [33, 200]")
    if not 0.05 <= args.approach_filter_flange_height_m <= 0.5:
        parser.error("approach-filter-flange-height-m must be in [0.05, 0.5]")
    if not 0.0 <= args.approach_filter_confirm_sec <= 1.0:
        parser.error("approach-filter-confirm-sec must be in [0, 1]")
    if not 0.03 <= args.approach_filter_time_constant_sec <= 0.5:
        parser.error("approach-filter-time-constant-sec must be in [0.03, 0.5]")
    if not 0.1 <= args.terminal_filter_gripper_threshold <= 0.6:
        parser.error("terminal-filter-gripper-threshold must be in [0.1, 0.6]")
    if not 0.05 <= args.terminal_filter_flange_height_m <= 0.5:
        parser.error("terminal-filter-flange-height-m must be in [0.05, 0.5]")
    if not 0.0 <= args.terminal_filter_confirm_sec <= 1.0:
        parser.error("terminal-filter-confirm-sec must be in [0, 1]")
    if not 0.03 <= args.terminal_filter_time_constant_sec <= 0.5:
        parser.error("terminal-filter-time-constant-sec must be in [0.03, 0.5]")
    if args.approach_filter_flange_height_m > args.terminal_filter_flange_height_m:
        parser.error(
            "approach-filter-flange-height-m must not exceed terminal-filter-flange-height-m"
        )
    if not 1.0 <= args.gripper_speed_mm_s <= 30.0:
        parser.error("gripper-speed-mm-s must be in [1, 30]")
    if not 0.2 <= args.gripper_force_n <= 2.0:
        parser.error("gripper-force-n must be in [0.2, 2]")
    if not 0.5 <= args.policy_gap_timeout_sec <= 5.0:
        parser.error("policy-gap-timeout-sec must be in [0.5, 5]")
    if not 0.1 <= args.lift_contact_force_n <= 2.0:
        parser.error("lift-contact-force-n must be in [0.1, 2]")
    if not 0.1 <= args.lift_gripper_threshold <= 0.5:
        parser.error("lift-gripper-threshold must be in [0.1, 0.5]")
    if not 1 <= args.lift_contact_confirmations <= 10:
        parser.error("lift-contact-confirmations must be in [1, 10]")
    if not 0.0 <= args.lift_settle_sec <= 2.0:
        parser.error("lift-settle-sec must be in [0, 2]")
    if not 10.0 <= args.lift_distance_mm <= 80.0:
        parser.error("lift-distance-mm must be in [10, 80]")
    if not 5.0 <= args.lift_speed_mm_s <= 30.0:
        parser.error("lift-speed-mm-s must be in [5, 30]")
    if not 0.0 <= args.lift_gripper_preload_mm <= 5.0:
        parser.error("lift-gripper-preload-mm must be in [0, 5]")
    if not 0.5 <= args.lift_rigid_contact_gap_mm <= 5.0:
        parser.error("lift-rigid-contact-gap-mm must be in [0.5, 5]")
    if not 1 <= args.lift_contact_loss_confirmations <= 10:
        parser.error("lift-contact-loss-confirmations must be in [1, 10]")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"Policy server is not reachable at {args.policy_host}:{args.policy_port}")

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    closed_mm, open_mm = load_gripper_calibration(args.gripper_calibration)
    fixed_noise = np.random.default_rng(args.noise_seed).standard_normal(
        (ACTION_HORIZON, 32), dtype=np.float32
    )
    external = V4L2CameraReader(
        args.external_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="external",
    )
    wrist = V4L2CameraReader(
        args.wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        name="wrist",
    )
    eth = NeroEthStateReader(args.robot_host)
    client: OpenPiPolicyClient | None = None
    robot = None
    gripper_driver = None
    backend: NeroCpvPositionBackend | None = None
    last_gripper_feedback_m: float | None = None
    rows: list[dict] = []
    inference_rows: list[dict] = []
    summary: dict = {
        "mode": "guarded_slow_pi05_arm_and_gripper_cpv",
        "executed": False,
        "prompt": args.prompt,
        "observation_state_source": "can",
        "eth_watchdog_enabled": True,
        "duration_limit_sec": args.duration,
        "noise_seed": args.noise_seed,
        "action_gain": args.action_gain,
        "action_hz": ACTION_HZ,
        "action_timeline": "continuous_feedback_progress",
        "gripper_action_timeline": "shared_arm_feedback_progress",
        "max_alignment_error_deg": args.max_alignment_error_deg,
        "alignment_search_margin_steps": args.alignment_search_margin_steps,
        "progress_arm_lead_steps": args.progress_arm_lead_steps,
        "progress_gripper_lead_steps": args.progress_gripper_lead_steps,
        "max_progress_steps_per_tick": args.max_progress_steps_per_tick,
        "chunk_blend_ms": args.chunk_blend_ms,
        "terminal_target_filter": args.terminal_target_filter,
        "approach_filter_flange_height_m": args.approach_filter_flange_height_m,
        "approach_filter_confirm_sec": args.approach_filter_confirm_sec,
        "approach_filter_time_constant_sec": args.approach_filter_time_constant_sec,
        "approach_filter_ticks": 0,
        "approach_filter_activations": 0,
        "approach_filter_first_activation_sec": None,
        "terminal_filter_gripper_threshold": args.terminal_filter_gripper_threshold,
        "terminal_filter_flange_height_m": args.terminal_filter_flange_height_m,
        "terminal_filter_confirm_sec": args.terminal_filter_confirm_sec,
        "terminal_filter_time_constant_sec": args.terminal_filter_time_constant_sec,
        "terminal_filter_ticks": 0,
        "terminal_filter_activations": 0,
        "terminal_filter_first_activation_sec": None,
        "alignment_command_weight": 0.5,
        "alignment_latency_weight": 0.05,
        "alignment_direction_weight": 0.05,
        "max_velocity_deg_s": args.max_velocity_deg_s,
        "max_acceleration_deg_s2": args.max_acceleration_deg_s2,
        "max_command_error_deg": args.max_command_error_deg,
        "feedback_governor_error_deg": args.feedback_governor_error_deg,
        "policy_horizon_displacement": "metrics_only",
        "gripper_commands_sent": 0,
        "gripper_speed_mm_s": args.gripper_speed_mm_s,
        "gripper_force_n": args.gripper_force_n,
        "gripper_control": "raw_policy_with_temporal_interpolation",
        "policy_gap_timeout_sec": args.policy_gap_timeout_sec,
        "policy_gap_holds": 0,
        "post_grasp_lift_assist": args.post_grasp_lift_assist,
        "lift_contact_force_n": args.lift_contact_force_n,
        "lift_gripper_threshold": args.lift_gripper_threshold,
        "lift_contact_confirmations": args.lift_contact_confirmations,
        "lift_settle_sec": args.lift_settle_sec,
        "lift_distance_mm": args.lift_distance_mm,
        "lift_speed_mm_s": args.lift_speed_mm_s,
        "lift_gripper_preload_mm": args.lift_gripper_preload_mm,
        "lift_rigid_contact_gap_mm": args.lift_rigid_contact_gap_mm,
        "lift_contact_loss_confirmations": args.lift_contact_loss_confirmations,
        "rigid_contact_validation_failed": False,
        "lift_assist_triggered": False,
        "lift_assist_triggered_sec": None,
        "exit_after_lift_assist": args.exit_after_lift_assist,
        "lift_assist_completed": False,
        "lift_assist_timed_out": False,
        "completion_reason": "duration_limit",
    }

    external.start()
    wrist.start()
    eth.start()
    try:
        external.wait_ready(8.0)
        wrist.wait_ready(8.0)
        eth.wait_ready(5.0, require_gripper=True)
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=5.0)

        config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.V120,
            interface="socketcan",
            channel=args.can_port,
            joint_limits=NERO_CPV_JOINT_LIMIT_OVERRIDES_RAD,
        )
        robot = AgxArmFactory.create_arm(config)
        gripper_driver = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.set_joint_limits_enabled(True)
        robot.connect()
        firmware = robot.get_firmware(timeout=2.0, min_interval=0.0)
        if firmware is None:
            raise RuntimeError("No firmware response on CAN; no command was sent")
        initial_can = wait_complete_joint_feedback(robot)
        initial_eth = np.asarray(eth.snapshot().joint_position_rad, dtype=np.float64)
        initial_mismatch_deg = np.rad2deg(initial_can - initial_eth)
        if float(np.max(np.abs(initial_mismatch_deg))) > args.max_can_eth_mismatch_deg:
            raise RuntimeError(f"CAN/ETH initial mismatch: {initial_mismatch_deg.tolist()}")
        arm_status = robot.get_arm_status()
        if arm_status is None or int(arm_status.msg.arm_status) != 0x00:
            raise RuntimeError("Arm status is missing or abnormal")
        check_driver_health(robot)
        check_gripper_health(gripper_driver)

        def capture_observation() -> tuple[dict, int, np.ndarray, float, float]:
            assert robot is not None
            assert gripper_driver is not None
            assembled_ns = time.monotonic_ns()
            joint_feedback = robot.get_joint_angles()
            gripper_feedback = gripper_driver.get_gripper_status()
            if joint_feedback is None or gripper_feedback is None:
                raise RuntimeError("Missing CAN feedback while assembling policy observation")
            if gripper_feedback.msg.mode != "width":
                raise RuntimeError("CAN gripper feedback is not in width mode")

            observation_q = np.asarray(joint_feedback.msg, dtype=np.float64)
            if observation_q.shape != (7,) or not np.isfinite(observation_q).all():
                raise RuntimeError(f"Invalid CAN joint observation: {observation_q}")
            measured_gripper = normalize_gripper(
                float(gripper_feedback.msg.value) * 1000.0,
                closed_mm,
                open_mm,
            )
            policy_gripper = measured_gripper
            external_frame = external.latest(max_age_sec=0.2)
            wrist_frame = wrist.latest(max_age_sec=0.2)
            observation = {
                "observation/external_image": prepare_external_model_image(
                    external_frame.image_rgb
                ),
                "observation/wrist_image": resize_with_pad(wrist_frame.image_rgb),
                "observation/state": np.asarray(
                    [*observation_q, policy_gripper], dtype=np.float32
                ),
                "prompt": args.prompt,
                "__openpi_noise": fixed_noise,
            }
            return (
                observation,
                assembled_ns,
                observation_q,
                measured_gripper,
                policy_gripper,
            )

        def infer_chunk(packet: tuple[dict, int, np.ndarray, float, float]) -> dict:
            assert client is not None
            (
                observation,
                observation_ns,
                observation_q,
                measured_gripper,
                policy_gripper,
            ) = packet
            request_started_ns = time.monotonic_ns()
            result, transport = client.infer_timed(observation)
            response_ns = time.monotonic_ns()
            actions = np.asarray(result.get("actions"), dtype=np.float64)
            if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid action chunk: {actions.shape}")
            actions = actions.copy()
            actions[:, :7] = observation_q[None, :] + args.action_gain * (
                actions[:, :7] - observation_q[None, :]
            )
            return {
                "actions": actions,
                "external_image": observation["observation/external_image"].copy(),
                "wrist_image": observation["observation/wrist_image"].copy(),
                "observation_ns": observation_ns,
                "observation_q": observation_q,
                "observation_gripper_measured": measured_gripper,
                "observation_gripper_policy": policy_gripper,
                "request_started_ns": request_started_ns,
                "response_ns": response_ns,
                "inference_ms": (response_ns - request_started_ns) / 1e6,
                "observation_to_response_ms": (response_ns - observation_ns) / 1e6,
                "transport": transport,
            }

        first_chunk = infer_chunk(capture_observation())
        first_metrics = validate_chunk(
            first_chunk,
            initial_can,
            max_first_action_deg=args.max_first_action_deg,
            max_consecutive_deg=args.max_consecutive_action_deg,
        )
        first_metrics.update({
            "alignment_latency_steps": (
                first_chunk["observation_to_response_ms"] / 1000.0 * ACTION_HZ
            ),
            "alignment_search_max_steps": 0.0,
            "alignment_max_feedback_error_deg": first_metrics[
                "max_first_from_current_deg"
            ],
            "alignment_rms_feedback_error_deg": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.rad2deg(
                                first_chunk["actions"][0, :7] - initial_can
                            )
                        )
                    )
                )
            ),
            "alignment_max_command_error_deg": first_metrics[
                "max_first_from_current_deg"
            ],
            "alignment_score": 0.0,
            "alignment_segment_index": 0,
            "alignment_segment_alpha": 0.0,
        })
        print(f"SLOW {args.duration:.0f}-SECOND PI0.5 ARM + GRIPPER CPV STREAM", flush=True)
        print(
            f"firmware={firmware} policy={first_chunk['inference_ms']:.0f}ms "
            f"q_deg={np.round(np.rad2deg(initial_can), 2).tolist()}",
            flush=True,
        )
        print(
            f"fixed_noise_seed={args.noise_seed} chunk_guard={first_metrics} "
            "observation_state=can eth_watchdog=on "
            f"gain={args.action_gain:.2f} "
            f"speed={args.max_velocity_deg_s:.2f}deg/s "
            f"accel={args.max_acceleration_deg_s2:.2f}deg/s^2 "
            f"blend={args.chunk_blend_ms:.0f}ms",
            flush=True,
        )
        print(
            f"gripper enabled: {args.gripper_speed_mm_s:.1f}mm/s, "
            f"force={args.gripper_force_n:.1f}N; keep the physical E-stop ready.",
            flush=True,
        )
        if args.confirm != CONFIRMATION:
            raise RuntimeError(
                f"Preflight passed but execution was not authorized; pass --confirm {CONFIRMATION!r}"
            )

        wait_enabled(robot)
        backend = NeroCpvPositionBackend(
            robot,
            max_command_step_rad=np.deg2rad(
                args.max_velocity_deg_s * 1.5 / ACTION_HZ
            ),
        )
        backend.prepare_hold(initial_can)
        wait_cpv_mode(robot)

        period = 1.0 / ACTION_HZ
        limits = np.asarray(
            [config["joint_limits"][f"joint{index}"] for index in range(1, 8)],
            dtype=np.float64,
        )
        summary["joint_limits_deg"] = np.rad2deg(limits).tolist()
        progress_buffer = FeedbackProgressActionChunk(
            action_hz=ACTION_HZ,
            arm_lead_steps=args.progress_arm_lead_steps,
            gripper_lead_steps=args.progress_gripper_lead_steps,
            max_progress_steps_per_tick=args.max_progress_steps_per_tick,
            blend_duration_sec=args.chunk_blend_ms / 1000.0,
            stale_after_sec=0.15,
        )
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=np.deg2rad(args.max_velocity_deg_s),
            max_acceleration_rad_s2=np.deg2rad(args.max_acceleration_deg_s2),
            max_target_feedback_error_rad=np.deg2rad(180.0),
            max_command_feedback_error_rad=np.deg2rad(args.max_command_error_deg),
            feedback_governor_error_rad=np.deg2rad(args.feedback_governor_error_deg),
            max_tick_interval_sec=3.0 * period,
            joint_limits_rad=limits,
        )
        started = time.monotonic()
        follower.initialize(initial_can, now=started - period)
        initial_gripper_status = gripper_driver.get_gripper_status()
        if initial_gripper_status is None or initial_gripper_status.msg.mode != "width":
            raise RuntimeError("No valid width-mode gripper feedback before execution")
        gripper_follower = RateLimitedGripperFollower(
            closed_m=closed_mm / 1000.0,
            open_m=open_mm / 1000.0,
            max_speed_m_s=args.gripper_speed_mm_s / 1000.0,
            contact_force_n=args.lift_contact_force_n,
            contact_preload_m=args.lift_gripper_preload_mm / 1000.0,
            max_tick_interval_sec=3.0 * period,
            force_hold_enabled=False,
        )
        gripper_follower.initialize(float(initial_gripper_status.msg.value), now=started - period)
        lift_assist = PostGraspLiftAssist(
            enabled=args.post_grasp_lift_assist,
            gripper_state_threshold=args.lift_gripper_threshold,
            contact_force_threshold_n=args.lift_contact_force_n,
            contact_confirmations=args.lift_contact_confirmations,
            settle_sec=args.lift_settle_sec,
            lift_distance_m=args.lift_distance_mm / 1000.0,
            lift_speed_m_s=args.lift_speed_mm_s / 1000.0,
        )
        terminal_filter = TerminalArmTargetFilter(
            enabled=args.terminal_target_filter,
            approach_flange_height_m=args.approach_filter_flange_height_m,
            approach_confirm_sec=args.approach_filter_confirm_sec,
            approach_time_constant_sec=args.approach_filter_time_constant_sec,
            gripper_threshold=args.terminal_filter_gripper_threshold,
            flange_height_m=args.terminal_filter_flange_height_m,
            confirm_sec=args.terminal_filter_confirm_sec,
            time_constant_sec=args.terminal_filter_time_constant_sec,
        )
        gripper_policy_raw_target = float(first_chunk["actions"][0, 7])
        gripper_policy_target = gripper_policy_raw_target
        initial_gripper_normalized = normalize_gripper(
            float(initial_gripper_status.msg.value) * 1000.0,
            closed_mm,
            open_mm,
        )
        gripper_observation_measured = initial_gripper_normalized
        gripper_observation_policy = initial_gripper_normalized
        # The first chunk was also the preflight sample. Its executable
        # timeline starts only now, after CPV preparation, while the arm held.
        first_response = started
        progress_buffer.push(
            first_chunk["actions"],
            initial_phase_steps=0.0,
            observed_at=first_response,
            received_at=first_response,
        )
        inference_rows.append({
            "chunk": 0,
            "inference_ms": first_chunk["inference_ms"],
            "observation_to_response_ms": first_chunk["observation_to_response_ms"],
            "observation_q": first_chunk["observation_q"].tolist(),
            "observation_gripper_measured": first_chunk["observation_gripper_measured"],
            "observation_gripper_policy": first_chunk["observation_gripper_policy"],
            "actions": first_chunk["actions"].tolist(),
            **first_metrics,
        })
        status_counts: Counter[str] = Counter()
        consecutive_holds = 0
        command_guard_started: float | None = None
        chunk_count = 1
        next_tick = started
        last_feedback = initial_can.copy()
        last_command = initial_can.copy()
        last_command_velocity = np.zeros(7, dtype=np.float64)
        last_alignment_offset_steps = 0.0
        last_alignment_latency_steps = first_metrics["alignment_latency_steps"]
        next_health_check = started
        last_gripper_send = started - 1.0
        last_gripper_width_m = float(initial_gripper_status.msg.value)
        gripper_status_counts: Counter[str] = Counter()
        policy_gap_started: float | None = None
        rigid_contact_loss_count = 0
        lift_assist_deadline: float | None = None

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-infer") as executor:
            future: Future = executor.submit(infer_chunk, capture_observation())
            while True:
                now = time.monotonic()
                if lift_assist_deadline is not None and now >= lift_assist_deadline:
                    summary["lift_assist_timed_out"] = True
                    summary["completion_reason"] = "post_grasp_lift_timeout"
                    print(
                        "POST-GRASP LIFT TIMEOUT: target height was not reached; "
                        "ending policy trial for automatic return",
                        flush=True,
                    )
                    break
                if now - started >= args.duration and not lift_assist.active:
                    break
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                scheduler_lag = now - next_tick
                next_tick += period
                if scheduler_lag > 3.0 * period:
                    raise RuntimeError(f"Control scheduler gap: {scheduler_lag * 1000:.1f}ms")

                if future.done():
                    completed = future.result()
                    completed_actions = np.asarray(completed["actions"], dtype=np.float64)
                    accepted_at = now
                    latency_steps = (
                        accepted_at - completed["observation_ns"] / 1e9
                    ) * ACTION_HZ
                    alignment = align_action_chunk_to_state(
                        completed_actions[:, :7],
                        feedback=last_feedback,
                        command=last_command,
                        command_velocity=last_command_velocity,
                        latency_steps=latency_steps,
                        max_alignment_error_rad=np.deg2rad(
                            args.max_alignment_error_deg
                        ),
                        search_margin_steps=args.alignment_search_margin_steps,
                    )
                    metrics = validate_chunk(
                        completed,
                        last_feedback,
                        max_first_action_deg=args.max_first_action_deg,
                        max_consecutive_deg=args.max_consecutive_action_deg,
                        aligned_action=alignment.action,
                        aligned_action_offset_steps=alignment.offset_steps,
                    )
                    metrics.update({
                        "alignment_latency_steps": alignment.latency_steps,
                        "alignment_search_max_steps": alignment.search_max_steps,
                        "alignment_max_feedback_error_deg": float(
                            np.rad2deg(alignment.max_feedback_error_rad)
                        ),
                        "alignment_rms_feedback_error_deg": float(
                            np.rad2deg(alignment.rms_feedback_error_rad)
                        ),
                        "alignment_max_command_error_deg": float(
                            np.rad2deg(alignment.max_command_error_rad)
                        ),
                        "alignment_score": alignment.score,
                        "alignment_segment_index": alignment.segment_index,
                        "alignment_segment_alpha": alignment.segment_alpha,
                    })
                    last_alignment_offset_steps = alignment.offset_steps
                    last_alignment_latency_steps = alignment.latency_steps
                    progress_buffer.push(
                        completed_actions,
                        initial_phase_steps=alignment.offset_steps,
                        observed_at=completed["observation_ns"] / 1e9,
                        received_at=accepted_at,
                    )
                    inference_rows.append({
                        "chunk": chunk_count,
                        "inference_ms": completed["inference_ms"],
                        "observation_to_response_ms": completed["observation_to_response_ms"],
                        "observation_q": completed["observation_q"].tolist(),
                        "observation_gripper_measured": completed[
                            "observation_gripper_measured"
                        ],
                        "observation_gripper_policy": completed[
                            "observation_gripper_policy"
                        ],
                        "actions": completed_actions.tolist(),
                        **metrics,
                    })
                    chunk_count += 1
                    future = executor.submit(infer_chunk, capture_observation())

                progress = progress_buffer.sample(
                    now,
                    arm_feedback=last_feedback,
                )
                arm_target = progress.arm_target
                if progress.gripper_target is None:
                    if progress.status not in {
                        "no_chunk_hold",
                        "stale_chunk_hold",
                    }:
                        raise RuntimeError(
                            "Gripper action timeline unavailable: "
                            f"{progress.status}"
                        )
                    summary["policy_gap_holds"] += 1
                else:
                    gripper_policy_raw_target = progress.gripper_target
                gripper_policy_target = gripper_policy_raw_target
                assisted_target = lift_assist.arm_target(
                    now=now,
                    measured_joint_rad=last_feedback,
                )
                if assisted_target is not None:
                    arm_target = assisted_target
                terminal_sample = terminal_filter.update(
                    arm_target,
                    gripper_target=gripper_policy_raw_target,
                    measured_flange_height_m=flange_height_m(last_feedback),
                    now=now,
                    bypass=assisted_target is not None,
                )
                arm_target = terminal_sample.target
                if terminal_sample.stage == "approach":
                    summary["approach_filter_ticks"] += 1
                    if summary["approach_filter_first_activation_sec"] is None:
                        summary["approach_filter_first_activation_sec"] = now - started
                        print(
                            "APPROACH TARGET FILTER ACTIVE: "
                            f"flange_z={flange_height_m(last_feedback):.3f}m "
                            f"tau={args.approach_filter_time_constant_sec:.3f}s",
                            flush=True,
                        )
                elif terminal_sample.stage == "terminal":
                    summary["terminal_filter_ticks"] += 1
                    if summary["terminal_filter_first_activation_sec"] is None:
                        summary["terminal_filter_first_activation_sec"] = now - started
                        print(
                            "TERMINAL TARGET FILTER ACTIVE: "
                            f"grip={gripper_policy_raw_target:.3f} "
                            f"flange_z={flange_height_m(last_feedback):.3f}m "
                            f"tau={args.terminal_filter_time_constant_sec:.3f}s",
                            flush=True,
                        )
                trajectory = TrajectorySample(
                    target=arm_target,
                    status=progress.status,
                    source_age_sec=progress.source_age_sec,
                    remaining_sec=progress.remaining_sec,
                )
                limited = follower.step(
                    trajectory,
                    measured=last_feedback,
                    now=now,
                )
                status_counts[limited.status] += 1
                guarded = limited.status in {
                    "no_chunk_hold",
                    "stale_chunk_hold",
                    "scheduler_gap_hold",
                    "joint_limit_rejected",
                    "target_feedback_rejected",
                    "command_feedback_guard_hold",
                }
                policy_gap_active = bool(
                    progress.gripper_target is None
                )
                if policy_gap_active:
                    consecutive_holds = 0
                    command_guard_started = None
                    if policy_gap_started is None:
                        policy_gap_started = now
                    elif now - policy_gap_started > args.policy_gap_timeout_sec:
                        raise RuntimeError(
                            "Policy action timeline did not recover within "
                            f"{args.policy_gap_timeout_sec:.1f} seconds"
                        )
                elif limited.status == "command_feedback_guard_hold":
                    policy_gap_started = None
                    consecutive_holds = 0
                    if command_guard_started is None:
                        command_guard_started = now
                    elif now - command_guard_started > 2.0:
                        hold_error_deg = np.rad2deg(limited.command - last_feedback)
                        raise RuntimeError(
                            "Command feedback did not catch up within 2 seconds; "
                            f"held_command_minus_feedback_deg={hold_error_deg.tolist()}"
                        )
                else:
                    policy_gap_started = None
                    command_guard_started = None
                    consecutive_holds = consecutive_holds + 1 if guarded else 0
                if consecutive_holds >= 3:
                    hold_error_deg = np.rad2deg(limited.command - last_feedback)
                    raise RuntimeError(
                        f"Executor remained guarded for three ticks: {limited.status}; "
                        f"held_command_minus_feedback_deg={hold_error_deg.tolist()}"
                    )

                backend.send(limited.command)
                summary["executed"] = True
                live_gripper = gripper_driver.get_gripper_status()
                if live_gripper is None or live_gripper.msg.mode != "width":
                    raise RuntimeError("Gripper feedback disappeared or changed mode")
                last_gripper_feedback_m = float(live_gripper.msg.value)
                gripper_observation_measured = normalize_gripper(
                    float(live_gripper.msg.value) * 1000.0,
                    closed_mm,
                    open_mm,
                )
                triggered = lift_assist.observe(
                    now=now,
                    joint_rad=last_feedback,
                    measured_gripper_state=gripper_observation_measured,
                    policy_gripper_target=gripper_policy_raw_target,
                    measured_force_n=float(live_gripper.msg.force),
                )
                if triggered:
                    summary["lift_assist_triggered"] = True
                    summary["lift_assist_triggered_sec"] = now - started
                    lift_assist_deadline = (
                        now
                        + args.lift_settle_sec
                        + args.lift_distance_mm / args.lift_speed_mm_s
                        + 5.0
                    )
                    print(
                        "POST-GRASP LIFT ASSIST TRIGGERED: "
                        f"grip={gripper_observation_measured:.3f} "
                        f"force={float(live_gripper.msg.force):+.3f}N",
                        flush=True,
                    )
                gripper_policy_target = lift_assist.gripper_target(
                    gripper_policy_raw_target
                )
                gripper_command = gripper_follower.step(
                    gripper_policy_target,
                    measured_width_m=float(live_gripper.msg.value),
                    measured_force_n=float(live_gripper.msg.force),
                    now=now,
                    contact_latch_enabled=lift_assist.active,
                )
                gripper_status_counts[gripper_command.status] += 1
                if (
                    lift_assist.state == "settling"
                    and gripper_follower.contact_latched
                ):
                    stable_contact = rigid_contact_is_stable(
                        measured_width_m=float(live_gripper.msg.value),
                        commanded_width_m=gripper_command.width_m,
                        measured_force_n=float(live_gripper.msg.force),
                        force_threshold_n=args.lift_contact_force_n,
                        minimum_width_gap_m=args.lift_rigid_contact_gap_mm / 1000.0,
                    )
                    rigid_contact_loss_count = (
                        0 if stable_contact else rigid_contact_loss_count + 1
                    )
                gripper_observation_policy = gripper_observation_measured
                gripper_guarded = gripper_command.status in {
                    "invalid_gripper_input_hold",
                    "gripper_scheduler_gap_hold",
                    "gripper_policy_rejected",
                }
                if gripper_guarded:
                    raise RuntimeError(f"Gripper controller guard: {gripper_command.status}")
                if (
                    abs(gripper_command.width_m - last_gripper_width_m) >= 0.0002
                    or now - last_gripper_send >= 0.5
                ):
                    gripper_driver.move_gripper_m(
                        value=gripper_command.width_m,
                        force=args.gripper_force_n,
                    )
                    last_gripper_width_m = gripper_command.width_m
                    last_gripper_send = now
                    summary["gripper_commands_sent"] += 1
                feedback = robot.get_joint_angles()
                status = robot.get_arm_status()
                if feedback is None or status is None:
                    raise RuntimeError("Complete CAN feedback or arm status disappeared")
                if (
                    int(status.msg.ctrl_mode) != 0x01
                    or int(status.msg.mode_feedback) != 0x05
                    or int(status.msg.arm_status) != 0x00
                ):
                    raise RuntimeError(
                        f"Mode/status changed: ctrl={status.msg.ctrl_mode} "
                        f"mode={status.msg.mode_feedback} arm={status.msg.arm_status}"
                    )
                measured = np.asarray(feedback.msg, dtype=np.float64)
                eth_measured = np.asarray(
                    eth.snapshot(max_age_sec=0.2).joint_position_rad,
                    dtype=np.float64,
                )
                command_error_deg = np.rad2deg(measured - limited.command)
                can_eth_deg = np.rad2deg(measured - eth_measured)
                if float(np.max(np.abs(command_error_deg))) > args.max_command_error_deg:
                    raise RuntimeError(f"Command tracking error: {command_error_deg.tolist()}")
                if float(np.max(np.abs(can_eth_deg))) > args.max_can_eth_mismatch_deg:
                    raise RuntimeError(f"CAN/ETH mismatch: {can_eth_deg.tolist()}")
                last_feedback = measured.copy()
                last_command = limited.command.copy()
                last_command_velocity = limited.velocity.copy()

                if now >= next_health_check:
                    check_driver_health(robot)
                    check_gripper_health(gripper_driver)
                    lift_status = lift_assist.status(measured)
                    print(
                        f"t={now - started:5.1f}s chunks={chunk_count:3d} "
                        f"status={limited.status:>10s} "
                        f"q_delta_deg={np.round(np.rad2deg(measured - initial_can), 2).tolist()} "
                        f"align={last_alignment_offset_steps:.1f}/"
                        f"{last_alignment_latency_steps:.1f} "
                        f"phase={progress.phase_steps if progress.phase_steps is not None else -1:.1f}"
                        f"->{progress.arm_target_phase_steps if progress.arm_target_phase_steps is not None else -1:.1f} "
                        f"terminal={terminal_sample.status} "
                        f"grip={float(live_gripper.msg.value) * 1000:.1f}mm "
                        f"target={gripper_policy_target:.3f} "
                        f"grip_obs={gripper_observation_measured:.3f} "
                        f"lift={lift_status.state}/"
                        f"{lift_status.measured_lift_m * 1000:.1f}mm",
                        flush=True,
                    )
                    next_health_check = now + 1.0

                rows.append({
                    "lift_assist": lift_assist.status(measured).__dict__,
                    "tick": len(rows),
                    "elapsed_sec": now - started,
                    "chunk_count": chunk_count,
                    "executor_status": limited.status,
                    "source_age_sec": trajectory.source_age_sec,
                    "remaining_sec": trajectory.remaining_sec,
                    "chunk_phase_steps": progress.phase_steps,
                    "arm_target_phase_steps": progress.arm_target_phase_steps,
                    "phase_feedback_error_deg": (
                        None
                        if progress.phase_feedback_error_rad is None
                        else float(np.rad2deg(progress.phase_feedback_error_rad))
                    ),
                    "gripper_timeline_status": progress.status,
                    "gripper_source_age_sec": progress.source_age_sec,
                    "gripper_remaining_sec": progress.remaining_sec,
                    "terminal_filter_status": terminal_sample.status,
                    "terminal_filter_active": terminal_sample.active,
                    "terminal_filter_stage": terminal_sample.stage,
                    "terminal_filter_alpha": terminal_sample.alpha,
                    "terminal_filter_raw_target": (
                        None
                        if progress.arm_target is None
                        else progress.arm_target.tolist()
                    ),
                    "command": limited.command.tolist(),
                    "command_velocity_deg_s": np.rad2deg(limited.velocity).tolist(),
                    "feedback_governed": limited.feedback_governed,
                    "desired": None if limited.desired is None else limited.desired.tolist(),
                    "can_feedback": measured.tolist(),
                    "eth_feedback": eth_measured.tolist(),
                    "command_error_deg": command_error_deg.tolist(),
                    "can_eth_deg": can_eth_deg.tolist(),
                    "gripper_policy_raw_target": gripper_policy_raw_target,
                    "gripper_policy_target": gripper_policy_target,
                    "flange_height_m": flange_height_m(measured),
                    "gripper_observation_measured": gripper_observation_measured,
                    "gripper_observation_policy": gripper_observation_policy,
                    "gripper_command_m": gripper_command.width_m,
                    "gripper_feedback_m": float(live_gripper.msg.value),
                    "gripper_force_n": float(live_gripper.msg.force),
                    "gripper_status": gripper_command.status,
                    "rigid_contact_loss_count": rigid_contact_loss_count,
                })
                if (
                    lift_assist.state == "settling"
                    and rigid_contact_loss_count
                    >= args.lift_contact_loss_confirmations
                ):
                    summary["rigid_contact_validation_failed"] = True
                    summary["completion_reason"] = "rigid_contact_validation_failed"
                    print(
                        "RIGID GRASP VALIDATION FAILED: contact force or width "
                        "stall was not sustained; ending trial without lift",
                        flush=True,
                    )
                    break
                if (
                    args.exit_after_lift_assist
                    and lift_assist.reached_target(measured)
                ):
                    summary["lift_assist_completed"] = True
                    summary["completion_reason"] = "post_grasp_lift_completed"
                    print(
                        "POST-GRASP LIFT COMPLETE: "
                        f"{lift_assist.status(measured).measured_lift_m * 1000:.1f}mm; "
                        "ending policy trial for automatic return",
                        flush=True,
                    )
                    break

        backend.hold()
        final_lift_status = lift_assist.status(last_feedback)
        summary.update({
            "lift_assist_final": final_lift_status.__dict__,
            "executed": True,
            "ticks": len(rows),
            "chunks": chunk_count,
            "elapsed_sec": time.monotonic() - started,
            "status_counts": dict(status_counts),
            "gripper_status_counts": dict(gripper_status_counts),
            "terminal_filter_activations": terminal_filter.activation_count,
            "approach_filter_activations": terminal_filter.approach_activation_count,
            "terminal_filter_stage_activations": terminal_filter.terminal_activation_count,
            "mean_inference_ms": float(np.mean([row["inference_ms"] for row in inference_rows])),
            "max_tracking_error_deg": float(
                np.max(np.abs([row["command_error_deg"] for row in rows]))
            ),
            "max_can_eth_mismatch_deg": float(
                np.max(np.abs([row["can_eth_deg"] for row in rows]))
            ),
            "mean_alignment_offset_steps": float(
                np.mean(
                    [row["aligned_action_offset_steps"] for row in inference_rows]
                )
            ),
            "mean_alignment_latency_steps": float(
                np.mean([row["alignment_latency_steps"] for row in inference_rows])
            ),
            "max_alignment_feedback_error_deg": float(
                np.max(
                    [
                        row["alignment_max_feedback_error_deg"]
                        for row in inference_rows
                    ]
                )
            ),
            "final_joint_rad": last_feedback.tolist(),
            "final_gripper_feedback_m": float(live_gripper.msg.value),
            "final_gripper_command_m": gripper_command.width_m,
            "final_gripper_policy_raw_target": gripper_policy_raw_target,
            "final_gripper_policy_target": gripper_policy_target,
            "final_gripper_observation_measured": gripper_observation_measured,
            "final_gripper_observation_policy": gripper_observation_policy,
        })
        print(f"PASS: slow policy stream finished and is holding; log={run_dir}", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if backend is not None and backend.prepared:
            try:
                backend.hold()
            except Exception:
                pass
        if gripper_driver is not None and last_gripper_feedback_m is not None:
            try:
                current_gripper = gripper_driver.get_gripper_status()
                hold_width = (
                    last_gripper_feedback_m
                    if current_gripper is None
                    else float(current_gripper.msg.value)
                )
                gripper_driver.move_gripper_m(value=hold_width, force=args.gripper_force_n)
            except Exception:
                pass
        with (run_dir / "steps.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        with (run_dir / "inference.jsonl").open("w", encoding="utf-8") as output:
            for row in inference_rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if robot is not None:
            robot.disconnect()
        if client is not None:
            client.close()
        eth.stop()
        external.stop()
        wrist.stop()


if __name__ == "__main__":
    main()
