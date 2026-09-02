#!/usr/bin/env python3

"""Preview or execute one heavily guarded NERO policy joint step."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.dual_can import FOLLOWER_CAN_PORT
from nero_vla.dual_can import require_can_role
from nero_vla.eth_state import NeroEthStateReader
from nero_vla.policy_client import OpenPiPolicyClient, port_open
from nero_vla.real_policy_dry_run import DEFAULT_EXTERNAL, DEFAULT_WRIST
from nero_vla.real_policy_dry_run import load_gripper_calibration
from nero_vla.real_policy_dry_run import normalize_gripper
from nero_vla.image_tools import prepare_external_model_image, resize_with_pad


CONFIRMATION = "MOVE POLICY STEP"
ACTION_HORIZON = 16
ACTION_FPS = 30


def wait_for(getter, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = getter()
        if value is not None:
            return value
        time.sleep(0.05)
    return None


def select_future_action_index(elapsed_sec: float, minimum_lead_sec: float) -> int:
    """Select the first action whose nominal target time still lies in the future."""
    index = math.ceil((elapsed_sec + minimum_lead_sec) * ACTION_FPS) - 1
    return max(0, index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview one policy target; execute one clamped arm-only step only with --execute"
    )
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
    parser.add_argument("--minimum-lead-ms", type=float, default=50.0)
    parser.add_argument("--max-policy-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-step-deg", type=float, default=0.25)
    parser.add_argument("--max-step-norm-deg", type=float, default=0.50)
    parser.add_argument("--max-start-speed-deg-s", type=float, default=2.0)
    parser.add_argument("--max-state-mismatch-deg", type=float, default=0.50)
    parser.add_argument("--max-departure-deg", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument("--motion-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "nero_ws/logs/guarded_policy_step",
    )
    args = parser.parse_args()
    require_can_role(args.can_port, "follower")

    positive = (
        args.minimum_lead_ms,
        args.max_policy_delta_deg,
        args.max_step_deg,
        args.max_step_norm_deg,
        args.max_start_speed_deg_s,
        args.max_state_mismatch_deg,
        args.max_departure_deg,
        args.motion_timeout,
    )
    if any(value <= 0 for value in positive):
        parser.error("all timing and guard values must be positive")
    if not 1 <= args.speed_percent <= 5:
        parser.error("speed-percent must remain in the guarded range 1..5")
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
    config = None
    record: dict = {
        "mode": "execute_one_guarded_arm_step" if args.execute else "preview_only",
        "executed": False,
        "prompt": args.prompt,
        "noise_seed": args.noise_seed,
        "guards": {
            "minimum_lead_ms": args.minimum_lead_ms,
            "max_policy_delta_deg": args.max_policy_delta_deg,
            "max_step_deg": args.max_step_deg,
            "max_step_norm_deg": args.max_step_norm_deg,
            "max_start_speed_deg_s": args.max_start_speed_deg_s,
            "max_state_mismatch_deg": args.max_state_mismatch_deg,
            "max_departure_deg": args.max_departure_deg,
            "speed_percent": args.speed_percent,
        },
    }

    print("ARM ONLY: the gripper command is deliberately ignored", flush=True)
    print("Default mode is preview-only; --execute still requires typed confirmation", flush=True)
    external.start()
    wrist.start()
    eth.start()
    try:
        external.wait_ready(8.0)
        wrist.wait_ready(8.0)
        eth.wait_ready(5.0, require_gripper=True)
        # Establish the persistent policy connection before timestamping the
        # observation. Connection setup is not part of recurring inference latency.
        client = OpenPiPolicyClient(args.policy_host, args.policy_port, open_timeout=5.0)
        if args.execute:
            config = create_agx_arm_config(
                robot=ArmModel.NERO,
                firmeware_version=NeroFW.V120,
                interface="socketcan",
                channel=args.can_port,
            )
            robot = AgxArmFactory.create_arm(config)
            gripper_driver = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            robot.set_joint_limits_enabled(True)
            if not robot.get_joint_limits_enabled():
                raise RuntimeError("SDK software joint limits could not be enabled")
            robot.connect()
            if wait_for(robot.get_joint_angles, 5.0) is None:
                raise RuntimeError("No decoded CAN joint feedback received")
            if wait_for(robot.get_arm_status, 5.0) is None:
                raise RuntimeError("No decoded CAN arm status received")
            gripper_status = wait_for(gripper_driver.get_gripper_status, 5.0)
            if gripper_status is None:
                raise RuntimeError("No decoded CAN gripper status received")
            gripper_flags = gripper_status.msg.foc_status
            gripper_faults = [
                name
                for name in (
                    "voltage_too_low",
                    "motor_overheating",
                    "driver_overcurrent",
                    "driver_overheating",
                    "sensor_status",
                    "driver_error_status",
                )
                if bool(getattr(gripper_flags, name, False))
            ]
            record["gripper_preflight"] = {
                "value": float(gripper_status.msg.value),
                "force_n": float(gripper_status.msg.force),
                "mode": str(gripper_status.msg.mode),
                "faults": gripper_faults,
            }
            if gripper_faults:
                raise RuntimeError(f"Gripper health guard rejected execution: {gripper_faults}")
            print("Keep a hand on the physical emergency stop and clear the workspace.", flush=True)
            print(
                "After confirmation, one fresh inference will immediately execute one "
                "clamped arm-only step.",
                flush=True,
            )
            if input(f"Type {CONFIRMATION!r} to authorize exactly one step: ") != CONFIRMATION:
                print("Cancelled; no enable or motion command was sent", flush=True)
                return

        state = eth.snapshot(max_age_sec=0.2)
        start_speed_deg_s = np.abs(np.rad2deg(state.joint_velocity_rad_s))
        if start_speed_deg_s.shape != (7,):
            raise RuntimeError(f"Expected 7 joint velocities, got {start_speed_deg_s.shape}")
        if float(start_speed_deg_s.max()) > args.max_start_speed_deg_s:
            raise RuntimeError(
                "Robot must be stationary before inference: "
                f"max speed={start_speed_deg_s.max():.2f}deg/s"
            )

        external_frame = external.latest(max_age_sec=0.2)
        wrist_frame = wrist.latest(max_age_sec=0.2)
        gripper = normalize_gripper(state.gripper_stroke_mm, closed_mm, open_mm)
        observation_state = np.asarray([*state.joint_position_rad, gripper], dtype=np.float32)
        observation_ns = state.joint_monotonic_ns
        assembled_ns = time.monotonic_ns()
        observation = {
            "observation/external_image": prepare_external_model_image(
                external_frame.image_rgb
            ),
            "observation/wrist_image": resize_with_pad(wrist_frame.image_rgb),
            "observation/state": observation_state,
            "prompt": args.prompt,
            "__openpi_noise": fixed_noise,
        }

        result, transport = client.infer_timed(observation)
        response_ns = time.monotonic_ns()
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid action chunk: shape={actions.shape}")

        elapsed_sec = (response_ns - observation_ns) / 1e9
        action_index = select_future_action_index(elapsed_sec, args.minimum_lead_ms / 1000.0)
        if action_index >= ACTION_HORIZON:
            raise RuntimeError(
                f"All policy actions are stale: infer={elapsed_sec * 1000:.1f}ms "
                f"selected_index={action_index}"
            )
        policy_target = actions[action_index, :7].astype(np.float64)
        raw_delta_deg = np.rad2deg(policy_target - observation_state[:7])
        if float(np.max(np.abs(raw_delta_deg))) > args.max_policy_delta_deg:
            raise RuntimeError(
                "Policy innovation guard rejected target: "
                f"raw_delta_deg={np.round(raw_delta_deg, 3).tolist()}"
            )

        limited_delta_deg = np.clip(raw_delta_deg, -args.max_step_deg, args.max_step_deg)
        norm_deg = float(np.linalg.norm(limited_delta_deg))
        if norm_deg > args.max_step_norm_deg:
            limited_delta_deg *= args.max_step_norm_deg / norm_deg

        record.update(
            {
                "observation_state": observation_state.tolist(),
                "input_age_ms": {
                    "joint_at_assembly": (assembled_ns - state.joint_monotonic_ns) / 1e6,
                    "external_at_assembly":
                        (assembled_ns - external_frame.monotonic_ns) / 1e6,
                    "wrist_at_assembly": (assembled_ns - wrist_frame.monotonic_ns) / 1e6,
                },
                "start_speed_deg_s": start_speed_deg_s.tolist(),
                "inference_ms": elapsed_sec * 1000.0,
                "transport": transport,
                "selected_action_index": action_index,
                "selected_nominal_time_ms": 1000.0 * (action_index + 1) / ACTION_FPS,
                "raw_policy_target": policy_target.tolist(),
                "raw_policy_delta_deg": raw_delta_deg.tolist(),
                "limited_delta_deg": limited_delta_deg.tolist(),
                "ignored_gripper_action": float(actions[action_index, 7]),
            }
        )
        np.save(run_dir / "actions.npy", actions)

        print(f"inference={elapsed_sec * 1000:.1f}ms selected action[{action_index}]", flush=True)
        print(f"current_deg={np.round(np.rad2deg(observation_state[:7]), 3).tolist()}", flush=True)
        print(f"raw_delta_deg={np.round(raw_delta_deg, 3).tolist()}", flush=True)
        print(f"LIMITED_delta_deg={np.round(limited_delta_deg, 3).tolist()}", flush=True)
        print(f"gripper action {actions[action_index, 7]:.3f} is NOT executed", flush=True)

        if not args.execute:
            print(f"PREVIEW PASS: no CAN connection; record={run_dir}", flush=True)
            return

        can_feedback = wait_for(robot.get_joint_angles, 5.0)
        status = wait_for(robot.get_arm_status, 5.0)
        if can_feedback is None or status is None:
            raise RuntimeError("No decoded CAN feedback/status received")
        can_current = np.asarray(can_feedback.msg, dtype=np.float64)
        mismatch_deg = np.rad2deg(can_current - observation_state[:7])
        if float(np.max(np.abs(mismatch_deg))) > args.max_state_mismatch_deg:
            raise RuntimeError(
                "ETH/CAN state mismatch guard rejected execution: "
                f"mismatch_deg={np.round(mismatch_deg, 3).tolist()}"
            )
        if status.msg.arm_status != 0x00:
            raise RuntimeError(
                f"Arm is not in an accepted state: arm={status.msg.arm_status} "
                f"motion={status.msg.motion_status}"
            )

        command_target = can_current + np.deg2rad(limited_delta_deg)
        configured_limits = np.asarray(
            [config["joint_limits"][f"joint{index}"] for index in range(1, 8)],
            dtype=np.float64,
        )
        outside_limits = np.flatnonzero(
            (command_target < configured_limits[:, 0])
            | (command_target > configured_limits[:, 1])
        )
        if outside_limits.size:
            joints = [int(index + 1) for index in outside_limits]
            raise RuntimeError(f"Command target violates configured joint limits: joints={joints}")
        record["can_current"] = can_current.tolist()
        record["eth_can_mismatch_deg"] = mismatch_deg.tolist()
        record["command_target"] = command_target.tolist()
        record["sdk_joint_limits_enabled"] = True
        print(f"CAN_current_deg={np.round(np.rad2deg(can_current), 3).tolist()}", flush=True)
        print(f"command_deg={np.round(np.rad2deg(command_target), 3).tolist()}", flush=True)

        robot.set_speed_percent(args.speed_percent)
        if not robot.enable(timeout=2.0):
            raise RuntimeError("Failed to switch to CAN control and enable all joints")
        can_mode_status = None
        can_mode_deadline = time.monotonic() + 0.5
        while time.monotonic() < can_mode_deadline:
            candidate = robot.get_arm_status()
            if candidate is not None and int(candidate.msg.ctrl_mode) == 0x01:
                can_mode_status = candidate
                break
            time.sleep(0.02)
        if can_mode_status is None:
            raise RuntimeError("Arm did not confirm CAN_CTRL mode after enable; no motion sent")
        record["ctrl_mode_after_enable"] = int(can_mode_status.msg.ctrl_mode)
        robot.move_j(command_target.tolist())
        record["executed"] = True
        deadline = time.monotonic() + args.motion_timeout
        reached = False
        while time.monotonic() < deadline:
            feedback = robot.get_joint_angles()
            status = robot.get_arm_status()
            if feedback is not None:
                actual = np.asarray(feedback.msg, dtype=np.float64)
                departure_deg = np.abs(np.rad2deg(actual - can_current))
                if float(departure_deg.max()) > args.max_departure_deg:
                    robot.electronic_emergency_stop()
                    record["emergency_stop"] = "departure_guard"
                    raise RuntimeError(
                        f"Departure guard triggered emergency stop: {departure_deg.tolist()}"
                    )
                error_deg = np.abs(np.rad2deg(actual - command_target))
                if float(error_deg.max()) <= 0.15:
                    reached = True
                    break
            if status is not None and status.msg.arm_status != 0x00:
                robot.electronic_emergency_stop()
                record["emergency_stop"] = "arm_status_guard"
                raise RuntimeError(
                    f"Arm status guard triggered emergency stop: arm={status.msg.arm_status} "
                    f"motion={status.msg.motion_status}"
                )
            time.sleep(0.02)
        if not reached:
            robot.electronic_emergency_stop()
            record["emergency_stop"] = "motion_timeout"
            raise TimeoutError("Target was not reached; electronic emergency stop was sent")
        final_feedback = robot.get_joint_angles()
        if final_feedback is not None:
            final = np.asarray(final_feedback.msg, dtype=np.float64)
            record["final_joint"] = final.tolist()
            print(f"final_deg={np.round(np.rad2deg(final), 3).tolist()}", flush=True)
        print(f"PASS: one guarded policy step executed; record={run_dir}", flush=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (run_dir / "record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
