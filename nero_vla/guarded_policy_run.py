#!/usr/bin/env python3

"""Run a short, slow, arm-only NERO policy closed loop with hard guards."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from nero_vla.camera_reader import V4L2CameraReader
from nero_vla.dual_can import FOLLOWER_CAN_PORT
from nero_vla.dual_can import require_can_role
from nero_vla.eth_state import NeroEthStateReader
from nero_vla.guarded_policy_step import ACTION_FPS, ACTION_HORIZON
from nero_vla.guarded_policy_step import select_future_action_index
from nero_vla.guarded_policy_step import wait_for
from nero_vla.policy_client import OpenPiPolicyClient, port_open
from nero_vla.real_policy_dry_run import DEFAULT_EXTERNAL, DEFAULT_WRIST
from nero_vla.real_policy_dry_run import load_gripper_calibration
from nero_vla.real_policy_dry_run import normalize_gripper
from nero_vla.image_tools import prepare_external_model_image, resize_with_pad


CONFIRMATION = "RUN ARM 10S"


def main() -> None:
    parser = argparse.ArgumentParser(description="Short guarded NERO arm-only policy run")
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
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--minimum-lead-ms", type=float, default=50.0)
    parser.add_argument("--max-policy-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-step-deg", type=float, default=1.0)
    parser.add_argument("--max-step-norm-deg", type=float, default=2.0)
    parser.add_argument("--max-state-mismatch-deg", type=float, default=2.0)
    parser.add_argument("--max-command-departure-deg", type=float, default=1.75)
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "nero_ws/logs/guarded_policy_run",
    )
    args = parser.parse_args()
    require_can_role(args.can_port, "follower")

    parser.error(
        "disabled: repeated move_j replanning caused stop-start real-robot motion; "
        "use the offline trajectory dry-run while the continuous backend is qualified"
    )

    if not 0 < args.duration <= 10.0:
        parser.error("duration must be in (0, 10] seconds for this first closed-loop stage")
    if not 0 < args.max_step_deg <= 1.0:
        parser.error("max-step-deg must be in (0, 1.0]")
    if not 0 < args.max_step_norm_deg <= 2.0:
        parser.error("max-step-norm-deg must be in (0, 2.0]")
    if not 1 <= args.speed_percent <= 5:
        parser.error("speed-percent must be in 1..5")
    if args.minimum_lead_ms <= 0 or args.max_policy_delta_deg <= 0:
        parser.error("timing and policy guards must be positive")
    if not port_open(args.policy_host, args.policy_port, 3.0):
        raise RuntimeError(f"Policy server is not reachable at {args.policy_host}:{args.policy_port}")

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    closed_mm, open_mm = load_gripper_calibration(args.gripper_calibration)
    noise = np.random.default_rng(args.noise_seed).standard_normal(
        (ACTION_HORIZON, 32), dtype=np.float32
    )
    external = V4L2CameraReader(
        args.external_camera, width=args.width, height=args.height,
        fps=args.camera_fps, name="external",
    )
    wrist = V4L2CameraReader(
        args.wrist_camera, width=args.width, height=args.height,
        fps=args.camera_fps, name="wrist",
    )
    eth = NeroEthStateReader(args.robot_host)
    client: OpenPiPolicyClient | None = None
    robot = None
    rows: list[dict] = []
    summary: dict = {
        "mode": "guarded_arm_only_closed_loop",
        "executed": False,
        "prompt": args.prompt,
        "duration_limit_sec": args.duration,
        "noise_seed": args.noise_seed,
        "gripper_commands_sent": 0,
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
        )
        robot = AgxArmFactory.create_arm(config)
        gripper_driver = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.set_joint_limits_enabled(True)
        robot.connect()
        initial_can = wait_for(robot.get_joint_angles, 5.0)
        initial_status = wait_for(robot.get_arm_status, 5.0)
        gripper_status = wait_for(gripper_driver.get_gripper_status, 5.0)
        if initial_can is None or initial_status is None or gripper_status is None:
            raise RuntimeError("Incomplete CAN preflight feedback")
        if initial_status.msg.arm_status != 0x00:
            raise RuntimeError(f"Arm preflight fault: {initial_status.msg.arm_status}")
        flags = gripper_status.msg.foc_status
        gripper_faults = [
            name for name in (
                "voltage_too_low", "motor_overheating", "driver_overcurrent",
                "driver_overheating", "sensor_status", "driver_error_status",
            ) if bool(getattr(flags, name, False))
        ]
        if gripper_faults:
            raise RuntimeError(f"Gripper preflight fault: {gripper_faults}")

        print("10-SECOND ARM-ONLY RUN", flush=True)
        print("gripper output remains disabled; keep a hand on the physical E-stop", flush=True)
        print(
            f"limits: {args.max_step_deg:.2f}deg/joint per replan, "
            f"{args.max_step_norm_deg:.2f}deg vector, speed={args.speed_percent}%",
            flush=True,
        )
        if input(f"Type {CONFIRMATION!r} to start: ") != CONFIRMATION:
            print("Cancelled; no enable or motion command was sent", flush=True)
            return

        robot.set_speed_percent(args.speed_percent)
        if not robot.enable(timeout=2.0):
            raise RuntimeError("Failed to switch to CAN control and enable joints")
        mode_deadline = time.monotonic() + 0.5
        while time.monotonic() < mode_deadline:
            status = robot.get_arm_status()
            if status is not None and int(status.msg.ctrl_mode) == 0x01:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("CAN_CTRL mode was not confirmed; no motion sent")

        started = time.monotonic()
        previous_can = np.asarray(initial_can.msg, dtype=np.float64)
        while time.monotonic() - started < args.duration:
            state = eth.snapshot(max_age_sec=0.2)
            external_frame = external.latest(max_age_sec=0.2)
            wrist_frame = wrist.latest(max_age_sec=0.2)
            gripper = normalize_gripper(state.gripper_stroke_mm, closed_mm, open_mm)
            observation_state = np.asarray([*state.joint_position_rad, gripper], dtype=np.float32)
            observation_ns = state.joint_monotonic_ns
            observation = {
                "observation/external_image": prepare_external_model_image(
                    external_frame.image_rgb
                ),
                "observation/wrist_image": resize_with_pad(wrist_frame.image_rgb),
                "observation/state": observation_state,
                "prompt": args.prompt,
                "__openpi_noise": noise,
            }
            result, transport = client.infer_timed(observation)
            response_ns = time.monotonic_ns()
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid action chunk: {actions.shape}")
            elapsed_sec = (response_ns - observation_ns) / 1e9
            action_index = select_future_action_index(
                elapsed_sec, args.minimum_lead_ms / 1000.0
            )
            if action_index >= ACTION_HORIZON:
                raise RuntimeError(f"Action chunk is stale: {elapsed_sec * 1000:.1f}ms")

            feedback = robot.get_joint_angles()
            status = robot.get_arm_status()
            grip = gripper_driver.get_gripper_status()
            if feedback is None or status is None or grip is None:
                raise RuntimeError("CAN feedback disappeared")
            if status.msg.arm_status != 0x00 or int(status.msg.ctrl_mode) != 0x01:
                raise RuntimeError(
                    f"Arm mode/status guard: ctrl={status.msg.ctrl_mode} "
                    f"arm={status.msg.arm_status}"
                )
            grip_flags = grip.msg.foc_status
            if any(bool(getattr(grip_flags, name, False)) for name in (
                "motor_overheating", "driver_overcurrent", "driver_overheating",
                "sensor_status", "driver_error_status",
            )):
                raise RuntimeError("Gripper health fault appeared during run")

            can_current = np.asarray(feedback.msg, dtype=np.float64)
            movement_since_last_deg = np.abs(np.rad2deg(can_current - previous_can))
            if float(movement_since_last_deg.max()) > args.max_command_departure_deg:
                robot.electronic_emergency_stop()
                raise RuntimeError(
                    "Unexpected inter-replan movement; emergency stop sent: "
                    f"{movement_since_last_deg.tolist()}"
                )
            mismatch_deg = np.rad2deg(can_current - observation_state[:7])
            if float(np.abs(mismatch_deg).max()) > args.max_state_mismatch_deg:
                raise RuntimeError(f"ETH/CAN mismatch: {mismatch_deg.tolist()}")

            policy_target = actions[action_index, :7].astype(np.float64)
            raw_delta_deg = np.rad2deg(policy_target - can_current)
            if float(np.abs(raw_delta_deg).max()) > args.max_policy_delta_deg:
                raise RuntimeError(f"Policy innovation rejected: {raw_delta_deg.tolist()}")
            limited_delta_deg = np.clip(
                raw_delta_deg, -args.max_step_deg, args.max_step_deg
            )
            norm_deg = float(np.linalg.norm(limited_delta_deg))
            if norm_deg > args.max_step_norm_deg:
                limited_delta_deg *= args.max_step_norm_deg / norm_deg
            target = can_current + np.deg2rad(limited_delta_deg)
            limits = np.asarray(
                [config["joint_limits"][f"joint{i}"] for i in range(1, 8)],
                dtype=np.float64,
            )
            if np.any((target < limits[:, 0]) | (target > limits[:, 1])):
                raise RuntimeError("Joint-limit guard rejected target")

            robot.move_j(target.tolist())
            summary["executed"] = True
            rows.append({
                "step": len(rows),
                "elapsed_sec": time.monotonic() - started,
                "inference_ms": elapsed_sec * 1000.0,
                "action_index": action_index,
                "can_current": can_current.tolist(),
                "raw_delta_deg": raw_delta_deg.tolist(),
                "limited_delta_deg": limited_delta_deg.tolist(),
                "target": target.tolist(),
                "ignored_gripper_action": float(actions[action_index, 7]),
                "transport": transport,
            })
            previous_can = can_current
            print(
                f"step={len(rows):02d} t={rows[-1]['elapsed_sec']:.2f}s "
                f"infer={elapsed_sec * 1000:.0f}ms idx={action_index} "
                f"delta={np.round(limited_delta_deg, 2).tolist()}",
                flush=True,
            )

        final_feedback = robot.get_joint_angles()
        summary.update({
            "steps": len(rows),
            "elapsed_sec": time.monotonic() - started,
            "final_joint": None if final_feedback is None else list(final_feedback.msg),
            "mean_inference_ms": float(np.mean([row["inference_ms"] for row in rows])),
        })
        print(f"PASS: guarded run finished; commands={len(rows)} record={run_dir}", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (run_dir / "steps.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
