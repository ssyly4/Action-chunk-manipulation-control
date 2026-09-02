"""Record passive NERO demonstrations into a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

from nero_vla.camera_reader import CameraFrame, SyntheticCameraReader, V4L2CameraReader
from nero_vla.eth_state import NeroEthSnapshot, NeroEthStateReader
from nero_vla.dual_can import BRIDGE_SOCKET_PATH
from nero_vla.dual_can import FOLLOWER_CAN_PORT
from nero_vla.dual_can import LEADER_CAN_PORT
from nero_vla.dual_can import bridge_command
from nero_vla.dual_can import bridge_status_if_running
from nero_vla.image_tools import rotate_external_image
from nero_vla.robot_config import CAPTURE_HOME_PROFILE


JOINT_NAMES = [f"joint_{index}.pos" for index in range(1, 8)]
STATE_NAMES = [*JOINT_NAMES, "gripper.pos"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FFplayWristPreview:
    """Display captured wrist frames without opening the V4L2 device twice."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> None:
        ffplay = Path(sys.executable).with_name("ffplay")
        if not ffplay.is_file():
            raise RuntimeError(f"ffplay was not found next to the active Python: {ffplay}")
        self._process = subprocess.Popen(
            [
                str(ffplay),
                "-loglevel", "error",
                "-window_title", "NERO wrist recording",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._write_frames, daemon=True)
        self._thread.start()

    def submit(self, image_rgb: np.ndarray) -> None:
        if self._error is not None:
            raise RuntimeError(f"Wrist preview failed: {self._error}")
        if image_rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"Unexpected preview frame shape: {image_rgb.shape}")
        try:
            self._frames.put_nowait(image_rgb)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(image_rgb)

    def stop(self) -> None:
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._process is not None:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def _write_frames(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            while True:
                frame = self._frames.get()
                if frame is None:
                    return
                self._process.stdin.write(frame.tobytes())
                self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._error = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class GripperCalibration:
    closed_mm: float
    open_mm: float

    @classmethod
    def load(cls, path: Path) -> "GripperCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = cls(float(payload["closed_mm"]), float(payload["open_mm"]))
        if abs(result.open_mm - result.closed_mm) < 5:
            raise ValueError("Gripper calibration span must be at least 5 mm")
        return result

    def normalize(self, stroke_mm: float | None) -> float:
        if stroke_mm is None:
            raise RuntimeError("No gripper stroke in ETH state")
        value = (stroke_mm - self.closed_mm) / (self.open_mm - self.closed_mm)
        return float(np.clip(value, 0, 1))


@dataclass(frozen=True)
class SynchronizedSample:
    scheduled_monotonic_ns: int
    state: NeroEthSnapshot
    state_vector: np.ndarray
    external: CameraFrame
    wrist: CameraFrame


def dataset_features(height: int, width: int, image_dtype: str) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES,
        },
        "observation.images.external": {
            "dtype": image_dtype,
            "shape": (width, height, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.images.wrist": {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
    }


def state_vector(state: NeroEthSnapshot, calibration: GripperCalibration) -> np.ndarray:
    values = [*state.joint_position_rad, calibration.normalize(state.gripper_stroke_mm)]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid NERO state vector: {result}")
    return result


def take_sample(
    scheduled_ns: int,
    eth: NeroEthStateReader,
    external: Any,
    wrist: Any,
    calibration: GripperCalibration,
) -> SynchronizedSample:
    state = eth.snapshot(max_age_sec=0.2)
    return SynchronizedSample(
        scheduled_monotonic_ns=scheduled_ns,
        state=state,
        state_vector=state_vector(state, calibration),
        external=external.latest(max_age_sec=0.2),
        wrist=wrist.latest(max_age_sec=0.2),
    )


def timing_record(index: int, observation: SynchronizedSample, action: SynchronizedSample) -> dict:
    tick = observation.scheduled_monotonic_ns
    return {
        "frame_index": index,
        "scheduled_monotonic_ns": tick,
        "joint_monotonic_ns": observation.state.joint_monotonic_ns,
        "external_monotonic_ns": observation.external.monotonic_ns,
        "wrist_monotonic_ns": observation.wrist.monotonic_ns,
        "action_source_joint_monotonic_ns": action.state.joint_monotonic_ns,
        "joint_age_ms": (tick - observation.state.joint_monotonic_ns) / 1e6,
        "external_age_ms": (tick - observation.external.monotonic_ns) / 1e6,
        "wrist_age_ms": (tick - observation.wrist.monotonic_ns) / 1e6,
        "action_lookahead_ms": (
            action.state.joint_monotonic_ns - observation.state.joint_monotonic_ns
        ) / 1e6,
        "external_sequence": observation.external.sequence,
        "wrist_sequence": observation.wrist.sequence,
    }


def joint_motion_speed_deg_s(
    observation: SynchronizedSample,
    action: SynchronizedSample,
) -> float:
    elapsed_sec = (
        action.scheduled_monotonic_ns - observation.scheduled_monotonic_ns
    ) / 1e9
    if elapsed_sec <= 0:
        return 0.0
    delta_rad = action.state_vector[:7] - observation.state_vector[:7]
    return float(np.max(np.abs(np.rad2deg(delta_rad))) / elapsed_sec)


def wait_until(deadline_ns: int) -> None:
    remaining = deadline_ns - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1e9)


def finish_requested() -> bool:
    """Consume an Enter press without blocking the 30 Hz recording loop."""
    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def choose_after_episode(auto_save: bool) -> str:
    if auto_save:
        return "save"
    while True:
        answer = input("Save episode [s], discard [d], or quit [q]? ").strip().lower()
        if answer in {"s", "save"}:
            return "save"
        if answer in {"d", "discard"}:
            return "discard"
        if answer in {"q", "quit"}:
            return "quit"


def return_to_capture_home(
    speed_percent: int,
    follower_can: str,
    leader_can: str,
    bridge_socket: str,
    timeout_sec: float,
    max_travel_deg: float,
) -> bool:
    bridge_status = bridge_status_if_running(bridge_socket)
    if bridge_status is None:
        print("WARNING: automatic dual-arm return requires a running CAN bridge", flush=True)
        return False
    if bridge_status.get("dry_run", False):
        print("WARNING: automatic dual-arm return cannot use a dry-run CAN bridge", flush=True)
        return False
    if not bridge_status.get("paused", False):
        bridge_command("pause", bridge_socket)
        print("leader/follower bridge paused for synchronized dual-arm return", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/hardware/nero_dual_return_home.py"),
        "--execute",
        "--speed-percent",
        str(speed_percent),
        "--timeout",
        str(timeout_sec),
        "--max-travel-deg",
        str(max_travel_deg),
        "--leader-can",
        leader_can,
        "--follower-can",
        follower_can,
        "--bridge-socket",
        bridge_socket,
    ]
    print(
        f"returning both arms to fixed capture home {CAPTURE_HOME_PROFILE} "
        f"at {speed_percent}% speed",
        flush=True,
    )
    started_ns = time.time_ns()
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return_logs = PROJECT_ROOT / "logs/dual_return_home"
        candidates = sorted(
            (
                path
                for path in return_logs.glob("dual_return_*.json")
                if path.stat().st_mtime_ns >= started_ns
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        recoverable_alignment_error = False
        if candidates:
            try:
                payload = json.loads(candidates[0].read_text(encoding="utf-8"))
                recoverable_alignment_error = (
                    "Final leader/follower alignment is unsafe"
                    in str(payload.get("error", ""))
                )
            except (OSError, json.JSONDecodeError):
                pass
        if recoverable_alignment_error:
            print(
                "final alignment settled outside the strict tolerance; "
                "retrying one guarded low-speed correction",
                flush=True,
            )
            result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"WARNING: automatic return failed with exit={result.returncode}; "
            "the recorded episode is still pending and the bridge remains paused",
            flush=True,
        )
        return False
    print("both arms reached fixed capture home; bridge resumed", flush=True)
    return True


def write_diagnostics(root: Path, episode_index: int, rows: list[dict]) -> Path:
    path = root / "diagnostics" / f"episode_{episode_index:06d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record NERO demonstrations in LeRobot v3 format")
    parser.add_argument("--host", default="10.90.0.150")
    parser.add_argument(
        "--external-camera",
        default="/dev/v4l/by-path/pci-0000:07:00.4-usb-0:1.1:1.0-video-index0",
    )
    parser.add_argument(
        "--wrist-camera",
        default="/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2:1.0-video-index0",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-seconds", type=float, default=20)
    parser.add_argument(
        "--episodes",
        "--successful-episodes",
        dest="episodes",
        type=int,
        default=1,
        help="number of successfully saved episodes; discarded attempts do not count",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--repo-id", default="local/nero_demo")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=Path.home() / "nero_ws/config/nero_gripper_calibration.json",
    )
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument("--synthetic-cameras", action="store_true")
    parser.add_argument(
        "--preview-wrist",
        action="store_true",
        help="show the same wrist frames that are written to the dataset",
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument(
        "--no-motion-start-detection",
        action="store_false",
        dest="motion_start_detection",
        help="start writing frames immediately after Enter",
    )
    parser.set_defaults(motion_start_detection=True)
    parser.add_argument("--motion-start-threshold-deg-s", type=float, default=2.0)
    parser.add_argument("--motion-start-consecutive", type=int, default=3)
    parser.add_argument("--motion-preroll-seconds", type=float, default=0.2)
    parser.add_argument(
        "--no-auto-return-home",
        action="store_false",
        dest="auto_return_home",
        help="do not return the follower arm after each recorded attempt",
    )
    parser.set_defaults(auto_return_home=True)
    parser.add_argument("--return-speed-percent", type=int, default=8)
    parser.add_argument("--return-delay-seconds", type=float, default=1.0)
    parser.add_argument("--leader-can-port", default=LEADER_CAN_PORT)
    parser.add_argument("--follower-can-port", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--can-bridge-socket", default=BRIDGE_SOCKET_PATH)
    parser.add_argument("--return-timeout-sec", type=float, default=75.0)
    parser.add_argument("--return-max-travel-deg", type=float, default=60.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing dataset until the final successful-episode target is reached",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.episode_seconds <= 0 or args.episodes <= 0:
        parser.error("fps, episode-seconds, and episodes must be positive")
    if (
        args.motion_start_threshold_deg_s <= 0
        or args.motion_start_consecutive <= 0
        or args.motion_preroll_seconds < 0
    ):
        parser.error(
            "motion threshold/consecutive must be positive and preroll must be non-negative"
        )
    if not 1 <= args.return_speed_percent <= 20:
        parser.error("return-speed-percent must be in [1, 20]")
    if (
        args.return_delay_seconds < 0
        or args.return_timeout_sec <= 0
        or args.return_max_travel_deg <= 0
    ):
        parser.error(
            "return delay must be non-negative; return timeout and max travel must be positive"
        )
    root_has_data = args.root.exists() and any(args.root.iterdir())
    if root_has_data and not args.resume:
        parser.error(f"dataset root already exists and is not empty: {args.root}")
    if args.resume and not root_has_data:
        parser.error(f"cannot resume: dataset root is missing or empty: {args.root}")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    calibration = GripperCalibration.load(args.gripper_calibration)
    image_dtype = args.image_storage
    use_video = image_dtype == "video"
    rgb_encoder = RGBEncoderConfig(
        vcodec="h264",
        crf=23,
        preset="ultrafast",
        g=args.fps,
    )
    dataset_options = {
        "root": args.root,
        "image_writer_threads": 0 if use_video else 4,
        "rgb_encoder": rgb_encoder,
        "streaming_encoding": use_video,
        "encoder_queue_maxsize": 120,
        "encoder_threads": 4,
    }
    def open_dataset() -> Any:
        if args.resume:
            result = LeRobotDataset.resume(args.repo_id, **dataset_options)
            if result.fps != args.fps:
                result.finalize()
                parser.error(
                    f"cannot resume at {args.fps} Hz: existing dataset uses {result.fps} Hz"
                )
            print(
                f"resuming dataset with {result.num_episodes} successful episodes; "
                f"final target={args.episodes}"
            )
            return result
        return LeRobotDataset.create(
            args.repo_id,
            args.fps,
            robot_type="nero",
            features=dataset_features(args.height, args.width, image_dtype),
            use_videos=use_video,
            **dataset_options,
        )

    if args.synthetic_cameras:
        external = SyntheticCameraReader("external", args.width, args.height, args.fps)
        wrist = SyntheticCameraReader("wrist", args.width, args.height, args.fps)
    else:
        external = V4L2CameraReader(
            args.external_camera, width=args.width, height=args.height, fps=args.fps, name="external"
        )
        wrist = V4L2CameraReader(
            args.wrist_camera, width=args.width, height=args.height, fps=args.fps, name="wrist"
        )

    eth = NeroEthStateReader(args.host)
    dataset = None
    external.start()
    wrist.start()
    eth.start()
    preview: FFplayWristPreview | None = None
    try:
        external.wait_ready()
        wrist.wait_ready()
        eth.wait_ready()
        dataset = open_dataset()
        if use_video:
            print("video_encoding=h264/ultrafast streaming=true queue=120 threads=4")
        if args.preview_wrist:
            preview = FFplayWristPreview(args.width, args.height, args.fps)
            preview.start()
            preview.submit(wrist.latest().image_rgb)
        print("inputs ready; recording is passive except for guarded automatic return-home")
        if args.auto_return_home and not return_to_capture_home(
            args.return_speed_percent,
            args.follower_can_port,
            args.leader_can_port,
            args.can_bridge_socket,
            args.return_timeout_sec,
            args.return_max_travel_deg,
        ):
            print(
                "WARNING: starting from the current manually prepared pose; "
                "automatic return requires follower CAN control",
                flush=True,
            )
        attempt_count = 0
        while dataset.num_episodes < args.episodes:
            attempt_count += 1
            saved_count = dataset.num_episodes
            if not args.auto_start:
                input(
                    f"Press Enter to arm successful episode "
                    f"{saved_count + 1}/{args.episodes} (attempt {attempt_count}): "
                )
            period_ns = round(1e9 / args.fps)
            frame_limit = round(args.episode_seconds * args.fps)
            next_tick = time.monotonic_ns()
            previous: SynchronizedSample | None = None
            diagnostics: list[dict] = []

            if args.motion_start_detection:
                preroll_frames = round(args.motion_preroll_seconds * args.fps)
                armed_samples: deque[SynchronizedSample] = deque(
                    maxlen=preroll_frames + args.motion_start_consecutive + 1
                )
                motion_streak = 0
                print(
                    "armed; waiting for sustained arm motion "
                    f">={args.motion_start_threshold_deg_s:.1f}deg/s for "
                    f"{args.motion_start_consecutive} frames; "
                    "armed waiting time is not recorded",
                    flush=True,
                )
                while True:
                    wait_until(next_tick)
                    current = take_sample(next_tick, eth, external, wrist, calibration)
                    if preview is not None:
                        preview.submit(current.wrist.image_rgb)
                    armed_samples.append(current)
                    if previous is not None:
                        speed_deg_s = joint_motion_speed_deg_s(previous, current)
                        motion_streak = (
                            motion_streak + 1
                            if speed_deg_s >= args.motion_start_threshold_deg_s
                            else 0
                        )
                        if motion_streak >= args.motion_start_consecutive:
                            print(
                                f"motion detected at {speed_deg_s:.2f}deg/s; "
                                f"recording with {preroll_frames} preroll frames",
                                flush=True,
                            )
                            next_tick += period_ns
                            break
                    previous = current
                    next_tick += period_ns

                buffered_samples = list(armed_samples)
                for observation, action in zip(
                    buffered_samples[:-1], buffered_samples[1:]
                ):
                    dataset.add_frame(
                        {
                            "observation.state": observation.state_vector,
                            "observation.images.external": rotate_external_image(
                                observation.external.image_rgb
                            ),
                            "observation.images.wrist": observation.wrist.image_rgb,
                            "action": action.state_vector,
                            "task": args.task,
                        }
                    )
                    diagnostics.append(
                        timing_record(len(diagnostics), observation, action)
                    )
                previous = buffered_samples[-1]

            print(
                f"recording task={args.task!r} max_frames={frame_limit}; "
                "press Enter when the demonstration is complete"
            )
            while len(diagnostics) < frame_limit:
                wait_until(next_tick)
                current = take_sample(next_tick, eth, external, wrist, calibration)
                if preview is not None:
                    preview.submit(current.wrist.image_rgb)
                if previous is not None:
                    frame = {
                        "observation.state": previous.state_vector,
                        "observation.images.external": rotate_external_image(
                            previous.external.image_rgb
                        ),
                        "observation.images.wrist": previous.wrist.image_rgb,
                        "action": current.state_vector,
                        "task": args.task,
                    }
                    dataset.add_frame(frame)
                    diagnostics.append(timing_record(len(diagnostics), previous, current))
                previous = current
                next_tick += period_ns
                if diagnostics and finish_requested():
                    print(f"episode stopped early at {len(diagnostics) / args.fps:.2f}s")
                    break

            if args.auto_return_home:
                if args.return_delay_seconds:
                    print(
                        f"recording stopped; waiting {args.return_delay_seconds:.1f}s "
                        "before synchronized return",
                        flush=True,
                    )
                    time.sleep(args.return_delay_seconds)
                return_to_capture_home(
                    args.return_speed_percent,
                    args.follower_can_port,
                    args.leader_can_port,
                    args.can_bridge_socket,
                    args.return_timeout_sec,
                    args.return_max_travel_deg,
                )

            decision = choose_after_episode(args.auto_save)
            if decision == "save":
                episode_index = dataset.num_episodes
                dataset.save_episode()
                diagnostic_path = write_diagnostics(args.root, episode_index, diagnostics)
                print(
                    f"saved episode={episode_index} frames={len(diagnostics)} "
                    f"success={dataset.num_episodes}/{args.episodes} "
                    f"attempts={attempt_count} diagnostics={diagnostic_path}"
                )
            else:
                dataset.clear_episode_buffer()
                if decision == "quit":
                    print(
                        f"episode discarded; quitting with "
                        f"success={dataset.num_episodes}/{args.episodes} "
                        f"attempts={attempt_count}"
                    )
                    break
                print(
                    f"episode discarded; success remains "
                    f"{dataset.num_episodes}/{args.episodes}; "
                    "a replacement attempt is required"
                )
        if dataset.num_episodes == args.episodes:
            print(
                f"target reached: {dataset.num_episodes} successful episodes "
                f"from {attempt_count} attempts"
            )
        rates = eth.rates_hz()
        print(
            f"source_rates_hz eth_joint={rates.get('/jointStates', 0):.1f} "
            f"external={external.measured_hz:.1f} wrist={wrist.measured_hz:.1f}"
        )
    except KeyboardInterrupt:
        if dataset is not None and dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        print("interrupted; pending episode discarded")
    finally:
        if preview is not None:
            preview.stop()
        eth.stop()
        external.stop()
        wrist.stop()
        if dataset is not None:
            dataset.finalize()


if __name__ == "__main__":
    main()
