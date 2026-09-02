"""Forward the NERO leader command stream between two isolated CAN buses."""

from __future__ import annotations

import argparse
import errno
import json
import os
import selectors
import signal
import socket
import struct
import threading
import time

from nero_vla.dual_can import BRIDGE_SOCKET_PATH
from nero_vla.dual_can import CAN_FRAME
from nero_vla.dual_can import CAN_SFF_MASK
from nero_vla.dual_can import FOLLOWER_CAN_PORT
from nero_vla.dual_can import LEADER_CAN_PORT
from nero_vla.dual_can import LEADER_FORWARD_IDS
from nero_vla.dual_can import arbitration_id
from nero_vla.dual_can import bridge_command
from nero_vla.dual_can import measure_role_pose_mismatch
from nero_vla.dual_can import require_can_role


CAN_RAW_FILTER = 1


class LeaderFollowerBridge:
    def __init__(
        self,
        leader_can: str,
        follower_can: str,
        socket_path: str,
        *,
        dry_run: bool,
        max_resume_mismatch_deg: float,
        forward_hz: float,
    ) -> None:
        if leader_can == follower_can:
            raise ValueError("leader and follower CAN interfaces must differ")
        self.leader_can = leader_can
        self.follower_can = follower_can
        self.socket_path = socket_path
        self.dry_run = dry_run
        self.max_resume_mismatch_deg = max_resume_mismatch_deg
        self.forward_hz = forward_hz
        self.forward_period_sec = 1.0 / forward_hz
        self.paused = not dry_run
        self.last_alignment: dict | None = None
        self.forwarded_frames = 0
        self.tx_dropped_frames = 0
        self.observed_frames = 0
        self.last_frame_monotonic: float | None = None
        self.last_forward_monotonic: float | None = None
        self.next_forward_monotonic = time.monotonic()
        self.latest_frames: dict[int, bytes] = {}
        self.started_monotonic = time.monotonic()
        self.last_can_bind_monotonic = self.started_monotonic
        self.can_rebinds = 0
        self.stopping = False

        self.leader_socket = self.open_leader_socket()
        self.follower_socket = self.open_follower_socket()

        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(socket_path):
            try:
                bridge_command("status", socket_path)
            except (OSError, RuntimeError, json.JSONDecodeError):
                os.unlink(socket_path)
            else:
                raise RuntimeError(f"another CAN bridge is already using {socket_path}")
        self.control_socket.bind(socket_path)
        self.control_socket.listen(4)
        self.control_socket.settimeout(0.5)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.leader_socket, selectors.EVENT_READ, "can")
        self.control_thread: threading.Thread | None = None

    def open_leader_socket(self) -> socket.socket:
        leader_socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        filters = b"".join(
            struct.pack("=II", frame_id, CAN_SFF_MASK)
            for frame_id in sorted(LEADER_FORWARD_IDS)
        )
        leader_socket.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_FILTER, filters)
        leader_socket.bind((self.leader_can,))
        return leader_socket

    def open_follower_socket(self) -> socket.socket:
        follower_socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        follower_socket.bind((self.follower_can,))
        follower_socket.setblocking(False)
        return follower_socket

    def rebind_can_sockets(self) -> None:
        self.selector.unregister(self.leader_socket)
        self.leader_socket.close()
        self.follower_socket.close()
        self.leader_socket = self.open_leader_socket()
        self.follower_socket = self.open_follower_socket()
        self.selector.register(self.leader_socket, selectors.EVENT_READ, "can")
        self.latest_frames.clear()
        self.next_forward_monotonic = time.monotonic()
        self.last_can_bind_monotonic = self.next_forward_monotonic
        self.can_rebinds += 1
        print(
            f"CAN sockets rebound after stale input; count={self.can_rebinds}",
            flush=True,
        )

    def status(self) -> dict:
        now = time.monotonic()
        return {
            "ok": True,
            "leader_can": self.leader_can,
            "follower_can": self.follower_can,
            "dry_run": self.dry_run,
            "paused": self.paused,
            "max_resume_mismatch_deg": self.max_resume_mismatch_deg,
            "forward_hz": self.forward_hz,
            "last_alignment": self.last_alignment,
            "observed_frames": self.observed_frames,
            "forwarded_frames": self.forwarded_frames,
            "tx_dropped_frames": self.tx_dropped_frames,
            "can_rebinds": self.can_rebinds,
            "uptime_sec": now - self.started_monotonic,
            "last_frame_age_sec": (
                None if self.last_frame_monotonic is None else now - self.last_frame_monotonic
            ),
            "last_forward_age_sec": (
                None
                if self.last_forward_monotonic is None
                else now - self.last_forward_monotonic
            ),
        }

    def handle_control(self) -> None:
        connection, _ = self.control_socket.accept()
        with connection:
            connection.settimeout(1.0)
            try:
                command = connection.recv(128).decode("ascii", errors="replace").strip()
            except TimeoutError:
                return
            if command == "status":
                response = self.status()
            elif command == "check":
                try:
                    self.last_alignment = measure_role_pose_mismatch(
                        self.leader_can, self.follower_can
                    )
                    response = self.status()
                except Exception as exc:
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "status": self.status(),
                    }
            elif command == "pause":
                self.paused = True
                response = self.status()
            elif command == "resume":
                try:
                    if not self.dry_run:
                        self.last_alignment = measure_role_pose_mismatch(
                            self.leader_can, self.follower_can
                        )
                        if (
                            self.last_alignment["max_abs_error_deg"]
                            > self.max_resume_mismatch_deg
                        ):
                            joint_errors = ", ".join(
                                f"J{index}={error:+.2f}deg"
                                for index, error in enumerate(
                                    self.last_alignment["error_deg"], start=1
                                )
                            )
                            raise RuntimeError(
                                "leader/follower pose mismatch is "
                                f"{self.last_alignment['max_abs_error_deg']:.2f}deg, "
                                f"limit={self.max_resume_mismatch_deg:.2f}deg; "
                                f"errors=[{joint_errors}]"
                            )
                    self.paused = False
                    response = self.status()
                except Exception as exc:
                    self.paused = True
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "status": self.status(),
                    }
            elif command == "stop":
                self.stopping = True
                response = self.status()
            else:
                response = {"ok": False, "error": f"unknown command: {command}"}
            try:
                connection.sendall(
                    (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                )
            except OSError:
                return

    def control_loop(self) -> None:
        while not self.stopping:
            try:
                self.handle_control()
            except TimeoutError:
                continue
            except OSError:
                if self.stopping:
                    return
                time.sleep(0.05)

    def handle_can(self) -> None:
        frame = self.leader_socket.recv(CAN_FRAME.size)
        if len(frame) != CAN_FRAME.size:
            return
        can_id, _, _ = CAN_FRAME.unpack(frame)
        frame_id = arbitration_id(can_id)
        if frame_id not in LEADER_FORWARD_IDS:
            return
        self.observed_frames += 1
        self.last_frame_monotonic = time.monotonic()
        self.latest_frames[frame_id] = frame
        if self.paused or self.dry_run:
            return
        now = time.monotonic()
        if now < self.next_forward_monotonic:
            return
        self.next_forward_monotonic = now + self.forward_period_sec
        for latest_id in sorted(self.latest_frames):
            try:
                self.follower_socket.send(self.latest_frames[latest_id])
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS):
                    raise
                self.tx_dropped_frames += 1
                continue
            self.forwarded_frames += 1
        self.last_forward_monotonic = now

    def run(self) -> None:
        self.control_thread = threading.Thread(
            target=self.control_loop,
            name="nero-can-bridge-control",
            daemon=True,
        )
        self.control_thread.start()
        while not self.stopping:
            events = self.selector.select(timeout=0.5)
            for _key, _ in events:
                self.handle_can()
            now = time.monotonic()
            last_input = (
                self.last_can_bind_monotonic
                if self.last_frame_monotonic is None
                else max(self.last_can_bind_monotonic, self.last_frame_monotonic)
            )
            if not self.paused and now - last_input > 1.0:
                self.rebind_can_sockets()
        self.control_thread.join(timeout=1.0)

    def close(self) -> None:
        self.selector.close()
        self.leader_socket.close()
        self.follower_socket.close()
        self.control_socket.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="NERO isolated dual-CAN leader/follower bridge")
    parser.add_argument(
        "command", choices=("run", "status", "check", "pause", "resume", "stop")
    )
    parser.add_argument("--leader-can", default=LEADER_CAN_PORT)
    parser.add_argument("--follower-can", default=FOLLOWER_CAN_PORT)
    parser.add_argument("--socket", default=BRIDGE_SOCKET_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-resume-mismatch-deg", type=float, default=15.0)
    parser.add_argument("--forward-hz", type=float, default=50.0)
    args = parser.parse_args()
    if args.max_resume_mismatch_deg <= 0:
        parser.error("max-resume-mismatch-deg must be positive")
    if not 1.0 <= args.forward_hz <= 200.0:
        parser.error("forward-hz must be in [1, 200]")

    if args.command != "run":
        print(json.dumps(bridge_command(args.command, args.socket), indent=2, sort_keys=True))
        return

    leader = require_can_role(args.leader_can, "leader")
    follower = require_can_role(args.follower_can, "follower")
    print(
        f"roles verified leader={leader.interface} follower={follower.interface} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    bridge = LeaderFollowerBridge(
        args.leader_can,
        args.follower_can,
        args.socket,
        dry_run=args.dry_run,
        max_resume_mismatch_deg=args.max_resume_mismatch_deg,
        forward_hz=args.forward_hz,
    )
    if not args.dry_run:
        print(
            "real bridge started PAUSED; use the resume command after pose alignment",
            flush=True,
        )

    def request_stop(_signum, _frame) -> None:
        bridge.stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        bridge.run()
    finally:
        print(json.dumps(bridge.status(), sort_keys=True), flush=True)
        bridge.close()


if __name__ == "__main__":
    main()
