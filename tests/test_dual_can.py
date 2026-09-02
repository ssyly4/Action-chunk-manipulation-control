import json
import socket
import threading
import unittest
from unittest.mock import patch

from nero_vla.dual_can import CanRoleError
from nero_vla.dual_can import CanTraffic
from nero_vla.dual_can import FOLLOWER_JOINT_IDS
from nero_vla.dual_can import LEADER_JOINT_IDS
from nero_vla.dual_can import arbitration_id
from nero_vla.dual_can import bridge_command
from nero_vla.dual_can import classify_can_traffic
from nero_vla.dual_can import require_can_role


class DualCanTests(unittest.TestCase):
    def test_classifies_isolated_can_roles(self) -> None:
        self.assertEqual(
            classify_can_traffic(LEADER_JOINT_IDS | {0x151, 0x159}), "leader"
        )
        self.assertEqual(
            classify_can_traffic(FOLLOWER_JOINT_IDS | {0x2A1}), "follower"
        )
        self.assertEqual(
            classify_can_traffic(LEADER_JOINT_IDS | FOLLOWER_JOINT_IDS), "mixed"
        )
        self.assertEqual(classify_can_traffic({0x123}), "unknown")

    def test_arbitration_id_rejects_non_standard_frames(self) -> None:
        self.assertEqual(arbitration_id(0x155), 0x155)
        self.assertEqual(arbitration_id(0x80000155), -1)

    @patch("nero_vla.dual_can.time.sleep", return_value=None)
    @patch("nero_vla.dual_can.sample_can_traffic")
    def test_role_check_recovers_from_transient_zero_frames(
        self, sample, _sleep
    ) -> None:
        sample.side_effect = [
            CanTraffic("can1", frozenset(), 0, 0.25),
            CanTraffic("can1", FOLLOWER_JOINT_IDS, 70, 0.25),
        ]

        traffic = require_can_role("can1", "follower", recovery_timeout_sec=3.0)

        self.assertEqual(traffic.frames, 70)
        self.assertEqual(sample.call_count, 2)

    @patch("nero_vla.dual_can.sample_can_traffic")
    def test_role_check_never_retries_confirmed_opposite_role(self, sample) -> None:
        sample.return_value = CanTraffic("can1", LEADER_JOINT_IDS, 40, 0.25)

        with self.assertRaises(CanRoleError):
            require_can_role("can1", "follower", recovery_timeout_sec=3.0)

        self.assertEqual(sample.call_count, 1)

    def test_bridge_command_round_trip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/bridge.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path)
            server.listen(1)

            def serve() -> None:
                connection, _ = server.accept()
                with connection:
                    self.assertEqual(connection.recv(128), b"status\n")
                    connection.sendall(b'{"ok":true,"paused":false}\n')
                server.close()

            thread = threading.Thread(target=serve)
            thread.start()
            try:
                self.assertEqual(
                    bridge_command("status", path), {"ok": True, "paused": False}
                )
            finally:
                thread.join(timeout=2)

    def test_bridge_command_rejects_unknown_command(self) -> None:
        with self.assertRaises(ValueError):
            bridge_command("invalid")
