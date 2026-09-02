import ast
import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nero_vla.state_monitor import advance_deadline
from nero_vla.state_monitor import build_snapshot
from nero_vla.state_monitor import firmware_driver


class Feedback:
    def __init__(self, msg, *, timestamp, hz):
        self.msg = msg
        self.timestamp = timestamp
        self.hz = hz


def driver_flags(**overrides):
    values = {
        "voltage_too_low": False,
        "motor_overheating": False,
        "driver_overcurrent": False,
        "driver_overheating": False,
        "collision_status": False,
        "driver_error_status": False,
        "driver_enable_status": True,
        "stall_status": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def gripper_flags(**overrides):
    values = {
        "voltage_too_low": False,
        "motor_overheating": False,
        "driver_overcurrent": False,
        "driver_overheating": False,
        "sensor_status": False,
        "driver_error_status": False,
        "driver_enable_status": True,
        "homing_status": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRobot:
    def __init__(self, *, timestamp, driver_fault=False):
        self.timestamp = timestamp
        self.driver_fault = driver_fault
        self.disconnected = False

    def get_arm_status(self):
        return Feedback(
            SimpleNamespace(
                ctrl_mode=1,
                arm_status=0,
                mode_feedback=1,
                motion_status=0,
            ),
            timestamp=self.timestamp,
            hz=200.0,
        )

    def get_joint_angles(self):
        return Feedback([0.1] * 7, timestamp=self.timestamp, hz=200.0)

    def get_flange_pose(self):
        return Feedback([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], timestamp=self.timestamp, hz=50.0)

    def get_motor_states(self, index):
        return Feedback(
            SimpleNamespace(
                position=0.1 * index,
                velocity=0.01 * index,
                current=0.2,
                torque=0.3,
            ),
            timestamp=self.timestamp,
            hz=200.0,
        )

    def get_driver_states(self, index):
        return Feedback(
            SimpleNamespace(
                vol=23.8,
                foc_temp=40.0,
                motor_temp=34.0,
                bus_current=0.2,
                foc_status=driver_flags(
                    driver_overcurrent=self.driver_fault and index == 3
                ),
            ),
            timestamp=self.timestamp,
            hz=50.0,
        )

    def has_comm_error(self):
        return False

    def get_comm_error(self):
        return None


class FakeGripper:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def get_gripper_status(self):
        return Feedback(
            SimpleNamespace(
                mode="width",
                value=0.05,
                force=1.0,
                foc_status=gripper_flags(),
            ),
            timestamp=self.timestamp,
            hz=50.0,
        )


class StateMonitorTests(unittest.TestCase):
    def make_snapshot(self, *, age_sec=0.01, driver_fault=False):
        wall_ns = time.time_ns()
        timestamp = wall_ns / 1e9 - age_sec
        return build_snapshot(
            FakeRobot(timestamp=timestamp, driver_fault=driver_fault),
            FakeGripper(timestamp),
            firmware="1.20",
            seq=3,
            wall_time_ns=wall_ns,
            monotonic_time_ns=2_000_000_000,
            scheduled_time_ns=1_999_000_000,
            period_ns=20_000_000,
            missed_deadlines=2,
        )

    def test_snapshot_schema_and_units(self):
        snapshot = self.make_snapshot()
        self.assertEqual(snapshot["schema_version"], "1.0")
        self.assertEqual(snapshot["seq"], 3)
        self.assertEqual(snapshot["joints"]["position_rad"], [0.1] * 7)
        self.assertAlmostEqual(snapshot["motors"][6]["velocity_rad_s"], 0.07)
        self.assertEqual(snapshot["flange_pose_m_rad"][:3], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(snapshot["gripper"]["value"], 0.05)
        self.assertEqual(snapshot["health"]["missed_deadlines"], 2)
        self.assertTrue(snapshot["health"]["ok"])
        json.dumps(snapshot, allow_nan=False)

    def test_stale_feedback_marks_snapshot_unhealthy(self):
        snapshot = self.make_snapshot(age_sec=0.5)
        self.assertFalse(snapshot["health"]["ok"])
        self.assertIn("joints", snapshot["health"]["stale_sources"])
        self.assertIn("flange", snapshot["health"]["stale_sources"])

    def test_driver_fault_is_reported(self):
        snapshot = self.make_snapshot(driver_fault=True)
        self.assertFalse(snapshot["health"]["ok"])
        self.assertIn("driver_3:driver_overcurrent", snapshot["health"]["faults"])

    def test_deadline_advance_does_not_accumulate_drift(self):
        next_deadline, skipped = advance_deadline(
            now_ns=1_045_000_000,
            deadline_ns=1_000_000_000,
            period_ns=20_000_000,
        )
        self.assertEqual(next_deadline, 1_060_000_000)
        self.assertEqual(skipped, 2)

    def test_firmware_selection(self):
        self.assertEqual(firmware_driver("1.20"), "v120")
        self.assertEqual(firmware_driver("1.12"), "v112")
        self.assertEqual(firmware_driver("1.11"), "v111")
        self.assertEqual(firmware_driver("1.10"), "default")

    def test_monitor_source_contains_no_control_calls(self):
        source_path = Path(__file__).parents[1] / "nero_vla" / "state_monitor.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {
            "enable",
            "disable",
            "reset",
            "electronic_emergency_stop",
            "set_motion_mode",
            "move_j",
            "move_p",
            "move_l",
            "move_c",
            "move_js",
            "move_mit",
            "move_cpv_pos",
            "move_cpv_vel",
            "move_gripper_m",
            "move_gripper_deg",
        }
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(calls & forbidden)


if __name__ == "__main__":
    unittest.main()
