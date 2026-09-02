import unittest

import numpy as np

from nero_vla.cpv_backend import NeroCpvPositionBackend


class FakeRobot:
    def __init__(self):
        self.calls = []

    def set_auto_set_motion_mode_enabled(self, enabled):
        self.calls.append(("auto_mode", enabled))

    def set_motion_mode(self, mode):
        self.calls.append(("motion_mode", mode))

    def move_cpv_pos(self, joint_index, position):
        self.calls.append(("cpv_pos", joint_index, position))


class NeroCpvPositionBackendTests(unittest.TestCase):
    def test_prepare_preloads_then_switches_mode_exactly_once(self):
        robot = FakeRobot()
        backend = NeroCpvPositionBackend(robot, max_command_step_rad=0.01)
        current = np.linspace(-0.3, 0.3, 7)
        backend.prepare_hold(current)

        self.assertEqual(robot.calls[0], ("auto_mode", False))
        mode_calls = [call for call in robot.calls if call[0] == "motion_mode"]
        position_calls = [call for call in robot.calls if call[0] == "cpv_pos"]
        self.assertEqual(mode_calls, [("motion_mode", "cpv")])
        self.assertEqual(len(position_calls), 14)
        np.testing.assert_allclose([call[2] for call in position_calls[:7]], current)
        np.testing.assert_allclose([call[2] for call in position_calls[7:]], current)

    def test_stream_writes_positions_without_more_mode_changes(self):
        robot = FakeRobot()
        backend = NeroCpvPositionBackend(robot, max_command_step_rad=0.01)
        current = np.linspace(-0.3, 0.3, 7)
        backend.prepare_hold(current)
        robot.calls.clear()

        backend.send(current + 0.005)
        backend.hold()
        self.assertEqual(len(robot.calls), 14)
        self.assertTrue(all(call[0] == "cpv_pos" for call in robot.calls))

    def test_unprepared_nonfinite_and_large_steps_are_rejected(self):
        robot = FakeRobot()
        backend = NeroCpvPositionBackend(robot, max_command_step_rad=0.01)
        with self.assertRaisesRegex(RuntimeError, "prepare_hold"):
            backend.send(np.zeros(7))
        with self.assertRaisesRegex(ValueError, "finite"):
            backend.prepare_hold(np.asarray([0, 0, 0, 0, 0, 0, np.nan]))

        backend = NeroCpvPositionBackend(robot, max_command_step_rad=0.01)
        backend.prepare_hold(np.zeros(7))
        call_count = len(robot.calls)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            backend.send(np.full(7, 0.02))
        self.assertEqual(len(robot.calls), call_count)


if __name__ == "__main__":
    unittest.main()
