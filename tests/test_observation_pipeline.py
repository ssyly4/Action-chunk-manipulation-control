import ast
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from nero_vla.observation_pipeline import SignalBuffer
from nero_vla.observation_pipeline import TimedValue
from nero_vla.observation_pipeline import interpolate


class ObservationPipelineTests(unittest.TestCase):
    def test_linear_interpolation(self):
        buffer = SignalBuffer()
        buffer.append(TimedValue(1_000_000_000, np.asarray([0.0, 2.0])))
        buffer.append(TimedValue(1_020_000_000, np.asarray([2.0, 4.0])))
        value, timing = interpolate(buffer, 1_005_000_000, 0.01)
        np.testing.assert_allclose(value, [0.5, 2.5])
        self.assertAlmostEqual(timing["alpha"], 0.25)
        self.assertAlmostEqual(timing["before_delta_ms"], 5.0)
        self.assertAlmostEqual(timing["after_delta_ms"], 15.0)

    def test_waits_for_future_sample(self):
        buffer = SignalBuffer()
        buffer.append(TimedValue(1_000_000_000, np.asarray([0.0])))

        def append_later():
            time.sleep(0.01)
            buffer.append(TimedValue(1_020_000_000, np.asarray([2.0])))

        thread = threading.Thread(target=append_later)
        thread.start()
        value, _ = interpolate(buffer, 1_010_000_000, 0.1)
        thread.join()
        np.testing.assert_allclose(value, [1.0])

    def test_duplicate_and_backward_timestamps_are_rejected(self):
        buffer = SignalBuffer()
        buffer.append(TimedValue(10, np.asarray([1.0])))
        buffer.append(TimedValue(10, np.asarray([2.0])))
        buffer.append(TimedValue(9, np.asarray([3.0])))
        self.assertEqual(buffer.duplicate_count, 1)
        self.assertEqual(buffer.backward_count, 1)

    def test_pipeline_contains_no_robot_control_calls(self):
        source = Path(__file__).parents[1] / "nero_vla" / "observation_pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        forbidden = {
            "enable", "disable", "reset", "electronic_emergency_stop",
            "set_motion_mode", "move_j", "move_p", "move_l", "move_c",
            "move_js", "move_mit", "move_cpv_pos", "move_cpv_vel",
            "move_gripper_m", "move_gripper_deg",
        }
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(calls & forbidden)


if __name__ == "__main__":
    unittest.main()
