import ast
import unittest
from pathlib import Path

import numpy as np

from nero_vla.policy_client import Packer
from nero_vla.policy_client import unpackb
from nero_vla.policy_probe import make_test_images
from nero_vla.policy_probe import normalize_gripper
from nero_vla.policy_probe import distribution


class PolicyProbeTests(unittest.TestCase):
    def test_test_images_match_policy_contract(self):
        exterior, wrist = make_test_images()
        for image in (exterior, wrist):
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(image.flags.c_contiguous)

    def test_gripper_droid_mapping(self):
        self.assertEqual(normalize_gripper(0.07, "width", 0.07), 0.0)
        self.assertEqual(normalize_gripper(0.0, "width", 0.07), 1.0)
        self.assertAlmostEqual(normalize_gripper(0.035, "width", 0.07), 0.5)

    def test_msgpack_numpy_round_trip(self):
        value = {"image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3)}
        restored = unpackb(Packer().pack(value))
        np.testing.assert_array_equal(restored["image"], value["image"])

    def test_distribution(self):
        stats = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)

    def test_probe_contains_no_robot_control_calls(self):
        source = Path(__file__).parents[1] / "nero_vla" / "policy_probe.py"
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
