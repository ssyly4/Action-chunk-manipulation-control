import unittest

import numpy as np

from nero_vla.terminal_target_filter import TerminalArmTargetFilter


class TerminalArmTargetFilterTests(unittest.TestCase):
    def make_filter(self, **kwargs):
        return TerminalArmTargetFilter(
            enabled=True,
            approach_flange_height_m=0.16,
            approach_confirm_sec=0.15,
            approach_time_constant_sec=0.10,
            gripper_threshold=0.35,
            flange_height_m=0.22,
            confirm_sec=0.15,
            time_constant_sec=0.12,
            **kwargs,
        )

    def test_passes_targets_before_approach_conditions(self):
        target = np.full(7, 0.2)
        sample = self.make_filter().update(
            target,
            gripper_target=0.8,
            measured_flange_height_m=0.25,
            now=1.0,
        )
        self.assertEqual(sample.status, "inactive")
        self.assertFalse(sample.active)
        np.testing.assert_allclose(sample.target, target)

    def test_requires_continuous_confirmation_before_approach_filtering(self):
        filt = self.make_filter()
        first = filt.update(
            np.zeros(7),
            gripper_target=0.8,
            measured_flange_height_m=0.15,
            now=1.0,
        )
        second = filt.update(
            np.ones(7),
            gripper_target=0.8,
            measured_flange_height_m=0.15,
            now=1.10,
        )
        third = filt.update(
            np.full(7, 2.0),
            gripper_target=0.8,
            measured_flange_height_m=0.15,
            now=1.16,
        )

        self.assertEqual(first.status, "approach_arming")
        self.assertEqual(second.status, "approach_arming")
        self.assertEqual(third.status, "approach_filtering")
        self.assertTrue(third.active)
        self.assertEqual(third.stage, "approach")
        self.assertAlmostEqual(third.alpha, 0.06 / (0.10 + 0.06))
        np.testing.assert_allclose(
            third.target,
            np.ones(7) + third.alpha * np.ones(7),
        )

    def test_terminal_stage_replaces_approach_with_stronger_filter(self):
        filt = self.make_filter()
        filt.update(
            np.zeros(7),
            gripper_target=0.8,
            measured_flange_height_m=0.15,
            now=1.0,
        )
        filt.update(
            np.ones(7),
            gripper_target=0.8,
            measured_flange_height_m=0.15,
            now=1.2,
        )
        filt.update(
            np.full(7, 2.0),
            gripper_target=0.3,
            measured_flange_height_m=0.15,
            now=1.23,
        )
        sample = filt.update(
            np.full(7, 3.0),
            gripper_target=0.3,
            measured_flange_height_m=0.15,
            now=1.39,
        )
        self.assertTrue(sample.active)
        self.assertEqual(sample.status, "terminal_filtering")
        self.assertEqual(sample.stage, "terminal")
        self.assertAlmostEqual(sample.alpha, 0.16 / (0.12 + 0.16))

    def test_approach_stage_does_not_require_close_gripper_intent(self):
        filt = self.make_filter()
        filt.update(
            np.zeros(7),
            gripper_target=0.9,
            measured_flange_height_m=0.15,
            now=1.0,
        )
        sample = filt.update(
            np.ones(7),
            gripper_target=0.9,
            measured_flange_height_m=0.15,
            now=1.2,
        )
        self.assertEqual(sample.status, "approach_filtering")
        self.assertEqual(sample.stage, "approach")

    def test_bypass_returns_raw_assist_target_and_clears_filter(self):
        filt = self.make_filter()
        filt.update(
            np.zeros(7),
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.0,
        )
        filt.update(
            np.ones(7),
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.2,
        )
        assist = np.full(7, 3.0)
        sample = filt.update(
            assist,
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.23,
            bypass=True,
        )
        self.assertEqual(sample.status, "bypassed")
        self.assertFalse(sample.active)
        np.testing.assert_allclose(sample.target, assist)

    def test_deliberate_full_open_falls_back_to_approach_filter(self):
        filt = self.make_filter()
        filt.update(
            np.zeros(7),
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.0,
        )
        filt.update(
            np.ones(7),
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.2,
        )
        target = np.full(7, 2.0)
        sample = filt.update(
            target,
            gripper_target=0.95,
            measured_flange_height_m=0.2,
            now=1.23,
        )
        self.assertTrue(sample.active)
        self.assertEqual(sample.status, "approach_filtering")
        self.assertEqual(sample.stage, "approach")

    def test_missing_target_resets_state(self):
        filt = self.make_filter()
        sample = filt.update(
            None,
            gripper_target=0.3,
            measured_flange_height_m=0.2,
            now=1.0,
        )
        self.assertEqual(sample.status, "no_target")
        self.assertIsNone(sample.target)
        self.assertFalse(filt.active)


if __name__ == "__main__":
    unittest.main()
