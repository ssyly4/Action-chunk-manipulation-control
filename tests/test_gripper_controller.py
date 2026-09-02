import unittest

from nero_vla.gripper_controller import ConfirmedCloseLatch
from nero_vla.gripper_controller import ConfirmedGraspState
from nero_vla.gripper_controller import RateLimitedGripperFollower


class ConfirmedCloseLatchTests(unittest.TestCase):
    def test_requires_two_consecutive_close_chunks(self):
        latch = ConfirmedCloseLatch(close_threshold=0.30, confirmations=2)

        self.assertAlmostEqual(latch.update(0.80), 0.80)
        self.assertAlmostEqual(latch.update(0.24), 0.80)
        self.assertFalse(latch.latched)
        self.assertAlmostEqual(latch.update(0.64), 0.64)
        self.assertEqual(latch.close_count, 0)

        self.assertAlmostEqual(latch.update(0.25), 0.64)
        self.assertAlmostEqual(latch.update(0.23), 0.23)
        self.assertTrue(latch.latched)

    def test_latched_target_can_close_further_but_never_reopens(self):
        latch = ConfirmedCloseLatch(close_threshold=0.30, confirmations=2)
        latch.update(0.70)
        latch.update(0.26)
        latch.update(0.24)

        self.assertAlmostEqual(latch.update(0.65), 0.24)
        self.assertAlmostEqual(latch.update(0.20), 0.20)
        self.assertAlmostEqual(latch.update(0.90), 0.20)

    def test_same_policy_chunk_counts_only_once(self):
        latch = ConfirmedCloseLatch(close_threshold=0.30, confirmations=2)

        self.assertAlmostEqual(latch.update(0.70, confirmation_token=1), 0.70)
        self.assertAlmostEqual(latch.update(0.24, confirmation_token=1), 0.70)
        self.assertAlmostEqual(latch.update(0.23, confirmation_token=1), 0.70)
        self.assertFalse(latch.latched)
        self.assertEqual(latch.close_count, 1)

        self.assertAlmostEqual(latch.update(0.22, confirmation_token=2), 0.22)
        self.assertTrue(latch.latched)

    def test_rejects_invalid_configuration_and_targets(self):
        with self.assertRaises(ValueError):
            ConfirmedCloseLatch(close_threshold=1.1)
        with self.assertRaises(ValueError):
            ConfirmedCloseLatch(confirmations=0)

        latch = ConfirmedCloseLatch()
        with self.assertRaises(ValueError):
            latch.update(float("nan"))


class ConfirmedGraspStateTests(unittest.TestCase):
    def test_requires_latched_close_and_confirmed_contact(self):
        state = ConfirmedGraspState(grasp_state=0.23, contact_confirmations=3)

        state.update(close_latched=False, contact_detected=True)
        self.assertAlmostEqual(state.policy_state(0.41), 0.41)
        state.update(close_latched=True, contact_detected=True)
        state.update(close_latched=True, contact_detected=True)
        self.assertFalse(state.latched)
        self.assertAlmostEqual(state.policy_state(0.41), 0.41)
        state.update(close_latched=True, contact_detected=True)

        self.assertTrue(state.latched)
        self.assertAlmostEqual(state.policy_state(0.41), 0.23)
        self.assertAlmostEqual(state.policy_state(0.20), 0.20)

    def test_interrupted_contact_resets_confirmation(self):
        state = ConfirmedGraspState(contact_confirmations=2)
        state.update(close_latched=True, contact_detected=True)
        state.update(close_latched=True, contact_detected=False)
        state.update(close_latched=True, contact_detected=True)

        self.assertFalse(state.latched)
        self.assertEqual(state.contact_count, 1)

    def test_rejects_invalid_configuration_and_state(self):
        with self.assertRaises(ValueError):
            ConfirmedGraspState(grasp_state=1.1)
        with self.assertRaises(ValueError):
            ConfirmedGraspState(contact_confirmations=0)
        with self.assertRaises(ValueError):
            ConfirmedGraspState().policy_state(float("nan"))


class RateLimitedGripperFollowerTests(unittest.TestCase):
    def make_follower(self):
        follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=0.08,
            max_speed_m_s=0.03,
            contact_force_n=0.8,
            contact_confirmations=3,
            contact_window_ticks=5,
        )
        follower.initialize(0.06, now=0.0)
        return follower

    def test_intermittent_confirmed_contact_latches_physical_width(self):
        follower = self.make_follower()
        forces = [0.9, 0.2, 0.85, 0.1, 0.95]
        measured = [0.050, 0.049, 0.048, 0.047, 0.046]
        results = []
        for tick, (force, width) in enumerate(zip(forces, measured), start=1):
            results.append(
                follower.step(
                    0.20,
                    measured_width_m=width,
                    measured_force_n=force,
                    now=tick * 0.01,
                    contact_latch_enabled=True,
                )
            )

        self.assertTrue(follower.contact_latched)
        self.assertEqual(results[-1].status, "gripper_contact_preload")
        self.assertAlmostEqual(results[-1].width_m, 0.0477)

        held = results[-1]
        for tick in range(6, 13):
            held = follower.step(
                0.0,
                measured_width_m=0.046,
                measured_force_n=0.0,
                now=tick * 0.01,
                contact_latch_enabled=True,
            )
        self.assertEqual(held.status, "gripper_contact_hold")
        self.assertAlmostEqual(held.width_m, 0.046)

    def test_contact_does_not_latch_before_close_intent_is_confirmed(self):
        follower = self.make_follower()
        for tick in range(1, 7):
            follower.step(
                0.20,
                measured_width_m=0.05,
                measured_force_n=1.0,
                now=tick * 0.01,
                contact_latch_enabled=False,
            )
        self.assertFalse(follower.contact_latched)

    def test_force_hold_can_be_disabled_for_raw_policy_tracking(self):
        follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=0.08,
            max_speed_m_s=0.03,
            contact_force_n=0.8,
            force_hold_enabled=False,
        )
        follower.initialize(0.06, now=0.0)

        command = follower.step(
            0.0,
            measured_width_m=0.05,
            measured_force_n=2.0,
            now=0.1,
            contact_latch_enabled=False,
        )

        self.assertEqual(command.status, "gripper_tracking")
        self.assertAlmostEqual(command.width_m, 0.057)

    def test_configurable_open_end_feedback_tolerance_clamps_to_calibration(self):
        follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=0.08,
            max_speed_m_s=0.03,
            contact_force_n=0.8,
            feedback_tolerance_m=0.01,
        )

        initialized = follower.initialize(0.087, now=0.0)

        self.assertAlmostEqual(initialized.width_m, 0.08)
        with self.assertRaises(ValueError):
            follower.step(
                1.0,
                measured_width_m=0.091,
                measured_force_n=0.0,
                now=0.01,
            )

    def test_default_feedback_tolerance_remains_three_millimeters(self):
        follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=0.08,
            max_speed_m_s=0.03,
            contact_force_n=0.8,
        )

        with self.assertRaises(ValueError):
            follower.initialize(0.084, now=0.0)

    def test_confirmed_contact_applies_rate_limited_preload(self):
        follower = RateLimitedGripperFollower(
            closed_m=0.0,
            open_m=0.08,
            max_speed_m_s=0.03,
            contact_force_n=0.8,
            contact_preload_m=0.0015,
            contact_confirmations=1,
            contact_window_ticks=1,
        )
        follower.initialize(0.046, now=0.0)
        confirmed = follower.step(
            0.20,
            measured_width_m=0.046,
            measured_force_n=0.9,
            now=0.01,
            contact_latch_enabled=True,
        )
        self.assertEqual(confirmed.status, "gripper_contact_preload")
        self.assertAlmostEqual(confirmed.width_m, 0.046)

        preload = follower.step(
            0.20,
            measured_width_m=0.046,
            measured_force_n=0.4,
            now=0.02,
            contact_latch_enabled=True,
        )
        self.assertEqual(preload.status, "gripper_contact_preload")
        self.assertAlmostEqual(preload.width_m, 0.0457)

        result = preload
        for tick in range(3, 8):
            result = follower.step(
                0.20,
                measured_width_m=0.046,
                measured_force_n=0.4,
                now=tick * 0.01,
                contact_latch_enabled=True,
            )
        self.assertEqual(result.status, "gripper_contact_hold")
        self.assertAlmostEqual(result.width_m, 0.0445)


if __name__ == "__main__":
    unittest.main()
