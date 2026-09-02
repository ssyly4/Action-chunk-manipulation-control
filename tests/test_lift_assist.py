import unittest

import numpy as np

from nero_vla.lift_assist import PostGraspLiftAssist
from nero_vla.lift_assist import PostReleaseHeightGuard
from nero_vla.lift_assist import PreGraspDescentAssist
from nero_vla.lift_assist import bounded_pose_target
from nero_vla.lift_assist import damped_pose_step
from nero_vla.lift_assist import flange_pose
from nero_vla.lift_assist import flange_position_m
from nero_vla.lift_assist import rigid_contact_is_stable


GRASP_Q = np.deg2rad([6.0, 46.6, 10.3, 78.1, -2.6, 14.7, -13.6])


class DampedPositionStepTests(unittest.TestCase):
    def test_one_millimeter_up_step_preserves_xy(self):
        before = flange_position_m(GRASP_Q)
        goal = flange_pose(GRASP_Q)
        goal[2] += 0.001
        target = damped_pose_step(GRASP_Q, goal)
        after = flange_pose(target)

        self.assertAlmostEqual(float(after[2] - before[2]), 0.001, delta=2e-5)
        self.assertLess(float(np.linalg.norm(after[:2] - before[:2])), 2e-5)
        self.assertLess(float(np.linalg.norm(after[3:] - goal[3:])), np.deg2rad(0.02))
        self.assertLessEqual(float(np.max(np.abs(target - GRASP_Q))), np.deg2rad(0.5))

    def test_bounded_target_accumulates_small_ik_steps(self):
        goal = flange_pose(GRASP_Q)
        goal[2] += 0.06
        single = damped_pose_step(GRASP_Q, goal)
        accumulated = bounded_pose_target(GRASP_Q, goal, iterations=4)

        self.assertGreater(
            float(np.max(np.abs(accumulated - GRASP_Q))),
            float(np.max(np.abs(single - GRASP_Q))),
        )
        self.assertLessEqual(
            float(np.max(np.abs(accumulated - GRASP_Q))),
            np.deg2rad(2.0) + 1e-9,
        )

    def test_bounded_target_respects_joint_limits(self):
        goal = flange_pose(GRASP_Q)
        goal[2] += 0.06
        limits = np.column_stack((GRASP_Q - 0.01, GRASP_Q + 0.01))

        target = bounded_pose_target(
            GRASP_Q,
            goal,
            iterations=10,
            joint_limits_rad=limits,
        )

        margin = np.deg2rad(0.25)
        self.assertTrue(np.all(target >= limits[:, 0] + margin - 1e-12))
        self.assertTrue(np.all(target <= limits[:, 1] - margin + 1e-12))


class RigidContactTests(unittest.TestCase):
    def test_sustained_rigid_contact_requires_force_and_width_stall(self):
        self.assertTrue(
            rigid_contact_is_stable(
                measured_width_m=0.030,
                commanded_width_m=0.026,
                measured_force_n=-0.8,
                force_threshold_n=0.55,
                minimum_width_gap_m=0.0015,
            )
        )
        self.assertFalse(
            rigid_contact_is_stable(
                measured_width_m=0.027,
                commanded_width_m=0.0268,
                measured_force_n=-0.14,
                force_threshold_n=0.55,
                minimum_width_gap_m=0.0015,
            )
        )


class PreGraspDescentAssistTests(unittest.TestCase):
    def make_assist(self):
        return PreGraspDescentAssist(
            enabled=True,
            descent_distance_m=0.005,
            close_threshold=0.5,
            confirmations=3,
        )

    def test_requires_three_close_intent_samples_while_open(self):
        assist = self.make_assist()
        for tick in range(2):
            self.assertFalse(
                assist.observe(
                    now=tick / 30,
                    joint_rad=GRASP_Q,
                    measured_gripper_state=0.9,
                    policy_gripper_target=0.45,
                )
            )
        self.assertTrue(
            assist.observe(
                now=2 / 30,
                joint_rad=GRASP_Q,
                measured_gripper_state=0.9,
                policy_gripper_target=0.45,
            )
        )
        self.assertEqual(assist.state, "descending")
        self.assertAlmostEqual(assist.gripper_target(0.1), 0.9)

    def test_holds_low_pose_until_gripper_has_closed(self):
        assist = self.make_assist()
        for tick in range(3):
            assist.observe(
                now=tick / 30,
                joint_rad=GRASP_Q,
                measured_gripper_state=0.9,
                policy_gripper_target=0.45,
            )
        q = GRASP_Q.copy()
        initial_z = flange_position_m(q)[2]
        for tick in range(1, 30):
            target = assist.arm_target(now=2 / 30 + tick / 30, measured_joint_rad=q)
            if assist.state == "closing_hold":
                break
            self.assertIsNotNone(target)
            q = target
        self.assertEqual(assist.state, "closing_hold")
        self.assertAlmostEqual(initial_z - flange_position_m(q)[2], 0.005, delta=0.001)
        self.assertAlmostEqual(assist.gripper_target(0.2), 0.2)
        self.assertIsNotNone(
            assist.arm_target(now=0.8, measured_joint_rad=q)
        )

        assist.observe(
            now=0.81,
            joint_rad=q,
            measured_gripper_state=0.2,
            policy_gripper_target=0.1,
        )

        self.assertEqual(assist.state, "completed")
        self.assertIsNone(assist.arm_target(now=0.82, measured_joint_rad=q))

    def test_contact_stall_completes_after_bounded_hold(self):
        assist = self.make_assist()
        for tick in range(3):
            assist.observe(
                now=tick / 30,
                joint_rad=GRASP_Q,
                measured_gripper_state=0.9,
                policy_gripper_target=0.45,
            )
        q = GRASP_Q.copy()
        for tick in range(1, 30):
            target = assist.arm_target(now=2 / 30 + tick / 30, measured_joint_rad=q)
            q = target
            if assist.state == "closing_hold":
                break

        self.assertEqual(assist.state, "closing_hold")
        assist.observe(
            now=3.0,
            joint_rad=q,
            measured_gripper_state=0.4,
            policy_gripper_target=0.1,
        )
        self.assertIsNone(assist.arm_target(now=3.0, measured_joint_rad=q))
        self.assertEqual(assist.state, "completed")

    def test_descent_timeout_still_rejects_no_motion(self):
        assist = self.make_assist()
        for tick in range(3):
            assist.observe(
                now=tick / 30,
                joint_rad=GRASP_Q,
                measured_gripper_state=0.9,
                policy_gripper_target=0.45,
            )

        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            assist.arm_target(now=3.0, measured_joint_rad=GRASP_Q)

    def test_never_triggers_after_gripper_is_nearly_closed(self):
        assist = self.make_assist()
        for tick in range(10):
            self.assertFalse(
                assist.observe(
                    now=tick / 30,
                    joint_rad=GRASP_Q,
                    measured_gripper_state=0.2,
                    policy_gripper_target=0.2,
                )
            )
        self.assertEqual(assist.state, "idle")


class PostGraspLiftAssistTests(unittest.TestCase):
    def make_assist(self):
        return PostGraspLiftAssist(
            enabled=True,
            gripper_state_threshold=0.32,
            contact_force_threshold_n=0.55,
            contact_confirmations=3,
            settle_sec=0.25,
            lift_distance_m=0.05,
            lift_speed_m_s=0.02,
        )

    def observe(self, assist, now, force, measured=0.28, target=0.28):
        return assist.observe(
            now=now,
            joint_rad=GRASP_Q,
            measured_gripper_state=measured,
            policy_gripper_target=target,
            measured_force_n=force,
        )

    def test_empty_close_does_not_trigger(self):
        assist = self.make_assist()
        for tick in range(100):
            self.assertFalse(self.observe(assist, tick / 30.0, 0.46))
        self.assertEqual(assist.state, "idle")

    def test_three_force_confirmations_trigger_and_hold_gripper(self):
        assist = self.make_assist()
        self.assertFalse(self.observe(assist, 0.0, 0.60))
        self.assertFalse(self.observe(assist, 1 / 30, -0.70))
        self.assertTrue(self.observe(assist, 2 / 30, -0.80))

        self.assertEqual(assist.state, "settling")
        self.assertAlmostEqual(assist.gripper_target(0.9), 0.28)

    def test_lift_reaches_fifty_millimeters_and_holds_xy(self):
        assist = self.make_assist()
        for tick, force in enumerate([0.6, -0.7, -0.8]):
            self.observe(assist, tick / 30.0, force)
        q = GRASP_Q.copy()
        initial = flange_position_m(q)

        for tick in range(1, 121):
            now = 2 / 30 + tick / 30
            target = assist.arm_target(now=now, measured_joint_rad=q)
            self.assertIsNotNone(target)
            q = target

        final = flange_position_m(q)
        self.assertAlmostEqual(float(final[2] - initial[2]), 0.05, delta=0.001)
        self.assertLess(float(np.linalg.norm(final[:2] - initial[:2])), 0.001)
        self.assertEqual(assist.state, "holding")
        self.assertTrue(assist.reached_target(q))

    def test_target_is_not_reached_from_nominal_time_alone(self):
        assist = self.make_assist()
        for tick, force in enumerate([0.6, -0.7, -0.8]):
            self.observe(assist, tick / 30.0, force)

        assist.arm_target(now=4.0, measured_joint_rad=GRASP_Q)

        self.assertEqual(assist.state, "holding")
        self.assertFalse(assist.reached_target(GRASP_Q))


class PostReleaseHeightGuardTests(unittest.TestCase):
    def make_guard(self, *, floor, recovery):
        return PostReleaseHeightGuard(
            enabled=True,
            floor_height_m=floor,
            recovery_height_m=recovery,
            confirmations=2,
        )

    def cycle_gripper(self, guard, q):
        for tick in range(2):
            guard.observe(now=tick / 30, joint_rad=q, measured_gripper_state=0.1)
        self.assertEqual(guard.state, "waiting_for_open")
        self.assertFalse(
            guard.observe(now=2 / 30, joint_rad=q, measured_gripper_state=0.9)
        )
        return guard.observe(now=3 / 30, joint_rad=q, measured_gripper_state=0.9)

    def test_open_without_prior_close_does_not_activate(self):
        z = float(flange_position_m(GRASP_Q)[2])
        guard = self.make_guard(floor=z - 0.01, recovery=z + 0.005)
        for tick in range(10):
            self.assertFalse(
                guard.observe(
                    now=tick / 30,
                    joint_rad=GRASP_Q,
                    measured_gripper_state=0.9,
                )
            )
        self.assertEqual(guard.state, "waiting_for_close")

    def test_low_release_recovers_height_while_holding_xy(self):
        initial = flange_position_m(GRASP_Q)
        guard = self.make_guard(
            floor=float(initial[2] + 0.005),
            recovery=float(initial[2] + 0.020),
        )
        self.assertTrue(self.cycle_gripper(guard, GRASP_Q))
        self.assertEqual(guard.state, "recovering")

        q = GRASP_Q.copy()
        for tick in range(1, 60):
            target = guard.arm_target(
                now=3 / 30 + tick / 30,
                measured_joint_rad=q,
                policy_joint_rad=GRASP_Q,
            )
            self.assertIsNotNone(target)
            q = target
            if guard.state == "guarding":
                break

        final = flange_position_m(q)
        self.assertEqual(guard.state, "guarding")
        self.assertGreaterEqual(float(final[2]), float(initial[2] + 0.017))
        self.assertLess(float(np.linalg.norm(final[:2] - initial[:2])), 0.001)

        guarded = guard.arm_target(
            now=2.0,
            measured_joint_rad=q,
            policy_joint_rad=GRASP_Q,
        )
        self.assertTrue(guard.overriding_policy)
        self.assertGreaterEqual(
            float(flange_position_m(guarded)[2]),
            float(initial[2] + 0.017),
        )

    def test_release_between_floor_and_recovery_still_recovers_first(self):
        initial = flange_position_m(GRASP_Q)
        guard = self.make_guard(
            floor=float(initial[2] - 0.02),
            recovery=float(initial[2] + 0.01),
        )

        self.assertTrue(self.cycle_gripper(guard, GRASP_Q))

        self.assertEqual(guard.state, "recovering")

    def test_guarding_reenters_recovery_when_feedback_drops_below_height(self):
        initial = flange_position_m(GRASP_Q)
        guard = self.make_guard(
            floor=float(initial[2] - 0.03),
            recovery=float(initial[2] + 0.01),
        )
        self.assertTrue(self.cycle_gripper(guard, GRASP_Q))
        self.assertEqual(guard.state, "recovering")
        q = GRASP_Q.copy()
        for tick in range(1, 60):
            q = guard.arm_target(
                now=3 / 30 + tick / 30,
                measured_joint_rad=q,
                policy_joint_rad=GRASP_Q,
            )
            if guard.state == "guarding":
                break
        self.assertEqual(guard.state, "guarding")

        # A later low feedback sample must not remain in the sweeping state.
        target = guard.arm_target(
            now=1.0,
            measured_joint_rad=GRASP_Q,
            policy_joint_rad=GRASP_Q,
        )

        self.assertEqual(guard.state, "recovering")
        self.assertGreater(float(flange_position_m(target)[2]), float(initial[2]))

    def test_guarding_applies_a_ramped_negative_x_forward_extension(self):
        initial = flange_pose(GRASP_Q)
        guard = PostReleaseHeightGuard(
            enabled=True,
            floor_height_m=float(initial[2] - 0.02),
            recovery_height_m=float(initial[2] - 0.01),
            confirmations=2,
            forward_extension_m=0.015,
            extension_ramp_sec=0.5,
        )
        self.assertTrue(self.cycle_gripper(guard, GRASP_Q))
        self.assertEqual(guard.state, "guarding")

        target = guard.arm_target(
            now=1.0,
            measured_joint_rad=GRASP_Q,
            policy_joint_rad=GRASP_Q,
        )

        self.assertTrue(guard.overriding_policy)
        self.assertAlmostEqual(guard.status(GRASP_Q).forward_extension_m, 0.015)
        self.assertLess(float(flange_pose(target)[0]), float(initial[0] - 0.005))

    def test_guarding_can_lock_policy_height_while_retaining_xy(self):
        initial = flange_pose(GRASP_Q)
        guard = PostReleaseHeightGuard(
            enabled=True,
            floor_height_m=float(initial[2] - 0.02),
            recovery_height_m=float(initial[2] - 0.01),
            confirmations=2,
            lock_height=True,
        )
        self.assertTrue(self.cycle_gripper(guard, GRASP_Q))
        target = guard.arm_target(
            now=1.0,
            measured_joint_rad=GRASP_Q,
            policy_joint_rad=GRASP_Q,
        )
        target_pose = flange_pose(target)
        self.assertTrue(guard.overriding_policy)
        self.assertAlmostEqual(float(target_pose[2]), float(initial[2] - 0.01), places=3)
        self.assertLess(float(np.linalg.norm(target_pose[:2] - initial[:2])), 0.003)


if __name__ == "__main__":
    unittest.main()
