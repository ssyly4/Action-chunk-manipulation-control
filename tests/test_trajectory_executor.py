import unittest

import numpy as np

from nero_vla.trajectory_executor import ActionChunkBuffer
from nero_vla.trajectory_executor import FeedbackProgressActionChunk
from nero_vla.trajectory_executor import RateLimitedJointFollower
from nero_vla.trajectory_executor import TrajectorySample
from nero_vla.trajectory_executor import align_action_chunk_to_state


class ChunkAlignmentTests(unittest.TestCase):
    @staticmethod
    def linear_actions(count=10):
        degrees = np.arange(count, dtype=np.float64)
        return np.deg2rad(degrees[:, None].repeat(7, axis=1))

    def test_projects_feedback_to_fractional_action_index(self):
        alignment = align_action_chunk_to_state(
            self.linear_actions(),
            feedback=np.deg2rad(np.full(7, 2.4)),
            command=np.deg2rad(np.full(7, 2.4)),
            latency_steps=6.0,
            max_alignment_error_rad=np.deg2rad(2.0),
            command_weight=0.0,
            latency_weight=0.0,
            direction_weight=0.0,
        )

        self.assertAlmostEqual(alignment.offset_steps, 2.4)
        np.testing.assert_allclose(np.rad2deg(alignment.action), [2.4] * 7)
        self.assertAlmostEqual(alignment.max_feedback_error_rad, 0.0, places=12)

    def test_search_is_bounded_by_latency_plus_margin(self):
        alignment = align_action_chunk_to_state(
            self.linear_actions(12),
            feedback=np.deg2rad(np.full(7, 8.0)),
            command=np.deg2rad(np.full(7, 8.0)),
            latency_steps=2.0,
            search_margin_steps=1.0,
            max_alignment_error_rad=np.deg2rad(10.0),
            command_weight=0.0,
            latency_weight=0.0,
            direction_weight=0.0,
        )

        self.assertAlmostEqual(alignment.search_max_steps, 3.0)
        self.assertAlmostEqual(alignment.offset_steps, 3.0)

    def test_search_keeps_one_future_action_in_horizon(self):
        alignment = align_action_chunk_to_state(
            self.linear_actions(12),
            feedback=np.deg2rad(np.full(7, 11.0)),
            command=np.deg2rad(np.full(7, 11.0)),
            latency_steps=20.0,
            max_alignment_error_rad=np.deg2rad(2.0),
            command_weight=0.0,
            latency_weight=0.0,
            direction_weight=0.0,
        )

        self.assertAlmostEqual(alignment.search_max_steps, 10.0)
        self.assertAlmostEqual(alignment.offset_steps, 10.0)

    def test_latency_is_only_a_weak_prior(self):
        alignment = align_action_chunk_to_state(
            self.linear_actions(),
            feedback=np.deg2rad(np.full(7, 2.4)),
            command=np.deg2rad(np.full(7, 2.4)),
            latency_steps=6.0,
            max_alignment_error_rad=np.deg2rad(2.0),
        )

        self.assertGreater(alignment.offset_steps, 2.3)
        self.assertLess(alignment.offset_steps, 2.6)

    def test_rejects_chunk_that_does_not_pass_near_feedback(self):
        with self.assertRaisesRegex(ValueError, "no safely aligned action"):
            align_action_chunk_to_state(
                self.linear_actions(3),
                feedback=np.deg2rad(np.full(7, 8.0)),
                command=np.deg2rad(np.full(7, 8.0)),
                latency_steps=2.0,
                max_alignment_error_rad=np.deg2rad(2.0),
            )


class ActionChunkBufferTests(unittest.TestCase):
    def test_interpolates_all_joint_targets_on_source_timeline(self):
        buffer = ActionChunkBuffer(action_hz=10.0, blend_duration_sec=0.0)
        actions = np.asarray([[0.0] * 7, [1.0] * 7, [2.0] * 7])
        buffer.push(actions, observed_at=0.0, received_at=0.04)
        sample = buffer.sample(0.15)
        self.assertEqual(sample.status, "tracking")
        np.testing.assert_allclose(sample.target, [0.5] * 7)

    def test_rejects_chunks_with_no_future_samples(self):
        buffer = ActionChunkBuffer(action_hz=10.0)
        with self.assertRaisesRegex(ValueError, "no future samples"):
            buffer.push(np.zeros((3, 7)), observed_at=0.0, received_at=0.31)

    def test_chunk_transition_starts_without_target_jump(self):
        buffer = ActionChunkBuffer(action_hz=10.0, blend_duration_sec=0.10)
        old = np.asarray([[0.0] * 7, [1.0] * 7, [2.0] * 7])
        buffer.push(old, observed_at=0.0, received_at=0.04)
        before = buffer.sample(0.15).target
        new = np.asarray([[10.0] * 7, [11.0] * 7, [12.0] * 7])
        buffer.push(new, observed_at=0.10, received_at=0.15)
        after = buffer.sample(0.15)
        self.assertEqual(after.status, "blending")
        np.testing.assert_allclose(after.target, before)

    def test_stale_chunk_requests_hold_instead_of_zero_position(self):
        buffer = ActionChunkBuffer(action_hz=10.0, stale_after_sec=0.05)
        actions = np.asarray([[1.0] * 7, [2.0] * 7])
        buffer.push(actions, observed_at=0.0, received_at=0.05)
        sample = buffer.sample(0.26)
        self.assertEqual(sample.status, "stale_chunk_hold")
        self.assertIsNone(sample.target)

    def test_scalar_action_timeline_interpolates_gripper_targets(self):
        buffer = ActionChunkBuffer(
            action_hz=10.0,
            action_dim=1,
            blend_duration_sec=0.0,
            first_action_offset_steps=0,
        )
        buffer.push(
            np.asarray([[0.8], [0.6], [0.2]]),
            observed_at=0.0,
            received_at=0.0,
        )

        self.assertAlmostEqual(float(buffer.sample(0.05).target[0]), 0.7)
        self.assertAlmostEqual(float(buffer.sample(0.15).target[0]), 0.4)

    def test_scalar_chunk_transition_blends_without_target_jump(self):
        buffer = ActionChunkBuffer(
            action_hz=10.0,
            action_dim=1,
            blend_duration_sec=0.10,
            first_action_offset_steps=0,
        )
        buffer.push(np.asarray([[0.8], [0.6], [0.4]]), observed_at=0.0, received_at=0.0)
        before = buffer.sample(0.05).target
        buffer.push(np.asarray([[0.2], [0.1], [0.0]]), observed_at=0.05, received_at=0.05)
        after = buffer.sample(0.05)

        self.assertEqual(after.status, "blending")
        np.testing.assert_allclose(after.target, before)

    def test_arm_and_gripper_share_fractional_alignment_phase(self):
        arm = ActionChunkBuffer(
            action_hz=10.0,
            blend_duration_sec=0.0,
            first_action_offset_steps=0,
        )
        gripper = ActionChunkBuffer(
            action_hz=10.0,
            action_dim=1,
            blend_duration_sec=0.0,
            first_action_offset_steps=0,
        )
        arm_actions = np.arange(4, dtype=np.float64)[:, None].repeat(7, axis=1)
        gripper_actions = np.asarray([[0.8], [0.6], [0.4], [0.2]])
        accepted_at = 10.0
        offset_steps = 2.5
        timeline_origin = accepted_at - offset_steps / 10.0

        arm.push(
            arm_actions,
            observed_at=timeline_origin,
            received_at=accepted_at,
        )
        gripper.push(
            gripper_actions,
            observed_at=timeline_origin,
            received_at=accepted_at,
        )

        np.testing.assert_allclose(arm.sample(accepted_at).target, [2.5] * 7)
        np.testing.assert_allclose(gripper.sample(accepted_at).target, [0.3])


class FeedbackProgressActionChunkTests(unittest.TestCase):
    @staticmethod
    def actions(count=5):
        arm = np.arange(count, dtype=np.float64)[:, None].repeat(7, axis=1)
        gripper = np.linspace(0.8, 0.0, count)[:, None]
        return np.concatenate([arm, gripper], axis=1)

    def make_buffer(self, **kwargs):
        return FeedbackProgressActionChunk(
            action_hz=10.0,
            arm_lead_steps=1.0,
            max_progress_steps_per_tick=1.0,
            blend_duration_sec=0.0,
            stale_after_sec=0.1,
            **kwargs,
        )

    def test_arm_lead_moves_without_advancing_gripper_phase(self):
        buffer = self.make_buffer()
        buffer.push(
            self.actions(),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )

        sample = buffer.sample(0.01, arm_feedback=np.zeros(7))

        self.assertAlmostEqual(sample.phase_steps, 0.0)
        self.assertAlmostEqual(sample.arm_target_phase_steps, 1.0)
        np.testing.assert_allclose(sample.arm_target, np.ones(7))
        self.assertAlmostEqual(sample.gripper_target, 0.8)

    def test_gripper_lead_reads_ahead_without_advancing_shared_phase(self):
        buffer = self.make_buffer(gripper_lead_steps=1.5)
        buffer.push(
            self.actions(),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )

        sample = buffer.sample(0.01, arm_feedback=np.zeros(7))

        self.assertAlmostEqual(sample.phase_steps, 0.0)
        self.assertAlmostEqual(sample.arm_target_phase_steps, 1.0)
        np.testing.assert_allclose(sample.arm_target, np.ones(7))
        self.assertAlmostEqual(sample.gripper_target, 0.5)

    def test_feedback_advances_arm_and_gripper_on_one_shared_phase(self):
        buffer = self.make_buffer()
        buffer.push(
            self.actions(),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )

        sample = buffer.sample(0.01, arm_feedback=np.full(7, 0.5))

        self.assertAlmostEqual(sample.phase_steps, 0.5)
        self.assertAlmostEqual(sample.arm_target_phase_steps, 1.5)
        np.testing.assert_allclose(sample.arm_target, np.full(7, 1.5))
        self.assertAlmostEqual(sample.gripper_target, 0.7)

    def test_phase_never_moves_backwards_when_feedback_regresses(self):
        buffer = self.make_buffer()
        buffer.push(
            self.actions(),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )
        first = buffer.sample(0.01, arm_feedback=np.ones(7))
        second = buffer.sample(0.02, arm_feedback=np.full(7, 0.2))

        self.assertAlmostEqual(first.phase_steps, 1.0)
        self.assertAlmostEqual(second.phase_steps, 1.0)
        self.assertAlmostEqual(second.gripper_target, 0.6)

    def test_progress_per_tick_is_bounded(self):
        buffer = FeedbackProgressActionChunk(
            action_hz=10.0,
            arm_lead_steps=1.0,
            max_progress_steps_per_tick=0.5,
            blend_duration_sec=0.0,
            stale_after_sec=0.1,
        )
        buffer.push(
            self.actions(),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )

        sample = buffer.sample(0.01, arm_feedback=np.full(7, 4.0))

        self.assertAlmostEqual(sample.phase_steps, 0.5)

    def test_new_chunk_blends_from_last_shared_targets(self):
        buffer = FeedbackProgressActionChunk(
            action_hz=10.0,
            arm_lead_steps=1.0,
            max_progress_steps_per_tick=1.0,
            blend_duration_sec=0.1,
            stale_after_sec=0.1,
        )
        first_actions = self.actions()
        buffer.push(
            first_actions,
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )
        before = buffer.sample(0.11, arm_feedback=np.zeros(7))
        second_actions = first_actions + np.asarray([10.0] * 7 + [0.1])
        buffer.push(
            second_actions,
            initial_phase_steps=0.0,
            observed_at=0.11,
            received_at=0.11,
        )
        after = buffer.sample(0.11, arm_feedback=np.full(7, 10.0))

        self.assertEqual(after.status, "blending")
        np.testing.assert_allclose(after.arm_target, before.arm_target)
        self.assertAlmostEqual(after.gripper_target, before.gripper_target)

    def test_stale_chunk_holds_both_arm_and_gripper(self):
        buffer = self.make_buffer()
        buffer.push(
            self.actions(2),
            initial_phase_steps=0.0,
            observed_at=0.0,
            received_at=0.0,
        )

        sample = buffer.sample(0.21, arm_feedback=np.zeros(7))

        self.assertEqual(sample.status, "stale_chunk_hold")
        self.assertIsNone(sample.arm_target)
        self.assertIsNone(sample.gripper_target)


class RateLimitedJointFollowerTests(unittest.TestCase):
    def make_follower(self):
        return RateLimitedJointFollower(
            max_velocity_rad_s=0.2,
            max_acceleration_rad_s2=1.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=0.5,
            max_tick_interval_sec=0.05,
        )

    def test_velocity_and_acceleration_are_bounded(self):
        follower = self.make_follower()
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)
        velocities = [np.zeros(7)]
        commands = [feedback.copy()]
        sample = TrajectorySample(np.ones(7), "tracking", 0.0, 1.0)
        for tick in range(1, 101):
            result = follower.step(sample, measured=feedback, now=tick * 0.01)
            feedback = result.command.copy()
            commands.append(result.command)
            velocities.append(result.velocity)
        velocity = np.stack(velocities)
        acceleration = np.diff(velocity, axis=0) / 0.01
        self.assertLessEqual(float(np.max(np.abs(velocity))), 0.2 + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 1.0 + 1e-9)
        self.assertTrue(np.all(np.diff(np.stack(commands)[:, 0]) >= -1e-12))

    def test_missing_target_holds_last_nonzero_command(self):
        follower = self.make_follower()
        initial = np.asarray([0.4, -0.3, 0.2, -0.1, 0.5, -0.6, 0.7])
        follower.initialize(initial, now=0.0)
        moving = follower.step(
            TrajectorySample(initial + 0.1, "tracking", 0.0, 1.0),
            measured=initial,
            now=0.01,
        )
        held = follower.step(
            TrajectorySample(None, "stale_chunk_hold", 1.0, -0.1),
            measured=moving.command,
            now=0.02,
        )
        self.assertEqual(held.status, "stale_chunk_hold")
        np.testing.assert_allclose(held.command, moving.command)
        self.assertFalse(np.allclose(held.command, np.zeros(7)))

    def test_scheduler_gap_and_far_target_hold(self):
        follower = self.make_follower()
        initial = np.full(7, 0.25)
        follower.initialize(initial, now=0.0)
        far = TrajectorySample(np.full(7, 3.0), "tracking", 0.0, 1.0)
        rejected = follower.step(far, measured=initial, now=0.01)
        self.assertEqual(rejected.status, "target_feedback_rejected")
        np.testing.assert_allclose(rejected.command, initial)
        gap = follower.step(
            TrajectorySample(initial + 0.1, "tracking", 0.0, 1.0),
            measured=initial,
            now=0.20,
        )
        self.assertEqual(gap.status, "scheduler_gap_hold")
        np.testing.assert_allclose(gap.command, initial)

    def test_feedback_lag_is_governed_before_hard_guard(self):
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=0.2,
            max_acceleration_rad_s2=1.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=0.05,
            max_tick_interval_sec=0.05,
        )
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)
        sample = TrajectorySample(np.ones(7), "tracking", 0.0, 1.0)
        velocities = [np.zeros(7)]
        results = []

        # Simulate a plant with enough lag to engage the feedback governor.
        for tick in range(1, 301):
            result = follower.step(sample, measured=feedback, now=tick * 0.01)
            results.append(result)
            velocities.append(result.velocity)
            feedback += 0.05 * (result.command - feedback)

        self.assertNotIn("command_feedback_guard_hold", [result.status for result in results])
        self.assertTrue(any(result.feedback_governed for result in results))
        self.assertLessEqual(
            float(np.max(np.abs(results[-1].command - feedback))),
            0.05 + 1e-12,
        )
        acceleration = np.diff(np.stack(velocities), axis=0) / 0.01
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 1.0 + 1e-9)

    def test_soft_feedback_limit_must_not_exceed_hard_limit(self):
        with self.assertRaisesRegex(ValueError, "governor"):
            RateLimitedJointFollower(
                max_velocity_rad_s=0.2,
                max_acceleration_rad_s2=1.0,
                max_target_feedback_error_rad=2.0,
                max_command_feedback_error_rad=0.05,
                feedback_governor_error_rad=0.06,
            )

    def test_discrete_integration_stops_at_joint_limit_without_overshoot(self):
        limits = np.asarray([[-1.0, 1.0]] * 7)
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=1.0,
            max_tick_interval_sec=0.10,
            joint_limits_rad=limits,
        )
        feedback = np.zeros(7)
        feedback[5] = -0.95
        follower.initialize(feedback, now=0.0)
        target = feedback.copy()
        target[5] = -1.0
        sample = TrajectorySample(target, "tracking", 0.0, 1.0)

        first = follower.step(sample, measured=feedback, now=0.05)
        second = follower.step(sample, measured=first.command, now=0.10)

        self.assertGreaterEqual(second.command[5], limits[5, 0])
        self.assertAlmostEqual(second.command[5], limits[5, 0])
        self.assertAlmostEqual(second.velocity[5], 0.0)
        self.assertTrue(second.joint_limit_clamped)
        self.assertEqual(second.status, "tracking")

    def test_discrete_integration_stops_at_target_without_reversing(self):
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=1.0,
            max_tick_interval_sec=0.10,
        )
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)
        accelerating = follower.step(
            TrajectorySample(np.ones(7), "tracking", 0.0, 1.0),
            measured=feedback,
            now=0.05,
        )
        nearby_target = np.full(7, 0.03)

        stopped = follower.step(
            TrajectorySample(nearby_target, "tracking", 0.0, 1.0),
            measured=accelerating.command,
            now=0.10,
        )

        np.testing.assert_allclose(stopped.command, nearby_target)
        np.testing.assert_allclose(stopped.velocity, np.zeros(7))

    def test_streaming_mode_does_not_stop_at_every_trajectory_sample(self):
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=0.2,
            max_acceleration_rad_s2=1.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=0.5,
            max_tick_interval_sec=0.05,
            tracking_mode="streaming_trajectory",
        )
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)
        results = []
        for tick in range(1, 31):
            target = np.full(7, 0.001 * tick)
            result = follower.step(
                TrajectorySample(target, "rtc_tracking", 0.0, 1.0),
                measured=feedback,
                now=tick * 0.01,
            )
            feedback = result.command.copy()
            results.append(result)

        velocities = np.stack([result.velocity for result in results])
        acceleration = np.diff(velocities, axis=0) / 0.01
        self.assertGreater(float(np.median(velocities[-10:, 0])), 0.05)
        self.assertLessEqual(float(np.max(np.abs(velocities))), 0.2 + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 1.0 + 1e-9)

    def test_streaming_mode_uses_explicit_target_velocity(self):
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=0.5,
            max_acceleration_rad_s2=20.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=1.0,
            max_tick_interval_sec=0.05,
            tracking_mode="streaming_trajectory",
        )
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)

        result = follower.step(
            TrajectorySample(
                feedback.copy(),
                "rtc_tracking",
                0.0,
                1.0,
                np.full(7, 0.1),
            ),
            measured=feedback,
            now=0.01,
        )

        np.testing.assert_allclose(result.velocity, np.full(7, 0.1))

    def test_streaming_mode_limits_jerk_during_velocity_reversal(self):
        follower = RateLimitedJointFollower(
            max_velocity_rad_s=0.5,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=20.0,
            max_target_feedback_error_rad=2.0,
            max_command_feedback_error_rad=1.0,
            max_tick_interval_sec=0.05,
            tracking_mode="streaming_trajectory",
        )
        feedback = np.zeros(7)
        follower.initialize(feedback, now=0.0)
        velocities = [np.zeros(7)]
        for tick in range(1, 41):
            direction = 1.0 if tick < 16 else -1.0
            target = np.full(7, direction * 0.3)
            result = follower.step(
                TrajectorySample(target, "rtc_tracking", 0.0, 1.0),
                measured=feedback,
                now=tick * 0.01,
            )
            feedback = result.command.copy()
            velocities.append(result.velocity)

        acceleration = np.diff(np.stack(velocities), axis=0) / 0.01
        jerk = np.diff(acceleration, axis=0) / 0.01
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 2.0 + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(jerk))), 20.0 + 1e-7)


if __name__ == "__main__":
    unittest.main()
