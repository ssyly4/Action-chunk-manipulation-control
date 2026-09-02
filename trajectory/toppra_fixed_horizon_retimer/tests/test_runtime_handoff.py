from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import replace

import numpy as np

from receding_toppra_queue import (
    ARM_COLUMNS,
    QuinticHandoffCorrection,
    RecedingToppraRtcQueue,
    scale_actions_to_envelope,
)
from follower_state_bridge import (
    FOLLOWER_STATE_REGISTRY,
    make_bridged_follower_class,
)


def make_chunk(
    base: np.ndarray | None = None, *, slope: float = 0.0001, steps: int = 24
) -> np.ndarray:
    values = np.zeros((steps, 16), dtype=np.float64)
    phase = np.arange(steps, dtype=np.float64)
    origin = np.zeros(14, dtype=np.float64) if base is None else np.asarray(base)
    for joint, column in enumerate(ARM_COLUMNS):
        values[:, column] = origin[joint] + slope * (joint + 1) * phase
    values[:, 7] = 1.0
    values[:, 15] = 1.0
    return values


class RuntimeHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        handle, path = tempfile.mkstemp(prefix="toppra-runtime-", suffix=".jsonl")
        os.close(handle)
        self.log_path = path
        os.environ["NERO_TOPPRA_RUNTIME_LOG"] = path
        os.environ.pop("NERO_TOPPRA_BOUNDARY_PATH_MODE", None)
        os.environ.pop("NERO_TOPPRA_CURVATURE_SPEED_MARGIN", None)
        os.environ.pop("NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF", None)
        os.environ.pop("NERO_TOPPRA_REPLAN_RESERVE_TICKS", None)
        os.environ.pop("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", None)
        FOLLOWER_STATE_REGISTRY.reset()

    def tearDown(self) -> None:
        os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
        os.environ.pop("NERO_TOPPRA_BOUNDARY_PATH_MODE", None)
        os.environ.pop("NERO_TOPPRA_CURVATURE_SPEED_MARGIN", None)
        os.environ.pop("NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF", None)
        os.environ.pop("NERO_TOPPRA_REPLAN_RESERVE_TICKS", None)
        os.environ.pop("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", None)
        try:
            os.unlink(self.log_path)
        except FileNotFoundError:
            pass

    def test_blend_starts_at_measured_position(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            for tick in range(1, 9):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)

            replacement = make_chunk(feedback + 0.01)
            queue.load(replacement)
            queue._retime_future.result(timeout=2.0)
            sample = queue.sample(9 / 30.0, feedback=feedback)
            self.assertIsNotNone(queue._blend)
            self.assertGreaterEqual(queue._blend.ticks, 2)
            self.assertLessEqual(queue._blend.ticks, 12)
            command = np.concatenate((sample.left_target, sample.right_target))
            np.testing.assert_allclose(command, feedback, atol=1e-10)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
        finally:
            queue.close()

    def test_safe_chunk_recovers_after_old_horizon_expires(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            for tick in range(1, 30):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)
            self.assertEqual(sample.status, "rtc_queue_hold")

            queue.load(make_chunk(feedback))
            queue._retime_future.result(timeout=2.0)
            sample = queue.sample(1.0, feedback=feedback)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
            self.assertEqual(queue.emitted_steps, 25)
        finally:
            queue.close()

    def test_hard_mismatch_rejects_and_keeps_old_reserve(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            old_plan = queue._active
            feedback = np.concatenate((sample.left_target, sample.right_target))

            queue.load(make_chunk(feedback + 0.1))
            queue._retime_future.result(timeout=2.0)
            sample = queue.sample(1 / 30.0, feedback=feedback)
            self.assertIs(queue._active, old_plan)
            self.assertIsNotNone(queue._ready)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
            with open(self.log_path, encoding="utf-8") as stream:
                self.assertIn("rejected_hard_measured_mismatch", stream.read())
        finally:
            queue.close()

    def test_retiming_does_not_block_old_reserve_sampling(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            old_plan = queue._active
            original_plan = queue._engine.plan

            def slow_plan(*args, **kwargs):
                time.sleep(0.05)
                return original_plan(*args, **kwargs)

            queue._engine.plan = slow_plan
            queue.load(make_chunk())
            self.assertFalse(queue._retime_future.done())
            feedback = np.concatenate((sample.left_target, sample.right_target))
            sample = queue.sample(1 / 30.0, feedback=feedback)
            self.assertIs(queue._active, old_plan)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
            queue._retime_future.result(timeout=2.0)
            queue.sample(2 / 30.0, feedback=feedback)
            self.assertIsNot(queue._active, old_plan)
        finally:
            queue.close()

    def test_minimum_commit_delays_takeover_and_blocks_another_request(self) -> None:
        os.environ["NERO_TOPPRA_MIN_COMMIT_TICKS"] = "12"
        os.environ["NERO_TOPPRA_PENDING_REQUEST_BLOCK_STEPS"] = "23"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            old_plan = queue._active

            queue.load(make_chunk())
            queue._retime_future.result(timeout=2.0)
            self.assertEqual(queue.remaining_steps, 23)

            for tick in range(1, 12):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)
                self.assertIs(queue._active, old_plan)
                self.assertIsNotNone(queue._ready)

            feedback = np.concatenate((sample.left_target, sample.right_target))
            sample = queue.sample(12 / 30.0, feedback=feedback)
            self.assertIsNot(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            self.assertEqual(queue._active_takeover_tick, 12)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_MIN_COMMIT_TICKS", None)
            os.environ.pop("NERO_TOPPRA_PENDING_REQUEST_BLOCK_STEPS", None)

    def test_minimum_commit_does_not_hold_after_active_reserve_exhausts(self) -> None:
        os.environ["NERO_TOPPRA_MIN_COMMIT_TICKS"] = "10"
        os.environ["NERO_TOPPRA_EMERGENCY_RESERVE_TICKS"] = "0"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk(steps=4))
            sample = queue.sample(0.0, feedback=feedback)
            old_plan = queue._active

            # A replacement becomes ready before the short active plan has
            # reached the configured 10-tick churn guard.
            queue.load(make_chunk())
            queue._retime_future.result(timeout=2.0)
            for tick in range(1, 5):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)

            self.assertIsNot(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_MIN_COMMIT_TICKS", None)
            os.environ.pop("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", None)

    def test_ready_chunk_can_replace_an_unfinished_blend(self) -> None:
        os.environ["NERO_TOPPRA_MIN_COMMIT_TICKS"] = "2"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            old_plan = queue._active
            queue._blend = QuinticHandoffCorrection(
                takeover_wall_tick=0,
                ticks=12,
                action_hz=30.0,
                position_offset=np.full(14, 0.002),
                velocity_offset=np.full(14, 0.01),
            )

            queue.load(make_chunk())
            queue._retime_future.result(timeout=2.0)
            for tick in range(1, 3):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)

            self.assertIsNot(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            self.assertEqual(queue._active_takeover_tick, 2)
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_MIN_COMMIT_TICKS", None)

    def test_low_reserve_uses_follower_handoff_when_bounded_blend_is_unavailable(
        self,
    ) -> None:
        os.environ["NERO_TOPPRA_EMERGENCY_RESERVE_TICKS"] = "3"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            for tick in range(1, 22):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)
            self.assertLessEqual(queue.remaining_steps, 3)
            old_plan = queue._active

            queue.load(make_chunk(feedback + 0.01))
            ready = queue._retime_future.result(timeout=2.0)
            ready.plan = replace(
                ready.plan,
                retiming=replace(
                    ready.plan.retiming,
                    status="retimed",
                    feasible=True,
                    fallback=False,
                    reason=None,
                ),
            )
            queue._select_bounded_blend = lambda *args, **kwargs: None
            queue.sample(22 / 30.0, feedback=feedback)

            self.assertIsNot(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            with open(self.log_path, encoding="utf-8") as stream:
                self.assertIn("reserve_exhaustion_follower_handoff", stream.read())
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", None)

    def test_low_reserve_discards_hard_mismatch_and_unblocks_replan(self) -> None:
        os.environ["NERO_TOPPRA_EMERGENCY_RESERVE_TICKS"] = "3"
        os.environ["NERO_TOPPRA_BOUNDARY_PATH_MODE"] = "path_curvature"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            for tick in range(1, 22):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)
            old_plan = queue._active

            queue.load(make_chunk(feedback + 0.1))
            queue._retime_future.result(timeout=2.0)
            queue.sample(22 / 30.0, feedback=feedback)

            self.assertIs(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            self.assertLessEqual(queue.remaining_steps, 3)
            with open(self.log_path, encoding="utf-8") as stream:
                self.assertIn("discarded_hard_mismatch_for_replan", stream.read())
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_EMERGENCY_RESERVE_TICKS", None)
            os.environ.pop("NERO_TOPPRA_BOUNDARY_PATH_MODE", None)

    def test_low_reserve_discards_unblendable_candidate_when_follower_disabled(
        self,
    ) -> None:
        os.environ["NERO_TOPPRA_MIN_COMMIT_TICKS"] = "20"
        os.environ["NERO_TOPPRA_EMERGENCY_RESERVE_TICKS"] = "3"
        os.environ["NERO_TOPPRA_REPLAN_RESERVE_TICKS"] = "8"
        os.environ["NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF"] = "0"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            for tick in range(1, 17):
                feedback = np.concatenate((sample.left_target, sample.right_target))
                sample = queue.sample(tick / 30.0, feedback=feedback)
            self.assertGreater(queue.remaining_steps, 3)
            self.assertLessEqual(queue.remaining_steps, 8)
            old_plan = queue._active

            queue.load(make_chunk(feedback + 0.01))
            ready = queue._retime_future.result(timeout=2.0)
            ready.plan = replace(
                ready.plan,
                retiming=replace(
                    ready.plan.retiming,
                    status="retimed",
                    feasible=True,
                    fallback=False,
                    reason=None,
                ),
            )
            queue._select_bounded_blend = lambda *args, **kwargs: None
            queue.sample(17 / 30.0, feedback=feedback)

            self.assertIs(queue._active, old_plan)
            self.assertIsNone(queue._ready)
            with open(self.log_path, encoding="utf-8") as stream:
                log = stream.read()
            self.assertIn("discarded_unblendable_for_early_replan", log)
            self.assertNotIn("reserve_exhaustion_follower_handoff", log)
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_MIN_COMMIT_TICKS", None)

    def test_rejected_candidate_is_rechecked_and_recovers(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk())
            sample = queue.sample(0.0, feedback=feedback)
            feedback = np.concatenate((sample.left_target, sample.right_target))

            queue.load(make_chunk(feedback + 0.1))
            queue._retime_future.result(timeout=2.0)
            queue.sample(1 / 30.0, feedback=feedback)
            self.assertIsNotNone(queue._ready)
            candidate = queue._ready.plan
            state = candidate.motion_state_at(queue.emitted_steps - candidate.start_wall_tick)
            queue._feedback_position = state.q.copy()
            queue._feedback_velocity = state.qd.copy()
            queue._feedback_acceleration = state.qdd.copy()
            queue._feedback_at = 2 / 30.0
            queue.sample(2 / 30.0, feedback=state.q)
            self.assertIs(queue._active, candidate)
            self.assertIsNone(queue._ready)
            with open(self.log_path, encoding="utf-8") as stream:
                log = stream.read()
            self.assertIn("rejected_hard_measured_mismatch", log)
            self.assertIn("recovered_direct_handoff", log)
        finally:
            queue.close()

    def test_quintic_correction_matches_position_and_velocity_boundaries(self) -> None:
        position_offset = np.linspace(-0.02, 0.02, 14)
        velocity_offset = np.linspace(-0.1, 0.1, 14)
        correction = QuinticHandoffCorrection(
            takeover_wall_tick=10,
            ticks=5,
            action_hz=30.0,
            position_offset=position_offset,
            velocity_offset=velocity_offset,
        )
        np.testing.assert_allclose(correction.evaluate(10), position_offset, atol=1e-12)
        np.testing.assert_allclose(
            correction.evaluate_velocity(10), velocity_offset, atol=1e-12
        )
        np.testing.assert_allclose(correction.evaluate(14), 0.0, atol=1e-12)
        np.testing.assert_allclose(correction.evaluate_velocity(14), 0.0, atol=1e-12)

    def test_action_gain_is_shared_and_does_not_modify_grippers(self) -> None:
        actions = make_chunk(slope=0.001)
        actions[:, 7] = np.linspace(0.0, 1.0, len(actions))
        actions[:, 15] = np.linspace(1.0, 0.0, len(actions))
        result = scale_actions_to_envelope(
            actions,
            anchor=np.zeros(14),
            initial_velocity=np.zeros(14),
            action_hz=30.0,
            max_velocity=np.deg2rad(25.0),
            max_acceleration=np.deg2rad(220.0),
            minimum_gain=0.5,
            gain_step=0.025,
            previous_gain=1.0,
            maximum_gain_rise=0.1,
        )
        self.assertLess(result.gain, 1.0)
        np.testing.assert_allclose(result.actions[:, 7], actions[:, 7])
        np.testing.assert_allclose(result.actions[:, 15], actions[:, 15])
        np.testing.assert_allclose(
            result.actions[1:, ARM_COLUMNS],
            result.gain * actions[1:, ARM_COLUMNS],
        )
        np.testing.assert_allclose(result.actions[0, ARM_COLUMNS], 0.0)

    def test_action_gain_rises_by_at_most_configured_step(self) -> None:
        actions = make_chunk(slope=0.0001)
        result = scale_actions_to_envelope(
            actions,
            anchor=np.zeros(14),
            initial_velocity=np.zeros(14),
            action_hz=30.0,
            max_velocity=np.deg2rad(25.0),
            max_acceleration=np.deg2rad(220.0),
            minimum_gain=0.5,
            gain_step=0.025,
            previous_gain=0.5,
            maximum_gain_rise=0.1,
        )
        self.assertAlmostEqual(result.gain, 0.6, places=9)

    def test_action_gain_anchors_elapsed_rtc_row_and_checks_only_future(self) -> None:
        actions = make_chunk(slope=0.001)
        anchor = np.linspace(-0.02, 0.02, 14)
        result = scale_actions_to_envelope(
            actions,
            anchor=anchor,
            initial_velocity=np.zeros(14),
            action_hz=30.0,
            max_velocity=np.deg2rad(25.0),
            max_acceleration=np.deg2rad(220.0),
            minimum_gain=0.5,
            gain_step=0.025,
            previous_gain=1.0,
            maximum_gain_rise=0.1,
            start_index=6,
        )
        np.testing.assert_allclose(result.actions[6, ARM_COLUMNS], anchor)
        np.testing.assert_allclose(
            result.actions[7:, ARM_COLUMNS],
            anchor + result.gain * (actions[7:, ARM_COLUMNS] - anchor),
        )

    def test_path_boundary_mode_preserves_scaled_waypoint(self) -> None:
        actions = make_chunk(slope=0.001)
        anchor = np.linspace(-0.02, 0.02, 14)
        result = scale_actions_to_envelope(
            actions,
            anchor=anchor,
            initial_velocity=np.zeros(14),
            action_hz=30.0,
            max_velocity=np.deg2rad(25.0),
            max_acceleration=np.deg2rad(220.0),
            minimum_gain=0.5,
            gain_step=0.025,
            previous_gain=1.0,
            maximum_gain_rise=0.1,
            start_index=6,
            anchor_start_waypoint=False,
        )
        expected = anchor + result.gain * (actions[6, ARM_COLUMNS] - anchor)
        np.testing.assert_allclose(result.actions[6, ARM_COLUMNS], expected)
        self.assertGreater(float(np.max(np.abs(expected - anchor))), 0.0)

    def test_skip_steps_start_plan_at_current_wall_tick_and_preserve_boundary(self) -> None:
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14)
            queue._feedback_position = feedback.copy()
            queue._feedback_at = 1.0
            queue._emitted = 10
            queue.load(make_chunk(), skip_steps=6)
            queue._retime_future.result(timeout=2.0)
            ready = queue._retime_future.result()
            self.assertEqual(ready.plan.start_wall_tick, 10)
            self.assertEqual(len(ready.plan.retiming.commands), 18)
            self.assertEqual(ready.plan.optimization_end_wall_tick, 28)
            np.testing.assert_allclose(ready.plan.actions[6, ARM_COLUMNS], feedback)
        finally:
            queue.close()

    def test_b_queue_marks_path_curvature_mode_without_overwriting_waypoint(self) -> None:
        os.environ["NERO_TOPPRA_BOUNDARY_PATH_MODE"] = "path_curvature"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.linspace(-0.01, 0.01, 14)
            queue._feedback_position = feedback.copy()
            queue._feedback_at = 1.0
            actions = make_chunk()
            queue.load(actions, skip_steps=6)
            ready = queue._retime_future.result(timeout=2.0)
            self.assertEqual(ready.boundary_path_mode, "path_curvature")
            expected = feedback + ready.action_gain.gain * (
                actions[6, ARM_COLUMNS] - feedback
            )
            np.testing.assert_allclose(ready.plan.actions[6, ARM_COLUMNS], expected)
            self.assertEqual(ready.plan.optimization_end_wall_tick, 18)
        finally:
            queue.close()

    def test_fallback_chunk_uses_position_only_handoff(self) -> None:
        os.environ["NERO_TOPPRA_MAX_ACCELERATION_DEG_S2"] = "20"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk(slope=0.0012))
            sample = queue.sample(0.0, feedback=feedback)
            self.assertEqual(queue._active.retiming.status, "fallback_original")
            feedback = np.concatenate((sample.left_target, sample.right_target))

            queue.load(make_chunk(feedback + 0.005, slope=0.0012))
            queue._retime_future.result(timeout=2.0)
            sample = queue.sample(1 / 30.0, feedback=feedback)
            self.assertIsNone(queue._ready)
            command = np.concatenate((sample.left_target, sample.right_target))
            np.testing.assert_allclose(command, feedback, atol=1e-10)
            with open(self.log_path, encoding="utf-8") as stream:
                self.assertIn("fallback_position_handoff", stream.read())
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_MAX_ACCELERATION_DEG_S2", None)

    def test_longer_dynamic_blend_recovers_large_safe_fallback_step(self) -> None:
        os.environ["NERO_TOPPRA_HARD_VELOCITY_DEG_S"] = "100"
        queue = RecedingToppraRtcQueue()
        try:
            feedback = np.zeros(14, dtype=np.float64)
            queue.load(make_chunk(slope=0.002))
            sample = queue.sample(0.0, feedback=feedback)
            feedback = np.concatenate((sample.left_target, sample.right_target))

            queue.load(make_chunk(feedback + 0.03, slope=0.002))
            queue._retime_future.result(timeout=2.0)
            sample = queue.sample(1 / 30.0, feedback=feedback)
            self.assertEqual(sample.status, "rtc_toppra_tracking")
            self.assertIsNone(queue._ready)
            self.assertIsNotNone(queue._blend)
            self.assertGreater(queue._blend.ticks, 5)
            with open(self.log_path, encoding="utf-8") as stream:
                self.assertIn("fallback_position_handoff", stream.read())
        finally:
            queue.close()
            os.environ.pop("NERO_TOPPRA_HARD_VELOCITY_DEG_S", None)

    def test_follower_bridge_publishes_combined_command_derivatives(self) -> None:
        class Sample:
            def __init__(self, command, velocity):
                self.command = np.asarray(command, dtype=np.float64)
                self.velocity = np.asarray(velocity, dtype=np.float64)

        class DummyFollower:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def initialize(self, measured, *, now):
                return Sample(measured, np.zeros(7))

            def step(self, sample, *, measured, now):
                del measured
                return sample

        bridged = make_bridged_follower_class(DummyFollower)
        left = bridged()
        right = bridged()
        left.initialize(np.zeros(7), now=1.0)
        right.initialize(np.ones(7), now=1.0)
        left.step(Sample(np.full(7, 0.1), np.full(7, 0.2)), measured=np.zeros(7), now=1.1)
        right.step(Sample(np.full(7, 1.1), np.full(7, 0.3)), measured=np.ones(7), now=1.1)
        state = FOLLOWER_STATE_REGISTRY.combined()
        self.assertIsNotNone(state)
        np.testing.assert_allclose(state.position[:7], 0.1)
        np.testing.assert_allclose(state.position[7:], 1.1)
        np.testing.assert_allclose(state.velocity[:7], 0.2)
        np.testing.assert_allclose(state.velocity[7:], 0.3)
        np.testing.assert_allclose(state.acceleration[:7], 2.0)
        np.testing.assert_allclose(state.acceleration[7:], 3.0)

    def test_queue_prefers_follower_command_boundary(self) -> None:
        class Sample:
            def __init__(self, command, velocity):
                self.command = np.asarray(command, dtype=np.float64)
                self.velocity = np.asarray(velocity, dtype=np.float64)

        FOLLOWER_STATE_REGISTRY.reset()
        left_slot = FOLLOWER_STATE_REGISTRY.register()
        right_slot = FOLLOWER_STATE_REGISTRY.register()
        FOLLOWER_STATE_REGISTRY.update(
            left_slot, Sample(np.full(7, 0.01), np.full(7, 0.02)), now=1.0
        )
        FOLLOWER_STATE_REGISTRY.update(
            right_slot, Sample(np.full(7, -0.01), np.full(7, -0.02)), now=1.0
        )
        queue = RecedingToppraRtcQueue()
        try:
            queue._feedback_position = np.zeros(14)
            queue._feedback_at = 1.0
            queue.load(make_chunk())
            queue._retime_future.result(timeout=2.0)
            ready = queue._retime_future.result()
            expected = np.concatenate((np.full(7, 0.01), np.full(7, -0.01)))
            np.testing.assert_allclose(ready.plan.actions[0, ARM_COLUMNS], expected)
            self.assertEqual(ready.boundary_state_source, "follower_command")
        finally:
            queue.close()


if __name__ == "__main__":
    unittest.main()
