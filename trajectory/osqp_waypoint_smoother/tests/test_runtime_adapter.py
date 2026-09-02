from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from osqp_casadi_rtc_queue import (
    PredictedTakeoverBoundary,
    RecedingOsqpCasadiRtcQueue,
)


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


class OsqpCasadiRuntimeAdapterTest(unittest.TestCase):
    def test_builds_osqp_then_casadi_plan_without_robot_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                phase = np.linspace(0.0, 1.0, 24)
                oscillation = np.zeros(24)
                oscillation[7:17] = np.deg2rad(0.25) * np.asarray([1.0, -1.0] * 5)
                for joint, column in enumerate(ARM_COLUMNS):
                    actions[:, column] = np.deg2rad(4.0) * phase + oscillation
                actions[:, 7] = np.linspace(1.0, 0.0, 24)
                actions[:, 15] = np.linspace(0.2, 0.8, 24)
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=actions[0, ARM_COLUMNS],
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertTrue(
                    ready.plan.retiming.metrics["osqp_feasible"],
                    ready.plan.retiming.metrics["osqp_reason"],
                )
                self.assertIn(
                    ready.plan.retiming.status,
                    {"osqp_casadi_optimized", "osqp_then_fallback_original"},
                )
                self.assertEqual(ready.action_gain.gain, 1.0)
                self.assertFalse(
                    ready.plan.retiming.metrics["osqp_action_gain_fallback"]
                )
                np.testing.assert_array_equal(
                    ready.plan.actions[:, 7], actions[:, 7]
                )
                np.testing.assert_array_equal(
                    ready.plan.actions[:, 15], actions[:, 15]
                )
                self.assertLessEqual(
                    ready.plan.retiming.metrics["osqp_smoothed_jerk_ratio"],
                    1.0003,
                )
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log

    def test_uses_action_gain_only_after_raw_osqp_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                # A sustained 40 deg/s path cannot fit under 28 deg/s by moving
                # each waypoint only 0.3 deg, but a bounded gain can recover it.
                ramp = np.deg2rad(np.arange(24) * (40.0 / 30.0))
                for column in ARM_COLUMNS:
                    actions[:, column] = ramp
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=actions[0, ARM_COLUMNS],
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertTrue(
                    ready.plan.retiming.metrics["osqp_action_gain_fallback"]
                )
                self.assertLess(ready.action_gain.gain, 1.0)
                self.assertTrue(ready.plan.retiming.metrics["osqp_feasible"])
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log

    def test_selects_precomputed_recovery_for_hard_handoff_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                ramp = np.deg2rad(np.arange(24) * (10.0 / 30.0))
                for column in ARM_COLUMNS:
                    actions[:, column] = ramp
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=np.zeros(14),
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertIsNotNone(ready.recovery_plan)
                primary_plan = ready.plan
                recovery_plan = ready.recovery_plan
                queue._ready = ready
                queue._emitted = 8
                queue._refresh_command_state = lambda: None
                queue._boundary_state = lambda: (
                    np.zeros(14),
                    np.zeros(14),
                    "test",
                )
                queue._feedback_position = np.zeros(14)

                queue._takeover_ready_plan()

                self.assertTrue(ready.recovery_selected)
                self.assertIsNot(queue._active, primary_plan)
                self.assertIs(queue._active, recovery_plan)
                self.assertLess(ready.action_gain.gain, 1.0)
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log

    def test_selects_recovery_when_primary_blend_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                ramp = np.deg2rad(np.arange(24) * (10.0 / 30.0))
                for column in ARM_COLUMNS:
                    actions[:, column] = ramp
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=np.zeros(14),
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertIsNotNone(ready.recovery_plan)
                primary_plan = ready.plan
                recovery_plan = ready.recovery_plan
                queue._ready = ready
                queue._emitted = 8
                queue._refresh_command_state = lambda: None
                queue._boundary_state = lambda: (
                    np.zeros(14),
                    np.zeros(14),
                    "test",
                )
                queue._feedback_position = np.zeros(14)
                queue.hard_position_error = np.deg2rad(5.0)
                queue._bounded_handoff_possible = (
                    lambda plan, **kwargs: plan is recovery_plan
                )

                queue._takeover_ready_plan()

                self.assertTrue(ready.recovery_selected)
                self.assertIsNot(queue._active, primary_plan)
                self.assertIs(queue._active, recovery_plan)
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log

    def test_predicts_recovery_boundary_at_commit_or_reserve_tick(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                ramp = np.deg2rad(np.arange(24) * 0.1)
                for column in ARM_COLUMNS:
                    actions[:, column] = ramp
                ready = queue._retime_job(
                    values=actions,
                    generation=0,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=actions[0, ARM_COLUMNS],
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                queue._active = ready.plan
                queue._active_takeover_tick = 0
                queue._emitted = 5
                queue.minimum_commit_ticks = 12
                queue.emergency_reserve_ticks = 3

                prediction = queue._predict_takeover_boundary(
                    action_count=24,
                    skip_steps=5,
                )

                self.assertIsNotNone(prediction)
                self.assertEqual(prediction.wall_tick, 12)
                self.assertEqual(prediction.action_index, 12)
                expected = ready.plan.motion_state_at(12)
                np.testing.assert_allclose(prediction.position, expected.q)
                np.testing.assert_allclose(prediction.velocity, expected.qd)

                queue._command_position = np.zeros(14)
                queue._command_velocity = np.zeros(14)
                queue._command_acceleration = np.zeros(14)
                follower_prediction = queue._predict_takeover_boundary(
                    action_count=24,
                    skip_steps=5,
                )
                self.assertEqual(
                    follower_prediction.source, "predicted_follower_rollout"
                )
                self.assertGreater(np.max(follower_prediction.position), 0.0)
                self.assertLess(
                    np.max(follower_prediction.position), np.max(expected.q)
                )
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log

    def test_recovery_plan_starts_on_predicted_takeover_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingOsqpCasadiRtcQueue(action_hz=30.0)
            try:
                actions = np.zeros((24, 16), dtype=np.float64)
                queue._predicted_recovery_boundaries[1] = PredictedTakeoverBoundary(
                    wall_tick=12,
                    action_index=12,
                    position=np.full(14, np.deg2rad(4.0)),
                    velocity=np.zeros(14),
                    source="test_future",
                )
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=0,
                    loaded_at_tick=5,
                    skip_steps=5,
                    started=time.perf_counter(),
                    anchor=np.zeros(14),
                    initial_velocity=np.zeros(14),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )

                self.assertIsNotNone(ready.recovery_plan)
                self.assertEqual(ready.recovery_plan.start_wall_tick, 12)
                self.assertEqual(ready.recovery_skip_steps, 12)
                self.assertEqual(ready.recovery_prediction_tick, 12)
                self.assertEqual(ready.recovery_prediction_source, "test_future")
                self.assertLess(ready.recovery_action_gain.gain, 1.0)

                primary_plan = ready.plan
                recovery_plan = ready.recovery_plan
                queue._ready = ready
                queue._emitted = 12
                queue._refresh_command_state = lambda: None
                queue._boundary_state = lambda: (
                    np.full(14, np.deg2rad(4.0)),
                    np.zeros(14),
                    "test_future",
                )
                queue._feedback_position = np.full(14, np.deg2rad(4.0))

                queue._takeover_ready_plan()

                self.assertTrue(ready.recovery_selected)
                self.assertIsNot(queue._active, primary_plan)
                self.assertIs(queue._active, recovery_plan)
                self.assertEqual(ready.skip_steps, 12)
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log


if __name__ == "__main__":
    unittest.main()
