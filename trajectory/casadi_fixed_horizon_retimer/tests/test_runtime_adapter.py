from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from casadi_rtc_queue import QuinticHandoffCorrection, RecedingCasadiRtcQueue


class CasadiRuntimeAdapterTest(unittest.TestCase):
    def test_builds_a_fixed_horizon_ready_plan_without_robot_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old_log = os.environ.get("NERO_TOPPRA_RUNTIME_LOG")
            os.environ["NERO_TOPPRA_RUNTIME_LOG"] = str(Path(directory) / "runtime.jsonl")
            queue = RecedingCasadiRtcQueue(action_hz=30.0)
            try:
                phase = np.arange(24, dtype=np.float64)
                arms = np.column_stack(
                    [0.001 * phase + 0.002 * joint for joint in range(14)]
                )
                actions = np.empty((24, 16), dtype=np.float64)
                actions[:, :7] = arms[:, :7]
                actions[:, 7] = 1.0
                actions[:, 8:15] = arms[:, 7:]
                actions[:, 15] = 1.0
                started = time.perf_counter()
                initial = queue._retime_job(
                    values=actions,
                    generation=0,
                    raw_start_tick=0,
                    loaded_at_tick=0,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=arms[0],
                    initial_velocity=np.full(14, 0.03),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertTrue(initial.plan.retiming.fallback)
                self.assertLess(time.perf_counter() - started, 0.05)
                ready = queue._retime_job(
                    values=actions,
                    generation=1,
                    raw_start_tick=1,
                    loaded_at_tick=1,
                    skip_steps=0,
                    started=time.perf_counter(),
                    anchor=arms[0],
                    initial_velocity=np.full(14, 0.03),
                    previous_gain=1.0,
                    boundary_state_source="test",
                    boundary_path_mode="path_curvature",
                )
                self.assertEqual(len(ready.plan.retiming.commands), 24)
                self.assertTrue(ready.plan.retiming.feasible, ready.plan.retiming.reason)
                self.assertEqual(ready.plan.retiming.status, "casadi_optimized")
                self.assertAlmostEqual(
                    ready.plan.retiming.metrics["boundary_duration_sec"], 24 / 30
                )
                queue._feedback_position = arms[0].copy()
                queue._feedback_velocity = np.zeros(14)
                queue._feedback_acceleration = np.zeros(14)
                queue._feedback_at = time.monotonic()
                queue._emitted = ready.plan.start_wall_tick
                state = ready.plan.motion_state_at(0)
                position_offset = np.full(14, 0.003)
                accepted = queue._select_bounded_blend(
                    ready.plan,
                    local=0,
                    position_offset=position_offset,
                    velocity_offset=-state.qd,
                    minimum_ticks=2,
                    match_velocity=True,
                )
                self.assertIsNotNone(accepted)

                original_max_jerk = queue.max_jerk
                queue.max_jerk = 1e-6
                rejected = queue._select_bounded_blend(
                    ready.plan,
                    local=0,
                    position_offset=position_offset,
                    velocity_offset=-state.qd,
                    minimum_ticks=2,
                    match_velocity=True,
                )
                queue.max_jerk = original_max_jerk
                self.assertIsNone(rejected)

                queue._active = ready.plan
                queue._emitted = ready.plan.start_wall_tick
                queue._blend = QuinticHandoffCorrection(
                    takeover_wall_tick=queue._emitted,
                    ticks=4,
                    action_hz=queue.action_hz,
                    position_offset=np.full(14, 0.002),
                    velocity_offset=np.full(14, 0.01),
                )
                expected_velocity = (
                    ready.plan.motion_state_at(0).qd
                    + queue._blend.evaluate_velocity(queue._emitted)
                )
                np.testing.assert_allclose(
                    queue._scheduled_velocity_at_tick(queue._emitted),
                    expected_velocity,
                )
            finally:
                queue.close()
                if old_log is None:
                    os.environ.pop("NERO_TOPPRA_RUNTIME_LOG", None)
                else:
                    os.environ["NERO_TOPPRA_RUNTIME_LOG"] = old_log


if __name__ == "__main__":
    unittest.main()
