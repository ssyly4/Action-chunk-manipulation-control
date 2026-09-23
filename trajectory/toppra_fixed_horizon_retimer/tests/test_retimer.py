from __future__ import annotations

import unittest

import numpy as np

from fixed_path_retimer import (
    FixedPathRetimer,
    RecedingConfig,
    RecedingFixedPathRetimer,
    RetimeConfig,
)


class FixedPathRetimerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.horizon = 24
        self.phase = np.arange(self.horizon, dtype=np.float64)
        self.actions = np.column_stack(
            (
                0.02 * np.sin(self.phase / 8.0),
                0.0015 * self.phase,
            )
        )
        self.retimer = FixedPathRetimer(
            RetimeConfig(
                action_hz=30.0,
                max_velocity=np.asarray((0.4, 0.2)),
                max_acceleration=np.asarray((3.0, 2.0)),
                max_spline_path_deviation=0.01,
            )
        )

    def test_fixed_boundary_and_endpoints(self) -> None:
        result = self.retimer.retime(self.actions)
        self.assertEqual(result.status, "retimed", result.reason)
        np.testing.assert_allclose(result.commands[0], self.actions[0], atol=1e-9)
        np.testing.assert_allclose(result.commands[-1], self.actions[-1], atol=1e-9)
        self.assertAlmostEqual(result.metrics["boundary_duration_sec"], 24 / 30)
        self.assertAlmostEqual(result.metrics["sample_span_sec"], 23 / 30)
        self.assertTrue(np.all(np.diff(result.phase_samples) >= 0.0))

    def test_rtc_projection_keeps_absolute_boundary(self) -> None:
        consumed = 7
        measured = self.actions[consumed] + np.asarray((0.0002, -0.0001))
        result = self.retimer.retime_rtc_replacement(
            self.actions,
            measured,
            consumed_steps=consumed,
            emitted_at_request=120,
        )
        self.assertEqual(result.remaining_ticks, self.horizon - consumed)
        self.assertEqual(result.replacement_start_tick, 127)
        self.assertEqual(result.replacement_end_tick, 144)
        self.assertEqual(result.request_boundary_tick, 144)
        self.assertEqual(len(result.retiming.commands), self.horizon - consumed)
        np.testing.assert_allclose(
            result.retiming.raw_phase_samples,
            np.arange(consumed, self.horizon),
        )

    def test_impossible_duration_falls_back_without_stretching(self) -> None:
        fast_actions = np.column_stack((self.phase * 0.2, self.phase * 0.1))
        result = self.retimer.retime(fast_actions)
        self.assertTrue(result.fallback)
        self.assertEqual(result.status, "fallback_original")
        self.assertAlmostEqual(result.metrics["boundary_duration_sec"], 24 / 30)
        self.assertEqual(len(result.commands), 24)

    def test_non_arm_events_remain_on_the_raw_action_clock(self) -> None:
        actions = np.column_stack(
            (
                self.actions,
                np.where(self.phase < 12.0, 0.0, 1.0),
            )
        )
        retimer = FixedPathRetimer(
            RetimeConfig(
                action_hz=30.0,
                max_velocity=np.asarray((0.4, 0.2)),
                max_acceleration=np.asarray((3.0, 2.0)),
                arm_columns=(0, 1),
                max_spline_path_deviation=0.01,
            )
        )
        result = retimer.retime(actions)
        expected = np.interp(result.raw_phase_samples, self.phase, actions[:, 2])
        np.testing.assert_allclose(result.commands[:, 2], expected, atol=1e-12)


class RecedingBoundaryVelocityTest(unittest.TestCase):
    def test_start_phase_speed_is_projected_from_follower_velocity(self) -> None:
        phase = np.arange(24, dtype=np.float64)
        actions = np.column_stack((0.002 * phase, 0.001 * phase))
        base = FixedPathRetimer(
            RetimeConfig(
                action_hz=30.0,
                max_velocity=np.asarray((1.0, 1.0)),
                max_acceleration=np.asarray((20.0, 20.0)),
                max_spline_path_deviation=0.01,
            )
        )
        receding = RecedingFixedPathRetimer(base, RecedingConfig())
        expected_phase_speed = 10.0
        start_velocity = np.asarray((0.02, 0.01))
        plan = receding.plan(
            actions,
            start_wall_tick=0,
            start_arm_velocity=start_velocity,
        )
        self.assertAlmostEqual(
            plan.requested_start_phase_speed, expected_phase_speed, places=6
        )

    def test_curvature_margin_caps_boundary_geometric_acceleration(self) -> None:
        phase = np.arange(24, dtype=np.float64)
        actions = np.column_stack(
            (0.003 * np.sin(phase / 2.0), 0.002 * np.cos(phase / 3.0))
        )
        max_acceleration = np.asarray((3.0, 3.0))
        base = FixedPathRetimer(
            RetimeConfig(
                action_hz=30.0,
                max_velocity=np.asarray((1.0, 1.0)),
                max_acceleration=max_acceleration,
                max_spline_path_deviation=0.01,
            )
        )
        receding = RecedingFixedPathRetimer(base, RecedingConfig())
        margin = 0.8
        plan = receding.plan(
            actions,
            start_wall_tick=0,
            start_arm_velocity=np.asarray((0.5, 0.5)),
            start_speed_curvature_margin=margin,
        )
        curvature = np.abs(plan.reference.evaluate(0.0, 2))
        geometric_acceleration = curvature * plan.requested_start_phase_speed**2
        self.assertTrue(
            np.all(geometric_acceleration <= margin**2 * max_acceleration + 1e-9)
        )


if __name__ == "__main__":
    unittest.main()
