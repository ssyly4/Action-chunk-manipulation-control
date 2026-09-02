from __future__ import annotations

import unittest

import numpy as np

from waypoint_smoother import OsqpWaypointSmoother, WaypointSmootherConfig


ARM_COLUMNS = tuple(range(7)) + tuple(range(8, 15))


def config(*, trust_deg: float = 1.0) -> WaypointSmootherConfig:
    return WaypointSmootherConfig(
        action_hz=30.0,
        arm_columns=ARM_COLUMNS,
        trust_region=np.deg2rad(trust_deg),
        max_velocity=np.deg2rad(40.0),
        max_acceleration=np.deg2rad(500.0),
        max_jerk=np.deg2rad(12000.0),
        tracking_weight=10.0,
        acceleration_weight=0.2,
        jerk_weight=2.0,
        solver_time_limit_sec=0.1,
    )


def noisy_actions() -> np.ndarray:
    actions = np.zeros((24, 16), dtype=np.float64)
    phase = np.linspace(0.0, 1.0, len(actions))
    base = np.deg2rad(5.0) * phase
    oscillation = np.zeros(len(actions), dtype=np.float64)
    oscillation[7:17] = np.deg2rad(0.35) * np.asarray(
        [1.0, -1.0] * 5, dtype=np.float64
    )
    for joint, column in enumerate(ARM_COLUMNS):
        actions[:, column] = (1.0 + 0.03 * joint) * base + oscillation
    actions[:, 7] = np.linspace(1.0, 0.0, len(actions))
    actions[:, 15] = np.linspace(0.2, 0.8, len(actions))
    return actions


class OsqpWaypointSmootherTest(unittest.TestCase):
    def test_smooths_jerk_without_changing_horizon_or_grippers(self) -> None:
        actions = noisy_actions()
        result = OsqpWaypointSmoother(config()).smooth(actions)

        self.assertTrue(result.feasible, result.reason)
        self.assertFalse(result.fallback)
        self.assertEqual(result.commands.shape, actions.shape)
        np.testing.assert_array_equal(result.commands[:, 7], actions[:, 7])
        np.testing.assert_array_equal(result.commands[:, 15], actions[:, 15])
        self.assertLess(
            result.metrics["smoothed_jerk_ratio"],
            result.metrics["raw_jerk_ratio"],
        )
        self.assertLessEqual(result.metrics["smoothed_jerk_ratio"], 1.0001)
        self.assertLessEqual(
            result.metrics["max_waypoint_deviation_rad"],
            np.deg2rad(1.0) + 2e-5,
        )

    def test_preserves_explicit_start_position_and_velocity(self) -> None:
        actions = noisy_actions()
        boundary_position = actions[0, ARM_COLUMNS].copy()
        boundary_velocity = np.full(14, np.deg2rad(5.0), dtype=np.float64)
        result = OsqpWaypointSmoother(config()).smooth(
            actions,
            boundary_position=boundary_position,
            boundary_velocity=boundary_velocity,
        )

        self.assertTrue(result.feasible, result.reason)
        smoothed = result.commands[:, ARM_COLUMNS]
        np.testing.assert_allclose(smoothed[0], boundary_position, atol=2e-5)
        np.testing.assert_allclose(
            (smoothed[1] - smoothed[0]) * 30.0,
            boundary_velocity,
            atol=2e-5,
        )

    def test_infeasible_boundary_returns_original_chunk(self) -> None:
        actions = noisy_actions()
        boundary_position = actions[0, ARM_COLUMNS] + np.deg2rad(5.0)
        result = OsqpWaypointSmoother(config(trust_deg=0.1)).smooth(
            actions,
            boundary_position=boundary_position,
        )

        self.assertFalse(result.feasible)
        self.assertTrue(result.fallback)
        np.testing.assert_array_equal(result.commands, actions)

    def test_warm_started_second_solve_remains_valid(self) -> None:
        smoother = OsqpWaypointSmoother(config())
        first = smoother.smooth(noisy_actions())
        shifted = noisy_actions()
        shifted[:, ARM_COLUMNS] += np.deg2rad(0.05)
        second = smoother.smooth(shifted)

        self.assertTrue(first.feasible, first.reason)
        self.assertTrue(second.feasible, second.reason)
        self.assertGreater(second.iterations, 0)

    def test_skip_steps_leave_consumed_prefix_unchanged(self) -> None:
        actions = noisy_actions()
        start_index = 7
        boundary = actions[start_index, ARM_COLUMNS].copy()
        result = OsqpWaypointSmoother(config()).smooth(
            actions,
            start_index=start_index,
            boundary_position=boundary,
        )

        self.assertTrue(result.feasible, result.reason)
        np.testing.assert_array_equal(result.commands[:start_index], actions[:start_index])
        np.testing.assert_array_equal(result.commands[:, 7], actions[:, 7])
        np.testing.assert_array_equal(result.commands[:, 15], actions[:, 15])


if __name__ == "__main__":
    unittest.main()
