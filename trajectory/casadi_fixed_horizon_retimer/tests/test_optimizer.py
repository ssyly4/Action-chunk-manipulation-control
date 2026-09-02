from __future__ import annotations

import unittest

import numpy as np

from fixed_phase_optimizer import CasadiPhaseOptimizer, PhaseOptimizerConfig


class CasadiPhaseOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        phase = np.arange(24, dtype=np.float64)
        self.actions = np.column_stack(
            (0.015 * np.sin(phase / 5.0), 0.0015 * phase)
        )
        self.optimizer = CasadiPhaseOptimizer(
            PhaseOptimizerConfig(
                action_hz=30.0,
                max_velocity=np.asarray((0.3, 0.2)),
                max_acceleration=np.asarray((3.0, 2.0)),
                max_jerk=np.asarray((120.0, 80.0)),
            )
        )

    def test_preserves_fixed_horizon_and_path_endpoints(self) -> None:
        result = self.optimizer.optimize(self.actions)
        self.assertEqual(result.status, "optimized", result.reason)
        self.assertAlmostEqual(result.metrics["sample_span_sec"], 23 / 30)
        self.assertAlmostEqual(result.metrics["boundary_duration_sec"], 24 / 30)
        np.testing.assert_allclose(result.commands[0], self.actions[0], atol=1e-9)
        np.testing.assert_allclose(result.commands[-1], self.actions[-1], atol=1e-9)
        self.assertTrue(np.all(np.diff(result.phase_samples) >= 0.0))
        self.assertGreater(result.metrics["terminal_phase_speed"], 0.0)
        self.assertLessEqual(result.metrics["max_path_error"], 1e-9)

    def test_impossible_fixed_duration_falls_back(self) -> None:
        phase = np.arange(24, dtype=np.float64)
        impossible = np.column_stack((phase * 0.5, phase * 0.25))
        result = self.optimizer.optimize(impossible)
        self.assertTrue(result.fallback)
        self.assertEqual(result.status, "fallback_original")
        self.assertEqual(len(result.commands), 24)
        self.assertAlmostEqual(result.metrics["boundary_duration_sec"], 24 / 30)

    def test_primal_and_dual_warm_start_remains_feasible(self) -> None:
        optimizer = CasadiPhaseOptimizer(
            PhaseOptimizerConfig(
                action_hz=30.0,
                max_velocity=np.asarray((0.3, 0.2)),
                max_acceleration=np.asarray((3.0, 2.0)),
                max_jerk=np.asarray((120.0, 80.0)),
                solver_warm_start_duals=True,
            )
        )
        first = optimizer.optimize(self.actions)
        shifted = self.actions.copy()
        shifted[:, 0] += 0.0002 * np.sin(np.arange(len(shifted)) / 3.0)
        second = optimizer.optimize(shifted)

        self.assertEqual(first.status, "optimized", first.reason)
        self.assertEqual(second.status, "optimized", second.reason)
        self.assertLessEqual(second.metrics["velocity_ratio"], 1.0002)
        self.assertLessEqual(second.metrics["acceleration_ratio"], 1.0002)
        self.assertLessEqual(second.metrics["jerk_ratio"], 1.0002)

    def test_non_arm_events_remain_on_the_raw_action_clock(self) -> None:
        phase = np.arange(24, dtype=np.float64)
        actions = np.column_stack(
            (
                0.015 * np.sin(phase / 5.0),
                0.0015 * phase,
                np.where(phase < 12.0, 0.0, 1.0),
            )
        )
        optimizer = CasadiPhaseOptimizer(
            PhaseOptimizerConfig(
                action_hz=30.0,
                max_velocity=np.asarray((0.3, 0.2)),
                max_acceleration=np.asarray((3.0, 2.0)),
                max_jerk=np.asarray((120.0, 80.0)),
                arm_columns=(0, 1),
            )
        )
        result = optimizer.optimize(actions)
        np.testing.assert_allclose(result.commands[:, 2], actions[:, 2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
