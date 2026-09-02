import unittest

import numpy as np

from nero_vla.guarded_policy_stream import validate_chunk


class PolicyChunkValidationTests(unittest.TestCase):
    def test_long_continuous_horizon_is_recorded_but_not_rejected(self):
        actions = np.zeros((16, 8), dtype=np.float64)
        actions[:, :7] = np.deg2rad(np.linspace(0.5, 16.0, 16))[:, None]
        actions[:, 7] = 0.22
        chunk = {
            "actions": actions,
            "observation_q": np.zeros(7, dtype=np.float64),
        }

        metrics = validate_chunk(
            chunk,
            np.zeros(7, dtype=np.float64),
            max_first_action_deg=2.0,
            max_consecutive_deg=2.5,
        )

        self.assertAlmostEqual(metrics["max_from_observation_deg"], 16.0)

    def test_discontinuous_first_action_is_still_rejected(self):
        actions = np.zeros((16, 8), dtype=np.float64)
        actions[:, :7] = np.deg2rad(3.0)
        actions[:, 7] = 0.22
        chunk = {
            "actions": actions,
            "observation_q": np.zeros(7, dtype=np.float64),
        }

        with self.assertRaisesRegex(RuntimeError, "first action"):
            validate_chunk(
                chunk,
                np.zeros(7, dtype=np.float64),
                max_first_action_deg=2.0,
                max_consecutive_deg=2.5,
            )

    def test_records_selected_action_far_from_feedback_without_changing_chunk_guards(self):
        actions = np.zeros((16, 8), dtype=np.float64)
        actions[:, :7] = np.deg2rad(np.arange(16))[:, None]
        actions[:, 7] = 0.22

        metrics = validate_chunk(
            {"actions": actions, "observation_q": np.zeros(7)},
            np.zeros(7),
            max_first_action_deg=2.0,
            max_consecutive_deg=2.5,
            aligned_action=actions[5],
            aligned_action_offset_steps=5.0,
        )

        self.assertAlmostEqual(metrics["max_aligned_from_current_deg"], 5.0)
        self.assertAlmostEqual(metrics["aligned_action_offset_steps"], 5.0)


if __name__ == "__main__":
    unittest.main()
