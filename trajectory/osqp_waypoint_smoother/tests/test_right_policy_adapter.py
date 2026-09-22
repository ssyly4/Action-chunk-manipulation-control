from __future__ import annotations

import unittest

import numpy as np

from right_policy_adapter import (
    RightOnlyPolicyClient,
    expand_right_actions,
    to_right_policy_observation,
)


class _Client:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.observation = None
        self.closed = False

    def infer_timed(self, observation):
        self.observation = observation
        return {"actions": self.actions}, {"latency_ms": 1.0}

    def close(self) -> None:
        self.closed = True


class RightPolicyAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = np.arange(16, dtype=np.float32)
        self.observation = {
            "observation/world_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation/left_wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
            "observation/right_wrist_image": np.full((8, 8, 3), 2, dtype=np.uint8),
            "observation/state": self.state,
            "prompt": "pick up the bottle and place it into the box",
            "__openpi_noise": np.zeros((16, 32), dtype=np.float32),
            "__openpi_num_steps": 3,
            "__openpi_rtc": {
                "prev_chunk_left_over": np.arange(8 * 16).reshape(8, 16),
                "inference_delay": 7,
                "execution_horizon": 8,
                "max_guidance_weight": 2.0,
            },
        }

    def test_maps_images_state_and_rtc_prefix_to_right_arm(self) -> None:
        mapped = to_right_policy_observation(self.observation)
        np.testing.assert_array_equal(mapped["observation/state"], self.state[8:16])
        np.testing.assert_array_equal(
            mapped["observation/wrist_image"],
            self.observation["observation/right_wrist_image"],
        )
        self.assertNotIn("observation/left_wrist_image", mapped)
        np.testing.assert_array_equal(
            mapped["__openpi_rtc"]["prev_chunk_left_over"],
            self.observation["__openpi_rtc"]["prev_chunk_left_over"][:, 8:16],
        )

    def test_expands_right_actions_with_constant_virtual_left(self) -> None:
        right = np.arange(16 * 8, dtype=np.float64).reshape(16, 8)
        expanded = expand_right_actions(right, bimanual_state=self.state)
        self.assertEqual(expanded.shape, (16, 16))
        np.testing.assert_array_equal(expanded[:, :8], np.repeat(self.state[None, :8], 16, axis=0))
        np.testing.assert_array_equal(expanded[:, 8:16], right)

    def test_client_preserves_transport_and_closes_inner_client(self) -> None:
        right = np.arange(16 * 8, dtype=np.float64).reshape(16, 8)
        inner = _Client(right)
        client = RightOnlyPolicyClient(inner)
        result, transport = client.infer_timed(self.observation)
        self.assertEqual(result["actions"].shape, (16, 16))
        self.assertEqual(transport, {"latency_ms": 1.0})
        np.testing.assert_array_equal(inner.observation["observation/state"], self.state[8:16])
        client.close()
        self.assertTrue(inner.closed)


if __name__ == "__main__":
    unittest.main()
