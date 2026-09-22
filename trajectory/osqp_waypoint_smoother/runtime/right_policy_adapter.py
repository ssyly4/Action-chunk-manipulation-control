"""Adapt a single-right-arm OpenPI policy to the bimanual trajectory runtime."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


RIGHT_STATE = slice(8, 16)
RIGHT_JOINTS = slice(8, 15)


def to_right_policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Map the production bimanual observation/RTC contract to right-arm 8D."""
    state = np.asarray(observation["observation/state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("bimanual observation/state must contain 16 finite values")

    result = {
        "observation/external_image": observation["observation/world_image"],
        "observation/wrist_image": observation["observation/right_wrist_image"],
        "observation/state": state[RIGHT_STATE].copy(),
        "prompt": observation["prompt"],
    }
    for key in ("__openpi_noise", "__openpi_num_steps"):
        if key in observation:
            result[key] = observation[key]

    if "__openpi_rtc" in observation:
        rtc = deepcopy(observation["__openpi_rtc"])
        previous = np.asarray(rtc["prev_chunk_left_over"], dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1] != 16:
            raise ValueError("bimanual RTC prefix must have shape (H, 16)")
        if not np.isfinite(previous).all():
            raise ValueError("bimanual RTC prefix must be finite")
        rtc["prev_chunk_left_over"] = previous[:, RIGHT_STATE].copy()
        result["__openpi_rtc"] = rtc
    return result


def expand_right_actions(
    actions: np.ndarray, *, bimanual_state: np.ndarray
) -> np.ndarray:
    """Embed Hx8 right actions in Hx16 with an explicit virtual-left hold."""
    right = np.asarray(actions, dtype=np.float64)
    state = np.asarray(bimanual_state, dtype=np.float64)
    if right.ndim != 2 or right.shape[1] != 8 or len(right) < 2:
        raise ValueError("right policy actions must have shape (H, 8), H >= 2")
    if state.shape != (16,):
        raise ValueError("bimanual state must have shape (16,)")
    if not np.isfinite(right).all() or not np.isfinite(state).all():
        raise ValueError("actions and state must be finite")

    expanded = np.repeat(state[None, :], len(right), axis=0)
    expanded[:, RIGHT_STATE] = right
    return expanded


class RightOnlyPolicyClient:
    """OpenPI client facade preserving the bimanual runtime action contract."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def infer_timed(self, observation: dict[str, Any]):
        bimanual_state = np.asarray(observation["observation/state"], dtype=np.float64)
        right_observation = to_right_policy_observation(observation)
        result, transport = self._client.infer_timed(right_observation)
        converted = dict(result)
        converted["actions"] = expand_right_actions(
            result.get("actions"), bimanual_state=bimanual_state
        )
        return converted, transport

    def close(self) -> None:
        self._client.close()
