"""Process-local command-state bridge for the TOPPRA runtime."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CombinedFollowerState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    updated_at: float


class FollowerStateRegistry:
    """Collect the two production follower states without editing production."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[int, CombinedFollowerState] = {}
        self._next_slot = 0

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._next_slot = 0

    def register(self) -> int:
        with self._lock:
            slot = self._next_slot
            self._next_slot += 1
        if slot > 1:
            raise RuntimeError("TOPPRA bridge expected exactly two arm followers")
        return slot

    def update(self, slot: int, sample: Any, *, now: float) -> None:
        position = np.asarray(sample.command, dtype=np.float64)
        velocity = np.asarray(sample.velocity, dtype=np.float64)
        if position.shape != (7,) or velocity.shape != (7,):
            raise ValueError("bridged follower state must contain seven joints")
        with self._lock:
            previous = self._states.get(slot)
            if previous is None or now <= previous.updated_at:
                acceleration = np.zeros(7, dtype=np.float64)
            else:
                acceleration = (velocity - previous.velocity) / (now - previous.updated_at)
            self._states[slot] = CombinedFollowerState(
                position=position.copy(),
                velocity=velocity.copy(),
                acceleration=acceleration,
                updated_at=float(now),
            )

    def combined(self) -> CombinedFollowerState | None:
        with self._lock:
            if 0 not in self._states or 1 not in self._states:
                return None
            left = self._states[0]
            right = self._states[1]
            return CombinedFollowerState(
                position=np.concatenate((left.position, right.position)),
                velocity=np.concatenate((left.velocity, right.velocity)),
                acceleration=np.concatenate((left.acceleration, right.acceleration)),
                updated_at=min(left.updated_at, right.updated_at),
            )


FOLLOWER_STATE_REGISTRY = FollowerStateRegistry()


class DesiredVelocityRegistry:
    """Publish one bimanual retimer velocity sample to the two followers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._velocity: np.ndarray | None = None

    def reset(self) -> None:
        with self._lock:
            self._velocity = None

    def publish(self, velocity: np.ndarray | None) -> None:
        if velocity is None:
            value = None
        else:
            value = np.asarray(velocity, dtype=np.float64)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError("retimer desired velocity must contain fourteen finite joints")
            value = value.copy()
        with self._lock:
            self._velocity = value

    def arm(self, slot: int) -> np.ndarray | None:
        if slot not in (0, 1):
            raise ValueError("follower slot must be left=0 or right=1")
        with self._lock:
            if self._velocity is None:
                return None
            start = 7 * slot
            return self._velocity[start : start + 7].copy()


DESIRED_VELOCITY_REGISTRY = DesiredVelocityRegistry()


def make_bridged_follower_class(base_class: type) -> type:
    """Return a subclass that mirrors CommandSample state into the registry."""

    class BridgedRateLimitedJointFollower(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._toppra_bridge_slot = FOLLOWER_STATE_REGISTRY.register()

        def initialize(self, measured: np.ndarray, *, now: float):
            sample = super().initialize(measured, now=now)
            FOLLOWER_STATE_REGISTRY.update(self._toppra_bridge_slot, sample, now=now)
            return sample

        def step(self, sample, *, measured: np.ndarray, now: float):
            result = super().step(sample, measured=measured, now=now)
            FOLLOWER_STATE_REGISTRY.update(self._toppra_bridge_slot, result, now=now)
            return result

    BridgedRateLimitedJointFollower.__name__ = "BridgedRateLimitedJointFollower"
    return BridgedRateLimitedJointFollower
