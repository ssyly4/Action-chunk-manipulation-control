"""Compatibility imports for the retired rolling-overlap runtime name."""

from receding_toppra_queue import (  # noqa: F401
    ARM_COLUMNS,
    QuinticHandoffCorrection,
    RecedingToppraRtcQueue,
)

RollingToppraRtcQueue = RecedingToppraRtcQueue

__all__ = [
    "ARM_COLUMNS",
    "QuinticHandoffCorrection",
    "RecedingToppraRtcQueue",
    "RollingToppraRtcQueue",
]
