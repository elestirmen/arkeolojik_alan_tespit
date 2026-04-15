"""Shared channel schema for training and inference.

5-kanal modeli:  R | G | B | SVF | SLRM
"""

from __future__ import annotations

from typing import Sequence, Tuple

MODEL_CHANNEL_NAMES: Tuple[str, ...] = (
    "R",
    "G",
    "B",
    "SVF",
    "SLRM",
)

LOCKED_TRAINED_ONLY_FIELDS: Tuple[str, ...] = ("tile", "overlap", "bands")

METADATA_SCHEMA_VERSION = 3


def expected_channel_names(in_channels: int) -> Tuple[str, ...]:
    """Return the canonical prefix for the given channel count."""
    if in_channels < 1 or in_channels > len(MODEL_CHANNEL_NAMES):
        raise ValueError(
            f"in_channels must be between 1 and {len(MODEL_CHANNEL_NAMES)}, got {in_channels}"
        )
    return MODEL_CHANNEL_NAMES[:in_channels]


def channel_names_match(channel_names: Sequence[str], in_channels: int) -> bool:
    """Compare channel names with the canonical schema for a channel count."""
    return tuple(str(name) for name in channel_names) == expected_channel_names(in_channels)
