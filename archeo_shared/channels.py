"""Shared channel schemas for training and inference.

GeoTIFF input is still RGB + DSM + DTM. Model tensors may use:
- rgb3 : R | G | B
- topo5: R | G | B | SVF | SLRM
- topo7: R | G | B | SVF | SLRM | Slope | nDSM
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

RGB_CHANNEL_NAMES: Tuple[str, ...] = (
    "R",
    "G",
    "B",
)

TOPO5_CHANNEL_NAMES: Tuple[str, ...] = (
    *RGB_CHANNEL_NAMES,
    "SVF",
    "SLRM",
)

TOPO7_CHANNEL_NAMES: Tuple[str, ...] = (
    *TOPO5_CHANNEL_NAMES,
    "Slope",
    "nDSM",
)

MODEL_CHANNEL_NAMES: Tuple[str, ...] = (
    *TOPO5_CHANNEL_NAMES,
)

FEATURE_MODE_CHANNEL_NAMES: Dict[str, Tuple[str, ...]] = {
    "rgb3": RGB_CHANNEL_NAMES,
    "topo5": TOPO5_CHANNEL_NAMES,
    "topo7": TOPO7_CHANNEL_NAMES,
}

LOCKED_TRAINED_ONLY_FIELDS: Tuple[str, ...] = ("tile", "overlap", "bands")

METADATA_SCHEMA_VERSION = 4


def normalize_feature_mode(raw: object) -> str:
    """Return a supported feature mode token."""
    feature_mode = str(raw).strip().lower()
    if feature_mode not in FEATURE_MODE_CHANNEL_NAMES:
        raise ValueError(
            f"unsupported feature_mode: {raw!r}. "
            f"Expected one of: {', '.join(FEATURE_MODE_CHANNEL_NAMES)}"
        )
    return feature_mode


def channel_names_for_feature_mode(feature_mode: object) -> Tuple[str, ...]:
    """Return canonical channel names for a feature mode."""
    return FEATURE_MODE_CHANNEL_NAMES[normalize_feature_mode(feature_mode)]


def input_band_count_for_feature_mode(feature_mode: object) -> int:
    """Return expected source GeoTIFF band selections for a feature mode."""
    return 3 if normalize_feature_mode(feature_mode) == "rgb3" else 5


def expected_channel_names(in_channels: int) -> Tuple[str, ...]:
    """Return the canonical channel schema for a supported channel count."""
    for names in FEATURE_MODE_CHANNEL_NAMES.values():
        if int(in_channels) == len(names):
            return names
    supported = ", ".join(str(len(names)) for names in FEATURE_MODE_CHANNEL_NAMES.values())
    raise ValueError(f"in_channels must be one of {supported}, got {in_channels}")


def canonicalize_channel_names(channel_names: Sequence[str]) -> Tuple[str, ...]:
    """Normalize case/spacing while preserving canonical spelling such as nDSM."""
    raw = tuple(str(name).strip() for name in channel_names if str(name).strip())
    if not raw:
        raise ValueError("channel_names cannot be empty")
    try:
        expected = expected_channel_names(len(raw))
    except ValueError as exc:
        raise ValueError(f"unsupported channel count: {len(raw)}") from exc
    expected_by_key = {name.strip().lower(): name for name in expected}
    normalized = tuple(expected_by_key.get(name.lower(), name) for name in raw)
    if normalized != expected:
        raise ValueError(f"expected channel schema {expected}, got {raw}")
    return expected


def channel_names_match(channel_names: Sequence[str], in_channels: int) -> bool:
    """Compare channel names with the canonical schema for a channel count."""
    try:
        return canonicalize_channel_names(channel_names) == expected_channel_names(in_channels)
    except ValueError:
        return False
