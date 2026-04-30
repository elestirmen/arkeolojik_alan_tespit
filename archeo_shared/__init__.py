"""Shared primitives used by training, data prep, and inference."""

from .channels import (
    LOCKED_TRAINED_ONLY_FIELDS,
    METADATA_SCHEMA_VERSION,
    MODEL_CHANNEL_NAMES,
    RGB_CHANNEL_NAMES,
    TOPO5_CHANNEL_NAMES,
    TOPO7_CHANNEL_NAMES,
    input_band_count_for_feature_mode,
)

__all__ = [
    "LOCKED_TRAINED_ONLY_FIELDS",
    "METADATA_SCHEMA_VERSION",
    "MODEL_CHANNEL_NAMES",
    "RGB_CHANNEL_NAMES",
    "TOPO5_CHANNEL_NAMES",
    "TOPO7_CHANNEL_NAMES",
    "input_band_count_for_feature_mode",
]
