"""Shared primitives used by training, data prep, and inference."""

from .channels import LOCKED_TRAINED_ONLY_FIELDS, METADATA_SCHEMA_VERSION, MODEL_CHANNEL_NAMES

__all__ = [
    "LOCKED_TRAINED_ONLY_FIELDS",
    "METADATA_SCHEMA_VERSION",
    "MODEL_CHANNEL_NAMES",
]
