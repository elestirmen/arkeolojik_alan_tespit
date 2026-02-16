from __future__ import annotations

import pytest

from egitim_verisi_olusturma import (
    _is_positive_for_balance,
    _split_windows_for_train_val,
    _validate_tile_generation_params,
)


def test_validate_tile_generation_params_rejects_zero_balance_ratio() -> None:
    with pytest.raises(ValueError, match="balance_ratio"):
        _validate_tile_generation_params(
            tile_size=256,
            overlap=64,
            min_positive_ratio=0.0,
            max_nodata_ratio=0.3,
            train_ratio=0.8,
            save_format="npz",
            balance_ratio=0.0,
            split_mode="spatial",
        )


def test_validate_tile_generation_params_rejects_overlap_ge_tile() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _validate_tile_generation_params(
            tile_size=128,
            overlap=128,
            min_positive_ratio=0.0,
            max_nodata_ratio=0.3,
            train_ratio=0.8,
            save_format="npz",
            balance_ratio=None,
            split_mode="spatial",
        )


def test_split_windows_spatial_discards_boundary_crossers() -> None:
    windows = [(0, 0), (64, 0), (192, 0), (256, 0)]
    train, val, discarded, mode = _split_windows_for_train_val(
        windows,
        train_ratio=0.5,
        tile_size=128,
        raster_height=320,
        split_mode="spatial",
        seed=42,
    )

    assert mode == "spatial"
    assert discarded == 1
    assert (64, 0) not in train
    assert (64, 0) not in val
    assert (0, 0) in train
    assert (192, 0) in val


def test_split_windows_spatial_falls_back_when_one_side_empty() -> None:
    windows = [(0, 0), (32, 0), (64, 0)]
    train, val, discarded, mode = _split_windows_for_train_val(
        windows,
        train_ratio=0.8,
        tile_size=128,
        raster_height=200,
        split_mode="spatial",
        seed=42,
    )

    assert mode == "random_fallback"
    assert discarded >= 0
    assert len(train) > 0
    assert len(val) > 0


def test_is_positive_for_balance_handles_zero_threshold_correctly() -> None:
    assert _is_positive_for_balance(positive_ratio=0.0, min_positive_ratio=0.0) is False
    assert _is_positive_for_balance(positive_ratio=0.01, min_positive_ratio=0.0) is True
    assert _is_positive_for_balance(positive_ratio=0.01, min_positive_ratio=0.02) is False
