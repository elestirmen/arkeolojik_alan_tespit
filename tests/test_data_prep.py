from __future__ import annotations

import json

import numpy as np
import pytest

from egitim_verisi_olusturma import (
    _is_positive_for_balance,
    _probability_to_rgb_grid,
    _split_windows_for_train_val,
    _validate_append_compatibility,
    _validate_tile_generation_params,
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


def test_validate_tile_generation_params_rejects_invalid_train_negative_keep_ratio() -> None:
    with pytest.raises(ValueError, match="train_negative_keep_ratio"):
        _validate_tile_generation_params(
            tile_size=256,
            overlap=64,
            min_positive_ratio=0.0,
            max_nodata_ratio=0.3,
            train_ratio=0.8,
            save_format="npz",
            split_mode="spatial",
            train_negative_keep_ratio=1.2,
        )


def test_validate_tile_generation_params_rejects_negative_train_negative_max() -> None:
    with pytest.raises(ValueError, match="train_negative_max"):
        _validate_tile_generation_params(
            tile_size=256,
            overlap=64,
            min_positive_ratio=0.0,
            max_nodata_ratio=0.3,
            train_ratio=0.8,
            save_format="npz",
            split_mode="spatial",
            train_negative_keep_ratio=1.0,
            train_negative_max=-1,
        )


def test_validate_tile_generation_params_rejects_non_positive_num_workers() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        _validate_tile_generation_params(
            tile_size=256,
            overlap=64,
            min_positive_ratio=0.0,
            max_nodata_ratio=0.3,
            train_ratio=0.8,
            save_format="npz",
            split_mode="spatial",
            num_workers=0,
        )


def test_probability_to_rgb_grid_maps_values_and_nodata() -> None:
    probs = pytest.importorskip("numpy").array(
        [[0.0, 0.5, 1.0, -1.0]],
        dtype="float32",
    )
    rgb = _probability_to_rgb_grid(probs, nodata_value=-1.0)

    assert rgb.shape == (3, 1, 4)

    # 0.0 -> blue
    assert int(rgb[0, 0, 0]) == 0
    assert int(rgb[1, 0, 0]) == 0
    assert int(rgb[2, 0, 0]) == 255

    # 0.5 -> yellow
    assert int(rgb[0, 0, 1]) == 255
    assert int(rgb[1, 0, 1]) == 255
    assert int(rgb[2, 0, 1]) == 0

    # 1.0 -> red
    assert int(rgb[0, 0, 2]) == 255
    assert int(rgb[1, 0, 2]) == 0
    assert int(rgb[2, 0, 2]) == 0

    # nodata -> black
    assert int(rgb[0, 0, 3]) == 0
    assert int(rgb[1, 0, 3]) == 0
    assert int(rgb[2, 0, 3]) == 0


def _write_minimal_npz_dataset(base_dir, tile_size: int = 256, channels: int = 5) -> None:
    for rel in ("train/images", "train/masks", "val/images", "val/masks"):
        (base_dir / rel).mkdir(parents=True, exist_ok=True)

    image = np.random.rand(channels, tile_size, tile_size).astype(np.float32)
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    mask[0, 0] = 1

    np.savez_compressed(base_dir / "train/images/sample.npz", image=image)
    np.savez_compressed(base_dir / "train/masks/sample.npz", mask=mask)
    np.savez_compressed(base_dir / "val/images/sample.npz", image=image)
    np.savez_compressed(base_dir / "val/masks/sample.npz", mask=mask)


def test_validate_append_compatibility_rejects_tile_size_mismatch(tmp_path) -> None:
    _write_minimal_npz_dataset(tmp_path, tile_size=256, channels=5)

    metadata = {
        "tile_size": 256,
        "bands": "1,2,3,4,5",
        "tpi_radii": [5, 15, 30],
        "normalize": True,
        "save_format": "npz",
        "num_channels": 5,
    }
    with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    with pytest.raises(ValueError, match="tile_size uyusmuyor"):
        _validate_append_compatibility(
            output_dir=tmp_path,
            tile_size=512,
            bands="1,2,3,4,5",
            tpi_radii=(5, 15, 30),
            normalize=True,
            save_format="npz",
        )


def test_validate_append_compatibility_rejects_normalize_mismatch(tmp_path) -> None:
    _write_minimal_npz_dataset(tmp_path, tile_size=256, channels=5)

    metadata = {
        "tile_size": 256,
        "bands": "1,2,3,4,5",
        "tpi_radii": [5, 15, 30],
        "normalize": True,
        "save_format": "npz",
        "num_channels": 5,
    }
    with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    with pytest.raises(ValueError, match="normalize uyusmuyor"):
        _validate_append_compatibility(
            output_dir=tmp_path,
            tile_size=256,
            bands="1,2,3,4,5",
            tpi_radii=(5, 15, 30),
            normalize=False,
            save_format="npz",
        )


def test_validate_append_compatibility_accepts_matching_settings(tmp_path) -> None:
    _write_minimal_npz_dataset(tmp_path, tile_size=256, channels=5)

    metadata = {
        "tile_size": 256,
        "bands": "1,2,3,4,5",
        "tpi_radii": [5, 15, 30],
        "normalize": True,
        "save_format": "npz",
        "num_channels": 5,
    }
    with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    _validate_append_compatibility(
        output_dir=tmp_path,
        tile_size=256,
        bands="1,2,3,4,5",
        tpi_radii=(5, 15, 30),
        normalize=True,
        save_format="npz",
    )
