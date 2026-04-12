from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window

from prepare_tile_classification_dataset import (
    _implicit_invalid_mask,
    positive_ratio_from_mask,
    read_window_data,
    save_tiles,
    SourcePair,
    TileRecord,
    validate_args,
    validate_source_raster,
)


def test_implicit_invalid_mask_detects_fill_patterns() -> None:
    data = np.ones((5, 2, 2), dtype=np.float32)
    data[:, 0, 0] = 0.0
    data[:3, 0, 1] = 0.0
    data[3:, 0, 1] = -10000.0

    invalid = _implicit_invalid_mask(data)

    assert bool(invalid[0, 0]) is True
    assert bool(invalid[0, 1]) is True
    assert bool(invalid[1, 1]) is False


def test_read_window_data_converts_implicit_fill_pixels_to_nan() -> None:
    arr = np.ones((5, 2, 2), dtype=np.float32)
    arr[:3, 0, 1] = 0.0
    arr[3:, 0, 1] = -10000.0

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 5,
        "dtype": "float32",
        "transform": from_origin(0, 2, 1, 1),
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr)
        with memfile.open() as src:
            data, valid = read_window_data(src, (1, 2, 3, 4, 5), Window(0, 0, 2, 2))

    assert bool(valid[0, 1]) is False
    assert np.isnan(data[:, 0, 1]).all()
    assert bool(valid[1, 1]) is True


def test_validate_source_raster_rejects_uint8_dsm_dtm() -> None:
    arr = np.zeros((5, 2, 2), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 5,
        "dtype": "uint8",
        "transform": from_origin(0, 2, 1, 1),
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr)
        with memfile.open() as src:
            with pytest.raises(ValueError, match="8-bit"):
                validate_source_raster(src, (1, 2, 3, 4, 5), Path("bad_uint8.tif"))


def test_positive_ratio_from_mask_ignores_invalid_pixels() -> None:
    mask = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [True, False],
            [True, False],
        ]
    )

    ratio, positive_pixels, total_pixels = positive_ratio_from_mask(mask, valid_mask=valid_mask)

    assert ratio == 1.0
    assert positive_pixels == 2
    assert total_pixels == 2


def test_positive_ratio_from_mask_respects_mask_nodata() -> None:
    mask = np.array(
        [
            [7.0, 3.0],
            [7.0, 7.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones((2, 2), dtype=bool)

    ratio, positive_pixels, total_pixels = positive_ratio_from_mask(
        mask,
        valid_mask=valid_mask,
        negative_value=7.0,
    )

    assert ratio == 0.25
    assert positive_pixels == 1
    assert total_pixels == 4


def test_validate_args_rejects_non_positive_num_workers() -> None:
    args = argparse.Namespace(
        tile_size=256,
        overlap=128,
        bands="1,2,3,4,5",
        tpi_radii="5,15,30",
        positive_ratio_threshold=0.02,
        valid_ratio_threshold=0.7,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        train_negative_keep_ratio=0.35,
        negative_to_positive_ratio=1.0,
        train_negative_max=None,
        seed=42,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="num_workers"):
        validate_args(args)


def test_save_tiles_supports_npy_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raster_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 5,
        "dtype": "float32",
        "transform": from_origin(0, 4, 1, 1),
    }
    raster_data = np.ones((5, 4, 4), dtype=np.float32)
    mask_data = np.zeros((1, 4, 4), dtype=np.float32)

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(raster_data)
    mask_profile = dict(profile)
    mask_profile["count"] = 1
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask_data)

    output_dir = tmp_path / "dataset"
    (output_dir / "train" / "Positive").mkdir(parents=True)
    metadata: dict[str, object] = {}
    pair = SourcePair(name="source", raster_path=raster_path, mask_path=mask_path)
    record = TileRecord(
        source_name="source",
        split="train",
        label="Positive",
        row_off=0,
        col_off=0,
        positive_ratio=0.5,
        valid_ratio=1.0,
        positive_pixels=8,
        total_pixels=16,
    )

    monkeypatch.setattr(
        "prepare_tile_classification_dataset.compute_tile_stack",
        lambda **_: np.ones((12, 4, 4), dtype=np.float32),
    )

    args = argparse.Namespace(
        bands="1,2,3,4,5",
        tpi_radii="5,15,30",
        format="npy",
        num_workers=1,
        tile_size=4,
        normalize=True,
        tile_prefix="",
    )

    save_tiles(
        output_dir=output_dir,
        pairs=[pair],
        selected_records=[record],
        metadata=metadata,
        args=args,
    )

    saved_path = output_dir / record.output_relpath
    assert saved_path.exists()
    assert saved_path.suffix == ".npy"
    loaded = np.load(saved_path)
    assert loaded.shape == (12, 4, 4)
    assert metadata["saved_tiles"] == 1


def test_save_tiles_selected_regions_skips_derivative_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 5,
        "dtype": "float32",
        "transform": from_origin(0, 4, 1, 1),
    }
    raster_data = np.ones((5, 4, 4), dtype=np.float32)
    mask_data = np.zeros((1, 4, 4), dtype=np.uint8)

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(raster_data)
    mask_profile = dict(profile)
    mask_profile["count"] = 1
    mask_profile["dtype"] = "uint8"
    mask_profile["nodata"] = 0
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask_data)

    output_dir = tmp_path / "dataset"
    (output_dir / "train" / "Negative").mkdir(parents=True)
    metadata: dict[str, object] = {}
    pair = SourcePair(name="source", raster_path=raster_path, mask_path=mask_path)
    record = TileRecord(
        source_name="source",
        split="train",
        label="Negative",
        row_off=0,
        col_off=0,
        positive_ratio=0.0,
        valid_ratio=1.0,
        positive_pixels=0,
        total_pixels=16,
    )

    monkeypatch.setattr(
        "prepare_tile_classification_dataset.compute_tile_stack",
        lambda **_: np.ones((12, 4, 4), dtype=np.float32),
    )

    def _unexpected_cache_call(**_: object) -> None:
        raise AssertionError("selected_regions modunda derivative cache hazirlanmamali")

    monkeypatch.setattr(
        "prepare_tile_classification_dataset.prepare_derivative_cache_for_source",
        _unexpected_cache_call,
    )

    args = argparse.Namespace(
        bands="1,2,3,4,5",
        tpi_radii="5,15,30",
        format="npy",
        num_workers=1,
        tile_size=4,
        normalize=True,
        tile_prefix="",
        sampling_mode="selected_regions",
    )

    save_tiles(
        output_dir=output_dir,
        pairs=[pair],
        selected_records=[record],
        metadata=metadata,
        args=args,
    )

    assert metadata["derivative_cache"][0]["mode"] == "selected_regions_direct"
