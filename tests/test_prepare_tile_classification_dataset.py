from __future__ import annotations

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
