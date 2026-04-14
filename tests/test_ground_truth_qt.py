from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("PySide6")

import ground_truth_kare_etiketleme_qt as gt


def _write_input_raster(path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 5,
        "count": 5,
        "dtype": "float32",
        "transform": from_origin(0, 4, 1, 1),
    }
    data = np.zeros((5, 4, 5), dtype=np.float32)
    data[0] = 10.0
    data[1] = 20.0
    data[2] = 30.0
    data[3] = 40.0
    data[4] = 50.0
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_mask(path: Path, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "transform": from_origin(0, 4, 1, 1),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data[np.newaxis, :, :].astype(np.uint8))


def _make_config(
    *,
    input_path: Path,
    output_path: Path,
    existing_mask: Path | None,
    existing_labels: Path | None,
) -> gt.AppConfig:
    return gt.AppConfig(
        input_path=input_path,
        output_path=output_path,
        existing_mask=existing_mask,
        existing_labels=existing_labels,
        preview_max_size=64,
        bands=(1, 2, 3),
        positive_value=1,
        negative_value=0,
        square_mode=True,
    )


def test_session_prefers_mask_tif_over_gpkg_when_both_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.tif"
    mask_path = tmp_path / "mask.tif"
    labels_path = tmp_path / "mask.gpkg"
    output_path = tmp_path / "out.tif"
    _write_input_raster(input_path)

    mask_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _write_mask(mask_path, mask_data)
    labels_path.write_text("placeholder", encoding="utf-8")

    gpkg_mask = np.ones((4, 5), dtype=np.uint8)
    monkeypatch.setattr(
        gt,
        "load_mask_from_annotations_gpkg",
        lambda **_: gpkg_mask.copy(),
    )

    session = gt.Session(
        _make_config(
            input_path=input_path,
            output_path=output_path,
            existing_mask=mask_path,
            existing_labels=labels_path,
        )
    )
    try:
        assert np.array_equal(session.mask_full, mask_data)
        assert not np.array_equal(session.mask_full, gpkg_mask)
    finally:
        session.close()


def test_session_uses_gpkg_when_mask_tif_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.tif"
    labels_path = tmp_path / "labels.gpkg"
    output_path = tmp_path / "out.tif"
    _write_input_raster(input_path)
    labels_path.write_text("placeholder", encoding="utf-8")

    gpkg_mask = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    monkeypatch.setattr(
        gt,
        "load_mask_from_annotations_gpkg",
        lambda **_: gpkg_mask.copy(),
    )

    session = gt.Session(
        _make_config(
            input_path=input_path,
            output_path=output_path,
            existing_mask=None,
            existing_labels=labels_path,
        )
    )
    try:
        assert np.array_equal(session.mask_full, gpkg_mask)
    finally:
        session.close()
