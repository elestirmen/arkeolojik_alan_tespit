from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from veri_birlestir_rgb_dsm_dtm import _compose_rgb_dtm_dsm_5band


def _write_raster(path: Path, data: np.ndarray, *, crs: str = "EPSG:3857") -> None:
    if data.ndim == 2:
        count = 1
        height, width = data.shape
        write_data = data[np.newaxis, ...]
    else:
        count, height, width = data.shape
        write_data = data
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(write_data.dtype),
        transform=from_origin(0.0, float(height), 1.0, 1.0),
        crs=crs,
    ) as dst:
        dst.write(write_data)


def test_compose_outputs_dsm_on_band4_and_dtm_on_band5(tmp_path: Path) -> None:
    rgb = np.stack(
        [
            np.full((4, 4), 10, dtype=np.float32),
            np.full((4, 4), 20, dtype=np.float32),
            np.full((4, 4), 30, dtype=np.float32),
        ],
        axis=0,
    )
    dtm = np.full((4, 4), 111, dtype=np.float32)
    dsm = np.full((4, 4), 222, dtype=np.float32)

    rgb_path = tmp_path / "rgb.tif"
    dtm_path = tmp_path / "dtm.tif"
    dsm_path = tmp_path / "dsm.tif"
    out_path = tmp_path / "merged.tif"

    _write_raster(rgb_path, rgb)
    _write_raster(dtm_path, dtm)
    _write_raster(dsm_path, dsm)

    _compose_rgb_dtm_dsm_5band(
        rgb_input_path=rgb_path,
        dtm_input_path=dtm_path,
        dsm_input_path=dsm_path,
        output_path=out_path,
        output_nodata=-9999.0,
        compression="LZW",
        show_progress=False,
        block_size=256,
    )

    with rasterio.open(out_path) as src:
        assert src.count == 5
        assert np.allclose(src.read(1), 10)
        assert np.allclose(src.read(2), 20)
        assert np.allclose(src.read(3), 30)
        assert np.allclose(src.read(4), 222)
        assert np.allclose(src.read(5), 111)
