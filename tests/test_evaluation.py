from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from evaluation import evaluate_predictions, evaluate_vectors

try:
    import geopandas  # noqa: F401

    HAS_GPD = True
except ImportError:
    HAS_GPD = False


def _write_single_band_raster(path: Path, arr: np.ndarray) -> None:
    meta = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": str(arr.dtype),
        "transform": from_origin(0.0, float(arr.shape[0]), 1.0, 1.0),
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr, 1)


def test_evaluate_predictions_auto_threshold_probability_map(tmp_path: Path) -> None:
    pred = np.array([[0.10, 0.20], [0.00, 0.90]], dtype=np.float32)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    pred_path = tmp_path / "pred_prob.tif"
    gt_path = tmp_path / "gt.tif"
    _write_single_band_raster(pred_path, pred)
    _write_single_band_raster(gt_path, gt)

    metrics = evaluate_predictions(pred_path, gt_path, threshold=None)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.true_negatives == 3


@pytest.mark.skipif(not HAS_GPD, reason="geopandas not installed")
def test_evaluate_vectors_coarse_resolution_is_safe(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    pred_path = tmp_path / "pred.gpkg"
    gt_path = tmp_path / "gt.gpkg"
    gdf.to_file(pred_path, driver="GPKG")
    gdf.to_file(gt_path, driver="GPKG")

    metrics = evaluate_vectors(pred_path, gt_path, rasterize_resolution=10.0)
    assert metrics.iou > 0.99


@pytest.mark.skipif(not HAS_GPD, reason="geopandas not installed")
def test_evaluate_vectors_rejects_non_positive_resolution(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    pred_path = tmp_path / "pred.gpkg"
    gt_path = tmp_path / "gt.gpkg"
    gdf.to_file(pred_path, driver="GPKG")
    gdf.to_file(gt_path, driver="GPKG")

    with pytest.raises(ValueError, match="rasterize_resolution must be > 0"):
        evaluate_vectors(pred_path, gt_path, rasterize_resolution=0.0)
