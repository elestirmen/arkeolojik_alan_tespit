from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

import vlm_lmstudio_detector as vlm


def test_auto_views_follow_available_bands():
    assert vlm._resolve_views(
        "auto",
        vlm.RasterBandLayout((1, 2, 3), None, None),
        logger=vlm.LOGGER,
    ) == ["rgb"]
    assert vlm._resolve_views(
        "auto",
        vlm.RasterBandLayout((1, 2, 3), 4, 5),
        logger=vlm.LOGGER,
    ) == ["rgb", "hillshade", "ndsm", "slope"]
    assert vlm._resolve_views(
        "auto",
        vlm.RasterBandLayout((1, 2, 3), None, 4),
        logger=vlm.LOGGER,
    ) == ["rgb", "hillshade", "slope"]
    assert vlm._resolve_views(
        "auto",
        vlm.RasterBandLayout((1, 2, 3), 4, None),
        logger=vlm.LOGGER,
    ) == ["rgb", "dsm"]


def test_auto_model_uses_loaded_lmstudio_model():
    class Model:
        id = "loaded-vision-model"

    class Models:
        data = [Model()]

    class Client:
        class models:
            @staticmethod
            def list():
                return Models()

    config = vlm.VlmLmStudioConfig(model="auto")

    assert vlm._resolve_lmstudio_model(Client(), config, logger=vlm.LOGGER) == "loaded-vision-model"


def test_base_url_without_v1_is_normalized():
    assert vlm._normalize_openai_base_url("http://127.0.0.1:8081") == "http://127.0.0.1:8081/v1"
    assert vlm._normalize_openai_base_url("http://127.0.0.1:8081/v1") == "http://127.0.0.1:8081/v1"


def test_prompt_includes_configured_gsd_scale():
    prompt = vlm._build_prompt(
        "rgb_only",
        ["rgb"],
        1024,
        1024,
        gsd_m=0.30,
    )

    assert "0.30 m ground sampling distance" in prompt
    assert "307 m x 307 m" in prompt
    assert "nadir imagery" in prompt


def test_wgs84_google_maps_fields_are_added_to_candidate_record():
    parsed = {
        "candidate": True,
        "confidence": 0.8,
        "candidate_type": "mound",
        "bbox_xyxy": [1, 1, 3, 3],
        "visual_evidence": "oval crop mark",
        "possible_false_positive": "soil variation",
        "recommended_check": "rgb",
    }
    base = {
        "tile_index": 1,
        "tile_row": 0,
        "tile_col": 0,
        "tile_width": 8,
        "tile_height": 8,
        "used_views": ["rgb"],
        "has_rgb": True,
        "has_dsm": False,
        "has_dtm": False,
        "analysis_mode": "rgb_only",
    }

    record = vlm._model_result_to_record(
        parsed=parsed,
        raw_response="{}",
        base_record=base,
        transform=from_origin(35.0, 39.0, 0.0001, 0.0001),
        crs=CRS.from_epsg(4326),
        raster_width=8,
        raster_height=8,
    )

    assert record["gps_lon"] is not None
    assert record["gps_lat"] is not None
    assert record["google_maps_url"].startswith("https://www.google.com/maps?q=")


def test_rgb_alpha_raster_is_not_treated_as_dsm(tmp_path: Path):
    tif_path = tmp_path / "rgba.tif"
    data = np.ones((4, 8, 8), dtype=np.uint8) * 255
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint8",
        transform=from_origin(0, 8, 1, 1),
    ) as dst:
        dst.write(data)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)

    with rasterio.open(tif_path) as src:
        layout = vlm._detect_band_layout(src, band_indexes=(1, 2, 3, 4, 5), logger=vlm.LOGGER)

    assert layout.has_rgb is True
    assert layout.has_dsm is False
    assert layout.has_dtm is False
    assert layout.analysis_mode == "rgb_only"


def test_empty_rgb_tile_is_skipped_before_model_call():
    rgb = [np.full((8, 8), np.nan, dtype=np.float32) for _ in range(3)]

    assert vlm._is_empty_rgb_tile(rgb) is True


def test_bad_json_response_is_recorded_with_raw_response(tmp_path: Path, monkeypatch):
    tif_path = tmp_path / "rgb.tif"
    data = np.ones((3, 8, 8), dtype=np.uint8) * 100
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        transform=from_origin(0, 8, 1, 1),
    ) as dst:
        dst.write(data)

    monkeypatch.setattr(vlm, "_make_openai_client", lambda config: object())
    monkeypatch.setattr(vlm, "_resolve_lmstudio_model", lambda client, config, logger: "loaded-vision-model")
    monkeypatch.setattr(vlm, "_request_vlm_json", lambda **kwargs: ("not json", True))

    summary = vlm.run_vlm_lmstudio_detection(
        input_path=tif_path,
        out_prefix=tmp_path / "out" / "rgb",
        config=vlm.VlmLmStudioConfig(tile=8, overlap=0, max_tiles=1),
    )

    assert summary.processed_tiles == 1
    assert summary.candidate_count == 0
    assert summary.error_count == 1
    assert summary.paths.jsonl.exists()
    assert summary.paths.csv.exists()
    assert summary.paths.xlsx.exists()
    assert summary.paths.geojson.exists()
    assert "raw_response" in summary.paths.raw_errors_jsonl.read_text(encoding="utf-8")
