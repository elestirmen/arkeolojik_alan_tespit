from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import topo_harita_uret as topo


def _write_dem(path: Path, *, count: int = 1) -> None:
    base = np.linspace(0, 20, 32 * 32, dtype=np.float32).reshape(32, 32)
    if count == 1:
        data = base[np.newaxis, ...]
    else:
        data = np.zeros((count, 32, 32), dtype=np.float32)
        for idx in range(count):
            data[idx] = base + idx
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=count,
        dtype="float32",
        transform=from_origin(500000.0, 4200000.0, 1.0, 1.0),
        crs="EPSG:32636",
        nodata=np.nan,
    ) as dst:
        dst.write(data)


def test_parse_products_expands_all_and_deduplicates() -> None:
    assert topo.parse_products("hillshade,svf,all,hs") == ("hillshade", "svf", "slrm")


def test_rvt_slrm_radius_is_normalized_to_supported_cell_range(monkeypatch) -> None:
    class DummyRvt:
        @staticmethod
        def slrm(**_kwargs):
            return np.zeros((1, 1), dtype=np.float32)

    monkeypatch.setattr(topo, "rvt_vis", DummyRvt)

    effective_m, requested_cell, effective_cell = topo._resolve_rvt_slrm_radius(10.0, 0.061)

    assert requested_cell == 164
    assert effective_cell == 50
    assert np.isclose(effective_m, 3.05)


def test_parser_uses_top_level_config_defaults(monkeypatch) -> None:
    monkeypatch.setitem(topo.CONFIG, "input", "configured_dem.tif")
    monkeypatch.setitem(topo.CONFIG, "elevation_band", 5)
    monkeypatch.setitem(topo.CONFIG, "output_dir", "configured_out")
    monkeypatch.setitem(topo.CONFIG, "prefix", "configured_prefix")
    monkeypatch.setitem(topo.CONFIG, "products", "hillshade,slrm")
    monkeypatch.setitem(topo.CONFIG, "workers", 7)
    monkeypatch.setitem(topo.CONFIG, "overwrite", True)

    args = topo.build_arg_parser().parse_args([])
    config = topo.config_from_args(args)

    assert config.input_path == Path("configured_dem.tif")
    assert config.elevation_band == 5
    assert config.output_dir == Path("configured_out")
    assert config.prefix == "configured_prefix"
    assert config.products == ("hillshade", "slrm")
    assert config.workers == 7
    assert config.overwrite is True


def test_cli_args_override_top_level_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(topo.CONFIG, "input", "configured_dem.tif")
    monkeypatch.setitem(topo.CONFIG, "elevation_band", 5)

    cli_input = tmp_path / "cli_dem.tif"
    args = topo.build_arg_parser().parse_args(
        [
            "--input",
            str(cli_input),
            "--dtm-band",
            "1",
            "--products",
            "slope",
            "--workers",
            "2",
            "--no-overwrite",
        ]
    )
    config = topo.config_from_args(args)

    assert config.input_path == cli_input
    assert config.elevation_band == 1
    assert config.products == ("slope",)
    assert config.workers == 2
    assert config.overwrite is False


def test_generate_separate_hillshade_and_slrm_maps(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path)

    outputs = topo.generate_topo_maps(
        topo.TopoMapConfig(
            input_path=dem_path,
            output_dir=tmp_path / "out",
            prefix="demo",
            products=("hillshade", "slrm"),
            chunk=16,
            overwrite=True,
        )
    )

    assert set(outputs) == {"hillshade", "slrm"}
    for product, path in outputs.items():
        assert path.name == f"demo_{product}.tif"
        with rasterio.open(path) as src:
            arr = src.read(1)
            assert src.count == 1
            assert src.width == 32
            assert src.height == 32
            assert src.crs.to_string() == "EPSG:32636"
            assert np.isfinite(arr).any()


def test_elevation_band_can_target_dtm_band_in_multiband_raster(tmp_path: Path) -> None:
    dem_path = tmp_path / "five_band.tif"
    _write_dem(dem_path, count=5)

    outputs = topo.generate_topo_maps(
        topo.TopoMapConfig(
            input_path=dem_path,
            elevation_band=5,
            output_dir=tmp_path / "out",
            products=("slope",),
            chunk=20,
            overwrite=True,
        )
    )

    with rasterio.open(outputs["slope"]) as src:
        assert src.read(1).shape == (32, 32)
