"""
DSM rasterini veya LAS/LAZ nokta bulutunu PDAL SMRF ile DTM'ye ceviren on-isleme araci.

Not:
- Bu betik PDAL Python bagimliligina ihtiyac duyar (`pip install pdal`).
- Raster girdide cikti son adimda kaynak DSM'in gridine zorla hizalanir;
  boyut, transform ve cozumunurluk birebir ayni olur.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.warp import reproject

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

LOGGER = logging.getLogger("on_isleme")
DEFAULT_NODATA = -9999.0
_TQDM_WARNED = False
_LAS_DEFAULT_CELL = 0.5

# ==================== CONFIG ====================
# Buradaki degerler komut satirindan arguman verilmezse varsayilan olarak kullanilir.
# Komut satiri argumanlari her zaman bu degerleri ezer.
CONFIG: dict[str, Any] = {
    # Girdi DSM GeoTIFF veya LAS/LAZ yolu.
    "input": "/veri/karlik_dag_dsm.tif",
    # Uretilecek DTM GeoTIFF yolu.
    "output": "/veri/karlik_dag_dtm_smrf.tif",
    # Isleme yontemi:
    # - "auto": Once SMRF dener, hata olursa (allow_fallback=True ise) fallback'e gecer.
    # - "smrf": Yalnizca SMRF (hata olursa durur).
    # - "fallback": SMRF'i atlar, dogrudan fallback calisir.
    # Not: Bu projede bircok sahada fallback kalitesi daha iyi oldugu icin varsayilan fallback.
    "method": "auto",
    # SMRF cell boyutu (metre). None -> raster girdide piksel boyutu, LAS/LAZ girdide
    # nokta yogunlugundan otomatik tahmin (hesaplanamazsa _LAS_DEFAULT_CELL).
    "cell": None,
    # SMRF slope: buyudukce zemin siniflamasi daha toleransli olur.
    "slope": 0.2,
    # SMRF threshold: zemin/saha ayrimi icin yukseklik fark esigi.
    "threshold": 0.45,
    # SMRF window: maksimum pencere boyutu (metre).
    "window": 16.0,
    # SMRF scalar: karar mekanizmasi carpan katsayisi.
    "scalar": 1.25,
    # >0 ise girdi bu piksel sayisini asarsa SMRF oncesi downsample uygulanir.
    # Kalite kritikse 0 yaparak otomatik downsample'i kapat.
    "smrf_max_pixels": 120_000_000,
    # Zorunlu downsample katsayisi: 1.0 kapali, 2.0 -> en/boy yariya iner.
    # Kalite kritikse 1.0 kullan.
    "smrf_downsample_factor": 1.0,
    # True ise SMRF islemi tum raster yerine ortusmeli karolarla yapilir.
    # Buyuk dosyalarda RAM'i ciddi dusurur; kaliteyi korumak icin onerilir.
    "smrf_tiled": True,
    # SMRF tiled modunda karo boyutu (piksel). Buyudukce hiz artar, RAM artar.
    "smrf_tile_size": 4096,
    # SMRF tiled modunda karo bindirme miktari (piksel).
    # 0 ise otomatik hesaplanir (window/cell tabanli).
    "smrf_overlap_px": 0,
    # SMRF hata verirse fallback (morfolojik DTM) kullanilsin mi?
    "allow_fallback": True,
    # Fallback morfolojik acma pencere boyutu (metre).
    "opening_meters": 6.0,
    # Fallback gaussian yumusatma sigma degeri (piksel).
    "smooth_sigma_px": 1.5,
    # Fallback karo (tile) boyutu (piksel).
    "tile_size": 2048,
    # Cikti nodata degeri.
    "nodata": -9999.0,
    # GeoTIFF sikistirma tipi: LZW | DEFLATE | NONE.
    "compression": "LZW",
    # Log seviyesi: DEBUG | INFO | WARNING | ERROR.
    "log_level": "INFO",
    # Ilerleme cubuklari acik/kapali.
    "progress": True,
}
# ===============================================


@dataclass(frozen=True)
class SmrfParams:
    """
    PDAL `filters.smrf` parametreleri.

    cell:
        Izgara hucre boyutu (metre). Buyuk deger daha az RAM/CPU, ama daha az detay.
    slope:
        Eğim toleransi. Buyudukce zemin siniflamasi daha yumusak olur.
    threshold:
        Yukseklik fark esigi. Buyuk degerde daha fazla nokta zemin kalabilir.
    window:
        Maksimum pencere boyutu (metre). Buyuk yapilarin zemin kabulune etki eder.
    scalar:
        SMRF karar mekanizmasi carpan katsayisi.
    """
    cell: float
    slope: float
    threshold: float
    window: float
    scalar: float


class _SimpleProgress:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.postfix = ""
        self._last_draw = 0.0
        self._draw(force=True)

    def update(self, value: int) -> None:
        if value <= 0:
            return
        self.current = min(self.total, self.current + int(value))
        now = time.time()
        if now - self._last_draw >= 0.2 or self.current >= self.total:
            self._draw(force=False)

    def close(self) -> None:
        self._draw(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def set_postfix_str(self, text: str, refresh: bool = False) -> None:
        self.postfix = text.strip()
        if refresh:
            self._draw(force=False)

    def _draw(self, force: bool) -> None:
        ratio = self.current / self.total
        pct = int(ratio * 100)
        bar_w = 28
        fill = int(bar_w * ratio)
        bar = "#" * fill + "-" * (bar_w - fill)
        postfix = f" | {self.postfix}" if self.postfix else ""
        sys.stderr.write(
            f"\r{self.desc}: [{bar}] {pct:3d}% ({self.current}/{self.total} {self.unit}){postfix}"
        )
        sys.stderr.flush()
        if force:
            self._last_draw = time.time()


def _normalize_path(raw_path: str) -> Path:
    """'/veri/...' gibi path'leri proje-kokune gore normalize eder."""
    value = raw_path.strip()
    if not value:
        raise ValueError("Bos path verilemez.")
    has_drive = len(value) >= 2 and value[1] == ":"
    if value[0] in ("/", "\\") and not has_drive:
        return Path(value.lstrip("/\\"))
    return Path(value)


def _build_pipeline(
    input_path: Path,
    temp_output_path: Path,
    params: SmrfParams,
    nodata: float,
    bounds: rasterio.coords.BoundingBox,
    source_dimension: str,
) -> list[dict[str, Any]]:
    bounds_str = f"([{bounds.left},{bounds.right}],[{bounds.bottom},{bounds.top}])"
    stages: list[dict[str, Any]] = [
        {
            "type": "readers.gdal",
            "filename": str(input_path),
        }
    ]
    if source_dimension != "Z":
        stages.append(
            {
                "type": "filters.ferry",
                "dimensions": f"{source_dimension}=>Z",
            }
        )
    stages.extend(
        [
        {
            "type": "filters.smrf",
            "cell": params.cell,
            "slope": params.slope,
            "threshold": params.threshold,
            "window": params.window,
            "scalar": params.scalar,
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]",
        },
        {
            "type": "writers.gdal",
            "filename": str(temp_output_path),
            "gdaldriver": "GTiff",
            "data_type": "float32",
            "dimension": "Z",
            "output_type": "idw",
            "resolution": params.cell,
            "radius": params.cell * 1.5,
            "nodata": nodata,
            "bounds": bounds_str,
        },
    ]
    )
    return stages


def _build_las_pipeline(
    input_path: Path,
    temp_output_path: Path,
    params: SmrfParams,
    nodata: float,
    compression: str,
) -> list[dict[str, Any]]:
    gdalopts: list[str] = []
    comp = str(compression).upper()
    if comp in {"LZW", "DEFLATE"}:
        gdalopts.append(f"COMPRESS={comp}")
    if comp != "NONE":
        gdalopts.append("PREDICTOR=3")

    writer_stage: dict[str, Any] = {
        "type": "writers.gdal",
        "filename": str(temp_output_path),
        "gdaldriver": "GTiff",
        "data_type": "float32",
        "dimension": "Z",
        "output_type": "idw",
        "resolution": params.cell,
        "radius": params.cell * 1.5,
        "nodata": nodata,
    }
    if gdalopts:
        writer_stage["gdalopts"] = ",".join(gdalopts)

    stages: list[dict[str, Any]] = [
        {
            "type": "readers.las",
            "filename": str(input_path),
        },
        {
            "type": "filters.smrf",
            "cell": params.cell,
            "slope": params.slope,
            "threshold": params.threshold,
            "window": params.window,
            "scalar": params.scalar,
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]",
        },
        writer_stage,
    ]
    return stages


def _estimate_las_cell_from_header(input_las_path: Path, fallback_cell: float) -> tuple[float, str]:
    """
    LAS/LAZ header bilgisinden ortalama XY nokta araligini tahmin eder.

    Yaklasim:
    - point_count: header'daki toplam nokta sayisi
    - area: (max_x - min_x) * (max_y - min_y)
    - avg_spacing ~= sqrt(area / point_count)
    """
    try:
        with input_las_path.open("rb") as fh:
            header = fh.read(375)
    except OSError as exc:
        return fallback_cell, f"header okunamadi ({exc})"

    if len(header) < 227:
        return fallback_cell, "header cok kisa"
    if header[0:4] != b"LASF":
        return fallback_cell, "LAS imzasi bulunamadi"

    try:
        version_major = int(header[24])
        version_minor = int(header[25])

        legacy_count = int(struct.unpack_from("<I", header, 107)[0])
        point_count = legacy_count
        if (version_major > 1 or (version_major == 1 and version_minor >= 4)) and len(header) >= 255:
            extended_count = int(struct.unpack_from("<Q", header, 247)[0])
            if extended_count > 0:
                point_count = extended_count

        scale_x, scale_y, _ = struct.unpack_from("<ddd", header, 131)
        max_x, min_x, max_y, min_y, _max_z, _min_z = struct.unpack_from("<dddddd", header, 179)

        span_x = float(max_x - min_x)
        span_y = float(max_y - min_y)
        if point_count <= 0:
            return fallback_cell, "nokta sayisi header'da sifir"
        if span_x <= 0 or span_y <= 0:
            return fallback_cell, "XY kapsami gecersiz"

        area = span_x * span_y
        avg_spacing = math.sqrt(area / float(point_count))
        if not math.isfinite(avg_spacing) or avg_spacing <= 0:
            return fallback_cell, "ortalama nokta araligi hesaplanamadi"

        # Cell, koordinat nicemleme adimindan kucuk olmamali.
        quant_step = max(abs(float(scale_x)), abs(float(scale_y)), 1e-6)
        est_cell = max(float(avg_spacing), quant_step)
        return est_cell, f"otomatik tahmin (n={point_count}, span=({span_x:.1f}m,{span_y:.1f}m))"
    except (ValueError, struct.error) as exc:
        return fallback_cell, f"header parse hatasi ({exc})"


def _run_pdal_pipeline(
    pipeline: list[dict[str, Any]],
    show_progress: bool,
    expected_points: Optional[int],
    log_completion: bool = True,
    status_label: Optional[str] = None,
    log_interval_sec: float = 60.0,
) -> None:
    try:
        import pdal
    except ImportError as exc:
        raise RuntimeError(
            "PDAL Python modulu bulunamadi. Kurulum: `pip install pdal` "
            "(Windows icin gerekirse conda-forge PDAL kurulumu gerekebilir)."
        ) from exc

    pipeline_json = json.dumps(pipeline)
    runner = pdal.Pipeline(pipeline_json)

    writer_path: Optional[Path] = None
    for stage in pipeline:
        if stage.get("type") == "writers.gdal":
            filename = stage.get("filename")
            if isinstance(filename, str) and filename.strip():
                writer_path = Path(filename)
                break

    expected_bytes: Optional[float] = None
    if expected_points is not None and expected_points > 0:
        expected_bytes = float(expected_points) * 4.0

    result: dict[str, int] = {}
    error: dict[str, Exception] = {}

    def _worker() -> None:
        try:
            result["count"] = int(runner.execute())
        except Exception as exc:  # pragma: no cover - passthrough
            error["exc"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    pbar = _open_progress(total=100, desc="SMRF (PDAL)", unit="%", enabled=show_progress)
    progress_value = 0
    last_log_elapsed = 0.0
    start_time = time.time()
    if show_progress:
        LOGGER.info(
            "SMRF gorunen ilerleme tahminidir; PDAL son asamada uzun sure 90-95%% araliginda kalabilir."
        )

    while worker.is_alive():
        elapsed = time.time() - start_time
        target_value = progress_value
        out_bytes: Optional[int] = None
        if writer_path is not None and expected_bytes is not None and expected_bytes > 0 and writer_path.exists():
            try:
                out_bytes = int(writer_path.stat().st_size)
                size_ratio = out_bytes / expected_bytes
                target_value = min(95, max(progress_value, int(size_ratio * 100)))
            except OSError:
                pass
        else:
            target_value = min(90, max(progress_value, int(elapsed // 4) + 1))

        if target_value > progress_value:
            if pbar is not None:
                pbar.update(target_value - progress_value)
            progress_value = target_value

        spinner = "|/-\\"[int(elapsed) % 4]
        postfix_parts = [f"elapsed={int(elapsed)}s"]
        if out_bytes is not None:
            postfix_parts.append(f"out={out_bytes / (1024 * 1024):.1f}MB")
        if progress_value >= 90:
            postfix_parts.append(f"finalizing {spinner}")
        _set_progress_postfix(pbar, " ".join(postfix_parts))

        if elapsed - last_log_elapsed >= float(log_interval_sec):
            prefix = f"{status_label} | " if status_label else ""
            LOGGER.info(
                "%sSMRF devam ediyor... %.0f sn (gorunen ilerleme: %d%%)",
                prefix,
                elapsed,
                progress_value,
            )
            last_log_elapsed = elapsed
        time.sleep(1.0)

    worker.join()
    if pbar is not None and progress_value < 100:
        pbar.update(100 - progress_value)
        pbar.close()

    if "exc" in error:
        raise RuntimeError(str(error["exc"])) from error["exc"]

    if log_completion:
        point_count = int(result.get("count", 0))
        LOGGER.info("PDAL pipeline calisti. Islenen nokta sayisi: %s", point_count)


def _detect_pdal_height_dimension(input_path: Path) -> str:
    try:
        import pdal
    except ImportError as exc:
        raise RuntimeError(
            "PDAL Python modulu bulunamadi. Kurulum: `pip install pdal` "
            "(Windows icin gerekirse conda-forge PDAL kurulumu gerekebilir)."
        ) from exc

    probe_pipeline = pdal.Pipeline(
        json.dumps(
            [
                {"type": "readers.gdal", "filename": str(input_path)},
                {"type": "filters.head", "count": 1},
            ]
        )
    )
    probe_pipeline.execute()
    arrays = probe_pipeline.arrays
    if not arrays:
        raise RuntimeError("PDAL readers.gdal rasterdan veri okumadi.")

    names = arrays[0].dtype.names or ()
    preferred = ("Z", "band_1", "band-1", "elevation", "Elevation", "Intensity")
    for name in preferred:
        if name in names:
            return name
    for name in names:
        if name not in {"X", "Y"}:
            return name
    raise RuntimeError(f"Yukseklik boyutu tespit edilemedi. Mevcut boyutlar: {names}")


def _open_progress(total: int, desc: str, unit: str, enabled: bool) -> Optional[Any]:
    global _TQDM_WARNED
    if not enabled:
        return None
    if tqdm is None:
        if not _TQDM_WARNED:
            LOGGER.warning("tqdm bulunamadi; dahili terminal progress kullanilacak.")
            _TQDM_WARNED = True
        return _SimpleProgress(total=total, desc=desc, unit=unit)
    return tqdm(total=total, desc=desc, unit=unit, leave=True, dynamic_ncols=True)


def _set_progress_postfix(pbar: Optional[Any], text: str) -> None:
    if pbar is None:
        return
    if not text:
        return
    try:
        pbar.set_postfix_str(text, refresh=False)
    except Exception:
        pass


def _prepare_temp_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.stem}_tmp_",
        suffix=output_path.suffix,
        dir=str(output_path.parent),
    )
    os.close(fd)
    return Path(temp_name)


def _run_las_smrf(
    input_las_path: Path,
    output_dtm_path: Path,
    params: SmrfParams,
    nodata: float,
    compression: str,
    show_progress: bool,
) -> None:
    temp_output_path = _prepare_temp_output_path(output_dtm_path)
    try:
        pipeline = _build_las_pipeline(
            input_path=input_las_path,
            temp_output_path=temp_output_path,
            params=params,
            nodata=nodata,
            compression=compression,
        )
        _run_pdal_pipeline(
            pipeline=pipeline,
            show_progress=show_progress,
            expected_points=None,
        )
        os.replace(temp_output_path, output_dtm_path)
    finally:
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except OSError:
                pass


def _prepare_smrf_input(
    input_path: Path,
    temp_dir: Path,
    max_pixels: Optional[int],
    downsample_factor: float,
    show_progress: bool,
) -> tuple[Path, int, float]:
    """
    SMRF icin girdiyi gerekirse downsample eder.

    Donus:
    - smrf_input_path: PDAL'e verilecek raster yolu
    - expected_points: SMRF tarafinda islenecek yaklasik nokta/piksel sayisi
    - smrf_base_cell: SMRF girdisinin piksel boyutu (metre)
    """
    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError(f"Girdi raster en az bir bant icermeli: {input_path}")
        src_points = int(src.width * src.height)
        xres, yres = src.res
        src_cell = max(abs(float(xres)), abs(float(yres)))

        factor = max(1.0, float(downsample_factor))
        if max_pixels is not None and max_pixels > 0 and src_points > max_pixels:
            auto_factor = math.sqrt(src_points / float(max_pixels))
            factor = max(factor, auto_factor)

        if factor <= 1.0001:
            return input_path, src_points, src_cell

        dst_width = max(1, int(round(src.width / factor)))
        dst_height = max(1, int(round(src.height / factor)))
        dst_points = int(dst_width * dst_height)
        actual_factor = math.sqrt(src_points / max(dst_points, 1))
        dst_cell = src_cell * actual_factor
        dst_transform = from_bounds(*src.bounds, dst_width, dst_height)

        smrf_input_path = temp_dir / "smrf_input_downsampled.tif"
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            width=dst_width,
            height=dst_height,
            transform=dst_transform,
            tiled=False,
            BIGTIFF="IF_SAFER",
        )
        if "compress" not in profile or not profile["compress"]:
            profile["compress"] = "LZW"
        if str(profile.get("dtype", "")).lower().startswith("float"):
            profile["predictor"] = 3

        LOGGER.warning(
            "SMRF bellek korumasi: girdi downsample uygulanacak. "
            "piksel=%d -> %d (%.2fx azalma), cell~%.4f -> %.4f",
            src_points,
            dst_points,
            src_points / max(dst_points, 1),
            src_cell,
            dst_cell,
        )

        pbar = _open_progress(
            total=src.count,
            desc="SMRF input downsample",
            unit="band",
            enabled=show_progress,
        )
        try:
            with rasterio.open(smrf_input_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        dst_nodata=src.nodata,
                        resampling=Resampling.bilinear,
                        num_threads=2,
                    )
                    if pbar is not None:
                        pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        return smrf_input_path, dst_points, dst_cell


def _compute_smrf_tiled(
    input_dsm_path: Path,
    output_dtm_path: Path,
    params: SmrfParams,
    nodata: float,
    compression: str,
    source_dimension: str,
    tile_size: int,
    overlap_px: int,
    show_progress: bool,
    temp_dir: Path,
) -> None:
    """
    RAM'i sabit tutmak icin SMRF'i ortusmeli karolarda calistirir.

    Her karoda pad (bindirme) bolgesiyle SMRF uygulanir, sadece cekirdek pencere
    global ciktiya yazilir. Bu yaklasim, downsample etmeden kaliteyi korumayi
    hedefler.
    """
    overlap_px = max(0, int(overlap_px))
    tile_size = max(128, int(tile_size))

    with rasterio.open(input_dsm_path) as src:
        if src.count < 1:
            raise ValueError(f"Girdi raster en az bir bant icermeli: {input_dsm_path}")
        if src.crs is None:
            raise ValueError(f"Girdi raster CRS icermiyor: {input_dsm_path}")

        total_rows = src.height
        total_cols = src.width
        tiles_y = (total_rows + tile_size - 1) // tile_size
        tiles_x = (total_cols + tile_size - 1) // tile_size
        total_tiles = tiles_y * tiles_x

        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            dtype="float32",
            count=1,
            nodata=nodata,
            compress=compression,
            predictor=3,
            tiled=False,
            BIGTIFF="IF_SAFER",
        )

        LOGGER.info(
            "SMRF tiled basladi: tile=%d overlap=%d toplam_karo=%d",
            tile_size,
            overlap_px,
            total_tiles,
        )

        pbar = _open_progress(
            total=total_tiles,
            desc="SMRF (PDAL tiled)",
            unit="tile",
            enabled=show_progress,
        )
        tile_idx = 0
        tiled_start = time.time()

        try:
            with rasterio.open(output_dtm_path, "w", **profile) as dst:
                for row0 in range(0, total_rows, tile_size):
                    row1 = min(row0 + tile_size, total_rows)
                    for col0 in range(0, total_cols, tile_size):
                        col1 = min(col0 + tile_size, total_cols)
                        tile_idx += 1

                        core_h = row1 - row0
                        core_w = col1 - col0
                        core_window = Window(col0, row0, core_w, core_h)

                        pad_row0 = max(0, row0 - overlap_px)
                        pad_row1 = min(total_rows, row1 + overlap_px)
                        pad_col0 = max(0, col0 - overlap_px)
                        pad_col1 = min(total_cols, col1 + overlap_px)
                        pad_h = pad_row1 - pad_row0
                        pad_w = pad_col1 - pad_col0
                        pad_window = Window(pad_col0, pad_row0, pad_w, pad_h)

                        tile_input_path = temp_dir / f"smrf_tile_{tile_idx:06d}_in.tif"
                        tile_raw_path = temp_dir / f"smrf_tile_{tile_idx:06d}_raw.tif"

                        arr_ma = src.read(1, window=pad_window, masked=True)
                        src_nodata = src.nodata if src.nodata is not None else nodata
                        arr = np.ma.filled(arr_ma.astype(np.float32), src_nodata)
                        tile_transform = src.window_transform(pad_window)
                        tile_bounds_tuple = src.window_bounds(pad_window)
                        tile_bounds = rasterio.coords.BoundingBox(
                            left=tile_bounds_tuple[0],
                            bottom=tile_bounds_tuple[1],
                            right=tile_bounds_tuple[2],
                            top=tile_bounds_tuple[3],
                        )

                        tile_profile = src.profile.copy()
                        tile_profile.pop("blockxsize", None)
                        tile_profile.pop("blockysize", None)
                        tile_profile.update(
                            width=pad_w,
                            height=pad_h,
                            transform=tile_transform,
                            count=1,
                            dtype="float32",
                            nodata=src_nodata,
                            compress="LZW",
                            predictor=3,
                            tiled=False,
                            BIGTIFF="IF_SAFER",
                        )
                        with rasterio.open(tile_input_path, "w", **tile_profile) as tile_dst:
                            tile_dst.write(arr, 1)
                            tile_mask = src.read_masks(1, window=pad_window)
                            tile_dst.write_mask(tile_mask)

                        pipeline = _build_pipeline(
                            input_path=tile_input_path,
                            temp_output_path=tile_raw_path,
                            params=params,
                            nodata=nodata,
                            bounds=tile_bounds,
                            source_dimension=source_dimension,
                        )
                        _set_progress_postfix(pbar, f"tile={tile_idx}/{total_tiles} running")
                        _run_pdal_pipeline(
                            pipeline=pipeline,
                            show_progress=False,
                            expected_points=pad_w * pad_h,
                            log_completion=False,
                            status_label=f"SMRF tiled karo {tile_idx}/{total_tiles}",
                            log_interval_sec=30.0,
                        )

                        core_arr = np.full((core_h, core_w), nodata, dtype=np.float32)
                        with rasterio.open(tile_raw_path) as tile_raw:
                            reproject(
                                source=rasterio.band(tile_raw, 1),
                                destination=core_arr,
                                src_transform=tile_raw.transform,
                                src_crs=tile_raw.crs if tile_raw.crs is not None else src.crs,
                                src_nodata=tile_raw.nodata if tile_raw.nodata is not None else nodata,
                                dst_transform=src.window_transform(core_window),
                                dst_crs=src.crs,
                                dst_nodata=nodata,
                                resampling=Resampling.bilinear,
                                num_threads=2,
                            )

                        dst.write(core_arr, 1, window=core_window)
                        core_mask = src.read_masks(1, window=core_window)
                        dst.write_mask(core_mask, window=core_window)

                        try:
                            tile_input_path.unlink(missing_ok=True)
                            tile_raw_path.unlink(missing_ok=True)
                        except OSError:
                            pass

                        if pbar is not None:
                            pbar.update(1)
                            elapsed = max(1e-6, time.time() - tiled_start)
                            avg_tile = elapsed / max(tile_idx, 1)
                            remaining = max(0, total_tiles - tile_idx)
                            eta_sec = int(round(avg_tile * remaining))
                            _set_progress_postfix(
                                pbar,
                                f"avg={avg_tile:.2f}s/tile eta={eta_sec}s",
                            )
                        elif tile_idx % 10 == 0 or tile_idx == total_tiles:
                            LOGGER.info("SMRF tiled ilerleme: %d/%d karo", tile_idx, total_tiles)
        finally:
            if pbar is not None:
                pbar.close()


def _compute_fallback_dtm(
    input_dsm_path: Path,
    output_dtm_path: Path,
    nodata: float,
    compression: str,
    opening_meters: float,
    smooth_sigma_px: float,
    tile_size: int,
    show_progress: bool,
) -> None:
    try:
        from scipy.ndimage import gaussian_filter, grey_opening
    except ImportError as exc:
        raise RuntimeError(
            "Fallback metod icin scipy gerekli. Kurulum: `pip install scipy` "
            "veya conda ortaminda `conda install scipy`."
        ) from exc

    with rasterio.open(input_dsm_path) as src:
        if src.count < 1:
            raise ValueError(f"Girdi raster en az bir bant icermeli: {input_dsm_path}")
        xres, yres = src.res
        pixel_size = max(abs(float(xres)), abs(float(yres)))
        opening_px = max(3, int(round(opening_meters / max(pixel_size, 1e-6))))
        if opening_px % 2 == 0:
            opening_px += 1
        overlap = max(1, opening_px // 2 + int(math.ceil(3.0 * smooth_sigma_px)))

        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            dtype="float32",
            count=1,
            nodata=nodata,
            compress=compression,
            predictor=3,
            tiled=False,
            BIGTIFF="IF_SAFER",
        )
        temp_output_path = _prepare_temp_output_path(output_dtm_path)

        total_rows = src.height
        total_cols = src.width
        tiles_y = (total_rows + tile_size - 1) // tile_size
        tiles_x = (total_cols + tile_size - 1) // tile_size
        total_tiles = tiles_y * tiles_x
        LOGGER.info(
            "Fallback DTM (morph) basladi: opening_px=%d sigma=%.2f overlap=%d tile=%d toplam_karo=%d",
            opening_px,
            smooth_sigma_px,
            overlap,
            tile_size,
            total_tiles,
        )

        tile_idx = 0
        pbar = _open_progress(
            total=total_tiles,
            desc="Fallback DTM",
            unit="tile",
            enabled=show_progress,
        )
        fallback_start = time.time()
        try:
            with rasterio.open(temp_output_path, "w", **profile) as dst:
                for row0 in range(0, total_rows, tile_size):
                    row1 = min(row0 + tile_size, total_rows)
                    for col0 in range(0, total_cols, tile_size):
                        col1 = min(col0 + tile_size, total_cols)
                        tile_idx += 1

                        core_h = row1 - row0
                        core_w = col1 - col0
                        core_window = Window(col0, row0, core_w, core_h)

                        pad_row0 = max(0, row0 - overlap)
                        pad_row1 = min(total_rows, row1 + overlap)
                        pad_col0 = max(0, col0 - overlap)
                        pad_col1 = min(total_cols, col1 + overlap)
                        pad_h = pad_row1 - pad_row0
                        pad_w = pad_col1 - pad_col0
                        pad_window = Window(pad_col0, pad_row0, pad_w, pad_h)

                        arr_ma = src.read(1, window=pad_window, masked=True)
                        arr = np.ma.filled(arr_ma.astype(np.float32), np.nan)
                        valid = np.isfinite(arr)

                        if np.any(valid):
                            safe_fill = float(np.nanmax(arr[valid]) + 100.0)
                        else:
                            safe_fill = 0.0
                        work = np.where(valid, arr, safe_fill).astype(np.float32, copy=False)

                        opened = grey_opening(work, size=(opening_px, opening_px)).astype(np.float32, copy=False)
                        if smooth_sigma_px > 0:
                            opened = gaussian_filter(opened, sigma=smooth_sigma_px).astype(np.float32, copy=False)

                        dtm_pad = np.where(valid, np.minimum(opened, arr), np.nan).astype(np.float32, copy=False)
                        rs = row0 - pad_row0
                        cs = col0 - pad_col0
                        dtm_core = dtm_pad[rs : rs + core_h, cs : cs + core_w]
                        out = np.where(np.isfinite(dtm_core), dtm_core, nodata).astype(np.float32, copy=False)

                        dst.write(out, 1, window=core_window)
                        src_mask = src.read_masks(1, window=core_window)
                        dst.write_mask(src_mask, window=core_window)

                        if pbar is not None:
                            pbar.update(1)
                            elapsed = max(1e-6, time.time() - fallback_start)
                            avg_tile = elapsed / max(tile_idx, 1)
                            remaining = max(0, total_tiles - tile_idx)
                            eta_sec = int(round(avg_tile * remaining))
                            _set_progress_postfix(
                                pbar,
                                f"avg={avg_tile:.2f}s/tile eta={eta_sec}s",
                            )
                        elif tile_idx % 25 == 0 or tile_idx == total_tiles:
                            LOGGER.info("Fallback DTM ilerleme: %d/%d karo", tile_idx, total_tiles)
            os.replace(temp_output_path, output_dtm_path)
        finally:
            if pbar is not None:
                pbar.close()
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except OSError:
                    pass


def _snap_to_reference_grid(
    reference_dsm_path: Path,
    raw_dtm_path: Path,
    output_dtm_path: Path,
    nodata: float,
    compression: str,
    show_progress: bool,
) -> None:
    with rasterio.open(reference_dsm_path) as ref, rasterio.open(raw_dtm_path) as raw:
        if ref.crs is None:
            raise ValueError(f"Girdi raster CRS icermiyor: {reference_dsm_path}")
        src_crs = raw.crs if raw.crs is not None else ref.crs
        src_nodata = raw.nodata if raw.nodata is not None else nodata

        profile = ref.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            count=1,
            dtype="float32",
            nodata=nodata,
            compress=compression,
            predictor=3,
            tiled=False,
            BIGTIFF="IF_SAFER",
        )

        temp_output_path = _prepare_temp_output_path(output_dtm_path)
        pbar = None
        try:
            with rasterio.open(temp_output_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(raw, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=raw.transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=nodata,
                    resampling=Resampling.bilinear,
                    num_threads=2,
                )
                block_rows, block_cols = ref.block_shapes[0]
                total_blocks = math.ceil(ref.height / block_rows) * math.ceil(ref.width / block_cols)
                pbar = _open_progress(
                    total=total_blocks,
                    desc="Grid maske kopyalama",
                    unit="block",
                    enabled=show_progress,
                )
                for idx, (_, window) in enumerate(ref.block_windows(1), start=1):
                    mask = ref.read_masks(1, window=window)
                    dst.write_mask(mask, window=window)
                    if pbar is not None:
                        pbar.update(1)
                    elif idx % 500 == 0 or idx == total_blocks:
                        LOGGER.info("Grid maske kopyalama: %d/%d blok", idx, total_blocks)
            os.replace(temp_output_path, output_dtm_path)
        finally:
            if pbar is not None:
                pbar.close()
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except OSError:
                    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DSM rasterini veya LAS/LAZ nokta bulutunu SMRF (PDAL) ile DTM'ye cevirir. "
            "Raster girdide cikti kaynak rasterin ayni piksel gridine hizalanir."
        )
    )
    parser.set_defaults(
        allow_fallback=bool(CONFIG["allow_fallback"]),
        progress=bool(CONFIG["progress"]),
        smrf_tiled=bool(CONFIG["smrf_tiled"]),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(CONFIG["input"]),
        help="DSM GeoTIFF veya LAS/LAZ yolu.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CONFIG["output"]),
        help="Uretilecek DTM GeoTIFF yolu.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=str(CONFIG["method"]),
        choices=["auto", "smrf", "fallback"],
        help=(
            "Yontem secimi: auto | smrf | fallback. "
            "auto: SMRF + opsiyonel fallback, smrf: sadece SMRF, fallback: sadece fallback "
            "(LAS/LAZ girdide fallback desteklenmez, SMRF kullanilir)."
        ),
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=CONFIG["cell"],
        help=(
            "SMRF cell (metre). Bos/NONE ise raster girdide piksel boyutu, "
            "LAS/LAZ girdide header'dan otomatik tahmin kullanilir. "
            "Buyuk deger daha az RAM/CPU, ama daha az detay."
        ),
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=float(CONFIG["slope"]),
        help="SMRF slope parametresi (egim toleransi).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(CONFIG["threshold"]),
        help="SMRF threshold parametresi (zemin ayrimi yukseklik esigi).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=float(CONFIG["window"]),
        help="SMRF max window (metre). Buyuk yapilarin zemin siniflamasina etki eder.",
    )
    parser.add_argument(
        "--scalar",
        type=float,
        default=float(CONFIG["scalar"]),
        help="SMRF scalar parametresi (karar mekanizmasi carpan katsayisi).",
    )
    parser.add_argument(
        "--smrf-max-pixels",
        type=int,
        default=int(CONFIG["smrf_max_pixels"]),
        help=(
            "SMRF icin maksimum piksel/nokta limiti. "
            "Asilirsa SMRF girdisi otomatik downsample edilir. "
            "Kalite kritikse <=0 ile kapat."
        ),
    )
    parser.add_argument(
        "--smrf-downsample-factor",
        type=float,
        default=float(CONFIG["smrf_downsample_factor"]),
        help=(
            "SMRF oncesi zorunlu downsample katsayisi. "
            "1.0 -> kapali, 2.0 -> genislik/yukseklik yariya iner. "
            "Kalite kritikse 1.0 kullan."
        ),
    )
    parser.add_argument(
        "--smrf-tiled",
        dest="smrf_tiled",
        action="store_true",
        help="SMRF'i ortusmeli karolarla calistir (RAM dusuk, kalite korunur, sure artabilir).",
    )
    parser.add_argument(
        "--no-smrf-tiled",
        dest="smrf_tiled",
        action="store_false",
        help="SMRF'i tek parcada calistir (hizli olabilir, RAM yuksek).",
    )
    parser.add_argument(
        "--smrf-tile-size",
        type=int,
        default=int(CONFIG["smrf_tile_size"]),
        help="SMRF tiled modunda karo boyutu (piksel). Buyudukce RAM artar.",
    )
    parser.add_argument(
        "--smrf-overlap-px",
        type=int,
        default=int(CONFIG["smrf_overlap_px"]),
        help="SMRF tiled modunda karo bindirmesi (piksel). 0 ise otomatik.",
    )
    parser.add_argument(
        "--allow-fallback",
        dest="allow_fallback",
        action="store_true",
        help="PDAL yoksa fallback (morph) DTM uretimine gec.",
    )
    parser.add_argument(
        "--no-fallback",
        dest="allow_fallback",
        action="store_false",
        help="PDAL yoksa hata ver; fallback kullanma.",
    )
    parser.add_argument(
        "--opening-meters",
        type=float,
        default=float(CONFIG["opening_meters"]),
        help="Fallback icin morfolojik acma pencere boyutu (metre).",
    )
    parser.add_argument(
        "--smooth-sigma-px",
        type=float,
        default=float(CONFIG["smooth_sigma_px"]),
        help="Fallback icin Gauss yumusatma sigma degeri (piksel).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=int(CONFIG["tile_size"]),
        help="Fallback isleme karo boyutu (piksel).",
    )
    parser.add_argument("--nodata", type=float, default=float(CONFIG["nodata"]), help="Cikti nodata degeri.")
    parser.add_argument(
        "--compression",
        type=str,
        default=str(CONFIG["compression"]),
        choices=["LZW", "DEFLATE", "NONE"],
        help="GeoTIFF sikistirma tipi.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=str(CONFIG["log_level"]),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Ilerleme cubuklarini goster.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Ilerleme cubuklarini kapat.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_path = _normalize_path(args.input)
    output_path = _normalize_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi DSM bulunamadi: {input_path}")

    method = str(args.method).strip().lower()
    is_las_input = input_path.suffix.lower() in {".las", ".laz"}

    if is_las_input:
        if method == "fallback":
            LOGGER.warning(
                "LAS/LAZ girdide fallback desteklenmedigi icin method=smrf olarak degistirildi."
            )
            method = "smrf"
        elif method == "auto" and bool(args.allow_fallback):
            LOGGER.info("LAS/LAZ girdide fallback uygulanmaz; auto mod yalnizca SMRF calistirir.")

        if args.cell and float(args.cell) > 0:
            las_cell = float(args.cell)
            LOGGER.info("LAS/LAZ girdide kullanici cell degeri kullaniliyor: %.4f m", las_cell)
        else:
            las_cell, reason = _estimate_las_cell_from_header(
                input_las_path=input_path,
                fallback_cell=float(_LAS_DEFAULT_CELL),
            )
            if abs(las_cell - float(_LAS_DEFAULT_CELL)) < 1e-9 and not reason.startswith("otomatik tahmin"):
                LOGGER.warning(
                    "LAS/LAZ girdide --cell verilmedi; otomatik tahmin basarisiz (%s). "
                    "Varsayilan cell=%.2f m kullaniliyor.",
                    reason,
                    float(_LAS_DEFAULT_CELL),
                )
            else:
                LOGGER.info(
                    "LAS/LAZ girdide --cell verilmedi; %s -> cell=%.4f m",
                    reason,
                    las_cell,
                )

        smrf_params = SmrfParams(
            cell=las_cell,
            slope=float(args.slope),
            threshold=float(args.threshold),
            window=float(args.window),
            scalar=float(args.scalar),
        )
        LOGGER.info("[1/2] Girdi nokta bulutu bilgisi okunuyor...")
        LOGGER.info("Yontem secimi: method=%s allow_fallback=%s", method, bool(args.allow_fallback))
        LOGGER.info(
            "SMRF parametreleri: cell=%.4f slope=%.4f threshold=%.4f window=%.2f scalar=%.3f",
            smrf_params.cell,
            smrf_params.slope,
            smrf_params.threshold,
            smrf_params.window,
            smrf_params.scalar,
        )
        LOGGER.info("[2/2] LAS/LAZ SMRF asamasi baslatiliyor...")
        _run_las_smrf(
            input_las_path=input_path,
            output_dtm_path=output_path,
            params=smrf_params,
            nodata=float(args.nodata),
            compression=str(args.compression),
            show_progress=bool(args.progress),
        )
        LOGGER.info("DTM basariyla yazildi (LAS SMRF): %s", output_path)
        return 0

    LOGGER.info("[1/4] Girdi raster bilgisi okunuyor...")
    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError(f"Girdi raster en az bir bant icermeli: {input_path}")
        xres, yres = src.res
        base_cell = max(abs(float(xres)), abs(float(yres)))
        bounds = src.bounds
        input_points = int(src.width * src.height)

    smrf_params = SmrfParams(
        cell=float(args.cell) if args.cell and args.cell > 0 else base_cell,
        slope=float(args.slope),
        threshold=float(args.threshold),
        window=float(args.window),
        scalar=float(args.scalar),
    )
    LOGGER.info("Yontem secimi: method=%s allow_fallback=%s", method, bool(args.allow_fallback))
    LOGGER.info(
        "SMRF parametreleri: cell=%.4f slope=%.4f threshold=%.4f window=%.2f scalar=%.3f",
        smrf_params.cell,
        smrf_params.slope,
        smrf_params.threshold,
        smrf_params.window,
        smrf_params.scalar,
    )
    smrf_max_pixels = int(args.smrf_max_pixels) if int(args.smrf_max_pixels) > 0 else None
    smrf_downsample_factor = max(1.0, float(args.smrf_downsample_factor))
    smrf_tiled = bool(args.smrf_tiled)
    smrf_tile_size = max(128, int(args.smrf_tile_size))
    smrf_overlap_px_arg = max(0, int(args.smrf_overlap_px))
    LOGGER.info(
        "SMRF bellek limiti: max_pixels=%s downsample_factor=%.2f girdi_nokta=%d",
        smrf_max_pixels if smrf_max_pixels is not None else "kapali",
        smrf_downsample_factor,
        input_points,
    )
    LOGGER.info(
        "SMRF isleme modu: tiled=%s tile_size=%d overlap_px=%d",
        smrf_tiled,
        smrf_tile_size,
        smrf_overlap_px_arg,
    )

    smrf_completed = False
    if method != "fallback":
        with tempfile.TemporaryDirectory(prefix="on_isleme_smrf_") as temp_dir:
            raw_dtm_path = Path(temp_dir) / "dtm_smrf_raw.tif"
            LOGGER.info("[2/4] SMRF asamasi baslatiliyor...")
            try:
                smrf_output_aligned = False
                if smrf_tiled:
                    source_dimension = _detect_pdal_height_dimension(input_path)
                    LOGGER.info("SMRF kaynak yukseklik boyutu: %s", source_dimension)

                    auto_overlap = max(
                        32,
                        int(math.ceil(smrf_params.window / max(smrf_params.cell, 1e-6))) + 8,
                    )
                    smrf_overlap_px = smrf_overlap_px_arg if smrf_overlap_px_arg > 0 else auto_overlap
                    LOGGER.info(
                        "SMRF tiled overlap: %d px (auto=%d)",
                        smrf_overlap_px,
                        auto_overlap,
                    )
                    if smrf_max_pixels is not None or smrf_downsample_factor > 1.0:
                        LOGGER.info(
                            "SMRF tiled modunda downsample kullanilmiyor; kaliteyi korumak icin orijinal cozumunurlukte isleniyor."
                        )

                    _compute_smrf_tiled(
                        input_dsm_path=input_path,
                        output_dtm_path=raw_dtm_path,
                        params=smrf_params,
                        nodata=float(args.nodata),
                        compression=str(args.compression),
                        source_dimension=source_dimension,
                        tile_size=smrf_tile_size,
                        overlap_px=smrf_overlap_px,
                        show_progress=bool(args.progress),
                        temp_dir=Path(temp_dir),
                    )
                    smrf_output_aligned = True
                    smrf_completed = raw_dtm_path.exists()
                else:
                    smrf_input_path, expected_points, smrf_base_cell = _prepare_smrf_input(
                        input_path=input_path,
                        temp_dir=Path(temp_dir),
                        max_pixels=smrf_max_pixels,
                        downsample_factor=smrf_downsample_factor,
                        show_progress=bool(args.progress),
                    )
                    if smrf_base_cell > smrf_params.cell:
                        LOGGER.warning(
                            "SMRF cell degeri downsample nedeniyle yukseltildi: %.4f -> %.4f",
                            smrf_params.cell,
                            smrf_base_cell,
                        )
                        smrf_params = replace(smrf_params, cell=smrf_base_cell)
                    if smrf_input_path == input_path:
                        LOGGER.info("SMRF girdi rasteri: orijinal cozumunurluk kullanilacak.")
                    else:
                        LOGGER.warning(
                            "SMRF girdi rasteri downsample edildi: %s (yaklasik nokta=%d)",
                            smrf_input_path,
                            expected_points,
                        )
                    source_dimension = _detect_pdal_height_dimension(smrf_input_path)
                    LOGGER.info("SMRF kaynak yukseklik boyutu: %s", source_dimension)
                    pipeline = _build_pipeline(
                        input_path=smrf_input_path,
                        temp_output_path=raw_dtm_path,
                        params=smrf_params,
                        nodata=float(args.nodata),
                        bounds=bounds,
                        source_dimension=source_dimension,
                    )
                    _run_pdal_pipeline(
                        pipeline=pipeline,
                        show_progress=bool(args.progress),
                        expected_points=expected_points,
                    )
                    smrf_completed = raw_dtm_path.exists()

                if smrf_completed:
                    if smrf_output_aligned:
                        os.replace(raw_dtm_path, output_path)
                        LOGGER.info("DTM basariyla yazildi (SMRF tiled): %s", output_path)
                    else:
                        LOGGER.info("[3/4] Cikti kaynak gridine hizalaniyor...")
                        _snap_to_reference_grid(
                            reference_dsm_path=input_path,
                            raw_dtm_path=raw_dtm_path,
                            output_dtm_path=output_path,
                            nodata=float(args.nodata),
                            compression=str(args.compression),
                            show_progress=bool(args.progress),
                        )
                        LOGGER.info("DTM basariyla yazildi (SMRF): %s", output_path)
                    return 0
            except RuntimeError as exc:
                if method == "smrf":
                    raise RuntimeError(
                        f"SMRF zorunlu secildi (method=smrf) ve calisamadi: {exc}"
                    ) from exc
                if not args.allow_fallback:
                    raise
                LOGGER.warning("SMRF calisamadi (%s). Fallback metoda geciliyor.", exc)
    else:
        LOGGER.info("[2/4] SMRF asamasi atlandi (method=fallback).")

    if method == "smrf":
        raise RuntimeError("SMRF cikti uretilemedi (method=smrf).")
    if method == "auto" and not args.allow_fallback:
        raise RuntimeError("SMRF cikti uretilemedi ve fallback kapali.")

    if method == "fallback":
        LOGGER.info("[3/4] Fallback asamasi baslatiliyor...")
    else:
        LOGGER.info("[4/4] Fallback asamasi baslatiliyor...")
    _compute_fallback_dtm(
        input_dsm_path=input_path,
        output_dtm_path=output_path,
        nodata=float(args.nodata),
        compression=str(args.compression),
        opening_meters=float(args.opening_meters),
        smooth_sigma_px=float(args.smooth_sigma_px),
        tile_size=int(args.tile_size),
        show_progress=bool(args.progress),
    )
    LOGGER.info("DTM basariyla yazildi (fallback): %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
