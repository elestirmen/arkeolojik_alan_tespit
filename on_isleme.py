"""
DSM rasterini PDAL SMRF ile DTM'ye ceviren on-isleme araci.

Not:
- Bu betik PDAL Python bagimliligina ihtiyac duyar (`pip install pdal`).
- Cikti son adimda kaynak DSM'in gridine zorla hizalanir; boyut, transform ve
  cozumunurluk birebir ayni olur.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
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

# ==================== CONFIG ====================
# Buradaki degerler komut satirindan arguman verilmezse varsayilan olarak kullanilir.
# Komut satiri argumanlari her zaman bu degerleri ezer.
CONFIG: dict[str, Any] = {
    "input": "/veri/karlik_dag_dsm.tif",
    "output": "/veri/karlik_dag_dtm_smrf.tif",
    "cell": None,  # None -> DSM piksel boyutu kullanilir
    "slope": 0.2,
    "threshold": 0.45,
    "window": 16.0,
    "scalar": 1.25,
    "smrf_max_pixels": 120_000_000,  # >0 ise asilirsa SMRF oncesi downsample yapilir
    "smrf_downsample_factor": 1.0,   # 1.0 -> kapali
    "allow_fallback": True,
    "opening_meters": 6.0,
    "smooth_sigma_px": 1.5,
    "tile_size": 2048,
    "nodata": -9999.0,
    "compression": "LZW",  # LZW | DEFLATE | NONE
    "log_level": "INFO",   # DEBUG | INFO | WARNING | ERROR
    "progress": True,
}
# ===============================================


@dataclass(frozen=True)
class SmrfParams:
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

    def _draw(self, force: bool) -> None:
        ratio = self.current / self.total
        pct = int(ratio * 100)
        bar_w = 28
        fill = int(bar_w * ratio)
        bar = "#" * fill + "-" * (bar_w - fill)
        sys.stderr.write(
            f"\r{self.desc}: [{bar}] {pct:3d}% ({self.current}/{self.total} {self.unit})"
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


def _run_pdal_pipeline(
    pipeline: list[dict[str, Any]],
    show_progress: bool,
    expected_points: Optional[int],
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
    last_log_elapsed = -60.0
    start_time = time.time()

    while worker.is_alive():
        elapsed = time.time() - start_time
        target_value = progress_value
        if writer_path is not None and expected_bytes is not None and expected_bytes > 0 and writer_path.exists():
            try:
                size_ratio = writer_path.stat().st_size / expected_bytes
                target_value = min(95, max(progress_value, int(size_ratio * 100)))
            except OSError:
                pass
        else:
            target_value = min(90, max(progress_value, int(elapsed // 4) + 1))

        if pbar is not None and target_value > progress_value:
            pbar.update(target_value - progress_value)
            progress_value = target_value
        elif pbar is None and elapsed - last_log_elapsed >= 60.0:
            LOGGER.info("SMRF devam ediyor... %.0f sn", elapsed)
            last_log_elapsed = elapsed
        time.sleep(1.0)

    worker.join()
    if pbar is not None and progress_value < 100:
        pbar.update(100 - progress_value)
        pbar.close()

    if "exc" in error:
        raise RuntimeError(str(error["exc"])) from error["exc"]

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


def _prepare_temp_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.stem}_tmp_",
        suffix=output_path.suffix,
        dir=str(output_path.parent),
    )
    os.close(fd)
    return Path(temp_name)


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
            "DSM rasterini SMRF (PDAL) ile DTM'ye cevirir ve ciktiyi "
            "kaynak rasterin ayni piksel gridine hizalar."
        )
    )
    parser.set_defaults(
        allow_fallback=bool(CONFIG["allow_fallback"]),
        progress=bool(CONFIG["progress"]),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(CONFIG["input"]),
        help="DSM GeoTIFF yolu.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CONFIG["output"]),
        help="Uretilecek DTM GeoTIFF yolu.",
    )
    parser.add_argument("--cell", type=float, default=CONFIG["cell"], help="SMRF cell parametresi (metre). Bossa DSM piksel boyutu kullanilir.")
    parser.add_argument("--slope", type=float, default=float(CONFIG["slope"]), help="SMRF slope parametresi.")
    parser.add_argument("--threshold", type=float, default=float(CONFIG["threshold"]), help="SMRF threshold parametresi.")
    parser.add_argument("--window", type=float, default=float(CONFIG["window"]), help="SMRF max window parametresi.")
    parser.add_argument("--scalar", type=float, default=float(CONFIG["scalar"]), help="SMRF scalar parametresi.")
    parser.add_argument(
        "--smrf-max-pixels",
        type=int,
        default=int(CONFIG["smrf_max_pixels"]),
        help=(
            "SMRF icin maksimum piksel/nokta limiti. "
            "Asilirsa SMRF girdisi otomatik downsample edilir. <=0 ile kapat."
        ),
    )
    parser.add_argument(
        "--smrf-downsample-factor",
        type=float,
        default=float(CONFIG["smrf_downsample_factor"]),
        help=(
            "SMRF oncesi zorunlu downsample katsayisi. "
            "1.0 -> kapali, 2.0 -> genislik/yukseklik yariya iner."
        ),
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
    LOGGER.info(
        "SMRF bellek limiti: max_pixels=%s downsample_factor=%.2f girdi_nokta=%d",
        smrf_max_pixels if smrf_max_pixels is not None else "kapali",
        smrf_downsample_factor,
        input_points,
    )

    smrf_completed = False
    with tempfile.TemporaryDirectory(prefix="on_isleme_smrf_") as temp_dir:
        raw_dtm_path = Path(temp_dir) / "dtm_smrf_raw.tif"
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
        LOGGER.info("[2/4] SMRF asamasi baslatiliyor...")
        try:
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
        except RuntimeError as exc:
            if not args.allow_fallback:
                raise
            LOGGER.warning("SMRF calisamadi (%s). Fallback metoda geciliyor.", exc)

        if smrf_completed:
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

    if not args.allow_fallback:
        raise RuntimeError("SMRF cikti uretilemedi ve fallback kapali.")

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
