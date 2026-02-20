"""
RGB(1,2,3) + DSM + DTM rasterlarini tek bir 5-band GeoTIFF dosyasina birlestirir.

Bant sirasi:
1: RGB Red
2: RGB Green
3: RGB Blue
4: DSM
5: DTM
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


LOGGER = logging.getLogger("birlestir")
DEFAULT_NODATA = -9999.0
_TQDM_WARNED = False

# ==================== CONFIG ====================
# Buradaki degerler komut satirindan arguman verilmezse varsayilan olarak kullanilir.
# Komut satiri argumanlari her zaman bu degerleri ezer.
CONFIG: dict[str, Any] = {
    # RGB raster yolu (en az 3 band).
    "rgb_input": "/on_veri/karlik_dag_rgb.tif",
    # DTM raster yolu (1 band).
    "dtm_input": "/on_veri/karlik_dag_dtm.tif",
    # DSM raster yolu (1 band).
    "dsm_input": "/on_veri/karlik_dag_dsm.tif",
    # Uretilecek 5-band GeoTIFF yolu.
    "output": "/on_veri/karlik_dag_rgb_dtm_dsm_5band.tif",
    # Cikti nodata degeri.
    "nodata": DEFAULT_NODATA,
    # GeoTIFF sikistirma tipi: LZW | DEFLATE | NONE.
    "compression": "LZW",
    # TIFF tile/blok boyutu (16'nin kati olacak sekilde duzeltilir).
    "block_size": 512,
    # Log seviyesi: DEBUG | INFO | WARNING | ERROR.
    "log_level": "INFO",
    # Ilerleme cubugu acik/kapali.
    "progress": True,
}
# ===============================================


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
        self._draw(force=False)

    def close(self) -> None:
        self._draw(force=True)
        print()

    def _draw(self, force: bool) -> None:
        ratio = self.current / self.total
        pct = int(ratio * 100)
        bar_w = 28
        fill = int(bar_w * ratio)
        bar = "#" * fill + "-" * (bar_w - fill)
        text = f"\r{self.desc}: [{bar}] {pct:3d}% ({self.current}/{self.total} {self.unit})"
        print(text, end="", flush=True)
        if force:
            self._last_draw = 0.0


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


def _normalize_path(raw_path: str) -> Path:
    value = raw_path.strip()
    if not value:
        raise ValueError("Bos path verilemez.")
    has_drive = len(value) >= 2 and value[1] == ":"
    if value[0] in ("/", "\\") and not has_drive:
        return Path(value.lstrip("/\\"))
    return Path(value)


def _prepare_temp_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.stem}_tmp_",
        suffix=output_path.suffix,
        dir=str(output_path.parent),
    )
    os.close(fd)
    return Path(temp_name)


def _compose_rgb_dtm_dsm_5band(
    rgb_input_path: Path,
    dtm_input_path: Path,
    dsm_input_path: Path,
    output_path: Path,
    output_nodata: float,
    compression: str,
    show_progress: bool,
    block_size: int,
) -> None:
    compression_norm = str(compression).strip().upper()
    if compression_norm not in {"LZW", "DEFLATE", "NONE"}:
        raise ValueError(f"Desteklenmeyen sikistirma tipi: {compression}")

    block_size = max(16, int(block_size))
    block_size = max(16, (block_size // 16) * 16)
    out_nodata32 = np.float32(output_nodata)

    temp_output_path = _prepare_temp_output_path(output_path)
    pbar = None
    try:
        with (
            rasterio.open(rgb_input_path) as rgb,
            rasterio.open(dtm_input_path) as dtm,
            rasterio.open(dsm_input_path) as dsm,
        ):
            if rgb.count < 3:
                raise ValueError(f"RGB raster en az 3 band icermeli: {rgb_input_path}")
            if dtm.count < 1:
                raise ValueError(f"DTM raster en az 1 band icermeli: {dtm_input_path}")
            if dsm.count < 1:
                raise ValueError(f"DSM raster en az 1 band icermeli: {dsm_input_path}")

            for name, ds, ds_path in (
                ("DTM", dtm, dtm_input_path),
                ("DSM", dsm, dsm_input_path),
            ):
                if ds.width != rgb.width or ds.height != rgb.height:
                    raise ValueError(
                        f"{name} boyutu RGB ile ayni degil: {ds_path} "
                        f"({ds.width}x{ds.height}) != {rgb_input_path} ({rgb.width}x{rgb.height})"
                    )
                if ds.crs != rgb.crs:
                    raise ValueError(
                        f"{name} CRS RGB ile ayni degil: {ds_path} ({ds.crs}) != {rgb_input_path} ({rgb.crs})"
                    )
                if not ds.transform.almost_equals(rgb.transform):
                    raise ValueError(
                        f"{name} grid/transform RGB ile ayni degil: {ds_path} != {rgb_input_path}"
                    )

            profile: dict[str, Any] = {
                "driver": "GTiff",
                "width": rgb.width,
                "height": rgb.height,
                "count": 5,
                "dtype": "float32",
                "crs": rgb.crs,
                "transform": rgb.transform,
                "nodata": float(output_nodata),
                "compress": compression_norm,
                "tiled": True,
                "blockxsize": block_size,
                "blockysize": block_size,
                "interleave": "band",
                "BIGTIFF": "IF_SAFER",
            }
            if compression_norm != "NONE":
                profile["predictor"] = 3

            with rasterio.open(temp_output_path, "w", **profile) as dst:
                block_rows, block_cols = dst.block_shapes[0]
                total_blocks = math.ceil(dst.height / block_rows) * math.ceil(dst.width / block_cols)
                pbar = _open_progress(
                    total=total_blocks,
                    desc="5-band birlestirme",
                    unit="block",
                    enabled=show_progress,
                )

                dtm_nodata = dtm.nodata
                dsm_nodata = dsm.nodata

                for idx, (_, window) in enumerate(dst.block_windows(1), start=1):
                    r = rgb.read(1, window=window).astype(np.float32, copy=False)
                    g = rgb.read(2, window=window).astype(np.float32, copy=False)
                    b = rgb.read(3, window=window).astype(np.float32, copy=False)
                    dtm_arr = dtm.read(1, window=window).astype(np.float32, copy=False)
                    dsm_arr = dsm.read(1, window=window).astype(np.float32, copy=False)

                    if dtm_nodata is not None:
                        dtm_nodata32 = np.float32(dtm_nodata)
                        if np.isnan(dtm_nodata32):
                            dtm_arr = np.where(np.isnan(dtm_arr), out_nodata32, dtm_arr)
                        else:
                            dtm_arr = np.where(dtm_arr == dtm_nodata32, out_nodata32, dtm_arr)
                    if dsm_nodata is not None:
                        dsm_nodata32 = np.float32(dsm_nodata)
                        if np.isnan(dsm_nodata32):
                            dsm_arr = np.where(np.isnan(dsm_arr), out_nodata32, dsm_arr)
                        else:
                            dsm_arr = np.where(dsm_arr == dsm_nodata32, out_nodata32, dsm_arr)

                    dst.write(r, 1, window=window)
                    dst.write(g, 2, window=window)
                    dst.write(b, 3, window=window)
                    dst.write(dsm_arr, 4, window=window)
                    dst.write(dtm_arr, 5, window=window)

                    if pbar is not None:
                        pbar.update(1)
                    elif idx % 500 == 0 or idx == total_blocks:
                        LOGGER.info("5-band birlestirme: %d/%d blok", idx, total_blocks)

        os.replace(temp_output_path, output_path)
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
        description="RGB(1,2,3) + DSM + DTM rasterlarini tek bir 5-band GeoTIFF olarak birlestir."
    )
    parser.set_defaults(progress=bool(CONFIG["progress"]))
    parser.add_argument(
        "--rgb-input",
        type=str,
        default=str(CONFIG["rgb_input"]),
        help="RGB raster yolu (en az 3 band).",
    )
    parser.add_argument(
        "--dtm-input",
        type=str,
        default=str(CONFIG["dtm_input"]),
        help="DTM raster yolu (1 band).",
    )
    parser.add_argument(
        "--dsm-input",
        type=str,
        default=str(CONFIG["dsm_input"]),
        help="DSM raster yolu (1 band).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CONFIG["output"]),
        help="Uretilecek 5-band GeoTIFF yolu.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=float(CONFIG["nodata"]),
        help="Cikti nodata degeri.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=str(CONFIG["compression"]),
        choices=["LZW", "DEFLATE", "NONE"],
        help="GeoTIFF sikistirma tipi.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=int(CONFIG["block_size"]),
        help="TIFF tile/blok boyutu (16'nin kati).",
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
        help="Ilerleme cubugunu goster.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Ilerleme cubugunu kapat.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rgb_input_path = _normalize_path(args.rgb_input)
    dtm_input_path = _normalize_path(args.dtm_input)
    dsm_input_path = _normalize_path(args.dsm_input)
    output_path = _normalize_path(args.output)

    for name, path in (
        ("RGB", rgb_input_path),
        ("DTM", dtm_input_path),
        ("DSM", dsm_input_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} girdi rasteri bulunamadi: {path}")

    LOGGER.info(
        "5-band birlestirme basliyor: RGB(1,2,3)=%s | DSM=%s | DTM=%s | OUTPUT=%s",
        rgb_input_path,
        dsm_input_path,
        dtm_input_path,
        output_path,
    )
    _compose_rgb_dtm_dsm_5band(
        rgb_input_path=rgb_input_path,
        dtm_input_path=dtm_input_path,
        dsm_input_path=dsm_input_path,
        output_path=output_path,
        output_nodata=float(args.nodata),
        compression=str(args.compression),
        show_progress=bool(args.progress),
        block_size=int(args.block_size),
    )
    LOGGER.info("5-band GeoTIFF basariyla yazildi: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
