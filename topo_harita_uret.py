"""Generate separate topographic visualization rasters.

Examples:
    python topo_harita_uret.py
    python topo_harita_uret.py --input dem.tif --elevation-band 1
    python topo_harita_uret.py --input rgb_dsm_dtm_5band.tif --elevation-band 5 --products hillshade,svf,slrm
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from archeo_shared.console import configure_utf8_console

configure_utf8_console()

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore[assignment]

try:
    from rvt import vis as rvt_vis
except ImportError:  # pragma: no cover - only required for SVF/RVT SLRM
    rvt_vis = None  # type: ignore[assignment]


LOGGER = logging.getLogger("topo_harita_uret")
DEFAULT_PRODUCTS = ("hillshade", "svf", "slrm")
PRODUCT_ALIASES = {
    "all": "all",
    "hepsi": "all",
    "hill": "hillshade",
    "hs": "hillshade",
    "hillshade": "hillshade",
    "svf": "svf",
    "sky_view_factor": "svf",
    "sky-view-factor": "svf",
    "slrm": "slrm",
    "lrm": "slrm",
    "local_relief": "slrm",
    "local-relief": "slrm",
    "slope": "slope",
    "egim": "slope",
}

# ==================== CONFIG ====================
# Bu scripti IDE'den veya sadece `python topo_harita_uret.py` ile calistirirsan
# asagidaki degerler kullanilir. Komut satiri argumanlari bu degerleri ezer.
CONFIG: Dict[str, Any] = {
    # Girdi GeoTIFF. Tek bant DTM/DSM olabilir veya RGB+DSM+DTM 5-band olabilir.
    "input": "workspace/on_veri/karlik_dag_set/karlik_dag_rgb_dtm_dsm_5band.tif",
    # Yukseklik bandi. Repo 5-band duzeni icin: 4=DSM, 5=DTM.
    "elevation_band": 5,
    # Bos/None ise girdi dosyasinin yanina yazar.
    "output_dir": "workspace/on_veri/karlik_dag_set/topo_haritalar",
    # Bos/None ise girdi dosya adi kullanilir.
    "prefix": "karlik_dag_dtm",
    # Secenekler: hillshade, svf, slrm, slope, all
    "products": "hillshade,svf,slrm",
    # Buyuk haritalar icin blok boyutu. Daha buyuk = hizli, daha cok RAM.
    "chunk": 2048,
    # None ise radius/sigma degerlerinden otomatik hesaplanir.
    "halo": None,
    # SVF ve SLRM parametreleri.
    "svf_radius_m": 10.0,
    "slrm_radius_m": 10.0,
    "slrm_sigma_px": 6.0,
    # Hillshade/slope parametreleri.
    "z_factor": 1.0,
    "azimuth": 315.0,
    "altitude": 45.0,
    # GeoTIFF sikistirma: LZW, DEFLATE, NONE.
    "compression": "DEFLATE",
    # Mevcut ciktilarin uzerine yazilsin mi?
    "overwrite": False,
    "log_level": "INFO",
}
# ===============================================


@dataclass(frozen=True)
class TopoMapConfig:
    input_path: Path
    elevation_band: int = 1
    output_dir: Optional[Path] = None
    prefix: Optional[str] = None
    products: Tuple[str, ...] = DEFAULT_PRODUCTS
    chunk: int = 2048
    halo: Optional[int] = None
    svf_radius_m: float = 10.0
    slrm_radius_m: float = 10.0
    slrm_sigma_px: float = 6.0
    z_factor: float = 1.0
    azimuth: float = 315.0
    altitude: float = 45.0
    compression: str = "DEFLATE"
    overwrite: bool = False
    log_level: str = "INFO"


def parse_products(raw: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    if not parts:
        return DEFAULT_PRODUCTS

    selected: List[str] = []
    for part in parts:
        product = PRODUCT_ALIASES.get(part.lower())
        if product is None:
            valid = ", ".join(sorted(k for k in PRODUCT_ALIASES if k not in {"all", "hepsi"}))
            raise ValueError(f"Bilinmeyen urun: {part}. Gecerli urunler: {valid}")
        if product == "all":
            for default_product in DEFAULT_PRODUCTS:
                if default_product not in selected:
                    selected.append(default_product)
            continue
        if product not in selected:
            selected.append(product)
    return tuple(selected)


def _config_text(name: str, fallback: str = "") -> str:
    value = CONFIG.get(name, fallback)
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _iter_windows(width: int, height: int, chunk: int) -> Iterable[Window]:
    if chunk <= 0:
        raise ValueError("chunk pozitif olmali")
    for row in range(0, height, chunk):
        for col in range(0, width, chunk):
            yield Window(col, row, min(chunk, width - col), min(chunk, height - row))


def _window_with_halo(window: Window, width: int, height: int, halo: int) -> Tuple[Window, Tuple[slice, slice]]:
    col = int(window.col_off)
    row = int(window.row_off)
    win_w = int(window.width)
    win_h = int(window.height)
    row0 = max(0, row - halo)
    col0 = max(0, col - halo)
    row1 = min(height, row + win_h + halo)
    col1 = min(width, col + win_w + halo)
    padded = Window(col0, row0, col1 - col0, row1 - row0)
    row_slice = slice(row - row0, row - row0 + win_h)
    col_slice = slice(col - col0, col - col0 + win_w)
    return padded, (row_slice, col_slice)


def _read_band_as_float(src: rasterio.io.DatasetReader, band_index: int, window: Window) -> np.ndarray:
    arr = src.read(band_index, window=window, masked=True)
    if hasattr(arr, "filled"):
        return np.asarray(arr.astype(np.float32).filled(np.nan), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _fill_nodata(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32), valid
    filled = data.copy()
    filled[~valid] = float(np.nanmedian(data[valid]))
    return filled, valid


def _pixel_size(src: rasterio.io.DatasetReader) -> float:
    transform = src.transform
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _hillshade(
    dem: np.ndarray,
    *,
    pixel_size: float,
    z_factor: float,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    filled, valid = _fill_nodata(dem)
    if z_factor != 1.0:
        filled = filled * float(z_factor)
    gy, gx = np.gradient(filled, pixel_size, pixel_size)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth_rad = math.radians(float(azimuth))
    altitude_rad = math.radians(float(altitude))
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(
        azimuth_rad - aspect
    )
    shaded = np.clip((shaded + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)
    shaded[~valid] = np.nan
    return shaded


def _slope_degrees(dem: np.ndarray, *, pixel_size: float, z_factor: float) -> np.ndarray:
    filled, valid = _fill_nodata(dem)
    if z_factor != 1.0:
        filled = filled * float(z_factor)
    gy, gx = np.gradient(filled, pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    slope[~valid] = np.nan
    return slope


def _rvt_as_float32(result_obj: Any, name: str) -> np.ndarray:
    if isinstance(result_obj, np.ndarray):
        return result_obj.astype(np.float32, copy=False)
    if isinstance(result_obj, dict):
        for key in (name, name.upper(), name.lower(), "svf", "SLRM", "slrm", "lrm"):
            if key in result_obj:
                return _rvt_as_float32(result_obj[key], name)
        for value in result_obj.values():
            try:
                return _rvt_as_float32(value, name)
            except TypeError:
                continue
    if isinstance(result_obj, (list, tuple)):
        for value in result_obj:
            try:
                return _rvt_as_float32(value, name)
            except TypeError:
                continue
    try:
        arr = np.asarray(result_obj)
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float32, copy=False)
    except Exception:
        pass
    raise TypeError(f"RVT '{name}' beklenmeyen tip dondurdu: {type(result_obj)}")


def _radius_to_cells(radius_m: float, pixel_size: float) -> int:
    return max(1, int(round(float(radius_m) / max(float(pixel_size), 1e-9))))


def _compute_svf(dem: np.ndarray, *, pixel_size: float, radius_m: float) -> np.ndarray:
    if rvt_vis is None:
        raise ImportError("SVF icin rvt-py/rvt gerekli: pip install rvt-py")
    filled, valid = _fill_nodata(dem)
    radius_px = _radius_to_cells(radius_m, pixel_size)
    attempts = (
        {
            "dem": filled,
            "resolution": pixel_size,
            "compute_svf": True,
            "compute_asvf": False,
            "compute_opns": False,
            "svf_r_max": radius_px,
            "svf_noise": 0,
            "no_data": None,
        },
        {
            "dem": filled,
            "resolution": pixel_size,
            "compute_svf": True,
            "compute_asvf": False,
            "compute_opns": False,
            "svf_r_max": radius_px,
            "svf_noise": 0,
        },
        {
            "dem": filled,
            "resolution": pixel_size,
            "compute_svf": True,
            "compute_opns": False,
            "svf_r_max": radius_px,
            "svf_noise": 0,
        },
    )
    last_type_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            svf = _rvt_as_float32(rvt_vis.sky_view_factor(**kwargs), "svf")
            svf[~valid] = np.nan
            return svf
        except TypeError as exc:
            last_type_error = exc
    try:
        svf = _rvt_as_float32(rvt_vis.sky_view_factor(dem=filled, resolution=pixel_size), "svf")
        svf[~valid] = np.nan
        return svf
    except Exception:
        if last_type_error is not None:
            raise last_type_error
        raise


def _compute_slrm(
    dem: np.ndarray,
    *,
    pixel_size: float,
    radius_m: float,
    sigma_px: float,
) -> np.ndarray:
    filled, valid = _fill_nodata(dem)
    radius_cell = _radius_to_cells(radius_m, pixel_size)

    if rvt_vis is not None and hasattr(rvt_vis, "slrm"):
        for kwargs in (
            {"dem": filled, "radius_cell": radius_cell, "no_data": None},
            {"dem": filled, "radius_cell": radius_cell},
        ):
            try:
                slrm = _rvt_as_float32(rvt_vis.slrm(**kwargs), "slrm")
                slrm[~valid] = np.nan
                return slrm
            except TypeError:
                continue

    if rvt_vis is not None and hasattr(rvt_vis, "local_relief_model"):
        for radius_key in ("search_radius", "radius", "r_max", "max_radius"):
            for extra_kwargs in ({"no_data": None}, {}):
                try:
                    slrm = _rvt_as_float32(
                        rvt_vis.local_relief_model(
                            dem=filled,
                            resolution=pixel_size,
                            **{radius_key: float(radius_m), **extra_kwargs},
                        ),
                        "local_relief_model",
                    )
                    slrm[~valid] = np.nan
                    return slrm
                except TypeError:
                    continue

    low_pass = gaussian_filter(filled, sigma=float(sigma_px))
    slrm = (filled - low_pass).astype(np.float32)
    slrm[~valid] = np.nan
    return slrm


def _output_profile(src: rasterio.io.DatasetReader, *, compression: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "height": int(src.height),
        "width": int(src.width),
        "count": 1,
        "dtype": "float32",
        "crs": src.crs,
        "transform": src.transform,
        "nodata": np.nan,
        "compress": str(compression).upper(),
        "BIGTIFF": "IF_SAFER",
        "interleave": "band",
    }
    if int(src.width) >= 16 and int(src.height) >= 16:
        blockx = min(512, (int(src.width) // 16) * 16)
        blocky = min(512, (int(src.height) // 16) * 16)
        profile.update({"tiled": True, "blockxsize": max(16, blockx), "blockysize": max(16, blocky)})
        if str(compression).upper() != "NONE":
            profile["predictor"] = 3
    return profile


def _resolve_output_paths(config: TopoMapConfig) -> Dict[str, Path]:
    output_dir = config.output_dir or config.input_path.parent
    prefix = config.prefix or config.input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return {product: output_dir / f"{prefix}_{product}.tif" for product in config.products}


def _auto_halo(config: TopoMapConfig, pixel_size: float) -> int:
    radii = [2]
    if "svf" in config.products:
        radii.append(_radius_to_cells(config.svf_radius_m, pixel_size))
    if "slrm" in config.products:
        radii.append(_radius_to_cells(config.slrm_radius_m, pixel_size))
        radii.append(int(math.ceil(4.0 * max(float(config.slrm_sigma_px), 0.0))))
    if "slope" in config.products:
        radii.append(2)
    return max(radii)


def generate_topo_maps(config: TopoMapConfig) -> Dict[str, Path]:
    output_paths = _resolve_output_paths(config)
    if not config.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            joined = "\n  ".join(existing)
            raise FileExistsError(f"Cikti dosyasi zaten var. --overwrite kullanin:\n  {joined}")
    if "svf" in config.products and rvt_vis is None:
        raise ImportError("SVF haritasi icin rvt-py/rvt gerekli: pip install rvt-py")

    with rasterio.open(config.input_path) as src:
        if config.elevation_band <= 0 or config.elevation_band > src.count:
            raise ValueError(f"elevation-band gecersiz: {config.elevation_band}; raster band sayisi={src.count}")

        pixel_size = _pixel_size(src)
        if src.crs is not None and getattr(src.crs, "is_geographic", False):
            LOGGER.warning(
                "Girdi CRS derece tabanli gorunuyor (%s). Radius metre varsayilir; once metre tabanli CRS'e reprojection onerilir.",
                src.crs,
            )
        halo = int(config.halo) if config.halo is not None else _auto_halo(config, pixel_size)
        profile = _output_profile(src, compression=config.compression)
        windows = list(_iter_windows(int(src.width), int(src.height), int(config.chunk)))
        LOGGER.info(
            "Topo haritalari uretiliyor: input=%s band=%d products=%s chunk=%d halo=%d px pixel=%.4f",
            config.input_path,
            config.elevation_band,
            ",".join(config.products),
            config.chunk,
            halo,
            pixel_size,
        )

        datasets: Dict[str, Any] = {}
        try:
            for product, path in output_paths.items():
                datasets[product] = rasterio.open(path, "w", **profile)
                datasets[product].set_band_description(1, product)

            iterator: Iterable[Window]
            if tqdm is not None:
                iterator = tqdm(windows, desc="Topo maps", unit="block", total=len(windows))
            else:
                iterator = windows

            for idx, window in enumerate(iterator, start=1):
                padded_window, core = _window_with_halo(window, int(src.width), int(src.height), halo)
                dem = _read_band_as_float(src, config.elevation_band, padded_window)
                row_slice, col_slice = core

                for product in config.products:
                    if product == "hillshade":
                        arr = _hillshade(
                            dem,
                            pixel_size=pixel_size,
                            z_factor=config.z_factor,
                            azimuth=config.azimuth,
                            altitude=config.altitude,
                        )
                    elif product == "svf":
                        arr = _compute_svf(dem, pixel_size=pixel_size, radius_m=config.svf_radius_m)
                    elif product == "slrm":
                        arr = _compute_slrm(
                            dem,
                            pixel_size=pixel_size,
                            radius_m=config.slrm_radius_m,
                            sigma_px=config.slrm_sigma_px,
                        )
                    elif product == "slope":
                        arr = _slope_degrees(dem, pixel_size=pixel_size, z_factor=config.z_factor)
                    else:  # pragma: no cover - parse_products prevents this
                        continue
                    datasets[product].write(arr[row_slice, col_slice].astype(np.float32, copy=False), 1, window=window)

                if tqdm is None and (idx % 100 == 0 or idx == len(windows)):
                    LOGGER.info("Ilerleme: %d/%d blok", idx, len(windows))
        finally:
            for ds in datasets.values():
                ds.close()

    LOGGER.info("Topo haritalari hazir:")
    for product, path in output_paths.items():
        LOGGER.info("  %s: %s", product, path)
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DTM/DSM bandindan ayri hillshade, SVF, SLRM GeoTIFF haritalari uret. "
            "Arguman verilmezse dosyanin basindaki CONFIG kullanilir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=_config_text("input"), help="Girdi GeoTIFF yolu.")
    parser.add_argument(
        "--elevation-band",
        "--dem-band",
        "--dtm-band",
        type=int,
        default=int(CONFIG["elevation_band"]),
        help="Hillshade/SVF/SLRM icin kullanilacak yukseklik bandi. 5-band RGB+DSM+DTM icin genelde 5.",
    )
    parser.add_argument("--output-dir", default=CONFIG.get("output_dir"), help="Cikti klasoru. Bos ise girdi dosyasinin yani.")
    parser.add_argument("--prefix", default=CONFIG.get("prefix"), help="Cikti dosya on eki. Bos ise girdi dosya adi.")
    parser.add_argument(
        "--products",
        default=_config_text("products", "hillshade,svf,slrm"),
        help="Virgullu urun listesi: hillshade, svf, slrm, slope veya all.",
    )
    parser.add_argument("--chunk", type=int, default=int(CONFIG["chunk"]), help="Islenecek blok boyutu piksel.")
    parser.add_argument("--halo", type=int, default=CONFIG.get("halo"), help="Blok kenari halo piksel. Bos ise otomatik.")
    parser.add_argument("--svf-radius-m", type=float, default=float(CONFIG["svf_radius_m"]), help="SVF arama yaricapi metre.")
    parser.add_argument("--slrm-radius-m", type=float, default=float(CONFIG["slrm_radius_m"]), help="SLRM yaricapi metre.")
    parser.add_argument("--slrm-sigma-px", type=float, default=float(CONFIG["slrm_sigma_px"]), help="RVT SLRM yoksa Gaussian fallback sigma piksel.")
    parser.add_argument("--z-factor", type=float, default=float(CONFIG["z_factor"]), help="Hillshade/slope yukseklik carpan katsayisi.")
    parser.add_argument("--azimuth", type=float, default=float(CONFIG["azimuth"]), help="Hillshade isik azimutu derece.")
    parser.add_argument("--altitude", type=float, default=float(CONFIG["altitude"]), help="Hillshade isik yuksekligi derece.")
    parser.add_argument(
        "--compression",
        choices=("LZW", "DEFLATE", "NONE"),
        default=str(CONFIG["compression"]).upper(),
        help="GeoTIFF sikistirma tipi.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true", help="Mevcut ciktilarin uzerine yaz.")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Mevcut cikti varsa dur.")
    parser.set_defaults(overwrite=bool(CONFIG["overwrite"]))
    parser.add_argument("--log-level", default=str(CONFIG["log_level"]), help="Log seviyesi.")
    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def config_from_args(args: argparse.Namespace) -> TopoMapConfig:
    input_text = str(args.input or "").strip()
    if not input_text:
        raise ValueError("Girdi yolu bos. CONFIG['input'] veya --input ile bir GeoTIFF belirtin.")
    return TopoMapConfig(
        input_path=Path(input_text),
        elevation_band=int(args.elevation_band),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        prefix=args.prefix,
        products=parse_products(args.products),
        chunk=int(args.chunk),
        halo=args.halo,
        svf_radius_m=float(args.svf_radius_m),
        slrm_radius_m=float(args.slrm_radius_m),
        slrm_sigma_px=float(args.slrm_sigma_px),
        z_factor=float(args.z_factor),
        azimuth=float(args.azimuth),
        altitude=float(args.altitude),
        compression=str(args.compression),
        overwrite=bool(args.overwrite),
        log_level=str(args.log_level),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    try:
        config = config_from_args(args)
        if not config.input_path.exists():
            raise FileNotFoundError(f"Girdi bulunamadi: {config.input_path}")
        generate_topo_maps(config)
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
