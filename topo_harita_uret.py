"""Generate separate topographic visualization rasters.

Examples:
    python topo_harita_uret.py              # argumansiz calistirma GUI acar
    python topo_harita_uret.py --cli        # CONFIG ile komut satiri uretimi
    python topo_harita_uret.py --gui
    python topo_harita_uret.py --input dem.tif --elevation-band 1
    python topo_harita_uret.py --input rgb_dsm_dtm_5band.tif --elevation-band 5 --products hillshade,svf,slrm
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
# RVT slrm trend yaricapi bu piksel araliginda olmali.
RVT_SLRM_MIN_CELL = 10
RVT_SLRM_MAX_CELL = 50
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
    # Paralel islenecek surec sayisi. 0 = otomatik (cekirdek-1), 1 = tek cekirdek.
    "workers": 0,
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
    workers: int = 0
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


def _resolve_rvt_slrm_radius(radius_m: float, pixel_size: float) -> Tuple[float, int, int]:
    """RVT slrm kullaniliyorsa metre yaricapi desteklenen piksel araligina oturtur."""
    requested_cell = _radius_to_cells(radius_m, pixel_size)
    if rvt_vis is not None and hasattr(rvt_vis, "slrm"):
        effective_cell = int(min(RVT_SLRM_MAX_CELL, max(RVT_SLRM_MIN_CELL, requested_cell)))
    else:
        effective_cell = requested_cell
    return float(effective_cell * pixel_size), requested_cell, effective_cell


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
        # RVT slrm trend yaricapini gecerli 10-50 piksel araligina sabitle.
        rvt_radius_cell = int(min(RVT_SLRM_MAX_CELL, max(RVT_SLRM_MIN_CELL, radius_cell)))
        for kwargs in (
            {"dem": filled, "radius_cell": rvt_radius_cell, "no_data": None},
            {"dem": filled, "radius_cell": rvt_radius_cell},
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


def _auto_halo(config: TopoMapConfig, pixel_size: float, *, slrm_radius_m: Optional[float] = None) -> int:
    radii = [2]
    if "svf" in config.products:
        radii.append(_radius_to_cells(config.svf_radius_m, pixel_size))
    if "slrm" in config.products:
        radii.append(_radius_to_cells(slrm_radius_m if slrm_radius_m is not None else config.slrm_radius_m, pixel_size))
        radii.append(int(math.ceil(4.0 * max(float(config.slrm_sigma_px), 0.0))))
    if "slope" in config.products:
        radii.append(2)
    return max(radii)


def _resolve_workers(requested: int) -> int:
    """0/negatif ise otomatik (cekirdek-1); pozitifse cekirdek sayisiyla sinirla."""
    cpu = os.cpu_count() or 1
    if requested and int(requested) > 0:
        return max(1, min(int(requested), cpu))
    return max(1, cpu - 1)


# Tuple semasi: (col_off, row_off, width, height) -- Window picklenebilir degil.
WindowTuple = Tuple[int, int, int, int]


def _compute_window_products(
    dem: np.ndarray,
    core: Tuple[slice, slice],
    *,
    pixel_size: float,
    products: Sequence[str],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Bir blok DEM'inden istenen tum urunlerin cekirdek (halo'suz) dilimini uretir."""
    row_slice, col_slice = core
    out: Dict[str, np.ndarray] = {}
    for product in products:
        if product == "hillshade":
            arr = _hillshade(
                dem,
                pixel_size=pixel_size,
                z_factor=params["z_factor"],
                azimuth=params["azimuth"],
                altitude=params["altitude"],
            )
        elif product == "svf":
            arr = _compute_svf(dem, pixel_size=pixel_size, radius_m=params["svf_radius_m"])
        elif product == "slrm":
            arr = _compute_slrm(
                dem,
                pixel_size=pixel_size,
                radius_m=params["slrm_radius_m"],
                sigma_px=params["slrm_sigma_px"],
            )
        elif product == "slope":
            arr = _slope_degrees(dem, pixel_size=pixel_size, z_factor=params["z_factor"])
        else:  # pragma: no cover - parse_products bunu engeller
            continue
        out[product] = arr[row_slice, col_slice].astype(np.float32, copy=False)
    return out


# Her isci surecte bir kez kurulan ve _worker_compute tarafindan kullanilan durum.
_WORKER_STATE: Dict[str, Any] = {}
_WORKER_CLEANUP_REGISTERED = False


def _worker_close() -> None:
    src = _WORKER_STATE.get("src")
    if src is not None:
        try:
            src.close()
        except Exception:
            pass
    _WORKER_STATE.clear()


def _worker_init(
    input_path: str,
    elevation_band: int,
    halo: int,
    width: int,
    height: int,
    pixel_size: float,
    products: Tuple[str, ...],
    params: Dict[str, Any],
) -> None:
    """ProcessPoolExecutor isci sureci basina bir kez calisir; rasteri acar."""
    global _WORKER_CLEANUP_REGISTERED
    _worker_close()
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        src=rasterio.open(input_path),
        elevation_band=int(elevation_band),
        halo=int(halo),
        width=int(width),
        height=int(height),
        pixel_size=float(pixel_size),
        products=tuple(products),
        params=dict(params),
    )
    if not _WORKER_CLEANUP_REGISTERED:
        import atexit

        atexit.register(_worker_close)
        _WORKER_CLEANUP_REGISTERED = True


def _worker_compute(window_tuple: WindowTuple) -> Tuple[WindowTuple, Dict[str, np.ndarray]]:
    """Isci surecte tek bir blogu okuyup tum urunleri hesaplar."""
    state = _WORKER_STATE
    window = Window(*window_tuple)
    padded_window, core = _window_with_halo(window, state["width"], state["height"], state["halo"])
    dem = _read_band_as_float(state["src"], state["elevation_band"], padded_window)
    result = _compute_window_products(
        dem,
        core,
        pixel_size=state["pixel_size"],
        products=state["products"],
        params=state["params"],
    )
    return window_tuple, result


def generate_topo_maps(
    config: TopoMapConfig,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Path]:
    """Topo haritalarini uretir.

    progress_callback(tamamlanan, toplam): her blok yazildiktan sonra cagrilir.
    cancel_check(): True donerse islem durur ve RuntimeError yukseltilir.
    """
    output_paths = _resolve_output_paths(config)
    if not config.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            joined = "\n  ".join(existing)
            raise FileExistsError(f"Cikti dosyasi zaten var. Uzerine yaz secenegini kullanin:\n  {joined}")
    if "svf" in config.products and rvt_vis is None:
        raise ImportError("SVF haritasi icin rvt-py/rvt gerekli: pip install rvt-py")

    def _cancelled() -> bool:
        return bool(cancel_check()) if cancel_check is not None else False

    with rasterio.open(config.input_path) as src:
        if config.elevation_band <= 0 or config.elevation_band > src.count:
            raise ValueError(f"elevation-band gecersiz: {config.elevation_band}; raster band sayisi={src.count}")

        pixel_size = _pixel_size(src)
        if src.crs is not None and getattr(src.crs, "is_geographic", False):
            LOGGER.warning(
                "Girdi CRS derece tabanli gorunuyor (%s). Radius metre varsayilir; once metre tabanli CRS'e reprojection onerilir.",
                src.crs,
            )
        effective_slrm_radius_m = float(config.slrm_radius_m)
        requested_slrm_cell: Optional[int] = None
        effective_slrm_cell: Optional[int] = None
        if "slrm" in config.products:
            effective_slrm_radius_m, requested_slrm_cell, effective_slrm_cell = _resolve_rvt_slrm_radius(
                config.slrm_radius_m,
                pixel_size,
            )

        halo = (
            int(config.halo)
            if config.halo is not None
            else _auto_halo(config, pixel_size, slrm_radius_m=effective_slrm_radius_m)
        )
        profile = _output_profile(src, compression=config.compression)
        width, height = int(src.width), int(src.height)
        windows = list(_iter_windows(width, height, int(config.chunk)))
        total = len(windows)
        workers = _resolve_workers(config.workers)
        params: Dict[str, Any] = {
            "svf_radius_m": float(config.svf_radius_m),
            "slrm_radius_m": effective_slrm_radius_m,
            "slrm_sigma_px": float(config.slrm_sigma_px),
            "z_factor": float(config.z_factor),
            "azimuth": float(config.azimuth),
            "altitude": float(config.altitude),
        }

        if (
            "slrm" in config.products
            and requested_slrm_cell is not None
            and effective_slrm_cell is not None
            and requested_slrm_cell != effective_slrm_cell
        ):
            LOGGER.info(
                "SLRM yaricapi raster cozumurlugune gore otomatik ayarlandi: %.2f m (%d px) -> %.2f m (%d px). "
                "RVT slrm destek araligi: %d-%d px.",
                float(config.slrm_radius_m),
                requested_slrm_cell,
                effective_slrm_radius_m,
                effective_slrm_cell,
                RVT_SLRM_MIN_CELL,
                RVT_SLRM_MAX_CELL,
            )

        LOGGER.info(
            "Topo haritalari uretiliyor: input=%s band=%d products=%s chunk=%d halo=%d px pixel=%.4f workers=%d blok=%d",
            config.input_path,
            config.elevation_band,
            ",".join(config.products),
            config.chunk,
            halo,
            pixel_size,
            workers,
            total,
        )

        if progress_callback is not None:
            progress_callback(0, total)

        datasets: Dict[str, Any] = {}
        done = 0
        cancelled = False
        use_tqdm = tqdm is not None and progress_callback is None
        try:
            for product, path in output_paths.items():
                datasets[product] = rasterio.open(path, "w", **profile)
                datasets[product].set_band_description(1, product)

            def _write(window_tuple: WindowTuple, result: Dict[str, np.ndarray]) -> None:
                nonlocal done
                window = Window(*window_tuple)
                for product, arr in result.items():
                    datasets[product].write(arr, 1, window=window)
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)

            if workers <= 1:
                # Tek cekirdek: bloklari sirayla isle.
                iterator: Iterable[Window]
                if use_tqdm:
                    iterator = tqdm(windows, desc="Topo maps", unit="block", total=total)
                else:
                    iterator = windows
                for window in iterator:
                    if _cancelled():
                        cancelled = True
                        break
                    padded_window, core = _window_with_halo(window, width, height, halo)
                    dem = _read_band_as_float(src, config.elevation_band, padded_window)
                    result = _compute_window_products(
                        dem, core, pixel_size=pixel_size, products=config.products, params=params
                    )
                    wt: WindowTuple = (
                        int(window.col_off), int(window.row_off),
                        int(window.width), int(window.height),
                    )
                    _write(wt, result)
            else:
                # Cok cekirdek: bloklari isci havuzunda paralel isle, ana surecte yaz.
                window_tuples: List[WindowTuple] = [
                    (int(w.col_off), int(w.row_off), int(w.width), int(w.height)) for w in windows
                ]
                max_pending = workers * 2
                progress = tqdm(total=total, desc="Topo maps", unit="block") if use_tqdm else None
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_worker_init,
                    initargs=(
                        str(config.input_path), config.elevation_band, halo,
                        width, height, pixel_size, tuple(config.products), params,
                    ),
                ) as executor:
                    try:
                        pending = set()
                        cursor = iter(window_tuples)
                        for _ in range(max_pending):
                            wt = next(cursor, None)
                            if wt is None:
                                break
                            pending.add(executor.submit(_worker_compute, wt))
                        while pending:
                            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for future in finished:
                                window_tuple, result = future.result()
                                _write(window_tuple, result)
                                if progress is not None:
                                    progress.update(1)
                            if _cancelled():
                                cancelled = True
                                break
                            for _ in range(len(finished)):
                                wt = next(cursor, None)
                                if wt is None:
                                    break
                                pending.add(executor.submit(_worker_compute, wt))
                    finally:
                        if cancelled:
                            for future in pending:
                                future.cancel()
                            executor.shutdown(wait=True, cancel_futures=True)
                        if progress is not None:
                            progress.close()
        finally:
            for ds in datasets.values():
                ds.close()

    if cancelled:
        raise RuntimeError(f"Islem iptal edildi ({done}/{total} blok yazildi).")

    LOGGER.info("Topo haritalari hazir:")
    for product, path in output_paths.items():
        LOGGER.info("  %s: %s", product, path)
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DTM/DSM bandindan ayri hillshade, SVF, SLRM GeoTIFF haritalari uret. "
            "Arguman verilmezse IDE dostu GUI acilir; --cli ile CONFIG komut satirindan calistirilir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=_config_text("input"), help="Girdi GeoTIFF yolu.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="Qt arayuzunu ac.")
    mode_group.add_argument("--cli", action="store_true", help="Arguman verilmeden de komut satiri uretimini zorla.")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=int(CONFIG["workers"]),
        help="Paralel surec sayisi. 0=otomatik (cekirdek-1), 1=tek cekirdek.",
    )
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
        workers=int(args.workers),
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


def _prepare_windows_qt_import() -> None:
    """Conda/Windows ICU DLL sirasi QtCore importunu bozmasin diye System32 ICU'yu once yukle."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for dll_name in ("icuuc.dll", "icuin.dll", "icudt.dll"):
            dll_path = system32 / dll_name
            if dll_path.exists():
                try:
                    ctypes.WinDLL(str(dll_path))
                except OSError:
                    pass
    except Exception:
        pass


def run_gui(initial_config: TopoMapConfig) -> int:
    """Qt arayuzunu calistirir; Qt yoksa yalnizca bu modda hata verir."""
    _prepare_windows_qt_import()
    qt_backend = ""
    pyside_error: Optional[Exception] = None
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSpinBox,
            QStatusBar,
            QVBoxLayout,
            QWidget,
        )

        qt_backend = "PySide6"
    except ImportError as exc:
        pyside_error = exc
        try:
            from PyQt6.QtCore import Qt, QTimer
            from PyQt6.QtWidgets import (
                QApplication,
                QCheckBox,
                QComboBox,
                QDoubleSpinBox,
                QFileDialog,
                QFormLayout,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPlainTextEdit,
                QProgressBar,
                QPushButton,
                QSpinBox,
                QStatusBar,
                QVBoxLayout,
                QWidget,
            )

            qt_backend = "PyQt6"
        except ImportError as pyqt_error:
            raise ImportError(
                "Qt arayuzu icin PySide6 veya PyQt6 gerekli.\n"
                f"PySide6 hatasi: {pyside_error}\n"
                f"PyQt6 hatasi: {pyqt_error}\n"
                "Kurulum: pip install PySide6"
            ) from pyqt_error

    import queue
    import threading
    import time
    import traceback

    class QueueLogHandler(logging.Handler):
        def __init__(self, output_queue: "queue.Queue[Tuple[str, Any]]") -> None:
            super().__init__()
            self.output_queue = output_queue

        def emit(self, record: logging.LogRecord) -> None:
            try:
                self.output_queue.put(("log", self.format(record)))
            except Exception:
                pass

    class TopoMapWindow(QMainWindow):
        def __init__(self, config: TopoMapConfig) -> None:
            super().__init__()
            self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
            self._thread: Optional[threading.Thread] = None
            self._cancel_event = threading.Event()
            self._last_outputs: Dict[str, Path] = {}
            self._last_input_text = ""

            self.setWindowTitle(f"Topo Harita Uretici ({qt_backend})")
            self.resize(900, 720)
            self.setStatusBar(QStatusBar(self))

            root = QWidget(self)
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            io_group = QGroupBox("Dosyalar", self)
            io_form = QFormLayout(io_group)
            self.edit_input = QLineEdit(str(config.input_path), self)
            self.btn_input = QPushButton("Sec...", self)
            self.btn_input.clicked.connect(self._choose_input)
            input_row = self._row(self.edit_input, self.btn_input)
            io_form.addRow("Girdi GeoTIFF:", input_row)

            self.edit_output_dir = QLineEdit(str(config.output_dir or ""), self)
            self.btn_output = QPushButton("Sec...", self)
            self.btn_output.clicked.connect(self._choose_output_dir)
            output_row = self._row(self.edit_output_dir, self.btn_output)
            io_form.addRow("Cikti klasoru:", output_row)

            self.edit_prefix = QLineEdit(str(config.prefix or ""), self)
            io_form.addRow("Cikti on eki:", self.edit_prefix)

            self.info_label = QLabel("", self)
            self.info_label.setWordWrap(True)
            io_form.addRow("Raster:", self.info_label)
            layout.addWidget(io_group)

            product_group = QGroupBox("Dahil edilecek ciktilar", self)
            product_layout = QVBoxLayout(product_group)
            product_check_row = QHBoxLayout()
            self.product_checks: Dict[str, QCheckBox] = {}
            self.product_titles = {
                "hillshade": "Hillshade",
                "svf": "SVF",
                "slrm": "SLRM",
                "slope": "Slope",
            }
            for product, title in (
                ("hillshade", "Hillshade"),
                ("svf", "SVF"),
                ("slrm", "SLRM"),
                ("slope", "Slope"),
            ):
                check = QCheckBox(title, self)
                check.setChecked(product in config.products)
                check.stateChanged.connect(self._refresh_output_preview)
                self.product_checks[product] = check
                product_check_row.addWidget(check)
            product_check_row.addStretch(1)
            self.btn_all_products = QPushButton("Tumunu sec", self)
            self.btn_all_products.clicked.connect(self._select_all_products)
            self.btn_clear_products = QPushButton("Temizle", self)
            self.btn_clear_products.clicked.connect(self._clear_products)
            product_check_row.addWidget(self.btn_all_products)
            product_check_row.addWidget(self.btn_clear_products)
            product_layout.addLayout(product_check_row)
            self.output_preview_label = QLabel("", self)
            self.output_preview_label.setWordWrap(True)
            product_layout.addWidget(self.output_preview_label)
            layout.addWidget(product_group)

            params_group = QGroupBox("Ayarlar", self)
            params_grid = QGridLayout(params_group)
            left_form = QFormLayout()
            right_form = QFormLayout()
            params_grid.addLayout(left_form, 0, 0)
            params_grid.addLayout(right_form, 0, 1)

            self.spin_band = QSpinBox(self)
            self.spin_band.setRange(1, 999)
            self.spin_band.setValue(max(1, int(config.elevation_band)))
            left_form.addRow("Yukseklik bandi:", self.spin_band)

            self.spin_chunk = QSpinBox(self)
            self.spin_chunk.setRange(128, 32768)
            self.spin_chunk.setSingleStep(128)
            self.spin_chunk.setValue(max(128, int(config.chunk)))
            left_form.addRow("Blok boyutu:", self.spin_chunk)

            self.spin_workers = QSpinBox(self)
            self.spin_workers.setRange(0, max(1, (os.cpu_count() or 1) * 2))
            self.spin_workers.setSpecialValueText("Otomatik")
            self.spin_workers.setValue(max(0, int(config.workers)))
            left_form.addRow("Worker:", self.spin_workers)

            self.spin_halo = QSpinBox(self)
            self.spin_halo.setRange(0, 10000)
            self.spin_halo.setSpecialValueText("Otomatik")
            self.spin_halo.setValue(0 if config.halo is None else max(0, int(config.halo)))
            left_form.addRow("Halo:", self.spin_halo)

            self.combo_compression = QComboBox(self)
            for item in ("DEFLATE", "LZW", "NONE"):
                self.combo_compression.addItem(item, item)
            idx = self.combo_compression.findData(str(config.compression).upper())
            self.combo_compression.setCurrentIndex(max(0, idx))
            left_form.addRow("Sikistirma:", self.combo_compression)

            self.check_overwrite = QCheckBox("Mevcut ciktilarin uzerine yaz", self)
            self.check_overwrite.setChecked(bool(config.overwrite))
            left_form.addRow("", self.check_overwrite)

            self.spin_svf_radius = self._double_spin(0.1, 10000.0, config.svf_radius_m, 1.0)
            right_form.addRow("SVF yaricap (m):", self.spin_svf_radius)

            self.spin_slrm_radius = self._double_spin(0.1, 10000.0, config.slrm_radius_m, 1.0)
            self.spin_slrm_radius.editingFinished.connect(self._normalize_slrm_radius_for_current_raster)
            right_form.addRow("SLRM yaricap (m):", self.spin_slrm_radius)

            self.spin_slrm_sigma = self._double_spin(0.1, 512.0, config.slrm_sigma_px, 0.5)
            right_form.addRow("SLRM sigma (px):", self.spin_slrm_sigma)

            self.spin_z_factor = self._double_spin(0.001, 1000.0, config.z_factor, 0.1)
            right_form.addRow("Z factor:", self.spin_z_factor)

            self.spin_azimuth = self._double_spin(0.0, 360.0, config.azimuth, 5.0)
            right_form.addRow("Azimut:", self.spin_azimuth)

            self.spin_altitude = self._double_spin(0.0, 90.0, config.altitude, 5.0)
            right_form.addRow("Yukseklik acisi:", self.spin_altitude)

            layout.addWidget(params_group)

            progress_group = QGroupBox("Calisma", self)
            progress_layout = QVBoxLayout(progress_group)
            self.progress = QProgressBar(self)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            progress_layout.addWidget(self.progress)
            self.log_view = QPlainTextEdit(self)
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumBlockCount(600)
            progress_layout.addWidget(self.log_view, 1)
            layout.addWidget(progress_group, 1)

            button_row = QHBoxLayout()
            self.btn_start = QPushButton("Uret", self)
            self.btn_start.clicked.connect(self._start_generation)
            self.btn_cancel = QPushButton("Iptal", self)
            self.btn_cancel.clicked.connect(self._cancel_generation)
            self.btn_cancel.setEnabled(False)
            self.btn_open_output = QPushButton("Cikti klasorunu ac", self)
            self.btn_open_output.clicked.connect(self._open_output_dir)
            self.btn_open_output.setEnabled(False)
            button_row.addStretch(1)
            button_row.addWidget(self.btn_open_output)
            button_row.addWidget(self.btn_cancel)
            button_row.addWidget(self.btn_start)
            layout.addLayout(button_row)

            self.timer = QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self._poll_queue)
            self.timer.start()
            self.edit_input.textChanged.connect(self._refresh_raster_info)
            self.edit_input.textChanged.connect(self._refresh_output_preview)
            self.edit_output_dir.textChanged.connect(self._refresh_output_preview)
            self.edit_prefix.textChanged.connect(self._refresh_output_preview)
            self._refresh_raster_info()
            self._refresh_output_preview()

        def _row(self, *widgets: QWidget) -> QWidget:
            row = QWidget(self)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            for index, widget in enumerate(widgets):
                layout.addWidget(widget, 1 if index == 0 else 0)
            return row

        def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox(self)
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(3)
            spin.setSingleStep(float(step))
            spin.setValue(float(value))
            return spin

        def _choose_input(self) -> None:
            start_text = self.edit_input.text().strip()
            start = Path(start_text).expanduser().parent if start_text else Path.cwd()
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Girdi GeoTIFF sec",
                str(start),
                "GeoTIFF (*.tif *.tiff);;Tum dosyalar (*.*)",
            )
            if path:
                self.edit_input.setText(path)

        def _choose_output_dir(self) -> None:
            start_text = self.edit_output_dir.text().strip()
            if not start_text:
                input_text = self.edit_input.text().strip()
                start_text = str(Path(input_text).expanduser().parent) if input_text else str(Path.cwd())
            path = QFileDialog.getExistingDirectory(self, "Cikti klasoru sec", start_text)
            if path:
                self.edit_output_dir.setText(path)

        def _select_all_products(self) -> None:
            for check in self.product_checks.values():
                check.setChecked(True)
            self._refresh_output_preview()

        def _clear_products(self) -> None:
            for check in self.product_checks.values():
                check.setChecked(False)
            self._refresh_output_preview()

        def _output_preview_base(self) -> Tuple[Path, str]:
            input_text = self.edit_input.text().strip()
            input_path = Path(input_text).expanduser() if input_text else Path()
            output_text = self.edit_output_dir.text().strip()
            output_dir = Path(output_text).expanduser() if output_text else input_path.parent
            prefix = self.edit_prefix.text().strip() or (input_path.stem if input_text else "cikti")
            return output_dir, prefix

        def _refresh_output_preview(self, *_args: Any) -> None:
            if not hasattr(self, "output_preview_label"):
                return
            selected = tuple(product for product, check in self.product_checks.items() if check.isChecked())
            if not selected:
                self.output_preview_label.setText("Secili cikti yok. Uretmek icin en az bir cikti secin.")
                return
            output_dir, prefix = self._output_preview_base()
            lines = []
            for product in selected:
                output_path = output_dir / f"{prefix}_{product}.tif"
                lines.append(f"{self.product_titles.get(product, product)} -> {output_path}")
            self.output_preview_label.setText("Uretilecek dosyalar:\n" + "\n".join(lines))

        def _normalize_slrm_radius_for_current_raster(self) -> None:
            input_text = self.edit_input.text().strip()
            if not input_text:
                return
            path = Path(input_text).expanduser()
            if not path.exists():
                return
            try:
                with rasterio.open(path) as src:
                    pixel_size = _pixel_size(src)
                requested_m = float(self.spin_slrm_radius.value())
                effective_m, requested_cell, effective_cell = _resolve_rvt_slrm_radius(requested_m, pixel_size)
                if requested_cell == effective_cell:
                    return
                self.spin_slrm_radius.blockSignals(True)
                try:
                    self.spin_slrm_radius.setValue(effective_m)
                finally:
                    self.spin_slrm_radius.blockSignals(False)
                self.statusBar().showMessage(
                    f"SLRM yaricapi otomatik ayarlandi: {requested_m:.2f} m ({requested_cell} px) -> "
                    f"{effective_m:.2f} m ({effective_cell} px)",
                    8000,
                )
            except Exception:
                return

        def _refresh_raster_info(self, *_args: Any) -> None:
            text = self.edit_input.text().strip()
            if text == self._last_input_text:
                return
            self._last_input_text = text
            path = Path(text).expanduser() if text else Path()
            if not text:
                self.info_label.setText("Girdi dosyasi bekleniyor.")
                return
            if not path.exists():
                self.info_label.setText("Dosya bulunamadi.")
                return
            try:
                with rasterio.open(path) as src:
                    band_count = int(src.count)
                    self.spin_band.setMaximum(max(1, band_count))
                    if self.spin_band.value() > band_count:
                        self.spin_band.setValue(band_count)
                    pixel_size = _pixel_size(src)
                    crs_text = src.crs.to_string() if src.crs is not None else "CRS yok"
                    self.info_label.setText(
                        f"{src.width} x {src.height} px  |  {band_count} bant  |  "
                        f"pixel={pixel_size:.4f}  |  {crs_text}"
                    )
                if not self.edit_prefix.text().strip():
                    self.edit_prefix.setText(path.stem)
                if not self.edit_output_dir.text().strip():
                    self.edit_output_dir.setPlaceholderText(str(path.parent))
                self._normalize_slrm_radius_for_current_raster()
                self._refresh_output_preview()
            except Exception as exc:
                self.info_label.setText(f"Raster okunamadi: {exc}")

        def _selected_products(self) -> Tuple[str, ...]:
            products = tuple(product for product, check in self.product_checks.items() if check.isChecked())
            if not products:
                raise ValueError("En az bir urun secin.")
            return products

        def _build_config(self) -> TopoMapConfig:
            input_text = self.edit_input.text().strip()
            if not input_text:
                raise ValueError("Girdi GeoTIFF yolu bos.")
            input_path = Path(input_text).expanduser()
            if not input_path.exists():
                raise FileNotFoundError(f"Girdi bulunamadi: {input_path}")
            output_text = self.edit_output_dir.text().strip()
            return TopoMapConfig(
                input_path=input_path,
                elevation_band=int(self.spin_band.value()),
                output_dir=Path(output_text).expanduser() if output_text else None,
                prefix=self.edit_prefix.text().strip() or None,
                products=self._selected_products(),
                chunk=int(self.spin_chunk.value()),
                workers=int(self.spin_workers.value()),
                halo=None if int(self.spin_halo.value()) == 0 else int(self.spin_halo.value()),
                svf_radius_m=float(self.spin_svf_radius.value()),
                slrm_radius_m=float(self.spin_slrm_radius.value()),
                slrm_sigma_px=float(self.spin_slrm_sigma.value()),
                z_factor=float(self.spin_z_factor.value()),
                azimuth=float(self.spin_azimuth.value()),
                altitude=float(self.spin_altitude.value()),
                compression=str(self.combo_compression.currentData()),
                overwrite=bool(self.check_overwrite.isChecked()),
                log_level=str(initial_config.log_level),
            )

        def _set_controls_enabled(self, enabled: bool) -> None:
            widgets = [
                self.edit_input,
                self.btn_input,
                self.edit_output_dir,
                self.btn_output,
                self.edit_prefix,
                *self.product_checks.values(),
                self.btn_all_products,
                self.btn_clear_products,
                self.spin_band,
                self.spin_chunk,
                self.spin_workers,
                self.spin_halo,
                self.combo_compression,
                self.check_overwrite,
                self.spin_svf_radius,
                self.spin_slrm_radius,
                self.spin_slrm_sigma,
                self.spin_z_factor,
                self.spin_azimuth,
                self.spin_altitude,
            ]
            for widget in widgets:
                widget.setEnabled(enabled)
            self.btn_start.setEnabled(enabled)
            self.btn_cancel.setEnabled(not enabled)

        def _start_generation(self) -> None:
            if self._thread is not None and self._thread.is_alive():
                return
            try:
                config = self._build_config()
            except Exception as exc:
                QMessageBox.critical(self, "Topo Harita Uretici", str(exc))
                return

            self._cancel_event.clear()
            self._last_outputs = {}
            self.btn_open_output.setEnabled(False)
            self.log_view.clear()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.statusBar().showMessage("Uretim baslatiliyor...")
            self._set_controls_enabled(False)

            def _progress(done: int, total: int) -> None:
                self._queue.put(("progress", (int(done), max(1, int(total)))))

            def _cancelled() -> bool:
                return self._cancel_event.is_set()

            def _worker() -> None:
                handler = QueueLogHandler(self._queue)
                handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
                LOGGER.addHandler(handler)
                try:
                    started = time.monotonic()
                    outputs = generate_topo_maps(config, progress_callback=_progress, cancel_check=_cancelled)
                    self._queue.put(("done", (outputs, time.monotonic() - started)))
                except Exception as exc:
                    self._queue.put(("error", (str(exc), traceback.format_exc())))
                finally:
                    LOGGER.removeHandler(handler)

            self._thread = threading.Thread(target=_worker, name="topo-map-generator", daemon=True)
            self._thread.start()

        def _cancel_generation(self) -> None:
            self._cancel_event.set()
            self.btn_cancel.setEnabled(False)
            self.statusBar().showMessage("Iptal isteniyor; aktif bloklar bitince duracak...")

        def _poll_queue(self) -> None:
            while True:
                try:
                    kind, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    done, total = payload
                    self.progress.setRange(0, max(1, int(total)))
                    self.progress.setValue(max(0, min(int(done), int(total))))
                    self.progress.setFormat(f"{done} / {total} blok (%p%)")
                    self.statusBar().showMessage(f"Uretim suruyor: {done}/{total} blok")
                elif kind == "log":
                    self.log_view.appendPlainText(str(payload))
                elif kind == "done":
                    outputs, elapsed = payload
                    self._last_outputs = dict(outputs)
                    self._set_controls_enabled(True)
                    self.btn_cancel.setEnabled(False)
                    self.btn_open_output.setEnabled(bool(outputs))
                    self.statusBar().showMessage(f"Tamamlandi ({elapsed:.1f} sn)", 8000)
                    paths = "\n".join(f"{product}: {path}" for product, path in outputs.items())
                    QMessageBox.information(self, "Topo Harita Uretici", f"Topo haritalari hazir:\n\n{paths}")
                elif kind == "error":
                    message, detail = payload
                    self._set_controls_enabled(True)
                    self.btn_cancel.setEnabled(False)
                    self.statusBar().showMessage("Uretim durdu", 8000)
                    if detail:
                        self.log_view.appendPlainText(detail)
                    if self._cancel_event.is_set():
                        QMessageBox.warning(self, "Topo Harita Uretici", f"Islem iptal edildi.\n\n{message}")
                    else:
                        QMessageBox.critical(self, "Topo Harita Uretici", f"Uretim basarisiz oldu:\n\n{message}")

        def _open_output_dir(self) -> None:
            if self._last_outputs:
                directory = next(iter(self._last_outputs.values())).parent
            else:
                text = self.edit_output_dir.text().strip()
                directory = Path(text).expanduser() if text else Path.cwd()
            try:
                if sys.platform == "win32":
                    os.startfile(str(directory))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    import subprocess

                    subprocess.Popen(["open", str(directory)])
                else:
                    import subprocess

                    subprocess.Popen(["xdg-open", str(directory)])
            except Exception as exc:
                QMessageBox.warning(self, "Topo Harita Uretici", f"Klasor acilamadi:\n{exc}")

        def closeEvent(self, event: Any) -> None:
            if self._thread is not None and self._thread.is_alive():
                answer = QMessageBox.question(
                    self,
                    "Topo Harita Uretici",
                    "Uretim devam ediyor. Iptal isteyip pencereyi acik birakayim mi?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._cancel_generation()
                    event.ignore()
                    return
            event.accept()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("Topo Harita Uretici")
    win = TopoMapWindow(initial_config)
    win.show()
    return int(app.exec())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    _configure_logging(args.log_level)
    try:
        config = config_from_args(args)
        if bool(args.gui) or (not raw_args and not bool(args.cli)):
            return run_gui(config)
        if not config.input_path.exists():
            raise FileNotFoundError(f"Girdi bulunamadi: {config.input_path}")
        generate_topo_maps(config)
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
