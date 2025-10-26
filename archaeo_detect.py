"""
Command-line tool for archaeological feature detection from multi-band GeoTIFFs.

The pipeline performs the following steps:
1. Reads RGB, DSM, and DTM bands from a single raster and prepares per-tile workloads.
2. Derives relief visualisation layers (sky-view factor, openness, local relief model,
   slope) from the DTM via the Relief Visualization Toolbox (rvt-py).
3. Builds a 9-channel tensor stack [R, G, B, SVF, PosOpen, NegOpen, LRM, Slope, nDSM],
   normalises channels with robust 2-98 percentile scaling, and handles nodata safely.
4. Runs tiled iInference with a pretrained U-Net style model (segmentation_models_pytorch),
   blending overlapping tiles to construct a seamless probability map.
5. Applies optional tall-object masking, thresholds to a binary mask, and writes both
   probability and mask GeoTIFF outputs with compression, preserving georeferencing.
6. Optionally vectorises detected features into polygons with area/score attributes and
   exports a GeoPackage for GIS workflows.

Assumptions:
- Input mosaic aligns RGB, DSM, and DTM bands in the same CRS/extent/pixel-grid.
- The pretrained model expects nine channels in the prescribed order.
- rvt-py and torch-compatible compute environment are installed ahead of execution.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, TextIO, Tuple, TypeVar

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    gaussian_gradient_magnitude,
    gaussian_laplace,
    grey_closing,
    grey_opening,
    uniform_filter,
)
import torch
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover - dependency required at runtime
    raise ImportError(
        "Install segmentation_models_pytorch: pip install segmentation-models-pytorch"
    ) from exc

try:
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform
except ImportError as exc:  # pragma: no cover - dependency required for vectorisation
    raise ImportError("Install shapely: pip install shapely") from exc

try:
    from pyproj import CRS, Transformer
except ImportError as exc:  # pragma: no cover - dependency required for vectorisation
    raise ImportError("Install pyproj: pip install pyproj") from exc

try:
    from rvt import vis as rvt_vis
except ImportError as exc:  # pragma: no cover - rvt is a hard requirement
    raise ImportError("Install rvt-py: pip install rvt") from exc

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # geopandas is optional; fallback writer is used otherwise

try:
    import fiona
except ImportError:
    fiona = None

try:
    from osgeo import gdal
except ImportError:  # pragma: no cover - optional, used to tame GDAL warnings
    gdal = None

LOGGER = logging.getLogger("archaeo_detect")

T = TypeVar("T")

_GDAL_HANDLER_INSTALLED = False
_SUPPRESSED_GDAL_MESSAGES = (
    "TIFFReadDirectory:Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel",
)


def install_gdal_warning_filter() -> None:
    """Suppress noisy but harmless GDAL warnings about mis-declared TIFF extrasamples."""
    global _GDAL_HANDLER_INSTALLED
    if _GDAL_HANDLER_INSTALLED or gdal is None:
        return

    def _handler(err_class: int, err_num: int, err_msg: str) -> None:
        if any(err_msg.startswith(prefix) for prefix in _SUPPRESSED_GDAL_MESSAGES):
            LOGGER.debug("Suppressed GDAL warning: %s", err_msg)
            return
        try:
            gdal.CPLDefaultErrorHandler(err_class, err_num, err_msg)
        except AttributeError:  # pragma: no cover - safety for stripped bindings
            LOGGER.warning("GDAL warning: %s", err_msg)

    gdal.PushErrorHandler(_handler)
    _GDAL_HANDLER_INSTALLED = True


def progress_bar(iterable: Iterable[T], **tqdm_kwargs) -> tqdm:
    """Wrapper around tqdm that prefers stdout so CLI runners can render progress."""
    stream: Optional[TextIO]
    if sys.stdout:
        stream = sys.stdout
    elif sys.stderr:
        stream = sys.stderr
    else:  # pragma: no cover - exceptionally rare in CLI usage
        stream = None
    if stream is not None:
        tqdm_kwargs.setdefault("file", stream)
    tqdm_kwargs.setdefault("dynamic_ncols", True)
    tqdm_kwargs.setdefault("mininterval", 0.5)
    tqdm_kwargs.setdefault("smoothing", 0.1)
    tqdm_kwargs.setdefault("leave", False)
    return tqdm(iterable, **tqdm_kwargs)

# Turkce not: Bu arac, ayni jeokoordinata sahip RGB, DSM ve DTM bantlarini tek bir GeoTIFF icinde bekler;
# DTM'den SVF/Openness/LRM/slope turetilip RGB ile birlikte 9 kanalli tensore donusturulur.
# Pretrained U-Net modeliyle kaydirma penceresi cikarimi yapilir; olasilik ve ikili maske GeoTIFF olarak yazilir;
# istenirse poligonlastirilir.

# ==== Pipeline defaults (tum ayarlar burada) ====
# Bu dataclass tum parametreleri tek merkezde toplar ve her alanin uzerindeki yorum parametrenin ne yaptigini aciklar.


@dataclass
class PipelineDefaults:
    # Girdide kullanilacak cok bantli GeoTIFF dosyasinin tam yolu; RGB, DSM ve DTM bantlarini icermelidir.
    input: str = field(
        default=r"C:\Users\ertug\Nextcloud\arkeolojik_alan_tespit\kesif_alani.tif",
        metadata={"help": "Full path to the multi-band GeoTIFF (RGB + DSM + DTM)."},
    )
    # Egitilmis .pth agirlik dosyasi; None ise zero-shot veya sablon agirlik akisi devreye girer.
    weights: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained weights (*.pth); leave empty to rely on zero-shot/template logic."},
    )
    # R,G,B,DSM,DTM bantlarinin 1-indexli sirasi; bir bant yoksa 0 yazabilirsiniz.
    bands: str = field(
        default="1,2,3,4,5",
        metadata={"help": "Comma separated band order (1-based) for R,G,B,DSM,DTM; use 0 if a band is absent."},
    )
    # Teker teker islenecek karo boyutu (piksel); daha buyuk deger GPU bellegini artirir, daha az karo anlamina gelir.
    tile: int = field(
        default=1024,
        metadata={"help": "Tile size in pixels processed per forward pass; larger tiles need more memory."},
    )
    # Komsu karolarin bindirme miktari (piksel); dikis hatlarini azaltir fakat toplam karo sayisini artirir.
    overlap: int = field(
        default=256,
        metadata={"help": "Overlap between tiles in pixels; balances seam quality vs. throughput."},
    )
    # DL olasilik haritasini maskeye cevirmek icin kullanilan esik (0-1 arasi); dusuk deger daha fazla aday uretir.
    th: float = field(
        default=0.6,
        metadata={"help": "Probability threshold applied to the DL output when generating the binary mask."},
    )
    # CUDA mevcudiyetinde otomatik float16 (half precision) kullan; bellek ve hiz kazanci saglar.
    half: bool = field(
        default=True,
        metadata={"help": "Use float16 autocast on CUDA devices (falls back to float32 when CUDA is unavailable)."},
    )
    # Tum karolar icin tek bir robust 2-98 yuzdelik normalizasyonu uygula; global kontrast tutarliligi saglar.
    global_norm: bool = field(
        default=True,
        metadata={"help": "Estimate a global percentile normalisation and reuse it for every tile."},
    )
    # Global normalizasyon icin ornekleyecegimiz karo sayisi; daha yuksek deger istatistikleri saglamlastirir.
    norm_sample_tiles: int = field(
        default=32,
        metadata={"help": "Number of tiles sampled to compute the global 2nd/98th percentiles."},
    )
    # Karo sinirlarinda kosinusu andiran agirliklarla yumusatma yaparak dikis izlerini azalt.
    feather: bool = field(
        default=True,
        metadata={"help": "Feather tile borders with smooth weights; disable if you need raw tile edges."},
    )
    # Klasik kabartma tabanli (RVT, Hessian, morfoloji) pipeline'i calistir.
    classic: bool = field(
        default=True,
        metadata={"help": "Run the classical relief pipeline (RVT log, Hessian, morphological filters)."},
    )
    # Klasik pipeline icin hangi modlar kullanilacak (rvtlog,hessian,morph,combo).
    classic_modes: str = field(
        default="combo",
        metadata={"help": "Classic modes to compute (e.g. 'combo', 'rvtlog,hessian')."},
    )
    # Her klasik modu ayri dosya olarak kaydet; referans icin kullanisli.
    classic_save_intermediate: bool = field(
        default=True,
        metadata={"help": "Write individual classic mode rasters besides the combined output."},
    )
    # Klasik pipeline icin sabit maske esigi (0-1); None ise otomatik Otsu kullanilir.
    classic_th: Optional[float] = field(
        default=None,
        metadata={"help": "Manual threshold for classical mask; leave None to use automatic Otsu."},
    )
    # DL ve klasik olasiliklarini birlestiren ekstra ciktiyi etkinlestir.
    fuse: bool = field(
        default=True,
        metadata={"help": "Blend DL and classical maps into a fused probability/mask pair."},
    )
    # Birlesimde DL olasiliginin agirligi (0-1); 1 yalniz DL, 0 yalniz klasik.
    alpha: float = field(
        default=0.5,
        metadata={"help": "Fusion mixing factor; 1.0 keeps only DL scores, 0.0 keeps only classic."},
    )
    # nDSM yuksekliklerine gore yuksek objeleri maskele; bu esigi asan piksel tespit edilmez.
    mask_talls: Optional[float] = field(
        default=2.5,
        metadata={"help": "Suppress detections where nDSM exceeds this height in metres."},
    )
    # Vektorelestirmede kabul edilecek minimum poligon alani (metrekare).
    min_area: float = field(
        default=80.0,
        metadata={"help": "Minimum polygon area to keep during vectorisation (square metres)."},
    )
    # Poligonlar icin Douglas-Peucker basitlestirme toleransi (metre); None ise uygulanmaz.
    simplify: Optional[float] = field(
        default=None,
        metadata={"help": "Polygon simplification tolerance in metres; set None to keep full detail."},
    )
    # Tespitleri GPKG poligon dosyasina aktar.
    vectorize: bool = field(
        default=True,
        metadata={"help": "Export detection masks as polygons (GeoPackage)."},
    )
    # segmentation_models_pytorch icinden secilecek mimari adi.
    arch: str = field(
        default="Unet",
        metadata={"help": "segmentation_models_pytorch architecture name (e.g. Unet, UnetPlusPlus)."},
    )
    # Tek model kosulurken kullanilacak varsayilan encoder.
    encoder: str = field(
        default="resnet34",
        metadata={"help": "Default encoder backbone when running a single model."},
    )
    # Virgul ayrimli encoder listesi; 'all' hepsini, 'none' yalniz varsayilan encoderi kullanir.
    encoders: str = field(
        default="all",
        metadata={"help": "Comma separated encoders to evaluate ('all', 'none', or explicit list)."},
    )
    # Encoder bazli agirlik dosya sablonu; {encoder} yer tutucusunu destekler.
    weights_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for encoder-specific weights (e.g. models/unet_{encoder}_9ch_best.pth)."},
    )
    # Cikti dosyalari icin on-ek veya dizin; None ise girdi adindan otomatik uretilir.
    out_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Output prefix or directory; defaults to using the input file stem."},
    )
    # Rastgelelik icin tohum degeri; numpy, torch ve random modullerine aktarilir.
    seed: int = field(
        default=42,
        metadata={"help": "Random seed applied to numpy, torch, and Python random."},
    )
    # Trained weights olmadan ImageNet encoderini 9 kanala genisletip zero-shot calistir.
    zero_shot_imagenet: bool = field(
        default=True,
        metadata={"help": "Enable zero-shot inference by inflating ImageNet encoders to 9 channels."},
    )
    # Log seviye kontrolu; 0 WARNING, 1 INFO, 2 DEBUG.
    verbose: int = field(
        default=1,
        metadata={"help": "Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG."},
    )


DEFAULTS = PipelineDefaults()


def default_for(name: str):
    return getattr(DEFAULTS, name)


def help_for(name: str) -> str:
    for f in fields(PipelineDefaults):
        if f.name == name:
            return f.metadata.get("help", "")
    raise KeyError(name)


def cli_help(name: str, extra: Optional[str] = None) -> str:
    base = help_for(name)
    if extra:
        return f"{base} {extra}".strip()
    return base


def build_config_from_args(args: argparse.Namespace) -> PipelineDefaults:
    values = {f.name: getattr(args, f.name) for f in fields(PipelineDefaults)}
    return PipelineDefaults(**values)



def configure_logging(verbosity: int) -> None:
    """Configure global logging level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seeds(seed: int) -> None:
    """Enforce deterministic behaviour across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_windowed(
    raster: rasterio.io.DatasetReader | str | Path,
    window: Window,
    indexes: Optional[Sequence[int]] = None,
    boundless: bool = True,
) -> np.ndarray:
    """Read a raster window from either an open dataset or a file path."""
    if hasattr(raster, "read"):
        dataset = raster
        return dataset.read(indexes=indexes, window=window, boundless=boundless)
    with rasterio.open(raster) as dataset:
        return dataset.read(indexes=indexes, window=window, boundless=boundless)


def compute_ndsm(dsm: Optional[np.ndarray], dtm: np.ndarray) -> np.ndarray:
    """Compute normalised DSM (nDSM). Returns zeros if DSM is missing."""
    ndsm = np.full_like(dtm, np.nan, dtype=np.float32)
    dtm_valid = np.isfinite(dtm)
    if dsm is None:
        ndsm[dtm_valid] = 0.0
        return ndsm
    dsm_valid = np.isfinite(dsm)
    mask = dtm_valid & dsm_valid
    ndsm[mask] = (dsm - dtm)[mask]
    return ndsm


def percentile_clip(
    arr: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Clip array values between given percentiles while respecting NaNs."""
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lower = np.percentile(arr[valid], low)
    upper = np.percentile(arr[valid], high)
    if upper - lower <= 1e-6:
        clipped = np.clip(arr - lower, 0.0, 1.0)
    else:
        clipped = np.clip((arr - lower) / (upper - lower), 0.0, 1.0)
    clipped[~valid] = 0.0
    return clipped.astype(np.float32)


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Convenience wrapper for percentile-based 0-1 scaling."""
    return percentile_clip(arr, low=2.0, high=98.0)


def _otsu_threshold_0to1(arr: np.ndarray, valid: np.ndarray) -> float:
    """Compute Otsu threshold on 0..1 data respecting validity mask."""
    data = arr[valid]
    if data.size == 0:
        return 0.5
    clipped = np.clip(data, 0.0, 1.0)
    hist, bin_edges = np.histogram(clipped, bins=256, range=(0.0, 1.0))
    if not np.any(hist):
        return float(np.nanmean(clipped)) if np.any(np.isfinite(clipped)) else 0.5
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    cumulative = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    global_mean = cumulative_mean[-1]
    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (1.0 - cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = numerator / np.maximum(denominator, 1e-12)
    sigma_b[cumulative == 0.0] = 0.0
    sigma_b[cumulative == 1.0] = 0.0
    best = np.argmax(sigma_b)
    threshold = (bin_edges[best] + bin_edges[best + 1]) * 0.5
    return float(np.clip(threshold, 0.0, 1.0))


def _local_variance(arr: np.ndarray, size: int = 7) -> np.ndarray:
    """Estimate local variance using uniform filtering."""
    if size <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    arr_f = arr.astype(np.float32, copy=False)
    mean = uniform_filter(arr_f, size=size, mode="nearest")
    mean_sq = uniform_filter(arr_f * arr_f, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return var.astype(np.float32)


def _hessian_response(im: np.ndarray, sigma: float) -> np.ndarray:
    """Compute normalised second eigenvalue magnitude from Hessian."""
    im_f = im.astype(np.float32, copy=False)
    Hxx = gaussian_filter(im_f, sigma=sigma, order=(2, 0), mode="nearest")
    Hxy = gaussian_filter(im_f, sigma=sigma, order=(1, 1), mode="nearest")
    Hyy = gaussian_filter(im_f, sigma=sigma, order=(0, 2), mode="nearest")
    tr = Hxx + Hyy
    det = Hxx * Hyy - Hxy * Hxy
    tmp = np.sqrt(np.maximum((tr * tr) / 4.0 - det, 0.0))
    l2 = tr / 2.0 - tmp
    return _norm01(np.abs(l2))


def robust_norm(
    arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    """Robust per-channel min-max normalisation with percentile clipping."""
    if arr.ndim != 3:
        raise ValueError("robust_norm expects a (C, H, W) array.")
    normed = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        channel = arr[idx]
        normed[idx] = percentile_clip(channel, low=p_low, high=p_high)
    return normed


def robust_norm_fixed(arr: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Normalize (C,H,W) with fixed per-channel lows/highs; NaNsâ†’0."""
    assert arr.ndim == 3
    out = np.zeros_like(arr, dtype=np.float32)
    C = arr.shape[0]
    for c in range(C):
        ch = arr[c]
        low = float(lows[c])
        high = float(highs[c])
        if high - low <= 1e-6:
            norm = np.clip(ch - low, 0.0, 1.0)
        else:
            norm = np.clip((ch - low) / (high - low), 0.0, 1.0)
        norm[~np.isfinite(ch)] = 0.0
        out[c] = norm.astype(np.float32)
    return out


def fill_nodata(
    arr: np.ndarray, fill_value: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaNs in an array with a constant (fallback median if not provided)."""
    filled = arr.astype(np.float32, copy=True)
    invalid_mask = ~np.isfinite(filled)
    if not np.any(invalid_mask):
        return filled, ~invalid_mask
    if fill_value is None:
        valid_vals = filled[~invalid_mask]
        fill_value = float(np.nanmedian(valid_vals)) if valid_vals.size else 0.0
    filled[invalid_mask] = fill_value
    return filled, ~invalid_mask


def compute_derivatives_with_rvt(
    dtm: np.ndarray,
    pixel_size: float,
    radii: Sequence[float] = (5.0, 10.0, 20.0, 30.0, 50.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVF, openness, LRM, and slope using RVT routines."""
    dtm_filled, valid_mask = fill_nodata(dtm)
    # Compatibility helpers for rvt-py API differences across versions
    def _as_float32_array(result_obj, name: str) -> np.ndarray:
        """Best-effort conversion of various RVT return types to float32 ndarray."""
        # Direct ndarray
        if isinstance(result_obj, np.ndarray):
            return result_obj.astype(np.float32)
        # Masked array
        if isinstance(result_obj, np.ma.MaskedArray):
            return np.ma.filled(result_obj, np.nan).astype(np.float32)
        # Dict of outputs (pick most relevant or first ndarray)
        if isinstance(result_obj, dict):
            preferred_keys = (
                "svf",
                "SVF",
                "positive_openness",
                "pos_open",
                "openness_positive",
                "negative_openness",
                "neg_open",
                "openness_negative",
                "lrm",
                "LRM",
                "slope",
                "Slope",
            )
            for k in preferred_keys:
                if k in result_obj and isinstance(result_obj[k], (np.ndarray, np.ma.MaskedArray)):
                    return _as_float32_array(result_obj[k], name)
            # fallback: first ndarray-looking value
            for v in result_obj.values():
                try:
                    arr = _as_float32_array(v, name)
                    if isinstance(arr, np.ndarray):
                        return arr
                except Exception:
                    continue
            raise TypeError(f"RVT '{name}' returned dict without ndarray values.")
        # Sequence of outputs: take first array-like
        if isinstance(result_obj, (list, tuple)):
            for v in result_obj:
                try:
                    arr = _as_float32_array(v, name)
                    if isinstance(arr, np.ndarray):
                        return arr
                except Exception:
                    continue
            raise TypeError(f"RVT '{name}' returned a sequence without ndarray values.")
        # Generic coercion
        try:
            arr = np.asarray(result_obj)
            if isinstance(arr, np.ndarray):
                return arr.astype(np.float32)
        except Exception:
            pass
        raise TypeError(f"RVT '{name}' returned unsupported type: {type(result_obj)}")

    def _call_with_radius(func, radius_val: float) -> np.ndarray:
        # try different keyword names for radius and nodata
        radius_keys = ("max_radius", "radius", "r_max", "search_radius", "max_search_radius")
        nodata_keys = ("no_data", "nodata")
        last_err: Optional[Exception] = None
        for rk in radius_keys:
            for nk in nodata_keys:
                try:
                    return func(dem=dtm_filled, resolution=pixel_size, **{rk: float(radius_val), nk: None})
                except TypeError as e:
                    last_err = e
                    continue
        # try without nodata keyword
        for rk in radius_keys:
            try:
                return func(dem=dtm_filled, resolution=pixel_size, **{rk: float(radius_val)})
            except TypeError as e:
                last_err = e
                continue
        # final fallback: minimal signature
        try:
            return func(dem=dtm_filled, resolution=pixel_size)
        except Exception as e:  # pragma: no cover - defensive
            if last_err is not None:
                raise last_err
            raise e

    svf_layers: List[np.ndarray] = []
    for radius in radii:
        svf_res = _call_with_radius(rvt_vis.sky_view_factor, float(radius))
        svf = _as_float32_array(svf_res, "sky_view_factor")
        svf_layers.append(svf)
    svf_avg = np.mean(np.stack(svf_layers, axis=0), axis=0)
    # Openness variants differ across rvt versions; compute both pos/neg robustly
    def _compute_openness(radius_val: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(rvt_vis, "positive_openness") and hasattr(rvt_vis, "negative_openness"):
            pos = _as_float32_array(
                _call_with_radius(rvt_vis.positive_openness, radius_val),
                "positive_openness",
            )
            neg = _as_float32_array(
                _call_with_radius(rvt_vis.negative_openness, radius_val),
                "negative_openness",
            )
            return pos, neg
        # Single 'openness' function that returns both
        if hasattr(rvt_vis, "openness"):
            res = _call_with_radius(rvt_vis.openness, radius_val)
            # dict case
            if isinstance(res, dict):
                # try common keys
                pos_keys = ("positive_openness", "openness_positive", "pos_open", "pos")
                neg_keys = ("negative_openness", "openness_negative", "neg_open", "neg")
                pos_arr = None
                neg_arr = None
                for k in pos_keys:
                    if k in res:
                        pos_arr = _as_float32_array(res[k], "positive_openness")
                        break
                for k in neg_keys:
                    if k in res:
                        neg_arr = _as_float32_array(res[k], "negative_openness")
                        break
                if pos_arr is not None and neg_arr is not None:
                    return pos_arr, neg_arr
                # fallback: try to pick two ndarray-like values in order
                arrays = []
                for v in res.values():
                    try:
                        arrays.append(_as_float32_array(v, "openness"))
                    except Exception:
                        continue
                if len(arrays) >= 2:
                    return arrays[0], arrays[1]
                if len(arrays) == 1:
                    return arrays[0], arrays[0]
                LOGGER.warning("RVT openness dict lacked array values; using zeros.")
            # sequence case
            if isinstance(res, (list, tuple)):
                seq = list(res)
                if len(seq) >= 2:
                    return (
                        _as_float32_array(seq[0], "positive_openness"),
                        _as_float32_array(seq[1], "negative_openness"),
                    )
                if len(seq) == 1:
                    arr = _as_float32_array(seq[0], "openness")
                    return arr, arr
                LOGGER.warning("RVT openness sequence empty; using zeros.")
            # single array case
            try:
                arr = _as_float32_array(res, "openness")
                return arr, arr
            except Exception:
                LOGGER.warning("RVT openness returned unsupported type; using zeros.")
        # fallback if no openness available
        zeros = np.zeros_like(dtm_filled, dtype=np.float32)
        return zeros, zeros

    pos_open, neg_open = _compute_openness(float(max(radii)))

    try:
        # LRM may also have different radius/nodata keywords
        def _call_lrm() -> np.ndarray:
            radius_keys = ("search_radius", "radius", "r_max", "max_radius", "max_search_radius")
            nodata_keys = ("no_data", "nodata")
            last_err: Optional[Exception] = None
            for rk in radius_keys:
                for nk in nodata_keys:
                    try:
                        return rvt_vis.local_relief_model(
                            dem=dtm_filled,
                            resolution=pixel_size,
                            **{rk: float(max(radii)), nk: None},
                        )
                    except TypeError as e:
                        last_err = e
                        continue
            for rk in radius_keys:
                try:
                    return rvt_vis.local_relief_model(
                        dem=dtm_filled,
                        resolution=pixel_size,
                        **{rk: float(max(radii))},
                    )
                except TypeError as e:
                    last_err = e
                    continue
            try:
                return rvt_vis.local_relief_model(dem=dtm_filled, resolution=pixel_size)
            except Exception as e:
                if last_err is not None:
                    raise last_err
                raise e

        lrm = _as_float32_array(_call_lrm(), "local_relief_model")
    except AttributeError:
        LOGGER.warning("rvt.vis.local_relief_model missing; using Gaussian fallback.")
        low_pass = gaussian_filter(dtm_filled, sigma=6.0)
        lrm = (dtm_filled - low_pass).astype(np.float32)

    # slope signature also varies across versions; try with/without keywords.
    # If slope is unavailable in rvt.vis, fall back to numpy gradient.
    slope: np.ndarray
    if hasattr(rvt_vis, "slope"):
        try:
            slope_res = rvt_vis.slope(
                dem=dtm_filled,
                resolution=pixel_size,
                output_units="degree",
                no_data=None,
            )
            slope = _as_float32_array(slope_res, "slope")
        except TypeError:
            try:
                slope = _as_float32_array(
                    rvt_vis.slope(dem=dtm_filled, resolution=pixel_size),
                    "slope",
                )
            except Exception as e:  # pragma: no cover - defensive
                raise e
    else:
        LOGGER.warning("rvt.vis.slope missing; using gradient-based fallback.")
        # Compute slope in degrees: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        # Use filled DTM to avoid NaNs breaking gradient
        gy, gx = np.gradient(dtm_filled.astype(np.float32), pixel_size, pixel_size)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    for derivative in (svf_avg, pos_open, neg_open, lrm, slope):
        derivative[~valid_mask] = np.nan
    return svf_avg, pos_open, neg_open, lrm, slope


def _score_rvtlog(dtm: np.ndarray, pixel_size: float) -> np.ndarray:
    """Composite RVT + LoG + gradient classical score."""
    dtm_filled, valid = fill_nodata(dtm)
    svf, _, neg, lrm, _ = compute_derivatives_with_rvt(dtm, pixel_size=pixel_size)
    sigmas = (1.0, 2.0, 4.0, 8.0)
    log_responses = [
        np.abs(gaussian_laplace(dtm_filled, sigma=s, mode="nearest"))
        for s in sigmas
    ]
    blob = np.maximum.reduce(log_responses)
    grad = gaussian_gradient_magnitude(dtm_filled, sigma=1.5, mode="nearest")
    svf_c = 1.0 - _norm01(svf)
    neg_n = _norm01(neg)
    lrm_n = _norm01(lrm)
    var_n = _norm01(_local_variance(dtm_filled, size=7))
    score = (
        0.30 * _norm01(blob)
        + 0.20 * lrm_n
        + 0.15 * _norm01(svf_c)
        + 0.15 * _norm01(grad)
        + 0.10 * neg_n
        + 0.10 * var_n
    )
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_hessian(dtm: np.ndarray) -> np.ndarray:
    """Multi-scale Hessian ridge/valley response."""
    dtm_filled, valid = fill_nodata(dtm)
    sigmas = (1.0, 2.0, 4.0, 8.0)
    responses = [_hessian_response(dtm_filled, sigma=s) for s in sigmas]
    score = np.maximum.reduce(responses)
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_morph(dtm: np.ndarray) -> np.ndarray:
    """Morphological white/black top-hat prominence score."""
    dtm_filled, valid = fill_nodata(dtm)
    radii = (3, 5, 9, 15)
    wth_list = []
    bth_list = []
    for r in radii:
        size = (int(r), int(r))
        opening = grey_opening(dtm_filled, size=size, mode="nearest")
        closing = grey_closing(dtm_filled, size=size, mode="nearest")
        wth = dtm_filled - opening
        bth = closing - dtm_filled
        wth_list.append(_norm01(wth))
        bth_list.append(_norm01(bth))
    hill = np.maximum.reduce(wth_list)
    moat = np.maximum.reduce(bth_list)
    score = np.maximum(hill, moat)
    score[~valid] = np.nan
    return score.astype(np.float32)


def stack_channels(
    rgb: np.ndarray,
    svf: np.ndarray,
    pos_open: np.ndarray,
    neg_open: np.ndarray,
    lrm: np.ndarray,
    slope: np.ndarray,
    ndsm: np.ndarray,
) -> np.ndarray:
    """Stack nine channels in the required order."""
    if rgb.shape[0] != 3:
        raise ValueError("RGB input must have three channels.")
    channels = [
        rgb[0],
        rgb[1],
        rgb[2],
        svf,
        pos_open,
        neg_open,
        lrm,
        slope,
        ndsm,
    ]
    stacked = np.stack([ch.astype(np.float32) for ch in channels], axis=0)
    return stacked


def build_model(
    arch: str = "Unet",
    encoder: str = "resnet34",
    in_ch: int = 9,
) -> torch.nn.Module:
    """nstantiate a segmentation model from segmentation_models_pytorch."""
    if not hasattr(smp, arch):
        raise ValueError(f"Architecture '{arch}' not found in segmentation_models_pytorch.")
    model_cls = getattr(smp, arch)
    model = model_cls(
        encoder_name=encoder,
        in_channels=in_ch,
        classes=1,
        activation=None,
    )
    return model


def inflate_conv1_to_n(conv_w: torch.Tensor, in_ch: int = 9, mode: str = "avg") -> torch.Tensor:
    """
    Inflate a first-conv weight tensor (out,c,kH,kW) to match the requested in_ch.
    - If channels already match, the tensor is returned unchanged.
    - If fewer channels are present, extra channels are filled using the average (or first) channel.
    - If more channels exist than requested, the tensor is truncated.
    """
    if conv_w.ndim != 4:
        raise ValueError("inflate_conv1_to_n expects a 4D convolution weight tensor.")
    out_ch, c_in, kH, kW = conv_w.shape
    if c_in == in_ch:
        return conv_w.clone()
    if in_ch < c_in:
        return conv_w[:, :in_ch].clone()

    if mode == "avg":
        base = conv_w.mean(dim=1, keepdim=True)
    elif mode == "first":
        base = conv_w[:, :1]
    else:
        raise ValueError(f"Unsupported inflate mode: {mode}")

    extra = base.repeat(1, in_ch - c_in, 1, 1)
    return torch.cat([conv_w, extra], dim=1).clone()


def build_model_with_imagenet_inflated(
    arch: str = "Unet", encoder: str = "resnet34", in_ch: int = 9
) -> torch.nn.Module:
    """
    Build a model that uses an ImageNet-pretrained 3-ch encoder and inflate its first conv to in_ch (default 9).
    Decoder weights come from the 3-ch SMP model where shapes match; others remain default-initialized.
    """
    if not hasattr(smp, arch):
        raise ValueError(f"Architecture '{arch}' not found in segmentation_models_pytorch.")

    model_3 = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1, activation=None
    )
    state_3 = model_3.state_dict()

    model_9 = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights=None, in_channels=in_ch, classes=1, activation=None
    )
    state_9 = model_9.state_dict()

    for k, v in state_3.items():
        if k in state_9 and state_9[k].shape == v.shape:
            state_9[k] = v.clone()

    def _first_conv_key(state_dict: dict) -> Optional[str]:
        convs: List[Tuple[str, torch.Tensor]] = []
        for key, val in state_dict.items():
            if not key.endswith(".weight"):
                continue
            if getattr(val, "ndim", 0) != 4:
                continue
            convs.append((key, val))
        if not convs:
            return None

        def priority(item: Tuple[str, torch.Tensor]) -> Tuple[int, str]:
            key, tensor = item
            score = 0
            if "conv1" in key:
                score -= 20
            if "stem" in key:
                score -= 10
            if key.startswith("encoder."):
                score -= 5
            # prefer smaller input channel counts (likely RGB)
            score += int(tensor.shape[1])
            return (score, key)

        # Prefer weights with exactly three input channels, then any other.
        for target_channels in (3, 1, None):
            filtered = [
                item for item in convs if target_channels is None or item[1].shape[1] == target_channels
            ]
            if not filtered:
                continue
            filtered.sort(key=priority)
            return filtered[0][0]

        convs.sort(key=priority)
        return convs[0][0]

    conv1_key_src = _first_conv_key(state_3)
    conv1_key_tgt = _first_conv_key(state_9)
    if conv1_key_src is None or conv1_key_tgt is None:
        raise RuntimeError("Could not locate a 4D first-conv weight to inflate.")

    w3 = state_3[conv1_key_src]  # (out,3,kH,kW)
    state_9[conv1_key_tgt] = inflate_conv1_to_n(w3, in_ch=in_ch, mode="avg")

    model_9.load_state_dict(state_9, strict=False)
    return model_9


def load_weights(
    model: torch.nn.Module,
    weights_path: Path,
    map_location: torch.device,
) -> None:
    """Load model weights with helpful diagnostics on mismatch."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Weights file did not contain a valid state dict.")

    expected_in_ch = model.encoder.conv1.weight.shape[1]
    first_conv_key = None
    for key, tensor in state.items():
        if tensor.ndim == 4:
            first_conv_key = key
            break
    if first_conv_key:
        in_channels = state[first_conv_key].shape[1]
        if in_channels != expected_in_ch:
            raise RuntimeError(
                f"Encoder expects {expected_in_ch} channels but weights provide {in_channels}. "
                "Ensure the model was trained with nine-channel input."
            )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        stripped_state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped_state, strict=True)


def generate_windows(
    width: int, height: int, tile: int, overlap: int
) -> Generator[Tuple[Window, int, int], None, None]:
    """Yield raster windows with the desired tile size and overlap."""
    if overlap >= tile:
        raise ValueError("--overlap must be smaller than --tile.")
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("Stride must be positive; reduce overlap or increase tile size.")
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            win_height = min(tile, height - row)
            win_width = min(tile, width - col)
            yield Window(col, row, win_width, win_height), row, col


def make_feather_weights(h: int, w: int, tile: int, overlap: int) -> np.ndarray:
    """Cosine feathering weights for a tile of size (h,w)."""
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)
    ramp = max(1, min(overlap, tile // 2))

    def _ramp(n: int) -> np.ndarray:
        win = np.ones(n, dtype=np.float32)
        # half-cosine ramps
        t = np.linspace(0, np.pi, ramp, endpoint=False, dtype=np.float32)
        up = (1 - np.cos(t)) * 0.5  # 0â†’1
        win[:ramp] = up
        win[-ramp:] = up[::-1]
        return win

    wr = _ramp(h)
    wc = _ramp(w)
    return np.outer(wr, wc).astype(np.float32)


@dataclass
class InferenceOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    transform: Affine
    crs: Optional[RasterioCRS]


@dataclass
class ClassicModeOutput:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    threshold: float


@dataclass
class ClassicOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float
    per_mode: Dict[str, ClassicModeOutput]


@dataclass
class FusionOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float


def infer_tiled(
    model: torch.nn.Module,
    input_path: Path,
    band_idx: Sequence[int],
    tile: int,
    overlap: int,
    device: torch.device,
    use_half: bool,
    threshold: float,
    mask_talls: Optional[float],
    out_prefix: Path,
    global_norm: bool,
    norm_sample_tiles: int,
    feather: bool,
) -> InferenceOutputs:
    """Run tiled iInference and save outputs."""
    model.eval()
    model.to(device)

    if mask_talls is not None and band_idx[3] <= 0:
        LOGGER.warning("Tall-object masking requested but DSM band missing; disabling.")
        mask_talls = None

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        prob_acc = np.zeros((height, width), dtype=np.float32)
        weight_acc = np.zeros((height, width), dtype=np.float32)
        ndsm_max = np.full((height, width), -np.inf, dtype=np.float32)
        valid_global = np.zeros((height, width), dtype=bool)

        total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
            width / max(tile - overlap, 1)
        )

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_half and device.type == "cuda"
            else nullcontext()
        )

        # ---- Global normalization pre-pass (optional) ----
        fixed_lows = None
        fixed_highs = None
        if global_norm:
            lows_list = []
            highs_list = []
            sampled = 0
            for window, row, col in generate_windows(width, height, tile, overlap):
                def read_band(idx: int) -> Optional[np.ndarray]:
                    if idx <= 0:
                        return None
                    data = src.read(idx, window=window, boundless=True, masked=True)
                    return np.ma.filled(data.astype(np.float32), np.nan)

                rgb_s = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
                dsm_s = read_band(band_idx[3])
                dtm_s = read_band(band_idx[4])
                if dtm_s is None:
                    break

                ndsm_s = compute_ndsm(dsm_s, dtm_s)
                svf_s, pos_s, neg_s, lrm_s, slope_s = compute_derivatives_with_rvt(
                    dtm_s, pixel_size=pixel_size
                )
                stack_s = stack_channels(rgb_s, svf_s, pos_s, neg_s, lrm_s, slope_s, ndsm_s)

                Hs, Ws = stack_s.shape[1], stack_s.shape[2]
                ch = min(256, Hs)
                cw = min(256, Ws)
                r0 = max(0, (Hs - ch) // 2)
                c0 = max(0, (Ws - cw) // 2)
                crop = stack_s[:, r0 : r0 + ch, c0 : c0 + cw]

                Cc = crop.shape[0]
                lows = np.zeros(Cc, dtype=np.float32)
                highs = np.zeros(Cc, dtype=np.float32)
                for ci in range(Cc):
                    v = crop[ci].ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        lows[ci] = 0.0
                        highs[ci] = 1.0
                    else:
                        lows[ci] = float(np.percentile(v, 2.0))
                        highs[ci] = float(np.percentile(v, 98.0))
                lows_list.append(lows)
                highs_list.append(highs)

                sampled += 1
                if sampled >= norm_sample_tiles:
                    break

            if lows_list and highs_list:
                fixed_lows = np.median(np.stack(lows_list, axis=0), axis=0)
                fixed_highs = np.median(np.stack(highs_list, axis=0), axis=0)

        for window, row, col in progress_bar(
            generate_windows(width, height, tile, overlap),
            total=total_tiles,
            desc="Inference",
            unit="tile",
        ):
            def read_band(idx: int) -> Optional[np.ndarray]:
                if idx <= 0:
                    return None
                data = src.read(idx, window=window, boundless=True, masked=True)
                return np.ma.filled(data.astype(np.float32), np.nan)

            rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
            dsm = read_band(band_idx[3])
            dtm = read_band(band_idx[4])
            if dtm is None:
                raise ValueError("DTM band is required (five-band input expected).")

            win_height = int(window.height)
            win_width = int(window.width)
            pad_h = max(0, tile - win_height)
            pad_w = max(0, tile - win_width)

            if pad_h or pad_w:
                rgb = np.pad(
                    rgb,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=np.nan,
                )
                dtm = np.pad(
                    dtm,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=np.nan,
                )
                if dsm is not None:
                    dsm = np.pad(
                        dsm,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=np.nan,
                    )

            valid_mask = np.isfinite(dtm)

            ndsm_tile = compute_ndsm(dsm, dtm)
            svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
                dtm, pixel_size=pixel_size
            )

            stacked = stack_channels(
                rgb=rgb,
                svf=svf,
                pos_open=pos_open,
                neg_open=neg_open,
                lrm=lrm,
                slope=slope,
                ndsm=ndsm_tile,
            )
            if global_norm and fixed_lows is not None and fixed_highs is not None:
                normed = robust_norm_fixed(stacked, fixed_lows, fixed_highs)
            else:
                normed = robust_norm(stacked)

            tensor = torch.from_numpy(normed).unsqueeze(0).to(device)
            with torch.no_grad(), autocast_ctx:
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            row_slice = slice(row, row + win_height)
            col_slice = slice(col, col + win_width)
            tile_rows = row_slice.stop - row_slice.start
            tile_cols = col_slice.stop - col_slice.start
            prob_tile = probs.astype(np.float32)[:tile_rows, :tile_cols]
            valid_tile = valid_mask[:tile_rows, :tile_cols]

            if feather:
                W = make_feather_weights(tile_rows, tile_cols, tile, overlap)
            else:
                W = np.ones((tile_rows, tile_cols), dtype=np.float32)

            prob_acc[row_slice, col_slice] += prob_tile * W * valid_tile
            weight_acc[row_slice, col_slice] += W * valid_tile.astype(np.float32)
            valid_global[row_slice, col_slice] |= valid_tile
            ndsm_chunk = ndsm_tile.astype(np.float32)[:tile_rows, :tile_cols]
            existing = ndsm_max[row_slice, col_slice]
            ndsm_chunk[~valid_tile] = existing[~valid_tile]
            ndsm_max[row_slice, col_slice] = np.maximum(existing, ndsm_chunk)

        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.divide(prob_acc, weight_acc, out=np.zeros_like(prob_acc), where=weight_acc > 0)
        prob_map[~valid_global] = np.nan

        if mask_talls is not None:
            tall_mask = ndsm_max > float(mask_talls)
            prob_map[tall_mask] = 0.0

        prob_map = prob_map.astype(np.float32)
        binary_mask = (prob_map >= threshold).astype(np.uint8)
        binary_mask[~valid_global] = 0

        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        prob_path = base_prefix.parent / f"{base_prefix.name}_prob.tif"
        mask_path = base_prefix.parent / f"{base_prefix.name}_mask.tif"

        meta.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
        with rasterio.open(prob_path, "w", **meta) as dst:
            dst.write(prob_map[np.newaxis, :, :])

        mask_meta = meta.copy()
        mask_meta.update(dtype="uint8", nodata=0)
        with rasterio.open(mask_path, "w", **mask_meta) as dst:
            dst.write(binary_mask[np.newaxis, :, :])

    return InferenceOutputs(
        prob_path=prob_path,
        mask_path=mask_path,
        prob_map=prob_map,
        mask=binary_mask,
        transform=transform,
        crs=crs,
    )


def infer_classic_tiled(
    input_path: Path,
    band_idx: Sequence[int],
    tile: int,
    overlap: int,
    out_prefix: Path,
    classic_th: Optional[float] = None,
    modes: Sequence[str] = ("rvtlog",),
    feather: bool = True,
    save_intermediate: bool = False,
) -> ClassicOutputs:
    """Run classical raster scoring methods in a tiled fashion."""
    if len(band_idx) < 5 or band_idx[4] <= 0:
        raise ValueError("DTM band index must be provided for classic inference.")
    base_modes = []
    for mode in modes:
        key = mode.strip().lower()
        if key not in base_modes:
            base_modes.append(key)
    if not base_modes:
        raise ValueError("At least one classic mode must be specified.")
    allowed = {"rvtlog", "hessian", "morph"}
    unknown = [m for m in base_modes if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported classic mode(s): {', '.join(unknown)}")
    base_prefix = out_prefix.with_suffix("")
    base_prefix.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        prob_acc: Dict[str, np.ndarray] = {
            mode: np.zeros((height, width), dtype=np.float32) for mode in base_modes
        }
        weight_acc: Dict[str, np.ndarray] = {
            mode: np.zeros((height, width), dtype=np.float32) for mode in base_modes
        }
        valid_global = np.zeros((height, width), dtype=bool)
        dtm_idx = band_idx[4]

        total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
            width / max(tile - overlap, 1)
        )

        for window, row, col in progress_bar(
            generate_windows(width, height, tile, overlap),
            total=total_tiles,
            desc="Classic",
            unit="tile",
        ):
            dtm_window = src.read(dtm_idx, window=window, boundless=True, masked=True)
            dtm_tile = np.ma.filled(dtm_window.astype(np.float32), np.nan)

            win_height = int(window.height)
            win_width = int(window.width)
            row_slice = slice(row, row + win_height)
            col_slice = slice(col, col + win_width)
            valid_tile = np.isfinite(dtm_tile)
            valid_global[row_slice, col_slice] |= valid_tile
            if not np.any(valid_tile):
                continue

            if feather:
                weights = make_feather_weights(win_height, win_width, tile, overlap)
            else:
                weights = np.ones((win_height, win_width), dtype=np.float32)

            for mode in base_modes:
                if mode == "rvtlog":
                    score_tile = _score_rvtlog(dtm_tile, pixel_size=pixel_size)
                elif mode == "hessian":
                    score_tile = _score_hessian(dtm_tile)
                elif mode == "morph":
                    score_tile = _score_morph(dtm_tile)
                else:  # pragma: no cover - guarded above
                    continue

                valid_mode = valid_tile & np.isfinite(score_tile)
                if not np.any(valid_mode):
                    continue
                score_tile_masked = np.where(valid_mode, score_tile, np.nan).astype(np.float32)
                prob_tile = _norm01(score_tile_masked)
                prob_tile[~valid_mode] = np.nan
                weight_tile = weights * valid_mode.astype(np.float32)
                prob_fill = np.nan_to_num(prob_tile, nan=0.0)

                prob_acc[mode][row_slice, col_slice] += prob_fill * weight_tile
                weight_acc[mode][row_slice, col_slice] += weight_tile

    per_mode_prob: Dict[str, np.ndarray] = {}
    per_mode_mask: Dict[str, np.ndarray] = {}
    per_mode_thresholds: Dict[str, float] = {}

    for mode in base_modes:
        acc = prob_acc[mode]
        weights = weight_acc[mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.divide(acc, weights, out=np.zeros_like(acc), where=weights > 0)
        prob_map = prob_map.astype(np.float32)
        prob_map[weights <= 0] = np.nan
        prob_map[~valid_global] = np.nan
        valid_mode = np.isfinite(prob_map)
        if np.any(valid_mode):
            prob_norm = _norm01(np.where(valid_mode, prob_map, np.nan))
            prob_norm[~valid_mode] = np.nan
        else:
            prob_norm = np.full_like(prob_map, np.nan, dtype=np.float32)
        if classic_th is not None:
            threshold = float(classic_th)
        else:
            threshold = _otsu_threshold_0to1(prob_norm, valid_mode)
        mask = np.zeros_like(prob_norm, dtype=np.uint8)
        mask[valid_mode & (prob_norm >= threshold)] = 1
        per_mode_prob[mode] = prob_norm.astype(np.float32)
        per_mode_mask[mode] = mask
        per_mode_thresholds[mode] = threshold

    if per_mode_prob:
        stack = np.stack([per_mode_prob[m] for m in base_modes], axis=0)
        combined_prob = np.nanmean(stack, axis=0).astype(np.float32)
    else:
        combined_prob = np.full((height, width), np.nan, dtype=np.float32)
    combined_prob[~valid_global] = np.nan
    combined_valid = np.isfinite(combined_prob)
    combined_prob = np.clip(combined_prob, 0.0, 1.0, out=combined_prob)

    if classic_th is not None:
        combined_threshold = float(classic_th)
    else:
        combined_threshold = _otsu_threshold_0to1(combined_prob, combined_valid)
    combined_mask = np.zeros_like(combined_prob, dtype=np.uint8)
    combined_mask[combined_valid & (combined_prob >= combined_threshold)] = 1

    float_meta = meta.copy()
    float_meta.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    mask_meta = float_meta.copy()
    mask_meta.update(dtype="uint8", nodata=0)

    classic_prob_path = base_prefix.parent / f"{base_prefix.name}_classic_prob.tif"
    classic_mask_path = base_prefix.parent / f"{base_prefix.name}_classic_mask.tif"
    with rasterio.open(classic_prob_path, "w", **float_meta) as dst:
        dst.write(combined_prob[np.newaxis, :, :])
    with rasterio.open(classic_mask_path, "w", **mask_meta) as dst:
        dst.write(combined_mask[np.newaxis, :, :])

    per_mode_outputs: Dict[str, ClassicModeOutput] = {}
    write_individual = save_intermediate or len(base_modes) == 1
    if write_individual:
        for mode in base_modes:
            prob_map = per_mode_prob.get(mode)
            mask_map = per_mode_mask.get(mode)
            if prob_map is None or mask_map is None:
                continue
            mode_prob_path = base_prefix.parent / f"{base_prefix.name}_classic_{mode}_prob.tif"
            mode_mask_path = base_prefix.parent / f"{base_prefix.name}_classic_{mode}_mask.tif"
            with rasterio.open(mode_prob_path, "w", **float_meta) as dst:
                dst.write(prob_map[np.newaxis, :, :])
            with rasterio.open(mode_mask_path, "w", **mask_meta) as dst:
                dst.write(mask_map[np.newaxis, :, :])
            per_mode_outputs[mode] = ClassicModeOutput(
                prob_path=mode_prob_path,
                mask_path=mode_mask_path,
                prob_map=prob_map,
                mask=mask_map,
                threshold=per_mode_thresholds.get(mode, combined_threshold),
            )

    return ClassicOutputs(
        prob_path=classic_prob_path,
        mask_path=classic_mask_path,
        prob_map=combined_prob,
        mask=combined_mask,
        transform=transform,
        crs=crs,
        threshold=combined_threshold,
        per_mode=per_mode_outputs,
    )


def vectorize_predictions(
    mask: np.ndarray,
    prob_map: np.ndarray,
    transform: Affine,
    crs: Optional[RasterioCRS],
    out_path: Path,
    min_area: float,
    simplify_tol: Optional[float],
) -> Optional[Path]:
    """Convert binary mask into polygons and write to GeoPackage."""
    if fiona is None and gpd is None:
        LOGGER.warning("Vector output skipped; install geopandas or fiona for vectorisation.")
        return None

    structure = np.ones((3, 3), dtype=int)
    labels, num_features = ndimage.label(mask.astype(bool), structure=structure)
    if num_features == 0:
        LOGGER.info("No features above threshold; skipping vectorisation.")
        return None

    label_ids = np.arange(1, num_features + 1)
    pixel_counts = ndimage.sum(mask.astype(np.uint8), labels, index=label_ids)
    prob_sums = ndimage.sum(prob_map.astype(np.float32), labels, index=label_ids)

    crs_obj: Optional[CRS] = None
    if crs:
        try:
            crs_obj = CRS.from_wkt(crs.to_wkt())
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("Unable to parse CRS; assuming projected coordinates for area.")

    to_area: Optional[Transformer] = None
    to_native: Optional[Transformer] = None
    if crs_obj and not crs_obj.is_projected:
        area_crs = CRS.from_epsg(6933)
        to_area = Transformer.from_crs(crs_obj, area_crs, always_xy=True)
        to_native = Transformer.from_crs(area_crs, crs_obj, always_xy=True)
    elif not crs_obj:
        LOGGER.warning(
            "Input CRS unknown; polygon areas assume coordinate units are meters."
        )

    records = []
    for geom, value in shapes(labels.astype(np.int32), mask=None, transform=transform):
        label_id = int(value)
        if label_id == 0:
            continue
        geom_shape = shape(geom)
        if to_area:
            geom_area_space = shapely_transform(to_area.transform, geom_shape)
            area_m2 = geom_area_space.area
        else:
            geom_area_space = geom_shape
            area_m2 = geom_shape.area
        if area_m2 < float(min_area):
            continue
        pixels = float(pixel_counts[label_id - 1])
        if pixels <= 0:
            continue
        mean_score = float(prob_sums[label_id - 1]) / pixels
        if simplify_tol and simplify_tol > 0:
            if to_area and to_native:
                simplified_area_geom = geom_area_space.simplify(
                    simplify_tol, preserve_topology=True
                )
                geom_shape = shapely_transform(to_native.transform, simplified_area_geom)
            else:
                geom_shape = geom_shape.simplify(simplify_tol, preserve_topology=True)
        records.append(
            {
                "id": int(label_id),
                "area_m2": float(area_m2),
                "score_mean": float(mean_score),
                "geometry": geom_shape,
            }
        )

    if not records:
        LOGGER.info("No polygons passed the min-area filter; skipping vector output.")
        return None

    out_path = out_path.with_suffix(".gpkg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gpd is not None:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf.to_file(out_path, driver="GPKG")
    else:
        if fiona is None:
            LOGGER.warning("Vector output skipped; fiona not available.")
            return None
        schema = {
            "geometry": "MultiPolygon"
            if any(rec["geometry"].geom_type == "MultiPolygon" for rec in records)
            else "Polygon",
            "properties": {
                "id": "int",
                "area_m2": "float",
                "score_mean": "float",
            },
        }
        crs_wkt = crs.to_wkt() if crs else None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with fiona.open(
            out_path,
            mode="w",
            driver="GPKG",
            schema=schema,
            crs_wkt=crs_wkt,
        ) as dst:
            for rec in records:
                dst.write(
                    {
                        "geometry": mapping(rec["geometry"]),
                        "properties": {
                            "id": rec["id"],
                            "area_m2": rec["area_m2"],
                            "score_mean": rec["score_mean"],
                        },
                    }
                )
    return out_path


def parse_band_indexes(band_string: str) -> Tuple[int, int, int, int, int]:
    """Parse CSV band specification to integer tuple."""
    parts = [int(val.strip()) for val in band_string.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("--bands must specify exactly five entries.")
    if any(idx <= 0 for idx in parts[:3]):
        raise argparse.ArgumentTypeError("RGB band indices must be positive (1-based).")
    if parts[4] <= 0:
        raise argparse.ArgumentTypeError("DTM band index must be provided (>=1).")
    return tuple(parts)  # type: ignore[return-value]


def resolve_out_prefix(input_path: Path, prefix: Optional[str]) -> Path:
    """Resolve output prefix path."""
    if prefix:
        out_path = Path(prefix)
        if out_path.is_dir():
            out_path = out_path / input_path.stem
        return out_path
    return input_path.with_suffix("")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Archaeological feature detection via pretrained U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=default_for("input"),
        help=cli_help("input", "(update PipelineDefaults.input to change the baked-in path)."),
    )
    parser.add_argument(
        "--weights",
        default=default_for("weights"),
        help=cli_help("weights"),
    )
    parser.add_argument(
        "--bands",
        default=default_for("bands"),
        help=cli_help("bands"),
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=default_for("tile"),
        help=cli_help("tile"),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=default_for("overlap"),
        help=cli_help("overlap"),
    )
    parser.add_argument(
        "--th",
        type=float,
        default=default_for("th"),
        help=cli_help("th"),
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=default_for("half"),
        help=cli_help("half", "(use --no-half to force float32 even when CUDA is present)."),
    )
    parser.add_argument(
        "--global-norm",
        action=argparse.BooleanOptionalAction,
        default=default_for("global_norm"),
        help=cli_help("global_norm", "(toggle with --no-global-norm for per-tile scaling)."),
    )
    parser.add_argument(
        "--norm-sample-tiles",
        type=int,
        default=default_for("norm_sample_tiles"),
        help=cli_help("norm_sample_tiles"),
    )
    parser.add_argument(
        "--feather",
        action=argparse.BooleanOptionalAction,
        default=default_for("feather"),
        help=cli_help("feather", "(switch off with --no-feather for raw tile edges)."),
    )
    parser.add_argument(
        "--classic",
        action=argparse.BooleanOptionalAction,
        default=default_for("classic"),
        help=cli_help("classic", "(disable with --no-classic if you only need DL inference)."),
    )
    parser.add_argument(
        "--classic-modes",
        default=default_for("classic_modes"),
        help=cli_help("classic_modes"),
    )
    parser.add_argument(
        "--classic-save-intermediate",
        action=argparse.BooleanOptionalAction,
        default=default_for("classic_save_intermediate"),
        help=cli_help("classic_save_intermediate", "(use --no-classic-save-intermediate to skip per-mode rasters)."),
    )
    parser.add_argument(
        "--classic-th",
        type=float,
        default=default_for("classic_th"),
        help=cli_help("classic_th"),
    )
    parser.add_argument(
        "--fuse",
        action=argparse.BooleanOptionalAction,
        default=default_for("fuse"),
        help=cli_help("fuse", "(turn off with --no-fuse to keep DL and classic outputs separate)."),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_for("alpha"),
        help=cli_help("alpha"),
    )
    parser.add_argument(
        "--mask-talls",
        type=float,
        default=default_for("mask_talls"),
        help=cli_help("mask_talls"),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=default_for("min_area"),
        help=cli_help("min_area"),
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=default_for("simplify"),
        help=cli_help("simplify"),
    )
    parser.add_argument(
        "--vectorize",
        action=argparse.BooleanOptionalAction,
        default=default_for("vectorize"),
        help=cli_help("vectorize", "(set --no-vectorize to keep only raster outputs)."),
    )
    parser.add_argument(
        "--arch",
        default=default_for("arch"),
        help=cli_help("arch"),
    )
    parser.add_argument(
        "--encoder",
        default=default_for("encoder"),
        help=cli_help("encoder"),
    )
    parser.add_argument(
        "--encoders",
        default=default_for("encoders"),
        help=cli_help(
            "encoders",
            "(examples: 'resnet34,resnet50', 'all', 'none').",
        ),
    )
    parser.add_argument(
        "--weights-template",
        default=default_for("weights_template"),
        help=cli_help(
            "weights_template",
            "(uses {encoder} placeholder; ignored if file missing).",
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=default_for("out_prefix"),
        help=cli_help("out_prefix"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_for("seed"),
        help=cli_help("seed"),
    )
    parser.add_argument(
        "--zero-shot-imagenet",
        action=argparse.BooleanOptionalAction,
        default=default_for("zero_shot_imagenet"),
        help=cli_help("zero_shot_imagenet", "(toggle with --no-zero-shot-imagenet when supplying weights)."),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=default_for("verbose"),
        help=cli_help("verbose"),
    )

    args = parser.parse_args(argv)
    config = build_config_from_args(args)

    ENCODER_ALIASES = {
        "efficientnet-b3": "timm-efficientnet-b3",
        "effnet-b3": "timm-efficientnet-b3",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
    }

    def normalize_encoder(name: str) -> Tuple[str, str]:
        key = (name or "").strip().lower()
        smp_name = ENCODER_ALIASES.get(key, key)
        suffix = "efficientnet-b3" if smp_name == "timm-efficientnet-b3" else smp_name
        return smp_name, suffix

    def available_encoders_list() -> List[str]:
        return ["resnet34", "resnet50", "efficientnet-b3"]

    if config.classic_th is not None and not (0.0 <= config.classic_th <= 1.0):
        parser.error("--classic-th must be between 0 and 1.")
    if not (0.0 <= config.alpha <= 1.0):
        parser.error("--alpha must be between 0 and 1.")

    configure_logging(config.verbose)
    install_gdal_warning_filter()
    set_random_seeds(config.seed)

    input_path = Path(config.input)
    if not input_path.exists():
        parser.error(f"Input raster not found: {input_path}")

    enc_mode = (config.encoders or "").strip().lower()
    if enc_mode in ("", "none"):
        if not config.zero_shot_imagenet and not config.weights:
            parser.error("Either provide --weights or use --zero-shot-imagenet for zero-shot inference.")

    weights_path: Optional[Path] = None
    if config.weights:
        weights_path = Path(config.weights)
        if not weights_path.exists():
            parser.error(f"Weights file not found: {weights_path}")

    bands = parse_band_indexes(config.bands)
    out_prefix = resolve_out_prefix(input_path, config.out_prefix)
    classic_outputs: Optional[ClassicOutputs] = None
    fusion_outputs: Optional[FusionOutputs] = None
    resolved_classic_modes: Optional[Tuple[str, ...]] = None
    if config.classic:
        raw_modes = [mode.strip() for mode in config.classic_modes.split(",") if mode.strip()]
        if not raw_modes:
            parser.error("--classic-modes must include at least one mode.")
        valid_modes = {"rvtlog", "hessian", "morph", "combo"}
        invalid = [m for m in raw_modes if m.lower() not in valid_modes]
        if invalid:
            parser.error(f"Unsupported classic mode(s): {', '.join(invalid)}")
        if len(raw_modes) == 1 and raw_modes[0].lower() == "combo":
            resolved_classic_modes = ("rvtlog", "hessian", "morph")
        elif any(m.lower() == "combo" for m in raw_modes):
            parser.error("'combo' cannot be combined with other classic modes.")
        else:
            resolved_classic_modes = tuple(m.lower() for m in raw_modes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if config.half and device.type != "cuda":
        LOGGER.warning("--half requested but CUDA not available; running in float32.")

    if enc_mode not in ("", "none"):
        enc_list = (
            available_encoders_list()
            if enc_mode == "all"
            else [enc.strip() for enc in config.encoders.split(",") if enc.strip()]
        )
        for enc in enc_list:
            smp_name, suffix = normalize_encoder(enc)
            try:
                _ = smp.encoders.get_encoder(smp_name, in_channels=3)
            except Exception:
                avail = ", ".join(available_encoders_list())
                LOGGER.error("Unknown encoder '%s'. Supported: %s", enc, avail)
                continue

            per_weights: Optional[Path] = None
            if config.weights_template:
                cand = Path(config.weights_template.format(encoder=suffix))
                if cand.exists():
                    per_weights = cand
                else:
                    LOGGER.info("[%s] No weights at %s; falling back to zero-shot.", suffix, cand)

            if per_weights is not None:
                LOGGER.info("Running %s with trained weights: %s", suffix, per_weights)
                model = build_model(arch=config.arch, encoder=smp_name, in_ch=9)
                load_weights(model, per_weights, map_location=device)
            else:
                LOGGER.info("Running %s in zero-shot (ImageNet 3->9) mode.", suffix)
                model = build_model_with_imagenet_inflated(arch=config.arch, encoder=smp_name, in_ch=9)

            enc_prefix = out_prefix.with_suffix("")
            enc_prefix = enc_prefix.parent / f"{enc_prefix.name}_{suffix}"

            outputs = infer_tiled(
                model=model,
                input_path=input_path,
                band_idx=bands,
                tile=config.tile,
                overlap=config.overlap,
                device=device,
                use_half=config.half,
                threshold=config.th,
                mask_talls=config.mask_talls,
                out_prefix=enc_prefix,
                global_norm=config.global_norm,
                norm_sample_tiles=config.norm_sample_tiles,
                feather=config.feather,
            )
            LOGGER.info("[%s] Probability: %s", suffix, outputs.prob_path)
            LOGGER.info("[%s] Mask: %s", suffix, outputs.mask_path)

            if config.vectorize:
                vector_base = outputs.mask_path.with_suffix("")
                gpkg = vectorize_predictions(
                    mask=outputs.mask,
                    prob_map=outputs.prob_map,
                    transform=outputs.transform,
                    crs=outputs.crs,
                    out_path=vector_base,
                    min_area=config.min_area,
                    simplify_tol=config.simplify,
                )
                if gpkg:
                    LOGGER.info("[%s] Vector: %s", suffix, gpkg)
        return

    if config.zero_shot_imagenet:
        LOGGER.info("Zero-shot mode: using ImageNet-pretrained encoder inflated to 9 channels.")
        model = build_model_with_imagenet_inflated(
            arch=config.arch, encoder=config.encoder, in_ch=9
        )
    else:
        model = build_model(arch=config.arch, encoder=config.encoder, in_ch=9)
        # weights_path is guaranteed to be set in this branch
        assert weights_path is not None
        load_weights(model, weights_path, map_location=device)

    if bands[3] <= 0:
        LOGGER.warning(
            "DSM band not provided; nDSM channel will be zero and tall-object masking will be disabled."
        )

    outputs = infer_tiled(
        model=model,
        input_path=input_path,
        band_idx=bands,
        tile=config.tile,
        overlap=config.overlap,
        device=device,
        use_half=config.half,
        threshold=config.th,
        mask_talls=config.mask_talls,
        out_prefix=out_prefix,
        global_norm=config.global_norm,
        norm_sample_tiles=config.norm_sample_tiles,
        feather=config.feather,
    )

    LOGGER.info("Probability map written to %s", outputs.prob_path)
    LOGGER.info("Binary mask written to %s", outputs.mask_path)

    fuse_enabled = config.classic and config.fuse
    if config.fuse and not config.classic:
        LOGGER.warning("--fuse requested without --classic; ignoring fusion request.")

    if config.classic and resolved_classic_modes is not None:
        classic_outputs = infer_classic_tiled(
            input_path=input_path,
            band_idx=bands,
            tile=config.tile,
            overlap=config.overlap,
            out_prefix=out_prefix,
            classic_th=config.classic_th,
            modes=resolved_classic_modes,
            feather=config.feather,
            save_intermediate=config.classic_save_intermediate,
        )
        LOGGER.info("Classic outputs written: %s, %s", classic_outputs.prob_path, classic_outputs.mask_path)
        if classic_outputs.per_mode:
            for mode_name, mode_out in classic_outputs.per_mode.items():
                LOGGER.info(
                    "Classic mode '%s' outputs: %s, %s",
                    mode_name,
                    mode_out.prob_path,
                    mode_out.mask_path,
                )

    if fuse_enabled and classic_outputs is not None:
        alpha = float(config.alpha)
        dl_prob = outputs.prob_map.astype(np.float32)
        classic_prob = classic_outputs.prob_map.astype(np.float32)
        dl_filled = np.nan_to_num(dl_prob, nan=0.0)
        classic_filled = np.nan_to_num(classic_prob, nan=0.0)
        fused_prob = alpha * dl_filled + (1.0 - alpha) * classic_filled
        fused_prob = fused_prob.astype(np.float32)
        fused_valid = np.isfinite(dl_prob) | np.isfinite(classic_prob)
        fused_prob = np.clip(fused_prob, 0.0, 1.0, out=fused_prob)
        fused_prob[~fused_valid] = np.nan
        fuse_threshold = config.th if config.th is not None else 0.5
        fused_mask = np.zeros_like(fused_prob, dtype=np.uint8)
        fused_mask[fused_valid & (fused_prob >= fuse_threshold)] = 1

        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        fused_prob_path = base_prefix.parent / f"{base_prefix.name}_fused_prob.tif"
        fused_mask_path = base_prefix.parent / f"{base_prefix.name}_fused_mask.tif"

        common_meta = {
            "driver": "GTiff",
            "height": fused_prob.shape[0],
            "width": fused_prob.shape[1],
            "count": 1,
            "transform": outputs.transform,
            "crs": outputs.crs,
            "compress": "deflate",
        }
        float_meta = common_meta.copy()
        float_meta.update(dtype="float32", nodata=np.nan)
        mask_meta = common_meta.copy()
        mask_meta.update(dtype="uint8", nodata=0)

        with rasterio.open(fused_prob_path, "w", **float_meta) as dst:
            dst.write(fused_prob[np.newaxis, :, :])
        with rasterio.open(fused_mask_path, "w", **mask_meta) as dst:
            dst.write(fused_mask[np.newaxis, :, :])

        fusion_outputs = FusionOutputs(
            prob_path=fused_prob_path,
            mask_path=fused_mask_path,
            prob_map=fused_prob,
            mask=fused_mask,
            transform=outputs.transform,
            crs=outputs.crs,
            threshold=fuse_threshold,
        )
        LOGGER.info("Fused outputs written to %s, %s", fused_prob_path, fused_mask_path)

    if config.vectorize:
        vector_jobs: List[Tuple[str, np.ndarray, np.ndarray, Affine, Optional[RasterioCRS], Path]] = [
            (
                "dl",
                outputs.mask,
                outputs.prob_map,
                outputs.transform,
                outputs.crs,
                outputs.mask_path.with_suffix(""),
            )
        ]
        if classic_outputs is not None:
            vector_jobs.append(
                (
                    "classic",
                    classic_outputs.mask,
                    classic_outputs.prob_map,
                    classic_outputs.transform,
                    classic_outputs.crs,
                    classic_outputs.mask_path.with_suffix(""),
                )
            )
            if config.classic_save_intermediate:
                for mode_name, mode_out in classic_outputs.per_mode.items():
                    vector_jobs.append(
                        (
                            f"classic_{mode_name}",
                            mode_out.mask,
                            mode_out.prob_map,
                            classic_outputs.transform,
                            classic_outputs.crs,
                            mode_out.mask_path.with_suffix(""),
                        )
                    )
        if fusion_outputs is not None:
            vector_jobs.append(
                (
                    "fused",
                    fusion_outputs.mask,
                    fusion_outputs.prob_map,
                    fusion_outputs.transform,
                    fusion_outputs.crs,
                    fusion_outputs.mask_path.with_suffix(""),
                )
            )

        for label, mask_arr, prob_arr, transform, crs_obj, out_base in vector_jobs:
            vector_file = vectorize_predictions(
                mask=mask_arr,
                prob_map=prob_arr,
                transform=transform,
                crs=crs_obj,
                out_path=out_base,
                min_area=config.min_area,
                simplify_tol=config.simplify,
            )
            if vector_file:
                LOGGER.info("Vector output (%s) written to %s", label, vector_file)


if __name__ == "__main__":
    main()

# Quick Start:
# (A) With trained weights:
# python archaeo_detect.py --input /data/site_multiband.tif --weights /models/unet_resnet34_arch.pth \
#     --bands 1,2,3,4,5 --tile 512 --overlap 64 --th 0.5 --mask-talls 2.5 --min-area 50 --vectorize
#
# (B) Zero-shot (no weights; inflate ImageNet encoder to 9-ch):
# python archaeo_detect.py --input /data/site_multiband.tif --bands 1,2,3,4,5 \
#     --tile 512 --overlap 64 --th 0.6 --mask-talls 2.5 --min-area 80 --vectorize --zero-shot-imagenet
