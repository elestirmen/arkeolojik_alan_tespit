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
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import gaussian_filter
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

LOGGER = logging.getLogger("archaeo_detect")

# Turkce not: Bu arac, ayni jeokoordinata sahip RGB, DSM ve DTM bantlarini tek bir GeoTIFF icinde bekler;
# DTM'den SVF/Openness/LRM/slope turetilip RGB ile birlikte 9 kanalli tensore donusturulur.
# Pretrained U-Net modeliyle kaydirma penceresi cikarimi yapilir; olasilik ve ikili maske GeoTIFF olarak yazilir;
# istenirse poligonlastirilir.

# ==== User-editable defaults (edit these as needed) ====
# Bu bloktaki degerleri kendi calismaniza gore degistirebilirsiniz.
USER_DEFAULTS = {
    "input": r"C:\\Users\\ertug\\Nextcloud\\arkeolojik_alan_tespiti\\kesif_alani.tif",
    "bands": "1,2,3,4,5",
    "tile": 1024,
    "overlap": 256,
    "th": 0.6,
    "min_area": 80.0,
    "mask_talls": 2.5,
    "vectorize": True,
    "zero_shot_imagenet": True,
    "verbose": 1,
}
# ======================================================


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
    """Normalize (C,H,W) with fixed per-channel lows/highs; NaNs→0."""
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


def inflate_conv1_to_n(conv_w_3: torch.Tensor, in_ch: int = 9, mode: str = "avg") -> torch.Tensor:
    """
    Inflate a 3-channel conv weight tensor (out,3,kH,kW) to (out,in_ch,kH,kW).
    mode='avg' uses the mean of RGB filters for extra channels.
    """
    out_ch, c3, kH, kW = conv_w_3.shape
    assert c3 == 3, "inflate_conv1_to_n expects 3-channel weights"
    base = conv_w_3.mean(dim=1, keepdim=True) if mode == "avg" else conv_w_3[:, 1:2]  # G channel
    if in_ch <= 3:
        return conv_w_3[:, :in_ch]
    extra = base.repeat(1, in_ch - 3, 1, 1)
    return torch.cat([conv_w_3, extra], dim=1)


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

    conv1_key_src = next((k for k in state_3.keys() if k.endswith("conv1.weight")), None)
    conv1_key_tgt = next((k for k in state_9.keys() if k.endswith("conv1.weight")), None)
    if conv1_key_src is None or conv1_key_tgt is None:
        raise RuntimeError("Could not locate encoder conv1 weights to inflate.")

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
        up = (1 - np.cos(t)) * 0.5  # 0→1
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

        for window, row, col in tqdm(
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
        required=False,
        default=USER_DEFAULTS["input"],
        help=("Path to multi-band GeoTIFF (default from USER_DEFAULTS at top of file)."),
    )
    parser.add_argument("--weights", required=False, help="Path to model weights (.pth).")
    parser.add_argument(
        "--bands",
        default="1,2,3,4,5",
        help="Comma-separated band order for R,G,B,DSM,DTM (use 0 if band missing).",
    )
    parser.add_argument("--tile", type=int, default=USER_DEFAULTS["tile"], help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=USER_DEFAULTS["overlap"], help="Overlap in pixels.")
    parser.add_argument("--th", type=float, default=USER_DEFAULTS["th"], help="Threshold for mask.")
    parser.add_argument("--half", action="store_true", help="Enable CUDA autocast (float16).")
    # Global normalization across tiles
    parser.add_argument(
        "--global-norm",
        action="store_true",
        default=False,
        help="Use a single set of robust p2–p98 thresholds computed from sample tiles to normalize all tiles.",
    )
    parser.add_argument(
        "--norm-sample-tiles",
        type=int,
        default=32,
        help="Number of tiles to sample when estimating global normalization thresholds (center 256x256 crop per tile).",
    )
    # Feather blending on tile edges
    parser.add_argument(
        "--feather",
        dest="feather",
        action="store_true",
        help="Feather tile edges with cosine weights.",
    )
    parser.add_argument(
        "--no-feather",
        dest="feather",
        action="store_false",
        help="Disable feather blending.",
    )
    parser.set_defaults(feather=True)
    parser.add_argument(
        "--mask-talls",
        type=float,
        default=USER_DEFAULTS["mask_talls"],
        help="Zero-out detections where nDSM exceeds this height (meters).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=USER_DEFAULTS["min_area"],
        help="Minimum polygon area in square meters.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=None,
        help="Simplify polygons with this tolerance (meters).",
    )
    parser.add_argument("--vectorize", action="store_true", default=USER_DEFAULTS["vectorize"], help="Export polygons to GPKG.")
    parser.add_argument("--arch", default="Unet", help="segmentation_models_pytorch architecture.")
    parser.add_argument("--encoder", default="resnet34", help="Encoder backbone.")
    parser.add_argument("--out-prefix", default=None, help="Output prefix or directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--zero-shot-imagenet",
        action="store_true",
        default=USER_DEFAULTS["zero_shot_imagenet"],
        help="Run without trained weights by inflating an ImageNet 3-ch encoder to 9-ch (exploratory mode).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=USER_DEFAULTS["verbose"],
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )

    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    set_random_seeds(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input raster not found: {input_path}")

    if not args.zero_shot_imagenet and not args.weights:
        parser.error("Either provide --weights or use --zero-shot-imagenet for zero-shot inference.")

    weights_path: Optional[Path] = None
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            parser.error(f"Weights file not found: {weights_path}")

    bands = parse_band_indexes(args.bands)
    out_prefix = resolve_out_prefix(input_path, args.out_prefix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if args.half and device.type != "cuda":
        LOGGER.warning("--half requested but CUDA not available; running in float32.")

    if args.zero_shot_imagenet:
        LOGGER.info("Zero-shot mode: using ImageNet-pretrained encoder inflated to 9 channels.")
        model = build_model_with_imagenet_inflated(
            arch=args.arch, encoder=args.encoder, in_ch=9
        )
    else:
        model = build_model(arch=args.arch, encoder=args.encoder, in_ch=9)
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
        tile=args.tile,
        overlap=args.overlap,
        device=device,
        use_half=args.half,
        threshold=args.th,
        mask_talls=args.mask_talls,
        out_prefix=out_prefix,
        global_norm=args.global_norm,
        norm_sample_tiles=args.norm_sample_tiles,
        feather=args.feather,
    )

    LOGGER.info("Probability map written to %s", outputs.prob_path)
    LOGGER.info("Binary mask written to %s", outputs.mask_path)

    if args.vectorize:
        vector_base = outputs.mask_path.with_suffix("")
        vector_file = vectorize_predictions(
            mask=outputs.mask,
            prob_map=outputs.prob_map,
            transform=outputs.transform,
            crs=outputs.crs,
            out_path=vector_base,
            min_area=args.min_area,
            simplify_tol=args.simplify,
        )
        if vector_file:
            LOGGER.info("Vector output written to %s", vector_file)


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
