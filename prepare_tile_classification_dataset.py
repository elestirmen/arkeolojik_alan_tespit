#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build an explicit tile-classification dataset from raster + mask pairs.

Output layout:

output_dir/
  train/
    Positive/
    Negative/
  val/
    Positive/
    Negative/
  test/                # optional when --test-ratio > 0
    Positive/
    Negative/

Each saved tile is a 12-channel numpy array (.npz or .npy) compatible with
training.py tile_classification mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from archeo_shared.channels import METADATA_SCHEMA_VERSION, MODEL_CHANNEL_NAMES

try:
    from archaeo_detect import (
        PrecomputedDerivatives,
        build_derivative_raster_cache,
        compute_derivatives_with_rvt,
        compute_ndsm,
        compute_tpi_multiscale,
        full_raster_cache_precompute_ok,
        get_cache_path,
        get_derivative_raster_cache_paths,
        load_derivative_raster_cache_info,
        precompute_derivatives,
        robust_norm,
        stack_channels,
    )
except ImportError:
    print("HATA: archaeo_detect.py bulunamadi.", file=sys.stderr)
    sys.exit(1)


CONFIG = {
    "pairs": [],
    "output_dir": "training_data_classification",
    "tile_size": 256,
    "overlap": 128,
    "bands": "1,2,3,4,5",
    "tpi_radii": "5,15,30",
    "sampling_mode": "full_grid",
    "positive_ratio_threshold": 0.02,
    "valid_ratio_threshold": 0.7,
    "negative_to_positive_ratio": 1.0,
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "test_ratio": 0.0,
    "train_negative_keep_ratio": 0.35,
    "train_negative_max": None,
    "normalize": True,
    "format": "npz",
    "num_workers": max(1, min(8, (os.cpu_count() or 1) // 2)),
    "derivative_cache_mode": "auto",
    "derivative_cache_dir": "",
    "recalculate_derivative_cache": False,
    "tile_prefix": "",
    "seed": 42,
    "overwrite": False,
}

LABEL_NEGATIVE = "Negative"
LABEL_POSITIVE = "Positive"
LABELS = (LABEL_NEGATIVE, LABEL_POSITIVE)
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
SAMPLING_MODE_FULL_GRID = "full_grid"
SAMPLING_MODE_SELECTED_REGIONS = "selected_regions"
SAMPLING_MODES = (
    SAMPLING_MODE_FULL_GRID,
    SAMPLING_MODE_SELECTED_REGIONS,
)

_SAVE_TILE_WORKER_CONTEXT: Dict[str, object] = {}
_SCAN_TILE_WORKER_CONTEXT: Dict[str, object] = {}


@dataclass
class PreparedDerivativeCache:
    mode: str = "none"
    location: Optional[str] = None
    precomputed: Optional[PrecomputedDerivatives] = None
    raster_cache_tif: Optional[Path] = None
    raster_cache_meta: Optional[Path] = None
    raster_band_map: Optional[Dict[str, int]] = None


def _close_scan_tile_worker_context() -> None:
    global _SCAN_TILE_WORKER_CONTEXT
    src = _SCAN_TILE_WORKER_CONTEXT.get("src")
    mask_src = _SCAN_TILE_WORKER_CONTEXT.get("mask_src")
    if src is not None:
        src.close()
    if mask_src is not None:
        mask_src.close()
    _SCAN_TILE_WORKER_CONTEXT = {}


def _init_scan_tile_worker(
    raster_path: str,
    mask_path: str,
    band_idx: Tuple[int, ...],
    tile_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    valid_ratio_threshold: float,
    positive_ratio_threshold: float,
) -> None:
    global _SCAN_TILE_WORKER_CONTEXT
    _close_scan_tile_worker_context()
    mask_src = rasterio.open(mask_path)
    _SCAN_TILE_WORKER_CONTEXT = {
        "src": rasterio.open(raster_path),
        "mask_src": mask_src,
        "mask_negative_value": get_mask_negative_value(mask_src),
        "band_idx": tuple(int(v) for v in band_idx),
        "tile_size": int(tile_size),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "valid_ratio_threshold": float(valid_ratio_threshold),
        "positive_ratio_threshold": float(positive_ratio_threshold),
    }


def _scan_window(
    *,
    src: rasterio.DatasetReader,
    mask_src: rasterio.DatasetReader,
    band_idx: Sequence[int],
    tile_size: int,
    row_off: int,
    col_off: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    valid_ratio_threshold: float,
    positive_ratio_threshold: float,
    mask_negative_value: Optional[float],
) -> dict:
    split_name = split_window_by_rows(
        row_off=int(row_off),
        tile_size=int(tile_size),
        raster_height=int(src.height),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    if split_name is None:
        return {
            "status": "boundary",
            "row_off": int(row_off),
            "col_off": int(col_off),
        }

    window = Window(col_off=int(col_off), row_off=int(row_off), width=int(tile_size), height=int(tile_size))
    _, valid_mask = read_window_data(src, band_idx, window)
    valid_ratio = float(valid_mask.mean()) if valid_mask.size > 0 else 0.0
    if valid_ratio < float(valid_ratio_threshold):
        return {
            "status": "invalid",
            "row_off": int(row_off),
            "col_off": int(col_off),
        }

    mask = mask_src.read(1, window=window).astype(np.float32)
    pos_ratio, positive_pixels, total_pixels = positive_ratio_from_mask(
        mask,
        valid_mask=valid_mask,
        negative_value=mask_negative_value,
    )
    label = LABEL_POSITIVE if pos_ratio >= float(positive_ratio_threshold) else LABEL_NEGATIVE
    return {
        "status": "ok",
        "row_off": int(row_off),
        "col_off": int(col_off),
        "split": str(split_name),
        "label": str(label),
        "positive_ratio": float(pos_ratio),
        "valid_ratio": float(valid_ratio),
        "positive_pixels": int(positive_pixels),
        "total_pixels": int(total_pixels),
    }


def _scan_single_window(task: Tuple[int, int]) -> dict:
    row_off, col_off = task
    ctx = _SCAN_TILE_WORKER_CONTEXT
    return _scan_window(
        src=ctx["src"],
        mask_src=ctx["mask_src"],
        band_idx=ctx["band_idx"],
        tile_size=int(ctx["tile_size"]),
        row_off=int(row_off),
        col_off=int(col_off),
        train_ratio=float(ctx["train_ratio"]),
        val_ratio=float(ctx["val_ratio"]),
        test_ratio=float(ctx["test_ratio"]),
        valid_ratio_threshold=float(ctx["valid_ratio_threshold"]),
        positive_ratio_threshold=float(ctx["positive_ratio_threshold"]),
        mask_negative_value=ctx.get("mask_negative_value"),
    )


def emit_progress(*, phase: str, current: int, total: int, message: str) -> None:
    payload = {
        "phase": str(phase),
        "current": int(current),
        "total": max(1, int(total)),
        "message": str(message),
    }
    print("PROGRESS_JSON\t" + json.dumps(payload, ensure_ascii=False), flush=True)


def sanitize_name(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    token = token.strip("_")
    return token or "source"


def parse_int_csv(raw: str, expected_len: Optional[int] = None) -> Tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if expected_len is not None and len(parts) != expected_len:
        raise ValueError(f"Beklenen {expected_len} tamsayi, verilen: {raw!r}")
    return parts


def companion_gpkg_path(mask_path: Path) -> Path:
    return mask_path.with_suffix(".gpkg")


def get_mask_negative_value(mask_src: rasterio.DatasetReader) -> Optional[float]:
    nodata_value = None
    if getattr(mask_src, "nodatavals", None):
        nodata_value = mask_src.nodatavals[0]
    if nodata_value is None:
        nodata_value = getattr(mask_src, "nodata", None)
    if nodata_value is None:
        return None
    try:
        value = float(nodata_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def mask_selected_pixels(mask: np.ndarray, negative_value: Optional[float] = None) -> np.ndarray:
    selected = np.isfinite(mask)
    if negative_value is None:
        selected &= mask > 0
    else:
        selected &= ~np.isclose(mask, float(negative_value), atol=1e-6, rtol=0.0)
    return selected


def positive_ratio_from_mask(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    negative_value: Optional[float] = None,
) -> tuple[float, int, int]:
    if mask.size <= 0:
        return 0.0, 0, 0
    if valid_mask is None:
        valid = np.isfinite(mask)
    else:
        if valid_mask.shape != mask.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} ile mask shape {mask.shape} uyusmuyor."
            )
        valid = valid_mask & np.isfinite(mask)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels <= 0:
        return 0.0, 0, 0
    positive_pixels = int(np.count_nonzero(valid & mask_selected_pixels(mask, negative_value=negative_value)))
    ratio = float(positive_pixels / valid_pixels)
    return ratio, positive_pixels, valid_pixels


def _declared_nodata_mask(
    src: rasterio.DatasetReader,
    band_idx: Sequence[int],
    data: np.ndarray,
) -> np.ndarray:
    nodata_mask = np.zeros(data.shape[1:], dtype=bool)
    for arr_idx, band_i in enumerate(band_idx):
        band_pos = int(band_i) - 1
        if band_pos < 0 or band_pos >= len(src.nodatavals):
            continue
        nodata_value = src.nodatavals[band_pos]
        if nodata_value is None:
            continue
        nodata_mask |= np.isclose(data[arr_idx], float(nodata_value), atol=1e-6, rtol=0.0)
    return nodata_mask


def _implicit_invalid_mask(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Beklenen (C,H,W) veri, alinan shape={data.shape}")
    tol = 1e-6
    all_zero = np.all(np.isclose(data, 0.0, atol=tol, rtol=0.0), axis=0)
    if data.shape[0] < 4:
        return all_zero

    rgb_zero = np.all(np.isclose(data[:3], 0.0, atol=tol, rtol=0.0), axis=0)
    aux = data[3:]
    aux_zero = np.all(np.isclose(aux, 0.0, atol=tol, rtol=0.0), axis=0)
    aux_constant_extreme = (
        np.all(np.isclose(aux, aux[:1], atol=tol, rtol=0.0), axis=0)
        & (np.abs(aux[0]) >= 9999.0)
    )
    return all_zero | (rgb_zero & (aux_zero | aux_constant_extreme))


def read_window_data(
    src: rasterio.DatasetReader,
    band_idx: Sequence[int],
    window: Window,
) -> tuple[np.ndarray, np.ndarray]:
    data = src.read(indexes=list(band_idx), window=window).astype(np.float32)
    masks = src.read_masks(indexes=list(band_idx), window=window)
    if masks.ndim == 2:
        valid = masks > 0
    else:
        valid = np.all(masks > 0, axis=0)
    valid &= np.all(np.isfinite(data), axis=0)
    valid &= ~_declared_nodata_mask(src, band_idx, data)
    valid &= ~_implicit_invalid_mask(data)
    if np.any(~valid):
        data[:, ~valid] = np.nan
    return data, valid


def valid_ratio_from_window(
    src: rasterio.DatasetReader,
    band_idx: Sequence[int],
    window: Window,
) -> float:
    _, valid = read_window_data(src, band_idx, window)
    return float(valid.mean()) if valid.size > 0 else 0.0


def build_windows(width: int, height: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    if width < tile_size or height < tile_size:
        return []
    rows = list(range(0, height - tile_size + 1, stride))
    cols = list(range(0, width - tile_size + 1, stride))
    return [(row_off, col_off) for row_off in rows for col_off in cols]


def build_split_window_index(
    *,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[List[Tuple[int, int]], Dict[str, List[Tuple[int, int]]], int]:
    windows = build_windows(width, height, tile_size, stride)
    windows_by_split: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    boundary_discarded = 0
    for row_off, col_off in windows:
        split_name = split_window_by_rows(
            row_off=int(row_off),
            tile_size=int(tile_size),
            raster_height=int(height),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
        )
        if split_name is None:
            boundary_discarded += 1
            continue
        windows_by_split[str(split_name)].append((int(row_off), int(col_off)))
    return windows, windows_by_split, boundary_discarded


def _candidate_offsets_for_extent(
    *,
    min_index: int,
    max_index: int,
    tile_size: int,
    stride: int,
    limit: int,
) -> List[int]:
    if limit < tile_size or max_index < 0 or min_index >= limit:
        return []
    start = max(0, int(min_index) - int(tile_size) + 1)
    end = min(int(max_index), int(limit) - int(tile_size))
    if end < start:
        return []
    first = ((start + int(stride) - 1) // int(stride)) * int(stride)
    if first > end:
        return []
    return list(range(first, end + 1, int(stride)))


def _add_candidate_windows_from_bounds(
    candidate_windows: set[Tuple[int, int]],
    *,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    height: int,
    width: int,
    tile_size: int,
    stride: int,
) -> None:
    row_offsets = _candidate_offsets_for_extent(
        min_index=int(row_min),
        max_index=int(row_max),
        tile_size=int(tile_size),
        stride=int(stride),
        limit=int(height),
    )
    col_offsets = _candidate_offsets_for_extent(
        min_index=int(col_min),
        max_index=int(col_max),
        tile_size=int(tile_size),
        stride=int(stride),
        limit=int(width),
    )
    for row_off in row_offsets:
        for col_off in col_offsets:
            candidate_windows.add((int(row_off), int(col_off)))


def _collect_positive_candidate_windows_from_gpkg(
    *,
    pair: SourcePair,
    src: rasterio.DatasetReader,
    tile_size: int,
    stride: int,
) -> Optional[set[Tuple[int, int]]]:
    gpkg_path = companion_gpkg_path(pair.mask_path)
    if not gpkg_path.exists():
        return None
    try:
        import geopandas as gpd

        try:
            gdf = gpd.read_file(gpkg_path, layer="annotations")
        except Exception:
            gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            return set()
        if src.crs is not None and gdf.crs is not None and str(gdf.crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)

        candidate_windows: set[Tuple[int, int]] = set()
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            bbox = from_bounds(*geom.bounds, transform=src.transform)
            row_min = int(math.floor(float(bbox.row_off)))
            row_max = int(math.ceil(float(bbox.row_off + bbox.height)) - 1)
            col_min = int(math.floor(float(bbox.col_off)))
            col_max = int(math.ceil(float(bbox.col_off + bbox.width)) - 1)
            _add_candidate_windows_from_bounds(
                candidate_windows,
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                height=int(src.height),
                width=int(src.width),
                tile_size=int(tile_size),
                stride=int(stride),
            )
        return candidate_windows
    except Exception as exc:
        print(
            f"BILGI: Pozitif aday pencereler GPKG'den okunamadi ({pair.name}): {exc}. Mask blok taramaya geciliyor.",
            flush=True,
        )
        return None


def _collect_positive_candidate_windows_from_mask(
    *,
    mask_src: rasterio.DatasetReader,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> set[Tuple[int, int]]:
    candidate_windows: set[Tuple[int, int]] = set()
    negative_value = get_mask_negative_value(mask_src)
    for _, block_window in mask_src.block_windows(1):
        mask_block = mask_src.read(1, window=block_window).astype(np.float32)
        if not np.any(mask_selected_pixels(mask_block, negative_value=negative_value)):
            continue
        row_min = int(block_window.row_off)
        row_max = int(block_window.row_off + block_window.height - 1)
        col_min = int(block_window.col_off)
        col_max = int(block_window.col_off + block_window.width - 1)
        _add_candidate_windows_from_bounds(
            candidate_windows,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
            height=int(height),
            width=int(width),
            tile_size=int(tile_size),
            stride=int(stride),
        )
    return candidate_windows


def collect_positive_candidate_windows(
    *,
    pair: SourcePair,
    src: rasterio.DatasetReader,
    mask_src: rasterio.DatasetReader,
    tile_size: int,
    stride: int,
) -> tuple[List[Tuple[int, int]], str]:
    from_gpkg = _collect_positive_candidate_windows_from_gpkg(
        pair=pair,
        src=src,
        tile_size=tile_size,
        stride=stride,
    )
    if from_gpkg is not None:
        return sorted(from_gpkg), "gpkg"
    return sorted(
        _collect_positive_candidate_windows_from_mask(
            mask_src=mask_src,
            width=int(src.width),
            height=int(src.height),
            tile_size=int(tile_size),
            stride=int(stride),
        )
    ), "mask_blocks"


def tile_record_from_scan_result(source_name: str, result: dict) -> TileRecord:
    return TileRecord(
        source_name=str(source_name),
        split=str(result["split"]),
        label=str(result["label"]),
        row_off=int(result["row_off"]),
        col_off=int(result["col_off"]),
        positive_ratio=float(result["positive_ratio"]),
        valid_ratio=float(result["valid_ratio"]),
        positive_pixels=int(result["positive_pixels"]),
        total_pixels=int(result["total_pixels"]),
    )


def validate_source_raster(
    src: rasterio.DatasetReader,
    band_idx: Sequence[int],
    raster_path: Path,
) -> None:
    if src.count < max(band_idx):
        raise ValueError(
            f"Raster {raster_path} icinde {src.count} bant var ama {band_idx} istendi."
        )

    dsm_dtype = np.dtype(src.dtypes[int(band_idx[3]) - 1])
    dtm_dtype = np.dtype(src.dtypes[int(band_idx[4]) - 1])
    if (
        np.issubdtype(dsm_dtype, np.integer)
        and np.issubdtype(dtm_dtype, np.integer)
        and dsm_dtype.itemsize <= 1
        and dtm_dtype.itemsize <= 1
    ):
        raise ValueError(
            "DSM/DTM bantlari 8-bit gorunuyor "
            f"({dsm_dtype}/{dtm_dtype}) -> {raster_path}. "
            "Bu durumda rakim verisi 0-255'e ezilmis olabilir; float32 DSM/DTM iceren bir 5-band GeoTIFF kullanin."
        )


def split_window_by_rows(
    *,
    row_off: int,
    tile_size: int,
    raster_height: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Optional[str]:
    train_end = int(math.floor(raster_height * train_ratio))
    val_end = int(math.floor(raster_height * (train_ratio + val_ratio)))
    tile_end = row_off + tile_size

    if tile_end <= train_end:
        return SPLIT_TRAIN
    if val_ratio > 0.0 and row_off >= train_end and tile_end <= val_end:
        return SPLIT_VAL
    if test_ratio > 0.0:
        test_start = val_end
        if row_off >= test_start and tile_end <= raster_height:
            return SPLIT_TEST
    if test_ratio <= 0.0 and row_off >= val_end and tile_end <= raster_height:
        return SPLIT_VAL
    return None


@dataclass(frozen=True)
class SourcePair:
    name: str
    raster_path: Path
    mask_path: Path


@dataclass
class TileRecord:
    source_name: str
    split: str
    label: str
    row_off: int
    col_off: int
    positive_ratio: float
    valid_ratio: float
    positive_pixels: int
    total_pixels: int
    output_relpath: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raster + mask ciftlerinden explicit Positive/Negative tile dataset uretir."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("RASTER_PATH", "MASK_PATH"),
        help="Raster + mask cifti. Birden fazla kaynak icin argumani tekrarlayin.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(CONFIG["output_dir"]))
    parser.add_argument("--tile-size", type=int, default=int(CONFIG["tile_size"]))
    parser.add_argument("--overlap", type=int, default=int(CONFIG["overlap"]))
    parser.add_argument("--bands", type=str, default=str(CONFIG["bands"]))
    parser.add_argument("--tpi-radii", type=str, default=str(CONFIG["tpi_radii"]))
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default=str(CONFIG["sampling_mode"]),
    )
    parser.add_argument(
        "--positive-ratio-threshold",
        type=float,
        default=float(CONFIG["positive_ratio_threshold"]),
    )
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=float(CONFIG["negative_to_positive_ratio"]),
    )
    parser.add_argument(
        "--valid-ratio-threshold",
        type=float,
        default=float(CONFIG["valid_ratio_threshold"]),
    )
    parser.add_argument("--train-ratio", type=float, default=float(CONFIG["train_ratio"]))
    parser.add_argument("--val-ratio", type=float, default=float(CONFIG["val_ratio"]))
    parser.add_argument("--test-ratio", type=float, default=float(CONFIG["test_ratio"]))
    parser.add_argument(
        "--train-negative-keep-ratio",
        type=float,
        default=float(CONFIG["train_negative_keep_ratio"]),
    )
    parser.add_argument("--train-negative-max", type=int, default=CONFIG["train_negative_max"])
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["normalize"]),
    )
    parser.add_argument("--format", choices=("npz", "npy"), default=str(CONFIG["format"]))
    parser.add_argument("--num-workers", type=int, default=int(CONFIG["num_workers"]))
    parser.add_argument(
        "--derivative-cache-mode",
        choices=("none", "auto", "npz", "raster"),
        default=str(CONFIG["derivative_cache_mode"]),
    )
    parser.add_argument(
        "--derivative-cache-dir",
        type=str,
        default=str(CONFIG["derivative_cache_dir"]),
        help="Bos birakilirsa cache, her girdi rasterinin yanindaki 'cache/' klasorune yazilir.",
    )
    parser.add_argument(
        "--recalculate-derivative-cache",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["recalculate_derivative_cache"]),
        help="Mevcut derivative cache varsa yeniden olustur.",
    )
    parser.add_argument("--tile-prefix", type=str, default=str(CONFIG["tile_prefix"]))
    parser.add_argument("--seed", type=int, default=int(CONFIG["seed"]))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=bool(CONFIG["overwrite"]),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors: List[str] = []
    if args.tile_size <= 0:
        errors.append(f"tile_size > 0 olmali, verilen: {args.tile_size}")
    if args.overlap < 0:
        errors.append(f"overlap >= 0 olmali, verilen: {args.overlap}")
    if args.overlap >= args.tile_size:
        errors.append(
            f"overlap tile_size'dan kucuk olmali ({args.overlap} >= {args.tile_size})"
        )
    try:
        band_idx = parse_int_csv(args.bands, expected_len=5)
        if any(b <= 0 for b in band_idx):
            errors.append(f"bands 1-bazli pozitif olmali, verilen: {args.bands}")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        tpi_radii = parse_int_csv(args.tpi_radii)
        if not tpi_radii:
            errors.append("tpi_radii bos olamaz.")
    except ValueError as exc:
        errors.append(str(exc))
    for name in (
        "positive_ratio_threshold",
        "valid_ratio_threshold",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "train_negative_keep_ratio",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} 0-1 araliginda olmali, verilen: {value}")
    if float(args.negative_to_positive_ratio) < 0.0:
        errors.append(
            "negative_to_positive_ratio >= 0 olmali, "
            f"verilen: {args.negative_to_positive_ratio}"
        )
    if float(args.train_ratio) <= 0.0:
        errors.append("train_ratio > 0 olmali.")
    if float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio) > 1.0 + 1e-9:
        errors.append("train_ratio + val_ratio + test_ratio 1.0'i gecemez.")
    if args.train_negative_max is not None and int(args.train_negative_max) < 0:
        errors.append(
            f"train_negative_max None veya >= 0 olmali, verilen: {args.train_negative_max}"
        )
    if int(args.seed) < 0:
        errors.append(f"seed >= 0 olmali, verilen: {args.seed}")
    if int(args.num_workers) < 1:
        errors.append(f"num_workers >= 1 olmali, verilen: {args.num_workers}")
    if errors:
        raise ValueError("Arguman dogrulama hatalari:\n- " + "\n- ".join(errors))


def derive_source_pairs(args: argparse.Namespace) -> List[SourcePair]:
    raw_pairs: List[Tuple[str, str, Optional[str]]] = []
    if args.pair:
        for raster_path, mask_path in args.pair:
            raw_pairs.append((raster_path, mask_path, None))
    else:
        for index, item in enumerate(CONFIG["pairs"]):
            if not isinstance(item, dict):
                raise ValueError(f"CONFIG['pairs'][{index}] dict olmali.")
            raw_pairs.append(
                (
                    str(item.get("raster_path", "")),
                    str(item.get("mask_path", "")),
                    item.get("name"),
                )
            )
    if not raw_pairs:
        raise ValueError("En az bir raster/mask cifti gereklidir.")
    pairs: List[SourcePair] = []
    used_names: set[str] = set()
    for index, (raster_raw, mask_raw, explicit_name) in enumerate(raw_pairs, start=1):
        raster_path = Path(raster_raw).expanduser().resolve()
        mask_path = Path(mask_raw).expanduser().resolve()
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster bulunamadi: {raster_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask bulunamadi: {mask_path}")
        base_name = explicit_name or raster_path.stem
        safe_name = sanitize_name(base_name)
        if safe_name in used_names:
            safe_name = f"{safe_name}_{index:02d}"
        used_names.add(safe_name)
        pairs.append(SourcePair(name=safe_name, raster_path=raster_path, mask_path=mask_path))
    return pairs


def prepare_output_dir(output_dir: Path, overwrite: bool, include_test: bool) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        existing_entries = list(output_dir.iterdir())
        if existing_entries and not overwrite:
            raise FileExistsError(
                f"Cikti dizini bos degil: {output_dir}\nYeni bir dizin secin veya --overwrite kullanin."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    split_names = [SPLIT_TRAIN, SPLIT_VAL] + ([SPLIT_TEST] if include_test else [])
    for split_name in split_names:
        for label in LABELS:
            (output_dir / split_name / label).mkdir(parents=True, exist_ok=True)


def _shape_suffix_from_array(array: np.ndarray) -> str:
    if array.ndim < 2:
        raise ValueError(f"Tile array shape en az 2 boyutlu olmali, alinan: {array.shape}")
    height = int(array.shape[-2])
    width = int(array.shape[-1])
    return f"{height}x{width}"


def make_tile_name(record: TileRecord, tile_prefix: str, *, shape_suffix: Optional[str] = None) -> str:
    base = f"{record.source_name}_r{record.row_off:05d}_c{record.col_off:05d}"
    if shape_suffix:
        base = f"{base}_{shape_suffix}"
    return f"{tile_prefix}_{base}" if tile_prefix else base


def select_train_negative_records(
    records: List[TileRecord],
    *,
    keep_ratio: float,
    max_count: Optional[int],
    seed: int,
) -> Tuple[List[TileRecord], Dict[str, int]]:
    train_positive = [r for r in records if r.split == SPLIT_TRAIN and r.label == LABEL_POSITIVE]
    train_negative = [r for r in records if r.split == SPLIT_TRAIN and r.label == LABEL_NEGATIVE]
    others = [r for r in records if r.split != SPLIT_TRAIN]
    target_count = len(train_negative)
    if keep_ratio < 1.0:
        target_count = int(round(target_count * keep_ratio))
    if max_count is not None:
        target_count = min(target_count, int(max_count))
    target_count = max(0, min(target_count, len(train_negative)))
    if target_count >= len(train_negative):
        selected_negative = list(train_negative)
    elif target_count == 0:
        selected_negative = []
    else:
        rng = random.Random(seed)
        selected_negative = rng.sample(train_negative, target_count)
        selected_negative.sort(key=lambda item: (item.source_name, item.row_off, item.col_off))
    selected_records = sorted(
        train_positive + selected_negative + others,
        key=lambda item: (SPLIT_ORDER.index(item.split), item.source_name, item.row_off, item.col_off),
    )
    stats = {
        "train_positive_count": len(train_positive),
        "train_negative_before_filter": len(train_negative),
        "train_negative_after_filter": len(selected_negative),
        "train_negative_removed": len(train_negative) - len(selected_negative),
    }
    return selected_records, stats


def select_records_for_selected_regions_mode(
    records: List[TileRecord],
    *,
    negative_to_positive_ratio: float,
    seed: int,
) -> Tuple[List[TileRecord], Dict[str, object]]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, str], List[TileRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.split, record.label)].append(record)
    selected_records: List[TileRecord] = []
    stats: Dict[str, object] = {
        "mode": SAMPLING_MODE_SELECTED_REGIONS,
        "negative_to_positive_ratio": float(negative_to_positive_ratio),
        "per_split": {},
    }
    for split_name in SPLIT_ORDER:
        positives = list(grouped.get((split_name, LABEL_POSITIVE), []))
        negatives_all = list(grouped.get((split_name, LABEL_NEGATIVE), []))
        negatives_zero_only = [
            record for record in negatives_all if float(record.positive_ratio) <= 0.0
        ]
        negative_pool = negatives_zero_only or negatives_all
        target_negative_count = int(round(len(positives) * float(negative_to_positive_ratio)))
        target_negative_count = max(0, min(target_negative_count, len(negative_pool)))
        if target_negative_count >= len(negative_pool):
            selected_negatives = list(negative_pool)
        elif target_negative_count == 0:
            selected_negatives = []
        else:
            selected_negatives = rng.sample(negative_pool, target_negative_count)
            selected_negatives.sort(key=lambda item: (item.source_name, item.row_off, item.col_off))
        selected_records.extend(positives)
        selected_records.extend(selected_negatives)
        stats["per_split"][split_name] = {
            "positive_count": len(positives),
            "negative_candidates_total": len(negatives_all),
            "negative_candidates_zero_only": len(negatives_zero_only),
            "negative_pool_used": len(negative_pool),
            "negative_selected": len(selected_negatives),
        }
    selected_records = sorted(
        selected_records,
        key=lambda item: (SPLIT_ORDER.index(item.split), item.source_name, item.row_off, item.col_off),
    )
    stats["total_selected"] = len(selected_records)
    return selected_records, stats


def iterate_scan_results(
    *,
    pair: SourcePair,
    windows: Sequence[Tuple[int, int]],
    band_idx: Sequence[int],
    args: argparse.Namespace,
) -> Iterable[dict]:
    window_list = [(int(row_off), int(col_off)) for row_off, col_off in windows]
    if not window_list:
        return
    num_workers = max(1, int(args.num_workers))
    worker_init_args = (
        str(pair.raster_path),
        str(pair.mask_path),
        tuple(int(v) for v in band_idx),
        int(args.tile_size),
        float(args.train_ratio),
        float(args.val_ratio),
        float(args.test_ratio),
        float(args.valid_ratio_threshold),
        float(args.positive_ratio_threshold),
    )
    if num_workers == 1:
        _init_scan_tile_worker(*worker_init_args)
        try:
            for result in map(_scan_single_window, window_list):
                yield result
        finally:
            _close_scan_tile_worker_context()
        return

    chunksize = max(1, min(256, len(window_list) // max(1, num_workers * 8)))
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_scan_tile_worker,
        initargs=worker_init_args,
    ) as executor:
        for result in executor.map(_scan_single_window, window_list, chunksize=chunksize):
            yield result


def collect_random_negative_records(
    *,
    pair: SourcePair,
    candidate_windows: Sequence[Tuple[int, int]],
    split_name: str,
    target_count: int,
    band_idx: Sequence[int],
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[List[TileRecord], int]:
    if target_count <= 0 or not candidate_windows:
        return [], 0

    remaining_windows = [(int(row_off), int(col_off)) for row_off, col_off in candidate_windows]
    rng.shuffle(remaining_windows)
    scanned_total = 0
    selected_records: List[TileRecord] = []
    batch_floor = max(64, int(args.num_workers) * 32)
    cursor = 0

    while cursor < len(remaining_windows) and len(selected_records) < int(target_count):
        remaining_target = int(target_count) - len(selected_records)
        batch_size = min(
            len(remaining_windows) - cursor,
            max(batch_floor, remaining_target * 6),
        )
        batch = remaining_windows[cursor:cursor + batch_size]
        cursor += batch_size
        batch_selected_before = len(selected_records)
        for result in iterate_scan_results(
            pair=pair,
            windows=batch,
            band_idx=band_idx,
            args=args,
        ):
            scanned_total += 1
            if str(result.get("status", "")) != "ok":
                continue
            if str(result["label"]) != LABEL_NEGATIVE or int(result["positive_pixels"]) != 0:
                continue
            if len(selected_records) < int(target_count):
                selected_records.append(tile_record_from_scan_result(pair.name, result))
        batch_selected = len(selected_records) - batch_selected_before
        emit_progress(
            phase="select",
            current=len(selected_records),
            total=max(1, int(target_count)),
            message=(
                f"Rastgele negatif seciliyor: "
                f"{len(selected_records)}/{int(target_count)} "
                f"({pair.name}/{split_name}, denenen {scanned_total}, batch +{batch_selected})"
            ),
        )

    return selected_records, scanned_total


def collect_selected_region_records(
    pairs: Sequence[SourcePair],
    args: argparse.Namespace,
) -> Tuple[List[TileRecord], Dict[str, object], Dict[str, object]]:
    band_idx = parse_int_csv(args.bands, expected_len=5)
    tile_size = int(args.tile_size)
    stride = int(args.tile_size - args.overlap)
    negative_to_positive_ratio = float(args.negative_to_positive_ratio)
    selection_contexts: List[Dict[str, object]] = []
    total_positive_candidates = 0

    for pair in pairs:
        with rasterio.open(pair.raster_path) as src, rasterio.open(pair.mask_path) as mask_src:
            if src.width != mask_src.width or src.height != mask_src.height:
                raise ValueError(
                    f"Raster/mask boyutu uyusmuyor: {pair.name} -> "
                    f"{src.width}x{src.height} vs {mask_src.width}x{mask_src.height}"
                )
            validate_source_raster(src, band_idx, pair.raster_path)
            all_windows, windows_by_split, boundary_discarded = build_split_window_index(
                width=int(src.width),
                height=int(src.height),
                tile_size=int(tile_size),
                stride=int(stride),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
            )
            if not all_windows:
                raise ValueError(
                    f"Raster icine tam tile sigmiyor: {pair.raster_path} (tile_size={tile_size})"
                )
            positive_candidate_windows, candidate_source = collect_positive_candidate_windows(
                pair=pair,
                src=src,
                mask_src=mask_src,
                tile_size=int(tile_size),
                stride=int(stride),
            )
            total_positive_candidates += len(positive_candidate_windows)
            selection_contexts.append(
                {
                    "pair": pair,
                    "width": int(src.width),
                    "height": int(src.height),
                    "windows_total": len(all_windows),
                    "windows_by_split": windows_by_split,
                    "boundary_discarded": int(boundary_discarded),
                    "positive_candidate_windows": positive_candidate_windows,
                    "positive_candidate_source": candidate_source,
                }
            )

    processed_positive_candidates = 0
    total_positive_candidates = max(1, int(total_positive_candidates))
    selected_records: List[TileRecord] = []
    source_summaries: List[Dict[str, object]] = []
    sampling_stats: Dict[str, object] = {
        "mode": SAMPLING_MODE_SELECTED_REGIONS,
        "strategy": "positive_from_annotations_random_unlabeled_negatives",
        "negative_to_positive_ratio": float(negative_to_positive_ratio),
        "per_split": {
            split_name: {
                "positive_count": 0,
                "negative_target": 0,
                "negative_pool_total": 0,
                "negative_scanned": 0,
                "negative_selected": 0,
                "negative_shortfall": 0,
            }
            for split_name in SPLIT_ORDER
        },
    }

    for source_index, context in enumerate(selection_contexts):
        pair = context["pair"]
        positive_candidate_windows = list(context["positive_candidate_windows"])
        windows_by_split = {
            str(split_name): list(items)
            for split_name, items in dict(context["windows_by_split"]).items()
        }
        source_counter: Counter[str] = Counter()
        candidate_rejected = 0
        invalid_discarded = 0
        boundary_discarded = int(context["boundary_discarded"])
        negative_scanned_total = 0

        with rasterio.open(pair.raster_path) as src, rasterio.open(pair.mask_path) as mask_src:
            positive_records: List[TileRecord] = []
            for result in iterate_scan_results(
                pair=pair,
                windows=positive_candidate_windows,
                band_idx=band_idx,
                args=args,
            ):
                processed_positive_candidates += 1
                if (
                    processed_positive_candidates == 1
                    or processed_positive_candidates == total_positive_candidates
                    or processed_positive_candidates % 25 == 0
                ):
                    emit_progress(
                        phase="scan",
                        current=processed_positive_candidates,
                        total=total_positive_candidates,
                        message=(
                            f"Etiketli pencere taraniyor: "
                            f"{processed_positive_candidates}/{total_positive_candidates} ({pair.name})"
                        ),
                    )
                status = str(result.get("status", ""))
                if status == "boundary":
                    boundary_discarded += 1
                    continue
                if status == "invalid":
                    invalid_discarded += 1
                    continue
                if status != "ok":
                    raise ValueError(f"Bilinmeyen scan sonucu: {status}")
                if str(result["label"]) != LABEL_POSITIVE:
                    candidate_rejected += 1
                    continue
                record = tile_record_from_scan_result(pair.name, result)
                positive_records.append(record)
                source_counter[f"{record.split}_{record.label}"] += 1

            positive_window_set = {(int(item.row_off), int(item.col_off)) for item in positive_records}
            rng = random.Random(int(args.seed) + source_index * 1009)
            negative_records: List[TileRecord] = []
            for split_name in SPLIT_ORDER:
                split_positive_count = sum(1 for item in positive_records if item.split == split_name)
                target_negative_count = int(round(split_positive_count * float(negative_to_positive_ratio)))
                candidate_pool = [
                    (int(row_off), int(col_off))
                    for row_off, col_off in windows_by_split.get(split_name, [])
                    if (int(row_off), int(col_off)) not in positive_window_set
                ]
                split_negative_records, scanned_for_split = collect_random_negative_records(
                    pair=pair,
                    candidate_windows=candidate_pool,
                    split_name=split_name,
                    target_count=target_negative_count,
                    band_idx=band_idx,
                    args=args,
                    rng=rng,
                )
                for record in split_negative_records:
                    negative_records.append(record)
                    source_counter[f"{record.split}_{record.label}"] += 1
                negative_scanned_total += scanned_for_split
                split_stats = sampling_stats["per_split"][split_name]
                split_stats["positive_count"] += int(split_positive_count)
                split_stats["negative_target"] += int(target_negative_count)
                split_stats["negative_pool_total"] += int(len(candidate_pool))
                split_stats["negative_scanned"] += int(scanned_for_split)
                split_stats["negative_selected"] += int(len(split_negative_records))
                split_stats["negative_shortfall"] += int(max(0, target_negative_count - len(split_negative_records)))

        selected_records.extend(positive_records)
        selected_records.extend(negative_records)
        source_summaries.append(
            {
                "name": pair.name,
                "raster_path": str(pair.raster_path),
                "mask_path": str(pair.mask_path),
                "width": int(context["width"]),
                "height": int(context["height"]),
                "windows_total": int(context["windows_total"]),
                "boundary_discarded": int(boundary_discarded),
                "invalid_discarded": int(invalid_discarded),
                "positive_candidate_source": str(context["positive_candidate_source"]),
                "positive_candidate_windows": len(positive_candidate_windows),
                "positive_candidate_rejected": int(candidate_rejected),
                "negative_random_scanned": int(negative_scanned_total),
                "counts": dict(source_counter),
            }
        )
        emit_progress(
            phase="scan",
            current=min(processed_positive_candidates, total_positive_candidates),
            total=total_positive_candidates,
            message=f"Secili bolge taramasi tamamlandi: {pair.name}",
        )

    selected_records = sorted(
        selected_records,
        key=lambda item: (SPLIT_ORDER.index(item.split), item.source_name, item.row_off, item.col_off),
    )
    sampling_stats["total_selected"] = len(selected_records)
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "task_type": "tile_classification",
        "dataset_layout": "classification_folders",
        "num_channels": len(MODEL_CHANNEL_NAMES),
        "channel_names": list(MODEL_CHANNEL_NAMES),
        "bands": str(args.bands),
        "tile_size": int(args.tile_size),
        "overlap": int(args.overlap),
        "stride": stride,
        "tpi_radii": list(parse_int_csv(args.tpi_radii)),
        "positive_ratio_threshold": float(args.positive_ratio_threshold),
        "valid_ratio_threshold": float(args.valid_ratio_threshold),
        "negative_to_positive_ratio": float(args.negative_to_positive_ratio),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "train_negative_keep_ratio": float(args.train_negative_keep_ratio),
        "train_negative_max": None if args.train_negative_max is None else int(args.train_negative_max),
        "normalize": bool(args.normalize),
        "save_format": str(args.format),
        "num_workers": int(args.num_workers),
        "derivative_cache_mode": str(args.derivative_cache_mode),
        "derivative_cache_dir": str(args.derivative_cache_dir).strip(),
        "recalculate_derivative_cache": bool(args.recalculate_derivative_cache),
        "sampling_mode": str(args.sampling_mode),
        "selection_strategy": "selected_regions_direct_compute",
        "sources": source_summaries,
    }
    return selected_records, metadata, sampling_stats


def collect_tile_records(
    pairs: Sequence[SourcePair],
    args: argparse.Namespace,
) -> Tuple[List[TileRecord], Dict[str, object]]:
    band_idx = parse_int_csv(args.bands, expected_len=5)
    tile_size = int(args.tile_size)
    num_workers = max(1, int(args.num_workers))
    stride = int(args.tile_size - args.overlap)
    all_records: List[TileRecord] = []
    source_summaries: List[Dict[str, object]] = []
    total_windows = 0
    for pair in pairs:
        with rasterio.open(pair.raster_path) as src:
            total_windows += len(build_windows(src.width, src.height, tile_size, stride))
    processed_windows = 0
    for pair in pairs:
        with rasterio.open(pair.raster_path) as src, rasterio.open(pair.mask_path) as mask_src:
            if src.width != mask_src.width or src.height != mask_src.height:
                raise ValueError(
                    f"Raster/mask boyutu uyusmuyor: {pair.name} -> "
                    f"{src.width}x{src.height} vs {mask_src.width}x{mask_src.height}"
                )
            validate_source_raster(src, band_idx, pair.raster_path)
            windows = build_windows(src.width, src.height, tile_size, stride)
            if not windows:
                raise ValueError(
                    f"Raster icine tam tile sigmiyor: {pair.raster_path} (tile_size={tile_size})"
                )
            source_counter: Counter[str] = Counter()
            boundary_discarded = 0
            invalid_discarded = 0
            def handle_scan_result(result: dict) -> None:
                nonlocal boundary_discarded, invalid_discarded
                status = str(result.get("status", ""))
                if status == "boundary":
                    boundary_discarded += 1
                    return
                if status == "invalid":
                    invalid_discarded += 1
                    return
                if status != "ok":
                    raise ValueError(f"Bilinmeyen scan sonucu: {status}")
                split_name = str(result["split"])
                label = str(result["label"])
                all_records.append(tile_record_from_scan_result(pair.name, result))
                source_counter[f"{split_name}_{label}"] += 1

            for result in iterate_scan_results(
                pair=pair,
                windows=windows,
                band_idx=band_idx,
                args=args,
            ):
                processed_windows += 1
                if processed_windows == 1 or processed_windows == total_windows or processed_windows % 50 == 0:
                    emit_progress(
                        phase="scan",
                        current=processed_windows,
                        total=total_windows,
                        message=f"Taranan pencere: {processed_windows}/{total_windows} ({pair.name})",
                    )
                handle_scan_result(result)
            source_summaries.append(
                {
                    "name": pair.name,
                    "raster_path": str(pair.raster_path),
                    "mask_path": str(pair.mask_path),
                    "width": int(src.width),
                    "height": int(src.height),
                    "windows_total": len(windows),
                    "boundary_discarded": int(boundary_discarded),
                    "invalid_discarded": int(invalid_discarded),
                    "counts": dict(source_counter),
                }
            )
            emit_progress(
                phase="scan",
                current=processed_windows,
                total=total_windows,
                message=f"Kaynak tamamlandi: {pair.name}",
            )
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "task_type": "tile_classification",
        "dataset_layout": "classification_folders",
        "num_channels": len(MODEL_CHANNEL_NAMES),
        "channel_names": list(MODEL_CHANNEL_NAMES),
        "bands": str(args.bands),
        "tile_size": int(args.tile_size),
        "overlap": int(args.overlap),
        "stride": stride,
        "tpi_radii": list(parse_int_csv(args.tpi_radii)),
        "positive_ratio_threshold": float(args.positive_ratio_threshold),
        "valid_ratio_threshold": float(args.valid_ratio_threshold),
        "negative_to_positive_ratio": float(args.negative_to_positive_ratio),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "train_negative_keep_ratio": float(args.train_negative_keep_ratio),
        "train_negative_max": None if args.train_negative_max is None else int(args.train_negative_max),
        "normalize": bool(args.normalize),
        "save_format": str(args.format),
        "num_workers": int(args.num_workers),
        "derivative_cache_mode": str(args.derivative_cache_mode),
        "derivative_cache_dir": str(args.derivative_cache_dir).strip(),
        "recalculate_derivative_cache": bool(args.recalculate_derivative_cache),
        "sampling_mode": str(args.sampling_mode),
        "sources": source_summaries,
    }
    return all_records, metadata


def resolve_derivative_cache_dir(args: argparse.Namespace, raster_path: Path) -> Path:
    raw = str(getattr(args, "derivative_cache_dir", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return raster_path.resolve().parent / "cache"


def prepare_derivative_cache_for_source(
    *,
    pair: SourcePair,
    band_idx: Sequence[int],
    tpi_radii: Sequence[int],
    args: argparse.Namespace,
) -> PreparedDerivativeCache:
    cache_mode = str(getattr(args, "derivative_cache_mode", "none")).strip().lower()
    if cache_mode == "none":
        return PreparedDerivativeCache(mode="none")

    cache_dir = resolve_derivative_cache_dir(args, pair.raster_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    enable_curvature = False
    enable_tpi = True
    recalculate = bool(getattr(args, "recalculate_derivative_cache", False))

    if cache_mode in ("auto", "npz"):
        precompute_ok, reason = full_raster_cache_precompute_ok(
            pair.raster_path,
            band_idx,
            enable_curvature=enable_curvature,
            enable_tpi=enable_tpi,
        )
        if precompute_ok:
            cache_path = get_cache_path(pair.raster_path, str(cache_dir))
            precomputed = precompute_derivatives(
                input_path=pair.raster_path,
                band_idx=band_idx,
                use_cache=True,
                cache_path=cache_path,
                recalculate=recalculate,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tuple(int(v) for v in tpi_radii),
            )
            if precomputed is not None:
                return PreparedDerivativeCache(
                    mode="npz",
                    location=str(cache_path),
                    precomputed=precomputed,
                )
        elif reason:
            print(
                "BILGI: Tam-raster NPZ cache bu veri icin uygun degil. "
                f"Detay: {reason} Raster-cache denenecek.",
                flush=True,
            )

    cache_tif_path, cache_meta_path = get_derivative_raster_cache_paths(pair.raster_path, str(cache_dir))
    raster_info = None

    def _remove_raster_cache_files() -> None:
        temp_tif_path = cache_tif_path.with_name(f"{cache_tif_path.stem}.building{cache_tif_path.suffix}")
        temp_meta_path = cache_meta_path.with_name(f"{cache_meta_path.stem}.building{cache_meta_path.suffix}")
        for p in (cache_tif_path, cache_meta_path, temp_tif_path, temp_meta_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    if not recalculate:
        raster_info = load_derivative_raster_cache_info(
            cache_tif_path,
            cache_meta_path,
            pair.raster_path,
            band_idx,
            rvt_radii=None,
            gaussian_lrm_sigma=None,
            enable_curvature=enable_curvature,
            enable_tpi=enable_tpi,
            tpi_radii=tpi_radii,
        )
    if raster_info is None:
        _remove_raster_cache_files()
        try:
            build_derivative_raster_cache(
                input_path=pair.raster_path,
                band_idx=band_idx,
                cache_tif_path=cache_tif_path,
                cache_meta_path=cache_meta_path,
                recalculate=True,
                rvt_radii=None,
                gaussian_lrm_sigma=None,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tuple(int(v) for v in tpi_radii),
                chunk_size=2048,
                worker_count=max(1, int(args.num_workers)),
                halo_px=None,
            )
            raster_info = load_derivative_raster_cache_info(
                cache_tif_path,
                cache_meta_path,
                pair.raster_path,
                band_idx,
                rvt_radii=None,
                gaussian_lrm_sigma=None,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tpi_radii,
            )
        except Exception as exc:
            _remove_raster_cache_files()
            print(
                f"UYARI: Raster-cache yeniden olusturulamadi ({pair.name}): {exc}",
                flush=True,
            )
            raster_info = None
    if raster_info is None:
        print(
            f"UYARI: Derivative cache hazirlanamadi, kayit asamasi tile-bazli hesaplamaya donecek ({pair.name}).",
            flush=True,
        )
        return PreparedDerivativeCache(mode="none")

    return PreparedDerivativeCache(
        mode="raster",
        location=str(cache_tif_path),
        raster_cache_tif=cache_tif_path,
        raster_cache_meta=cache_meta_path,
        raster_band_map={str(k): int(v) for k, v in dict(raster_info.get("band_map", {})).items()},
    )


def compute_tile_stack(
    *,
    src: rasterio.DatasetReader,
    window: Window,
    band_idx: Sequence[int],
    tpi_radii: Sequence[int],
    normalize: bool,
) -> np.ndarray:
    data, _ = read_window_data(src, band_idx, window)
    rgb = data[:3]
    dsm = data[3]
    dtm = data[4]
    pixel_size = float((abs(src.transform.a) + abs(src.transform.e)) / 2.0)
    ndsm = compute_ndsm(dsm, dtm)
    svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
        dtm,
        pixel_size=pixel_size,
        show_progress=False,
        log_steps=False,
    )
    tpi = compute_tpi_multiscale(dtm, radii=tuple(int(r) for r in tpi_radii))
    stacked = stack_channels(
        rgb=rgb,
        dsm=dsm,
        dtm=dtm,
        svf=svf,
        pos_open=pos_open,
        neg_open=neg_open,
        lrm=lrm,
        slope=slope,
        ndsm=ndsm,
        tpi=tpi,
    )
    if normalize:
        stacked = robust_norm(stacked)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_tile_stack_from_precomputed(
    *,
    precomputed: PrecomputedDerivatives,
    window: Window,
    normalize: bool,
) -> np.ndarray:
    row_start = int(window.row_off)
    col_start = int(window.col_off)
    row_end = row_start + int(window.height)
    col_end = col_start + int(window.width)

    rgb = precomputed.rgb[:, row_start:row_end, col_start:col_end].copy()
    dsm = (
        precomputed.dsm[row_start:row_end, col_start:col_end].copy()
        if precomputed.dsm is not None
        else None
    )
    dtm = precomputed.dtm[row_start:row_end, col_start:col_end].copy()
    svf = precomputed.svf[row_start:row_end, col_start:col_end].copy()
    pos_open = precomputed.pos_open[row_start:row_end, col_start:col_end].copy()
    neg_open = precomputed.neg_open[row_start:row_end, col_start:col_end].copy()
    lrm = precomputed.lrm[row_start:row_end, col_start:col_end].copy()
    slope = precomputed.slope[row_start:row_end, col_start:col_end].copy()
    ndsm = precomputed.ndsm[row_start:row_end, col_start:col_end].copy()
    tpi = (
        precomputed.tpi[row_start:row_end, col_start:col_end].copy()
        if precomputed.tpi is not None
        else None
    )
    stacked = stack_channels(
        rgb=rgb,
        dsm=dsm,
        dtm=dtm,
        svf=svf,
        pos_open=pos_open,
        neg_open=neg_open,
        lrm=lrm,
        slope=slope,
        ndsm=ndsm,
        tpi=tpi,
    )
    if normalize:
        stacked = robust_norm(stacked)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_tile_stack_from_raster_cache(
    *,
    src: rasterio.DatasetReader,
    derivative_src: rasterio.DatasetReader,
    derivative_band_map: Dict[str, int],
    window: Window,
    band_idx: Sequence[int],
    normalize: bool,
) -> np.ndarray:
    data, _ = read_window_data(src, band_idx, window)
    rgb = data[:3]
    dsm = data[3]
    dtm = data[4]
    band_names = ["svf", "pos_open", "neg_open", "lrm", "slope", "ndsm", "tpi"]
    indexes = [int(derivative_band_map[name]) for name in band_names]
    deriv_stack = derivative_src.read(indexes=indexes, window=window, masked=True)
    deriv_stack = np.ma.filled(deriv_stack.astype(np.float32), np.nan)
    svf, pos_open, neg_open, lrm, slope, ndsm, tpi = deriv_stack
    stacked = stack_channels(
        rgb=rgb,
        dsm=dsm,
        dtm=dtm,
        svf=svf,
        pos_open=pos_open,
        neg_open=neg_open,
        lrm=lrm,
        slope=slope,
        ndsm=ndsm,
        tpi=tpi,
    )
    if normalize:
        stacked = robust_norm(stacked)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _save_tile_array(output_path: Path, stacked: np.ndarray, file_ext: str) -> None:
    if file_ext == "npy":
        np.save(output_path, stacked)
    else:
        np.savez_compressed(output_path, image=stacked)


def _init_save_tile_worker(
    raster_path: str,
    band_idx: Tuple[int, ...],
    tpi_radii: Tuple[int, ...],
    normalize: bool,
    tile_size: int,
    output_dir: str,
    file_ext: str,
    tile_prefix: str,
    derivative_cache_tif: Optional[str] = None,
    derivative_band_map: Optional[Tuple[Tuple[str, int], ...]] = None,
) -> None:
    global _SAVE_TILE_WORKER_CONTEXT
    derivative_src = None
    derivative_map: Optional[Dict[str, int]] = None
    if derivative_cache_tif:
        derivative_src = rasterio.open(str(derivative_cache_tif))
        derivative_map = {str(k): int(v) for k, v in (derivative_band_map or ())}
    _SAVE_TILE_WORKER_CONTEXT = {
        "src": rasterio.open(raster_path),
        "derivative_src": derivative_src,
        "derivative_band_map": derivative_map,
        "band_idx": tuple(int(v) for v in band_idx),
        "tpi_radii": tuple(int(v) for v in tpi_radii),
        "normalize": bool(normalize),
        "tile_size": int(tile_size),
        "output_dir": Path(output_dir),
        "file_ext": str(file_ext),
        "tile_prefix": str(tile_prefix),
    }


def _save_tile_worker(record: TileRecord) -> str:
    ctx = _SAVE_TILE_WORKER_CONTEXT
    src = ctx["src"]
    derivative_src = ctx.get("derivative_src")
    derivative_band_map = ctx.get("derivative_band_map")
    tile_size = int(ctx["tile_size"])
    window = Window(
        col_off=int(record.col_off),
        row_off=int(record.row_off),
        width=tile_size,
        height=tile_size,
    )
    if derivative_src is not None and derivative_band_map is not None:
        stacked = compute_tile_stack_from_raster_cache(
            src=src,
            derivative_src=derivative_src,
            derivative_band_map=derivative_band_map,
            window=window,
            band_idx=ctx["band_idx"],
            normalize=bool(ctx["normalize"]),
        )
    else:
        stacked = compute_tile_stack(
            src=src,
            window=window,
            band_idx=ctx["band_idx"],
            tpi_radii=ctx["tpi_radii"],
            normalize=bool(ctx["normalize"]),
        )
    output_dir = ctx["output_dir"]
    file_ext = str(ctx["file_ext"])
    tile_name = make_tile_name(
        record,
        str(ctx["tile_prefix"]).strip(),
        shape_suffix=_shape_suffix_from_array(stacked),
    )
    output_path = output_dir / record.split / record.label / f"{tile_name}.{file_ext}"
    _save_tile_array(output_path, stacked, file_ext)
    return str(output_path.relative_to(output_dir)).replace("\\", "/")


def _build_write_stage_message(
    *,
    source_name: str,
    source_saved: int,
    source_total: int,
    worker_count: int,
    pending_count: Optional[int] = None,
    warmup: bool = False,
) -> str:
    if source_saved <= 0 or warmup:
        base = f"Ilk tilelar hesaplaniyor: {source_saved}/{source_total} tamamlandi ({source_name})"
    else:
        base = f"Tilelar hesaplaniyor: {source_saved}/{source_total} tamamlandi ({source_name})"

    details: List[str] = []
    if worker_count > 0:
        details.append(f"worker {worker_count}")
    if pending_count is not None:
        details.append(f"bekleyen is {pending_count}")

    if not details:
        return base
    return f"{base} - {', '.join(details)}"


def save_tiles(
    *,
    output_dir: Path,
    pairs: Sequence[SourcePair],
    selected_records: Sequence[TileRecord],
    metadata: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    if not selected_records:
        raise ValueError("Kaydedilecek tile kalmadi.")
    pair_by_name = {pair.name: pair for pair in pairs}
    records_by_source: Dict[str, List[TileRecord]] = defaultdict(list)
    for record in selected_records:
        records_by_source[record.source_name].append(record)
    band_idx = parse_int_csv(args.bands, expected_len=5)
    tpi_radii = parse_int_csv(args.tpi_radii)
    file_ext = str(args.format)
    num_workers = int(args.num_workers)
    total_records = len(selected_records)
    saved_count = 0
    cache_records: List[Dict[str, object]] = []
    total_sources = max(1, len(records_by_source))
    direct_selected_region_mode = str(getattr(args, "sampling_mode", "")).strip().lower() == SAMPLING_MODE_SELECTED_REGIONS
    for source_name in sorted(records_by_source):
        pair = pair_by_name[source_name]
        source_records = sorted(
            records_by_source[source_name],
            key=lambda item: (SPLIT_ORDER.index(item.split), item.row_off, item.col_off),
        )
        if direct_selected_region_mode:
            emit_progress(
                phase="cache",
                current=len(cache_records) + 1,
                total=total_sources,
                message=f"Turev cache atlandi: {pair.name} (selected_regions)",
            )
            prepared_cache = PreparedDerivativeCache(mode="selected_regions_direct")
        else:
            emit_progress(
                phase="cache",
                current=len(cache_records) + 1,
                total=total_sources,
                message=f"Turev cache hazirlaniyor: {pair.name}",
            )
            prepared_cache = prepare_derivative_cache_for_source(
                pair=pair,
                band_idx=band_idx,
                tpi_radii=tpi_radii,
                args=args,
            )
        cache_records.append(
            {
                "source_name": pair.name,
                "mode": prepared_cache.mode,
                "location": prepared_cache.location,
            }
        )
        source_total_records = len(source_records)
        source_saved_before = saved_count

        if prepared_cache.precomputed is not None:
            emit_progress(
                phase="write",
                current=saved_count,
                total=total_records,
                message=_build_write_stage_message(
                    source_name=pair.name,
                    source_saved=0,
                    source_total=source_total_records,
                    worker_count=0,
                    pending_count=source_total_records,
                    warmup=True,
                ),
            )
            for record in source_records:
                window = Window(
                    col_off=int(record.col_off),
                    row_off=int(record.row_off),
                    width=int(args.tile_size),
                    height=int(args.tile_size),
                )
                stacked = compute_tile_stack_from_precomputed(
                    precomputed=prepared_cache.precomputed,
                    window=window,
                    normalize=bool(args.normalize),
                )
                tile_name = make_tile_name(
                    record,
                    str(args.tile_prefix).strip(),
                    shape_suffix=_shape_suffix_from_array(stacked),
                )
                output_path = output_dir / record.split / record.label / f"{tile_name}.{file_ext}"
                _save_tile_array(output_path, stacked, file_ext)
                record.output_relpath = str(output_path.relative_to(output_dir)).replace("\\", "/")
                saved_count += 1
                source_saved = saved_count - source_saved_before
                if saved_count == 1 or saved_count == total_records or saved_count % 25 == 0:
                    emit_progress(
                        phase="write",
                        current=saved_count,
                        total=total_records,
                        message=_build_write_stage_message(
                            source_name=pair.name,
                            source_saved=source_saved,
                            source_total=source_total_records,
                            worker_count=0,
                            pending_count=max(0, source_total_records - source_saved),
                        ),
                    )
            continue

        if num_workers == 1:
            emit_progress(
                phase="write",
                current=saved_count,
                total=total_records,
                message=_build_write_stage_message(
                    source_name=pair.name,
                    source_saved=0,
                    source_total=source_total_records,
                    worker_count=1,
                    pending_count=source_total_records,
                    warmup=True,
                ),
            )
            derivative_src = None
            if prepared_cache.raster_cache_tif is not None and prepared_cache.raster_band_map is not None:
                derivative_src = rasterio.open(prepared_cache.raster_cache_tif)
            with rasterio.open(pair.raster_path) as src:
                for record in source_records:
                    window = Window(
                        col_off=int(record.col_off),
                        row_off=int(record.row_off),
                        width=int(args.tile_size),
                        height=int(args.tile_size),
                    )
                    if derivative_src is not None and prepared_cache.raster_band_map is not None:
                        stacked = compute_tile_stack_from_raster_cache(
                            src=src,
                            derivative_src=derivative_src,
                            derivative_band_map=prepared_cache.raster_band_map,
                            window=window,
                            band_idx=band_idx,
                            normalize=bool(args.normalize),
                        )
                    else:
                        stacked = compute_tile_stack(
                            src=src,
                            window=window,
                            band_idx=band_idx,
                            tpi_radii=tpi_radii,
                            normalize=bool(args.normalize),
                        )
                    tile_name = make_tile_name(
                        record,
                        str(args.tile_prefix).strip(),
                        shape_suffix=_shape_suffix_from_array(stacked),
                    )
                    output_path = output_dir / record.split / record.label / f"{tile_name}.{file_ext}"
                    _save_tile_array(output_path, stacked, file_ext)
                    record.output_relpath = str(output_path.relative_to(output_dir)).replace("\\", "/")
                    saved_count += 1
                    source_saved = saved_count - source_saved_before
                    if saved_count == 1 or saved_count == total_records or saved_count % 25 == 0:
                        emit_progress(
                            phase="write",
                            current=saved_count,
                            total=total_records,
                            message=_build_write_stage_message(
                                source_name=pair.name,
                                source_saved=source_saved,
                                source_total=source_total_records,
                                worker_count=1,
                                pending_count=max(0, source_total_records - source_saved),
                            ),
                        )
            if derivative_src is not None:
                derivative_src.close()
            continue

        worker_count = max(1, min(num_workers, source_total_records))
        emit_progress(
            phase="write",
            current=saved_count,
            total=total_records,
            message=_build_write_stage_message(
                source_name=pair.name,
                source_saved=0,
                source_total=source_total_records,
                worker_count=worker_count,
                pending_count=source_total_records,
                warmup=True,
            ),
        )
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_save_tile_worker,
            initargs=(
                str(pair.raster_path),
                tuple(int(v) for v in band_idx),
                tuple(int(v) for v in tpi_radii),
                bool(args.normalize),
                int(args.tile_size),
                str(output_dir),
                file_ext,
                str(args.tile_prefix).strip(),
                str(prepared_cache.raster_cache_tif) if prepared_cache.raster_cache_tif is not None else None,
                tuple(sorted((prepared_cache.raster_band_map or {}).items())),
            ),
        ) as executor:
            future_to_record = {
                executor.submit(_save_tile_worker, record): record for record in source_records
            }
            pending_futures = set(future_to_record)
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=5.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    source_saved = saved_count - source_saved_before
                    emit_progress(
                        phase="write",
                        current=saved_count,
                        total=total_records,
                        message=_build_write_stage_message(
                            source_name=pair.name,
                            source_saved=source_saved,
                            source_total=source_total_records,
                            worker_count=worker_count,
                            pending_count=len(pending_futures),
                            warmup=source_saved <= 0,
                        ),
                    )
                    continue
                for future in done_futures:
                    record = future_to_record[future]
                    record.output_relpath = future.result()
                    saved_count += 1
                source_saved = saved_count - source_saved_before
                if (
                    saved_count == 1
                    or saved_count == total_records
                    or saved_count % 25 == 0
                    or source_saved == 1
                    or source_saved == source_total_records
                ):
                    emit_progress(
                        phase="write",
                        current=saved_count,
                        total=total_records,
                        message=_build_write_stage_message(
                            source_name=pair.name,
                            source_saved=source_saved,
                            source_total=source_total_records,
                            worker_count=worker_count,
                            pending_count=len(pending_futures),
                        ),
                    )
    metadata["saved_tiles"] = int(saved_count)
    metadata["derivative_cache"] = cache_records


def write_manifests(
    *,
    output_dir: Path,
    selected_records: Sequence[TileRecord],
    metadata: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    manifest_path = output_dir / "tiles_manifest.csv"
    tile_labels_path = output_dir / "tile_labels.csv"
    total_records = max(1, len(selected_records))
    emit_progress(
        phase="manifest",
        current=0,
        total=total_records,
        message="Manifest dosyalari yaziliyor...",
    )
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_fp, open(
        tile_labels_path, "w", newline="", encoding="utf-8"
    ) as tile_labels_fp:
        manifest_writer = csv.writer(manifest_fp)
        tile_labels_writer = csv.writer(tile_labels_fp)
        manifest_writer.writerow(
            [
                "tile_name",
                "split",
                "label",
                "image_relpath",
                "source_name",
                "row_off",
                "col_off",
                "positive_ratio",
                "valid_ratio",
                "positive_pixels",
                "total_pixels",
            ]
        )
        tile_labels_writer.writerow(
            [
                "tile_name",
                "split",
                "image_relpath",
                "mask_relpath",
                "row_off",
                "col_off",
                "tile_label",
                "positive_ratio",
                "positive_pixels",
                "total_pixels",
                "label_threshold",
            ]
        )
        counts_by_split_label: Counter[str] = Counter()
        for index, record in enumerate(selected_records, start=1):
            tile_name = Path(record.output_relpath).stem
            manifest_writer.writerow(
                [
                    tile_name,
                    record.split,
                    record.label,
                    record.output_relpath,
                    record.source_name,
                    record.row_off,
                    record.col_off,
                    f"{record.positive_ratio:.8f}",
                    f"{record.valid_ratio:.8f}",
                    record.positive_pixels,
                    record.total_pixels,
                ]
            )
            tile_labels_writer.writerow(
                [
                    tile_name,
                    record.split,
                    record.output_relpath,
                    "",
                    record.row_off,
                    record.col_off,
                    1 if record.label == LABEL_POSITIVE else 0,
                    f"{record.positive_ratio:.8f}",
                    record.positive_pixels,
                    record.total_pixels,
                    f"{float(args.positive_ratio_threshold):.8f}",
                ]
            )
            counts_by_split_label[f"{record.split}_{record.label}"] += 1
            if index == 1 or index == len(selected_records) or index % 100 == 0:
                emit_progress(
                    phase="manifest",
                    current=index,
                    total=total_records,
                    message=f"Manifest yaziliyor: {index}/{len(selected_records)}",
                )
    metadata["selected_counts"] = dict(counts_by_split_label)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        pairs = derive_source_pairs(args)
        prepare_output_dir(
            args.output_dir,
            overwrite=bool(args.overwrite),
            include_test=float(args.test_ratio) > 0.0,
        )
        emit_progress(phase="start", current=0, total=1, message="Kaynaklar taraniyor...")
        if args.sampling_mode == SAMPLING_MODE_SELECTED_REGIONS:
            selected_records, metadata, sampling_stats = collect_selected_region_records(
                pairs,
                args,
            )
        else:
            records, metadata = collect_tile_records(pairs, args)
            selected_records, sampling_stats = select_train_negative_records(
                records,
                keep_ratio=float(args.train_negative_keep_ratio),
                max_count=args.train_negative_max,
                seed=int(args.seed),
            )
        metadata["sampling_stats"] = sampling_stats
        if not selected_records:
            raise ValueError("Secilen ayarlar ile hic tile kalmadi.")
        emit_progress(
            phase="select",
            current=len(selected_records),
            total=max(1, len(selected_records) if args.sampling_mode == SAMPLING_MODE_SELECTED_REGIONS else len(records)),
            message=(
                f"Secilen tile sayisi: {len(selected_records)}"
                if args.sampling_mode == SAMPLING_MODE_SELECTED_REGIONS
                else f"Secilen tile sayisi: {len(selected_records)} / {len(records)}"
            ),
        )
        save_tiles(
            output_dir=args.output_dir,
            pairs=pairs,
            selected_records=selected_records,
            metadata=metadata,
            args=args,
        )
        write_manifests(
            output_dir=args.output_dir,
            selected_records=selected_records,
            metadata=metadata,
            args=args,
        )
        emit_progress(phase="done", current=1, total=1, message="Tile dataset hazir.")
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "selected_tiles": len(selected_records),
                    "sampling_mode": args.sampling_mode,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
