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
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from archeo_shared.channels import METADATA_SCHEMA_VERSION, MODEL_CHANNEL_NAMES

try:
    from archaeo_detect import (
        compute_derivatives_with_rvt,
        compute_ndsm,
        compute_tpi_multiscale,
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


def positive_ratio_from_mask(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
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
    positive_pixels = int(np.count_nonzero(valid & (mask > 0)))
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


def make_tile_name(record: TileRecord, tile_prefix: str) -> str:
    base = f"{record.source_name}_r{record.row_off:05d}_c{record.col_off:05d}"
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


def collect_tile_records(
    pairs: Sequence[SourcePair],
    args: argparse.Namespace,
) -> Tuple[List[TileRecord], Dict[str, object]]:
    band_idx = parse_int_csv(args.bands, expected_len=5)
    tile_size = int(args.tile_size)
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
            for row_off, col_off in windows:
                processed_windows += 1
                if processed_windows == 1 or processed_windows == total_windows or processed_windows % 200 == 0:
                    emit_progress(
                        phase="scan",
                        current=processed_windows,
                        total=total_windows,
                        message=f"Taranan pencere: {processed_windows}/{total_windows} ({pair.name})",
                    )
                split_name = split_window_by_rows(
                    row_off=row_off,
                    tile_size=tile_size,
                    raster_height=int(src.height),
                    train_ratio=float(args.train_ratio),
                    val_ratio=float(args.val_ratio),
                    test_ratio=float(args.test_ratio),
                )
                if split_name is None:
                    boundary_discarded += 1
                    continue
                window = Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size)
                _, valid_mask = read_window_data(src, band_idx, window)
                valid_ratio = float(valid_mask.mean()) if valid_mask.size > 0 else 0.0
                if valid_ratio < float(args.valid_ratio_threshold):
                    invalid_discarded += 1
                    continue
                mask = mask_src.read(1, window=window).astype(np.float32)
                pos_ratio, positive_pixels, total_pixels = positive_ratio_from_mask(mask, valid_mask=valid_mask)
                label = LABEL_POSITIVE if pos_ratio >= float(args.positive_ratio_threshold) else LABEL_NEGATIVE
                all_records.append(
                    TileRecord(
                        source_name=pair.name,
                        split=split_name,
                        label=label,
                        row_off=int(row_off),
                        col_off=int(col_off),
                        positive_ratio=float(pos_ratio),
                        valid_ratio=float(valid_ratio),
                        positive_pixels=positive_pixels,
                        total_pixels=total_pixels,
                    )
                )
                source_counter[f"{split_name}_{label}"] += 1
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
        "sampling_mode": str(args.sampling_mode),
        "sources": source_summaries,
    }
    return all_records, metadata


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
    total_records = len(selected_records)
    saved_count = 0
    for source_name in sorted(records_by_source):
        pair = pair_by_name[source_name]
        with rasterio.open(pair.raster_path) as src:
            source_records = sorted(
                records_by_source[source_name],
                key=lambda item: (SPLIT_ORDER.index(item.split), item.row_off, item.col_off),
            )
            for record in source_records:
                window = Window(
                    col_off=int(record.col_off),
                    row_off=int(record.row_off),
                    width=int(args.tile_size),
                    height=int(args.tile_size),
                )
                stacked = compute_tile_stack(
                    src=src,
                    window=window,
                    band_idx=band_idx,
                    tpi_radii=tpi_radii,
                    normalize=bool(args.normalize),
                )
                tile_name = make_tile_name(record, str(args.tile_prefix).strip())
                output_path = output_dir / record.split / record.label / f"{tile_name}.{file_ext}"
                if file_ext == "npy":
                    np.save(output_path, stacked)
                else:
                    np.savez_compressed(output_path, image=stacked)
                record.output_relpath = str(output_path.relative_to(output_dir)).replace("\\", "/")
                saved_count += 1
                if saved_count == 1 or saved_count == total_records or saved_count % 50 == 0:
                    emit_progress(
                        phase="write",
                        current=saved_count,
                        total=total_records,
                        message=f"Tile kaydi: {saved_count}/{total_records}",
                    )
    metadata["saved_tiles"] = int(saved_count)


def write_manifests(
    *,
    output_dir: Path,
    selected_records: Sequence[TileRecord],
    metadata: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    manifest_path = output_dir / "tiles_manifest.csv"
    tile_labels_path = output_dir / "tile_labels.csv"
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
        for record in selected_records:
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
        records, metadata = collect_tile_records(pairs, args)
        if args.sampling_mode == SAMPLING_MODE_SELECTED_REGIONS:
            selected_records, sampling_stats = select_records_for_selected_regions_mode(
                records,
                negative_to_positive_ratio=float(args.negative_to_positive_ratio),
                seed=int(args.seed),
            )
        else:
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
            total=max(1, len(records)),
            message=f"Secilen tile sayisi: {len(selected_records)} / {len(records)}",
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
