#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arkeolojik Alan Tespiti - 5 Kanallı Eğitim Verisi Oluşturma Scripti

Bu script, çok bantlı GeoTIFF dosyalarından ve karşılık gelen ground truth 
maskelerinden 5 kanallı eğitim tile'ları oluşturur.

Kanal Yapısı (5 kanal):
    [0-2]: RGB (R, G, B)
    [3]: SVF (Sky-View Factor)
    [4]: SLRM (Simplified Local Relief Model)

Kullanım:
    python egitim_verisi_olusturma.py --input kesif_alani.tif --mask ground_truth.tif --output training_data
    python egitim_verisi_olusturma.py  # CONFIG bolumundeki varsayilanlarla calisir

Gereksinimler:
    - Çok bantlı GeoTIFF (RGB + DSM + DTM)
    - Ground truth maske (arkeolojik alanlar = 1, arka plan = 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window
from tqdm import tqdm

from archeo_shared.channels import METADATA_SCHEMA_VERSION, MODEL_CHANNEL_NAMES

# Mevcut projedeki fonksiyonları import et
try:
    from archaeo_detect import (
        compute_derivatives_with_rvt,
        stack_channels,
        robust_norm,
    )
except ImportError:
    print("HATA: archaeo_detect.py bulunamadı!")
    print("Bu scripti archaeo_detect.py ile aynı dizinde çalıştırın.")
    sys.exit(1)

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("egitim_verisi")

# ==================== CONFIG ====================
# IDE uzerinden dogrudan "Run" ettiginizde bu degerler kullanilir.
# Komut satirindan verilen argumanlar her zaman bu degerleri ezer.
CONFIG: dict[str, object] = {
    # input:
    # Cok bantli raster dosyasinin yolu.
    # Beklenen icerik: RGB + DSM + DTM (hangi bant hangi role gidecek, bands ile belirlenir).
    "input": "on_veri/karlik_vadi_rgb_dtm_dsm_5band.tif",
    #"input": "on_veri/karlik_dag_rgb_dtm_dsm_5band.tif",

    # mask:
    # Ground-truth maske dosyasinin yolu.
    # 0 disindaki tum degerler pozitif sinif olarak ele alinir (otomatik 0/1'e cevrilir).
    "mask": "on_veri/karlik_vadi_rgb_ground_truth.tif",
    #"mask": "on_veri/karlik_dag_rgb_ground_truth.tif",


    # output:
    # Cikti kok dizini.
    # train/val, images/masks, metadata ve tile_presence dosyalari bu dizine yazilir.
    "output": "training_data",

    # tile_size:
    # Her ornegin (tile) uzamsal boyutu.
    # Buyuk tile daha fazla baglam verir; disk, RAM ve islem suresini artirir.
    "tile_size": 256,

    # overlap:
    # Komsu tile'larin kac piksel ust uste binecegi.
    # Yuksek overlap daha fazla ornek ureterek sinir gecislerini yumusatir, tekrarli veri artar.
    "overlap": 128,

    # bands:
    # Girdi rasterindaki bant sirasi: "R,G,B,DSM,DTM".
    # GeoTIFF bant indeksleri 1 tabanlidir (1,2,3,...).
    "bands": "1,2,3,4,5",

    # min_positive:
    # Tile'in kabul edilmesi icin gereken minimum pozitif piksel orani.
    # 0.0 oldugunda tamamen negatif tile'lar da dahil edilir.
    "min_positive": 0.0,

    # tile_label_min_positive_ratio:
    # Tile-level classification icin explicit tile etiketi uretilirken kullanilan
    # pozitif piksel orani esigi.
    # 0.0 ise tile icinde en az bir pozitif piksel olmasi yeterlidir.
    # Bu ayar tile'i veri setine dahil etme filtresi DEGILDIR; sadece tile label
    # uretimi icindir.
    "tile_label_min_positive_ratio": 0.02,

    # max_nodata:
    # Bir tile icin izin verilen maksimum gecersiz/nodata orani.
    # Bu esik asilirsa tile atlanir.
    "max_nodata": 0.3,

    # train_ratio:
    # Train/val bolme orani (0-1).
    # Ornek: 0.8 -> pencerelerin yaklasik %80'i train, %20'si val.
    "train_ratio": 0.8,

    # split_mode:
    # Train/val ayirma stratejisi.
    # spatial: mekansal sizintiyi azaltmak icin siniri kesen tile'lari dislar.
    # random: klasik rastgele bolme.
    "split_mode": "spatial",  # spatial | random

    # normalize:
    # True ise her kanal robust sekilde 0-1 araligina cekilir (persentil kirpma tabanli).
    # Bu, farkli olceklerdeki kanallari (RGB/slope/curvature vb.) benzer araliga getirir.
    # False yaparsan ham degerler korunur; model egitimi genelde daha zorlasir.
    "normalize": True,

    # format:
    # Kayit formati.
    # npz: sikistirilmis (daha az disk), npy: sikistirmasiz (genelde daha hizli I/O).
    "format": "npz",  # npy | npz

    # num_workers:
    # Tile uretiminde kullanilacak paralel isci sayisi.
    # 1 = sekansiyel; yuksek deger CPU ve disk I/O yukunu artirir.
    "num_workers": 8,                 # max(1, (os.cpu_count() or 1) - 1) / 2,

    # train_negative_keep_ratio:
    # Train setindeki tamamen negatif tile'larin ne kadarinin tutulacagi (0-1).
    # 1.0 = hepsini tut, 0.0 = hepsini at. Pozitif tile'lar bu filtreden etkilenmez.
    "train_negative_keep_ratio": 0.2,

    # train_negative_max:
    # Train setinde tutulacak negatif tile sayisina ust sinir.
    # None ise limit uygulanmaz.
    "train_negative_max": None,

    # tile_prefix:
    # Tile dosya adlarina eklenecek on ek.
    # Bos birakilirsa tile_size/overlap/bands/timestamp ile otomatik ad uretilir.
    "tile_prefix": "",

    # append:
    # True: mevcut output icine yeni tile ekler (silmeden).
    # False: once eski tile dosyalarini temizler, sonra yeniden uretir.
    "append": True,
}
# ===============================================


def _validate_tile_generation_params(
    tile_size: int,
    overlap: int,
    min_positive_ratio: float,
    max_nodata_ratio: float,
    train_ratio: float,
    save_format: str,
    split_mode: str,
    tile_label_min_positive_ratio: float = 0.0,
    train_negative_keep_ratio: float = 1.0,
    train_negative_max: Optional[int] = None,
    num_workers: int = 1,
) -> None:
    errors: List[str] = []

    if tile_size <= 0:
        errors.append(f"tile_size pozitif olmali, verilen: {tile_size}")
    if overlap < 0:
        errors.append(f"overlap negatif olamaz, verilen: {overlap}")
    if overlap >= tile_size:
        errors.append(f"overlap ({overlap}) tile_size'dan ({tile_size}) kucuk olmali")

    if not 0.0 <= min_positive_ratio <= 1.0:
        errors.append(
            f"min_positive_ratio 0-1 arasinda olmali, verilen: {min_positive_ratio}"
        )
    if not 0.0 <= tile_label_min_positive_ratio <= 1.0:
        errors.append(
            "tile_label_min_positive_ratio 0-1 arasinda olmali, "
            f"verilen: {tile_label_min_positive_ratio}"
        )
    if not 0.0 <= max_nodata_ratio <= 1.0:
        errors.append(
            f"max_nodata_ratio 0-1 arasinda olmali, verilen: {max_nodata_ratio}"
        )
    if not 0.0 < train_ratio < 1.0:
        errors.append(f"train_ratio 0 ile 1 arasinda olmali, verilen: {train_ratio}")

    if save_format not in {"npy", "npz"}:
        errors.append(f"save_format gecersiz: {save_format} (npy/npz)")

    if split_mode not in {"spatial", "random"}:
        errors.append(f"split_mode gecersiz: {split_mode} (spatial/random)")

    if not 0.0 <= float(train_negative_keep_ratio) <= 1.0:
        errors.append(
            "train_negative_keep_ratio 0-1 arasinda olmali, "
            f"verilen: {train_negative_keep_ratio}"
        )

    if train_negative_max is not None and int(train_negative_max) < 0:
        errors.append(
            f"train_negative_max None veya >=0 olmali, verilen: {train_negative_max}"
        )

    if int(num_workers) < 1:
        errors.append(f"num_workers >= 1 olmali, verilen: {num_workers}")

    if errors:
        raise ValueError("Parametre dogrulama hatalari:\n- " + "\n- ".join(errors))


def _sanitize_source_name(name: str) -> str:
    """Filesystem-safe token for source names."""
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name)
    safe = safe.strip("_")
    return safe or "source"


def _prepare_output_dirs(
    output_dir: Path,
    *,
    save_format: str,
    clean_output: bool,
) -> Tuple[Path, Path, Path, Path]:
    """
    Prepare train/val image+mask directories.

    clean_output=True:
        remove old tile files to avoid stale mixing.
    clean_output=False:
        keep old files but forbid mixing .npy and .npz.
    """
    train_images_dir = output_dir / "train" / "images"
    train_masks_dir = output_dir / "train" / "masks"
    val_images_dir = output_dir / "val" / "images"
    val_masks_dir = output_dir / "val" / "masks"
    tile_dirs = [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]

    for d in tile_dirs:
        d.mkdir(parents=True, exist_ok=True)

    if clean_output:
        removed_count = 0
        for d in tile_dirs:
            for pattern in ("*.npy", "*.npz"):
                for tile_path in d.glob(pattern):
                    tile_path.unlink()
                    removed_count += 1

        for stale_artifact in (
            output_dir / "metadata.json",
            output_dir / "tile_presence_scores.csv",
            output_dir / "tile_labels.csv",
            output_dir / "tile_presence_grid.tif",
            output_dir / "tile_presence_grid_rgb.tif",
        ):
            if stale_artifact.exists():
                stale_artifact.unlink()

        if removed_count > 0:
            LOGGER.info("Onceki calismadan %d tile dosyasi temizlendi.", removed_count)
    else:
        other_ext = ".npz" if save_format == "npy" else ".npy"
        mixed_dir_samples: List[str] = []
        for d in tile_dirs:
            if any(d.glob(f"*{other_ext}")):
                mixed_dir_samples.append(str(d))

        if mixed_dir_samples:
            raise ValueError(
                "clean_output=False iken farkli formatta eski dosyalar bulundu. "
                f"Beklenen format: .{save_format}, bulunan eski format: {other_ext}. "
                "Format karisimi olmamasi icin --append kullanmayin veya dizini temizleyin. "
                f"Sorunlu dizinler: {mixed_dir_samples}"
            )

    return train_images_dir, train_masks_dir, val_images_dir, val_masks_dir


def _parse_csv_int_tuple(raw: Any) -> Tuple[int, ...]:
    """Parse CSV/list-like value into integer tuple."""
    if raw is None:
        return tuple()
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    text = str(raw).strip()
    if not text:
        return tuple()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _load_first_array_shape(data_dir: Path, *, save_format: str, key: str) -> Optional[Tuple[int, ...]]:
    """Load shape of first array under data_dir for a given save format."""
    pattern = f"*.{save_format}"
    for path in sorted(data_dir.glob(pattern)):
        if save_format == "npy":
            arr = np.load(path, mmap_mode="r")
            return tuple(int(v) for v in arr.shape)
        with np.load(path) as packed:
            if key not in packed:
                raise ValueError(f"Append kontrolu: '{key}' anahtari bulunamadi: {path}")
            arr = packed[key]
            return tuple(int(v) for v in arr.shape)
    return None


def _validate_append_compatibility(
    output_dir: Path,
    *,
    tile_size: int,
    bands: str,
    normalize: bool,
    save_format: str,
    expected_channels: int = 5,
) -> None:
    """
    Ensure append mode does not silently mix incompatible tile datasets.

    Checks metadata and existing tile shapes/channels.
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    mismatches: List[str] = []

    # 1) Existing tile shape/channel checks (works even if metadata is missing).
    image_shape: Optional[Tuple[int, ...]] = None
    mask_shape: Optional[Tuple[int, ...]] = None
    for split in ("train", "val"):
        if image_shape is None:
            image_shape = _load_first_array_shape(
                output_dir / split / "images",
                save_format=save_format,
                key="image",
            )
        if mask_shape is None:
            mask_shape = _load_first_array_shape(
                output_dir / split / "masks",
                save_format=save_format,
                key="mask",
            )

    if image_shape is not None:
        if len(image_shape) != 3:
            mismatches.append(
                f"mevcut image shape gecersiz ({image_shape}); beklenen (C,H,W)"
            )
        else:
            old_channels, old_h, old_w = image_shape
            if (old_h, old_w) != (int(tile_size), int(tile_size)):
                mismatches.append(
                    f"tile_size uyusmuyor (mevcut: {old_h}x{old_w}, yeni: {tile_size}x{tile_size})"
                )
            if int(old_channels) != int(expected_channels):
                mismatches.append(
                    f"kanal sayisi uyusmuyor (mevcut: {old_channels}, yeni: {expected_channels})"
                )

    if mask_shape is not None:
        if len(mask_shape) != 2:
            mismatches.append(
                f"mevcut mask shape gecersiz ({mask_shape}); beklenen (H,W)"
            )
        elif mask_shape != (int(tile_size), int(tile_size)):
            mismatches.append(
                f"mask tile_size uyusmuyor (mevcut: {mask_shape[0]}x{mask_shape[1]}, yeni: {tile_size}x{tile_size})"
            )

    # 2) Metadata checks for semantic consistency.
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            raise ValueError(
                f"Append kontrolu icin metadata okunamadi: {metadata_path} ({exc})"
            ) from exc

        if not isinstance(metadata, dict):
            raise ValueError(f"Append kontrolu: metadata dict olmali ({metadata_path})")

        def _check(name: str, old_val: Any, new_val: Any) -> None:
            if old_val is None:
                return
            if old_val != new_val:
                mismatches.append(
                    f"{name} uyusmuyor (mevcut: {old_val}, yeni: {new_val})"
                )

        _check("save_format", metadata.get("save_format"), str(save_format))
        if metadata.get("tile_size") is not None:
            _check("tile_size", int(metadata["tile_size"]), int(tile_size))
        if metadata.get("num_channels") is not None:
            _check(
                "num_channels",
                int(metadata["num_channels"]),
                int(expected_channels),
            )
        if metadata.get("normalize") is not None:
            _check("normalize", bool(metadata["normalize"]), bool(normalize))
        if metadata.get("bands") is not None:
            try:
                old_bands = _parse_csv_int_tuple(metadata["bands"])
                new_bands = _parse_csv_int_tuple(bands)
                _check("bands", old_bands, new_bands)
            except ValueError:
                mismatches.append(
                    f"bands degeri parse edilemedi (mevcut: {metadata.get('bands')}, yeni: {bands})"
                )
    elif image_shape is not None or mask_shape is not None:
        LOGGER.warning(
            "Append modu: metadata.json bulunamadi, yalnizca tile shape/kanal uyumlulugu kontrol edildi."
        )

    if mismatches:
        lines = "\n- ".join(mismatches)
        raise ValueError(
            "Append modu uyumsuz veri karisimi olusturur:\n- "
            + lines
            + "\nCozum: ayni ayarlarla append edin, farkli ayarlar icin ayri output dizini kullanin "
            "veya --no-append ile temiz uretim yapin."
        )


def _split_windows_for_train_val(
    windows: List[Tuple[int, int]],
    *,
    train_ratio: float,
    tile_size: int,
    raster_height: int,
    split_mode: str,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int, str]:
    """
    Split windows into train/val sets.

    spatial mode:
        Uses a horizontal split line and only keeps tiles fully above/below that line.
        Tiles crossing the split line are discarded to avoid train/val spatial overlap leakage.
    """
    if not windows:
        return [], [], 0, split_mode

    rng = np.random.RandomState(seed)
    windows_copy = list(windows)

    if split_mode == "random":
        rng.shuffle(windows_copy)
        train_count = int(len(windows_copy) * train_ratio)
        return windows_copy[:train_count], windows_copy[train_count:], 0, "random"

    split_row = int(raster_height * train_ratio)
    train_windows: List[Tuple[int, int]] = []
    val_windows: List[Tuple[int, int]] = []
    boundary_discarded = 0

    for row_off, col_off in windows_copy:
        if row_off + tile_size <= split_row:
            train_windows.append((row_off, col_off))
        elif row_off >= split_row:
            val_windows.append((row_off, col_off))
        else:
            boundary_discarded += 1

    if len(train_windows) == 0 or len(val_windows) == 0:
        rng.shuffle(windows_copy)
        train_count = int(len(windows_copy) * train_ratio)
        return windows_copy[:train_count], windows_copy[train_count:], boundary_discarded, "random_fallback"

    rng.shuffle(train_windows)
    rng.shuffle(val_windows)
    return train_windows, val_windows, boundary_discarded, "spatial"


def _is_positive_for_balance(
    positive_ratio: float,
    min_positive_ratio: float,
) -> bool:
    """
    Decide whether a tile is positive for balancing.

    Bug fix:
    - `min_positive_ratio=0.0` should still treat fully empty masks (`ratio=0`) as negative.
    """
    if positive_ratio <= 0.0:
        return False
    threshold = float(min_positive_ratio)
    if threshold <= 0.0:
        return True
    return positive_ratio >= threshold


def _positive_ratio_to_tile_label(
    positive_ratio: float,
    min_positive_ratio: float,
) -> int:
    """Pozitif oranini tile-level 0/1 etikete cevir."""
    return int(_is_positive_for_balance(positive_ratio, min_positive_ratio))


def _sample_windows_without_replacement(
    windows: List[Tuple[int, int]],
    *,
    target_count: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Select random windows without replacement."""
    if target_count <= 0 or not windows:
        return []
    if target_count >= len(windows):
        return list(windows)
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(windows), size=target_count, replace=False)
    return [windows[int(i)] for i in sampled_indices]


def _filter_train_negative_windows(
    train_windows: List[Tuple[int, int]],
    *,
    mask_src,
    tile_size: int,
    keep_ratio: float,
    max_count: Optional[int],
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], dict]:
    """
    Downsample fully-negative windows on train split.

    Positive windows (mask contains any >0 pixel) are always kept.
    """
    train_windows = list(train_windows)
    if not train_windows:
        return [], {
            "train_negative_before_filter": 0,
            "train_negative_after_filter": 0,
            "train_negative_removed_by_filter": 0,
            "train_positive_before_filter": 0,
            "train_positive_after_filter": 0,
        }

    positive_train: List[Tuple[int, int]] = []
    negative_train: List[Tuple[int, int]] = []

    for row_off, col_off in tqdm(train_windows, desc="Train negatif analiz", leave=False):
        window = Window(col_off, row_off, tile_size, tile_size)
        mask = mask_src.read(1, window=window)
        if np.any(mask > 0):
            positive_train.append((row_off, col_off))
        else:
            negative_train.append((row_off, col_off))

    negative_before = len(negative_train)
    if keep_ratio >= 1.0 and max_count is None:
        selected_negative = list(negative_train)
    else:
        target_negative_count = int(negative_before * float(keep_ratio))
        target_negative_count = min(target_negative_count, negative_before)
        if max_count is not None:
            target_negative_count = min(target_negative_count, int(max_count))
        target_negative_count = max(0, int(target_negative_count))
        selected_negative = _sample_windows_without_replacement(
            negative_train,
            target_count=target_negative_count,
            seed=seed,
        )

    filtered_train = positive_train + selected_negative
    rng = np.random.RandomState(seed)
    rng.shuffle(filtered_train)

    filter_stats = {
        "train_negative_before_filter": negative_before,
        "train_negative_after_filter": len(selected_negative),
        "train_negative_removed_by_filter": negative_before - len(selected_negative),
        "train_positive_before_filter": len(positive_train),
        "train_positive_after_filter": len(positive_train),
    }
    return filtered_train, filter_stats


def _probability_to_rgb_grid(
    probabilities: np.ndarray,
    *,
    nodata_value: float = -1.0,
) -> np.ndarray:
    """
    Convert probability grid [0,1] to RGB heatmap (3,H,W) uint8.

    Color ramp:
      0.0 -> blue
      0.5 -> yellow
      1.0 -> red
      nodata -> black
    """
    probs = probabilities.astype(np.float32, copy=False)
    h, w = probs.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)

    valid = np.isfinite(probs) & (probs >= 0.0) & (probs != float(nodata_value))
    if not np.any(valid):
        return rgb

    t = np.clip(probs, 0.0, 1.0)
    low = valid & (t <= 0.5)
    high = valid & (t > 0.5)

    if np.any(low):
        u = (t[low] / 0.5).astype(np.float32)
        rgb[0][low] = (255.0 * u).astype(np.uint8)           # R
        rgb[1][low] = (255.0 * u).astype(np.uint8)           # G
        rgb[2][low] = (255.0 * (1.0 - u)).astype(np.uint8)   # B

    if np.any(high):
        u = ((t[high] - 0.5) / 0.5).astype(np.float32)
        rgb[0][high] = 255                                   # R
        rgb[1][high] = (255.0 * (1.0 - u)).astype(np.uint8)  # G
        rgb[2][high] = 0                                     # B

    return rgb


_TILE_WORKER_CONTEXT: dict[str, object] = {}


def _close_tile_worker_context() -> None:
    global _TILE_WORKER_CONTEXT
    src = _TILE_WORKER_CONTEXT.get("src")
    mask_src = _TILE_WORKER_CONTEXT.get("mask_src")
    if src is not None:
        src.close()
    if mask_src is not None:
        mask_src.close()
    _TILE_WORKER_CONTEXT = {}


def _init_tile_worker(
    input_tif: str,
    mask_tif: str,
    band_idx: Tuple[int, ...],
    tile_size: int,
    pixel_size: float,
    min_positive_ratio: float,
    max_nodata_ratio: float,
    normalize: bool,
    save_format: str,
    tile_prefix: str,
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: str,
    val_masks_dir: str,
) -> None:
    global _TILE_WORKER_CONTEXT
    _close_tile_worker_context()
    _TILE_WORKER_CONTEXT = {
        "src": rasterio.open(input_tif),
        "mask_src": rasterio.open(mask_tif),
        "band_idx": tuple(int(b) for b in band_idx),
        "tile_size": int(tile_size),
        "pixel_size": float(pixel_size),
        "min_positive_ratio": float(min_positive_ratio),
        "max_nodata_ratio": float(max_nodata_ratio),
        "normalize": bool(normalize),
        "save_format": str(save_format),
        "tile_prefix": str(tile_prefix),
        "train_images_dir": str(train_images_dir),
        "train_masks_dir": str(train_masks_dir),
        "val_images_dir": str(val_images_dir),
        "val_masks_dir": str(val_masks_dir),
    }


def _process_single_tile(task: Tuple[int, int, str]) -> dict:
    row_off, col_off, split_name = task
    ctx = _TILE_WORKER_CONTEXT
    src = ctx["src"]
    mask_src = ctx["mask_src"]
    tile_size = int(ctx["tile_size"])
    window = Window(col_off, row_off, tile_size, tile_size)

    mask_raw = mask_src.read(1, window=window).astype(np.float32)
    mask = np.where(np.isfinite(mask_raw) & (mask_raw > 0), 1, 0).astype(np.uint8)

    def read_band(band_i: int) -> Optional[np.ndarray]:
        if band_i <= 0:
            return None
        data = src.read(band_i, window=window)
        return data.astype(np.float32)

    band_idx = ctx["band_idx"]
    rgb = np.stack([read_band(int(band_idx[i])) for i in range(3)], axis=0)
    dsm = read_band(int(band_idx[3]))
    dtm = read_band(int(band_idx[4]))

    if dtm is None:
        return {
            "status": "skipped_nodata",
            "row_off": int(row_off),
            "col_off": int(col_off),
        }

    nodata_ratio = np.sum(~np.isfinite(dtm)) / dtm.size
    if nodata_ratio > float(ctx["max_nodata_ratio"]):
        return {
            "status": "skipped_nodata",
            "row_off": int(row_off),
            "col_off": int(col_off),
        }

    positive_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    positive_ratio = (positive_pixels / total_pixels) if total_pixels > 0 else 0.0

    if (
        float(ctx["min_positive_ratio"]) > 0.0
        and positive_ratio < float(ctx["min_positive_ratio"])
    ):
        return {
            "status": "skipped_empty",
            "row_off": int(row_off),
            "col_off": int(col_off),
        }

    try:
        svf, slrm = compute_derivatives_with_rvt(
            dtm,
            pixel_size=float(ctx["pixel_size"]),
            show_progress=False,
            log_steps=False,
        )
    except Exception as exc:
        return {
            "status": "skipped_nodata",
            "row_off": int(row_off),
            "col_off": int(col_off),
            "error": str(exc),
        }

    stacked = stack_channels(rgb, svf, slrm)

    if bool(ctx["normalize"]):
        stacked = robust_norm(stacked)

    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask = np.nan_to_num(mask, nan=0.0).astype(np.uint8)

    base_tile_name = f"tile_{row_off:05d}_{col_off:05d}"
    tile_prefix = str(ctx["tile_prefix"])
    tile_name = f"{tile_prefix}_{base_tile_name}" if tile_prefix else base_tile_name

    if split_name == "train":
        images_dir = str(ctx["train_images_dir"])
        masks_dir = str(ctx["train_masks_dir"])
    else:
        images_dir = str(ctx["val_images_dir"])
        masks_dir = str(ctx["val_masks_dir"])

    save_format = str(ctx["save_format"])
    if save_format == "npy":
        np.save(os.path.join(images_dir, f"{tile_name}.npy"), stacked)
        np.save(os.path.join(masks_dir, f"{tile_name}.npy"), mask)
    else:
        np.savez_compressed(
            os.path.join(images_dir, f"{tile_name}.npz"),
            image=stacked,
        )
        np.savez_compressed(
            os.path.join(masks_dir, f"{tile_name}.npz"),
            mask=mask,
        )

    return {
        "status": "ok",
        "split": str(split_name),
        "tile_name": tile_name,
        "row_off": int(row_off),
        "col_off": int(col_off),
        "positive_ratio": float(positive_ratio),
        "positive_pixels": int(positive_pixels),
        "total_pixels": int(total_pixels),
    }


def create_training_tiles(
    input_tif: Path,
    mask_tif: Path,
    output_dir: Path,
    tile_size: int = 256,
    overlap: int = 128,
    bands: str = "1,2,3,4,5",
    min_positive_ratio: float = 0.0,
    tile_label_min_positive_ratio: float = 0.0,
    max_nodata_ratio: float = 0.3,
    train_ratio: float = 0.8,
    normalize: bool = True,
    save_format: str = "npz",
    num_workers: int = 1,
    split_mode: str = "spatial",
    train_negative_keep_ratio: float = 1.0,
    train_negative_max: Optional[int] = None,
    tile_prefix: str = "",
    clean_output: bool = True,
) -> dict:
    """
    GeoTIFF'ten 5 kanallı eğitim tile'ları oluşturur (R, G, B, SVF, SLRM).
    
    Args:
        input_tif: Çok bantlı GeoTIFF (RGB + DSM + DTM)
        mask_tif: Ground truth maske dosyası (arkeolojik alanlar = 1)
        output_dir: Çıktı dizini
        tile_size: Tile boyutu (piksel)
        overlap: Örtüşme miktarı
        bands: Bant indeksleri "R,G,B,DSM,DTM" formatında
        min_positive_ratio: Dataset'e dahil etmek için minimum pozitif piksel oranı (0-1)
        tile_label_min_positive_ratio: Tile-level classification etiketi için minimum
            pozitif piksel oranı (0-1)
        max_nodata_ratio: Maksimum nodata oranı (0-1)
        train_ratio: Eğitim/doğrulama bölme oranı
        normalize: Tile'ları normalize et
        save_format: Kayıt formatı ("npy" veya "npz")
        num_workers: Paralel worker sayisi (1=sekansiyel)
        split_mode: Train/val bölme modu.
                   "spatial" (önerilen): mekansal sızıntıyı azaltmak için
                   sınırı kesen tile'ları atar.
                   "random": klasik rastgele bölme (daha yüksek leakage riski).
        train_negative_keep_ratio: Train setindeki tamamen negatif tile'ların tutulma oranı (0-1).
                                   0=hepsini at, 1=hepsini tut.
        train_negative_max: Train setinde tutulacak negatif tile sayısına üst sınır.
                            None ise sınır yok.
        tile_prefix: Kayit edilen tile dosya adlarina eklenecek on ek.
                     Bos birakilirsa tile_size, overlap, bands ve zamanla otomatik olusturulur.

    Returns:
        İstatistik sözlüğü
    """
    _validate_tile_generation_params(
        tile_size=tile_size,
        overlap=overlap,
        min_positive_ratio=min_positive_ratio,
        tile_label_min_positive_ratio=tile_label_min_positive_ratio,
        max_nodata_ratio=max_nodata_ratio,
        train_ratio=train_ratio,
        save_format=save_format,
        split_mode=split_mode,
        train_negative_keep_ratio=train_negative_keep_ratio,
        train_negative_max=train_negative_max,
        num_workers=num_workers,
    )

    output_dir = Path(output_dir)
    tile_prefix = str(tile_prefix).strip()
    if not tile_prefix:
        band_tokens = [token.strip() for token in str(bands).split(",") if token.strip()]
        band_sig = "-".join(band_tokens) if band_tokens else "na"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tile_prefix = f"t{int(tile_size)}_ov{int(overlap)}_b{band_sig}_{ts}"
    if any(ch in tile_prefix for ch in ("/", "\\")):
        raise ValueError("tile_prefix '/' veya '\\' iceremez.")

    num_workers = int(num_workers)

    if not bool(clean_output):
        _validate_append_compatibility(
            output_dir=output_dir,
            tile_size=int(tile_size),
            bands=str(bands),
            normalize=bool(normalize),
            save_format=str(save_format),
            expected_channels=len(MODEL_CHANNEL_NAMES),
        )

    # Dizin yapısı oluştur
    train_images_dir, train_masks_dir, val_images_dir, val_masks_dir = _prepare_output_dirs(
        output_dir,
        save_format=save_format,
        clean_output=bool(clean_output),
    )
    
    band_idx = [int(b) for b in bands.split(",")]
    
    stats = {
        "total_tiles": 0,
        "train_tiles": 0,
        "val_tiles": 0,
        "skipped_nodata": 0,
        "skipped_empty": 0,
        "worker_error_count": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
        "split_mode_requested": split_mode,
        "split_mode_effective": split_mode,
        "split_boundary_discarded": 0,
        "num_workers": int(num_workers),
        "train_negative_keep_ratio": float(train_negative_keep_ratio),
        "train_negative_max": None if train_negative_max is None else int(train_negative_max),
        "train_negative_before_filter": 0,
        "train_negative_after_filter": 0,
        "train_negative_removed_by_filter": 0,
        "train_positive_before_filter": 0,
        "train_positive_after_filter": 0,
    }

    tile_presence_csv_path = output_dir / "tile_presence_scores.csv"
    tile_labels_csv_path = output_dir / "tile_labels.csv"
    tile_presence_grid_path = output_dir / "tile_presence_grid.tif"
    tile_presence_grid_rgb_path = output_dir / "tile_presence_grid_rgb.tif"
    
    LOGGER.info("=" * 60)
    LOGGER.info("EĞİTİM VERİSİ OLUŞTURMA")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Girdi: {input_tif}")
    LOGGER.info(f"Maske: {mask_tif}")
    LOGGER.info(f"Çıktı: {output_dir}")
    LOGGER.info(f"Tile boyutu: {tile_size}x{tile_size}")
    LOGGER.info(f"Örtüşme: {overlap}")
    LOGGER.info(f"Bölme modu: {split_mode}")
    LOGGER.info(f"Bant sırası: {bands}")
    LOGGER.info(
        "Tile label eşiği (classification): %.4f",
        float(tile_label_min_positive_ratio),
    )
    LOGGER.info(f"Paralel worker: {num_workers}")
    LOGGER.info(f"Tile prefix: {tile_prefix}")
    LOGGER.info(f"Cikti temizleme: {'aktif' if clean_output else 'kapali (append modu)'}")
    LOGGER.info(
        "Train negatif filtre: keep_ratio=%.3f, max=%s",
        float(train_negative_keep_ratio),
        "None" if train_negative_max is None else int(train_negative_max),
    )
    LOGGER.info("=" * 60)
    
    with rasterio.open(input_tif) as src, rasterio.open(mask_tif) as mask_src:
        height, width = src.height, src.width
        pixel_size = float((abs(src.transform.a) + abs(src.transform.e)) / 2.0)
        source_transform = src.transform
        source_crs = src.crs
        
        LOGGER.info(f"Raster boyutu: {width}x{height} piksel")
        LOGGER.info(f"Piksel boyutu: {pixel_size:.2f} m")
        
        # Maske boyut kontrolü
        if (mask_src.height, mask_src.width) != (height, width):
            raise ValueError(
                f"Maske boyutu ({mask_src.width}x{mask_src.height}) "
                f"girdi boyutuyla ({width}x{height}) eşleşmiyor!"
            )
        
        stride = tile_size - overlap
        grid_rows = max(0, ((height - tile_size) // stride) + 1)
        grid_cols = max(0, ((width - tile_size) // stride) + 1)
        tile_presence_grid = (
            np.full((grid_rows, grid_cols), -1.0, dtype=np.float32)
            if grid_rows > 0 and grid_cols > 0
            else None
        )
        
        # Tüm geçerli pencereleri topla
        windows = []
        for row_off in range(0, height - tile_size + 1, stride):
            for col_off in range(0, width - tile_size + 1, stride):
                windows.append((row_off, col_off))
        
        total_windows = len(windows)
        LOGGER.info(f"Toplam potansiyel tile sayısı: {total_windows}")
        
        # Tum tile pencereleri train/val bolumunden once hazir.
        # Tek dengeleme mekanizmasi: tum tile pencereleri korunur,
        # negatif azaltma yalnizca train splitte uygulanir.
        all_windows = windows

        train_windows, val_windows, boundary_discarded, split_mode_effective = _split_windows_for_train_val(
            all_windows,
            train_ratio=train_ratio,
            tile_size=tile_size,
            raster_height=height,
            split_mode=split_mode,
            seed=42,
        )
        stats["split_mode_effective"] = split_mode_effective
        stats["split_boundary_discarded"] = int(boundary_discarded)

        if split_mode_effective == "random_fallback":
            LOGGER.warning(
                "Mekansal bölme train/val için yeterli tile üretemedi; random bölmeye geri dönüldü."
            )

        if len(train_windows) == 0:
            raise ValueError(
                "Train penceresi olusmadi. tile_size/overlap/train_ratio ayarlarini kontrol edin."
            )
        if len(val_windows) == 0:
            raise ValueError(
                "Val penceresi olusmadi. tile_size/overlap/train_ratio ayarlarini kontrol edin."
            )

        if train_negative_keep_ratio < 1.0 or train_negative_max is not None:
            train_windows, train_filter_stats = _filter_train_negative_windows(
                train_windows,
                mask_src=mask_src,
                tile_size=tile_size,
                keep_ratio=float(train_negative_keep_ratio),
                max_count=train_negative_max,
                seed=42,
            )
            stats.update(train_filter_stats)
            LOGGER.info(
                "Train negatif filtre sonucu: once=%d, sonra=%d, atilan=%d",
                stats["train_negative_before_filter"],
                stats["train_negative_after_filter"],
                stats["train_negative_removed_by_filter"],
            )
            if len(train_windows) == 0:
                raise ValueError(
                    "Train seti negatif filtre sonrasi bos kaldi. "
                    "train_negative_keep_ratio veya train_negative_max degerlerini artirin."
                )

        tile_tasks: List[Tuple[int, int, str]] = (
            [(row_off, col_off, "train") for row_off, col_off in train_windows]
            + [(row_off, col_off, "val") for row_off, col_off in val_windows]
        )

        if not tile_tasks:
            raise ValueError(
                "Islenecek tile bulunamadi. max_nodata/min_positive ve split ayarlarini kontrol edin."
            )

        LOGGER.info(
            "Train/Val planı: train=%d, val=%d, sınırda atılan=%d (%s)",
            len(train_windows),
            len(val_windows),
            boundary_discarded,
            split_mode_effective,
        )
        
        with open(tile_presence_csv_path, "w", newline="", encoding="utf-8") as tile_presence_fp, open(
            tile_labels_csv_path, "w", newline="", encoding="utf-8"
        ) as tile_labels_fp:
            tile_presence_writer = csv.writer(tile_presence_fp)
            tile_labels_writer = csv.writer(tile_labels_fp)
            tile_presence_writer.writerow(
                [
                    "tile_name",
                    "split",
                    "row_off",
                    "col_off",
                    "is_positive",
                    "positive_ratio",
                    "presence_probability",
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

            LOGGER.info("\nTile'lar işleniyor...")
            worker_error_count = 0
            worker_error_samples: List[str] = []

            def handle_tile_result(result: dict) -> None:
                nonlocal worker_error_count, worker_error_samples
                status = str(result.get("status", ""))

                if status == "skipped_nodata":
                    stats["skipped_nodata"] += 1
                    if "error" in result:
                        worker_error_count += 1
                        if len(worker_error_samples) < 5:
                            worker_error_samples.append(
                                "(%s, %s): %s"
                                % (
                                    result.get("row_off"),
                                    result.get("col_off"),
                                    result.get("error"),
                                )
                            )
                    return

                if status == "skipped_empty":
                    stats["skipped_empty"] += 1
                    return

                if status != "ok":
                    raise ValueError(f"Bilinmeyen tile sonucu: {status}")

                row_off = int(result["row_off"])
                col_off = int(result["col_off"])
                split_name = str(result["split"])
                tile_name = str(result["tile_name"])
                positive_ratio = float(result["positive_ratio"])
                positive_pixels = int(result["positive_pixels"])
                total_pixels = int(result["total_pixels"])

                if positive_ratio > 0:
                    stats["positive_tiles"] += 1
                else:
                    stats["negative_tiles"] += 1

                if split_name == "train":
                    stats["train_tiles"] += 1
                elif split_name == "val":
                    stats["val_tiles"] += 1
                else:
                    raise ValueError(f"Gecersiz split sonucu: {split_name}")

                if tile_presence_grid is not None:
                    grid_r = int(row_off // stride)
                    grid_c = int(col_off // stride)
                    if (
                        0 <= grid_r < tile_presence_grid.shape[0]
                        and 0 <= grid_c < tile_presence_grid.shape[1]
                    ):
                        tile_presence_grid[grid_r, grid_c] = np.float32(positive_ratio)

                tile_presence_writer.writerow(
                    [
                        tile_name,
                        split_name,
                        row_off,
                        col_off,
                        int(positive_ratio > 0.0),
                        f"{positive_ratio:.8f}",
                        f"{positive_ratio:.8f}",
                        positive_pixels,
                        total_pixels,
                    ]
                )

                tile_file_ext = str(save_format)
                image_relpath = Path(split_name) / "images" / f"{tile_name}.{tile_file_ext}"
                mask_relpath = Path(split_name) / "masks" / f"{tile_name}.{tile_file_ext}"
                tile_label = _positive_ratio_to_tile_label(
                    positive_ratio,
                    float(tile_label_min_positive_ratio),
                )
                tile_labels_writer.writerow(
                    [
                        tile_name,
                        split_name,
                        str(image_relpath).replace("\\", "/"),
                        str(mask_relpath).replace("\\", "/"),
                        row_off,
                        col_off,
                        tile_label,
                        f"{positive_ratio:.8f}",
                        positive_pixels,
                        total_pixels,
                        f"{float(tile_label_min_positive_ratio):.8f}",
                    ]
                )

                stats["total_tiles"] += 1

            worker_init_args = (
                str(input_tif),
                str(mask_tif),
                tuple(int(b) for b in band_idx),
                int(tile_size),
                float(pixel_size),
                float(min_positive_ratio),
                float(max_nodata_ratio),
                bool(normalize),
                str(save_format),
                str(tile_prefix),
                str(train_images_dir),
                str(train_masks_dir),
                str(val_images_dir),
                str(val_masks_dir),
            )

            if num_workers == 1:
                _init_tile_worker(*worker_init_args)
                try:
                    for result in tqdm(
                        map(_process_single_tile, tile_tasks),
                        total=len(tile_tasks),
                        desc="Isleniyor",
                    ):
                        handle_tile_result(result)
                finally:
                    _close_tile_worker_context()
            else:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_tile_worker,
                    initargs=worker_init_args,
                ) as executor:
                    futures = [
                        executor.submit(_process_single_tile, task)
                        for task in tile_tasks
                    ]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Isleniyor",
                    ):
                        handle_tile_result(future.result())

            if worker_error_count > 0:
                stats["worker_error_count"] = int(worker_error_count)
                LOGGER.warning(
                    "Turev hesaplama hatasi nedeniyle atlanan tile sayisi: %d",
                    worker_error_count,
                )
                for idx, sample in enumerate(worker_error_samples, start=1):
                    LOGGER.warning("  Ornek %d: %s", idx, sample)
    if stats["train_tiles"] == 0 or stats["val_tiles"] == 0:
        raise ValueError(
            "Uretilen veri train/val dengesini saglamiyor "
            f"(train={stats['train_tiles']}, val={stats['val_tiles']}). "
            "max_nodata/min_positive veya split ayarlarini gevsetin."
        )

    if tile_presence_grid is not None and tile_presence_grid.size > 0:
        grid_transform = source_transform * Affine.scale(stride, stride)
        with rasterio.open(
            tile_presence_grid_path,
            "w",
            driver="GTiff",
            height=int(tile_presence_grid.shape[0]),
            width=int(tile_presence_grid.shape[1]),
            count=1,
            dtype="float32",
            crs=source_crs,
            transform=grid_transform,
            nodata=-1.0,
            compress="lzw",
        ) as dst:
            dst.write(tile_presence_grid, 1)

        tile_presence_grid_rgb = _probability_to_rgb_grid(
            tile_presence_grid,
            nodata_value=-1.0,
        )
        with rasterio.open(
            tile_presence_grid_rgb_path,
            "w",
            driver="GTiff",
            height=int(tile_presence_grid.shape[0]),
            width=int(tile_presence_grid.shape[1]),
            count=3,
            dtype="uint8",
            crs=source_crs,
            transform=grid_transform,
            compress="lzw",
        ) as dst:
            dst.write(tile_presence_grid_rgb)

    # Metadata kaydet
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "input_file": str(input_tif),
        "mask_file": str(mask_tif),
        "tile_size": tile_size,
        "overlap": overlap,
        "bands": bands,
        "min_positive_ratio": float(min_positive_ratio),
        "tile_label_min_positive_ratio": float(tile_label_min_positive_ratio),
        "normalize": normalize,
        "save_format": save_format,
        "clean_output": bool(clean_output),
        "num_workers": int(num_workers),
        "tile_prefix": tile_prefix,
        "tile_presence_file": str(tile_presence_csv_path),
        "tile_labels_file": str(tile_labels_csv_path),
        "tile_presence_grid_file": str(tile_presence_grid_path),
        "tile_presence_grid_rgb_file": str(tile_presence_grid_rgb_path),
        "num_channels": len(MODEL_CHANNEL_NAMES),
        "channel_names": list(MODEL_CHANNEL_NAMES),
        "stats": stats,
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Sonuç raporu
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SONUÇ RAPORU")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Toplam tile sayısı: {stats['total_tiles']}")
    LOGGER.info(f"  → Eğitim: {stats['train_tiles']}")
    LOGGER.info(f"  → Doğrulama: {stats['val_tiles']}")
    LOGGER.info(f"Pozitif tile (arkeolojik içerik): {stats['positive_tiles']}")
    LOGGER.info(f"Negatif tile (arka plan): {stats['negative_tiles']}")
    LOGGER.info(f"Atlanan (nodata): {stats['skipped_nodata']}")
    LOGGER.info(f"Atlanan (boş): {stats['skipped_empty']}")
    
    if train_negative_keep_ratio < 1.0 or train_negative_max is not None:
        LOGGER.info("\nTrain Negatif Filtre Istatistikleri:")
        LOGGER.info(
            "  Train pozitif (once/sonra): %d / %d",
            stats.get("train_positive_before_filter", 0),
            stats.get("train_positive_after_filter", 0),
        )
        LOGGER.info(
            "  Train negatif (once/sonra): %d / %d",
            stats.get("train_negative_before_filter", 0),
            stats.get("train_negative_after_filter", 0),
        )
        LOGGER.info(
            "  Train'den atilan negatif: %d",
            stats.get("train_negative_removed_by_filter", 0),
        )
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"Çıktı dizini: {output_dir}")
    LOGGER.info(f"  → Eğitim görüntüleri: {train_images_dir}")
    LOGGER.info(f"  → Eğitim maskeleri: {train_masks_dir}")
    LOGGER.info(f"  → Doğrulama görüntüleri: {val_images_dir}")
    LOGGER.info(f"  → Doğrulama maskeleri: {val_masks_dir}")
    LOGGER.info(f"  → Tile varlık skorları (CSV): {tile_presence_csv_path}")
    LOGGER.info(f"  → Tile label manifesti (CSV): {tile_labels_csv_path}")
    LOGGER.info(f"  → Tile varlık grid rasterı: {tile_presence_grid_path}")
    LOGGER.info(f"  → Tile varlık renkli rasterı: {tile_presence_grid_rgb_path}")
    LOGGER.info(f"  → Metadata: {output_dir / 'metadata.json'}")
    LOGGER.info("=" * 60)
    
    return stats


def create_tiles_from_multiple_sources(
    sources: List[Tuple[Path, Path]],
    output_dir: Path,
    **kwargs,
) -> dict:
    """
    Birden fazla kaynak dosyadan tile oluşturur.
    
    Args:
        sources: (input_tif, mask_tif) tuple listesi
        output_dir: Çıktı dizini
        **kwargs: create_training_tiles parametreleri
    """
    all_stats = {
        "total_tiles": 0,
        "train_tiles": 0,
        "val_tiles": 0,
        "skipped_nodata": 0,
        "skipped_empty": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_sources: List[dict] = []
    
    for i, (input_tif, mask_tif) in enumerate(sources):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"KAYNAK {i+1}/{len(sources)}: {input_tif.name}")
        LOGGER.info(f"{'='*60}")
        
        # Her kaynak için ayrı alt dizin
        source_name = _sanitize_source_name(input_tif.stem)
        source_output = output_dir / f"source_{i:02d}_{source_name}"
        
        stats = create_training_tiles(
            input_tif=input_tif,
            mask_tif=mask_tif,
            output_dir=source_output,
            **kwargs,
        )

        summary_sources.append(
            {
                "source_index": int(i),
                "name": source_name,
                "input_file": str(input_tif),
                "mask_file": str(mask_tif),
                "output_dir": str(source_output),
                "stats": stats,
            }
        )
        
        # İstatistikleri birleştir
        for key in all_stats:
            all_stats[key] += stats.get(key, 0)
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("TÜM KAYNAKLAR İÇİN TOPLAM")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Toplam tile: {all_stats['total_tiles']}")
    LOGGER.info(f"Eğitim: {all_stats['train_tiles']}")
    LOGGER.info(f"Doğrulama: {all_stats['val_tiles']}")
    
    summary = {
        "created_at": datetime.now().isoformat(),
        "num_sources": len(summary_sources),
        "sources": summary_sources,
        "aggregate_stats": all_stats,
    }
    with open(output_dir / "multi_source_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return all_stats


def main():
    config_input = str(CONFIG.get("input", "")).strip()
    config_mask = str(CONFIG.get("mask", "")).strip()
    config_output = str(CONFIG.get("output", "training_data")).strip() or "training_data"

    config_train_negative_keep_ratio = float(CONFIG.get("train_negative_keep_ratio", 1.0))
    config_train_negative_max = CONFIG.get("train_negative_max", None)
    if config_train_negative_max is not None:
        config_train_negative_max = int(config_train_negative_max)
    config_tile_label_min_positive_ratio = float(
        CONFIG.get("tile_label_min_positive_ratio", 0.0)
    )
    config_num_workers = int(CONFIG.get("num_workers", max(1, (os.cpu_count() or 1) - 1)))
    config_tile_prefix = str(CONFIG.get("tile_prefix", "")).strip()
    config_append = bool(CONFIG.get("append", False))

    parser = argparse.ArgumentParser(
        description="5 kanalli arkeolojik tespit egitim verisi olusturma",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=config_input if config_input else None,
        help="Cok bantli GeoTIFF dosyasi (RGB + DSM + DTM)",
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        default=config_mask if config_mask else None,
        help="Ground truth maske dosyasi (arkeolojik alanlar = 1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=config_output,
        help="Cikti dizini",
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=int(CONFIG.get("tile_size", 256)),
        help="Tile boyutu (piksel)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(CONFIG.get("overlap", 128)),
        help="Tile ortusme miktari (piksel)",
    )
    parser.add_argument(
        "--bands", "-b",
        type=str,
        default=str(CONFIG.get("bands", "1,2,3,4,5")),
        help="Bant sirasi (R,G,B,DSM,DTM)",
    )
    parser.add_argument(
        "--min-positive",
        type=float,
        default=float(CONFIG.get("min_positive", 0.0)),
        help="Minimum pozitif piksel orani (0-1). 0=tum tile'lar dahil",
    )
    parser.add_argument(
        "--tile-label-min-positive-ratio",
        type=float,
        default=config_tile_label_min_positive_ratio,
        help=(
            "Tile-level classification etiketi icin minimum pozitif piksel orani (0-1). "
            "0 ise tile icinde en az bir pozitif piksel olmasi yeterlidir."
        ),
    )
    parser.add_argument(
        "--max-nodata",
        type=float,
        default=float(CONFIG.get("max_nodata", 0.3)),
        help="Maksimum nodata orani (0-1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=float(CONFIG.get("train_ratio", 0.8)),
        help="Egitim/toplam orani",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Normalizasyon yapma",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "npz"],
        default=str(CONFIG.get("format", "npz")),
        help="Kayit formati",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config_num_workers,
        help="Paralel worker sayisi. 1=sekansiyel.",
    )
    parser.add_argument(
        "--train-negative-keep-ratio",
        type=float,
        default=config_train_negative_keep_ratio,
        help=(
            "Train setindeki tamamen negatif tile'larin ne kadarinin tutulacagi (0-1). "
            "0=hepsini at, 1=hepsini tut."
        ),
    )
    parser.add_argument(
        "--train-negative-max",
        type=int,
        default=config_train_negative_max,
        help=(
            "Train setinde tutulacak negatif tile sayisina ust sinir. "
            "None ise sinir yok."
        ),
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["spatial", "random"],
        default=str(CONFIG.get("split_mode", "spatial")),
        help="Train/val bolme modu. spatial onerilir (mekansal sizintiyi azaltir).",
    )
    parser.add_argument(
        "--tile-prefix",
        type=str,
        default=config_tile_prefix,
        help=(
            "Tile dosya adlarina eklenecek on ek (ornek: s2). "
            "Bos ise otomatik: t{tile}_ov{overlap}_b{bands}_{timestamp}"
        ),
    )
    parser.add_argument(
        "--append",
        dest="append",
        action="store_true",
        help=(
            "Mevcut cikti dizinindeki tile dosyalarini silmeden yeni tile ekle. "
            "Varsayilan davranis eski tile dosyalarini temizlemektir."
        ),
    )
    parser.add_argument(
        "--no-append",
        dest="append",
        action="store_false",
        help="Mevcut cikti dizinindeki tile dosyalarini temizleyerek yeniden olustur.",
    )
    parser.set_defaults(no_normalize=not bool(CONFIG.get("normalize", True)))
    parser.set_defaults(append=config_append)

    args = parser.parse_args()

    if not args.input:
        parser.error("Girdi dosyasi icin CONFIG['input'] veya --input belirtin.")
    if not args.mask:
        parser.error("Maske dosyasi icin CONFIG['mask'] veya --mask belirtin.")

    input_path = Path(args.input)
    mask_path = Path(args.mask)

    if not input_path.exists():
        print(f"HATA: Girdi dosyasi bulunamadi: {input_path}")
        sys.exit(1)

    if not mask_path.exists():
        print(f"HATA: Maske dosyasi bulunamadi: {mask_path}")
        sys.exit(1)

    try:
        create_training_tiles(
            input_tif=input_path,
            mask_tif=mask_path,
            output_dir=Path(args.output),
            tile_size=args.tile_size,
            overlap=args.overlap,
            bands=args.bands,
            min_positive_ratio=args.min_positive,
            tile_label_min_positive_ratio=args.tile_label_min_positive_ratio,
            max_nodata_ratio=args.max_nodata,
            train_ratio=args.train_ratio,
            normalize=not args.no_normalize,
            save_format=args.format,
            num_workers=args.num_workers,
            split_mode=args.split_mode,
            train_negative_keep_ratio=args.train_negative_keep_ratio,
            train_negative_max=args.train_negative_max,
            tile_prefix=args.tile_prefix,
            clean_output=not args.append,
        )
    except Exception as e:
        LOGGER.error(f"Hata: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nOK - Egitim verisi olusturma tamamlandi!")
    print("  Simdi training.py ile model egitebilirsiniz:")
    print(f"  python training.py --data {args.output}")
    print(f"  python training.py --data {args.output} --task tile_classification")


if __name__ == "__main__":
    main()
