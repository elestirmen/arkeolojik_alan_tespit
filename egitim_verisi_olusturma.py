#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arkeolojik Alan Tespiti - 12 Kanallı Eğitim Verisi Oluşturma Scripti

Bu script, çok bantlı GeoTIFF dosyalarından ve karşılık gelen ground truth 
maskelerinden 12 kanallı eğitim tile'ları oluşturur.

Kanal Yapısı (12 kanal):
    [0-2]: RGB
    [3]: SVF (Sky-View Factor)
    [4]: Positive Openness
    [5]: Negative Openness
    [6]: LRM (Local Relief Model)
    [7]: Slope
    [8]: nDSM (normalize edilmiş DSM)
    [9]: Plan Curvature
    [10]: Profile Curvature
    [11]: TPI (Topographic Position Index)

Kullanım:
    python egitim_verisi_olusturma.py --input kesif_alani.tif --mask ground_truth.tif --output training_data
    python egitim_verisi_olusturma.py  # CONFIG bolumundeki varsayilanlarla calisir

Gereksinimler:
    - Çok bantlı GeoTIFF (RGB + DSM + DTM)
    - Ground truth maske (arkeolojik alanlar = 1, arka plan = 0)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import json
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Mevcut projedeki fonksiyonları import et
try:
    from archaeo_detect import (
        compute_derivatives_with_rvt,
        compute_ndsm,
        compute_curvatures,
        compute_tpi_multiscale,
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
    # Girdi cok bantli raster (RGB + DSM + DTM)
    "input": "on_veri/karlik_dag_rgb_dtm_dsm_5band.tif",
    # Ground-truth maske (arkeolojik alan=1, arka plan=0)
    "mask": "ground_truth.tif",
    # Cikti dizini
    "output": "training_data",
    # Tile ayarlari
    "tile_size": 256,
    "overlap": 64,
    # Bant sirasi: R,G,B,DSM,DTM
    "bands": "1,2,3,4,5",
    # TPI yaricaplari
    "tpi_radii": (5, 15, 30),
    # Filtreleme/split
    "min_positive": 0.0,
    "max_nodata": 0.3,
    "train_ratio": 0.8,
    "split_mode": "spatial",  # spatial | random
    # On-isleme/kayit
    "normalize": True,
    "format": "npz",  # npy | npz
    "balance_ratio": None,  # Ornek: 0.4
    "tile_prefix": "",  # Bos ise otomatik: t{tile}_ov{overlap}_b{bands}_{timestamp}
}
# ===============================================


def _validate_tile_generation_params(
    tile_size: int,
    overlap: int,
    min_positive_ratio: float,
    max_nodata_ratio: float,
    train_ratio: float,
    save_format: str,
    balance_ratio: Optional[float],
    split_mode: str,
) -> None:
    errors: List[str] = []

    if tile_size <= 0:
        errors.append(f"tile_size pozitif olmalı, verilen: {tile_size}")
    if overlap < 0:
        errors.append(f"overlap negatif olamaz, verilen: {overlap}")
    if overlap >= tile_size:
        errors.append(f"overlap ({overlap}) tile_size'dan ({tile_size}) küçük olmalı")

    if not 0.0 <= min_positive_ratio <= 1.0:
        errors.append(
            f"min_positive_ratio 0-1 arasında olmalı, verilen: {min_positive_ratio}"
        )
    if not 0.0 <= max_nodata_ratio <= 1.0:
        errors.append(
            f"max_nodata_ratio 0-1 arasında olmalı, verilen: {max_nodata_ratio}"
        )
    if not 0.0 < train_ratio < 1.0:
        errors.append(f"train_ratio 0 ile 1 arasında olmalı, verilen: {train_ratio}")

    if save_format not in {"npy", "npz"}:
        errors.append(f"save_format geçersiz: {save_format} (npy/npz)")

    if balance_ratio is not None and not 0.0 < balance_ratio < 1.0:
        errors.append(
            f"balance_ratio None veya (0,1) aralığında olmalı, verilen: {balance_ratio}"
        )

    if split_mode not in {"spatial", "random"}:
        errors.append(f"split_mode geçersiz: {split_mode} (spatial/random)")

    if errors:
        raise ValueError("Parametre doğrulama hataları:\n- " + "\n- ".join(errors))


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


def create_training_tiles(
    input_tif: Path,
    mask_tif: Path,
    output_dir: Path,
    tile_size: int = 256,
    overlap: int = 64,
    bands: str = "1,2,3,4,5",
    tpi_radii: Tuple[int, ...] = (5, 15, 30),
    min_positive_ratio: float = 0.0,
    max_nodata_ratio: float = 0.3,
    train_ratio: float = 0.8,
    normalize: bool = True,
    save_format: str = "npz",
    balance_ratio: Optional[float] = 0.3,
    split_mode: str = "spatial",
    tile_prefix: str = "",
) -> dict:
    """
    GeoTIFF'ten 12 kanallı eğitim tile'ları oluşturur.
    
    Args:
        input_tif: Çok bantlı GeoTIFF (RGB + DSM + DTM)
        mask_tif: Ground truth maske dosyası (arkeolojik alanlar = 1)
        output_dir: Çıktı dizini
        tile_size: Tile boyutu (piksel)
        overlap: Örtüşme miktarı
        bands: Bant indeksleri "R,G,B,DSM,DTM" formatında
        tpi_radii: TPI yarıçapları
        min_positive_ratio: Minimum pozitif piksel oranı (0-1)
        max_nodata_ratio: Maksimum nodata oranı (0-1)
        train_ratio: Eğitim/doğrulama bölme oranı
        normalize: Tile'ları normalize et
        save_format: Kayıt formatı ("npy" veya "npz")
        balance_ratio: Pozitif/negatif dengeleme oranı (0-1). 
                       None = dengeleme yok, tüm tile'lar kullanılır.
                       Örn: 0.4 = %40 pozitif, %60 negatif.
                       Önerilen: 0.4 (hafif dengeli)
        split_mode: Train/val bölme modu.
                   "spatial" (önerilen): mekansal sızıntıyı azaltmak için
                   sınırı kesen tile'ları atar.
                   "random": klasik rastgele bölme (daha yüksek leakage riski).
        tile_prefix: Kayit edilen tile dosya adlarina eklenecek on ek.
                     Bos birakilirsa tile_size, overlap, bands ve zamanla otomatik olusturulur.

    Returns:
        İstatistik sözlüğü
    """
    _validate_tile_generation_params(
        tile_size=tile_size,
        overlap=overlap,
        min_positive_ratio=min_positive_ratio,
        max_nodata_ratio=max_nodata_ratio,
        train_ratio=train_ratio,
        save_format=save_format,
        balance_ratio=balance_ratio,
        split_mode=split_mode,
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
    
    # Dizin yapısı oluştur
    train_images_dir = output_dir / "train" / "images"
    train_masks_dir = output_dir / "train" / "masks"
    val_images_dir = output_dir / "val" / "images"
    val_masks_dir = output_dir / "val" / "masks"
    
    for d in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    band_idx = [int(b) for b in bands.split(",")]
    
    stats = {
        "total_tiles": 0,
        "train_tiles": 0,
        "val_tiles": 0,
        "skipped_nodata": 0,
        "skipped_empty": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
        "balanced_selection": balance_ratio is not None,
        "original_positive_count": 0,
        "original_negative_count": 0,
        "selected_positive_count": 0,
        "selected_negative_count": 0,
        "discarded_negative_count": 0,
        "ignored_low_positive_count": 0,
        "split_mode_requested": split_mode,
        "split_mode_effective": split_mode,
        "split_boundary_discarded": 0,
    }
    
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
    LOGGER.info(f"TPI yarıçapları: {tpi_radii}")
    LOGGER.info(f"Tile prefix: {tile_prefix}")
    if balance_ratio is not None:
        LOGGER.info(f"Dengeli seçim: Aktif (hedef: %{balance_ratio*100:.0f} pozitif, %{(1-balance_ratio)*100:.0f} negatif)")
    else:
        LOGGER.info("Dengeli seçim: Kapalı (tüm tile'lar kullanılacak)")
    LOGGER.info("=" * 60)
    
    with rasterio.open(input_tif) as src, rasterio.open(mask_tif) as mask_src:
        height, width = src.height, src.width
        pixel_size = float((abs(src.transform.a) + abs(src.transform.e)) / 2.0)
        
        LOGGER.info(f"Raster boyutu: {width}x{height} piksel")
        LOGGER.info(f"Piksel boyutu: {pixel_size:.2f} m")
        
        # Maske boyut kontrolü
        if (mask_src.height, mask_src.width) != (height, width):
            raise ValueError(
                f"Maske boyutu ({mask_src.width}x{mask_src.height}) "
                f"girdi boyutuyla ({width}x{height}) eşleşmiyor!"
            )
        
        stride = tile_size - overlap
        
        # Tüm geçerli pencereleri topla
        windows = []
        for row_off in range(0, height - tile_size + 1, stride):
            for col_off in range(0, width - tile_size + 1, stride):
                windows.append((row_off, col_off))
        
        total_windows = len(windows)
        LOGGER.info(f"Toplam potansiyel tile sayısı: {total_windows}")
        
        # Dengeli seçim için önce pozitif/negatif tile'ları ayır
        if balance_ratio is not None:
            LOGGER.info("\nTile'lar pozitif/negatif olarak analiz ediliyor...")
            positive_windows = []
            negative_windows = []
            ignored_low_positive = 0
            
            for row_off, col_off in tqdm(windows, desc="Analiz"):
                window = Window(col_off, row_off, tile_size, tile_size)
                
                # Sadece maske kontrolü (hızlı)
                try:
                    mask = mask_src.read(1, window=window).astype(np.float32)
                    positive_ratio = np.sum(mask > 0) / mask.size
                    
                    if _is_positive_for_balance(
                        positive_ratio=positive_ratio,
                        min_positive_ratio=min_positive_ratio,
                    ):
                        positive_windows.append((row_off, col_off))
                    elif positive_ratio > 0:
                        # Contains labels but does not meet positivity threshold for training tiles.
                        ignored_low_positive += 1
                    else:
                        negative_windows.append((row_off, col_off))
                except Exception:
                    continue
            
            stats["original_positive_count"] = len(positive_windows)
            stats["original_negative_count"] = len(negative_windows)
            stats["ignored_low_positive_count"] = int(ignored_low_positive)
            
            LOGGER.info(f"Pozitif tile (içerik var): {len(positive_windows)}")
            LOGGER.info(f"Negatif tile (sadece arka plan): {len(negative_windows)}")
            if ignored_low_positive > 0:
                LOGGER.info(
                    "Pozitif içerikli ama min_positive_ratio altında kalan tile: %d",
                    ignored_low_positive,
                )
            
            # Negatif tile'ları örnekle
            if len(positive_windows) > 0:
                target_negative_count = int(len(positive_windows) * (1 - balance_ratio) / balance_ratio)
                target_negative_count = min(target_negative_count, len(negative_windows))
                
                np.random.seed(42)
                if target_negative_count < len(negative_windows):
                    sampled_indices = np.random.choice(
                        len(negative_windows),
                        size=target_negative_count,
                        replace=False
                    )
                    selected_negative = [negative_windows[i] for i in sampled_indices]
                    stats["discarded_negative_count"] = len(negative_windows) - target_negative_count
                else:
                    selected_negative = negative_windows
                    stats["discarded_negative_count"] = 0
                
                stats["selected_positive_count"] = len(positive_windows)
                stats["selected_negative_count"] = len(selected_negative)
                
                # Birleştir ve karıştır
                all_windows = positive_windows + selected_negative
                np.random.seed(42)
                np.random.shuffle(all_windows)
                
                LOGGER.info("\nDengeli seçim sonucu:")
                LOGGER.info(f"  Seçilen pozitif: {len(positive_windows)}")
                LOGGER.info(f"  Seçilen negatif: {len(selected_negative)}")
                LOGGER.info(f"  Atılan negatif: {stats['discarded_negative_count']}")
                LOGGER.info(f"  Toplam seçilen: {len(all_windows)}")
                if len(all_windows) > 0:
                    actual_ratio = len(positive_windows) / len(all_windows)
                    LOGGER.info(f"  Gerçek pozitif oranı: %{actual_ratio*100:.1f}")
            else:
                LOGGER.warning("Pozitif tile bulunamadı! Tüm tile'lar kullanılacak.")
                all_windows = windows
        else:
            # Dengeli seçim yok, tüm tile'ları kullan
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

        all_windows = train_windows + val_windows
        train_window_set = set(train_windows)

        LOGGER.info(
            "Train/Val planı: train=%d, val=%d, sınırda atılan=%d (%s)",
            len(train_windows),
            len(val_windows),
            boundary_discarded,
            split_mode_effective,
        )
        
        LOGGER.info("\nTile'lar işleniyor...")
        for row_off, col_off in tqdm(all_windows, desc="İşleniyor"):
            window = Window(col_off, row_off, tile_size, tile_size)
            
            # Maske oku
            mask = mask_src.read(1, window=window).astype(np.float32)
            
            # Bantları oku
            def read_band(band_i: int) -> Optional[np.ndarray]:
                if band_i <= 0:
                    return None
                data = src.read(band_i, window=window)
                return data.astype(np.float32)
            
            rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
            dsm = read_band(band_idx[3])
            dtm = read_band(band_idx[4])
            
            # Nodata kontrolü
            if dtm is None:
                stats["skipped_nodata"] += 1
                continue
            
            nodata_ratio = np.sum(~np.isfinite(dtm)) / dtm.size
            if nodata_ratio > max_nodata_ratio:
                stats["skipped_nodata"] += 1
                continue
            
            # Pozitif piksel oranı kontrolü
            positive_ratio = np.sum(mask > 0) / mask.size
            
            # Boş tile kontrolü (min_positive_ratio > 0 ise)
            if min_positive_ratio > 0 and positive_ratio < min_positive_ratio:
                stats["skipped_empty"] += 1
                continue
            
            # Türevleri hesapla
            try:
                ndsm = compute_ndsm(dsm, dtm)
                svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
                    dtm, pixel_size=pixel_size
                )
                plan_curv, profile_curv = compute_curvatures(dtm, pixel_size=pixel_size)
                tpi = compute_tpi_multiscale(dtm, radii=tpi_radii)
            except Exception as e:
                LOGGER.warning(f"Türev hesaplama hatası ({row_off}, {col_off}): {e}")
                stats["skipped_nodata"] += 1
                continue
            
            # 12 kanallı tensor oluştur
            stacked = stack_channels(
                rgb=rgb,
                svf=svf,
                pos_open=pos_open,
                neg_open=neg_open,
                lrm=lrm,
                slope=slope,
                ndsm=ndsm,
                plan_curv=plan_curv,
                profile_curv=profile_curv,
                tpi=tpi,
            )
            
            # Normalize et
            if normalize:
                stacked = robust_norm(stacked)
            
            # NaN temizle
            stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.nan_to_num(mask, nan=0.0)
            
            # İstatistikler
            if positive_ratio > 0:
                stats["positive_tiles"] += 1
            else:
                stats["negative_tiles"] += 1
            
            # Train/Val ayrımı
            is_train = (row_off, col_off) in train_window_set
            
            if is_train:
                images_dir = train_images_dir
                masks_dir = train_masks_dir
                stats["train_tiles"] += 1
            else:
                images_dir = val_images_dir
                masks_dir = val_masks_dir
                stats["val_tiles"] += 1
            
            # Kaydet
            base_tile_name = f"tile_{row_off:05d}_{col_off:05d}"
            tile_name = f"{tile_prefix}_{base_tile_name}" if tile_prefix else base_tile_name
            
            if save_format == "npy":
                np.save(images_dir / f"{tile_name}.npy", stacked.astype(np.float32))
                np.save(masks_dir / f"{tile_name}.npy", mask.astype(np.uint8))
            else:  # npz
                np.savez_compressed(
                    images_dir / f"{tile_name}.npz",
                    image=stacked.astype(np.float32),
                )
                np.savez_compressed(
                    masks_dir / f"{tile_name}.npz",
                    mask=mask.astype(np.uint8),
                )
            
            stats["total_tiles"] += 1
    
    # Metadata kaydet
    metadata = {
        "created_at": datetime.now().isoformat(),
        "input_file": str(input_tif),
        "mask_file": str(mask_tif),
        "tile_size": tile_size,
        "overlap": overlap,
        "bands": bands,
        "tpi_radii": list(tpi_radii),
        "normalize": normalize,
        "tile_prefix": tile_prefix,
        "num_channels": 12,
        "channel_names": [
            "R", "G", "B", "SVF", "Pos_Openness", "Neg_Openness",
            "LRM", "Slope", "nDSM", "Plan_Curvature", "Profile_Curvature", "TPI"
        ],
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
    
    if stats.get("balanced_selection", False):
        LOGGER.info("\nDengeli Seçim İstatistikleri:")
        LOGGER.info(f"  Orijinal pozitif: {stats.get('original_positive_count', 0)}")
        LOGGER.info(f"  Orijinal negatif: {stats.get('original_negative_count', 0)}")
        LOGGER.info(f"  Seçilen pozitif: {stats.get('selected_positive_count', 0)}")
        LOGGER.info(f"  Seçilen negatif: {stats.get('selected_negative_count', 0)}")
        LOGGER.info(f"  Atılan negatif: {stats.get('discarded_negative_count', 0)}")
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"Çıktı dizini: {output_dir}")
    LOGGER.info(f"  → Eğitim görüntüleri: {train_images_dir}")
    LOGGER.info(f"  → Eğitim maskeleri: {train_masks_dir}")
    LOGGER.info(f"  → Doğrulama görüntüleri: {val_images_dir}")
    LOGGER.info(f"  → Doğrulama maskeleri: {val_masks_dir}")
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
    
    for i, (input_tif, mask_tif) in enumerate(sources):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"KAYNAK {i+1}/{len(sources)}: {input_tif.name}")
        LOGGER.info(f"{'='*60}")
        
        # Her kaynak için ayrı alt dizin
        source_output = output_dir / f"source_{i:02d}_{input_tif.stem}"
        
        stats = create_training_tiles(
            input_tif=input_tif,
            mask_tif=mask_tif,
            output_dir=source_output,
            **kwargs,
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
    
    return all_stats


def main():
    config_input = str(CONFIG.get("input", "")).strip()
    config_mask = str(CONFIG.get("mask", "")).strip()
    config_output = str(CONFIG.get("output", "training_data")).strip() or "training_data"
    config_tpi_radii = CONFIG.get("tpi_radii", (5, 15, 30))
    if isinstance(config_tpi_radii, (list, tuple)):
        tpi_default = ",".join(str(int(r)) for r in config_tpi_radii)
    else:
        tpi_default = str(config_tpi_radii).strip() or "5,15,30"

    config_balance_ratio = CONFIG.get("balance_ratio", None)
    if config_balance_ratio is not None:
        config_balance_ratio = float(config_balance_ratio)
    config_tile_prefix = str(CONFIG.get("tile_prefix", "")).strip()

    parser = argparse.ArgumentParser(
        description="12 kanalli arkeolojik tespit egitim verisi olusturma",
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
        default=int(CONFIG.get("overlap", 64)),
        help="Tile ortusme miktari (piksel)",
    )
    parser.add_argument(
        "--bands", "-b",
        type=str,
        default=str(CONFIG.get("bands", "1,2,3,4,5")),
        help="Bant sirasi (R,G,B,DSM,DTM)",
    )
    parser.add_argument(
        "--tpi-radii",
        type=str,
        default=tpi_default,
        help="TPI yaricaplari (virgulle ayrilmis)",
    )
    parser.add_argument(
        "--min-positive",
        type=float,
        default=float(CONFIG.get("min_positive", 0.0)),
        help="Minimum pozitif piksel orani (0-1). 0=tum tile'lar dahil",
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
        "--balance-ratio",
        type=float,
        default=config_balance_ratio,
        help=(
            "Pozitif/negatif dengeleme orani (0-1). Ornek: 0.4 = %%40 pozitif, %%60 negatif. "
            "None = dengeleme yok. Onerilen: 0.4 (hafif dengeli)"
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
    parser.set_defaults(no_normalize=not bool(CONFIG.get("normalize", True)))

    args = parser.parse_args()

    if not args.input:
        parser.error("Girdi dosyasi icin CONFIG['input'] veya --input belirtin.")
    if not args.mask:
        parser.error("Maske dosyasi icin CONFIG['mask'] veya --mask belirtin.")

    try:
        tpi_radii = tuple(
            int(r.strip()) for r in str(args.tpi_radii).split(",") if r.strip()
        )
    except ValueError:
        parser.error(f"Gecersiz --tpi-radii degeri: {args.tpi_radii}")

    if not tpi_radii:
        parser.error("TPI yaricaplari bos olamaz. Ornek: 5,15,30")

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
            tpi_radii=tpi_radii,
            min_positive_ratio=args.min_positive,
            max_nodata_ratio=args.max_nodata,
            train_ratio=args.train_ratio,
            normalize=not args.no_normalize,
            save_format=args.format,
            balance_ratio=args.balance_ratio,
            split_mode=args.split_mode,
            tile_prefix=args.tile_prefix,
        )
    except Exception as e:
        LOGGER.error(f"Hata: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nOK - Egitim verisi olusturma tamamlandi!")
    print("  Simdi training.py ile model egitebilirsiniz:")
    print(f"  python training.py --data {args.output}")


if __name__ == "__main__":
    main()

