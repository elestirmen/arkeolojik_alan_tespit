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
    
    Veya interaktif mod:
    python egitim_verisi_olusturma.py

Gereksinimler:
    - Çok bantlı GeoTIFF (RGB + DSM + DTM)
    - Ground truth maske (arkeolojik alanlar = 1, arka plan = 0)
"""

from __future__ import annotations

import argparse
import logging
import os
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
        DEFAULTS,
        fill_nodata,
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
    save_format: str = "npy",
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
        
    Returns:
        İstatistik sözlüğü
    """
    output_dir = Path(output_dir)
    
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
    }
    
    LOGGER.info("=" * 60)
    LOGGER.info("EĞİTİM VERİSİ OLUŞTURMA")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Girdi: {input_tif}")
    LOGGER.info(f"Maske: {mask_tif}")
    LOGGER.info(f"Çıktı: {output_dir}")
    LOGGER.info(f"Tile boyutu: {tile_size}x{tile_size}")
    LOGGER.info(f"Örtüşme: {overlap}")
    LOGGER.info(f"Bant sırası: {bands}")
    LOGGER.info(f"TPI yarıçapları: {tpi_radii}")
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
        
        # Rastgele karıştır (train/val bölme için)
        np.random.seed(42)
        np.random.shuffle(windows)
        
        train_count = int(len(windows) * train_ratio)
        
        all_tiles = []
        
        LOGGER.info("\nTile'lar işleniyor...")
        for idx, (row_off, col_off) in enumerate(tqdm(windows, desc="İşleniyor")):
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
            is_train = idx < train_count
            
            if is_train:
                images_dir = train_images_dir
                masks_dir = train_masks_dir
                stats["train_tiles"] += 1
            else:
                images_dir = val_images_dir
                masks_dir = val_masks_dir
                stats["val_tiles"] += 1
            
            # Kaydet
            tile_name = f"tile_{row_off:05d}_{col_off:05d}"
            
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
    parser = argparse.ArgumentParser(
        description="12 kanallı arkeolojik tespit eğitim verisi oluşturma",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Çok bantlı GeoTIFF dosyası (RGB + DSM + DTM)",
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        help="Ground truth maske dosyası (arkeolojik alanlar = 1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="training_data",
        help="Çıktı dizini",
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=256,
        help="Tile boyutu (piksel)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Tile örtüşme miktarı (piksel)",
    )
    parser.add_argument(
        "--bands", "-b",
        type=str,
        default="1,2,3,4,5",
        help="Bant sırası (R,G,B,DSM,DTM)",
    )
    parser.add_argument(
        "--tpi-radii",
        type=str,
        default="5,15,30",
        help="TPI yarıçapları (virgülle ayrılmış)",
    )
    parser.add_argument(
        "--min-positive",
        type=float,
        default=0.0,
        help="Minimum pozitif piksel oranı (0-1). 0=tüm tile'lar dahil",
    )
    parser.add_argument(
        "--max-nodata",
        type=float,
        default=0.3,
        help="Maksimum nodata oranı (0-1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Eğitim/toplam oranı",
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
        default="npy",
        help="Kayıt formatı",
    )
    
    args = parser.parse_args()
    
    # İnteraktif mod (argüman verilmemişse)
    if args.input is None or args.mask is None:
        print("\n" + "=" * 60)
        print("ARKEOLOJİK ALAN TESPİTİ - EĞİTİM VERİSİ OLUŞTURMA")
        print("=" * 60)
        print("\nLütfen gerekli dosyaları girin:")
        
        args.input = input("Girdi GeoTIFF dosyası [kesif_alani.tif]: ").strip()
        if not args.input:
            args.input = "kesif_alani.tif"
        
        args.mask = input("Ground truth maske dosyası: ").strip()
        if not args.mask:
            print("HATA: Maske dosyası gerekli!")
            print("\nİpucu: Ground truth maske, arkeolojik alanların")
            print("       işaretlendiği tek bantlı bir GeoTIFF olmalıdır.")
            print("       Arkeolojik alan = 1, Arka plan = 0")
            sys.exit(1)
        
        args.output = input(f"Çıktı dizini [{args.output}]: ").strip() or args.output
        
        tile_input = input(f"Tile boyutu [{args.tile_size}]: ").strip()
        if tile_input:
            args.tile_size = int(tile_input)
    
    # TPI yarıçaplarını parse et
    tpi_radii = tuple(int(r) for r in args.tpi_radii.split(","))
    
    # Dosya kontrolü
    input_path = Path(args.input)
    mask_path = Path(args.mask)
    
    if not input_path.exists():
        print(f"HATA: Girdi dosyası bulunamadı: {input_path}")
        sys.exit(1)
    
    if not mask_path.exists():
        print(f"HATA: Maske dosyası bulunamadı: {mask_path}")
        sys.exit(1)
    
    # Tile oluştur
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
        )
    except Exception as e:
        LOGGER.error(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Eğitim verisi oluşturma tamamlandı!")
    print(f"  Şimdi training.py ile model eğitebilirsiniz:")
    print(f"  python training.py --data {args.output}")


if __name__ == "__main__":
    main()


