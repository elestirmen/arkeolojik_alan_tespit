#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qt tabanli GeoTIFF etiketleme araci (PySide6 / PyQt6).

Ozellikler:
- Sol fare: rectangle cizim
- Sag fare: pan
- Tekerlek: zoom
- Draw / Erase modlari
- Secilen / secilmeyen maske degerleri ayarlanabilir
- Undo (Ctrl+Z), clear, reset, fit
- Save / Save As (GeoTIFF mask)
- Sol panel katman yonetimi (gorunurluk, sira, saydamlik)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform

QT_BACKEND = ""
_pyside_import_error: Optional[Exception] = None

if sys.platform == "win32":
    # Conda ICU DLL'leri, Qt'nin bekledigi sembollerle uyusmayabiliyor.
    # System32 ICU'yu once yukleyerek QtCore import hatasini engelliyoruz.
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for dll_name in ("icuuc.dll", "icuin.dll", "icudt.dll"):
        dll_path = system32 / dll_name
        if dll_path.exists():
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError:
                pass

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import (
        QAction, QActionGroup, QBrush, QColor, QCursor, QDragEnterEvent, QDropEvent,
        QImage, QKeySequence, QPainter, QPen, QPixmap, QTransform,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QProgressDialog,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    QT_BACKEND = "PySide6"
except ImportError as exc:
    _pyside_import_error = exc
    try:
        from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
        from PyQt6.QtGui import (
            QAction, QActionGroup, QBrush, QColor, QCursor, QDragEnterEvent, QDropEvent,
            QImage, QKeySequence, QPainter, QPen, QPixmap, QTransform,
        )
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGraphicsPixmapItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsView,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QProgressDialog,
            QProgressBar,
            QPushButton,
            QSlider,
            QSpinBox,
            QStatusBar,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )
        QT_BACKEND = "PyQt6"
    except ImportError as exc:
        raise ImportError(
            "PySide6/PyQt6 import edilemedi.\n"
            f"PySide6 hatasi: {_pyside_import_error}\n"
            f"PyQt6 hatasi: {exc}\n"
            "Kurulum: pip install PySide6 (veya pip install PyQt6)"
        ) from exc

# ==================== CONFIG ====================
# Bu bolum, aracin duzenlenebilir varsayilanlarini tek yerde toplar.
# Amac: davranisi kodun icine dagitmak yerine burada merkezi ve okunur tutmak.
# Buradaki degerleri degistirince hem arayuz varsayilanlari hem de bazi CLI
# varsayilanlari birlikte etkilenir.
CONFIG: dict[str, object] = {
    # app_title:
    # Uygulama penceresinin basliginda gorunen ad.
    # Windows gorev cubugunda ve bilgi/uyari pencerelerinde de bu metin kullanilir.
    "app_title": "Ground Truth Kare Etiketleme (Qt)",
    # overlay_alpha:
    # Maske katmaninin goruntu ustundeki saydamlik seviyesi.
    # 0 tamamen gorunmez, 255 ise maskeyi neredeyse tamamen opak yapar.
    # 80-120 araligi genelde alttaki goruntuyu gormeye devam ederken maskeyi de belirgin tutar.
    "overlay_alpha": 96,
    # layer_key_base:
    # Ana raster katmani icin dahili kimlik.
    # Katman listesi ve detay gorunumu icinde "ana goruntu"yu bulmak icin kullanilir.
    "layer_key_base": "__base__",
    # layer_key_mask:
    # Maske katmani icin dahili kimlik.
    # Kod icinde secim overlay'ini diger ek raster katmanlardan ayirmaya yarar.
    "layer_key_mask": "__mask__",
    # annotations_layer_name:
    # GeoPackage icine polygon anotasyonlar yazildiginda kullanilan katman adi.
    # Disarida QGIS vb. ile acarken bu isim gorunur.
    "annotations_layer_name": "annotations",
    # dataset_export_script:
    # "Tile Dataset Export" aksiyonunda arka planda calistirilan script dosyasi.
    # Farkli bir export scripti varsa burada degistirilebilir.
    "dataset_export_script": "prepare_tile_classification_dataset.py",
    # dataset_default_tile_size:
    # Export dialogunda ilk acilista gelen tile boyutu.
    # Buyuk deger daha fazla baglam verir ama secilen bolgenin disina tasan pozitif tile sayisini artirabilir.
    "dataset_default_tile_size": 256,
    # dataset_default_overlap:
    # Ard ardina gelen tile'lar arasindaki ortusme miktari.
    # Overlap arttikca daha yogun sampling olur; veri miktari ve disk kullanimi da artar.
    "dataset_default_overlap": 128,
    # dataset_default_sampling_mode:
    # Export dialogunda varsayilan uretim stratejisi.
    # "selected_regions": secili alanlari merkez alip pozitifleri uretir, negatifleri ayrica ornekler.
    # "full_grid": tum rasteri kayan pencere gibi tarar.
    "dataset_default_sampling_mode": "selected_regions",
    # dataset_default_feature_mode:
    # Export edilen tile tensorunun semasi.
    # "topo5" -> R,G,B,SVF,SLRM
    # "rgb3"  -> R,G,B
    "dataset_default_feature_mode": "topo5",
    
    # dataset_default_positive_ratio:
    # Bir tile'in Positive etiket alabilmesi icin maskeyle ortusmesi gereken minimum oran.
    # Ornek: 0.02 -> tile alaninin yalnizca %2'si seciliyse bile Positive olabilir.
    # "Yanlis yerden tile gelmis gibi" hissi olursa ilk bakilacak ayarlardan biri budur.
    "dataset_default_positive_ratio": 0.1,
    
    # dataset_default_valid_ratio:
    # Girdi rasterinde gecerli veri iceren piksellerin minimum orani.
    # Kenarlarda bos/nodata alan coksa bu esik altinda kalan tile'lar tamamen elenir.
    "dataset_default_valid_ratio": 0.9,

    # dataset_default_train_negative_keep:
    # Sadece full_grid modunda kullanilir.
    # Uretilen train negatiflerinin ne kadarinin tutulacagini belirler; veri dengesini ve toplam dataset boyutunu etkiler.
    "dataset_default_train_negative_keep": 0.35,

    # dataset_default_neg_to_pos_ratio:
    # selected_regions modunda hedeflenen negatif / pozitif tile orani.
    # 1.0 -> pozitif sayisi kadar negatif secmeye calisir; 2.0 -> iki kati negatif hedefler.
    "dataset_default_neg_to_pos_ratio": 1.0,
    # dataset_default_format:
    # Tile'larin diske hangi formatta kaydedilecegi.
    # "npz" daha kucuk dosya uretir, "npy" ise sikistirmasiz oldugu icin bazi sistemlerde daha hizli olabilir.
    "dataset_default_format": "npz",
    # dataset_default_num_workers:
    # Export sirasinda ayni anda kullanilacak is parcacigi / process sayisi.
    # Yuksek deger CPU ve disk yukunu artirir; yavas disklerde her zaman daha hizli olmayabilir.
    "dataset_default_num_workers": max(1, min(8, (os.cpu_count() or 1) // 2)),
    
    # dataset_default_derivative_cache_mode:
    # Turev raster cache davranisinin varsayilan modu.
    # "auto" genelde en guvenli secenektir; veri boyutuna gore uygun strateji secmeye calisir.
    "dataset_default_derivative_cache_mode": "auto",
    # dataset_default_derivative_cache_dir:
    # Turev cache'in yazilacagi klasor.
    # Bos birakilirsa ilgili girdi rasterinin yaninda "cache/" klasoru kullanilir.
    "dataset_default_derivative_cache_dir": "",
    # dataset_default_overwrite:
    # Cikti klasoru doluysa nasil davranilacaginin varsayilani.
    # True ise eski klasor temizlenip yeniden yazilir; False ise kullanicidan bos klasor beklenir.
    "dataset_default_overwrite": True,
    # default_window_width:
    # Uygulama ilk acildiginda ana pencerenin genisligi.
    # Sadece ilk boyutu belirler; kullanici sonradan pencereyi istedigi gibi buyutup kucultebilir.
    "default_window_width": 1500,
    # default_window_height:
    # Uygulama ilk acildiginda ana pencerenin yuksekligi.
    "default_window_height": 950,
    # default_preview_max_size:
    # Onizleme rasterinin maksimum kenar boyu.
    # 0 verilirse preview icin downsample yapilmaz; buyuk rasterlerde bellek ve acilis suresi ciddi artabilir.
    "default_preview_max_size": 4096,
    # default_preview_bands_raw:
    # Ilk acilista goruntuleme icin kullanilacak bant kombinasyonu.
    # Ornek "1,2,3" -> RGB. Tek bantli dosyalarda CLI ile farkli kombinasyon verilebilir.
    "default_preview_bands_raw": "1,2,3",
    # default_model_bands_raw:
    # Tile export ve model tarafinda beklenen bant sirasi.
    # topo5 icin format: R,G,B,DSM,DTM
    # rgb3 icin format: R,G,B
    "default_model_bands_raw": "1,2,3,4,5",
    "default_model_bands_rgb_raw": "1,2,3",
    # default_positive_value:
    # Maske rasterine secili alan yazilirken kullanilan piksel degeri.
    # Genelde 1 yeterlidir; baska sistemlerle uyum gerekiyorsa degistirilebilir.
    "default_positive_value": 1,
    # default_negative_value:
    # Maske rasterinde secilmemis / arkaplan alanlarin degeri.
    # Bu deger ayni zamanda GeoTIFF nodata olarak da yazildigi icin export akisini etkiler.
    "default_negative_value": 0,
    # default_square_mode:
    # Cizim yaparken dikdortgenin zorunlu kare olup olmayacaginin varsayilani.
    # True ise baslangictan itibaren kare kilidi acik gelir.
    "default_square_mode": False,
    # detail_refresh_delay_ms:
    # Kullanici zoom/pan yaptiktan sonra detay gorunumu olusturmadan once beklenecek sure.
    # Dusuk deger daha canli hissettirir, ama cok dusuk olursa gereksiz yeniden hesaplama artar.
    "detail_refresh_delay_ms": 90,
    # detail_min_zoom:
    # Detay gorunumune gecmek icin gereken minimum zoom seviyesi.
    # Kullanici yeterince yakinlasmadiysa pahali detay patch hesaplamasi yapilmaz.
    "detail_min_zoom": 1.0,
    # detail_native_scale_threshold:
    # Preview olcegi zaten neredeyse bire bir ise detay patch'e gecmemek icin kullanilan esik.
    # Boyut farki cok azsa ikinci bir detay katmani uretmenin faydasi yoktur.
    "detail_native_scale_threshold": 1.01,
    # detail_max_output_side:
    # Tek seferde uretilen detay patch'inin maksimum genislik/yuksekligi.
    # Asiri buyuk patch'lerin RAM tuketimini ve gecikmesini kontrol altinda tutar.
    "detail_max_output_side": 4096,
    # export_progress_phase_labels:
    # Arka plandaki export script'inden gelen faz adlarinin arayuzde nasil gorunecegi.
    # Gerekirse teknik isimleri degistirmeden sadece kullaniciya gorunen etiketler burada duzenlenebilir.
    "export_progress_phase_labels": {
        "start": "Hazirlik",
        "scan": "Tarama",
        "select": "Secim",
        "cache": "Turev Cache",
        "write": "Yazma",
        "manifest": "Manifest",
        "done": "Tamamlandi",
    },
}
# ===============================================

APP_TITLE = str(CONFIG["app_title"])
OVERLAY_ALPHA = int(CONFIG["overlay_alpha"])
LAYER_KEY_BASE = str(CONFIG["layer_key_base"])
LAYER_KEY_MASK = str(CONFIG["layer_key_mask"])
ANNOTATIONS_LAYER_NAME = str(CONFIG["annotations_layer_name"])
DATASET_EXPORT_SCRIPT = str(CONFIG["dataset_export_script"])
DATASET_DEFAULT_TILE_SIZE = int(CONFIG["dataset_default_tile_size"])
DATASET_DEFAULT_OVERLAP = int(CONFIG["dataset_default_overlap"])
DATASET_DEFAULT_SAMPLING_MODE = str(CONFIG["dataset_default_sampling_mode"])
DATASET_DEFAULT_FEATURE_MODE = str(CONFIG["dataset_default_feature_mode"])
DATASET_DEFAULT_POSITIVE_RATIO = float(CONFIG["dataset_default_positive_ratio"])
DATASET_DEFAULT_VALID_RATIO = float(CONFIG["dataset_default_valid_ratio"])
DATASET_DEFAULT_TRAIN_NEG_KEEP = float(CONFIG["dataset_default_train_negative_keep"])
DATASET_DEFAULT_NEG_TO_POS_RATIO = float(CONFIG["dataset_default_neg_to_pos_ratio"])
DATASET_DEFAULT_FORMAT = str(CONFIG["dataset_default_format"])
DATASET_DEFAULT_NUM_WORKERS = int(CONFIG["dataset_default_num_workers"])
DATASET_DEFAULT_DERIVATIVE_CACHE_MODE = str(CONFIG["dataset_default_derivative_cache_mode"])
DATASET_DEFAULT_DERIVATIVE_CACHE_DIR = str(CONFIG["dataset_default_derivative_cache_dir"])
DATASET_DEFAULT_OVERWRITE = bool(CONFIG["dataset_default_overwrite"])
DEFAULT_WINDOW_WIDTH = int(CONFIG["default_window_width"])
DEFAULT_WINDOW_HEIGHT = int(CONFIG["default_window_height"])
DEFAULT_PREVIEW_MAX_SIZE = int(CONFIG["default_preview_max_size"])
DEFAULT_PREVIEW_BANDS_RAW = str(CONFIG["default_preview_bands_raw"])
DEFAULT_MODEL_BANDS_RAW = str(CONFIG["default_model_bands_raw"])
DEFAULT_MODEL_BANDS_RGB_RAW = str(CONFIG["default_model_bands_rgb_raw"])
DEFAULT_POSITIVE_VALUE = int(CONFIG["default_positive_value"])
DEFAULT_NEGATIVE_VALUE = int(CONFIG["default_negative_value"])
DEFAULT_SQUARE_MODE = bool(CONFIG["default_square_mode"])
DETAIL_REFRESH_DELAY_MS = int(CONFIG["detail_refresh_delay_ms"])
DETAIL_MIN_ZOOM = float(CONFIG["detail_min_zoom"])
DETAIL_NATIVE_SCALE_THRESHOLD = float(CONFIG["detail_native_scale_threshold"])
DETAIL_MAX_OUTPUT_SIDE = int(CONFIG["detail_max_output_side"])
StretchBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
EXPORT_PROGRESS_PHASE_LABELS: dict[str, str] = dict(CONFIG["export_progress_phase_labels"])

# ---------------------------------------------------------------------------
# Light Fresh Theme Stylesheet
# ---------------------------------------------------------------------------
APP_STYLE = """
QMainWindow, QWidget {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
}
QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ffffff, stop:1 #f1f5f9);
    border-bottom: 1px solid #e2e8f0;
    padding: 4px 6px;
    spacing: 3px;
}
QToolBar::separator {
    width: 1px;
    background: #cbd5e1;
    margin: 4px 6px;
}
QToolButton {
    background: transparent;
    color: #334155;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    min-width: 28px;
}
QToolButton:hover {
    background: #e0f2fe;
    border-color: #7dd3fc;
}
QToolButton:pressed {
    background: #bae6fd;
}
QToolButton:checked {
    background: #0284c7;
    border-color: #0369a1;
    color: #ffffff;
}
QToolButton:disabled {
    color: #94a3b8;
}
QStatusBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f1f5f9, stop:1 #e2e8f0);
    border-top: 1px solid #cbd5e1;
    padding: 2px 8px;
    font-size: 12px;
}
QStatusBar QLabel {
    color: #475569;
    padding: 0 6px;
}
QGraphicsView {
    background: #e2e8f0;
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #f1f5f9;
    width: 10px;
    height: 10px;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #94a3b8;
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #64748b;
}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
    border: none;
    height: 0; width: 0;
}
QMessageBox {
    background-color: #f8fafc;
}
QMessageBox QLabel {
    color: #1e293b;
}
QMessageBox QPushButton {
    background: #e0f2fe;
    color: #0c4a6e;
    border: 1px solid #7dd3fc;
    border-radius: 6px;
    padding: 6px 18px;
    min-width: 70px;
    font-size: 13px;
}
QMessageBox QPushButton:hover {
    background: #bae6fd;
    border-color: #38bdf8;
}
QMessageBox QPushButton:pressed {
    background: #7dd3fc;
}
QFileDialog {
    background-color: #f8fafc;
    color: #1e293b;
}
"""


def parse_bands(raw: str, count: int) -> tuple[int, int, int]:
    """Band string'ini parse et. Tek bantlı dosyalar için otomatik gri tonlama."""
    parts = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not parts:
        # Tek bantlı dosya için otomatik
        parts = [1]
    if len(parts) == 1:
        parts = [parts[0], parts[0], parts[0]]
    elif len(parts) == 2:
        parts = [parts[0], parts[1], parts[1]]
    else:
        parts = parts[:3]
    for b in parts:
        if b < 1 or b > count:
            raise ValueError(f"Band indeksi gecersiz: {b} (1-{count})")
    return parts[0], parts[1], parts[2]


def compute_stretch_bounds(
    arr: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> tuple[float, float]:
    arr_f32 = np.asarray(arr, dtype=np.float32)
    if np.ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        if np.any(mask):
            arr_f32 = arr_f32.copy()
            arr_f32[mask] = np.nan
    valid = np.isfinite(arr_f32)
    if not np.any(valid):
        return 0.0, 255.0
    vals = arr_f32[valid]
    lo = float(np.percentile(vals, low))
    hi = float(np.percentile(vals, high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 255.0
    return lo, hi


def stretch_to_uint8(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    arr_f32 = np.asarray(arr, dtype=np.float32)
    if np.ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        if np.any(mask):
            arr_f32 = arr_f32.copy()
            arr_f32[mask] = np.nan
    valid = np.isfinite(arr_f32)
    if not np.any(valid):
        return out
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        clipped = np.clip(arr_f32, 0.0, 255.0)
        out[valid] = clipped[valid].astype(np.uint8)
        return out
    scaled = (arr_f32 - float(low)) / float(high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    return out


def scaled_transform(
    src_transform: Affine,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> Affine:
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("Hedef boyutlar pozitif olmali")
    return src_transform * Affine.scale(float(src_w) / float(dst_w), float(src_h) / float(dst_h))


def resampled_window_transform(
    src_transform: Affine,
    window: Window,
    out_w: int,
    out_h: int,
) -> Affine:
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Hedef boyutlar pozitif olmali")
    return window_transform(window, src_transform) * Affine.scale(
        float(window.width) / float(out_w),
        float(window.height) / float(out_h),
    )


def sanitize_name(value: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    safe = safe.strip("_")
    return safe or "source"


def normalize_feature_mode(raw: object) -> str:
    value = str(raw).strip().lower()
    if value not in {"rgb3", "topo5"}:
        raise ValueError(f"Desteklenmeyen feature_mode: {raw!r}")
    return value


def default_model_bands_for_feature_mode(feature_mode: object) -> str:
    return (
        DEFAULT_MODEL_BANDS_RGB_RAW
        if normalize_feature_mode(feature_mode) == "rgb3"
        else DEFAULT_MODEL_BANDS_RAW
    )


def parse_int_csv(raw: str, expected_len: Optional[int] = None) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if expected_len is not None and len(parts) != expected_len:
        raise ValueError(f"Beklenen {expected_len} tamsayi, verilen: {raw!r}")
    return parts


def qimage_from_rgb(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return img.copy()


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()


def mask_to_rgba(mask: np.ndarray, negative_value: int) -> np.ndarray:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    idx = mask != np.uint8(negative_value)
    rgba[idx, 0] = 255
    rgba[idx, 3] = OVERLAY_ALPHA
    return rgba


def _read_rgb(
    ds,
    bands: tuple[int, int, int],
    out_w: int,
    out_h: int,
    *,
    window: Optional[Window] = None,
    stretch_bounds: Optional[StretchBounds] = None,
) -> tuple[np.ndarray, StretchBounds]:
    """Secilen bantlardan hedef boyutta RGB goruntu uret."""
    unique_bands: list[int] = []
    band_to_idx: dict[int, int] = {}
    for b in bands[:3]:
        if b not in band_to_idx:
            band_to_idx[b] = len(unique_bands)
            unique_bands.append(b)

    data = ds.read(
        unique_bands,
        out_shape=(len(unique_bands), out_h, out_w),
        window=window,
        resampling=Resampling.bilinear,
        masked=True,
    )

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    resolved_bounds: list[tuple[float, float]] = []
    for ch, b in enumerate(bands[:3]):
        src_idx = band_to_idx[b]
        arr = data[src_idx].astype(np.float32, copy=False)
        if stretch_bounds is None:
            low, high = compute_stretch_bounds(arr)
        else:
            low, high = stretch_bounds[ch]
        rgb[:, :, ch] = stretch_to_uint8(arr, low, high)
        resolved_bounds.append((float(low), float(high)))
    return rgb, (resolved_bounds[0], resolved_bounds[1], resolved_bounds[2])


def _read_rgb_preview(
    ds,
    bands: tuple[int, int, int],
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, StretchBounds]:
    """Seçilen bantlardan hedef boyutta RGB preview üret."""
    return _read_rgb(ds, bands, out_w, out_h)


def _read_rgb_aligned_to_grid(
    ds,
    bands: tuple[int, int, int],
    out_w: int,
    out_h: int,
    *,
    dst_transform: Affine,
    dst_crs,
    stretch_bounds: Optional[StretchBounds] = None,
) -> tuple[np.ndarray, StretchBounds]:
    vrt_kwargs: dict[str, object] = {
        "transform": dst_transform,
        "width": int(out_w),
        "height": int(out_h),
        "resampling": Resampling.bilinear,
    }
    if dst_crs is not None:
        vrt_kwargs["crs"] = dst_crs
    with WarpedVRT(ds, **vrt_kwargs) as vrt:
        return _read_rgb(vrt, bands, out_w, out_h, stretch_bounds=stretch_bounds)


def read_preview_rgb(path: Path, bands: tuple[int, int, int], out_w: int, out_h: int) -> np.ndarray:
    """Raster dosyasını hedef preview boyutunda RGB uint8'e dönüştür."""
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Hedef preview boyutu gecerli degil")
    with rasterio.open(path) as ds:
        rgb, _ = _read_rgb_preview(ds, bands, out_w, out_h)
        return rgb


def preview_to_full_box(
    box: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    full_w: int,
    full_h: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    x0 = max(0, min(int(math.floor(xmin * scale_x)), full_w - 1))
    y0 = max(0, min(int(math.floor(ymin * scale_y)), full_h - 1))
    x1 = max(x0 + 1, min(int(math.ceil((xmax + 1) * scale_x)), full_w))
    y1 = max(y0 + 1, min(int(math.ceil((ymax + 1) * scale_y)), full_h))
    return x0, y0, x1, y1


def trim_process_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def companion_gpkg_path(mask_path: Path) -> Path:
    return mask_path.with_suffix(".gpkg")


def write_annotations_gpkg(
    *,
    gpkg_path: Path,
    mask: np.ndarray,
    transform,
    crs,
    positive_value: int,
    negative_value: int,
    source_raster_path: Path,
    mask_path: Path,
) -> Optional[Path]:
    selected = (mask.astype(np.uint8, copy=False) != np.uint8(negative_value)).astype(np.uint8)
    if int(np.count_nonzero(selected)) <= 0:
        if gpkg_path.exists():
            gpkg_path.unlink(missing_ok=True)
        return None

    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape

    features: list[dict[str, object]] = []
    annotation_id = 1
    for geom, value in shapes(selected, mask=selected.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        polygon = shapely_shape(geom)
        if polygon.is_empty:
            continue
        features.append(
            {
                "annotation_id": int(annotation_id),
                "class_name": "Positive",
                "class_value": int(positive_value),
                "source_raster": str(source_raster_path),
                "mask_path": str(mask_path),
                "geometry": polygon,
            }
        )
        annotation_id += 1

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink(missing_ok=True)
    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=crs)
    gdf.to_file(gpkg_path, layer=ANNOTATIONS_LAYER_NAME, driver="GPKG")
    return gpkg_path


def load_mask_from_annotations_gpkg(
    *,
    gpkg_path: Path,
    width: int,
    height: int,
    transform,
    crs,
    positive_value: int,
    negative_value: int,
) -> np.ndarray:
    import geopandas as gpd

    try:
        gdf = gpd.read_file(gpkg_path, layer=ANNOTATIONS_LAYER_NAME)
    except Exception:
        gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return np.full((height, width), np.uint8(negative_value), dtype=np.uint8)

    if crs is not None and gdf.crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return np.full((height, width), np.uint8(negative_value), dtype=np.uint8)

    out = np.full((height, width), np.uint8(negative_value), dtype=np.uint8)
    burned = rasterize(
        ((geom, int(positive_value)) for geom in geometries),
        out_shape=(height, width),
        transform=transform,
        fill=int(negative_value),
        dtype="uint8",
    )
    out[:, :] = burned.astype(np.uint8, copy=False)
    return out


@dataclass
class AppConfig:
    input_path: Path
    output_path: Path
    existing_mask: Optional[Path]
    existing_labels: Optional[Path]
    preview_max_size: int
    bands: tuple[int, int, int]
    positive_value: int
    negative_value: int
    square_mode: bool


@dataclass
class LayerState:
    key: str
    name: str
    item: QGraphicsPixmapItem
    kind: str  # base | mask | raster
    source_path: Optional[Path] = None
    opacity: float = 1.0
    visible: bool = True
    bands: Optional[tuple[int, int, int]] = None
    preview_stretch_bounds: Optional[StretchBounds] = None
    detail_item: Optional[QGraphicsPixmapItem] = None


@dataclass
class UndoEntry:
    full_box: tuple[int, int, int, int]
    preview_box: tuple[int, int, int, int]
    full_prev: np.ndarray
    preview_prev: np.ndarray


class Session:
    """GeoTIFF etiketleme oturumu – performans optimizasyonlu."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.src = rasterio.open(cfg.input_path)
        self.profile = self.src.profile.copy()
        self.full_h = int(self.src.height)
        self.full_w = int(self.src.width)

        (
            self.preview_rgb,
            self.preview_stretch_bounds,
            self.scale_x,
            self.scale_y,
        ) = self._build_preview(cfg.preview_max_size, cfg.bands)
        self.preview_h, self.preview_w = self.preview_rgb.shape[:2]
        self.preview_transform = scaled_transform(
            self.src.transform,
            self.full_w,
            self.full_h,
            self.preview_w,
            self.preview_h,
        )

        pos_val = np.uint8(cfg.positive_value)
        neg_val = np.uint8(cfg.negative_value)
        self.mask_full = self._load_initial_mask(cfg.existing_mask, cfg.existing_labels)
        selected_preview = cv2.resize(
            (self.mask_full != neg_val).astype(np.uint8),
            (self.preview_w, self.preview_h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.mask_preview = np.full((self.preview_h, self.preview_w), neg_val, dtype=np.uint8)
        self.mask_preview[selected_preview > 0] = pos_val

        self.initial_mask_full = self.mask_full.copy()
        self.initial_mask_preview = self.mask_preview.copy()

        # --- Undo stack: bölgesel önceki değerler ---
        self.history: list[UndoEntry] = []
        self.dirty = False

        # --- O(1) stats counter ---
        self._pos_count = int(np.count_nonzero(self.mask_full != neg_val))
        self._total = int(self.mask_full.size)
        self.render_revision = 0

        # --- Persistent overlay RGBA buffer ---
        self.overlay_rgba = np.zeros((self.preview_h, self.preview_w, 4), dtype=np.uint8)
        self._rebuild_overlay_full()

    def close(self) -> None:
        try:
            self.src.close()
        except Exception:
            pass

    def _build_preview(
        self,
        max_size: int,
        bands: tuple[int, int, int],
    ) -> tuple[
        np.ndarray,
        StretchBounds,
        float,
        float,
    ]:
        h, w = self.src.height, self.src.width
        if max_size <= 0:
            scale = 1.0
        else:
            scale = min(1.0, float(max_size) / float(max(h, w)))
        ph = max(1, int(round(h * scale)))
        pw = max(1, int(round(w * scale)))
        rgb, stretch_bounds = _read_rgb_preview(self.src, bands, pw, ph)
        return rgb, stretch_bounds, float(w) / float(pw), float(h) / float(ph)

    def _load_initial_mask(
        self,
        mask_path: Optional[Path],
        labels_path: Optional[Path],
    ) -> np.ndarray:
        neg_val = np.uint8(self.cfg.negative_value)
        pos_val = np.uint8(self.cfg.positive_value)
        if mask_path is not None and mask_path.exists():
            with rasterio.open(mask_path) as ds:
                if ds.width != self.full_w or ds.height != self.full_h:
                    raise ValueError("Mevcut maske boyutu raster ile ayni olmali")
                mask = ds.read(1)
            # GeoTIFF maske varsa oturumda onu esas aliyoruz; export da ayni rasteri kullanacak.
            if np.any(mask == neg_val):
                selected = mask != neg_val
            else:
                selected = mask > 0
            out = np.full((self.full_h, self.full_w), neg_val, dtype=np.uint8)
            out[selected] = pos_val
            return out
        if labels_path is not None and labels_path.exists():
            return load_mask_from_annotations_gpkg(
                gpkg_path=labels_path,
                width=self.full_w,
                height=self.full_h,
                transform=self.src.transform,
                crs=self.src.crs,
                positive_value=int(pos_val),
                negative_value=int(neg_val),
            )
        return np.full((self.full_h, self.full_w), neg_val, dtype=np.uint8)

    # --- Overlay helpers ---
    def _rebuild_overlay_full(self) -> None:
        """Tüm overlay RGBA buffer'ını mask_preview'dan yeniden oluştur."""
        self.overlay_rgba[:, :] = mask_to_rgba(self.mask_preview, int(self.cfg.negative_value))

    def _update_overlay_region(self, py0: int, py1: int, px0: int, px1: int) -> None:
        """Overlay RGBA buffer'ın sadece belirli bölgesini güncelle."""
        region_mask = self.mask_preview[py0:py1, px0:px1]
        region = self.overlay_rgba[py0:py1, px0:px1]
        region[:, :] = mask_to_rgba(region_mask, int(self.cfg.negative_value))

    def visible_preview_rect_to_full(
        self,
        preview_rect: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        px0, py0, px1, py1 = preview_rect
        if px1 <= px0 or py1 <= py0:
            raise ValueError("Preview rect bos olamaz")
        return preview_to_full_box(
            (px0, py0, px1 - 1, py1 - 1),
            self.scale_x,
            self.scale_y,
            self.full_w,
            self.full_h,
        )

    def render_detail_patch(
        self,
        preview_rect: tuple[int, int, int, int],
        target_w: int,
        target_h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        window, out_w, out_h, _detail_transform = self.resolve_detail_request(
            preview_rect,
            target_w,
            target_h,
        )
        full_x0 = int(window.col_off)
        full_y0 = int(window.row_off)
        full_x1 = full_x0 + int(window.width)
        full_y1 = full_y0 + int(window.height)
        rgb, _ = _read_rgb(
            self.src,
            self.cfg.bands,
            out_w,
            out_h,
            window=window,
            stretch_bounds=self.preview_stretch_bounds,
        )

        mask_patch = self.mask_full[full_y0:full_y1, full_x0:full_x1]
        if mask_patch.shape[1] != out_w or mask_patch.shape[0] != out_h:
            mask_patch = cv2.resize(mask_patch, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        rgba = mask_to_rgba(mask_patch.astype(np.uint8, copy=False), int(self.cfg.negative_value))
        return rgb, rgba

    def resolve_detail_request(
        self,
        preview_rect: tuple[int, int, int, int],
        target_w: int,
        target_h: int,
    ) -> tuple[Window, int, int, Affine]:
        full_x0, full_y0, full_x1, full_y1 = self.visible_preview_rect_to_full(preview_rect)
        full_window_w = max(1, full_x1 - full_x0)
        full_window_h = max(1, full_y1 - full_y0)
        out_w = max(1, min(int(target_w), int(full_window_w), DETAIL_MAX_OUTPUT_SIDE))
        out_h = max(1, min(int(target_h), int(full_window_h), DETAIL_MAX_OUTPUT_SIDE))
        window = Window(
            col_off=int(full_x0),
            row_off=int(full_y0),
            width=int(full_window_w),
            height=int(full_window_h),
        )
        detail_transform = resampled_window_transform(self.src.transform, window, out_w, out_h)
        return window, out_w, out_h, detail_transform

    def apply_box(self, box: tuple[int, int, int, int], mode: str) -> None:
        px0, py0, px1, py1 = box
        x0, y0, x1, y1 = preview_to_full_box(box, self.scale_x, self.scale_y, self.full_w, self.full_h)
        pxi1 = min(self.preview_w, px1 + 1)
        pyi1 = min(self.preview_h, py1 + 1)

        pos_val = np.uint8(self.cfg.positive_value)
        neg_val = np.uint8(self.cfg.negative_value)
        val = pos_val if mode == "draw" else neg_val
        full_view = self.mask_full[y0:y1, x0:x1]
        preview_view = self.mask_preview[py0:pyi1, px0:pxi1]

        prev_pos = int(np.count_nonzero(full_view != neg_val))
        preview_changed = bool(np.any(preview_view != val))
        if mode == "draw":
            full_changed = prev_pos < full_view.size
        else:
            full_changed = prev_pos > 0
        if not (full_changed or preview_changed):
            return

        # Önceki değerler: hızlı undo için kaydedilir.
        full_prev = full_view.copy()
        preview_prev = preview_view.copy()

        # Maskeleri güncelle
        full_view[:, :] = val
        preview_view[:, :] = val

        if mode == "draw":
            self._pos_count += (full_view.size - prev_pos)
        else:
            self._pos_count -= prev_pos

        # Overlay'ı sadece değişen bölgede güncelle
        self._update_overlay_region(py0, pyi1, px0, pxi1)

        self.history.append(
            UndoEntry(
                full_box=(x0, y0, x1, y1),
                preview_box=(px0, py0, pxi1, pyi1),
                full_prev=full_prev,
                preview_prev=preview_prev,
            )
        )
        self.dirty = True
        self.render_revision += 1

    def undo(self) -> None:
        if not self.history:
            return
        entry = self.history.pop()
        x0, y0, x1, y1 = entry.full_box
        px0, py0, pxi1, pyi1 = entry.preview_box

        neg_val = np.uint8(self.cfg.negative_value)
        full_view = self.mask_full[y0:y1, x0:x1]
        cur_pos = int(np.count_nonzero(full_view != neg_val))
        prev_pos = int(np.count_nonzero(entry.full_prev != neg_val))
        full_view[:, :] = entry.full_prev
        self._pos_count += (prev_pos - cur_pos)

        self.mask_preview[py0:pyi1, px0:pxi1] = entry.preview_prev
        self._update_overlay_region(py0, pyi1, px0, pxi1)
        self.dirty = bool(self.history)
        self.render_revision += 1

    def clear(self) -> None:
        neg_val = np.uint8(self.cfg.negative_value)
        self.mask_full.fill(neg_val)
        self.mask_preview.fill(neg_val)
        self.overlay_rgba.fill(0)
        self.history.clear()
        self._pos_count = 0
        self.dirty = True
        self.render_revision += 1

    def reset(self) -> None:
        neg_val = np.uint8(self.cfg.negative_value)
        self.mask_full[:, :] = self.initial_mask_full
        self.mask_preview[:, :] = self.initial_mask_preview
        self.history.clear()
        self._pos_count = int(np.count_nonzero(self.initial_mask_full != neg_val))
        self._rebuild_overlay_full()
        self.dirty = True
        self.render_revision += 1

    def set_positive_value(self, new_value: int) -> bool:
        """Pozitif sınıf değerini güncelle ve mevcut seçili alanları yeni değere eşitle."""
        clamped = max(1, min(int(new_value), 255))
        return self.set_class_values(clamped, int(self.cfg.negative_value))

    def set_negative_value(self, new_value: int) -> bool:
        clamped = max(0, min(int(new_value), 255))
        return self.set_class_values(int(self.cfg.positive_value), clamped)

    def set_class_values(self, positive_value: int, negative_value: int) -> bool:
        new_pos = max(1, min(int(positive_value), 255))
        new_neg = max(0, min(int(negative_value), 255))
        if new_pos == new_neg:
            raise ValueError("Secili ve secilmeyen deger ayni olamaz")

        old_pos = int(self.cfg.positive_value)
        old_neg = int(self.cfg.negative_value)
        if new_pos == old_pos and new_neg == old_neg:
            return False

        old_neg_u8 = np.uint8(old_neg)
        selected_full = self.mask_full != old_neg_u8
        selected_preview = self.mask_preview != old_neg_u8
        selected_init_full = self.initial_mask_full != old_neg_u8
        selected_init_preview = self.initial_mask_preview != old_neg_u8

        has_selected = bool(np.any(selected_full) or np.any(selected_init_full))
        pixels_changed = (new_neg != old_neg) or ((new_pos != old_pos) and has_selected)

        if pixels_changed:
            new_pos_u8 = np.uint8(new_pos)
            new_neg_u8 = np.uint8(new_neg)

            self.mask_full.fill(new_neg_u8)
            self.mask_full[selected_full] = new_pos_u8

            self.mask_preview.fill(new_neg_u8)
            self.mask_preview[selected_preview] = new_pos_u8

            self.initial_mask_full.fill(new_neg_u8)
            self.initial_mask_full[selected_init_full] = new_pos_u8

            self.initial_mask_preview.fill(new_neg_u8)
            self.initial_mask_preview[selected_init_preview] = new_pos_u8

            self._rebuild_overlay_full()
            self.dirty = True
            self.render_revision += 1

        self.cfg.positive_value = new_pos
        self.cfg.negative_value = new_neg
        self._pos_count = int(np.count_nonzero(selected_full))

        # Sınıf değerleri değiştiğinde undo kayıtları eski değerleri taşıyacağı için temizlenir.
        self.history.clear()
        return pixels_changed

    def save(self, path: Path) -> None:
        profile = self.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=int(self.cfg.negative_value),
            compress="deflate",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.mask_full[np.newaxis, :, :].astype(np.uint8, copy=False))
        gpkg_path = companion_gpkg_path(path)
        write_annotations_gpkg(
            gpkg_path=gpkg_path,
            mask=self.mask_full,
            transform=self.src.transform,
            crs=self.src.crs,
            positive_value=int(self.cfg.positive_value),
            negative_value=int(self.cfg.negative_value),
            source_raster_path=self.cfg.input_path,
            mask_path=path,
        )
        self.cfg.output_path = path
        self.cfg.existing_mask = path
        self.cfg.existing_labels = gpkg_path if gpkg_path.exists() else None
        self.dirty = False

    def stats(self) -> tuple[int, int, float]:
        """O(1) istatistik – counter tabanlı."""
        ratio = (100.0 * self._pos_count / self._total) if self._total > 0 else 0.0
        return self._pos_count, self._total, ratio

class AnnotView(QGraphicsView):
    box_committed = Signal(int, int, int, int)
    zoom_changed = Signal(float)
    viewport_changed = Signal()

    # Cursors
    _CURSOR_DRAW = Qt.CursorShape.CrossCursor
    _CURSOR_ERASE = Qt.CursorShape.PointingHandCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setAcceptDrops(True)

        self.image_w = 1
        self.image_h = 1
        self.square_mode = True
        self.mode = "draw"
        self.wheel_inverted = False

        self._panning = False
        self._pan_last_pos = None
        self._drawing = False
        self._start = None
        self._rect_item: Optional[QGraphicsRectItem] = None

        self._apply_mode_cursor()

    def _apply_mode_cursor(self) -> None:
        if self.mode == "draw":
            self.viewport().setCursor(self._CURSOR_DRAW)
        else:
            self.viewport().setCursor(self._CURSOR_ERASE)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._apply_mode_cursor()

    def set_image_size(self, w: int, h: int) -> None:
        self.image_w = int(w)
        self.image_h = int(h)

    def fit_all(self) -> None:
        self.fitInView(QRectF(0, 0, self.image_w, self.image_h), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.transform().m11())
        self.viewport_changed.emit()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if self.wheel_inverted:
            delta = -delta
        factor = 1.2 if delta > 0 else (1.0 / 1.2)
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        self.viewport_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewport_changed.emit()

    def _clamp_point(self, p: QPointF) -> QPointF:
        x = min(max(p.x(), 0.0), float(self.image_w - 1))
        y = min(max(p.y(), 0.0), float(self.image_h - 1))
        return QPointF(x, y)

    def _norm_box(self, p0: QPointF, p1: QPointF) -> tuple[int, int, int, int]:
        x0, y0 = p0.x(), p0.y()
        x1, y1 = p1.x(), p1.y()
        if self.square_mode:
            dx, dy = x1 - x0, y1 - y0
            side = max(abs(dx), abs(dy))
            if side > 0:
                x1 = x0 + side if dx >= 0 else x0 - side
                y1 = y0 + side if dy >= 0 else y0 - side
        xmin = int(np.clip(np.floor(min(x0, x1)), 0, self.image_w - 1))
        ymin = int(np.clip(np.floor(min(y0, y1)), 0, self.image_h - 1))
        xmax = int(np.clip(np.ceil(max(x0, x1)), 0, self.image_w - 1))
        ymax = int(np.clip(np.ceil(max(y0, y1)), 0, self.image_h - 1))
        return xmin, ymin, xmax, ymax

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start = self._clamp_point(self.mapToScene(event.pos()))
            color = Qt.GlobalColor.green if self.mode == "draw" else Qt.GlobalColor.yellow
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            fill = QColor(color)
            fill.setAlpha(64)
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(pen)
            self._rect_item.setBrush(QBrush(fill))
            self._rect_item.setZValue(1_000_000.0)
            self._rect_item.setRect(QRectF(self._start.x(), self._start.y(), 1.0, 1.0))
            if self.scene() is not None:
                self.scene().addItem(self._rect_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_last_pos is not None:
            d = event.pos() - self._pan_last_pos
            self._pan_last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        if self._drawing and self._start is not None and self._rect_item is not None:
            p1 = self._clamp_point(self.mapToScene(event.pos()))
            xmin, ymin, xmax, ymax = self._norm_box(self._start, p1)
            self._rect_item.setRect(
                QRectF(
                    float(xmin),
                    float(ymin),
                    float(max(1, xmax - xmin + 1)),
                    float(max(1, ymax - ymin + 1)),
                )
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self._pan_last_pos = None
            self.unsetCursor()
            self.viewport_changed.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drawing and self._start is not None:
            p1 = self._clamp_point(self.mapToScene(event.pos()))
            xmin, ymin, xmax, ymax = self._norm_box(self._start, p1)
            if xmax > xmin and ymax > ymin:
                self.box_committed.emit(xmin, ymin, xmax, ymax)
            self._drawing = False
            self._start = None
            if self._rect_item is not None and self.scene() is not None:
                self.scene().removeItem(self._rect_item)
            self._rect_item = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Band Selection Dialog
# ---------------------------------------------------------------------------
class BandSelectionDialog(QDialog):
    """Çok bantlı dosyalarda R/G/B band seçim dialog'u."""

    def __init__(self, band_count: int, default_raw: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Band Seçimi")
        self.setMinimumWidth(360)
        self.setStyleSheet(APP_STYLE)

        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(f"Dosyada <b>{band_count}</b> bant mevcut. Görüntüleme için band atayın.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Preset combo
        preset_group = QGroupBox("Hazır Ayar")
        preset_layout = QFormLayout(preset_group)
        self._preset_combo = QComboBox()
        presets = []
        if band_count >= 3:
            presets.append(("RGB (1, 2, 3)", "1,2,3"))
        if band_count >= 5:
            presets.append(("RGB (3, 2, 1) – BGR", "3,2,1"))
            presets.append(("Band 4, 3, 2 – Yakın Kızılötesi", "4,3,2"))
        presets.append(("Gri Tonlama (Band 1)", "1"))
        presets.append(("Özel…", ""))
        self._presets = presets
        for label, _ in presets:
            self._preset_combo.addItem(label)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addRow("Seçim:", self._preset_combo)
        layout.addWidget(preset_group)

        # Custom band spinboxes
        self._custom_group = QGroupBox("Özel Band Seçimi")
        custom_layout = QFormLayout(self._custom_group)

        self._spin_r = QSpinBox()
        self._spin_r.setRange(1, band_count)
        self._spin_r.setValue(1)
        custom_layout.addRow("Kırmızı (R):", self._spin_r)

        self._spin_g = QSpinBox()
        self._spin_g.setRange(1, band_count)
        self._spin_g.setValue(min(2, band_count))
        custom_layout.addRow("Yeşil (G):", self._spin_g)

        self._spin_b = QSpinBox()
        self._spin_b.setRange(1, band_count)
        self._spin_b.setValue(min(3, band_count))
        custom_layout.addRow("Mavi (B):", self._spin_b)

        self._custom_group.setVisible(False)
        layout.addWidget(self._custom_group)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Try to match the default
        self._select_default(default_raw)

    def _select_default(self, raw: str) -> None:
        raw_clean = raw.replace(" ", "")
        for i, (_, val) in enumerate(self._presets):
            if val == raw_clean:
                self._preset_combo.setCurrentIndex(i)
                return
        # Default to first preset
        self._preset_combo.setCurrentIndex(0)

    def _on_preset_changed(self, index: int) -> None:
        _, val = self._presets[index]
        is_custom = val == ""
        self._custom_group.setVisible(is_custom)

    def get_bands_raw(self) -> str:
        idx = self._preset_combo.currentIndex()
        _, val = self._presets[idx]
        if val:
            return val
        return f"{self._spin_r.value()},{self._spin_g.value()},{self._spin_b.value()}"


class MaskValuesDialog(QDialog):
    """Maske sınıf değerleri için ince ayar dialog'u."""

    def __init__(self, positive_value: int, negative_value: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maske Ayarlari")
        self.setMinimumWidth(360)
        self.setStyleSheet(APP_STYLE)

        layout = QVBoxLayout(self)

        info = QLabel("Secilen ve secilmeyen piksel degerlerini ayarlayin.")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.spin_positive = QSpinBox(self)
        self.spin_positive.setRange(1, 255)
        self.spin_positive.setValue(max(1, min(int(positive_value), 255)))
        self.spin_positive.setToolTip("Cizim (Draw) ile isaretlenen pikseller")
        form.addRow("Secilen:", self.spin_positive)

        self.spin_negative = QSpinBox(self)
        self.spin_negative.setRange(0, 255)
        self.spin_negative.setValue(max(0, min(int(negative_value), 255)))
        self.spin_negative.setToolTip("Silme (Erase) ve arkaplan pikselleri")
        form.addRow("Secilmeyen:", self.spin_negative)

        layout.addLayout(form)

        self._validation = QLabel()
        self._validation.setStyleSheet("color: #b91c1c;")
        layout.addWidget(self._validation)

        self._btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btn_box.accepted.connect(self.accept)
        self._btn_box.rejected.connect(self.reject)
        layout.addWidget(self._btn_box)

        self.spin_positive.valueChanged.connect(self._sync_validation)
        self.spin_negative.valueChanged.connect(self._sync_validation)
        self._sync_validation()

    def _sync_validation(self) -> None:
        same = self.spin_positive.value() == self.spin_negative.value()
        ok_btn = self._btn_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(not same)
        if same:
            self._validation.setText("Secilen ve secilmeyen deger ayni olamaz.")
        else:
            self._validation.setText("")

    def values(self) -> tuple[int, int]:
        return int(self.spin_positive.value()), int(self.spin_negative.value())


class TileDatasetExportDialog(QDialog):
    """Pozitif/Negatif tile dataset export ayarlari."""

    def __init__(
        self,
        *,
        default_output_dir: Path,
        default_feature_mode: str,
        default_bands_raw: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Tile Dataset Export")
        self.setMinimumWidth(520)
        self.setStyleSheet(APP_STYLE)
        self._auto_output_dir = Path(default_output_dir)
        self._output_dir_manually_changed = False

        layout = QVBoxLayout(self)

        info = QLabel(
            "Mevcut secimler, Positive/Negative klasor yapisina sahip "
            "tile-classification datasetine donusturulecek."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        output_wrap = QWidget(self)
        output_row = QHBoxLayout(output_wrap)
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(6)
        self.edit_output_dir = QLineEdit(str(default_output_dir))
        self.edit_output_dir.setPlaceholderText("Dataset cikti klasoru")
        self.edit_output_dir.textEdited.connect(self._mark_output_dir_manual)
        output_row.addWidget(self.edit_output_dir, 1)
        self.btn_browse_output = QPushButton("Sec...")
        self.btn_browse_output.clicked.connect(self._choose_output_dir)
        output_row.addWidget(self.btn_browse_output)
        form.addRow("Cikti klasoru:", output_wrap)

        self.combo_feature_mode = QComboBox(self)
        self.combo_feature_mode.addItem("topo5 - RGB + SVF + SLRM", "topo5")
        self.combo_feature_mode.addItem("rgb3 - sadece RGB", "rgb3")
        default_feature_index = self.combo_feature_mode.findData(
            normalize_feature_mode(default_feature_mode)
        )
        self.combo_feature_mode.setCurrentIndex(max(0, default_feature_index))
        form.addRow("Feature modu:", self.combo_feature_mode)

        self.edit_model_bands = QLineEdit(default_bands_raw)
        self.edit_model_bands.setPlaceholderText(DEFAULT_MODEL_BANDS_RAW)
        self.edit_model_bands.setToolTip("topo5: R,G,B,DSM,DTM | rgb3: R,G,B")
        form.addRow("Model bantlari:", self.edit_model_bands)

        self.spin_tile_size = QSpinBox(self)
        self.spin_tile_size.setRange(64, 4096)
        self.spin_tile_size.setValue(DATASET_DEFAULT_TILE_SIZE)
        self.spin_tile_size.valueChanged.connect(self._sync_overlap_range)
        form.addRow("Tile boyutu:", self.spin_tile_size)

        self.spin_overlap = QSpinBox(self)
        self.spin_overlap.setRange(0, DATASET_DEFAULT_TILE_SIZE - 1)
        self.spin_overlap.setValue(DATASET_DEFAULT_OVERLAP)
        form.addRow("Overlap:", self.spin_overlap)

        self.combo_sampling_mode = QComboBox(self)
        self.combo_sampling_mode.addItem("Secimlerden tile + rastgele negatif", "selected_regions")
        self.combo_sampling_mode.addItem("Tum rasterda kayan pencere", "full_grid")
        default_sampling_index = self.combo_sampling_mode.findData(DATASET_DEFAULT_SAMPLING_MODE)
        self.combo_sampling_mode.setCurrentIndex(max(0, default_sampling_index))
        form.addRow("Uretim modu:", self.combo_sampling_mode)

        self.spin_positive_ratio = QDoubleSpinBox(self)
        self.spin_positive_ratio.setRange(0.0, 1.0)
        self.spin_positive_ratio.setDecimals(4)
        self.spin_positive_ratio.setSingleStep(0.005)
        self.spin_positive_ratio.setValue(DATASET_DEFAULT_POSITIVE_RATIO)
        form.addRow("Pozitif esigi:", self.spin_positive_ratio)

        self.spin_negative_to_positive = QDoubleSpinBox(self)
        self.spin_negative_to_positive.setRange(0.0, 20.0)
        self.spin_negative_to_positive.setDecimals(3)
        self.spin_negative_to_positive.setSingleStep(0.25)
        self.spin_negative_to_positive.setValue(DATASET_DEFAULT_NEG_TO_POS_RATIO)
        form.addRow("Neg/Poz oran:", self.spin_negative_to_positive)

        self.spin_valid_ratio = QDoubleSpinBox(self)
        self.spin_valid_ratio.setRange(0.0, 1.0)
        self.spin_valid_ratio.setDecimals(3)
        self.spin_valid_ratio.setSingleStep(0.05)
        self.spin_valid_ratio.setValue(DATASET_DEFAULT_VALID_RATIO)
        form.addRow("Min. gecerli oran:", self.spin_valid_ratio)

        self.spin_train_negative_keep = QDoubleSpinBox(self)
        self.spin_train_negative_keep.setRange(0.0, 1.0)
        self.spin_train_negative_keep.setDecimals(3)
        self.spin_train_negative_keep.setSingleStep(0.05)
        self.spin_train_negative_keep.setValue(DATASET_DEFAULT_TRAIN_NEG_KEEP)
        form.addRow("Train negatif tut:", self.spin_train_negative_keep)

        self.combo_format = QComboBox(self)
        self.combo_format.addItem("NPZ - sikistirilmis, daha kucuk", "npz")
        self.combo_format.addItem("NPY - sikistirmasiz, genelde daha hizli", "npy")
        self.combo_format.setCurrentIndex(0 if DATASET_DEFAULT_FORMAT == "npz" else 1)
        form.addRow("Dosya formati:", self.combo_format)

        self.spin_num_workers = QSpinBox(self)
        self.spin_num_workers.setRange(1, 64)
        self.spin_num_workers.setValue(DATASET_DEFAULT_NUM_WORKERS)
        self.spin_num_workers.setToolTip("Daha yuksek deger CPU ve disk kullanimini artirir.")
        form.addRow("Worker sayisi:", self.spin_num_workers)

        self.combo_derivative_cache_mode = QComboBox(self)
        self.combo_derivative_cache_mode.addItem("Otomatik (onerilen)", "auto")
        self.combo_derivative_cache_mode.addItem("Kapali", "none")
        self.combo_derivative_cache_mode.addItem("NPZ (RAM uygunsa)", "npz")
        self.combo_derivative_cache_mode.addItem("Raster-cache (buyuk raster)", "raster")
        default_cache_index = self.combo_derivative_cache_mode.findData(DATASET_DEFAULT_DERIVATIVE_CACHE_MODE)
        self.combo_derivative_cache_mode.setCurrentIndex(max(0, default_cache_index))
        self.combo_derivative_cache_mode.setToolTip(
            "Bos birakilan cache klasoru icin varsayilan yol: <girdi_raster_klasoru>/cache"
        )
        form.addRow("Turev cache:", self.combo_derivative_cache_mode)

        cache_wrap = QWidget(self)
        cache_row = QHBoxLayout(cache_wrap)
        cache_row.setContentsMargins(0, 0, 0, 0)
        cache_row.setSpacing(6)
        self.edit_derivative_cache_dir = QLineEdit(DATASET_DEFAULT_DERIVATIVE_CACHE_DIR)
        self.edit_derivative_cache_dir.setPlaceholderText("<girdi_raster_klasoru>/cache")
        cache_row.addWidget(self.edit_derivative_cache_dir, 1)
        self.btn_browse_derivative_cache_dir = QPushButton("Sec...")
        self.btn_browse_derivative_cache_dir.clicked.connect(self._choose_derivative_cache_dir)
        cache_row.addWidget(self.btn_browse_derivative_cache_dir)
        form.addRow("Cache klasoru:", cache_wrap)

        self.combo_overwrite = QComboBox(self)
        self.combo_overwrite.addItem("Evet - klasoru temizle", True)
        self.combo_overwrite.addItem("Hayir - klasor bos olmali", False)
        default_overwrite_index = self.combo_overwrite.findData(DATASET_DEFAULT_OVERWRITE)
        self.combo_overwrite.setCurrentIndex(max(0, default_overwrite_index))
        form.addRow("Uzerine yaz:", self.combo_overwrite)

        layout.addLayout(form)

        hint = QLabel(
            "Onerilen baslangic: tile=256, overlap=128, pozitif esigi=0.02.\n"
            "Not: 0.02 esigi, tile'in sadece %2'si secili olsa bile Positive yazabilir; "
            "buyuk tile boyutlarinda secim tile'in kenarinda kalmis gibi gorunebilir.\n"
            "topo5 mevcut modelle uyumludur; rgb3 sadece RGB tile dataset uretir.\n"
            "Derivative cache varsayilan olarak <girdi_raster_klasoru>/cache altina yazilir.\n"
            "Hiz onemliyse NPY + auto cache + orta seviye worker sayisi iyi baslangictir."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #475569; font-size: 12px;")
        layout.addWidget(hint)

        self._validation = QLabel()
        self._validation.setStyleSheet("color: #b91c1c;")
        layout.addWidget(self._validation)

        self._btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btn_box.accepted.connect(self.accept)
        self._btn_box.rejected.connect(self.reject)
        layout.addWidget(self._btn_box)

        self.edit_output_dir.textChanged.connect(self._sync_validation)
        self.combo_feature_mode.currentIndexChanged.connect(self._sync_feature_mode_ui)
        self.edit_model_bands.textChanged.connect(self._sync_validation)
        self.spin_overlap.valueChanged.connect(self._sync_validation)
        self.spin_tile_size.valueChanged.connect(self._sync_validation)
        self.combo_sampling_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self._sync_overlap_range()
        self._sync_mode_ui()
        self._sync_feature_mode_ui()
        self._sync_validation()

    def _choose_output_dir(self) -> None:
        current = self.edit_output_dir.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Cikti klasorunu sec", current)
        if chosen:
            self._output_dir_manually_changed = True
            self.edit_output_dir.setText(chosen)

    def _mark_output_dir_manual(self, _text: str) -> None:
        self._output_dir_manually_changed = True

    def _choose_derivative_cache_dir(self) -> None:
        current = self.edit_derivative_cache_dir.text().strip() or str(Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Cache klasorunu sec", current)
        if chosen:
            self.edit_derivative_cache_dir.setText(chosen)

    def _sync_overlap_range(self) -> None:
        tile_size = max(1, int(self.spin_tile_size.value()))
        current = int(self.spin_overlap.value())
        self.spin_overlap.setMaximum(max(0, tile_size - 1))
        if current >= tile_size:
            self.spin_overlap.setValue(max(0, tile_size - 1))

    def _sync_mode_ui(self) -> None:
        selected_mode = str(self.combo_sampling_mode.currentData())
        is_selected_regions = selected_mode == "selected_regions"
        self.spin_negative_to_positive.setEnabled(is_selected_regions)
        self.spin_train_negative_keep.setEnabled(not is_selected_regions)

    def _sync_feature_mode_ui(self) -> None:
        feature_mode = normalize_feature_mode(self.combo_feature_mode.currentData())
        expected_bands = 3 if feature_mode == "rgb3" else 5
        placeholder = default_model_bands_for_feature_mode(feature_mode)
        self.edit_model_bands.setPlaceholderText(placeholder)
        if not self.edit_model_bands.text().strip():
            self.edit_model_bands.setText(placeholder)
        current_bands = self.edit_model_bands.text().strip()
        try:
            parse_int_csv(current_bands, expected_len=expected_bands)
        except Exception:
            self.edit_model_bands.setText(placeholder)
        uses_derivatives = feature_mode == "topo5"
        self.combo_derivative_cache_mode.setEnabled(uses_derivatives)
        self.edit_derivative_cache_dir.setEnabled(uses_derivatives)
        self.btn_browse_derivative_cache_dir.setEnabled(uses_derivatives)
        if not uses_derivatives:
            idx = self.combo_derivative_cache_mode.findData("none")
            if idx >= 0:
                self.combo_derivative_cache_mode.setCurrentIndex(idx)
        if not self._output_dir_manually_changed:
            auto_output = self._auto_output_dir
            if auto_output.name:
                stem = auto_output.name
                suffixes = ("_rgb3", "_topo5")
                for suffix in suffixes:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                auto_output = auto_output.with_name(f"{stem}_{feature_mode}")
            self.edit_output_dir.setText(str(auto_output))

    def _sync_validation(self) -> None:
        output_ok = bool(self.edit_output_dir.text().strip())
        overlap_ok = int(self.spin_overlap.value()) < int(self.spin_tile_size.value())
        feature_mode = normalize_feature_mode(self.combo_feature_mode.currentData())
        expected_len = 3 if feature_mode == "rgb3" else 5
        try:
            bands = parse_int_csv(self.edit_model_bands.text().strip(), expected_len=expected_len)
            bands_ok = all(int(v) > 0 for v in bands)
        except Exception:
            bands_ok = False
        ok_btn = self._btn_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(output_ok and overlap_ok and bands_ok)
        if not output_ok:
            self._validation.setText("Bir cikti klasoru belirtin.")
        elif not bands_ok:
            if feature_mode == "rgb3":
                self._validation.setText("RGB3 icin model bantlari 3 tamsayi olmali: R,G,B")
            else:
                self._validation.setText("TOPO5 icin model bantlari 5 tamsayi olmali: R,G,B,DSM,DTM")
        elif not overlap_ok:
            self._validation.setText("Overlap, tile boyutundan kucuk olmali.")
        else:
            self._validation.setText("")

    def values(self) -> dict[str, object]:
        output_dir = Path(self.edit_output_dir.text().strip()).expanduser().resolve()
        return {
            "output_dir": output_dir,
            "feature_mode": normalize_feature_mode(self.combo_feature_mode.currentData()),
            "bands_raw": self.edit_model_bands.text().strip(),
            "tile_size": int(self.spin_tile_size.value()),
            "overlap": int(self.spin_overlap.value()),
            "sampling_mode": str(self.combo_sampling_mode.currentData()),
            "positive_ratio_threshold": float(self.spin_positive_ratio.value()),
            "negative_to_positive_ratio": float(self.spin_negative_to_positive.value()),
            "valid_ratio_threshold": float(self.spin_valid_ratio.value()),
            "train_negative_keep_ratio": float(self.spin_train_negative_keep.value()),
            "format": str(self.combo_format.currentData()),
            "num_workers": int(self.spin_num_workers.value()),
            "derivative_cache_mode": str(self.combo_derivative_cache_mode.currentData()),
            "derivative_cache_dir": self.edit_derivative_cache_dir.text().strip(),
            "overwrite": bool(self.combo_overwrite.currentData()),
        }


class MainWindow(QMainWindow):
    def __init__(
        self,
        preview_max_size: int,
        bands_raw: str,
        positive_value: int,
        negative_value: int,
        square_mode: bool,
        session: Optional[Session] = None,
    ):
        super().__init__()
        self.s: Optional[Session] = None
        self.mode = "draw"
        self.square_mode = bool(square_mode)
        self.preview_max_size = int(preview_max_size)
        self.bands_raw = bands_raw
        self.model_feature_mode = DATASET_DEFAULT_FEATURE_MODE
        self.model_bands_raw = default_model_bands_for_feature_mode(self.model_feature_mode)
        self._shown_dataset_input_warnings: set[tuple[str, str, str]] = set()
        self.positive_value = int(positive_value)
        self.negative_value = int(negative_value)

        # --- Apply light theme ---
        self.setStyleSheet(APP_STYLE)

        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setAcceptDrops(True)

        self.scene = QGraphicsScene(self)
        self.view = AnnotView(self)
        self.view.setScene(self.scene)
        self.view.square_mode = self.square_mode
        self.view.mode = self.mode
        self.view.box_committed.connect(self.on_box)
        self.view.zoom_changed.connect(self.update_status)
        self.view.zoom_changed.connect(self._schedule_detail_refresh)
        self.view.viewport_changed.connect(self._schedule_detail_refresh)
        self.view.horizontalScrollBar().valueChanged.connect(self._schedule_detail_refresh)
        self.view.verticalScrollBar().valueChanged.connect(self._schedule_detail_refresh)

        self._layers_updating_ui = False
        self._extra_layer_counter = 0
        self.layers: list[LayerState] = []
        self._detail_cache_key: Optional[tuple[object, ...]] = None
        self._detail_refresh_in_progress = False
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.timeout.connect(self._refresh_detail_view)

        self.base_item = QGraphicsPixmapItem()
        self.base_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.base_detail_item = QGraphicsPixmapItem()
        self.base_detail_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.base_detail_item.setVisible(False)
        self.mask_detail_item = QGraphicsPixmapItem()
        self.mask_detail_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.mask_detail_item.setVisible(False)
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.base_detail_item)
        self.scene.addItem(self.mask_item)
        self.scene.addItem(self.mask_detail_item)

        self.layer_panel = self._build_layer_panel()
        center = QWidget(self)
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.layer_panel, 0)
        center_layout.addWidget(self.view, 1)
        self.setCentralWidget(center)

        self._reset_layer_stack()

        self._build_toolbar()
        self._build_status_bar()
        self._set_actions_enabled(False)
        self._set_window_title()
        self.show_empty_state()
        self.update_status()

        if session is not None:
            self.set_session(session)

    def _set_window_title(self) -> None:
        if self.s is None:
            self.setWindowTitle(f"{APP_TITLE} [{QT_BACKEND}]")
            return
        self.setWindowTitle(f"{APP_TITLE} - {self.s.cfg.input_path.name} [{QT_BACKEND}]")

    # ------------------------------------------------------------------
    # Layer panel
    # ------------------------------------------------------------------
    _PANEL_STYLE = """
        QWidget#LayerPanel {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8fafc);
            border-right: 1px solid #e2e8f0;
        }
        QLabel#LayerTitle {
            font-weight: 700;
            font-size: 14px;
            color: #0f172a;
            padding: 2px 0;
        }
        QPushButton {
            background: #e0f2fe;
            color: #0c4a6e;
            border: 1px solid #bae6fd;
            border-radius: 5px;
            padding: 4px 10px;
            font-size: 12px;
        }
        QPushButton:hover { background: #bae6fd; border-color: #7dd3fc; }
        QPushButton:pressed { background: #7dd3fc; }
        QPushButton:disabled { background: #f1f5f9; color: #94a3b8; border-color: #e2e8f0; }
        QListWidget {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 4px;
            font-size: 13px;
            outline: none;
        }
        QListWidget::item {
            padding: 5px 6px;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background: #dbeafe;
            color: #1e3a5f;
        }
        QListWidget::item:hover {
            background: #f0f9ff;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -5px 0;
            background: #0284c7;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #0369a1;
        }
        QSlider::sub-page:horizontal {
            background: #7dd3fc;
            border-radius: 3px;
        }
    """

    def _build_layer_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("LayerPanel")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(380)
        panel.setStyleSheet(self._PANEL_STYLE)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Katmanlar")
        title.setObjectName("LayerTitle")
        layout.addWidget(title)

        # --- Add / Remove buttons ---
        top_row = QHBoxLayout()
        self.btn_add_layer = QPushButton("\U0001f9f1 Ekle")
        self.btn_add_layer.setToolTip("Goruntuye ekstra raster katmani ekle")
        self.btn_add_layer.clicked.connect(self.add_visual_layer)
        top_row.addWidget(self.btn_add_layer)

        self.btn_remove_layer = QPushButton("\u2796 Sil")
        self.btn_remove_layer.setToolTip("Secili ekstra katmani kaldir")
        self.btn_remove_layer.clicked.connect(self.remove_selected_layer)
        top_row.addWidget(self.btn_remove_layer)
        layout.addLayout(top_row)

        # --- Move buttons ---
        move_row = QHBoxLayout()
        self.btn_layer_up = QPushButton("\u2b06 Yukari")
        self.btn_layer_up.setToolTip("Secili katmani yukariya tasi (veya surukle)")
        self.btn_layer_up.clicked.connect(self.move_selected_layer_up)
        move_row.addWidget(self.btn_layer_up)

        self.btn_layer_down = QPushButton("\u2b07 Asagi")
        self.btn_layer_down.setToolTip("Secili katmani asagiya tasi (veya surukle)")
        self.btn_layer_down.clicked.connect(self.move_selected_layer_down)
        move_row.addWidget(self.btn_layer_down)
        layout.addLayout(move_row)

        # --- Layer list with drag-reorder ---
        self.layer_list = QListWidget(self)
        self.layer_list.setAlternatingRowColors(True)
        self.layer_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.layer_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.layer_list.setDragEnabled(True)
        self.layer_list.model().rowsMoved.connect(self._on_layer_rows_moved)
        self.layer_list.itemChanged.connect(self._on_layer_item_changed)
        self.layer_list.currentRowChanged.connect(self._on_layer_selection_changed)
        self.layer_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self._open_layer_context_menu)
        layout.addWidget(self.layer_list, 1)

        # --- Opacity slider ---
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Saydamlik"))

        self.layer_opacity = QSlider(Qt.Orientation.Horizontal, self)
        self.layer_opacity.setRange(0, 100)
        self.layer_opacity.setValue(100)
        self.layer_opacity.valueChanged.connect(self._on_layer_opacity_changed)
        opacity_row.addWidget(self.layer_opacity, 1)

        self.layer_opacity_value = QLabel("%100")
        self.layer_opacity_value.setMinimumWidth(46)
        opacity_row.addWidget(self.layer_opacity_value)
        layout.addLayout(opacity_row)

        hint = QLabel("\u2b06\u2b07 surukleyerek veya butonlarla siralama degistirin.\n"
                      "\u2611 kutusu ile gorunurlugu acin/kapatin.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(hint)

        return panel

    def _reset_layer_stack(self) -> None:
        for layer in self.layers:
            if layer.kind == "raster":
                self.scene.removeItem(layer.item)
                if layer.detail_item is not None:
                    self.scene.removeItem(layer.detail_item)

        self._extra_layer_counter = 0
        self._clear_detail_view()
        self.base_item.setVisible(True)
        self.base_item.setOpacity(1.0)
        self.mask_item.setVisible(True)
        self.mask_item.setOpacity(1.0)

        self.layers = [
            LayerState(
                key=LAYER_KEY_MASK,
                name="\U0001f534 Maske",
                item=self.mask_item,
                kind="mask",
                opacity=1.0,
                visible=True,
            ),
            LayerState(
                key=LAYER_KEY_BASE,
                name="\U0001f5bc  Ana Goruntu",
                item=self.base_item,
                kind="base",
                opacity=1.0,
                visible=True,
            ),
        ]
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=LAYER_KEY_MASK)

    def _find_layer(self, key: str) -> Optional[LayerState]:
        for layer in self.layers:
            if layer.key == key:
                return layer
        return None

    def _selected_layer_key(self) -> Optional[str]:
        item = self.layer_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value is not None else None

    def _selected_layer_row(self) -> int:
        key = self._selected_layer_key()
        if key is None:
            return -1
        for index, layer in enumerate(self.layers):
            if layer.key == key:
                return index
        return -1

    def _selected_layer(self) -> Optional[LayerState]:
        key = self._selected_layer_key()
        if key is None:
            return None
        return self._find_layer(key)

    def _layer_detail_item(self, layer: LayerState) -> Optional[QGraphicsPixmapItem]:
        if layer.key == LAYER_KEY_BASE:
            return self.base_detail_item
        if layer.key == LAYER_KEY_MASK:
            return self.mask_detail_item
        return layer.detail_item

    def _detail_items(self) -> list[QGraphicsPixmapItem]:
        items = [self.base_detail_item, self.mask_detail_item]
        for layer in self.layers:
            if layer.kind == "raster" and layer.detail_item is not None:
                items.append(layer.detail_item)
        return items

    def _apply_layer_order(self) -> None:
        total = len(self.layers)
        for idx, layer in enumerate(self.layers):
            layer.item.setZValue(float(total - idx))
            layer.item.setVisible(layer.visible)
            layer.item.setOpacity(layer.opacity)
        self._sync_detail_layer_state()

    def _sync_detail_layer_state(self) -> None:
        managed_items: set[int] = set()
        for layer in self.layers:
            detail_item = self._layer_detail_item(layer)
            if detail_item is None:
                continue
            managed_items.add(id(detail_item))
            detail_item.setZValue(layer.item.zValue() + 0.25)
            detail_item.setOpacity(layer.opacity)
            detail_item.setVisible(layer.visible and not detail_item.pixmap().isNull())
        for item in self._detail_items():
            if id(item) not in managed_items:
                item.setVisible(False)

    def _clear_detail_view(self) -> None:
        self._detail_timer.stop()
        self._detail_cache_key = None
        self._set_detail_loading(None)
        for item in self._detail_items():
            item.setPixmap(QPixmap())
            item.setTransform(QTransform())
            item.setPos(0.0, 0.0)
            item.setVisible(False)

    def _schedule_detail_refresh(self, *_args) -> None:
        if self.s is None:
            self._clear_detail_view()
            return
        self._detail_timer.start(DETAIL_REFRESH_DELAY_MS)

    def _viewport_preview_rect(self) -> Optional[tuple[int, int, int, int]]:
        if self.s is None:
            return None
        visible_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        px0 = max(0, int(math.floor(visible_rect.left())))
        py0 = max(0, int(math.floor(visible_rect.top())))
        px1 = min(self.s.preview_w, int(math.ceil(visible_rect.right())))
        py1 = min(self.s.preview_h, int(math.ceil(visible_rect.bottom())))
        if px1 <= px0 or py1 <= py0:
            return None
        return px0, py0, px1, py1

    def _apply_detail_pixmap(
        self,
        item: QGraphicsPixmapItem,
        image: QImage,
        preview_rect: tuple[int, int, int, int],
    ) -> None:
        px0, py0, px1, py1 = preview_rect
        scene_w = max(1.0, float(px1 - px0))
        scene_h = max(1.0, float(py1 - py0))
        pixmap = QPixmap.fromImage(image)
        item.setPixmap(pixmap)
        item.setPos(float(px0), float(py0))
        if pixmap.width() > 0 and pixmap.height() > 0:
            item.setTransformOriginPoint(0.0, 0.0)
            item.setTransform(
                QTransform.fromScale(
                    scene_w / float(pixmap.width()),
                    scene_h / float(pixmap.height()),
                )
            )

    def _render_layer_detail_rgb(
        self,
        layer: LayerState,
        preview_rect: tuple[int, int, int, int],
        target_w: int,
        target_h: int,
    ) -> Optional[np.ndarray]:
        if (
            self.s is None
            or layer.source_path is None
            or layer.bands is None
            or layer.preview_stretch_bounds is None
        ):
            return None

        try:
            _window, out_w, out_h, detail_transform = self.s.resolve_detail_request(
                preview_rect,
                target_w,
                target_h,
            )
        except Exception:
            return None

        try:
            with rasterio.open(layer.source_path) as ds:
                rgb, _ = _read_rgb_aligned_to_grid(
                    ds,
                    layer.bands,
                    out_w,
                    out_h,
                    dst_transform=detail_transform,
                    dst_crs=self.s.src.crs,
                    stretch_bounds=layer.preview_stretch_bounds,
                )
        except Exception:
            return None
        return rgb

    def _refresh_detail_view(self) -> None:
        if self._detail_refresh_in_progress:
            return
        if self.s is None:
            self._clear_detail_view()
            return

        zoom = float(self.view.transform().m11())
        if max(self.s.scale_x, self.s.scale_y) <= DETAIL_NATIVE_SCALE_THRESHOLD or zoom < DETAIL_MIN_ZOOM:
            self._clear_detail_view()
            return

        preview_rect = self._viewport_preview_rect()
        if preview_rect is None:
            self._clear_detail_view()
            return

        px0, py0, px1, py1 = preview_rect
        dpr = max(1.0, float(self.devicePixelRatioF()))
        target_w = max(
            1,
            min(
                DETAIL_MAX_OUTPUT_SIDE,
                int(math.ceil((px1 - px0) * zoom * dpr)),
            ),
        )
        target_h = max(
            1,
            min(
                DETAIL_MAX_OUTPUT_SIDE,
                int(math.ceil((py1 - py0) * zoom * dpr)),
            ),
        )
        base_layer = self._find_layer(LAYER_KEY_BASE)
        mask_layer = self._find_layer(LAYER_KEY_MASK)
        extra_layers_state = tuple(
            (
                layer.key,
                bool(layer.visible),
                round(float(layer.opacity), 4),
                str(layer.source_path) if layer.source_path is not None else "",
                layer.bands,
            )
            for layer in self.layers
            if layer.kind == "raster"
        )
        cache_key = (
            preview_rect,
            target_w,
            target_h,
            int(self.s.render_revision),
            bool(base_layer.visible if base_layer is not None else False),
            bool(mask_layer.visible if mask_layer is not None else False),
            extra_layers_state,
        )
        if cache_key == self._detail_cache_key:
            return

        visible_extra_layers = sum(1 for layer in self.layers if layer.kind == "raster" and layer.visible)
        show_loading = visible_extra_layers > 0 or (target_w * target_h) >= 1_000_000
        loading_text = (
            f"Detay yukleniyor ({visible_extra_layers} ek katman)..."
            if visible_extra_layers > 0
            else "Detay yukleniyor..."
        )

        self._detail_refresh_in_progress = True
        if show_loading:
            self._set_detail_loading(loading_text, flush=True)
        try:
            try:
                rgb, rgba = self.s.render_detail_patch(preview_rect, target_w, target_h)
            except Exception:
                self._clear_detail_view()
                return

            self._apply_detail_pixmap(self.base_detail_item, qimage_from_rgb(rgb), preview_rect)
            self._apply_detail_pixmap(self.mask_detail_item, qimage_from_rgba(rgba), preview_rect)
            for layer in self.layers:
                if layer.kind != "raster" or layer.detail_item is None:
                    continue
                layer_rgb = self._render_layer_detail_rgb(layer, preview_rect, target_w, target_h)
                if layer_rgb is None:
                    layer.detail_item.setPixmap(QPixmap())
                    layer.detail_item.setTransform(QTransform())
                    layer.detail_item.setPos(0.0, 0.0)
                    continue
                self._apply_detail_pixmap(layer.detail_item, qimage_from_rgb(layer_rgb), preview_rect)
            self._detail_cache_key = cache_key
            self._sync_detail_layer_state()
        finally:
            self._detail_refresh_in_progress = False
            if show_loading:
                self._set_detail_loading(None)

    def _rebuild_layer_list(self, select_key: Optional[str] = None) -> None:
        if select_key is None:
            selected = self._selected_layer()
            select_key = selected.key if selected is not None else None

        self._layers_updating_ui = True
        self.layer_list.clear()

        for layer in self.layers:
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.key)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            # Enable drag for all layers
            flags |= Qt.ItemFlag.ItemIsDragEnabled
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self.layer_list.addItem(item)

        if self.layer_list.count() > 0:
            target_row = 0
            if select_key is not None:
                for row in range(self.layer_list.count()):
                    key = self.layer_list.item(row).data(Qt.ItemDataRole.UserRole)
                    if key == select_key:
                        target_row = row
                        break
            self.layer_list.setCurrentRow(target_row)

        self._layers_updating_ui = False
        self._sync_layer_controls()

    def _on_layer_rows_moved(self, *_args) -> None:
        """QListWidget drag-reorder sonrasi layer sirasini senkronize et."""
        if self._layers_updating_ui:
            return
        # Reconstruct self.layers from the current QListWidget order
        key_to_layer = {layer.key: layer for layer in self.layers}
        new_order: list[LayerState] = []
        for row in range(self.layer_list.count()):
            key = self.layer_list.item(row).data(Qt.ItemDataRole.UserRole)
            layer = key_to_layer.get(str(key))
            if layer is not None:
                new_order.append(layer)
        if len(new_order) == len(self.layers):
            self.layers = new_order
        self._apply_layer_order()
        self._sync_layer_controls()

    def _open_layer_context_menu(self, _pos) -> None:
        global_pos = QCursor.pos()
        local_pos = self.layer_list.viewport().mapFromGlobal(global_pos)
        item = self.layer_list.itemAt(local_pos)
        if item is not None:
            self.layer_list.setCurrentItem(item)

        selected = self._selected_layer()
        has_session = self.s is not None
        row = self._selected_layer_row()

        can_move_up = has_session and selected is not None and row > 0
        can_move_down = has_session and selected is not None and 0 <= row < (len(self.layers) - 1)
        can_remove = has_session and selected is not None and selected.kind == "raster"
        can_toggle_visibility = has_session and selected is not None

        menu = QMenu(self.layer_list)
        act_add = menu.addAction("Katman Ekle")
        act_add.setEnabled(has_session)
        act_add.triggered.connect(self.add_visual_layer)

        menu.addSeparator()

        act_up = menu.addAction("Yukari tasi")
        act_up.setEnabled(can_move_up)
        act_up.triggered.connect(self.move_selected_layer_up)

        act_down = menu.addAction("Asagi tasi")
        act_down.setEnabled(can_move_down)
        act_down.triggered.connect(self.move_selected_layer_down)

        act_remove = menu.addAction("Katman Sil")
        act_remove.setEnabled(can_remove)
        act_remove.triggered.connect(self.remove_selected_layer)

        menu.addSeparator()

        act_visible = menu.addAction("Gorunur")
        act_visible.setCheckable(True)
        act_visible.setChecked(bool(selected.visible) if selected is not None else False)
        act_visible.setEnabled(can_toggle_visibility)
        act_visible.toggled.connect(self._set_selected_layer_visible)

        menu.exec(global_pos)

    def _set_selected_layer_visible(self, visible: bool) -> None:
        row = self._selected_layer_row()
        if row < 0:
            return
        item = self.layer_list.item(row)
        if item is None:
            return
        target_state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
        if item.checkState() == target_state:
            return
        item.setCheckState(target_state)

    def _sync_layer_controls(self) -> None:
        selected = self._selected_layer()
        has_session = self.s is not None

        self.btn_add_layer.setEnabled(has_session)

        row = self._selected_layer_row()
        can_move_up = has_session and selected is not None and row > 0
        can_move_down = has_session and selected is not None and 0 <= row < (len(self.layers) - 1)
        self.btn_layer_up.setEnabled(can_move_up)
        self.btn_layer_down.setEnabled(can_move_down)

        is_raster = has_session and selected is not None and selected.kind == "raster"
        self.btn_remove_layer.setEnabled(is_raster)
        if hasattr(self, "act_remove_layer"):
            self.act_remove_layer.setEnabled(is_raster)
        if hasattr(self, "act_add_layer"):
            self.act_add_layer.setEnabled(has_session)

        self.layer_opacity.setEnabled(has_session and selected is not None)
        if selected is None:
            self.layer_opacity.blockSignals(True)
            self.layer_opacity.setValue(100)
            self.layer_opacity.blockSignals(False)
            self.layer_opacity_value.setText("%100")
            return

        opacity_pct = int(round(selected.opacity * 100.0))
        self.layer_opacity.blockSignals(True)
        self.layer_opacity.setValue(opacity_pct)
        self.layer_opacity.blockSignals(False)
        self.layer_opacity_value.setText(f"%{opacity_pct}")

    def _on_layer_item_changed(self, item: QListWidgetItem) -> None:
        if self._layers_updating_ui:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        layer = self._find_layer(str(key))
        if layer is None:
            return
        layer.visible = item.checkState() == Qt.CheckState.Checked
        layer.item.setVisible(layer.visible)
        self._sync_detail_layer_state()

    def _on_layer_selection_changed(self, _row: int) -> None:
        if self._layers_updating_ui:
            return
        self._sync_layer_controls()

    def _on_layer_opacity_changed(self, value: int) -> None:
        selected = self._selected_layer()
        if selected is None:
            self.layer_opacity_value.setText("%100")
            return
        opacity = max(0.0, min(1.0, float(value) / 100.0))
        selected.opacity = opacity
        selected.item.setOpacity(opacity)
        self.layer_opacity_value.setText(f"%{value}")
        self._sync_detail_layer_state()

    def move_selected_layer_up(self) -> None:
        row = self._selected_layer_row()
        if row <= 0:
            return
        layer = self.layers.pop(row)
        self.layers.insert(row - 1, layer)
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=layer.key)

    def move_selected_layer_down(self) -> None:
        row = self._selected_layer_row()
        if row < 0 or row >= len(self.layers) - 1:
            return
        layer = self.layers.pop(row)
        self.layers.insert(row + 1, layer)
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=layer.key)

    def add_visual_layer(self) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Katman olarak GeoTIFF sec",
            "",
            "GeoTIFF (*.tif *.tiff);;All (*.*)",
        )
        if not path:
            return
        layer_path = Path(path).expanduser()
        bands_raw = self._choose_bands_raw(layer_path, self.bands_raw)
        if bands_raw is None:
            return

        try:
            with rasterio.open(layer_path) as ds:
                bands = parse_bands(bands_raw, ds.count)
                rgb, stretch_bounds = _read_rgb_aligned_to_grid(
                    ds,
                    bands,
                    self.s.preview_w,
                    self.s.preview_h,
                    dst_transform=self.s.preview_transform,
                    dst_crs=self.s.src.crs,
                )
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Katman okunamadi:\n{exc}")
            return

        item = QGraphicsPixmapItem()
        item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        item.setPixmap(QPixmap.fromImage(qimage_from_rgb(rgb)))
        self.scene.addItem(item)
        detail_item = QGraphicsPixmapItem()
        detail_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        detail_item.setVisible(False)
        self.scene.addItem(detail_item)

        self._extra_layer_counter += 1
        key = f"extra_{self._extra_layer_counter}"
        layer = LayerState(
            key=key,
            name=f"{layer_path.name} ({bands_raw})",
            item=item,
            kind="raster",
            source_path=layer_path,
            opacity=1.0,
            visible=True,
            bands=bands,
            preview_stretch_bounds=stretch_bounds,
            detail_item=detail_item,
        )

        insert_idx = 1 if self.layers and self.layers[0].key == LAYER_KEY_MASK else 0
        self.layers.insert(insert_idx, layer)
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=key)
        self._schedule_detail_refresh()

    def remove_selected_layer(self) -> None:
        row = self._selected_layer_row()
        if row < 0 or row >= len(self.layers):
            return
        layer = self.layers[row]
        if layer.kind != "raster":
            QMessageBox.information(self, APP_TITLE, "Ana goruntu ve maske katmani silinemez.")
            return
        self.scene.removeItem(layer.item)
        if layer.detail_item is not None:
            self.scene.removeItem(layer.detail_item)
        self.layers.pop(row)
        self._apply_layer_order()
        if self.layers:
            next_row = min(row, len(self.layers) - 1)
            self._rebuild_layer_list(select_key=self.layers[next_row].key)
        else:
            self._rebuild_layer_list()
        self._schedule_detail_refresh()

    def _apply_mask_values(self, positive: int, negative: int) -> None:
        pos = max(1, min(int(positive), 255))
        neg = max(0, min(int(negative), 255))
        if pos == neg:
            QMessageBox.warning(self, APP_TITLE, "Secilen ve secilmeyen deger ayni olamaz.")
            return

        self.positive_value = pos
        self.negative_value = neg

        if self.s is not None:
            try:
                changed = self.s.set_class_values(pos, neg)
            except Exception as exc:
                QMessageBox.critical(self, APP_TITLE, f"Maske degeri guncellenemedi:\n{exc}")
                return
            if changed:
                self.refresh_overlay()
                return

        self.update_status()

    def open_mask_settings_dialog(self) -> None:
        dlg = MaskValuesDialog(self.positive_value, self.negative_value, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        pos, neg = dlg.values()
        self._apply_mask_values(pos, neg)

    def _set_actions_enabled(self, enabled: bool) -> None:
        for act in (
            self.act_open_labels,
            self.act_save,
            self.act_save_as,
            self.act_export_dataset,
            self.act_add_layer,
            self.act_remove_layer,
            self.act_draw,
            self.act_erase,
            self.act_square,
            self.act_undo,
            self.act_clear,
            self.act_reset,
            self.act_fit,
            self.act_invert,
        ):
            act.setEnabled(enabled)
        self._sync_layer_controls()

    def show_empty_state(self) -> None:
        self._reset_layer_stack()
        self.base_item.setPixmap(QPixmap())
        self.mask_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF(0, 0, 800, 600))
        self.view.set_image_size(800, 600)
        self.view.resetTransform()
        # Empty state text
        if not hasattr(self, '_empty_text'):
            self._empty_text = self.scene.addText("")
        self._empty_text.setPlainText(
            "📂  Dosya Aç  veya  GeoTIFF sürükle-bırak ile başlayın\n\n"
            "Kısayollar:  Ctrl+O  Aç  |  D  Çiz  |  E  Sil  |  F  Sığdır  |  Ctrl+Z  Geri Al"
        )
        self._empty_text.setDefaultTextColor(QColor("#475569"))
        font = self._empty_text.font()
        font.setPointSize(14)
        self._empty_text.setFont(font)
        self._empty_text.setPos(120, 250)
        self._empty_text.setVisible(True)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Araçlar", self)
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())  # keep default icon size
        self.addToolBar(tb)

        # --- File actions ---
        self.act_open = QAction("📂 Aç", self)
        self.act_open.setToolTip("Girdi GeoTIFF dosyası aç  (Ctrl+O)")
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_input)
        tb.addAction(self.act_open)

        self.act_open_labels = QAction("🗂️ Etiket Aç", self)
        self.act_open_labels.setToolTip("Mevcut maske veya GPKG etiketten devam et")
        self.act_open_labels.triggered.connect(self.open_existing_labels)
        tb.addAction(self.act_open_labels)

        self.act_save = QAction("💾 Kaydet", self)
        self.act_save.setToolTip("Maskeyi kaydet  (Ctrl+S)")
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save)
        tb.addAction(self.act_save)

        self.act_save_as = QAction("📄 Farklı Kaydet", self)
        self.act_save_as.setToolTip("Maskeyi farklı konuma kaydet  (Ctrl+Shift+S)")
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.save_as)
        tb.addAction(self.act_save_as)

        self.act_export_dataset = QAction("🧩 Tile Dataset", self)
        self.act_export_dataset.setToolTip("Positive/Negative tile dataset uret")
        self.act_export_dataset.triggered.connect(self.open_tile_dataset_export_dialog)
        tb.addAction(self.act_export_dataset)

        self.act_add_layer = QAction("🧱 Katman Ekle", self)
        self.act_add_layer.setToolTip("Ek bir raster katman ekle")
        self.act_add_layer.triggered.connect(self.add_visual_layer)
        tb.addAction(self.act_add_layer)

        self.act_remove_layer = QAction("➖ Katman Sil", self)
        self.act_remove_layer.setToolTip("Secili ekstra katmani sil")
        self.act_remove_layer.triggered.connect(self.remove_selected_layer)
        tb.addAction(self.act_remove_layer)

        self.act_mask_settings = QAction("⚙️ Maske Ayarlari", self)
        self.act_mask_settings.setToolTip("Secilen/secilmeyen piksel degerlerini acilan panelde ayarla")
        self.act_mask_settings.triggered.connect(self.open_mask_settings_dialog)
        tb.addAction(self.act_mask_settings)
        tb.addSeparator()

        # --- Mode actions (exclusive group) ---
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)

        self.act_draw = QAction("✏️ Çiz", self)
        self.act_draw.setToolTip("Çizim modu – pozitif alan işaretle  (D)")
        self.act_draw.setShortcut("D")
        self.act_draw.setCheckable(True)
        self.act_draw.setChecked(True)
        self.act_draw.triggered.connect(lambda: self.set_mode("draw"))
        mode_group.addAction(self.act_draw)
        tb.addAction(self.act_draw)

        self.act_erase = QAction("🧹 Sil", self)
        self.act_erase.setToolTip("Silme modu – işaretli alanı kaldır  (E)")
        self.act_erase.setShortcut("E")
        self.act_erase.setCheckable(True)
        self.act_erase.triggered.connect(lambda: self.set_mode("erase"))
        mode_group.addAction(self.act_erase)
        tb.addAction(self.act_erase)

        self.act_square = QAction("⬜ Kare", self)
        self.act_square.setToolTip("Kare kilidi – kare şeklinde çizim  (S)")
        self.act_square.setShortcut("S")
        self.act_square.setCheckable(True)
        self.act_square.setChecked(self.square_mode)
        self.act_square.triggered.connect(self.toggle_square)
        tb.addAction(self.act_square)
        tb.addSeparator()

        # --- History actions ---
        self.act_undo = QAction("↩️ Geri Al", self)
        self.act_undo.setToolTip("Son işlemi geri al  (Ctrl+Z)")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.undo)
        tb.addAction(self.act_undo)

        self.act_clear = QAction("🗑️ Temizle", self)
        self.act_clear.setToolTip("Tüm maskeyi sıfırla")
        self.act_clear.triggered.connect(self.clear)
        tb.addAction(self.act_clear)

        self.act_reset = QAction("🔄 Başa Dön", self)
        self.act_reset.setToolTip("İlk yüklenen maskeye geri dön")
        self.act_reset.triggered.connect(self.reset)
        tb.addAction(self.act_reset)
        tb.addSeparator()

        # --- View actions ---
        self.act_fit = QAction("🔍 Sığdır", self)
        self.act_fit.setToolTip("Görüntüyü pencereye sığdır  (F)")
        self.act_fit.setShortcut("F")
        self.act_fit.triggered.connect(self.view.fit_all)
        tb.addAction(self.act_fit)

        self.act_invert = QAction("🖱️ Wheel", self)
        self.act_invert.setToolTip("Fare tekerleği yönünü tersle  (I)")
        self.act_invert.setShortcut("I")
        self.act_invert.setCheckable(True)
        self.act_invert.triggered.connect(self.invert_wheel)
        tb.addAction(self.act_invert)

    # ------------------------------------------------------------------
    # Rich Status Bar
    # ------------------------------------------------------------------
    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        # Mode badge
        self._status_mode = QLabel()
        self._status_mode.setStyleSheet(
            "font-weight: bold; padding: 2px 10px; border-radius: 4px;"
        )
        sb.addWidget(self._status_mode)

        # Square badge
        self._status_square = QLabel()
        sb.addWidget(self._status_square)

        # Positive value label
        self._status_positive = QLabel()
        sb.addWidget(self._status_positive)

        # Zoom label
        self._status_zoom = QLabel()
        sb.addWidget(self._status_zoom)

        # Wheel label
        self._status_wheel = QLabel()
        sb.addWidget(self._status_wheel)

        # Detail loading label
        self._status_detail = QLabel()
        self._status_detail.setStyleSheet("color: #0f766e; font-weight: 600;")
        self._status_detail.setVisible(False)
        sb.addWidget(self._status_detail)

        # Spacer
        spacer = QWidget()
        spacer.setFixedWidth(30)
        sb.addWidget(spacer)

        # Stats label (permanent, right side)
        self._status_stats = QLabel()
        sb.addPermanentWidget(self._status_stats)

        # Undo count
        self._status_undo = QLabel()
        sb.addPermanentWidget(self._status_undo)

    def _set_detail_loading(self, text: Optional[str], *, flush: bool = False) -> None:
        if not hasattr(self, "_status_detail"):
            return
        clean = str(text).strip() if text else ""
        self._status_detail.setVisible(bool(clean))
        self._status_detail.setText(f"  {clean}  " if clean else "")
        if flush:
            QApplication.processEvents()

    def _show_busy_indicator(
        self,
        text: str,
        *,
        maximum: int = 0,
        value: int = 0,
    ) -> QProgressDialog:
        dlg = QProgressDialog(text, "", 0, int(maximum), self)
        dlg.setWindowTitle(APP_TITLE)
        dlg.setCancelButton(None)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(520)
        dlg.setValue(int(value))
        progress_bar = dlg.findChild(QProgressBar)
        if progress_bar is not None:
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%v / %m  (%p%)")
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _update_busy_indicator(
        self,
        dlg: Optional[QProgressDialog],
        *,
        text: str,
        value: int,
        maximum: int,
    ) -> None:
        if dlg is None:
            return
        dlg.setMaximum(int(maximum))
        dlg.setLabelText(str(text))
        dlg.setValue(int(value))
        QApplication.processEvents()

    def _hide_busy_indicator(self, dlg: Optional[QProgressDialog]) -> None:
        if dlg is None:
            return
        dlg.close()
        dlg.deleteLater()
        QApplication.processEvents()

    def _parse_export_progress_payload(self, line: str) -> Optional[dict[str, object]]:
        prefix = "PROGRESS_JSON\t"
        if not line.startswith(prefix):
            return None
        try:
            payload = json.loads(line[len(prefix):].strip())
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _progress_phase_label(self, phase: object) -> str:
        key = str(phase).strip().lower()
        if not key:
            return "Islem"
        return EXPORT_PROGRESS_PHASE_LABELS.get(key, key.replace("_", " ").title())

    def _format_export_progress_text(
        self,
        payload: dict[str, object],
        *,
        idle_seconds: Optional[int] = None,
    ) -> str:
        total = max(1, int(payload.get("total", 1)))
        current = max(0, min(int(payload.get("current", 0)), total))
        pct = int(round((100.0 * current) / total))
        phase_label = self._progress_phase_label(payload.get("phase", ""))
        message = str(payload.get("message", "")).strip()
        lines = [f"{phase_label}  |  %{pct}  |  {current}/{total}"]
        if message:
            lines.append(message)
        if idle_seconds is not None and idle_seconds > 0:
            lines.append(f"Son guncellemeden beri yeni log yok: {idle_seconds} sn")
        return "\n".join(lines)

    def _format_export_progress_status(self, payload: dict[str, object]) -> str:
        total = max(1, int(payload.get("total", 1)))
        current = max(0, min(int(payload.get("current", 0)), total))
        pct = int(round((100.0 * current) / total))
        phase_label = self._progress_phase_label(payload.get("phase", ""))
        message = str(payload.get("message", "")).strip()
        summary = f"{phase_label}: {current}/{total} (%{pct})"
        if not message:
            return summary
        return f"{summary} - {message}"

    def _default_dataset_output_dir(self, feature_mode: Optional[str] = None) -> Path:
        feature_suffix = normalize_feature_mode(feature_mode or self.model_feature_mode)
        base_dir = Path(__file__).resolve().parent
        if self.s is None:
            return base_dir / "workspace" / f"training_data_classification_{feature_suffix}"
        return (
            base_dir
            / "workspace"
            / f"training_data_classification_{sanitize_name(self.s.cfg.input_path.stem)}_{feature_suffix}"
        )

    def _ensure_mask_saved_for_export(self) -> bool:
        if self.s is None:
            return False
        try:
            # Export script'i GeoTIFF maskeyi okurken aday pencereler icin GPKG'yi de kullanabiliyor.
            # Bu ikisini export oncesi her zaman ayni oturum verisinden yeniden yazarak senkron tutuyoruz.
            self.s.save(self.s.cfg.output_path)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Export oncesi etiketler kaydedilemedi:\n{exc}")
            return False
        self.update_status()
        return True

    def _dataset_export_input_issue(
        self,
        input_path: Path,
        feature_mode: str,
        bands_raw: str,
    ) -> Optional[str]:
        normalized_mode = normalize_feature_mode(feature_mode)
        expected_len = 3 if normalized_mode == "rgb3" else 5
        try:
            band_idx = parse_int_csv(str(bands_raw), expected_len=expected_len)
        except Exception as exc:
            return f"Model bantlari okunamadi: {exc}"

        try:
            with rasterio.open(input_path) as src:
                if src.count < max(band_idx):
                    return f"Raster {src.count} bant iceriyor ama model bantlari {band_idx} istiyor."
                if normalized_mode == "topo5":
                    dsm_dtype = np.dtype(src.dtypes[int(band_idx[3]) - 1])
                    dtm_dtype = np.dtype(src.dtypes[int(band_idx[4]) - 1])
                    if (
                        np.issubdtype(dsm_dtype, np.integer)
                        and np.issubdtype(dtm_dtype, np.integer)
                        and dsm_dtype.itemsize <= 1
                        and dtm_dtype.itemsize <= 1
                    ):
                        return (
                            "Secilen DSM/DTM bantlari 8-bit gorunuyor "
                            f"({dsm_dtype}/{dtm_dtype}).\n\n"
                            "Bu durumda yukseklik verisi 0-255'e ezilmis olabilir; "
                            "SVF, SLRM ve benzeri topografik bantlar guvenilir uretilemez.\n\n"
                            "Float32 DSM/DTM iceren dogru 5-band GeoTIFF ile tekrar deneyin."
                        )
        except Exception as exc:
            return str(exc)
        return None

    def _warn_dataset_export_input_if_needed(self, input_path: Path) -> None:
        issue = self._dataset_export_input_issue(
            input_path,
            self.model_feature_mode,
            self.model_bands_raw,
        )
        if not issue:
            return
        warning_key = (
            str(input_path.resolve()),
            str(self.model_feature_mode),
            str(self.model_bands_raw),
        )
        if warning_key in self._shown_dataset_input_warnings:
            return
        self._shown_dataset_input_warnings.add(warning_key)
        QMessageBox.warning(
            self,
            APP_TITLE,
            "Bu raster acildi ancak mevcut model bant ayariyla tile dataset export sorunlu olabilir.\n\n"
            f"Girdi:\n{input_path}\n\n"
            f"Feature modu:\n{self.model_feature_mode}\n\n"
            f"Model bantlari:\n{self.model_bands_raw}\n\n"
            f"Detay:\n{issue}",
        )

    def _preflight_dataset_export(self, options: dict[str, object]) -> bool:
        if self.s is None:
            return False
        issue = self._dataset_export_input_issue(
            self.s.cfg.input_path,
            str(options["feature_mode"]),
            str(options["bands_raw"]),
        )
        if issue:
            QMessageBox.critical(self, APP_TITLE, f"Dataset export on kontrolu basarisiz:\n{issue}")
            return False
        return True

    def _build_dataset_export_command(self, options: dict[str, object]) -> list[str]:
        if self.s is None:
            raise RuntimeError("Aktif oturum yok")
        script_path = Path(__file__).resolve().with_name(DATASET_EXPORT_SCRIPT)
        if not script_path.exists():
            raise FileNotFoundError(f"Dataset export script bulunamadi: {script_path}")
        output_dir = Path(options["output_dir"]).expanduser().resolve()
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--pair",
            str(self.s.cfg.input_path),
            str(self.s.cfg.output_path),
            "--output-dir",
            str(output_dir),
            "--tile-size",
            str(int(options["tile_size"])),
            "--overlap",
            str(int(options["overlap"])),
            "--feature-mode",
            str(options["feature_mode"]),
            "--bands",
            str(options["bands_raw"]),
            "--sampling-mode",
            str(options["sampling_mode"]),
            "--positive-ratio-threshold",
            str(float(options["positive_ratio_threshold"])),
            "--negative-to-positive-ratio",
            str(float(options["negative_to_positive_ratio"])),
            "--valid-ratio-threshold",
            str(float(options["valid_ratio_threshold"])),
            "--train-negative-keep-ratio",
            str(float(options["train_negative_keep_ratio"])),
            "--format",
            str(options["format"]),
            "--num-workers",
            str(int(options["num_workers"])),
            "--derivative-cache-mode",
            str(options["derivative_cache_mode"]),
        ]
        if str(options["derivative_cache_dir"]).strip():
            cmd.extend(
                [
                    "--derivative-cache-dir",
                    str(options["derivative_cache_dir"]).strip(),
                ]
            )
        if bool(options["overwrite"]):
            cmd.append("--overwrite")
        return cmd

    def open_tile_dataset_export_dialog(self) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        dlg = TileDatasetExportDialog(
            default_output_dir=self._default_dataset_output_dir(self.model_feature_mode),
            default_feature_mode=self.model_feature_mode,
            default_bands_raw=self.model_bands_raw,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        values = dlg.values()
        self.model_feature_mode = str(values["feature_mode"])
        self.model_bands_raw = str(values["bands_raw"])
        self.export_tile_dataset(values)

    def export_tile_dataset(self, options: dict[str, object]) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        if not self._ensure_mask_saved_for_export():
            return
        if not self._preflight_dataset_export(options):
            return
        import queue
        import threading
        import time

        try:
            cmd = self._build_dataset_export_command(options)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Export komutu hazirlanamadi:\n{exc}")
            return

        busy = self._show_busy_indicator("Tile dataset hazirlaniyor...", maximum=1, value=0)
        output_lines: list[str] = []
        export_env = os.environ.copy()
        export_env.setdefault("PYTHONIOENCODING", "utf-8")
        export_env.setdefault("PYTHONUTF8", "1")
        try:
            process = subprocess.Popen(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=export_env,
            )
        except Exception as exc:
            self._hide_busy_indicator(busy)
            QMessageBox.critical(self, APP_TITLE, f"Dataset export baslatilamadi:\n{exc}")
            return

        output_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        reader_errors: list[str] = []

        def _reader() -> None:
            try:
                if process.stdout is None:
                    return
                for raw_line in process.stdout:
                    output_queue.put(raw_line)
            except Exception as exc:
                reader_errors.append(str(exc))
            finally:
                output_queue.put(None)

        reader_thread = threading.Thread(
            target=_reader,
            name="dataset-export-output-reader",
            daemon=True,
        )
        reader_thread.start()

        try:
            stream_closed = process.stdout is None
            last_output_at = time.monotonic()
            last_heartbeat_at = last_output_at
            last_progress_payload: Optional[dict[str, object]] = None
            while True:
                drained_any = False
                while True:
                    try:
                        raw_line = output_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    if raw_line is None:
                        stream_closed = True
                        break
                    last_output_at = time.monotonic()
                    line = raw_line.rstrip()
                    payload = self._parse_export_progress_payload(line)
                    if payload is not None:
                        current = int(payload.get("current", 0))
                        total = max(1, int(payload.get("total", 1)))
                        current = max(0, min(current, total))
                        last_progress_payload = dict(payload)
                        self._update_busy_indicator(
                            busy,
                            text=self._format_export_progress_text(payload),
                            value=current,
                            maximum=total,
                        )
                        self.statusBar().showMessage(self._format_export_progress_status(payload))
                        continue
                    if line.strip():
                        output_lines.append(line)
                        if last_progress_payload is None:
                            self.statusBar().showMessage(trim_process_output("\n".join(output_lines), max_lines=1))

                if stream_closed and process.poll() is not None:
                    break

                now = time.monotonic()
                if not drained_any and now - last_heartbeat_at >= 1.0:
                    last_heartbeat_at = now
                    idle_seconds = int(max(0.0, now - last_output_at))
                    if busy is not None:
                        if last_progress_payload is not None:
                            heartbeat_payload = dict(last_progress_payload)
                            heartbeat_text = self._format_export_progress_text(
                                heartbeat_payload,
                                idle_seconds=idle_seconds,
                            )
                            heartbeat_status = self._format_export_progress_status(heartbeat_payload)
                            heartbeat_value = max(
                                0,
                                min(
                                    int(heartbeat_payload.get("current", 0)),
                                    max(1, int(heartbeat_payload.get("total", 1))),
                                ),
                            )
                            heartbeat_max = max(1, int(heartbeat_payload.get("total", 1)))
                        else:
                            heartbeat_text = (
                                "Hazirlik  |  %0  |  0/1\n"
                                f"Tile dataset hazirlaniyor... Son log: {idle_seconds} sn once"
                            )
                            heartbeat_status = f"Tile dataset hazirlaniyor... ({idle_seconds} sn yeni log yok)"
                            heartbeat_value = int(busy.value())
                            heartbeat_max = max(1, int(busy.maximum()))
                        self._update_busy_indicator(
                            busy,
                            text=heartbeat_text,
                            value=heartbeat_value,
                            maximum=heartbeat_max,
                        )
                        self.statusBar().showMessage(heartbeat_status)
                    else:
                        QApplication.processEvents()
                else:
                    QApplication.processEvents()
                time.sleep(0.05)
            result_code = int(process.wait())
        finally:
            reader_thread.join(timeout=1.0)
            if process.stdout is not None:
                process.stdout.close()
        if reader_errors:
            output_lines.extend(f"UYARI: cikti okuyucu hatasi: {item}" for item in reader_errors)

        output_dir = Path(options["output_dir"]).expanduser().resolve()
        details = trim_process_output("\n".join(output_lines))
        self._hide_busy_indicator(busy)
        if result_code != 0:
            message = f"Dataset export basarisiz oldu.\n\nCikti klasoru:\n{output_dir}"
            if details:
                message += f"\n\nSon log satirlari:\n{details}"
            QMessageBox.critical(self, APP_TITLE, message)
            return

        self.statusBar().showMessage(f"Tile dataset hazir: {output_dir}", 8000)
        message = (
            "Tile dataset hazirlandi.\n\n"
            f"Maske:\n{self.s.cfg.output_path}\n\n"
            f"Dataset klasoru:\n{output_dir}"
        )
        if details:
            message += f"\n\nOzet:\n{details}"
        QMessageBox.information(self, APP_TITLE, message)

    def _confirm_save_if_dirty(self) -> bool:
        if self.s is None or not self.s.dirty:
            return True
        ans = QMessageBox.question(
            self,
            APP_TITLE,
            "Kaydedilmemis degisiklikler var. Devam etmeden once kaydetmek ister misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if ans == QMessageBox.StandardButton.Cancel:
            return False
        if ans == QMessageBox.StandardButton.Yes:
            try:
                self.s.save(self.s.cfg.output_path)
            except Exception as exc:
                QMessageBox.critical(self, APP_TITLE, f"Kaydetme hatasi:\n{exc}")
                return False
        return True

    def _choose_bands_raw(self, input_path: Path, default_raw: str) -> Optional[str]:
        try:
            with rasterio.open(input_path) as tmp:
                band_count = tmp.count
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Dosya acilamadi:\n{exc}")
            return None

        if band_count == 1:
            return "1"
        if band_count >= 3:
            dlg = BandSelectionDialog(band_count, default_raw, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            return dlg.get_bands_raw()
        return "1,2"

    def open_input(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Girdi GeoTIFF seç", "", "GeoTIFF (*.tif *.tiff);;All (*.*)")
        if not path:
            return
        input_path = Path(path).expanduser()
        selected_bands = self._choose_bands_raw(input_path, self.bands_raw)
        if selected_bands is None:
            return
        self.bands_raw = selected_bands
        self.load_input(input_path)

    def open_existing_labels(self) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        if not self._confirm_save_if_dirty():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Mevcut etiket dosyasini sec",
            str(self.s.cfg.output_path.parent),
            "Etiket Dosyalari (*.gpkg *.tif *.tiff);;GeoPackage (*.gpkg);;GeoTIFF (*.tif *.tiff)",
        )
        if not path:
            return
        label_path = Path(path).expanduser()
        suffix = label_path.suffix.lower()
        existing_mask = label_path if suffix in {".tif", ".tiff"} else None
        existing_labels = label_path if suffix == ".gpkg" else None
        self.load_input(
            self.s.cfg.input_path,
            output_path=self.s.cfg.output_path,
            existing_mask=existing_mask,
            existing_labels=existing_labels,
        )

    def load_input(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        existing_mask: Optional[Path] = None,
        existing_labels: Optional[Path] = None,
    ) -> bool:
        input_path = input_path.expanduser()
        if not input_path.exists():
            QMessageBox.critical(self, APP_TITLE, f"Girdi dosyasi bulunamadi:\n{input_path}")
            return False

        out_path = output_path.expanduser() if output_path is not None else input_path.with_name(f"{input_path.stem}_ground_truth.tif")
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".tif")

        mask_path = existing_mask.expanduser() if existing_mask is not None else None
        labels_path = existing_labels.expanduser() if existing_labels is not None else None

        if mask_path is None and out_path.exists():
            mask_path = out_path
        if labels_path is None:
            auto_gpkg = companion_gpkg_path(out_path)
            if auto_gpkg.exists():
                labels_path = auto_gpkg

        if mask_path is not None and not mask_path.exists():
            QMessageBox.critical(self, APP_TITLE, f"Mevcut maske dosyasi bulunamadi:\n{mask_path}")
            return False
        if labels_path is not None and not labels_path.exists():
            QMessageBox.critical(self, APP_TITLE, f"Mevcut etiket GPKG dosyasi bulunamadi:\n{labels_path}")
            return False

        try:
            with rasterio.open(input_path) as tmp:
                bands = parse_bands(self.bands_raw, tmp.count)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Band/raster hatası:\n{exc}")
            return False

        cfg = AppConfig(
            input_path=input_path,
            output_path=out_path,
            existing_mask=mask_path,
            existing_labels=labels_path,
            preview_max_size=self.preview_max_size,
            bands=bands,
            positive_value=self.positive_value,
            negative_value=self.negative_value,
            square_mode=self.square_mode,
        )

        try:
            new_session = Session(cfg)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Oturum başlatılamadı:\n{exc}")
            return False

        self.set_session(new_session)
        self._warn_dataset_export_input_if_needed(input_path)
        return True

    def set_session(self, session: Session) -> None:
        prev_session = self.s
        self.s = session
        if prev_session is not None:
            prev_session.close()

        # Hide empty state text
        if hasattr(self, '_empty_text'):
            self._empty_text.setVisible(False)

        self.square_mode = bool(session.cfg.square_mode)
        self.act_square.setChecked(self.square_mode)
        self.view.square_mode = self.square_mode
        self.positive_value = int(session.cfg.positive_value)
        self.negative_value = int(session.cfg.negative_value)
        self.view.set_mode(self.mode)
        self.view.set_image_size(self.s.preview_w, self.s.preview_h)
        self._reset_layer_stack()

        base_layer = self._find_layer(LAYER_KEY_BASE)
        if base_layer is not None:
            base_layer.name = f"\U0001f5bc  {self.s.cfg.input_path.name}"
        self._rebuild_layer_list(select_key=LAYER_KEY_MASK)

        self._set_actions_enabled(True)
        self._set_window_title()
        self.refresh_base()
        self.refresh_overlay()
        self.view.fit_all()
        self.update_status()

    def refresh_base(self) -> None:
        if self.s is None:
            return
        self.base_item.setPixmap(QPixmap.fromImage(qimage_from_rgb(self.s.preview_rgb)))
        self.scene.setSceneRect(QRectF(0, 0, self.s.preview_w, self.s.preview_h))
        self._schedule_detail_refresh()

    def refresh_overlay(self) -> None:
        if self.s is None:
            return
        self.mask_item.setPixmap(QPixmap.fromImage(qimage_from_rgba(self.s.overlay_rgba)))
        self._schedule_detail_refresh()
        self.update_status()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.view.set_mode(mode)
        self.act_draw.setChecked(mode == "draw")
        self.act_erase.setChecked(mode == "erase")
        self.update_status()

    def toggle_square(self) -> None:
        self.square_mode = self.act_square.isChecked()
        self.view.square_mode = self.square_mode
        if self.s is not None:
            self.s.cfg.square_mode = self.square_mode
        self.update_status()

    def invert_wheel(self) -> None:
        self.view.wheel_inverted = not self.view.wheel_inverted
        self.act_invert.setChecked(self.view.wheel_inverted)
        self.update_status()

    def on_box(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.s is None:
            return
        self.s.apply_box((x0, y0, x1, y1), self.mode)
        self.refresh_overlay()

    def undo(self) -> None:
        if self.s is None:
            return
        self.s.undo()
        self.refresh_overlay()

    def clear(self) -> None:
        if self.s is None:
            return
        ans = QMessageBox.question(
            self, APP_TITLE,
            "Tüm maske sıfırlanacak. Emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.s.clear()
        self.refresh_overlay()

    def reset(self) -> None:
        if self.s is None:
            return
        ans = QMessageBox.question(
            self, APP_TITLE,
            "Maske ilk yüklenen haline döndürülecek. Emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.s.reset()
        self.refresh_overlay()

    def save(self) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        try:
            self.s.save(self.s.cfg.output_path)
            QMessageBox.information(self, APP_TITLE, f"Kaydedildi:\n{self.s.cfg.output_path}")
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Kaydetme hatasi:\n{exc}")
        self.update_status()

    def save_as(self) -> None:
        if self.s is None:
            QMessageBox.information(self, APP_TITLE, "Once bir girdi dosyasi acin.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Farkli Kaydet", str(self.s.cfg.output_path), "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".tif")
        try:
            self.s.save(p)
            QMessageBox.information(self, APP_TITLE, f"Kaydedildi:\n{p}")
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Kaydetme hatasi:\n{exc}")
        self.update_status()

    def update_status(self, *_args) -> None:
        z = self.view.transform().m11()

        # Mode badge
        if self.mode == "draw":
            self._status_mode.setText("  ✏️  ÇİZ  ")
            self._status_mode.setStyleSheet(
                "font-weight:bold; padding:2px 10px; border-radius:4px;"
                "background:#dcfce7; color:#166534; border:1px solid #86efac;"
            )
        else:
            self._status_mode.setText("  🧹  SİL  ")
            self._status_mode.setStyleSheet(
                "font-weight:bold; padding:2px 10px; border-radius:4px;"
                "background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;"
            )

        # Square badge
        sq_icon = "⬜" if self.square_mode else "▭"
        sq_text = "Kare" if self.square_mode else "Serbest"
        self._status_square.setText(f"  {sq_icon} {sq_text}  ")

        self._status_positive.setText(f"  Maske S:{self.positive_value}  B:{self.negative_value}  ")

        # Zoom
        self._status_zoom.setText(f"  🔍 {z:.0%}  ")

        # Wheel direction
        wdir = "↕ Ters" if self.view.wheel_inverted else "↕ Normal"
        self._status_wheel.setText(f"  {wdir}  ")

        if self.s is None:
            self._status_stats.setText("  📂 Dosya bekleniyor  ")
            self._status_undo.setText("")
            return

        pos, total, ratio = self.s.stats()
        self._status_stats.setText(
            f"  Pozitif: {pos:,} / {total:,}  ({ratio:.2f}%)  "
        )
        undo_n = len(self.s.history)
        self._status_undo.setText(f"  ↩️ {undo_n}  " if undo_n > 0 else "")

    # ------------------------------------------------------------------
    # Drag & Drop Support
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: "QDragEnterEvent") -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith((".tif", ".tiff")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: "QDropEvent") -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".tif", ".tiff")):
                if not self._confirm_save_if_dirty():
                    return
                input_path = Path(path)
                selected_bands = self._choose_bands_raw(input_path, self.bands_raw)
                if selected_bands is None:
                    return
                self.bands_raw = selected_bands
                self.load_input(input_path)
                return

    def closeEvent(self, event) -> None:
        if not self._confirm_save_if_dirty():
            event.ignore()
            return
        if self.s is not None:
            self.s.close()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qt tabanli GeoTIFF etiketleyici (PySide6/PyQt6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=str, default="")
    p.add_argument("--output", "-o", type=str, default="")
    p.add_argument("--existing-mask", type=str, default="")
    p.add_argument("--existing-labels", type=str, default="")
    p.add_argument("--preview-max-size", type=int, default=DEFAULT_PREVIEW_MAX_SIZE)
    p.add_argument("--bands", type=str, default=DEFAULT_PREVIEW_BANDS_RAW)
    p.add_argument("--positive-value", type=int, default=DEFAULT_POSITIVE_VALUE)
    p.add_argument("--negative-value", type=int, default=DEFAULT_NEGATIVE_VALUE)
    p.add_argument(
        "--square-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SQUARE_MODE,
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    preview_max = int(args.preview_max_size)
    positive = int(args.positive_value)
    negative = int(args.negative_value)
    if preview_max < 0:
        QMessageBox.critical(None, APP_TITLE, "--preview-max-size 0 veya pozitif olmali (0=tam cozumurluk)")
        return 1
    if not (1 <= positive <= 255):
        QMessageBox.critical(None, APP_TITLE, "--positive-value 1-255 araliginda olmali")
        return 1
    if not (0 <= negative <= 255):
        QMessageBox.critical(None, APP_TITLE, "--negative-value 0-255 araliginda olmali")
        return 1
    if positive == negative:
        QMessageBox.critical(None, APP_TITLE, "--positive-value ve --negative-value ayni olamaz")
        return 1

    win = MainWindow(
        preview_max_size=preview_max,
        bands_raw=args.bands,
        positive_value=positive,
        negative_value=negative,
        square_mode=bool(args.square_mode),
    )

    if args.input:
        input_path = Path(args.input).expanduser()
        output_path = Path(args.output).expanduser() if args.output else input_path.with_name(f"{input_path.stem}_ground_truth.tif")
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".tif")
        existing_mask = Path(args.existing_mask).expanduser() if args.existing_mask else None
        existing_labels = Path(args.existing_labels).expanduser() if args.existing_labels else None
        if not win.load_input(
            input_path,
            output_path=output_path,
            existing_mask=existing_mask,
            existing_labels=existing_labels,
        ):
            return 1

    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
