#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qt tabanli GeoTIFF etiketleme araci (PySide6 / PyQt6).

Ozellikler:
- Sol fare: rectangle cizim
- Sag fare: pan
- Tekerlek: zoom
- Draw / Erase modlari
- Undo (Ctrl+Z), clear, reset, fit
- Save / Save As (GeoTIFF mask)
- Sol panel katman yonetimi (gorunurluk, sira, saydamlik)
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

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
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import (
        QAction, QActionGroup, QBrush, QColor, QCursor, QDragEnterEvent, QDropEvent,
        QImage, QKeySequence, QPainter, QPen, QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
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
        from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal as Signal
        from PyQt6.QtGui import (
            QAction, QActionGroup, QBrush, QColor, QCursor, QDragEnterEvent, QDropEvent,
            QImage, QKeySequence, QPainter, QPen, QPixmap,
        )
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QGraphicsPixmapItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsView,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
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

APP_TITLE = "Ground Truth Kare Etiketleme (Qt)"
OVERLAY_ALPHA = 96
LAYER_KEY_BASE = "__base__"
LAYER_KEY_MASK = "__mask__"

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


def stretch_to_uint8(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    vals = arr[valid].astype(np.float32, copy=False)
    lo = float(np.percentile(vals, low))
    hi = float(np.percentile(vals, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        clipped = np.clip(vals, 0.0, 255.0)
        out[valid] = clipped.astype(np.uint8)
        return out
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    return out


def qimage_from_rgb(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return img.copy()


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()


def _read_rgb_preview(ds, bands: tuple[int, int, int], out_w: int, out_h: int) -> np.ndarray:
    """Seçilen bantlardan hedef boyutta RGB preview üret."""
    unique_bands: list[int] = []
    band_to_idx: dict[int, int] = {}
    for b in bands[:3]:
        if b not in band_to_idx:
            band_to_idx[b] = len(unique_bands)
            unique_bands.append(b)

    data = ds.read(
        unique_bands,
        out_shape=(len(unique_bands), out_h, out_w),
        resampling=Resampling.bilinear,
    )

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for ch, b in enumerate(bands[:3]):
        src_idx = band_to_idx[b]
        rgb[:, :, ch] = stretch_to_uint8(data[src_idx].astype(np.float32, copy=False))
    return rgb


def read_preview_rgb(path: Path, bands: tuple[int, int, int], out_w: int, out_h: int) -> np.ndarray:
    """Raster dosyasını hedef preview boyutunda RGB uint8'e dönüştür."""
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Hedef preview boyutu gecerli degil")
    with rasterio.open(path) as ds:
        return _read_rgb_preview(ds, bands, out_w, out_h)


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


@dataclass
class AppConfig:
    input_path: Path
    output_path: Path
    existing_mask: Optional[Path]
    preview_max_size: int
    bands: tuple[int, int, int]
    positive_value: int
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

        self.preview_rgb, self.scale_x, self.scale_y = self._build_preview(cfg.preview_max_size, cfg.bands)
        self.preview_h, self.preview_w = self.preview_rgb.shape[:2]

        self.mask_full = self._load_initial_mask(cfg.existing_mask)
        self.mask_preview = cv2.resize(
            (self.mask_full > 0).astype(np.uint8),
            (self.preview_w, self.preview_h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.mask_preview[self.mask_preview > 0] = np.uint8(cfg.positive_value)

        self.initial_mask_full = self.mask_full.copy()
        self.initial_mask_preview = self.mask_preview.copy()

        # --- Undo stack: bölgesel önceki değerler ---
        self.history: list[UndoEntry] = []
        self.dirty = False

        # --- O(1) stats counter ---
        self._pos_count = int(np.count_nonzero(self.mask_full))
        self._total = int(self.mask_full.size)

        # --- Persistent overlay RGBA buffer ---
        self.overlay_rgba = np.zeros((self.preview_h, self.preview_w, 4), dtype=np.uint8)
        self._rebuild_overlay_full()

    def close(self) -> None:
        try:
            self.src.close()
        except Exception:
            pass

    def _build_preview(self, max_size: int, bands: tuple[int, int, int]) -> tuple[np.ndarray, float, float]:
        h, w = self.src.height, self.src.width
        if max_size <= 0:
            scale = 1.0
        else:
            scale = min(1.0, float(max_size) / float(max(h, w)))
        ph = max(1, int(round(h * scale)))
        pw = max(1, int(round(w * scale)))
        rgb = _read_rgb_preview(self.src, bands, pw, ph)
        return rgb, float(w) / float(pw), float(h) / float(ph)

    def _load_initial_mask(self, mask_path: Optional[Path]) -> np.ndarray:
        if mask_path is None:
            return np.zeros((self.full_h, self.full_w), dtype=np.uint8)
        with rasterio.open(mask_path) as ds:
            if ds.width != self.full_w or ds.height != self.full_h:
                raise ValueError("Mevcut maske boyutu raster ile ayni olmali")
            mask = ds.read(1)
        out = np.zeros((self.full_h, self.full_w), dtype=np.uint8)
        out[mask > 0] = np.uint8(self.cfg.positive_value)
        return out

    # --- Overlay helpers ---
    def _rebuild_overlay_full(self) -> None:
        """Tüm overlay RGBA buffer'ını mask_preview'dan yeniden oluştur."""
        self.overlay_rgba.fill(0)
        idx = self.mask_preview > 0
        self.overlay_rgba[idx, 0] = 255
        self.overlay_rgba[idx, 3] = OVERLAY_ALPHA

    def _update_overlay_region(self, py0: int, py1: int, px0: int, px1: int) -> None:
        """Overlay RGBA buffer'ın sadece belirli bölgesini güncelle."""
        region_mask = self.mask_preview[py0:py1, px0:px1]
        region = self.overlay_rgba[py0:py1, px0:px1]
        region[:] = 0
        idx = region_mask > 0
        region[idx, 0] = 255
        region[idx, 3] = OVERLAY_ALPHA

    def apply_box(self, box: tuple[int, int, int, int], mode: str) -> None:
        px0, py0, px1, py1 = box
        x0, y0, x1, y1 = preview_to_full_box(box, self.scale_x, self.scale_y, self.full_w, self.full_h)
        pxi1 = min(self.preview_w, px1 + 1)
        pyi1 = min(self.preview_h, py1 + 1)

        val = np.uint8(self.cfg.positive_value) if mode == "draw" else np.uint8(0)
        full_view = self.mask_full[y0:y1, x0:x1]
        preview_view = self.mask_preview[py0:pyi1, px0:pxi1]

        prev_pos = int(np.count_nonzero(full_view))
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

    def undo(self) -> None:
        if not self.history:
            return
        entry = self.history.pop()
        x0, y0, x1, y1 = entry.full_box
        px0, py0, pxi1, pyi1 = entry.preview_box

        full_view = self.mask_full[y0:y1, x0:x1]
        cur_pos = int(np.count_nonzero(full_view))
        prev_pos = int(np.count_nonzero(entry.full_prev))
        full_view[:, :] = entry.full_prev
        self._pos_count += (prev_pos - cur_pos)

        self.mask_preview[py0:pyi1, px0:pxi1] = entry.preview_prev
        self._update_overlay_region(py0, pyi1, px0, pxi1)
        self.dirty = bool(self.history)

    def clear(self) -> None:
        self.mask_full.fill(0)
        self.mask_preview.fill(0)
        self.overlay_rgba.fill(0)
        self.history.clear()
        self._pos_count = 0
        self.dirty = True

    def reset(self) -> None:
        self.mask_full[:, :] = self.initial_mask_full
        self.mask_preview[:, :] = self.initial_mask_preview
        self.history.clear()
        self._pos_count = int(np.count_nonzero(self.initial_mask_full))
        self._rebuild_overlay_full()
        self.dirty = True

    def set_positive_value(self, new_value: int) -> bool:
        """Pozitif sınıf değerini güncelle ve mevcut seçili alanları yeni değere eşitle."""
        clamped = max(1, min(int(new_value), 255))
        if clamped == int(self.cfg.positive_value):
            return False

        new_val = np.uint8(clamped)
        has_selected = bool(np.any(self.mask_full > 0))
        if has_selected:
            self.mask_full[self.mask_full > 0] = new_val
            self.mask_preview[self.mask_preview > 0] = new_val
            self.initial_mask_full[self.initial_mask_full > 0] = new_val
            self.initial_mask_preview[self.initial_mask_preview > 0] = new_val
            self.dirty = True

        self.cfg.positive_value = clamped
        # Eski değer ile kaydedilmiş undo kayıtlarını tutarsızlığa düşürmemek için temizle.
        self.history.clear()
        return has_selected

    def save(self, path: Path) -> None:
        profile = self.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.mask_full[np.newaxis, :, :].astype(np.uint8, copy=False))
        self.cfg.output_path = path
        self.dirty = False

    def stats(self) -> tuple[int, int, float]:
        """O(1) istatistik – counter tabanlı."""
        ratio = (100.0 * self._pos_count / self._total) if self._total > 0 else 0.0
        return self._pos_count, self._total, ratio

class AnnotView(QGraphicsView):
    box_committed = Signal(int, int, int, int)
    zoom_changed = Signal(float)

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

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if self.wheel_inverted:
            delta = -delta
        factor = 1.2 if delta > 0 else (1.0 / 1.2)
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())

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


class MainWindow(QMainWindow):
    def __init__(
        self,
        preview_max_size: int,
        bands_raw: str,
        positive_value: int,
        square_mode: bool,
        session: Optional[Session] = None,
    ):
        super().__init__()
        self.s: Optional[Session] = None
        self.mode = "draw"
        self.square_mode = bool(square_mode)
        self.preview_max_size = int(preview_max_size)
        self.bands_raw = bands_raw
        self.positive_value = int(positive_value)

        # --- Apply light theme ---
        self.setStyleSheet(APP_STYLE)

        self.resize(1500, 950)
        self.setAcceptDrops(True)

        self.scene = QGraphicsScene(self)
        self.view = AnnotView(self)
        self.view.setScene(self.scene)
        self.view.square_mode = self.square_mode
        self.view.mode = self.mode
        self.view.box_committed.connect(self.on_box)
        self.view.zoom_changed.connect(self.update_status)

        self._layers_updating_ui = False
        self._extra_layer_counter = 0
        self.layers: list[LayerState] = []

        self.base_item = QGraphicsPixmapItem()
        self.base_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.mask_item)

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

        self._extra_layer_counter = 0
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

    def _selected_layer(self) -> Optional[LayerState]:
        row = self.layer_list.currentRow()
        if 0 <= row < len(self.layers):
            return self.layers[row]
        return None

    def _apply_layer_order(self) -> None:
        total = len(self.layers)
        for idx, layer in enumerate(self.layers):
            layer.item.setZValue(float(total - idx))
            layer.item.setVisible(layer.visible)
            layer.item.setOpacity(layer.opacity)

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

    def _sync_layer_controls(self) -> None:
        selected = self._selected_layer()
        has_session = self.s is not None

        self.btn_add_layer.setEnabled(has_session)

        row = self.layer_list.currentRow()
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

    def move_selected_layer_up(self) -> None:
        row = self.layer_list.currentRow()
        if row <= 0:
            return
        layer = self.layers.pop(row)
        self.layers.insert(row - 1, layer)
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=layer.key)

    def move_selected_layer_down(self) -> None:
        row = self.layer_list.currentRow()
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
            rgb = read_preview_rgb(layer_path, bands, self.s.preview_w, self.s.preview_h)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Katman okunamadi:\n{exc}")
            return

        item = QGraphicsPixmapItem()
        item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        item.setPixmap(QPixmap.fromImage(qimage_from_rgb(rgb)))
        self.scene.addItem(item)

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
        )

        insert_idx = 1 if self.layers and self.layers[0].key == LAYER_KEY_MASK else 0
        self.layers.insert(insert_idx, layer)
        self._apply_layer_order()
        self._rebuild_layer_list(select_key=key)

    def remove_selected_layer(self) -> None:
        row = self.layer_list.currentRow()
        if row < 0 or row >= len(self.layers):
            return
        layer = self.layers[row]
        if layer.kind != "raster":
            QMessageBox.information(self, APP_TITLE, "Ana goruntu ve maske katmani silinemez.")
            return
        self.scene.removeItem(layer.item)
        self.layers.pop(row)
        self._apply_layer_order()
        if self.layers:
            next_row = min(row, len(self.layers) - 1)
            self._rebuild_layer_list(select_key=self.layers[next_row].key)
        else:
            self._rebuild_layer_list()

    def _set_positive_spin_value(self, value: int) -> None:
        if not hasattr(self, "spin_positive"):
            return
        clamped = max(1, min(int(value), 255))
        self.spin_positive.blockSignals(True)
        self.spin_positive.setValue(clamped)
        self.spin_positive.blockSignals(False)

    def on_positive_value_changed(self, value: int) -> None:
        clamped = max(1, min(int(value), 255))
        self.positive_value = clamped
        if self.s is not None:
            self.s.set_positive_value(clamped)
        self.update_status()

    def _set_actions_enabled(self, enabled: bool) -> None:
        for act in (
            self.act_save,
            self.act_save_as,
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

        self.act_add_layer = QAction("🧱 Katman Ekle", self)
        self.act_add_layer.setToolTip("Ek bir raster katman ekle")
        self.act_add_layer.triggered.connect(self.add_visual_layer)
        tb.addAction(self.act_add_layer)

        self.act_remove_layer = QAction("➖ Katman Sil", self)
        self.act_remove_layer.setToolTip("Secili ekstra katmani sil")
        self.act_remove_layer.triggered.connect(self.remove_selected_layer)
        tb.addAction(self.act_remove_layer)

        tb.addSeparator()
        self._toolbar_positive_label = QLabel("Maske Degeri")
        self._toolbar_positive_label.setStyleSheet("padding-left: 6px; padding-right: 2px;")
        tb.addWidget(self._toolbar_positive_label)

        self.spin_positive = QSpinBox(self)
        self.spin_positive.setRange(1, 255)
        self.spin_positive.setValue(self.positive_value)
        self.spin_positive.setFixedWidth(72)
        self.spin_positive.setToolTip("Secili piksellerin kayit degeri (1-255)")
        self.spin_positive.valueChanged.connect(self.on_positive_value_changed)
        tb.addWidget(self.spin_positive)
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

    def load_input(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        existing_mask: Optional[Path] = None,
    ) -> bool:
        input_path = input_path.expanduser()
        if not input_path.exists():
            QMessageBox.critical(self, APP_TITLE, f"Girdi dosyasi bulunamadi:\n{input_path}")
            return False

        out_path = output_path.expanduser() if output_path is not None else input_path.with_name(f"{input_path.stem}_ground_truth.tif")
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".tif")

        mask_path = existing_mask.expanduser() if existing_mask is not None else None
        if mask_path is not None and not mask_path.exists():
            QMessageBox.critical(self, APP_TITLE, f"Mevcut maske dosyasi bulunamadi:\n{mask_path}")
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
            preview_max_size=self.preview_max_size,
            bands=bands,
            positive_value=self.positive_value,
            square_mode=self.square_mode,
        )

        try:
            new_session = Session(cfg)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Oturum başlatılamadı:\n{exc}")
            return False

        self.set_session(new_session)
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
        self._set_positive_spin_value(self.positive_value)
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

    def refresh_overlay(self) -> None:
        if self.s is None:
            return
        self.mask_item.setPixmap(QPixmap.fromImage(qimage_from_rgba(self.s.overlay_rgba)))
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

        self._status_positive.setText(f"  Değer {self.positive_value}  ")

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
    p.add_argument("--preview-max-size", type=int, default=0)
    p.add_argument("--bands", type=str, default="1,2,3")
    p.add_argument("--positive-value", type=int, default=1)
    p.add_argument("--square-mode", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    preview_max = int(args.preview_max_size)
    positive = int(args.positive_value)
    if preview_max < 0:
        QMessageBox.critical(None, APP_TITLE, "--preview-max-size 0 veya pozitif olmali (0=tam cozumurluk)")
        return 1
    if not (1 <= positive <= 255):
        QMessageBox.critical(None, APP_TITLE, "--positive-value 1-255 araliginda olmali")
        return 1

    win = MainWindow(
        preview_max_size=preview_max,
        bands_raw=args.bands,
        positive_value=positive,
        square_mode=bool(args.square_mode),
    )

    if args.input:
        input_path = Path(args.input).expanduser()
        output_path = Path(args.output).expanduser() if args.output else input_path.with_name(f"{input_path.stem}_ground_truth.tif")
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".tif")
        existing_mask = Path(args.existing_mask).expanduser() if args.existing_mask else None
        if not win.load_input(input_path, output_path=output_path, existing_mask=existing_mask):
            return 1

    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
