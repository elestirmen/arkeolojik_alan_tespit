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
"""

from __future__ import annotations

import argparse
import ctypes
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
    from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsView,
        QMainWindow,
        QMessageBox,
        QToolBar,
    )
    QT_BACKEND = "PySide6"
except ImportError as exc:
    _pyside_import_error = exc
    try:
        from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal as Signal
        from PyQt6.QtGui import QAction, QImage, QKeySequence, QPainter, QPen, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QFileDialog,
            QGraphicsPixmapItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsView,
            QMainWindow,
            QMessageBox,
            QToolBar,
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


def parse_bands(raw: str, count: int) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("Band listesi bos olamaz")
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


def preview_to_full_box(
    box: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    full_w: int,
    full_h: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    x0 = int(np.floor(xmin * scale_x))
    y0 = int(np.floor(ymin * scale_y))
    x1 = int(np.ceil((xmax + 1) * scale_x))
    y1 = int(np.ceil((ymax + 1) * scale_y))
    x0 = int(np.clip(x0, 0, full_w - 1))
    y0 = int(np.clip(y0, 0, full_h - 1))
    x1 = int(np.clip(x1, x0 + 1, full_w))
    y1 = int(np.clip(y1, y0 + 1, full_h))
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


class Session:
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
        self.history: list[tuple[int, int, int, int, np.ndarray, int, int, int, int, np.ndarray]] = []
        self.dirty = False

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
        rgb = np.zeros((ph, pw, 3), dtype=np.uint8)
        for i, b in enumerate(bands):
            arr = self.src.read(b, out_shape=(ph, pw), resampling=Resampling.bilinear).astype(np.float32, copy=False)
            rgb[:, :, i] = stretch_to_uint8(arr)
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

    def apply_box(self, box: tuple[int, int, int, int], mode: str) -> None:
        px0, py0, px1, py1 = box
        x0, y0, x1, y1 = preview_to_full_box(box, self.scale_x, self.scale_y, self.full_w, self.full_h)
        pxi1 = min(self.preview_w, px1 + 1)
        pyi1 = min(self.preview_h, py1 + 1)
        prev_full = self.mask_full[y0:y1, x0:x1].copy()
        prev_prev = self.mask_preview[py0:pyi1, px0:pxi1].copy()
        if mode == "draw":
            self.mask_full[y0:y1, x0:x1] = np.uint8(self.cfg.positive_value)
            self.mask_preview[py0:pyi1, px0:pxi1] = np.uint8(self.cfg.positive_value)
        else:
            self.mask_full[y0:y1, x0:x1] = np.uint8(0)
            self.mask_preview[py0:pyi1, px0:pxi1] = np.uint8(0)
        self.history.append((x0, y0, x1, y1, prev_full, px0, py0, pxi1, pyi1, prev_prev))
        self.dirty = True

    def undo(self) -> None:
        if not self.history:
            return
        x0, y0, x1, y1, prev_full, px0, py0, pxi1, pyi1, prev_prev = self.history.pop()
        self.mask_full[y0:y1, x0:x1] = prev_full
        self.mask_preview[py0:pyi1, px0:pxi1] = prev_prev
        self.dirty = True

    def clear(self) -> None:
        self.mask_full.fill(0)
        self.mask_preview.fill(0)
        self.history.clear()
        self.dirty = True

    def reset(self) -> None:
        self.mask_full[:, :] = self.initial_mask_full
        self.mask_preview[:, :] = self.initial_mask_preview
        self.history.clear()
        self.dirty = True

    def save(self, path: Path) -> None:
        profile = self.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.mask_full[np.newaxis, :, :].astype(np.uint8, copy=False))
        self.cfg.output_path = path
        self.dirty = False

    def stats(self) -> tuple[int, int, float]:
        pos = int(np.count_nonzero(self.mask_full))
        total = int(self.mask_full.size)
        ratio = (100.0 * pos / total) if total > 0 else 0.0
        return pos, total, ratio

class AnnotView(QGraphicsView):
    box_committed = Signal(int, int, int, int)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

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
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(pen)
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
            self._rect_item.setRect(QRectF(float(xmin), float(ymin), float(max(1, xmax - xmin)), float(max(1, ymax - ymin))))
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

        self.resize(1500, 950)

        self.scene = QGraphicsScene(self)
        self.view = AnnotView(self)
        self.view.setScene(self.scene)
        self.view.square_mode = self.square_mode
        self.view.mode = self.mode
        self.view.box_committed.connect(self.on_box)
        self.view.zoom_changed.connect(self.update_status)
        self.setCentralWidget(self.view)

        self.base_item = QGraphicsPixmapItem()
        self.base_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.mask_item)

        self.make_toolbar()
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

    def _set_actions_enabled(self, enabled: bool) -> None:
        for act in (
            self.act_save,
            self.act_save_as,
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

    def show_empty_state(self) -> None:
        self.base_item.setPixmap(QPixmap())
        self.mask_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF(0, 0, 1, 1))
        self.view.set_image_size(1, 1)
        self.view.resetTransform()

    def make_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        self.addToolBar(tb)

        self.act_open = QAction("Dosya Ac", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_input)
        tb.addAction(self.act_open)
        tb.addSeparator()

        self.act_save = QAction("Kaydet", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save)
        tb.addAction(self.act_save)

        self.act_save_as = QAction("Farkli Kaydet", self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.save_as)
        tb.addAction(self.act_save_as)
        tb.addSeparator()

        self.act_draw = QAction("Ciz", self)
        self.act_draw.setCheckable(True)
        self.act_draw.setChecked(True)
        self.act_draw.triggered.connect(lambda: self.set_mode("draw"))
        tb.addAction(self.act_draw)

        self.act_erase = QAction("Sil", self)
        self.act_erase.setCheckable(True)
        self.act_erase.triggered.connect(lambda: self.set_mode("erase"))
        tb.addAction(self.act_erase)

        self.act_square = QAction("Kare Kilidi", self)
        self.act_square.setCheckable(True)
        self.act_square.setChecked(self.square_mode)
        self.act_square.triggered.connect(self.toggle_square)
        tb.addAction(self.act_square)
        tb.addSeparator()

        self.act_undo = QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.undo)
        tb.addAction(self.act_undo)

        self.act_clear = QAction("Temizle", self)
        self.act_clear.triggered.connect(self.clear)
        tb.addAction(self.act_clear)

        self.act_reset = QAction("Basa Don", self)
        self.act_reset.triggered.connect(self.reset)
        tb.addAction(self.act_reset)
        tb.addSeparator()

        self.act_fit = QAction("Sigdir", self)
        self.act_fit.setShortcut("F")
        self.act_fit.triggered.connect(self.view.fit_all)
        tb.addAction(self.act_fit)

        self.act_invert = QAction("Wheel Tersle", self)
        self.act_invert.setShortcut("I")
        self.act_invert.triggered.connect(self.invert_wheel)
        tb.addAction(self.act_invert)

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

    def open_input(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Girdi GeoTIFF sec", "", "GeoTIFF (*.tif *.tiff);;All (*.*)")
        if not path:
            return
        self.load_input(Path(path).expanduser())

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
            QMessageBox.critical(self, APP_TITLE, f"Band/raste hatasi:\n{exc}")
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
            QMessageBox.critical(self, APP_TITLE, f"Oturum baslatilamadi:\n{exc}")
            return False

        self.set_session(new_session)
        return True

    def set_session(self, session: Session) -> None:
        prev_session = self.s
        self.s = session
        if prev_session is not None:
            prev_session.close()

        self.square_mode = bool(session.cfg.square_mode)
        self.act_square.setChecked(self.square_mode)
        self.view.square_mode = self.square_mode
        self.view.mode = self.mode
        self.view.set_image_size(self.s.preview_w, self.s.preview_h)

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
        rgba = np.zeros((self.s.preview_h, self.s.preview_w, 4), dtype=np.uint8)
        idx = self.s.mask_preview > 0
        rgba[idx, 0] = 255
        rgba[idx, 3] = OVERLAY_ALPHA
        self.mask_item.setPixmap(QPixmap.fromImage(qimage_from_rgba(rgba)))
        self.update_status()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.view.mode = mode
        self.act_draw.setChecked(mode == "draw")
        self.act_erase.setChecked(mode == "erase")

    def toggle_square(self) -> None:
        self.square_mode = self.act_square.isChecked()
        self.view.square_mode = self.square_mode
        if self.s is not None:
            self.s.cfg.square_mode = self.square_mode
        self.update_status()

    def invert_wheel(self) -> None:
        self.view.wheel_inverted = not self.view.wheel_inverted
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
        self.s.clear()
        self.refresh_overlay()

    def reset(self) -> None:
        if self.s is None:
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
        wdir = "ters" if self.view.wheel_inverted else "normal"
        if self.s is None:
            self.statusBar().showMessage(
                f"dosya=bekleniyor | mod={self.mode} | kare={'on' if self.square_mode else 'off'} | "
                f"zoom={z:.2f}x | wheel={wdir}"
            )
            return
        pos, total, ratio = self.s.stats()
        self.statusBar().showMessage(
            f"mod={self.mode} | kare={'on' if self.square_mode else 'off'} | zoom={z:.2f}x | wheel={wdir} | "
            f"pozitif={pos}/{total} ({ratio:.2f}%) | undo={len(self.s.history)}"
        )

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
