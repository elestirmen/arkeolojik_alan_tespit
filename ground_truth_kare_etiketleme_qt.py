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
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
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
    def __init__(self, session: Session):
        super().__init__()
        self.s = session
        self.mode = "draw"
        self.square_mode = session.cfg.square_mode

        self.setWindowTitle(f"{APP_TITLE} - {session.cfg.input_path.name} [{QT_BACKEND}]")
        self.resize(1500, 950)

        self.scene = QGraphicsScene(self)
        self.view = AnnotView(self)
        self.view.setScene(self.scene)
        self.view.set_image_size(self.s.preview_w, self.s.preview_h)
        self.view.square_mode = self.square_mode
        self.view.mode = self.mode
        self.view.box_committed.connect(self.on_box)
        self.view.zoom_changed.connect(self.update_status)
        self.setCentralWidget(self.view)

        self.base_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.mask_item)

        self.make_toolbar()
        self.refresh_base()
        self.refresh_overlay()
        self.view.fit_all()
        self.update_status()

    def make_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        self.addToolBar(tb)

        act_save = QAction("Kaydet", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save)
        tb.addAction(act_save)

        act_save_as = QAction("Farkli Kaydet", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self.save_as)
        tb.addAction(act_save_as)
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

        act_undo = QAction("Undo", self)
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self.undo)
        tb.addAction(act_undo)

        act_clear = QAction("Temizle", self)
        act_clear.triggered.connect(self.clear)
        tb.addAction(act_clear)

        act_reset = QAction("Basa Don", self)
        act_reset.triggered.connect(self.reset)
        tb.addAction(act_reset)
        tb.addSeparator()

        act_fit = QAction("Sigdir", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.view.fit_all)
        tb.addAction(act_fit)

        act_invert = QAction("Wheel Tersle", self)
        act_invert.setShortcut("I")
        act_invert.triggered.connect(self.invert_wheel)
        tb.addAction(act_invert)

    def refresh_base(self) -> None:
        self.base_item.setPixmap(QPixmap.fromImage(qimage_from_rgb(self.s.preview_rgb)))
        self.scene.setSceneRect(QRectF(0, 0, self.s.preview_w, self.s.preview_h))

    def refresh_overlay(self) -> None:
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

    def invert_wheel(self) -> None:
        self.view.wheel_inverted = not self.view.wheel_inverted
        self.update_status()

    def on_box(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.s.apply_box((x0, y0, x1, y1), self.mode)
        self.refresh_overlay()

    def undo(self) -> None:
        self.s.undo()
        self.refresh_overlay()

    def clear(self) -> None:
        self.s.clear()
        self.refresh_overlay()

    def reset(self) -> None:
        self.s.reset()
        self.refresh_overlay()

    def save(self) -> None:
        try:
            self.s.save(self.s.cfg.output_path)
            QMessageBox.information(self, APP_TITLE, f"Kaydedildi:\n{self.s.cfg.output_path}")
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Kaydetme hatasi:\n{exc}")
        self.update_status()

    def save_as(self) -> None:
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
        pos, total, ratio = self.s.stats()
        z = self.view.transform().m11()
        wdir = "ters" if self.view.wheel_inverted else "normal"
        self.statusBar().showMessage(
            f"mod={self.mode} | kare={'on' if self.square_mode else 'off'} | zoom={z:.2f}x | wheel={wdir} | "
            f"pozitif={pos}/{total} ({ratio:.2f}%) | undo={len(self.s.history)}"
        )

    def closeEvent(self, event) -> None:
        if self.s.dirty:
            ans = QMessageBox.question(
                self,
                APP_TITLE,
                "Kaydedilmemis degisiklikler var. Cikmadan once kaydetmek ister misiniz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.StandardButton.Yes:
                try:
                    self.s.save(self.s.cfg.output_path)
                except Exception as exc:
                    QMessageBox.critical(self, APP_TITLE, f"Kaydetme hatasi:\n{exc}")
                    event.ignore()
                    return
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
    p.add_argument("--preview-max-size", type=int, default=1800)
    p.add_argument("--bands", type=str, default="1,2,3")
    p.add_argument("--positive-value", type=int, default=1)
    p.add_argument("--square-mode", action="store_true")
    return p


def pick_input_if_needed(current: Optional[Path]) -> Optional[Path]:
    if current is not None:
        return current
    path, _ = QFileDialog.getOpenFileName(None, "Girdi GeoTIFF sec", "", "GeoTIFF (*.tif *.tiff);;All (*.*)")
    if not path:
        return None
    return Path(path).expanduser()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    input_path = Path(args.input).expanduser() if args.input else None
    input_path = pick_input_if_needed(input_path)
    if input_path is None:
        return 0
    if not input_path.exists():
        QMessageBox.critical(None, APP_TITLE, f"Girdi dosyasi bulunamadi:\n{input_path}")
        return 1

    output_path = Path(args.output).expanduser() if args.output else input_path.with_name(f"{input_path.stem}_ground_truth.tif")
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".tif")

    existing_mask = Path(args.existing_mask).expanduser() if args.existing_mask else None
    preview_max = int(args.preview_max_size)
    positive = int(args.positive_value)
    if preview_max <= 0:
        QMessageBox.critical(None, APP_TITLE, "--preview-max-size pozitif olmali")
        return 1
    if not (1 <= positive <= 255):
        QMessageBox.critical(None, APP_TITLE, "--positive-value 1-255 araliginda olmali")
        return 1

    try:
        with rasterio.open(input_path) as tmp:
            bands = parse_bands(args.bands, tmp.count)
    except Exception as exc:
        QMessageBox.critical(None, APP_TITLE, f"Band/raste hatasi:\n{exc}")
        return 1

    cfg = AppConfig(
        input_path=input_path,
        output_path=output_path,
        existing_mask=existing_mask,
        preview_max_size=preview_max,
        bands=bands,
        positive_value=positive,
        square_mode=bool(args.square_mode),
    )

    try:
        session = Session(cfg)
    except Exception as exc:
        QMessageBox.critical(None, APP_TITLE, f"Oturum baslatilamadi:\n{exc}")
        return 1

    win = MainWindow(session)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
