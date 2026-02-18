#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground-truth mask editor (rectangle/square) for GeoTIFF files.

This tool opens a preview window and lets you draw boxes with the mouse.
The output is a single-band GeoTIFF mask:
    background = 0
    labeled area = positive_value (default: 1)

Controls:
    Left mouse drag : draw a box
    Right drag      : pan map
    Mouse wheel     : zoom in/out at cursor
    Top-right buttons : Save / Save As / Save+Exit / Quit
    d               : draw mode
    e               : erase mode
    k               : toggle square lock
    z               : reset zoom to fit
    u               : undo last box
    Ctrl+Z          : undo last box
    + / -           : zoom in / zoom out (center)
    i               : invert mouse wheel zoom direction
    c               : clear all labels
    r               : reset to initial mask
    s               : save and continue
    a               : save as
    x               : save and exit
    q / ESC         : quit (asks for unsaved changes)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover - ortama gore tkinter olmayabilir
    tk = None
    filedialog = None
    messagebox = None


# ==================== CONFIG ====================
# IDE uzerinden dogrudan "Run" ettiginizde bu degerler kullanilir.
# Komut satiri argumanlari bu degerleri ezer.
CONFIG: dict[str, object] = {
    "input": "kesif_alani.tif",
    "output": "ground_truth_manual.tif",
    "existing_mask": "",  # Ornek: "ground_truth.tif"
    "preview_max_size": 1800,  # max(width, height) for display image
    "bands": "1,2,3",  # preview bands (1-based)
    "positive_value": 1,  # labeled pixels
    "square_mode": True,  # True => lock drawing to square
    "use_launcher": True,  # acilista dosya/secenek penceresi
}
# ===============================================

WINDOW_NAME = "Ground Truth Kare Etiketleme"
OVERLAY_COLOR_BGR = np.array([0, 0, 255], dtype=np.float32)
OVERLAY_ALPHA = 0.35

ACTION_BUTTON_SPECS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("save", "Kaydet", (56, 175, 64)),
    ("save_as", "Farkli Kaydet", (66, 133, 244)),
    ("save_exit", "Kaydet+Cik", (31, 97, 141)),
    ("quit", "Cikis", (120, 120, 120)),
]


def _create_tk_root() -> Optional[Any]:
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def _pick_output_path(current_output: Path, input_path: Path) -> Optional[Path]:
    if tk is None or filedialog is None:
        return current_output

    root = _create_tk_root()
    if root is None:
        return current_output
    try:
        selected = filedialog.asksaveasfilename(
            title="Maske dosyasini kaydet",
            initialdir=str(input_path.parent if input_path.exists() else current_output.parent),
            initialfile=str(current_output.name),
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Tum dosyalar", "*.*")],
        )
    finally:
        root.destroy()

    if not selected:
        return None
    return Path(selected).expanduser()


def _confirm_unsaved_changes(output_path: Path) -> str:
    """
    Returns:
        "save" | "discard" | "cancel"
    """
    if tk is not None and messagebox is not None:
        root = _create_tk_root()
        if root is not None:
            try:
                answer = messagebox.askyesnocancel(
                    "Kaydedilmemis Degisiklikler",
                    f"Kaydedilmemis degisiklikler var.\n\nKaydetmek ister misiniz?\n\nHedef: {output_path}",
                )
            finally:
                root.destroy()
            if answer is None:
                return "cancel"
            return "save" if bool(answer) else "discard"

    print("Kaydedilmemis degisiklik var.")
    print("[s] Kaydet | [d] Kaydetmeden cik | [c] Iptal")
    while True:
        choice = input("Secim (s/d/c): ").strip().lower()
        if choice in ("s", "save"):
            return "save"
        if choice in ("d", "discard"):
            return "discard"
        if choice in ("c", "cancel"):
            return "cancel"


def _show_error_dialog(title: str, message: str) -> None:
    if tk is None or messagebox is None:
        print(f"HATA: {message}")
        return
    root = _create_tk_root()
    if root is None:
        print(f"HATA: {message}")
        return
    try:
        messagebox.showerror(title, message)
    finally:
        root.destroy()


def _build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ground_truth.tif")


def _stretch_to_uint8(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile stretch to uint8 for visualization."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out

    values = arr[valid].astype(np.float32, copy=False)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        clipped = np.clip(values, 0.0, 255.0)
        out[valid] = clipped.astype(np.uint8)
        return out

    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    return out


def _parse_bands(raw: str, band_count: int) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not parts:
        raise ValueError("Band listesi bos olamaz. Ornek: 1,2,3")

    if len(parts) == 1:
        parts = [parts[0], parts[0], parts[0]]
    elif len(parts) == 2:
        parts = [parts[0], parts[1], parts[1]]
    else:
        parts = parts[:3]

    for band in parts:
        if band < 1 or band > band_count:
            raise ValueError(f"Gecersiz band indeksi: {band} (1-{band_count})")
    return parts[0], parts[1], parts[2]


def _open_launcher_dialog(defaults: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Dost bir acilis penceresi.

    Returns:
        Baslatildiysa secilen degerleri dict olarak dondurur.
        Iptal edilirse None dondurur.
    """
    if tk is None or filedialog is None or messagebox is None:
        return None

    root = tk.Tk()
    root.title("Ground Truth Kare Etiketleme")
    root.geometry("860x420")
    root.resizable(True, False)

    result: dict[str, Any] = {}
    cancelled = {"value": True}

    input_var = tk.StringVar(value=str(defaults.get("input", "")))
    output_var = tk.StringVar(value=str(defaults.get("output", "")))
    existing_var = tk.StringVar(value=str(defaults.get("existing_mask", "")))
    bands_var = tk.StringVar(value=str(defaults.get("bands", "1,2,3")))
    preview_var = tk.StringVar(value=str(defaults.get("preview_max_size", 1800)))
    positive_var = tk.StringVar(value=str(defaults.get("positive_value", 1)))
    square_var = tk.BooleanVar(value=bool(defaults.get("square_mode", True)))

    root.columnconfigure(1, weight=1)

    def _browse_input() -> None:
        selected = filedialog.askopenfilename(
            title="Girdi GeoTIFF sec",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Tum dosyalar", "*.*")],
        )
        if selected:
            input_var.set(selected)
            if not output_var.get().strip():
                output_var.set(str(_build_default_output_path(Path(selected).expanduser())))

    def _browse_existing() -> None:
        selected = filedialog.askopenfilename(
            title="Mevcut maske sec (opsiyonel)",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Tum dosyalar", "*.*")],
        )
        if selected:
            existing_var.set(selected)

    def _browse_output() -> None:
        input_raw = input_var.get().strip()
        input_path = Path(input_raw).expanduser() if input_raw else Path("ground_truth_manual.tif")
        current_output = Path(output_var.get().strip() or _build_default_output_path(input_path))
        selected = filedialog.asksaveasfilename(
            title="Cikti maskesini kaydet",
            initialdir=str(input_path.parent) if input_path.parent.exists() else ".",
            initialfile=current_output.name,
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Tum dosyalar", "*.*")],
        )
        if selected:
            output_var.set(selected)

    def _validate_and_start() -> None:
        input_path = Path(input_var.get().strip()).expanduser()
        if not input_var.get().strip():
            messagebox.showerror("Eksik Bilgi", "Lutfen bir girdi GeoTIFF secin.")
            return
        if not input_path.exists():
            messagebox.showerror("Gecersiz Dosya", f"Girdi dosyasi bulunamadi:\n{input_path}")
            return

        existing_raw = existing_var.get().strip()
        existing_path = Path(existing_raw).expanduser() if existing_raw else None
        if existing_path is not None and not existing_path.exists():
            messagebox.showerror("Gecersiz Dosya", f"Mevcut maske bulunamadi:\n{existing_path}")
            return

        output_raw = output_var.get().strip()
        output_path = Path(output_raw).expanduser() if output_raw else _build_default_output_path(input_path)

        try:
            positive_value = int(positive_var.get().strip())
        except ValueError:
            messagebox.showerror("Gecersiz Deger", "Pozitif etiket degeri tam sayi olmali.")
            return
        if positive_value < 1 or positive_value > 255:
            messagebox.showerror("Gecersiz Deger", "Pozitif etiket degeri 1-255 araliginda olmali.")
            return

        try:
            preview_max_size = int(preview_var.get().strip())
        except ValueError:
            messagebox.showerror("Gecersiz Deger", "Onizleme boyutu tam sayi olmali.")
            return
        if preview_max_size <= 0:
            messagebox.showerror("Gecersiz Deger", "Onizleme boyutu pozitif olmali.")
            return

        if not bands_var.get().strip():
            messagebox.showerror("Gecersiz Deger", "Band listesi bos olamaz. Ornek: 1,2,3")
            return

        result.update(
            input=str(input_path),
            output=str(output_path),
            existing_mask=str(existing_path) if existing_path is not None else "",
            bands=bands_var.get().strip(),
            preview_max_size=preview_max_size,
            positive_value=positive_value,
            square_mode=bool(square_var.get()),
        )
        cancelled["value"] = False
        root.destroy()

    def _cancel() -> None:
        cancelled["value"] = True
        root.destroy()

    row = 0
    tk.Label(root, text="Girdi GeoTIFF").grid(row=row, column=0, padx=12, pady=(14, 8), sticky="w")
    tk.Entry(root, textvariable=input_var).grid(row=row, column=1, padx=8, pady=(14, 8), sticky="ew")
    tk.Button(root, text="Sec...", width=10, command=_browse_input).grid(row=row, column=2, padx=(4, 12), pady=(14, 8))

    row += 1
    tk.Label(root, text="Mevcut Maske (ops.)").grid(row=row, column=0, padx=12, pady=8, sticky="w")
    tk.Entry(root, textvariable=existing_var).grid(row=row, column=1, padx=8, pady=8, sticky="ew")
    tk.Button(root, text="Sec...", width=10, command=_browse_existing).grid(row=row, column=2, padx=(4, 12), pady=8)

    row += 1
    tk.Label(root, text="Cikti Maske").grid(row=row, column=0, padx=12, pady=8, sticky="w")
    tk.Entry(root, textvariable=output_var).grid(row=row, column=1, padx=8, pady=8, sticky="ew")
    tk.Button(root, text="Kaydet...", width=10, command=_browse_output).grid(row=row, column=2, padx=(4, 12), pady=8)

    row += 1
    tk.Label(root, text="Bandlar (RGB)").grid(row=row, column=0, padx=12, pady=8, sticky="w")
    tk.Entry(root, textvariable=bands_var).grid(row=row, column=1, padx=8, pady=8, sticky="ew")

    row += 1
    tk.Label(root, text="Pozitif Deger (1-255)").grid(row=row, column=0, padx=12, pady=8, sticky="w")
    tk.Entry(root, textvariable=positive_var).grid(row=row, column=1, padx=8, pady=8, sticky="ew")

    row += 1
    tk.Label(root, text="Onizleme Max Boyut").grid(row=row, column=0, padx=12, pady=8, sticky="w")
    tk.Entry(root, textvariable=preview_var).grid(row=row, column=1, padx=8, pady=8, sticky="ew")

    row += 1
    tk.Checkbutton(root, text="Kare Cizim Modu Acik (1:1)", variable=square_var).grid(
        row=row, column=0, columnspan=3, padx=12, pady=(10, 4), sticky="w"
    )

    row += 1
    tk.Label(
        root,
        text=(
            "Editorde: Sol fare ile ciz, sag fare ile gezin (pan), tekerlek ile zoom.\n"
            "u/Ctrl+Z: geri al | +/-: zoom | i: tekerlek yonu | s: kaydet | q/ESC: cikis"
        ),
        justify="left",
        fg="#333333",
    ).grid(row=row, column=0, columnspan=3, padx=12, pady=(4, 10), sticky="w")

    row += 1
    button_frame = tk.Frame(root)
    button_frame.grid(row=row, column=0, columnspan=3, pady=(2, 12))
    tk.Button(button_frame, text="Baslat", width=14, command=_validate_and_start).pack(side="left", padx=8)
    tk.Button(button_frame, text="Iptal", width=14, command=_cancel).pack(side="left", padx=8)

    root.protocol("WM_DELETE_WINDOW", _cancel)
    root.mainloop()

    if cancelled["value"]:
        return None
    return result


def _build_preview(
    src: rasterio.io.DatasetReader,
    bands: tuple[int, int, int],
    preview_max_size: int,
) -> tuple[np.ndarray, float, float]:
    h = src.height
    w = src.width
    scale = min(1.0, float(preview_max_size) / float(max(h, w)))
    preview_h = max(1, int(round(h * scale)))
    preview_w = max(1, int(round(w * scale)))

    rgb = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
    for i, band in enumerate(bands):
        arr = src.read(
            band,
            out_shape=(preview_h, preview_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32, copy=False)
        rgb[:, :, i] = _stretch_to_uint8(arr)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale_x = float(w) / float(preview_w)
    scale_y = float(h) / float(preview_h)
    return bgr, scale_x, scale_y


def _load_initial_mask(
    existing_mask: Optional[Path],
    height: int,
    width: int,
    positive_value: int,
) -> np.ndarray:
    if existing_mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    if not existing_mask.exists():
        raise FileNotFoundError(f"Mevcut maske bulunamadi: {existing_mask}")

    with rasterio.open(existing_mask) as mask_ds:
        if mask_ds.height != height or mask_ds.width != width:
            raise ValueError(
                "Mevcut maske boyutu girdi raster boyutuyla ayni olmali. "
                f"Mask: {mask_ds.width}x{mask_ds.height}, Raster: {width}x{height}"
            )
        mask = mask_ds.read(1)

    out = np.zeros((height, width), dtype=np.uint8)
    out[mask > 0] = np.uint8(positive_value)
    return out


def _normalize_preview_box(
    p0: tuple[int, int],
    p1: tuple[int, int],
    max_w: int,
    max_h: int,
    square_mode: bool,
) -> tuple[int, int, int, int]:
    x0, y0 = p0
    x1, y1 = p1

    if square_mode:
        dx = x1 - x0
        dy = y1 - y0
        side = max(abs(dx), abs(dy))
        if side > 0:
            x1 = x0 + side if dx >= 0 else x0 - side
            y1 = y0 + side if dy >= 0 else y0 - side

    x0 = int(np.clip(x0, 0, max_w - 1))
    x1 = int(np.clip(x1, 0, max_w - 1))
    y0 = int(np.clip(y0, 0, max_h - 1))
    y1 = int(np.clip(y1, 0, max_h - 1))

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    return xmin, ymin, xmax, ymax


def _preview_box_to_full(
    pbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    full_w: int,
    full_h: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = pbox

    x0 = int(np.floor(xmin * scale_x))
    y0 = int(np.floor(ymin * scale_y))
    x1 = int(np.ceil((xmax + 1) * scale_x))
    y1 = int(np.ceil((ymax + 1) * scale_y))

    x0 = int(np.clip(x0, 0, full_w - 1))
    y0 = int(np.clip(y0, 0, full_h - 1))
    x1 = int(np.clip(x1, x0 + 1, full_w))
    y1 = int(np.clip(y1, y0 + 1, full_h))
    return x0, y0, x1, y1


def _get_wheel_delta(flags: int) -> int:
    """Decode OpenCV wheel delta from mouse callback flags."""
    if hasattr(cv2, "getMouseWheelDelta"):
        try:
            return int(cv2.getMouseWheelDelta(flags))
        except Exception:
            pass
    raw = (int(flags) >> 16) & 0xFFFF
    if raw >= 0x8000:
        raw -= 0x10000
    return int(raw)


@dataclass
class EditorState:
    preview_bgr: np.ndarray
    mask_full: np.ndarray
    mask_preview: np.ndarray
    scale_x: float
    scale_y: float
    positive_value: int
    square_mode: bool
    mode: str = "draw"  # draw | erase
    dragging: bool = False
    start_pt: Optional[tuple[int, int]] = None
    current_pt: Optional[tuple[int, int]] = None
    history: list[tuple[int, int, int, int, np.ndarray, int, int, int, int, np.ndarray]] = field(
        default_factory=list
    )
    initial_mask: Optional[np.ndarray] = None
    initial_mask_preview: Optional[np.ndarray] = None
    zoom: float = 1.0
    zoom_min: float = 1.0
    zoom_max: float = 24.0
    view_x: float = 0.0
    view_y: float = 0.0
    panning: bool = False
    pan_start_display: Optional[tuple[int, int]] = None
    pan_start_view: Optional[tuple[float, float]] = None
    dirty: bool = False
    button_regions: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    pending_action: Optional[str] = None
    wheel_direction: int = 1  # 1:normal, -1:inverted
    src: Optional[Any] = None
    src_bands: tuple[int, int, int] = (1, 1, 1)
    native_bg_cache_key: Optional[tuple[int, int, int, int, int, int]] = None
    native_bg_cache_bgr: Optional[np.ndarray] = None

    @property
    def preview_h(self) -> int:
        return int(self.preview_bgr.shape[0])

    @property
    def preview_w(self) -> int:
        return int(self.preview_bgr.shape[1])

    @property
    def full_h(self) -> int:
        return int(self.mask_full.shape[0])

    @property
    def full_w(self) -> int:
        return int(self.mask_full.shape[1])

    def _view_size(self) -> tuple[float, float]:
        return float(self.preview_w) / self.zoom, float(self.preview_h) / self.zoom

    def clamp_view(self) -> None:
        view_w, view_h = self._view_size()
        max_x = max(0.0, float(self.preview_w) - view_w)
        max_y = max(0.0, float(self.preview_h) - view_h)
        self.view_x = float(np.clip(self.view_x, 0.0, max_x))
        self.view_y = float(np.clip(self.view_y, 0.0, max_y))

    def display_to_preview(self, x: int, y: int) -> tuple[int, int]:
        self.clamp_view()
        px = self.view_x + (float(x) / self.zoom)
        py = self.view_y + (float(y) / self.zoom)
        px_i = int(np.clip(np.round(px), 0, self.preview_w - 1))
        py_i = int(np.clip(np.round(py), 0, self.preview_h - 1))
        return px_i, py_i

    def zoom_at_display(self, x: int, y: int, wheel_delta: int) -> None:
        old_zoom = float(self.zoom)
        factor = 1.2 if wheel_delta > 0 else (1.0 / 1.2)
        new_zoom = float(np.clip(old_zoom * factor, self.zoom_min, self.zoom_max))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        focus_x = self.view_x + (float(x) / old_zoom)
        focus_y = self.view_y + (float(y) / old_zoom)
        self.zoom = new_zoom
        self.view_x = focus_x - (float(x) / new_zoom)
        self.view_y = focus_y - (float(y) / new_zoom)
        self.clamp_view()

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

    def start_pan(self, x: int, y: int) -> None:
        self.panning = True
        self.pan_start_display = (x, y)
        self.pan_start_view = (self.view_x, self.view_y)

    def update_pan(self, x: int, y: int) -> None:
        if not self.panning or self.pan_start_display is None or self.pan_start_view is None:
            return
        start_x, start_y = self.pan_start_display
        origin_x, origin_y = self.pan_start_view
        dx = float(x - start_x) / self.zoom
        dy = float(y - start_y) / self.zoom
        self.view_x = origin_x - dx
        self.view_y = origin_y - dy
        self.clamp_view()

    def stop_pan(self) -> None:
        self.panning = False
        self.pan_start_display = None
        self.pan_start_view = None

    def apply_box_from_preview(self, pbox: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = _preview_box_to_full(
            pbox=pbox,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            full_w=self.full_w,
            full_h=self.full_h,
        )
        px0, py0, px1_inc, py1_inc = pbox[0], pbox[1], pbox[2] + 1, pbox[3] + 1
        px0 = int(np.clip(px0, 0, self.preview_w - 1))
        py0 = int(np.clip(py0, 0, self.preview_h - 1))
        px1_inc = int(np.clip(px1_inc, px0 + 1, self.preview_w))
        py1_inc = int(np.clip(py1_inc, py0 + 1, self.preview_h))

        if x1 <= x0 or y1 <= y0:
            return

        previous = self.mask_full[y0:y1, x0:x1].copy()
        previous_preview = self.mask_preview[py0:py1_inc, px0:px1_inc].copy()
        if self.mode == "draw":
            self.mask_full[y0:y1, x0:x1] = np.uint8(self.positive_value)
            self.mask_preview[py0:py1_inc, px0:px1_inc] = np.uint8(self.positive_value)
        else:
            self.mask_full[y0:y1, x0:x1] = np.uint8(0)
            self.mask_preview[py0:py1_inc, px0:px1_inc] = np.uint8(0)
        self.history.append((x0, y0, x1, y1, previous, px0, py0, px1_inc, py1_inc, previous_preview))
        self.dirty = True

    def undo(self) -> None:
        if not self.history:
            return
        x0, y0, x1, y1, previous, px0, py0, px1_inc, py1_inc, previous_preview = self.history.pop()
        self.mask_full[y0:y1, x0:x1] = previous
        self.mask_preview[py0:py1_inc, px0:px1_inc] = previous_preview
        self.dirty = True

    def clear_all(self) -> None:
        if not np.any(self.mask_full):
            return
        self.mask_full.fill(0)
        self.mask_preview.fill(0)
        self.history.clear()
        self.dirty = True

    def reset_initial(self) -> None:
        if self.initial_mask is not None:
            if np.array_equal(self.mask_full, self.initial_mask):
                return
            self.mask_full[:, :] = self.initial_mask
            if self.initial_mask_preview is not None:
                self.mask_preview[:, :] = self.initial_mask_preview
            self.history.clear()
            self.dirty = True


def _draw_action_buttons(canvas: np.ndarray, state: EditorState) -> None:
    state.button_regions.clear()
    pad = 12
    button_w = 170
    button_h = 30
    gap = 8
    x1 = canvas.shape[1] - pad
    x0 = max(pad, x1 - button_w)
    y = pad

    for action, label, color in ACTION_BUTTON_SPECS:
        y0 = y
        y1 = y0 + button_h
        if y1 >= canvas.shape[0] - pad:
            break

        state.button_regions[action] = (x0, y0, x1, y1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (245, 245, 245), 1, cv2.LINE_AA)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        text_x = x0 + max(8, (button_w - text_size[0]) // 2)
        text_y = y0 + (button_h + text_size[1]) // 2 - 2
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (25, 25, 25), 1, cv2.LINE_AA)
        y = y1 + gap


def _hit_action_button(state: EditorState, x: int, y: int) -> Optional[str]:
    for action, rect in state.button_regions.items():
        x0, y0, x1, y1 = rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            return action
    return None


def _get_viewport_bounds(state: EditorState) -> tuple[int, int, int, int]:
    state.clamp_view()
    view_w_f, view_h_f = state._view_size()
    vx0 = int(np.floor(state.view_x))
    vy0 = int(np.floor(state.view_y))
    vx1 = int(np.ceil(state.view_x + view_w_f))
    vy1 = int(np.ceil(state.view_y + view_h_f))
    vx0 = int(np.clip(vx0, 0, state.preview_w - 1))
    vy0 = int(np.clip(vy0, 0, state.preview_h - 1))
    vx1 = int(np.clip(vx1, vx0 + 1, state.preview_w))
    vy1 = int(np.clip(vy1, vy0 + 1, state.preview_h))
    return vx0, vy0, vx1, vy1


def _get_native_background(
    state: EditorState,
    vx0: int,
    vy0: int,
    vx1: int,
    vy1: int,
) -> np.ndarray:
    if state.src is None:
        raise RuntimeError("Raster source not available for native rendering.")

    display_h, display_w = state.preview_h, state.preview_w
    fx0 = int(np.floor(vx0 * state.scale_x))
    fy0 = int(np.floor(vy0 * state.scale_y))
    fx1 = int(np.ceil(vx1 * state.scale_x))
    fy1 = int(np.ceil(vy1 * state.scale_y))
    fx0 = int(np.clip(fx0, 0, state.full_w - 1))
    fy0 = int(np.clip(fy0, 0, state.full_h - 1))
    fx1 = int(np.clip(fx1, fx0 + 1, state.full_w))
    fy1 = int(np.clip(fy1, fy0 + 1, state.full_h))

    fw = fx1 - fx0
    fh = fy1 - fy0
    cache_key = (fx0, fy0, fw, fh, display_w, display_h)
    if state.native_bg_cache_key == cache_key and state.native_bg_cache_bgr is not None:
        return state.native_bg_cache_bgr.copy()

    window = rasterio.windows.Window(col_off=fx0, row_off=fy0, width=fw, height=fh)
    rgb = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    for i, band in enumerate(state.src_bands):
        arr = state.src.read(
            band,
            window=window,
            out_shape=(display_h, display_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32, copy=False)
        rgb[:, :, i] = _stretch_to_uint8(arr)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    state.native_bg_cache_key = cache_key
    state.native_bg_cache_bgr = bgr.copy()
    return bgr


def _render_canvas(state: EditorState) -> np.ndarray:
    vx0, vy0, vx1, vy1 = _get_viewport_bounds(state)
    if state.src is not None and state.zoom > 1.0:
        canvas = _get_native_background(state, vx0, vy0, vx1, vy1)
    else:
        bg_crop = state.preview_bgr[vy0:vy1, vx0:vx1]
        bg_interp = cv2.INTER_LINEAR if state.zoom >= 1.0 else cv2.INTER_AREA
        canvas = cv2.resize(bg_crop, (state.preview_w, state.preview_h), interpolation=bg_interp)

    mask_crop = state.mask_preview[vy0:vy1, vx0:vx1]
    mask_canvas = cv2.resize(mask_crop, (state.preview_w, state.preview_h), interpolation=cv2.INTER_NEAREST)
    mask_idx = mask_canvas > 0
    if np.any(mask_idx):
        blended = (1.0 - OVERLAY_ALPHA) * canvas[mask_idx].astype(np.float32) + OVERLAY_ALPHA * OVERLAY_COLOR_BGR
        canvas[mask_idx] = blended.astype(np.uint8)

    if state.dragging and state.start_pt is not None and state.current_pt is not None:
        xmin, ymin, xmax, ymax = _normalize_preview_box(
            state.start_pt,
            state.current_pt,
            max_w=state.preview_w,
            max_h=state.preview_h,
            square_mode=state.square_mode,
        )
        sx = float(state.preview_w) / float(vx1 - vx0)
        sy = float(state.preview_h) / float(vy1 - vy0)
        dx0 = int(np.round((xmin - vx0) * sx))
        dy0 = int(np.round((ymin - vy0) * sy))
        dx1 = int(np.round((xmax - vx0) * sx))
        dy1 = int(np.round((ymax - vy0) * sy))
        dx0 = int(np.clip(dx0, 0, state.preview_w - 1))
        dy0 = int(np.clip(dy0, 0, state.preview_h - 1))
        dx1 = int(np.clip(dx1, 0, state.preview_w - 1))
        dy1 = int(np.clip(dy1, 0, state.preview_h - 1))
        color = (0, 255, 0) if state.mode == "draw" else (0, 255, 255)
        cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1), color, 2)

    pos_count = int(np.count_nonzero(state.mask_full))
    total = int(state.mask_full.size)
    ratio = (100.0 * pos_count / total) if total > 0 else 0.0
    dirty_text = "yes" if state.dirty else "no"
    line1 = (
        f"mode: {state.mode} | square: {'on' if state.square_mode else 'off'} "
        f"| zoom: {state.zoom:.2f}x | labels: {len(state.history)} | unsaved: {dirty_text}"
    )
    line2 = f"positive pixels: {pos_count} / {total} ({ratio:.2f}%)"
    line3 = "mouse: left draw | right drag pan | wheel zoom | d/e mode | k square | z fit"
    wheel_dir_txt = "normal" if state.wheel_direction > 0 else "inverted"
    line4 = "keys: u/Ctrl+Z undo | c clear | r reset | + / - zoom | i invert wheel"
    line5 = f"keys: s save | a save as | x save+exit | q quit | wheel-dir: {wheel_dir_txt}"
    line6 = "buttons: top-right Save / Save As / Save+Exit / Quit"
    cv2.putText(canvas, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line4, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line4, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line5, (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line5, (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line6, (12, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line6, (12, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1, cv2.LINE_AA)
    _draw_action_buttons(canvas, state)
    return canvas


def _mouse_callback(event: int, x: int, y: int, flags: int, state: EditorState) -> None:
    x = int(np.clip(x, 0, state.preview_w - 1))
    y = int(np.clip(y, 0, state.preview_h - 1))
    if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
        delta = _get_wheel_delta(flags)
        if delta != 0:
            # Zoom yonu sabit tutulur; yalnizca kullanici 'i' tusuyla tersleyebilir.
            state.zoom_at_display(x=x, y=y, wheel_delta=int(delta * state.wheel_direction))
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        state.start_pan(x, y)
        return
    if event == cv2.EVENT_MOUSEMOVE and state.panning:
        state.update_pan(x, y)
        return
    if event == cv2.EVENT_RBUTTONUP and state.panning:
        state.stop_pan()
        return

    px, py = state.display_to_preview(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        button_action = _hit_action_button(state, x, y)
        if button_action is not None:
            state.pending_action = button_action
            return
        if state.panning:
            return
        state.dragging = True
        state.start_pt = (px, py)
        state.current_pt = (px, py)
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        state.current_pt = (px, py)
    elif event == cv2.EVENT_LBUTTONUP and state.dragging:
        state.current_pt = (px, py)
        if state.start_pt is not None and state.current_pt is not None:
            pbox = _normalize_preview_box(
                state.start_pt,
                state.current_pt,
                max_w=state.preview_w,
                max_h=state.preview_h,
                square_mode=state.square_mode,
            )
            xmin, ymin, xmax, ymax = pbox
            if xmax > xmin and ymax > ymin:
                state.apply_box_from_preview(pbox)
        state.dragging = False
        state.start_pt = None
        state.current_pt = None


def _save_mask(output_path: Path, profile: dict, mask: np.ndarray) -> None:
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(mask[np.newaxis, :, :].astype(np.uint8, copy=False))


def _save_with_feedback(state: EditorState, output_path: Path, profile: dict) -> bool:
    try:
        _save_mask(output_path=output_path, profile=profile, mask=state.mask_full)
    except Exception as exc:
        msg = f"Maske kaydedilemedi:\n{output_path}\n\nDetay: {exc}"
        print(f"HATA: {msg}")
        _show_error_dialog("Kaydetme Hatasi", msg)
        return False

    pos_count = int(np.count_nonzero(state.mask_full))
    ratio = 100.0 * pos_count / float(state.mask_full.size)
    print(f"Kaydedildi: {output_path}")
    print(f"Pozitif piksel: {pos_count} ({ratio:.2f}%)")
    state.dirty = False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GeoTIFF uzerinde kare/rectangle cizip ground-truth maskesi olusturur.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=str(CONFIG.get("input", "")), help="Girdi GeoTIFF")
    parser.add_argument("--output", "-o", type=str, default=str(CONFIG.get("output", "ground_truth_manual.tif")), help="Cikti maske GeoTIFF")
    parser.add_argument(
        "--existing-mask",
        type=str,
        default=str(CONFIG.get("existing_mask", "")),
        help="Var olan maskeyi acip ustune cizmek icin (bos birakirsan sifirdan baslar).",
    )
    parser.add_argument(
        "--preview-max-size",
        type=int,
        default=int(CONFIG.get("preview_max_size", 1800)),
        help="Onizleme penceresindeki en buyuk kenar piksel boyutu.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=str(CONFIG.get("bands", "1,2,3")),
        help="Onizleme icin band listesi (1-based), ornek: 1,2,3",
    )
    parser.add_argument(
        "--positive-value",
        type=int,
        default=int(CONFIG.get("positive_value", 1)),
        help="Pozitif etiket piksel degeri.",
    )
    parser.add_argument(
        "--square-mode",
        action="store_true",
        help="Cizimi kareye kilitle (aspect 1:1).",
    )
    parser.add_argument(
        "--free-rectangle",
        action="store_true",
        help="Kare kilidini kapatip serbest dikdortgen ciz.",
    )
    parser.add_argument(
        "--launcher",
        dest="launcher",
        action="store_true",
        help="Acilista dosya secme ve etiketleme ayarlari penceresini ac.",
    )
    parser.add_argument(
        "--no-launcher",
        dest="launcher",
        action="store_false",
        help="Acilis penceresini atla, sadece CLI argumanlarini kullan.",
    )
    parser.set_defaults(launcher=bool(CONFIG.get("use_launcher", True)))
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if bool(args.launcher):
        if tk is None or filedialog is None or messagebox is None:
            print("UYARI: tkinter GUI bulunamadi, launcher acilamadi. CLI argumanlari kullaniliyor.")
        else:
            current_square_mode = bool(CONFIG.get("square_mode", True))
            if bool(args.square_mode):
                current_square_mode = True
            if bool(args.free_rectangle):
                current_square_mode = False
            launcher_defaults = {
                "input": args.input,
                "output": args.output,
                "existing_mask": args.existing_mask,
                "bands": args.bands,
                "preview_max_size": args.preview_max_size,
                "positive_value": args.positive_value,
                "square_mode": current_square_mode,
            }
            launcher_values = _open_launcher_dialog(launcher_defaults)
            if launcher_values is None:
                print("Islem kullanici tarafindan iptal edildi.")
                return 0
            args.input = launcher_values["input"]
            args.output = launcher_values["output"]
            args.existing_mask = launcher_values["existing_mask"]
            args.bands = launcher_values["bands"]
            args.preview_max_size = launcher_values["preview_max_size"]
            args.positive_value = launcher_values["positive_value"]
            args.square_mode = bool(launcher_values["square_mode"])
            args.free_rectangle = not bool(launcher_values["square_mode"])

    input_path = Path(str(args.input).strip()).expanduser()
    if not str(args.output).strip():
        output_path = _build_default_output_path(input_path)
    else:
        output_path = Path(str(args.output).strip()).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".tif")

    if not input_path.exists():
        parser.error(f"Girdi dosyasi bulunamadi: {input_path}")
    if int(args.preview_max_size) <= 0:
        parser.error("--preview-max-size pozitif olmali.")
    if int(args.positive_value) <= 0 or int(args.positive_value) > 255:
        parser.error("--positive-value 1-255 araliginda olmali.")

    existing_mask: Optional[Path] = None
    if str(args.existing_mask).strip():
        existing_mask = Path(str(args.existing_mask).strip()).expanduser()

    src = None
    try:
        src = rasterio.open(input_path)
        bands = _parse_bands(str(args.bands), src.count)
        preview_bgr, scale_x, scale_y = _build_preview(src, bands, int(args.preview_max_size))
        profile = src.profile.copy()
        initial_mask = _load_initial_mask(
            existing_mask=existing_mask,
            height=src.height,
            width=src.width,
            positive_value=int(args.positive_value),
        )
    except Exception as exc:
        if src is not None:
            src.close()
        print(f"HATA: Raster acilamadi veya onizleme olusturulamadi: {exc}")
        return 1

    config_square = bool(CONFIG.get("square_mode", True))
    square_mode = config_square
    if bool(args.square_mode):
        square_mode = True
    if bool(args.free_rectangle):
        square_mode = False

    initial_mask_preview = cv2.resize(
        (initial_mask > 0).astype(np.uint8),
        (preview_bgr.shape[1], preview_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    initial_mask_preview[initial_mask_preview > 0] = np.uint8(int(args.positive_value))

    state = EditorState(
        preview_bgr=preview_bgr,
        mask_full=initial_mask.copy(),
        mask_preview=initial_mask_preview.copy(),
        scale_x=scale_x,
        scale_y=scale_y,
        positive_value=int(args.positive_value),
        square_mode=square_mode,
        initial_mask=initial_mask.copy(),
        initial_mask_preview=initial_mask_preview.copy(),
        zoom_max=max(24.0, float(max(scale_x, scale_y)) * 2.0),
        src=src,
        src_bands=bands,
    )

    print("=" * 78)
    print("Ground-truth editor aciliyor...")
    print(f"Girdi raster : {input_path}")
    print(f"Cikti maske  : {output_path}")
    print(f"Raster boyutu: {state.full_w}x{state.full_h}")
    print("Kontroller   :")
    print("  Sol fare: ciz | Sag fare: pan | Tekerlek: zoom | z: zoom reset")
    print("  d/e: mode | k: square | u veya Ctrl+Z: undo | c: clear | r: reset")
    print("  + / -: zoom in/out | i: tekerlek zoom yonunu tersle")
    print("  s: kaydet | a: farkli kaydet | x: kaydet+cik | q/ESC: cik")
    print("  Sag ust butonlar: Kaydet | Farkli Kaydet | Kaydet+Cik | Cikis")
    print("=" * 78)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(state.preview_w, 1600), min(state.preview_h, 1000))
        cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, state)
    except cv2.error as exc:
        if src is not None:
            src.close()
        print("HATA: OpenCV pencere acilamadi. Masaustu GUI ortaminda calistirin.")
        print(f"Detay: {exc}")
        return 1

    saved_once = False
    quit_without_save = False
    while True:
        cv2.imshow(WINDOW_NAME, _render_canvas(state))
        key = cv2.waitKey(20) & 0xFF
        action: Optional[str] = None

        if state.pending_action is not None:
            action = state.pending_action
            state.pending_action = None

        if action is None and key != 255:
            key_char = chr(key).lower() if 32 <= key <= 126 else ""
            if key == 27 or key_char == "q":
                action = "quit"
            elif key == 26:  # Ctrl+Z
                action = "undo"
            elif key_char == "d":
                action = "draw"
            elif key_char == "e":
                action = "erase"
            elif key_char == "k":
                action = "toggle_square"
            elif key_char == "u":
                action = "undo"
            elif key_char == "c":
                action = "clear"
            elif key_char == "r":
                action = "reset_initial"
            elif key_char == "z":
                action = "zoom_reset"
            elif key_char in ("+", "="):
                action = "zoom_in"
            elif key_char in ("-", "_"):
                action = "zoom_out"
            elif key_char == "i":
                action = "invert_wheel"
            elif key_char == "s":
                action = "save"
            elif key_char == "a":
                action = "save_as"
            elif key_char == "x":
                action = "save_exit"

        if action is None:
            continue

        if action == "quit":
            if state.dirty:
                decision = _confirm_unsaved_changes(output_path)
                if decision == "cancel":
                    continue
                if decision == "save":
                    if _save_with_feedback(state=state, output_path=output_path, profile=profile):
                        saved_once = True
                        break
                    continue
                if decision == "discard":
                    quit_without_save = True
                    break
                continue
            break

        if action == "draw":
            state.mode = "draw"
        elif action == "erase":
            state.mode = "erase"
        elif action == "toggle_square":
            state.square_mode = not state.square_mode
        elif action == "undo":
            state.undo()
        elif action == "clear":
            state.clear_all()
        elif action == "reset_initial":
            state.reset_initial()
        elif action == "zoom_reset":
            state.reset_zoom()
        elif action == "zoom_in":
            state.zoom_at_display(x=state.preview_w // 2, y=state.preview_h // 2, wheel_delta=120)
        elif action == "zoom_out":
            state.zoom_at_display(x=state.preview_w // 2, y=state.preview_h // 2, wheel_delta=-120)
        elif action == "invert_wheel":
            state.wheel_direction *= -1
        elif action == "save":
            if _save_with_feedback(state=state, output_path=output_path, profile=profile):
                saved_once = True
        elif action == "save_as":
            maybe_path = _pick_output_path(current_output=output_path, input_path=input_path)
            if maybe_path is not None:
                output_path = maybe_path
                if _save_with_feedback(state=state, output_path=output_path, profile=profile):
                    saved_once = True
        elif action == "save_exit":
            if _save_with_feedback(state=state, output_path=output_path, profile=profile):
                saved_once = True
                break

    cv2.destroyAllWindows()
    if src is not None:
        src.close()
    if quit_without_save:
        print("Kaydedilmeden cikildi.")
    elif not saved_once and not state.dirty:
        print("Degisiklik yok, cikildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
