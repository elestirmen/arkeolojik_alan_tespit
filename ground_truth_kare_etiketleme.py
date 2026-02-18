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
    d               : draw mode
    e               : erase mode
    k               : toggle square lock
    u               : undo last box
    c               : clear all labels
    r               : reset to initial mask
    s               : save and exit
    q / ESC         : quit without save
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


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
}
# ===============================================

WINDOW_NAME = "Ground Truth Kare Etiketleme"
OVERLAY_COLOR_BGR = np.array([0, 0, 255], dtype=np.float32)
OVERLAY_ALPHA = 0.35


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


@dataclass
class EditorState:
    preview_bgr: np.ndarray
    mask_full: np.ndarray
    scale_x: float
    scale_y: float
    positive_value: int
    square_mode: bool
    mode: str = "draw"  # draw | erase
    dragging: bool = False
    start_pt: Optional[tuple[int, int]] = None
    current_pt: Optional[tuple[int, int]] = None
    history: list[tuple[int, int, int, int, np.ndarray]] = field(default_factory=list)
    initial_mask: Optional[np.ndarray] = None

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

    def apply_box_from_preview(self, pbox: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = _preview_box_to_full(
            pbox=pbox,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            full_w=self.full_w,
            full_h=self.full_h,
        )
        if x1 <= x0 or y1 <= y0:
            return

        previous = self.mask_full[y0:y1, x0:x1].copy()
        if self.mode == "draw":
            self.mask_full[y0:y1, x0:x1] = np.uint8(self.positive_value)
        else:
            self.mask_full[y0:y1, x0:x1] = np.uint8(0)
        self.history.append((x0, y0, x1, y1, previous))

    def undo(self) -> None:
        if not self.history:
            return
        x0, y0, x1, y1, previous = self.history.pop()
        self.mask_full[y0:y1, x0:x1] = previous

    def clear_all(self) -> None:
        self.mask_full.fill(0)
        self.history.clear()

    def reset_initial(self) -> None:
        if self.initial_mask is not None:
            self.mask_full[:, :] = self.initial_mask
            self.history.clear()


def _render_canvas(state: EditorState) -> np.ndarray:
    canvas = state.preview_bgr.copy()
    mask_preview = cv2.resize(
        (state.mask_full > 0).astype(np.uint8),
        (state.preview_w, state.preview_h),
        interpolation=cv2.INTER_NEAREST,
    )
    idx = mask_preview > 0
    if np.any(idx):
        blended = (1.0 - OVERLAY_ALPHA) * canvas[idx].astype(np.float32) + OVERLAY_ALPHA * OVERLAY_COLOR_BGR
        canvas[idx] = blended.astype(np.uint8)

    if state.dragging and state.start_pt is not None and state.current_pt is not None:
        xmin, ymin, xmax, ymax = _normalize_preview_box(
            state.start_pt,
            state.current_pt,
            max_w=state.preview_w,
            max_h=state.preview_h,
            square_mode=state.square_mode,
        )
        color = (0, 255, 0) if state.mode == "draw" else (0, 255, 255)
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

    pos_count = int(np.count_nonzero(state.mask_full))
    total = int(state.mask_full.size)
    ratio = (100.0 * pos_count / total) if total > 0 else 0.0
    line1 = f"mode: {state.mode} | square: {'on' if state.square_mode else 'off'} | labels: {len(state.history)}"
    line2 = f"positive pixels: {pos_count} / {total} ({ratio:.2f}%)"
    line3 = "keys: d draw | e erase | k square | u undo | c clear | r reset | s save | q quit"
    cv2.putText(canvas, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _mouse_callback(event: int, x: int, y: int, _flags: int, state: EditorState) -> None:
    x = int(np.clip(x, 0, state.preview_w - 1))
    y = int(np.clip(y, 0, state.preview_h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        state.dragging = True
        state.start_pt = (x, y)
        state.current_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        state.current_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state.dragging:
        state.current_pt = (x, y)
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
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        parser.error(f"Girdi dosyasi bulunamadi: {input_path}")
    if int(args.preview_max_size) <= 0:
        parser.error("--preview-max-size pozitif olmali.")
    if int(args.positive_value) <= 0 or int(args.positive_value) > 255:
        parser.error("--positive-value 1-255 araliginda olmali.")

    existing_mask: Optional[Path] = None
    if str(args.existing_mask).strip():
        existing_mask = Path(str(args.existing_mask).strip()).expanduser()

    try:
        with rasterio.open(input_path) as src:
            bands = _parse_bands(args.bands, src.count)
            preview_bgr, scale_x, scale_y = _build_preview(src, bands, int(args.preview_max_size))
            profile = src.profile.copy()
            initial_mask = _load_initial_mask(
                existing_mask=existing_mask,
                height=src.height,
                width=src.width,
                positive_value=int(args.positive_value),
            )
    except Exception as exc:
        print(f"HATA: Raster acilamadi veya onizleme olusturulamadi: {exc}")
        return 1

    config_square = bool(CONFIG.get("square_mode", True))
    square_mode = config_square
    if args.square_mode:
        square_mode = True
    if args.free_rectangle:
        square_mode = False

    state = EditorState(
        preview_bgr=preview_bgr,
        mask_full=initial_mask.copy(),
        scale_x=scale_x,
        scale_y=scale_y,
        positive_value=int(args.positive_value),
        square_mode=square_mode,
        initial_mask=initial_mask.copy(),
    )

    print("=" * 72)
    print("Ground-truth editor aciliyor...")
    print(f"Girdi raster : {input_path}")
    print(f"Cikti maske  : {output_path}")
    print(f"Raster boyutu: {state.full_w}x{state.full_h}")
    print("Kontroller   :")
    print("  Sol fare: ciz | d: draw | e: erase | k: square toggle")
    print("  u: undo | c: clear | r: reset | s: save | q/ESC: quit")
    print("=" * 72)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(state.preview_w, 1600), min(state.preview_h, 1000))
        cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, state)
    except cv2.error as exc:
        print("HATA: OpenCV pencere acilamadi. Masaustu GUI ortaminda calistirin.")
        print(f"Detay: {exc}")
        return 1

    saved = False
    while True:
        cv2.imshow(WINDOW_NAME, _render_canvas(state))
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key in (27, ord("q")):
            break
        if key == ord("d"):
            state.mode = "draw"
        elif key == ord("e"):
            state.mode = "erase"
        elif key == ord("k"):
            state.square_mode = not state.square_mode
        elif key == ord("u"):
            state.undo()
        elif key == ord("c"):
            state.clear_all()
        elif key == ord("r"):
            state.reset_initial()
        elif key == ord("s"):
            _save_mask(output_path=output_path, profile=profile, mask=state.mask_full)
            pos_count = int(np.count_nonzero(state.mask_full))
            ratio = 100.0 * pos_count / float(state.mask_full.size)
            print(f"Kaydedildi: {output_path}")
            print(f"Pozitif piksel: {pos_count} ({ratio:.2f}%)")
            saved = True
            break

    cv2.destroyAllWindows()
    if not saved:
        print("Kaydedilmeden cikildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

