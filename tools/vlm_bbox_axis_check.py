"""VLM bbox eksen-sirasi dogrulama araci (Faz 2a, salt-okunur).

Amac: Gemma 4'un bbox'i native ``[y1,x1,y2,x2]`` (y-once) mi yoksa kodun bekledigi
``[x1,y1,x2,y2]`` (x-once) mi dondurdugunu AMPIRIK olarak gostermek.

Bir run klasorundeki adaylardan secilenler icin kaynak tile'i okur ve bbox'i IKI
yorumla cizer:
    KIRMIZI  = x-once  [x1,y1,x2,y2]  (kodun su anki yorumu, _validate_bbox)
    YESIL    = y-once  [y1,x1,y2,x2]  (Gemma 4 native / TF konvansiyonu)
Hangi kutu, kaydin ``visual_evidence`` metninde tarif edilen yere/yonelime oturuyorsa
dogru yorum odur.

Pipeline'i HIC degistirmez; sadece okur ve PNG uretir.

Kullanim (archeo env ile):
    .../envs/archeo/python.exe tools/vlm_bbox_axis_check.py --run-folder <klasor> --count 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw


def _read_run_config(run_folder: Path) -> Dict[str, Any]:
    hits = glob.glob(str(run_folder / "*_vlm_run_config.json"))
    if not hits:
        raise FileNotFoundError(f"run_config bulunamadi: {run_folder}")
    return json.load(open(hits[0], encoding="utf-8"))


def _candidates_jsonl(run_folder: Path) -> Path:
    hits = sorted(glob.glob(str(run_folder / "*_vlm_candidates.jsonl")))
    if not hits:
        raise FileNotFoundError(f"candidates.jsonl bulunamadi: {run_folder}")
    return Path(hits[0])


def _scale_uint8(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.uint8)
    lo = float(np.percentile(data[valid], low))
    hi = float(np.percentile(data[valid], high))
    if hi - lo <= 1e-6:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _render_tile_rgb(src: rasterio.io.DatasetReader, window: Window) -> Image.Image:
    count = int(src.count)
    idx = [1, 2, 3] if count >= 3 else [1, 1, 1]
    bands = [src.read(i, window=window, boundless=True, fill_value=0) for i in idx]
    rgb = np.stack([_scale_uint8(b) for b in bands], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _is_distinguishable(bbox: List[float], tw: int, th: int) -> bool:
    """Iki yorumun gorsel olarak farkli oldugu kutulari sec (kare/tam-tile olmayan)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 4 or h <= 4:
        return False
    if w * h >= 0.7 * tw * th:  # tam-tile yorumlari ayni gosterir
        return False
    longer = max(w, h)
    if longer <= 0:
        return False
    return abs(w - h) / longer >= 0.3  # belirgin dikdortgen


def _draw(img: Image.Image, bbox: List[float], caption: str) -> Image.Image:
    canvas = img.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox
    # KIRMIZI: x-once (kodun yorumu)
    d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=4)
    # YESIL: y-once (native) -> stored [a,b,c,d] = [y1,x1,y2,x2]
    d.rectangle([y1, x1, y2, x2], outline=(40, 230, 40), width=2)
    d.text((6, 6), caption, fill=(255, 255, 0))
    d.text((6, 18), "KIRMIZI=x-once(kod)  YESIL=y-once(native)", fill=(255, 255, 0))
    return canvas


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="VLM bbox eksen-sirasi dogrulama (salt-okunur).")
    p.add_argument("--run-folder", required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    run_folder = Path(args.run_folder)
    cfg = _read_run_config(run_folder)
    input_path = Path(str(cfg.get("input")))
    if not input_path.exists():
        print(f"HATA: girdi raster yok: {input_path}")
        return 2
    jsonl = _candidates_jsonl(run_folder)
    out_dir = Path(args.out_dir) if args.out_dir else (run_folder / "bbox_axis_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ayirt edilebilir kutulu adaylari topla
    picked: List[Dict[str, Any]] = []
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        b = r.get("bbox_xyxy")
        ve = (r.get("visual_evidence") or "").strip()
        tw = r.get("tile_width")
        th = r.get("tile_height")
        if not (b and ve and tw and th):
            continue
        if not _is_distinguishable([float(x) for x in b], int(tw), int(th)):
            continue
        picked.append(r)

    if not picked:
        print("Ayirt edilebilir kutulu aday bulunamadi.")
        return 1

    # dosya boyunca dagit
    step = max(1, len(picked) // args.count)
    chosen = picked[::step][: args.count]
    print(f"Girdi raster : {input_path}")
    print(f"candidates   : {jsonl.name}  (ayirt edilebilir aday: {len(picked)})")
    print(f"Cikti klasoru: {out_dir}")
    print("-" * 70)

    with rasterio.open(input_path) as src:
        for n, r in enumerate(chosen, 1):
            col = int(r.get("tile_col") or 0)
            row = int(r.get("tile_row") or 0)
            tw = int(r.get("tile_width"))
            th = int(r.get("tile_height"))
            b = [float(x) for x in r.get("bbox_xyxy")]
            window = Window(col, row, tw, th)
            try:
                img = _render_tile_rgb(src, window)
            except Exception as exc:
                print(f"[{n}] tile okunamadi (tile_index={r.get('tile_index')}): {exc}")
                continue
            ctype = str(r.get("candidate_type"))
            ve = str(r.get("visual_evidence") or "")[:200]
            cap = f"#{r.get('tile_index')} {ctype} bbox={[round(v) for v in b]}"
            out = _draw(img, b, cap)
            fname = out_dir / f"axis_{n:02d}_t{r.get('tile_index')}_{ctype}.png"
            out.save(fname)
            print(f"[{n}] {ctype:12s} bbox(stored)={[round(v) for v in b]}")
            print(f"     x-once(KIRMIZI): x{round(b[0])}-{round(b[2])}, y{round(b[1])}-{round(b[3])}")
            print(f"     y-once(YESIL)  : x{round(b[1])}-{round(b[3])}, y{round(b[0])}-{round(b[2])}")
            print(f"     evidence: {ve}")
            print(f"     -> {fname.name}")
            print("-" * 70)
    print("Goruntuleri acip KIRMIZI mi YESIL mi feature'a/evidence'a oturuyor bakin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
