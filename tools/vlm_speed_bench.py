"""VLM hiz olcum araci (salt-okunur). Gercek bir stage1 cagrisini taklit eder.

Tek bir goruntu (varsayilan 768x768) + gercek stage1 promptu gonderir, JSON uretir
ve sureyi olcer. llama.cpp 'timings' alanini (prompt-eval ve generation t/s) okur;
yoksa wall-clock'tan hesaplar. Isinma icin ilk kosu atilir.

Kullanim:
    python tools/vlm_speed_bench.py --base-url http://127.0.0.1:8080 --api-key llama-server
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional


def _png_data_url(size: int) -> str:
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    row = b"\x00" + (bytes((90, 90, 90)) * size)
    idat = zlib.compress(row * size, 6)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _post(url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_model(base_url: str, api_key: str) -> str:
    req = urllib.request.Request(f"{base_url}/models", headers={"Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ids = [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
    gemma = [i for i in ids if "gemma" in i.lower()]
    return (gemma or ids)[0]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="VLM hiz olcumu (salt-okunur).")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default="llama-server")
    p.add_argument("--image-size", type=int, default=768)
    p.add_argument("--prompt-file", default="prompts/cappadocia_stage1.txt")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--runs", type=int, default=2, help="olculecek kosu sayisi (ilk isinma haric)")
    p.add_argument("--label", default="", help="ciktiyi etiketlemek icin (orn. ngl99)")
    p.add_argument("--model", default="", help="model id; bos ise /v1/models'tan auto-resolve")
    args = p.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    model = args.model.strip() or _resolve_model(base_url, args.api_key)

    prompt_text = "Analyze this tile for archaeological features. Return strict JSON."
    pf = Path(args.prompt_file)
    if pf.exists():
        prompt_text = pf.read_text(encoding="utf-8").strip() + "\nReturn exactly one JSON object."

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": _png_data_url(args.image_size)}},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }

    label = f"[{args.label}] " if args.label else ""
    print(f"{label}model={model} image={args.image_size}px max_tokens={args.max_tokens}")

    def one_run() -> Dict[str, Any]:
        t0 = time.monotonic()
        resp = _post(f"{base_url}/chat/completions", payload, args.api_key, timeout=600)
        wall = time.monotonic() - t0
        usage = resp.get("usage") or {}
        timings = resp.get("timings") or {}
        return {"wall": wall, "usage": usage, "timings": timings}

    # isinma
    try:
        one_run()
    except Exception as exc:
        print(f"HATA (isinma): {exc}")
        return 2

    walls: List[float] = []
    prompt_tps: List[float] = []
    gen_tps: List[float] = []
    pt = ct = 0
    for i in range(max(1, args.runs)):
        r = one_run()
        walls.append(r["wall"])
        u = r["usage"]
        t = r["timings"]
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        if t.get("prompt_per_second"):
            prompt_tps.append(float(t["prompt_per_second"]))
        if t.get("predicted_per_second"):
            gen_tps.append(float(t["predicted_per_second"]))
        print(f"  run {i+1}: wall={r['wall']:.1f}s  prompt_tok={pt} gen_tok={ct}", end="")
        if t:
            print(f"  prompt_eval={t.get('prompt_per_second',0):.1f} t/s  gen={t.get('predicted_per_second',0):.1f} t/s")
        else:
            print()

    med_wall = sorted(walls)[len(walls) // 2]
    print("-" * 60)
    print(f"{label}MEDIAN wall = {med_wall:.1f} s/istek  (prompt_tok={pt}, gen_tok={ct})")
    if prompt_tps:
        print(f"{label}prompt-eval ort = {sum(prompt_tps)/len(prompt_tps):.1f} t/s  (1120 goruntu token'inin islenme hizi)")
    if gen_tps:
        print(f"{label}generation ort  = {sum(gen_tps)/len(gen_tps):.1f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
