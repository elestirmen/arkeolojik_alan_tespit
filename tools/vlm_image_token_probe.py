"""Gemma 4 / LM Studio gorsel token butcesi olcum araci (salt-okunur).

Bu arac pipeline'i HIC degistirmez; sadece LM Studio OpenAI-uyumlu endpoint'ine
pipeline ile ayni sekilde (PNG data URL) tek bir istek gonderir ve donen
``usage.prompt_tokens`` degerinden goruntu basina harcanan token sayisini olcer.

Gemma 4'un goruntu token butcesi sabit "soft token" sayisidir (70/140/280/560/1120)
ve girdi piksel boyutundan bagimsizdir. Goruntulu ve goruntusuz iki istegin
prompt_tokens farki ~bu butceyi verir:

    delta ~ 280   -> varsayilan (genel sohbet katmani)
    delta ~ 1120  -> kucuk-nesne/OCR katmani (arkeolojik tespit icin onerilen)

Kullanim:
    python tools/vlm_image_token_probe.py
    python tools/vlm_image_token_probe.py --model google/gemma-4-26b-a4b --sizes 512,1024
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_config_defaults() -> Dict[str, Any]:
    """config_vlm.local.yaml / config_vlm.yaml icinden base_url/api_key/model oku (varsa)."""
    defaults = {"base_url": "http://127.0.0.1:8081", "api_key": "lm-studio", "model": "auto"}
    try:
        import yaml  # type: ignore
    except ImportError:
        return defaults
    for name in ("config_vlm.local.yaml", "config_vlm.yaml"):
        path = Path(name)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        for key in ("base_url", "api_key", "model"):
            if data.get(key):
                defaults[key] = data[key]
        break
    return defaults


def _normalize_base_url(base_url: str) -> str:
    text = str(base_url or "").strip().rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


def _http_json(url: str, payload: Optional[Dict[str, Any]], api_key: str, timeout: float) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_model(base_url: str, api_key: str, requested: str, timeout: float) -> str:
    if requested and requested.strip().lower() not in {"auto", "active", "current", "loaded"}:
        return requested.strip()
    try:
        models = _http_json(f"{base_url}/models", None, api_key, timeout)
        ids = [str(m.get("id")) for m in models.get("data", []) if m.get("id")]
    except Exception:
        ids = []
    # embedding olmayan, tercihen gemma-4 olan ilk modeli sec
    gemma = [i for i in ids if "gemma-4" in i.lower()]
    if gemma:
        return gemma[0]
    non_embed = [i for i in ids if "embed" not in i.lower()]
    if non_embed:
        return non_embed[0]
    raise RuntimeError("LM Studio'da uygun model bulunamadi (/v1/models bos).")


def _png_bytes(width: int, height: int, rgb: tuple = (90, 90, 90)) -> bytes:
    """Bagimsiz, stdlib-only RGB PNG kodlayici (PIL gerektirmez)."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    row = b"\x00" + (bytes(rgb) * width)  # her satir basina filter byte 0
    raw = row * height
    idat = zlib.compress(raw, 6)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _solid_png_data_url(size: int) -> str:
    encoded = base64.b64encode(_png_bytes(size, size)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _prompt_tokens(
    *, base_url: str, api_key: str, model: str, with_image: bool, size: int, timeout: float
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": "Reply with the single word OK."}]
    if with_image:
        content.append({"type": "image_url", "image_url": {"url": _solid_png_data_url(size)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    resp = _http_json(f"{base_url}/chat/completions", payload, api_key, timeout)
    usage = resp.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "usage_present": bool(usage),
    }


def _nearest_budget(delta: Optional[int]) -> str:
    if delta is None:
        return "bilinmiyor"
    budgets = [70, 140, 280, 560, 1120]
    nearest = min(budgets, key=lambda b: abs(b - delta))
    return f"~{nearest}"


def main(argv: Optional[List[str]] = None) -> int:
    cfg = _load_config_defaults()
    parser = argparse.ArgumentParser(description="Gemma 4 / LM Studio gorsel token butcesi olcumu (salt-okunur).")
    parser.add_argument("--base-url", default=cfg["base_url"])
    parser.add_argument("--api-key", default=cfg["api_key"])
    parser.add_argument("--model", default=cfg["model"], help="Model id veya 'auto'.")
    parser.add_argument("--sizes", default="512,1024", help="Test edilecek piksel boyutlari (CSV).")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    base_url = _normalize_base_url(args.base_url)
    try:
        model = _resolve_model(base_url, args.api_key, args.model, min(args.timeout, 10.0))
    except Exception as exc:
        print(f"HATA: {exc}")
        return 2

    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]
    print(f"Endpoint : {base_url}")
    print(f"Model    : {model}")
    print("-" * 64)

    try:
        baseline = _prompt_tokens(
            base_url=base_url, api_key=args.api_key, model=model, with_image=False, size=sizes[0], timeout=args.timeout
        )
    except Exception as exc:
        print(f"HATA (goruntusuz istek): {exc}")
        return 2

    if not baseline["usage_present"] or baseline["prompt_tokens"] is None:
        print("UYARI: Sunucu 'usage.prompt_tokens' dondurmedi; bu olcum yontemi calismaz.")
        print("LM Studio surumunuzde usage raporlamasini acin veya server log'undan token sayisini izleyin.")
        return 1

    base_tok = int(baseline["prompt_tokens"])
    print(f"Goruntusuz prompt_tokens (taban) : {base_tok}")
    print("-" * 64)

    results = []
    for size in sizes:
        try:
            r = _prompt_tokens(
                base_url=base_url, api_key=args.api_key, model=model, with_image=True, size=size, timeout=args.timeout
            )
        except Exception as exc:
            print(f"{size}x{size}: HATA -> {exc}")
            continue
        img_tok = r["prompt_tokens"]
        if img_tok is None:
            print(f"{size}x{size}: prompt_tokens yok")
            continue
        delta = int(img_tok) - base_tok
        results.append((size, int(img_tok), delta))
        print(f"{size}x{size}: prompt_tokens={img_tok}  ->  goruntu ~{delta} token  (en yakin butce: {_nearest_budget(delta)})")

    print("-" * 64)
    if results:
        deltas = [d for _, _, d in results]
        nearest = _nearest_budget(round(sum(deltas) / len(deltas)))
        print(f"SONUC: goruntu basina {nearest} soft token kullaniliyor.")
        if "1120" in nearest:
            print("  -> 1120 AKTIF. Kucuk-nesne tespiti icin dogru ayar.")
        elif "280" in nearest:
            print("  -> Varsayilan 280. Onerilen: LM Studio Load Config'te image-min/max-tokens=1120, ubatch>=2048.")
        else:
            print("  -> 1120 degil. LM Studio Load Config'ten image token butcesini 1120'ye cekin.")
        if len(set(deltas)) == 1:
            print("  (Not: farkli piksel boyutlari ayni token verdi -> beklendigi gibi sabit soft-token butcesi.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
