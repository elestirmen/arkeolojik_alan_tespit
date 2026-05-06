"""Standalone VLM archaeological candidate scanner.

This CLI intentionally does not import ``archaeo_detect.py``. The old
classic/DL/YOLO pipeline can keep its own dependencies, while this entrypoint
only loads the raster/VLM pieces needed for LM Studio based scanning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from archeo_shared.console import configure_utf8_console

configure_utf8_console()

import rasterio

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency message is clearer at runtime
    yaml = None  # type: ignore[assignment]

from vlm_lmstudio_detector import VlmLmStudioConfig, run_vlm_lmstudio_detection


LOGGER = logging.getLogger("vlm_detect")
SESSION_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKSPACE_OUTPUTS_DIR = Path("workspace") / "ciktilar"
KNOWN_OUTPUT_SUFFIXES = {
    ".tif",
    ".tiff",
    ".gpkg",
    ".xlsx",
    ".csv",
    ".geojson",
    ".json",
    ".jsonl",
    ".shp",
    ".vrt",
}

PATH_FIELDS = {"input", "out_prefix", "output_root", "resume_jsonl_path", "prompt_stage1_path", "prompt_stage2_path"}

VLM_KEY_ALIASES = {
    "vlm_base_url": "base_url",
    "vlm_api_key": "api_key",
    "vlm_model": "model",
    "vlm_tile": "tile",
    "vlm_overlap": "overlap",
    "vlm_views": "views",
    "vlm_source_kind": "source_kind",
    "vlm_gsd_m": "gsd_m",
    "vlm_confidence_threshold": "confidence_threshold",
    "vlm_max_tiles": "max_tiles",
    "vlm_export_every": "export_every",
    "vlm_resume": "resume",
    "vlm_resume_jsonl_path": "resume_jsonl_path",
    "vlm_prompt_stage1_path": "prompt_stage1_path",
    "vlm_prompt_stage2_path": "prompt_stage2_path",
    "vlm_reload_every_tiles": "reload_every_tiles",
    "vlm_reload_pause_seconds": "reload_pause_seconds",
    "vlm_timeout": "timeout",
    "vlm_temperature": "temperature",
}


@dataclass(frozen=True)
class StandaloneVlmConfig:
    input: str = "kesif_alani.tif"
    out_prefix: Optional[str] = None
    output_root: str = str(WORKSPACE_OUTPUTS_DIR)
    bands: str = "auto"
    base_url: str = "http://127.0.0.1:8081"
    api_key: str = "lm-studio"
    model: str = "auto"
    tile: int = 1024
    overlap: int = 256
    views: str = "auto"
    source_kind: str = "auto"
    gsd_m: Optional[float] = 0.0
    confidence_threshold: float = 0.75
    max_tiles: int = 0
    export_every: int = 150
    resume: bool = True
    resume_jsonl_path: Optional[str] = None
    prompt_stage1_path: Optional[str] = None
    prompt_stage2_path: Optional[str] = None
    reload_every_tiles: int = 250
    reload_pause_seconds: float = 5.0
    timeout: int = 120
    temperature: float = 0.0
    log_level: str = "INFO"


def default_config_path() -> str:
    """Prefer VLM-specific config files."""
    for candidate in ("config_vlm.local.yaml", "config_vlm.yaml"):
        if Path(candidate).exists():
            return candidate
    return "config_vlm.yaml"


def load_yaml_config(yaml_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("YAML destegi icin PyYAML yukleyin: pip install pyyaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config kok degeri sozluk olmali: {yaml_path}")
    return data


def normalize_config_keys(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both VLM-only keys and legacy vlm_* keys."""
    known = {field.name for field in fields(StandaloneVlmConfig)}
    normalized: Dict[str, Any] = {}
    for raw_key, value in raw_config.items():
        key = VLM_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key in known:
            normalized[key] = value
    return normalized


def _resolve_path_value(raw: Any, base_dir: Path) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return raw
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve(strict=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM-only archaeological candidate scan with LM Studio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=default_config_path(), help="VLM YAML config path.")
    parser.add_argument("--input", dest="input", help="Input GeoTIFF path.")
    parser.add_argument("--out-prefix", dest="out_prefix", help="Output base name or prefix.")
    parser.add_argument("--output-root", dest="output_root", help="Root output directory.")
    parser.add_argument("--bands", dest="bands", help="Band spec: auto, 1,2,3, or 1,2,3,4,5.")
    parser.add_argument("--base-url", "--vlm-base-url", dest="base_url", help="LM Studio base URL.")
    parser.add_argument("--api-key", "--vlm-api-key", dest="api_key", help="LM Studio API key.")
    parser.add_argument("--model", "--vlm-model", dest="model", help="LM Studio model name or auto.")
    parser.add_argument("--tile", "--vlm-tile", dest="tile", type=int, help="Tile size in pixels.")
    parser.add_argument("--overlap", "--vlm-overlap", dest="overlap", type=int, help="Tile overlap in pixels.")
    parser.add_argument("--views", "--vlm-views", dest="views", help="VLM views: auto or CSV.")
    parser.add_argument(
        "--source-kind",
        "--vlm-source-kind",
        dest="source_kind",
        help="Input interpretation: auto, rgb, hillshade, single_band, dsm, dtm, or mixed_topo.",
    )
    parser.add_argument("--gsd-m", "--vlm-gsd-m", dest="gsd_m", type=float, help="Ground sample distance in m/px; 0 auto.")
    parser.add_argument(
        "--confidence-threshold",
        "--vlm-confidence-threshold",
        dest="confidence_threshold",
        type=float,
        help="Minimum first-stage confidence for export/review.",
    )
    parser.add_argument("--max-tiles", "--vlm-max-tiles", dest="max_tiles", type=int, help="Maximum tiles; 0 unlimited.")
    parser.add_argument(
        "--export-every",
        "--vlm-export-every",
        dest="export_every",
        type=int,
        help="Refresh side outputs every N new tiles; 0 only at end.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=None, help="Resume from previous JSONL.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None, help="Disable resume.")
    parser.add_argument("--resume-jsonl-path", dest="resume_jsonl_path", help="Explicit VLM JSONL resume source.")
    parser.add_argument(
        "--prompt-stage1-path",
        "--vlm-prompt-stage1-path",
        dest="prompt_stage1_path",
        help="Stage 1 decision guidance prompt text file.",
    )
    parser.add_argument(
        "--prompt-stage2-path",
        "--vlm-prompt-stage2-path",
        dest="prompt_stage2_path",
        help="Stage 2 review guidance prompt text file.",
    )
    parser.add_argument(
        "--reload-every-tiles",
        "--vlm-reload-every-tiles",
        dest="reload_every_tiles",
        type=int,
        help="Unload/load LM Studio model every N new tiles; 0 disabled.",
    )
    parser.add_argument(
        "--reload-pause-seconds",
        "--vlm-reload-pause-seconds",
        dest="reload_pause_seconds",
        type=float,
        help="Pause between LM Studio unload and load.",
    )
    parser.add_argument("--timeout", "--vlm-timeout", dest="timeout", type=int, help="API timeout seconds.")
    parser.add_argument("--temperature", "--vlm-temperature", dest="temperature", type=float, help="Chat temperature.")
    parser.add_argument("--log-level", dest="log_level", help="Logging level.")
    return parser


def build_config_from_args(args: argparse.Namespace) -> StandaloneVlmConfig:
    values = {field.name: getattr(StandaloneVlmConfig(), field.name) for field in fields(StandaloneVlmConfig)}
    yaml_fields: set[str] = set()
    yaml_dir: Optional[Path] = None

    if args.config:
        yaml_path = Path(args.config)
        raw_config = load_yaml_config(yaml_path)
        yaml_values = normalize_config_keys(raw_config)
        values.update(yaml_values)
        yaml_fields = set(yaml_values)
        yaml_dir = yaml_path.resolve(strict=False).parent

    cli_fields: set[str] = set()
    for field in fields(StandaloneVlmConfig):
        if not hasattr(args, field.name):
            continue
        value = getattr(args, field.name)
        if value is None:
            continue
        values[field.name] = value
        cli_fields.add(field.name)

    cwd = Path.cwd()
    for name in PATH_FIELDS:
        if name not in values:
            continue
        if name in cli_fields:
            base_dir = cwd
        elif name in yaml_fields and yaml_dir is not None:
            base_dir = yaml_dir
        else:
            base_dir = cwd
        values[name] = _resolve_path_value(values[name], base_dir)

    return StandaloneVlmConfig(**values)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    token = token.strip("._-")
    return token or "run"


def _output_base_path(path: Path) -> Path:
    base = path
    while base.suffix.lower() in KNOWN_OUTPUT_SUFFIXES:
        base = base.with_suffix("")
    return base


def build_vlm_session_folder_name(input_path: Path, config: StandaloneVlmConfig) -> str:
    stem = _safe_token(input_path.stem)[:28].rstrip("_-") or "input"
    tile_part = f"t{int(config.tile)}"
    overlap_part = f"o{int(config.overlap)}"
    return f"{SESSION_RUN_ID}_{stem}_vlm_{tile_part}{overlap_part}"


def resolve_out_prefix(input_path: Path, config: StandaloneVlmConfig) -> Path:
    fallback_name = _output_base_path(input_path).name
    if config.out_prefix:
        out_path = Path(config.out_prefix)
        if out_path.is_dir():
            out_path = out_path / input_path.stem
        out_name = _output_base_path(out_path).name or fallback_name
    else:
        out_name = fallback_name

    output_root = Path(config.output_root)
    session_folder = build_vlm_session_folder_name(input_path, config)
    return output_root / session_folder / out_name


def find_latest_vlm_resume_jsonl(out_prefix: Path, output_root: Path) -> Optional[Path]:
    try:
        base = _output_base_path(out_prefix)
        current = base.parent / f"{base.name}_vlm_candidates.jsonl"
        current_resolved = current.resolve(strict=False)
        pattern = f"*/{base.name}_vlm_candidates.jsonl"
        candidates = []
        for path in output_root.glob(pattern):
            resolved = path.resolve(strict=False)
            if resolved == current_resolved:
                continue
            if path.is_file() and path.stat().st_size > 0:
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    except Exception as exc:
        LOGGER.warning("VLM resume JSONL aranirken hata olustu: %s", exc)
        return None


def parse_manual_band_indexes(band_string: str) -> Tuple[int, ...]:
    parts = [int(part.strip()) for part in str(band_string).split(",") if part.strip()]
    if len(parts) not in (3, 5):
        raise ValueError("bands 3 veya 5 deger icermeli; ornek: auto, 1,2,3, 1,2,3,4,5")
    if any(idx <= 0 for idx in parts[:3]):
        raise ValueError("RGB band indeksleri 1 veya daha buyuk olmali.")
    if len(parts) == 5 and any(idx < 0 for idx in parts[3:]):
        raise ValueError("DSM/DTM band indeksleri 0 veya daha buyuk olmali.")
    return tuple(parts)


def resolve_vlm_band_indexes(input_path: Path, band_spec: str) -> Optional[Tuple[int, ...]]:
    """Resolve VLM bands. None means let vlm_lmstudio_detector auto-detect 3+ band rasters."""
    spec = str(band_spec or "").strip()
    if not spec or spec.lower() == "auto":
        with rasterio.open(input_path) as src:
            band_count = int(src.count)
        if band_count == 1:
            return (1, 1, 1)
        if band_count >= 3:
            return None
        raise ValueError(f"bands='auto' icin desteklenmeyen bant sayisi: {band_count}")

    parsed = parse_manual_band_indexes(spec)
    with rasterio.open(input_path) as src:
        band_count = int(src.count)
    for idx in parsed:
        if idx > band_count:
            raise ValueError(f"Band indeksi raster bant sayisini asiyor: {idx} > {band_count}")
    return parsed


def write_run_manifest(out_prefix: Path, config: StandaloneVlmConfig, band_indexes: Optional[Tuple[int, ...]]) -> Path:
    manifest_path = _output_base_path(out_prefix).parent / f"{_output_base_path(out_prefix).name}_vlm_run_config.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["resolved_band_indexes"] = list(band_indexes) if band_indexes is not None else "auto"
    payload["session_run_id"] = SESSION_RUN_ID
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run(config: StandaloneVlmConfig) -> int:
    input_path = Path(config.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi GeoTIFF bulunamadi: {input_path}")

    out_prefix = resolve_out_prefix(input_path, config)
    band_indexes = resolve_vlm_band_indexes(input_path, config.bands)
    output_root = Path(config.output_root)
    resume_jsonl_path: Optional[Path] = None
    if config.resume:
        if config.resume_jsonl_path:
            resume_jsonl_path = Path(config.resume_jsonl_path)
        else:
            resume_jsonl_path = find_latest_vlm_resume_jsonl(out_prefix, output_root)
        if resume_jsonl_path is not None:
            LOGGER.info("VLM resume JSONL bulundu: %s", resume_jsonl_path)

    manifest_path = write_run_manifest(out_prefix, config, band_indexes)
    LOGGER.info("VLM-only tarama basliyor")
    LOGGER.info("Girdi: %s", input_path)
    LOGGER.info("Cikti on eki: %s", out_prefix)
    LOGGER.info("Run config: %s", manifest_path)

    summary = run_vlm_lmstudio_detection(
        input_path=input_path,
        out_prefix=out_prefix,
        band_indexes=band_indexes,
        config=VlmLmStudioConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            tile=config.tile,
            overlap=config.overlap,
            views=config.views,
            source_kind=config.source_kind,
            gsd_m=config.gsd_m,
            confidence_threshold=config.confidence_threshold,
            max_tiles=config.max_tiles,
            timeout=config.timeout,
            temperature=config.temperature,
            export_every=config.export_every,
            resume=config.resume,
            resume_jsonl_path=resume_jsonl_path,
            prompt_stage1_path=config.prompt_stage1_path,
            prompt_stage2_path=config.prompt_stage2_path,
            reload_every_tiles=config.reload_every_tiles,
            reload_pause_seconds=config.reload_pause_seconds,
        ),
        logger=LOGGER,
    )

    LOGGER.info(
        "VLM tamamlandi: %d/%d tile, resume=%d, model_aday=%d, esik_ustu=%d, hata=%d. Mod=%s, view=%s",
        summary.processed_tiles,
        summary.total_tiles,
        summary.resumed_tiles,
        summary.raw_candidate_count,
        summary.candidate_count,
        summary.error_count,
        summary.analysis_mode,
        ",".join(summary.used_views),
    )
    LOGGER.info("VLM JSONL: %s", summary.paths.jsonl)
    LOGGER.info("VLM CSV: %s", summary.paths.csv)
    LOGGER.info("VLM Excel: %s", summary.paths.xlsx)
    LOGGER.info("VLM GeoJSON: %s", summary.paths.geojson)
    LOGGER.info("VLM GPKG ilk asama pozitifler: %s", summary.paths.gpkg_stage1)
    LOGGER.info("VLM GPKG ikinci asama dogrulanan: %s", summary.paths.gpkg_stage2)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config_from_args(args)
        configure_logging(config.log_level)
        return run(config)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
