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
import urllib.parse
import urllib.request
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

from vlm_lmstudio_detector import (
    VlmLmStudioConfig,
    VlmOutputPaths,
    _default_stage1_guidance,
    _default_stage2_guidance,
    _normalize_openai_base_url,
    run_vlm_lmstudio_detection,
    _write_candidate_outputs,
    _candidate_record_lists,
)


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
    "vlm_max_candidates_per_tile": "max_candidates_per_tile",
    "vlm_reload_every_tiles": "reload_every_tiles",
    "vlm_reload_pause_seconds": "reload_pause_seconds",
    "vlm_timeout": "timeout",
    "vlm_temperature": "temperature",
    "vlm_featureless_std_threshold": "featureless_std_threshold",
    "vlm_cross_tile_iou_threshold": "cross_tile_iou_threshold",
    "vlm_batch": "batch",
    "vlm_max_tokens": "max_tokens",
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
    max_candidates_per_tile: int = 3
    reload_every_tiles: int = 250
    reload_pause_seconds: float = 5.0
    timeout: int = 120
    temperature: float = 0.0
    log_level: str = "INFO"
    featureless_std_threshold: float = 4.0
    cross_tile_iou_threshold: float = 0.5
    batch: bool = False
    backend: str = "lmstudio"
    max_tokens: int = 512


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


def _apply_backend_profile(raw_config: Dict[str, Any], cli_backend: Optional[str] = None) -> Dict[str, Any]:
    """Tek config icinde backend profili (orn. lmstudio / llama) secimi.

    ``backends:`` (veya ``profiles:``) altindaki secili profilin alanlari ortak
    ayarlarin uzerine yazilir. Profil bolumu yoksa config oldugu gibi kullanilir
    (eski duz configlerle geriye donuk uyumlu). CLI ``--backend`` config'teki
    ``backend`` degerini ezer.
    """
    raw = dict(raw_config)
    profiles = raw.pop("backends", None)
    if profiles is None:
        profiles = raw.pop("profiles", None)

    backend = cli_backend or raw.get("backend")

    if profiles:
        if not isinstance(profiles, dict):
            raise ValueError("config 'backends' bir sozluk olmali.")
        if not backend:
            raise ValueError(
                f"Config 'backends' iceriyor ancak 'backend' secili degil. "
                f"Secenekler: {sorted(profiles)} (config'te 'backend:' yazin veya --backend verin)."
            )
        profile = profiles.get(str(backend))
        if profile is None:
            raise ValueError(
                f"backend='{backend}' icin profil bulunamadi. Mevcut profiller: {sorted(profiles)}."
            )
        if not isinstance(profile, dict):
            raise ValueError(f"'{backend}' backend profili bir sozluk olmali.")
        raw.update(profile)  # profil, ortak degerleri ezer

    if backend is not None:
        raw["backend"] = backend
    return raw


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
        description="VLM-only archaeological candidate scan with an OpenAI-compatible local VLM backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=default_config_path(), help="VLM YAML config path.")
    parser.add_argument(
        "--backend",
        dest="backend",
        help="Backend profili sec: config 'backends' bolumundeki bir anahtar (orn. lmstudio, llama).",
    )
    parser.add_argument("--input", dest="input", help="Input GeoTIFF path.")
    parser.add_argument("--out-prefix", dest="out_prefix", help="Output base name or prefix.")
    parser.add_argument("--output-root", dest="output_root", help="Root output directory.")
    parser.add_argument("--bands", dest="bands", help="Band spec: auto, 1,2,3, or 1,2,3,4,5.")
    parser.add_argument("--base-url", "--vlm-base-url", dest="base_url", help="OpenAI-compatible VLM base URL.")
    parser.add_argument("--api-key", "--vlm-api-key", dest="api_key", help="OpenAI-compatible VLM API key.")
    parser.add_argument("--model", "--vlm-model", dest="model", help="VLM model name or auto.")
    parser.add_argument("--tile", "--vlm-tile", dest="tile", type=int, help="Tile size in pixels.")
    parser.add_argument("--overlap", "--vlm-overlap", dest="overlap", type=int, help="Tile overlap in pixels.")
    parser.add_argument("--views", "--vlm-views", dest="views", help="VLM views: auto or CSV.")
    parser.add_argument(
        "--source-kind",
        "--vlm-source-kind",
        dest="source_kind",
        help=(
            "Input interpretation: auto, rgb, hillshade, slrm, svf, slope, single_band, "
            "dsm, dtm, mixed_topo, or terrain_derivative."
        ),
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
    parser.add_argument("--batch", dest="batch", action="store_true", default=None, help="Klasor girdisindeki her bir dosyayi ayri bir karo olarak sirayla tara ve birlesik cikti uret.")
    parser.add_argument("--no-batch", dest="batch", action="store_false", default=None, help="Batch modunu kapat (klasor girdilerini ek view'lar olarak algilar).")
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
        "--max-candidates-per-tile",
        "--vlm-max-candidates-per-tile",
        dest="max_candidates_per_tile",
        type=int,
        help="Maximum distinct first-stage candidates to keep from one tile.",
    )
    parser.add_argument(
        "--reload-every-tiles",
        "--vlm-reload-every-tiles",
        dest="reload_every_tiles",
        type=int,
        help="Unload/load LM Studio model every N new tiles; ignored for non-LM Studio backends; 0 disabled.",
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
    parser.add_argument("--max-tokens", "--vlm-max-tokens", dest="max_tokens", type=int, help="Maximum output tokens for each VLM response.")
    parser.add_argument("--log-level", dest="log_level", help="Logging level.")
    return parser


def build_config_from_args(args: argparse.Namespace) -> StandaloneVlmConfig:
    values = {field.name: getattr(StandaloneVlmConfig(), field.name) for field in fields(StandaloneVlmConfig)}
    yaml_fields: set[str] = set()
    yaml_dir: Optional[Path] = None

    if args.config:
        yaml_path = Path(args.config)
        raw_config = load_yaml_config(yaml_path)
        raw_config = _apply_backend_profile(raw_config, getattr(args, "backend", None))
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


def build_vlm_batch_folder_name(input_path: Path) -> str:
    stem = _safe_token(input_path.stem)[:28].rstrip("_-") or "input"
    return f"{SESSION_RUN_ID}_{stem}_merged_batch"


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


def _latest_vlm_resume_jsonl_mtime(folder: Path) -> Optional[float]:
    try:
        mtimes = [
            path.stat().st_mtime
            for path in folder.glob("*_vlm_candidates.jsonl")
            if path.is_file() and path.stat().st_size > 0
        ]
    except OSError:
        return None
    if not mtimes:
        return None
    return max(mtimes)


def find_latest_vlm_batch_folder(input_path: Path, output_root: Path) -> Optional[Path]:
    stem = _safe_token(input_path.stem)[:28].rstrip("_-") or "input"
    pattern = f"*_{stem}_merged_batch"
    try:
        candidates = []
        for path in output_root.glob(pattern):
            if not path.is_dir():
                continue
            latest_jsonl_mtime = _latest_vlm_resume_jsonl_mtime(path)
            if latest_jsonl_mtime is not None:
                candidates.append((latest_jsonl_mtime, path.stat().st_mtime, path))
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
    except Exception as exc:
        LOGGER.warning("VLM batch resume klasoru aranirken hata olustu: %s", exc)
        return None


def resolve_batch_output_folder(input_path: Path, config: StandaloneVlmConfig) -> Path:
    output_root = Path(config.output_root)
    if config.out_prefix:
        requested = _output_base_path(Path(config.out_prefix))
        return requested
    if config.resume:
        resume_folder = find_latest_vlm_batch_folder(input_path, output_root)
        if resume_folder is not None:
            LOGGER.info("VLM batch resume klasoru bulundu: %s", resume_folder)
            return resume_folder
    return output_root / build_vlm_batch_folder_name(input_path)


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
    if input_path.is_dir():
        if not spec or spec.lower() == "auto":
            return None
        return parse_manual_band_indexes(spec)
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


def _validate_startup_preconditions(input_path: Path, config: StandaloneVlmConfig) -> None:
    """Çalışma başlamadan önce kritik ön koşulları doğrula; manifest yazılmadan hata fırlat."""
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi GeoTIFF bulunamadi: {input_path}")
    if input_path.is_dir():
        has_raster = any(
            path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            for path in input_path.iterdir()
        )
        if not has_raster:
            has_raster = any(
                path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
                for path in input_path.rglob("*")
            )
        if not has_raster:
            raise FileNotFoundError(f"Girdi klasorunde GeoTIFF bulunamadi: {input_path}")

    for label, prompt_path in (
        ("Stage 1 prompt", config.prompt_stage1_path),
        ("Stage 2 prompt", config.prompt_stage2_path),
    ):
        if prompt_path:
            p = Path(prompt_path)
            if not p.exists():
                raise FileNotFoundError(f"{label} dosyasi bulunamadi: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"{label} dosyasi bos: {p}")

    parsed = urllib.parse.urlparse(str(config.base_url or ""))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"vlm_base_url gecerli bir HTTP/HTTPS URL olmali, alindi: {config.base_url!r}"
        )

    models_url = _normalize_openai_base_url(config.base_url).rstrip("/") + "/models"
    try:
        req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    except Exception as exc:
        backend_label = str(config.backend or "openai-compatible")
        LOGGER.warning(
            "%s VLM sunucusuna erisilemedi (%s): %s — sunucu hazir degil mi?",
            backend_label,
            models_url,
            exc,
        )


def write_run_manifest(out_prefix: Path, config: StandaloneVlmConfig, band_indexes: Optional[Tuple[int, ...]]) -> Path:
    manifest_path = _output_base_path(out_prefix).parent / f"{_output_base_path(out_prefix).name}_vlm_run_config.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["resolved_band_indexes"] = list(band_indexes) if band_indexes is not None else "auto"
    payload["session_run_id"] = SESSION_RUN_ID
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _load_prompt_for_run_info(prompt_path: Optional[str], default_text: str) -> Tuple[str, str]:
    if prompt_path is None:
        return "default", default_text.strip()
    path = Path(prompt_path)
    return str(path), path.read_text(encoding="utf-8").strip()


def _run_info_parameter_value(name: str, value: Any) -> Any:
    if name in {"api_key"}:
        return "<redacted>"
    return value


def write_run_info_txt(out_prefix: Path, config: StandaloneVlmConfig, band_indexes: Optional[Tuple[int, ...]]) -> Path:
    base = _output_base_path(out_prefix)
    info_path = base.parent / f"{base.name}_vlm_parameters_prompts.txt"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = asdict(config)
    parameters["resolved_band_indexes"] = list(band_indexes) if band_indexes is not None else "auto"
    parameters["session_run_id"] = SESSION_RUN_ID

    stage1_source, stage1_prompt = _load_prompt_for_run_info(config.prompt_stage1_path, _default_stage1_guidance())
    stage2_source, stage2_prompt = _load_prompt_for_run_info(config.prompt_stage2_path, _default_stage2_guidance())

    lines = [
        "VLM run parameters",
        "==================",
    ]
    for name in sorted(parameters):
        value = _run_info_parameter_value(name, parameters[name])
        lines.append(f"{name}: {value}")

    lines.extend(
        [
            "",
            "Stage 1 prompt",
            "==============",
            f"source: {stage1_source}",
            "",
            stage1_prompt,
            "",
            "Stage 2 prompt",
            "==============",
            f"source: {stage2_source}",
            "",
            stage2_prompt,
            "",
        ]
    )
    info_path.write_text("\n".join(lines), encoding="utf-8")
    return info_path


def run_batch(config: StandaloneVlmConfig) -> int:
    input_dir = Path(config.input)
    if not input_dir.is_dir():
        LOGGER.error("Girdi yolu bir klasor olmalidir: %s", input_dir)
        return 1

    tif_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tif_paths:
        tif_paths = sorted(
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )

    if not tif_paths:
        LOGGER.error("Klasor icinde hic .tif veya .tiff dosyasi bulunamadi: %s", input_dir)
        return 1

    LOGGER.info("Batch modunda toplam %d adet cografi karo (tile) bulundu.", len(tif_paths))
    for idx, path in enumerate(tif_paths, 1):
        LOGGER.info("  %d. %s", idx, path.name)

    combined_folder = resolve_batch_output_folder(input_dir, config)
    combined_folder.mkdir(parents=True, exist_ok=True)

    combined_paths = VlmOutputPaths(
        jsonl=combined_folder / "combined_vlm_candidates.jsonl",
        csv=combined_folder / "combined_vlm_candidates.csv",
        xlsx=combined_folder / "combined_vlm_candidates.xlsx",
        geojson=combined_folder / "combined_vlm_candidates.geojson",
        gpkg=combined_folder / "combined_vlm_stage2_verified.gpkg",
        gpkg_stage1=combined_folder / "combined_vlm_stage1_positives.gpkg",
        gpkg_stage2=combined_folder / "combined_vlm_stage2_verified.gpkg",
        raw_errors_jsonl=combined_folder / "combined_vlm_raw_errors.jsonl",
    )

    all_records: List[Dict[str, Any]] = []
    all_error_records: List[Dict[str, Any]] = []
    combined_crs = None

    for idx, tif_path in enumerate(tif_paths, 1):
        LOGGER.info("================================================================================")
        LOGGER.info("KARO TARAMA BASLIYOR (%d/%d): %s", idx, len(tif_paths), tif_path.name)
        LOGGER.info("================================================================================")

        try:
            with rasterio.open(tif_path) as src:
                if combined_crs is None:
                    combined_crs = src.crs

            tile_config = StandaloneVlmConfig(
                input=str(tif_path),
                out_prefix=str(combined_folder / tif_path.stem),
                output_root=config.output_root,
                bands=config.bands,
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                backend=config.backend,
                tile=config.tile,
                overlap=config.overlap,
                views=config.views,
                source_kind=config.source_kind,
                gsd_m=config.gsd_m,
                confidence_threshold=config.confidence_threshold,
                max_tiles=config.max_tiles,
                export_every=config.export_every,
                resume=config.resume,
                resume_jsonl_path=None,
                prompt_stage1_path=config.prompt_stage1_path,
                prompt_stage2_path=config.prompt_stage2_path,
                max_candidates_per_tile=config.max_candidates_per_tile,
                reload_every_tiles=config.reload_every_tiles,
                reload_pause_seconds=config.reload_pause_seconds,
                timeout=config.timeout,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                log_level=config.log_level,
                featureless_std_threshold=config.featureless_std_threshold,
                cross_tile_iou_threshold=config.cross_tile_iou_threshold,
                batch=False,
            )

            _validate_startup_preconditions(tif_path, tile_config)
            band_indexes = resolve_vlm_band_indexes(tif_path, tile_config.bands)
            sub_out_prefix = combined_folder / tif_path.stem
            write_run_manifest(sub_out_prefix, tile_config, band_indexes)
            write_run_info_txt(sub_out_prefix, tile_config, band_indexes)

            summary = run_vlm_lmstudio_detection(
                input_path=tif_path,
                out_prefix=sub_out_prefix,
                band_indexes=band_indexes,
                config=VlmLmStudioConfig(
                    base_url=tile_config.base_url,
                    api_key=tile_config.api_key,
                    model=tile_config.model,
                    backend=tile_config.backend,
                    tile=tile_config.tile,
                    overlap=tile_config.overlap,
                    views=tile_config.views,
                    source_kind=tile_config.source_kind,
                    gsd_m=tile_config.gsd_m,
                    confidence_threshold=tile_config.confidence_threshold,
                    max_tiles=tile_config.max_tiles,
                    timeout=tile_config.timeout,
                    temperature=tile_config.temperature,
                    max_tokens=tile_config.max_tokens,
                    export_every=tile_config.export_every,
                    resume=tile_config.resume,
                    resume_jsonl_path=sub_out_prefix.parent / f"{sub_out_prefix.name}_vlm_candidates.jsonl",
                    prompt_stage1_path=tile_config.prompt_stage1_path,
                    prompt_stage2_path=tile_config.prompt_stage2_path,
                    max_candidates_per_tile=tile_config.max_candidates_per_tile,
                    reload_every_tiles=tile_config.reload_every_tiles,
                    reload_pause_seconds=tile_config.reload_pause_seconds,
                    featureless_std_threshold=tile_config.featureless_std_threshold,
                    cross_tile_iou_threshold=tile_config.cross_tile_iou_threshold,
                ),
                logger=LOGGER,
            )

            tile_jsonl = sub_out_prefix.parent / f"{sub_out_prefix.name}_vlm_candidates.jsonl"
            tile_errors_jsonl = sub_out_prefix.parent / f"{sub_out_prefix.name}_vlm_raw_errors.jsonl"

            if tile_jsonl.exists():
                with tile_jsonl.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            record = json.loads(line)
                            record["batch_tile_name"] = tif_path.name
                            all_records.append(record)

            if tile_errors_jsonl.exists():
                with tile_errors_jsonl.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            err_record = json.loads(line)
                            err_record["batch_tile_name"] = tif_path.name
                            all_error_records.append(err_record)

        except Exception as exc:
            LOGGER.error("%s karosu taranirken kritik hata olustu: %s", tif_path.name, exc)
            continue

    if not all_records:
        LOGGER.warning("Taramalar sonucunda hicbir kayit uretilemedi.")
        return 0

    LOGGER.info("================================================================================")
    LOGGER.info("TUM KAROLAR TAMAMLANDI. BIRLESTIRILMIS CIKTILAR YAZILIYOR...")
    LOGGER.info("================================================================================")

    with combined_paths.jsonl.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    with combined_paths.raw_errors_jsonl.open("w", encoding="utf-8") as fh:
        for record in all_error_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    first_stage_records, candidate_records = _candidate_record_lists(all_records, config.confidence_threshold)

    _write_candidate_outputs(
        paths=combined_paths,
        records=candidate_records,
        crs=combined_crs,
        first_stage_records=first_stage_records,
        all_records=all_records,
        confidence_threshold=config.confidence_threshold,
        cross_tile_iou_threshold=config.cross_tile_iou_threshold,
        logger=LOGGER,
    )

    LOGGER.info("Birlesik batch tarama basariyla tamamlandi!")
    LOGGER.info("Birlesik Sonuclar Klasoru: %s", combined_folder)
    LOGGER.info("Birlesik JSONL: %s", combined_paths.jsonl)
    LOGGER.info("Birlesik CSV: %s", combined_paths.csv)
    if combined_paths.xlsx.exists():
        LOGGER.info("Birlesik Excel: %s", combined_paths.xlsx)
    LOGGER.info("Birlesik GeoJSON: %s", combined_paths.geojson)
    if combined_paths.gpkg_stage1.exists():
        LOGGER.info("Birlesik GPKG ilk asama pozitifler: %s", combined_paths.gpkg_stage1)
        LOGGER.info("Birlesik GPKG ikinci asama dogrulanan: %s", combined_paths.gpkg_stage2)

    return 0


def run(config: StandaloneVlmConfig) -> int:
    input_path = Path(config.input)
    _validate_startup_preconditions(input_path, config)

    if config.batch:
        if not input_path.is_dir():
            LOGGER.error("Batch modu icin girdi yolu bir klasor olmalidir: %s", input_path)
            return 1
        return run_batch(config)

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
    run_info_path = write_run_info_txt(out_prefix, config, band_indexes)
    LOGGER.info("VLM-only tarama basliyor")
    LOGGER.info("Girdi: %s", input_path)
    LOGGER.info("Cikti on eki: %s", out_prefix)
    LOGGER.info("Run config: %s", manifest_path)
    LOGGER.info("Parametre/prompt TXT: %s", run_info_path)

    summary = run_vlm_lmstudio_detection(
        input_path=input_path,
        out_prefix=out_prefix,
        band_indexes=band_indexes,
        config=VlmLmStudioConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            backend=config.backend,
            tile=config.tile,
            overlap=config.overlap,
            views=config.views,
            source_kind=config.source_kind,
            gsd_m=config.gsd_m,
            confidence_threshold=config.confidence_threshold,
            max_tiles=config.max_tiles,
            timeout=config.timeout,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            export_every=config.export_every,
            resume=config.resume,
            resume_jsonl_path=resume_jsonl_path,
            prompt_stage1_path=config.prompt_stage1_path,
            prompt_stage2_path=config.prompt_stage2_path,
            max_candidates_per_tile=config.max_candidates_per_tile,
            reload_every_tiles=config.reload_every_tiles,
            reload_pause_seconds=config.reload_pause_seconds,
            featureless_std_threshold=config.featureless_std_threshold,
            cross_tile_iou_threshold=config.cross_tile_iou_threshold,
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
