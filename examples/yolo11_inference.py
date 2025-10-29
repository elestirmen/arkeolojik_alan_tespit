#!/usr/bin/env python
"""
Helper script to run YOLOv11 detections on GeoTIFF inputs or derivatives
NPZ caches produced by the archaeo_detect pipeline. The script extracts
RGB bands, applies a percentile stretch to 8bit, runs Ultralytics YOLOv11,
and (optionally) writes detections to a GeoPackage for GIS usage. When
no explicit source is given the script reads `config.yaml` to decide
which raster/cache should be processed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import Affine
from rasterio.transform import xy as transform_xy

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - auxiliary script
    raise SystemExit(
        "Ultralytics is required for YOLOv11 inference. Install with `pip install ultralytics>=8.1.0`."
    ) from exc

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is optional at runtime
    gpd = None

try:
    from shapely.geometry import Polygon
except ImportError as exc:  # pragma: no cover - shapely is a core project dependency
    raise SystemExit(
        "Shapely is required to export detections. Install with `pip install shapely`."
    ) from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - required for config-driven mode
    raise SystemExit("PyYAML is required. Install with `pip install pyyaml`.") from exc


def parse_band_indexes(spec: str) -> Tuple[int, int, int]:
    """Parse comma separated band indexes."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Exactly three comma-separated band indexes are required (e.g. '1,2,3').")
    try:
        bands = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Band indexes must be integer values.") from exc
    if any(idx <= 0 for idx in bands):
        raise argparse.ArgumentTypeError("Band indexes are 1-based; provide values >= 1.")
    return bands  # type: ignore[return-value]


def stretch_to_uint8(rgb: np.ndarray, low: float, high: float) -> np.ndarray:
    """Stretch float/int bands to uint8 using percentile bounds."""
    stretched = np.zeros_like(rgb, dtype=np.uint8)
    for idx in range(rgb.shape[0]):
        band = rgb[idx].astype(np.float32)
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        lo = float(np.nanpercentile(band, low))
        hi = float(np.nanpercentile(band, high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(band[valid]))
            hi = float(np.nanmax(band[valid]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        scaled = np.zeros_like(band, dtype=np.float32)
        scaled[valid] = (band[valid] - lo) / (hi - lo)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        stretched[idx] = (scaled * 255.0).astype(np.uint8)
    return stretched


def xyxy_to_polygon(transform: Affine, xmin: float, ymin: float, xmax: float, ymax: float) -> Polygon:
    """Convert pixel-space bounding box to georeferenced polygon."""
    rows = [ymin, ymin, ymax, ymax, ymin]
    cols = [xmin, xmax, xmax, xmin, xmin]
    xs, ys = transform_xy(transform, rows, cols, offset="center")
    return Polygon(zip(xs, ys))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 detections on RGB bands of a GeoTIFF produced for archaeological mapping."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Path to the multi-band GeoTIFF or derivatives NPZ (expects RGB data). "
        "If omitted, the path is resolved from config.yaml.",
    )
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLOv11 weights file (e.g. 'yolo11n.pt').")
    parser.add_argument("--output", default="ciktilar/yolo11", help="Directory where inference artefacts will be stored.")
    parser.add_argument(
        "--bands",
        default="1,2,3",
        help="Comma separated band indexes for RGB extraction (1-based, default '1,2,3').",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO detections.")
    parser.add_argument("--device", default=None, help="Device override passed to Ultralytics (e.g. '0', 'cpu').")
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for 8-bit stretch (default: 2.0).",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for 8-bit stretch (default: 98.0).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for the YOLO run; defaults to '<source_stem>_yolo11'.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional GeoTIFF path used to recover transform/CRS when --source is a derivatives NPZ.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Pipeline configuration file used when --source is omitted (default: config.yaml).",
    )
    parser.add_argument(
        "--source-mode",
        choices=("auto", "npz", "geotiff"),
        default="auto",
        help="If --source is omitted: 'auto' prefers an existing derivatives NPZ; "
        "'npz' forces cache usage; 'geotiff' forces original raster.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size (pixels) for sliding-window inference. Set <=0 to disable tiling.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Overlap (pixels) between adjacent tiles when tiling is enabled.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save YOLO visualisations for each tile/run under the specified output directory.",
    )
    return parser.parse_args()


def resolve_reference_path(npz_path: Path, reference_arg: Optional[Path], metadata: dict) -> Optional[Path]:
    """Resolve a suitable reference GeoTIFF for georeferencing."""
    if reference_arg:
        ref = Path(reference_arg)
        return ref if ref.exists() else None

    input_path = metadata.get("input_path")
    if isinstance(input_path, str):
        candidate = Path(input_path)
        if candidate.exists():
            return candidate
        alt = npz_path.parent / candidate.name
        if alt.exists():
            return alt
    return None


def load_config_inputs(config_path: Path) -> Tuple[Path, Optional[Path]]:
    """Load input raster and optional cache directory from the YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if "input" not in data or data["input"] is None:
        raise ValueError("Config does not define an 'input' path.")

    base_dir = config_path.parent
    raw_input = data["input"]
    input_path = Path(raw_input)
    if not input_path.is_absolute():
        input_path = (base_dir / input_path).resolve()

    cache_dir_raw = data.get("cache_dir")
    cache_dir = None
    if cache_dir_raw:
        cache_dir = Path(cache_dir_raw)
        if not cache_dir.is_absolute():
            cache_dir = (base_dir / cache_dir).resolve()

    return input_path, cache_dir


def resolve_source_and_reference(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    """Determine source (GeoTIFF or NPZ) and an optional reference GeoTIFF."""
    if args.source:
        source_path = Path(args.source).resolve()
        reference_path = Path(args.reference).resolve() if args.reference else None
        return source_path, reference_path

    config_path = Path(args.config).resolve()
    input_path, cache_dir = load_config_inputs(config_path)

    if cache_dir is not None:
        derivative_path = cache_dir / f"{input_path.stem}_derivatives.npz"
    else:
        derivative_path = input_path.with_suffix(".derivatives.npz")

    chosen_source: Path
    mode = args.source_mode

    if mode in ("auto", "npz"):
        if derivative_path.exists():
            chosen_source = derivative_path
        elif mode == "npz":
            raise FileNotFoundError(f"Derivatives cache not found: {derivative_path}")
        else:
            chosen_source = input_path
    else:
        chosen_source = input_path

    reference_path = Path(args.reference).resolve() if args.reference else None
    if chosen_source.suffix.lower() == ".npz" and reference_path is None:
        reference_path = input_path

    return chosen_source.resolve(), reference_path


def compute_steps(length: int, tile: int, overlap: int) -> List[int]:
    """Compute tile start indices ensuring full coverage with desired overlap."""
    if tile <= 0 or tile >= length:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("Tile overlap must be smaller than tile size.")
    positions = list(range(0, max(1, length - tile + 1), stride))
    last_start = max(0, length - tile)
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    return sorted(set(max(0, p) for p in positions))


def load_rgb_data(
    source: Path, bands: Tuple[int, int, int], reference: Optional[Path]
) -> Tuple[np.ndarray, Optional[Affine], Optional[RasterioCRS]]:
    """Load RGB data and georeferencing info from GeoTIFF or derivatives NPZ."""
    if source.suffix.lower() == ".npz":
        npz = np.load(source, allow_pickle=True)
        if "rgb" not in npz:
            raise ValueError(f"'rgb' array not found in {source}")
        rgb = npz["rgb"].astype(np.float32)
        if rgb.shape[0] != 3:
            raise ValueError(f"'rgb' array must have 3 channels; found shape {rgb.shape}")
        metadata: dict = {}
        if "_metadata" in npz:
            meta_obj = npz["_metadata"]
            if hasattr(meta_obj, "item"):
                try:
                    metadata_candidate = meta_obj.item()
                    if isinstance(metadata_candidate, dict):
                        metadata = metadata_candidate
                except Exception:
                    metadata = {}
        ref_path = resolve_reference_path(source, reference, metadata)
        transform: Optional[Affine] = None
        crs: Optional[RasterioCRS] = None
        if ref_path and ref_path.exists():
            with rasterio.open(ref_path) as src:
                transform = src.transform
                crs = src.crs
        else:
            print(
                "Warning: No reference GeoTIFF found for georeferencing. "
                "Detections will be exported without geographic coordinates."
            )
        return rgb, transform, crs

    with rasterio.open(source) as src:
        rgb = src.read(bands).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return rgb, transform, crs


def main() -> None:
    args = parse_args()
    bands = parse_band_indexes(args.bands)
    source, reference = resolve_source_and_reference(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb, transform, crs = load_rgb_data(source, bands, reference)

    rgb_uint8 = stretch_to_uint8(rgb, args.low_percentile, args.high_percentile)
    image_rgb = np.transpose(rgb_uint8, (1, 2, 0))
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    height, width, _ = image_bgr.shape

    model = YOLO(args.weights)
    run_name = args.run_name or f"{source.stem}_yolo11"
    predict_kwargs = dict(
        conf=args.conf,
        verbose=False,
        save=args.save_vis,
        project=str(output_dir),
        exist_ok=True,
    )
    if args.device:
        predict_kwargs["device"] = args.device

    tile_size = args.tile_size
    overlap = args.overlap if args.overlap is not None else 0
    tiling_enabled = tile_size and tile_size > 0 and (tile_size < max(height, width))

    if tiling_enabled and overlap >= tile_size:
        raise ValueError("Overlap must be smaller than tile size when tiling is enabled.")

    records: List[dict] = []
    geometries: List[Optional[Polygon]] = []
    total_tiles = 0

    def run_predict(image: np.ndarray, name: str) -> "ultralytics.engine.results.Results":  # type: ignore[name-defined]
        kwargs = predict_kwargs.copy()
        kwargs["name"] = name
        results = model.predict(image, **kwargs)
        return results[0]

    if tiling_enabled:
        row_starts = compute_steps(height, tile_size, overlap)
        col_starts = compute_steps(width, tile_size, overlap)
        tile_count = 0
        for row_start in row_starts:
            row_end = min(row_start + tile_size, height)
            for col_start in col_starts:
                col_end = min(col_start + tile_size, width)
                tile = image_bgr[row_start:row_end, col_start:col_end]
                tile_name = (
                    f"{run_name}_r{row_start}_c{col_start}"
                    if args.save_vis
                    else run_name
                )
                result = run_predict(tile, tile_name)
                total_tiles += 1
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names
                for bbox, conf, cls_id in zip(boxes, confidences, classes):
                    xmin, ymin, xmax, ymax = bbox.tolist()
                    xmin_glob = xmin + col_start
                    xmax_glob = xmax + col_start
                    ymin_glob = ymin + row_start
                    ymax_glob = ymax + row_start
                    polygon = (
                        xyxy_to_polygon(transform, xmin_glob, ymin_glob, xmax_glob, ymax_glob)
                        if transform is not None
                        else None
                    )
                    center_row = (ymin_glob + ymax_glob) / 2.0
                    center_col = (xmin_glob + xmax_glob) / 2.0
                    if transform is not None:
                        geo_x, geo_y = transform_xy(transform, [center_row], [center_col], offset="center")
                        center_x = float(geo_x[0])
                        center_y = float(geo_y[0])
                    else:
                        center_x = None
                        center_y = None
                    record = {
                        "tile_row_start": int(row_start),
                        "tile_col_start": int(col_start),
                        "tile_row_end": int(row_end),
                        "tile_col_end": int(col_end),
                        "class_id": int(cls_id),
                        "class_name": names.get(int(cls_id), str(cls_id)),
                        "confidence": float(conf),
                        "pixel_xmin": float(xmin_glob),
                        "pixel_ymin": float(ymin_glob),
                        "pixel_xmax": float(xmax_glob),
                        "pixel_ymax": float(ymax_glob),
                        "center_x": center_x,
                        "center_y": center_y,
                        "geometry_wkt": polygon.wkt if polygon is not None else None,
                    }
                    records.append(record)
                    geometries.append(polygon)
                tile_count += 1
        if len(records) == 0:
            print(f"Tiled inference completed ({total_tiles} tiles); no detections above threshold.")
    else:
        result = run_predict(image_bgr, run_name)
        total_tiles = 1
        if result.boxes is None or len(result.boxes) == 0:
            print("YOLO inference completed; no detections above the confidence threshold.")
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for bbox, conf, cls_id in zip(boxes, confidences, classes):
                xmin, ymin, xmax, ymax = bbox.tolist()
                polygon = (
                    xyxy_to_polygon(transform, xmin, ymin, xmax, ymax) if transform is not None else None
                )
                center_row = (ymin + ymax) / 2.0
                center_col = (xmin + xmax) / 2.0
                if transform is not None:
                    geo_x, geo_y = transform_xy(transform, [center_row], [center_col], offset="center")
                    center_x = float(geo_x[0])
                    center_y = float(geo_y[0])
                else:
                    center_x = None
                    center_y = None
                record = {
                    "tile_row_start": 0,
                    "tile_col_start": 0,
                    "tile_row_end": height,
                    "tile_col_end": width,
                    "class_id": int(cls_id),
                    "class_name": names.get(int(cls_id), str(cls_id)),
                    "confidence": float(conf),
                    "pixel_xmin": float(xmin),
                    "pixel_ymin": float(ymin),
                    "pixel_xmax": float(xmax),
                    "pixel_ymax": float(ymax),
                    "center_x": center_x,
                    "center_y": center_y,
                    "geometry_wkt": polygon.wkt if polygon is not None else None,
                }
                records.append(record)
                geometries.append(polygon)

    if records:
        if (
            gpd is not None
            and transform is not None
            and crs is not None
            and any(geom is not None for geom in geometries)
        ):
            gpkg_path = output_dir / f"{run_name}_detections.gpkg"
            gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"Saved georeferenced detections to {gpkg_path}")
        else:
            csv_path = output_dir / f"{run_name}_detections.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
            print(f"Detections exported as CSV to {csv_path} (GeoPackage unavailable).")
    else:
        print("No detections recorded; nothing to export.")

    if args.save_vis:
        print(f"YOLO visual outputs saved under {output_dir / run_name}* directories.")
    print(f"Inference finished over {total_tiles} tile(s); detections: {len(records)}")


if __name__ == "__main__":
    main()
