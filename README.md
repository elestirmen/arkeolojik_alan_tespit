# 🏛️ Archaeological Site Detection (Deep Learning + Classical Image Processing)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Turkish documentation: [`README_TR.md`](README_TR.md).

> **Advanced AI system for automatic detection of archaeological structures from multi-band GeoTIFF data. Primarily designed for derivatives built from UAV (drone) nadir imagery—orthophotos, DSM/DTM, and stacked relief channels. Satellite imagery and other aerial/LiDAR sources are also supported when provided in the same multi-band GeoTIFF format.**

This project combines **deep learning** and **classical image processing** methods to detect archaeological traces (tumuli, ditches, mounds, wall remains, etc.) from multi-band GeoTIFF data (RGB, DSM, DTM). Input data is most commonly produced from **UAV photogrammetry**; **satellite imagery or other aerial products** can also be used as long as the band layout and georeferencing are compatible.

### Current default workflow (repository `config.yaml`)

If `config.local.yaml` exists, the CLI prefers it automatically; otherwise it falls back to `config.yaml`.

The checked-in profile targets **tile-level classification** (`dl_task: tile_classification`) with a **single trained checkpoint** (`trained_model_only: true`). In that mode:

- Use **`weights`** (your `.pth` file) and **`training_metadata`** (JSON from training).
- **`tile`**, **`overlap`**, and **`bands`** are taken from `training_metadata.json` during inference—do not "fix" mismatches by editing overlap in YAML; retrain with the desired overlap if needed.
- After a successful `training.py` run, the best weights are published to `workspace/checkpoints/active/model.pth` and metadata to `workspace/checkpoints/active/training_metadata.json` (you may point `weights` to another file in `workspace/checkpoints/active/` if you prefer).

**Model input channels (current code):** the deep-learning stack is **5 channels** — **R, G, B, SVF, SLRM** — in that order (`archeo_shared/channels.py` → `MODEL_CHANNEL_NAMES`). The GeoTIFF remains **5 bands** (RGB + DSM + DTM). **SVF** (Sky-View Factor) and **SLRM** (Simple Local Relief Model from RVT, computed on DTM) are **derived inside** `archaeo_detect.py` / the dataset scripts; they are not separate GeoTIFF bands. Older documentation that referred to a 12-channel tensor (nDSM, multi-scale TPI, extra RVT openness channels, etc.) describes a **previous schema**, not the current training + inference path.

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🎯 What It Does](#-what-it-does)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔗 Band Merge Tool (`veri_birlestir_rgb_dsm_dtm.py`)](#-band-merge-tool-veri_birlestir_rgb_dsm_dtmpy)
- [DSM to DTM Preprocessing (`dtm_uret.py`)](#dsm-to-dtm-preprocessing-dtm_uretpy)
- [🏷️ Ground Truth Labeling Tool (`ground_truth_kare_etiketleme_qt.py`)](#%EF%B8%8F-ground-truth-labeling-tool-ground_truth_kare_etiketleme_qtpy)
- [🗂️ Tile Classification Dataset Tool (`prepare_tile_classification_dataset.py`)](#%EF%B8%8F-tile-classification-dataset-tool-prepare_tile_classification_datasetpy)
- [🎮 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [📂 Output Files](#-output-files)
- [🔬 How It Works](#-how-it-works)
- [💡 Use Cases](#-use-cases)
- [🎨 Visualization](#-visualization)
- [⚡ Performance Optimization](#-performance-optimization)
- [🐛 Troubleshooting](#-troubleshooting)
- [❓ FAQ](#-faq)
- [🎓 Model Training Guide](#-model-training-guide)
- [🔬 Advanced Features](#-advanced-features)
- [📚 Technical Details](#-technical-details)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

### 🧠 Four Powerful Methods
- **Deep Learning**: U-Net, DeepLabV3+ and other modern segmentation architectures
- **YOLO11 (NEW!)**: Fast object detection and segmentation with Ultralytics YOLO11 + labeled terrain inventory 🏷️
  - ⚠️ **Note:** Fine-tuning required for nadir (bird's-eye) imagery (see docs/YOLO11_NADIR_TRAINING.md)
- **Classical Image Processing**: RVT (Relief Visualization Toolbox), Hessian matrix, Morphological operators
- **Hybrid Fusion**: Smart fusion combining strengths of each method

### 🎯 Smart Detection Features
- ✅ **Multi-Encoder Support**: ResNet, EfficientNet, VGG, DenseNet, MobileNet and more
- ✅ **Zero-Shot Learning**: Works even without trained models using ImageNet weights
- ✅ **Ensemble Learning**: Combines results from multiple encoders for more reliable detection
- ✅ **Multi-Scale Analysis**: Detects structures of different sizes
- ✅ **🆕 Labeled Object Detection**: Automatic labeling of 80 different object classes (trees, buildings, vehicles, etc.) with YOLO11
- ✅ **🆕 5-channel DL stack**: R, G, B from the raster plus **SVF** and **SLRM** (from DTM via RVT), assembled in code—not extra GeoTIFF bands
- ✅ **🆕 CBAM attention (optional)**: Supported in `training.py` when attention is enabled (off in the checked-in `CONFIG` by default)

### 🔧 Technical Features
- 🚀 **Tile-Based Processing**: Memory-efficient processing for large images
- 🎨 **Seamless Mosaicking**: No artifacts at tile boundaries with cosine feathering
- 📊 **Robust Normalization**: Global or local percentile-based normalization
- ⚡ **Cache System**: 10-100x speedup by caching RVT calculations
- 🎯 **Smart Masking**: Automatic filtering of tall structures (trees, buildings)
- 📐 **Vectorization**: Converts results to GIS-compatible polygons
- 🏷️ **Ground Truth Labeling**: Interactive Qt-based GeoTIFF annotation tool with layer management

### 🌐 GIS Integration
- 📁 Vector output in GeoPackage (.gpkg) format
- 🗺️ Geographic coordinate system (CRS) preserved
- 📏 Area calculation and filtering
- 🎯 Compatible with QGIS, ArcGIS and similar software

---

## 🎯 What It Does

This system can detect the following archaeological features:

| Structure Type | Description | Detection Method |
|----------------|-------------|------------------|
| 🏔️ **Tumuli** | Raised burial mounds | RVT + Hessian + DL |
| 🏛️ **Mounds** | Settlement mounds | All methods |
| 🧱 **Wall Remains** | Linear structure traces | Hessian + DL |
| ⭕ **Ring Ditches** | Circular defensive structures | Morphological + DL |
| 🏰 **Fortress Remains** | Large structure complexes | Fusion (most effective) |
| 🏺 **Settlement Traces** | Irregular topographic anomalies | Classical + DL |
| 🛤️ **Ancient Roads** | Linear elevation changes | Hessian + RVT |

---

## 🚀 Quick Start

### End-to-end: labels → tiles → train → detect

```bash
pip install -r requirements.txt

# 1a) Legacy paired tiles (images + masks) for segmentation or backward compatibility
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data

# 1b) Direct tile-classification dataset with explicit Positive/Negative folders
python prepare_tile_classification_dataset.py \
  --pair kesif_alani.tif ground_truth.tif \
  --output-dir workspace/training_data_classification \
  --sampling-mode selected_regions \
  --overwrite

# 2) Train (training.py now accepts both layouts)
python training.py --data workspace/training_data_classification --task tile_classification --epochs 50

# 3) Inference (uses config.yaml; publishes to workspace/checkpoints/active/ are described in training output)
python archaeo_detect.py
```

Artifacts after training:

- `workspace/checkpoints/active/model.pth` — best checkpoint copied for inference
- `workspace/checkpoints/active/training_metadata.json` — **source of truth** for `tile` / `overlap` / `bands` when `trained_model_only: true`

**Important:** With `trained_model_only: true`, do not raise `overlap` only in YAML to "match" training—metadata locks those fields. Change overlap in data prep + retrain if you need a different overlap.

**Without a trained model yet:** use zero-shot / classical paths (see [Usage](#-usage)) or start from `configs/tile_classification_baseline.example.yaml` if provided.

### Run detection only (dependencies already installed)

```bash
python archaeo_detect.py
```

Uses `config.yaml` (paths to input raster, methods, thresholds). Outputs go under `workspace/ciktilar/<session>/`.

### IDE / no CLI for data prep

`egitim_verisi_olusturma.py` includes a `CONFIG` dict (default `input`, `mask`, `output`, `tile_size`, `overlap`, `bands`, …). If you run the script without `--input` / `--mask`, it **requires** those keys to be set in `CONFIG`—there is no interactive file dialog for paths.

---

## 📦 Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8 GB | 16 GB+ |
| **Disk Space** | 2 GB | 5 GB+ |
| **GPU** | None (works with CPU) | NVIDIA CUDA-capable GPU |

### Step-by-Step Installation

#### 1️⃣ Check Python and Pip

```bash
python --version  # Should be Python 3.10 or higher
pip --version     # pip should be installed
```

#### 2️⃣ Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv310
.venv310\Scripts\activate

# Linux/Mac
python -m venv .venv310
source .venv310/bin/activate
```

**Note:** `.venv310` is optional. If you use Conda (e.g., `archeo`), you can skip/remove `.venv310`.

#### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
- `torch>=2.0.0` - PyTorch (deep learning)
- `torchvision>=0.15.0` - Image processing
- `segmentation-models-pytorch>=0.3.2` - Segmentation models
- `rasterio>=1.3.0` - Raster data read/write
- `fiona>=1.9.0` - Vector data processing
- `geopandas>=0.12.0` - Geographic data analysis
- `opencv-python>=4.7.0` - Image processing
- `scikit-image>=0.20.0` - Advanced image processing
- `scipy>=1.10.0` - Scientific computing
- `numpy>=1.24.0` - Numerical operations
- `rvt-py>=1.2.0` (Python < 3.11) or `rvt>=2.0.0` (Python >= 3.11) - Relief Visualization Toolbox
- `pyyaml>=6.0` - YAML configuration files

#### 4️⃣ GDAL Installation (Optional but Recommended)

**Windows:**
```bash
# Via OSGeo4W or Conda
conda install -c conda-forge gdal
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install gdal-bin python3-gdal
```

**Mac:**
```bash
brew install gdal
```

#### 5️⃣ GPU Support (Optional)

If you have an NVIDIA GPU, install CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU check:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

---

## 🔗 Band Merge Tool (`veri_birlestir_rgb_dsm_dtm.py`)

`veri_birlestir_rgb_dsm_dtm.py` combines separate RGB, DSM, and DTM GeoTIFF files into a single 5-band GeoTIFF ready for the detection and training pipeline.

**Band order in the output:**
| Band | Content |
|------|---------|
| 1 | Red (RGB) |
| 2 | Green (RGB) |
| 3 | Blue (RGB) |
| 4 | DSM |
| 5 | DTM |

### Quick Run

```bash
python veri_birlestir_rgb_dsm_dtm.py \
  --rgb-input veri/ortofoto_rgb.tif \
  --dsm-input veri/dsm.tif \
  --dtm-input veri/dtm.tif \
  --output veri/combined_5band.tif
```

### Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rgb-input` | _(required)_ | RGB raster (minimum 3 bands) |
| `--dsm-input` | _(required)_ | DSM raster (1 band) |
| `--dtm-input` | _(required)_ | DTM raster (1 band) |
| `--output` | _(required)_ | Output 5-band GeoTIFF path |
| `--nodata` | `-9999.0` | NoData value for output |
| `--compression` | `LZW` | TIFF compression (`LZW` \| `DEFLATE` \| `NONE`) |
| `--block-size` | `512` | TIFF tile/block size (rounded to nearest 16) |
| `--log-level` | `INFO` | Logging level (`DEBUG` \| `INFO` \| `WARNING` \| `ERROR`) |
| `--progress` / `--no-progress` | on | Progress bar toggle |

### Notes

- All inputs are resampled to the resolution and extent of the RGB raster.
- The script validates that source CRS values are compatible before merging.
- If run from an IDE without CLI arguments, set the `CONFIG` dict at the top of the script.

---

## DSM to DTM Preprocessing (`dtm_uret.py`)

`dtm_uret.py` converts DSM GeoTIFF or LAS/LAZ point cloud input into a DTM GeoTIFF.

Current processing flow in code:
- `[1/4]` Read input raster metadata.
- `[2/4]` Run PDAL SMRF pipeline.
- `[3/4]` Snap output DTM to the exact source grid (resolution/transform/extent).
- `[4/4]` If SMRF fails and fallback is enabled, run morphological fallback DTM.

### Quick Run

```bash
python dtm_uret.py \
  --input veri/karlik_dag_dsm.tif \
  --output veri/karlik_dag_dtm_smrf.tif \
  --progress
```

LAS/LAZ input example:

```bash
python dtm_uret.py \
  --input veri/karlik_dag_dsm.las \
  --output veri/karlik_dag_dtm_smrf.tif \
  --method smrf \
  --cell 0.5 \
  --progress
```

LAS/LAZ tiled example (large files):

```bash
python dtm_uret.py \
  --input veri/karlik_dag_dsm.las \
  --output veri/karlik_dag_dtm_smrf.tif \
  --method smrf \
  --smrf-tiled \
  --smrf-tile-size 4096 \
  --smrf-overlap-px 0 \
  --smrf-tile-workers 2 \
  --progress
```

### Key CLI Parameters

- `--input`, `--output`
- `--method` (`auto` | `smrf` | `fallback`, default: `fallback`)
- `--cell`, `--slope`, `--threshold`, `--window`, `--scalar` (SMRF parameters)
  - LAS/LAZ input: if `--cell` is not provided, sibling `*.tif/*.tiff` pixel size is preferred; otherwise LAS header density is used.
- `--las-crs` (LAS/LAZ output CRS override, e.g. `EPSG:32636`)
- `--smrf-max-pixels`, `--smrf-downsample-factor` (SMRF RAM control)
- `--smrf-tiled`, `--smrf-tile-size`, `--smrf-overlap-px` (quality-preserving low-RAM SMRF for raster and LAS/LAZ)
- `--smrf-tile-workers` (parallel tile workers for SMRF tiled mode)
- `--allow-fallback` / `--no-fallback`
- `--opening-meters`, `--smooth-sigma-px`, `--tile-size` (fallback tuning)
- `--nodata`, `--compression`, `--log-level`
- `--progress` / `--no-progress`
- Note: `fallback` is raster-only; LAS/LAZ input runs with SMRF.

### Dependencies and Environment Notes

- SMRF requires Python PDAL module in the same environment:

```bash
conda install -n <env_name> -c conda-forge pdal python-pdal
```

- Fallback method requires `scipy`.
- LAS/LAZ input requires PDAL (`readers.las` + `writers.gdal`).
- Default `smrf_max_pixels` is `120000000`; if input exceeds this, SMRF input is downsampled automatically to reduce memory pressure.
- `smrf_tiled=true` keeps quality by processing SMRF in overlapping tiles instead of downsampling.
- For large files and quality-critical runs, prefer:

```bash
python dtm_uret.py \
  --method smrf \
  --smrf-tiled \
  --smrf-max-pixels 0 \
  --smrf-downsample-factor 1.0 \
  --smrf-tile-size 4096 \
  --smrf-overlap-px 0
```

- On Windows, keep geospatial stack consistent in one environment (avoid mixing `pip` `gdal/rasterio` with conda GDAL libraries), otherwise GDAL plugin DLL errors may occur.
- Runtime defaults are defined in `dtm_uret.py` (`CONFIG` dict) and can be overridden via CLI.

---

## 🏷️ Ground Truth Labeling Tool (`ground_truth_kare_etiketleme_qt.py`)

Interactive Qt-based tool for creating binary ground truth masks on GeoTIFF imagery. Draw rectangles on a preview of your raster data and export pixel-accurate GeoTIFF masks for model training.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🖱️ Rectangle Drawing** | Left-click + drag to draw/erase annotation rectangles |
| **🔍 Zoom & Pan** | Mouse wheel to zoom, right-click to pan |
| **📐 Square Lock** | Constrain drawing to perfect squares |
| **↩️ Undo** | Full undo history (Ctrl+Z) |
| **🎨 Band Selection** | Auto-detects bands; dialog for multi-band files (RGB, BGR, NIR presets) |
| **🗂️ Layer Panel** | Side panel with visibility toggles, opacity slider, drag-reorder |
| **➕ Extra Layers** | Load additional GeoTIFF rasters as overlay layers |
| **💾 GeoTIFF Export** | Saves mask with source CRS, transform, and DEFLATE compression |
| **🖼️ Drag & Drop** | Drop `.tif` files directly onto the window |
| **🎨 Light Theme** | Modern light UI with gradient toolbar and styled controls |
| **🔌 Dual Backend** | Works with PySide6 or PyQt6 |

### 🚀 Quick Start

```bash
# No arguments — opens file dialog
python ground_truth_kare_etiketleme_qt.py

# With arguments
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --output kesif_alani_ground_truth.tif

# Resume editing an existing mask
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --existing-mask kesif_alani_ground_truth.tif

# Single-band DEM with preview downsample
python ground_truth_kare_etiketleme_qt.py \
  --input karlik_dag_dsm.tif \
  --preview-max-size 4096
```

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open GeoTIFF |
| `Ctrl+S` | Save mask |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+Z` | Undo |
| `D` | Draw mode |
| `E` | Erase mode |
| `S` | Toggle square lock |
| `F` | Fit to window |
| `W` | Invert mouse wheel direction |

### 📋 CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|--------:|
| `--input`, `-i` | Input GeoTIFF path | _(file dialog)_ |
| `--output`, `-o` | Output mask path | `<input>_ground_truth.tif` |
| `--existing-mask` | Pre-existing mask to continue editing | _(none)_ |
| `--preview-max-size` | Max preview dimension in pixels (0 = full res) | `0` |
| `--bands` | Comma-separated band indices for RGB display | `1,2,3` |
| `--positive-value` | Pixel value for positive class (1–255) | `1` |
| `--square-mode` | Start with square lock enabled | `false` |

### 🎵 Band Selection

When opening a file, the tool automatically detects the number of bands:

| Band Count | Behavior |
|:----------:|----------|
| **1** | Automatic grayscale — no dialog |
| **2** | Uses bands 1,2 — no dialog |
| **3+** | Shows **Band Selection Dialog** with presets |

**Available Presets (3+ bands):**
- **RGB (1, 2, 3)** — standard true-color
- **BGR (3, 2, 1)** — reversed band order
- **NIR (4, 3, 2)** — near-infrared false color (5+ bands)
- **Grayscale (Band 1)** — single band
- **Custom** — pick any band for R/G/B via spin boxes

### 🗂️ Layer Panel

The left-side panel manages display layers:

- **☑️ Visibility** — checkbox per layer to show/hide
- **🔀 Reorder** — drag items or use ⬆/⬇ buttons (top = foreground)
- **🎚️ Opacity** — slider (0–100%) per selected layer
- **➕ Add Layer** — load extra GeoTIFFs as visual overlays
- **➖ Remove Layer** — delete extra layers (base image and mask cannot be removed)

Default layers:
1. 🔴 **Maske** — the annotation overlay (red, semi-transparent)
2. 🖼️ **Ana Görüntü** — the base raster

### 🔧 Dependencies

```bash
pip install rasterio opencv-python numpy
pip install PySide6   # or: pip install PyQt6
```

---

## 🗂️ Tile Classification Dataset Tool (`prepare_tile_classification_dataset.py`)

Dedicated script for building an **explicit Positive/Negative tile classification dataset** from one or more raster + mask pairs. Unlike `egitim_verisi_olusturma.py` (which produces paired images/masks for segmentation or legacy tile_classification), this script outputs pre-sorted `Positive/` and `Negative/` folders directly, which is the canonical layout for `training.py --task tile_classification`.

### Output Layout

```
output_dir/
  train/
    Positive/   ← tiles where positive_ratio >= threshold
    Negative/
  val/
    Positive/
    Negative/
  test/          ← optional (when --test-ratio > 0)
    Positive/
    Negative/
  metadata.json
  tiles_manifest.csv
  tile_labels.csv
```

Each tile is a 5-channel `.npz` (or `.npy`) file compatible with `training.py tile_classification` mode (same order as `MODEL_CHANNEL_NAMES`).

### Quick Run

```bash
# Single raster + mask pair
python prepare_tile_classification_dataset.py \
  --pair kesif_alani.tif ground_truth.tif \
  --output-dir workspace/training_data_classification \
  --sampling-mode selected_regions \
  --overwrite

# Multiple sources
python prepare_tile_classification_dataset.py \
  --pair bolge1.tif bolge1_mask.tif \
  --pair bolge2.tif bolge2_mask.tif \
  --output-dir workspace/training_data_classification \
  --train-negative-keep-ratio 0.35
```

### Sampling Modes

| Mode | Description |
|------|-------------|
| `full_grid` | Slide over the entire raster with `tile_size` stride |
| `selected_regions` | Only emit tiles that overlap mask-marked regions (much smaller dataset, higher positive density) |

### Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pair RASTER MASK` | _(required)_ | Raster + mask pair; repeat for multiple sources |
| `--output-dir` | `workspace/training_data_classification` | Output root directory |
| `--tile-size` | `256` | Tile size in pixels |
| `--overlap` | `128` | Sliding-window overlap in pixels |
| `--bands` | `1,2,3,4,5` | 1-based band indices: R, G, B, DSM, DTM |
| `--sampling-mode` | `full_grid` | `full_grid` or `selected_regions` |
| `--positive-ratio-threshold` | `0.02` | Min fraction of positive pixels to call a tile Positive |
| `--valid-ratio-threshold` | `0.70` | Min valid-pixel fraction required to keep a tile |
| `--negative-to-positive-ratio` | `1.0` | Max negatives per positive (for `selected_regions` mode) |
| `--train-ratio` | `0.8` | Train fraction of total tiles |
| `--val-ratio` | `0.2` | Val fraction of total tiles |
| `--test-ratio` | `0.0` | Optional held-out test split |
| `--train-negative-keep-ratio` | `0.35` | Fraction of all-negative **train** tiles to keep |
| `--train-negative-max` | `None` | Optional hard cap on kept negative train tiles |
| `--normalize` / `--no-normalize` | on | Apply robust 2-98% normalization to stacked channels |
| `--format` | `npz` | Output format: `npz` or `npy` |
| `--num-workers` | auto | Parallel worker processes |
| `--derivative-cache-mode` | `auto` | `none` \| `auto` \| `npz` \| `raster` |
| `--derivative-cache-dir` | _(sibling `workspace/cache/`)_ | Where to store/read RVT derivative cache |
| `--recalculate-derivative-cache` | off | Force re-computation even if cache exists |
| `--tile-prefix` | `""` | Optional prefix for tile filenames |
| `--seed` | `42` | Random seed for reproducible splits |
| `--overwrite` | off | Delete and recreate output directory if it exists |

### Manifest Files

The script writes two companion CSV files alongside `metadata.json`:

- **`tiles_manifest.csv`** — full record per tile: `tile_name`, `split`, `label`, `image_relpath`, `source_name`, `row_off`, `col_off`, `positive_ratio`, `valid_ratio`
- **`tile_labels.csv`** — compact label index: `tile_name`, `split`, `tile_label`, `positive_ratio`, used by `training.py` for fast label counts without scanning files

---

## 🎮 Usage

### Basic Usage

#### Running with Default Settings

```bash
python archaeo_detect.py
```

This command uses settings from the `config.yaml` file and processes the GeoTIFF file defined as input.

#### Running with Command-Line Parameters

```bash
# Change threshold value
python archaeo_detect.py --th 0.7

# Adjust tile size
python archaeo_detect.py --tile 512 --overlap 128

# Enable verbose mode (detailed log)
python archaeo_detect.py -v

# Use a different input file
python archaeo_detect.py --input new_area.tif

# Multiple parameters
python archaeo_detect.py --th 0.7 --tile 1024 --enable-fusion -v
```

### Common Usage Examples

#### 🔰 Example 1: First-Time Use (Zero-Shot)

Without trained models, using only ImageNet weights:

```bash
python archaeo_detect.py \
  --encoder resnet34 \
  --zero-shot-imagenet \
  --enable-classic \
  --enable-fusion \
  -v
```

#### 🎯 Example 2: Classical Method Only (Fast)

If no GPU or for quick testing:

```bash
python archaeo_detect.py \
  --no-enable-deep-learning \
  --enable-classic \
  --classic-modes combo \
  --cache-derivatives
```

#### 🚀 Example 3: Ensemble (Multi-Encoder)

For highest accuracy with multiple encoders:

```bash
python archaeo_detect.py \
  --encoders all \
  --enable-deep-learning \
  --enable-classic \
  --enable-fusion \
  --fuse-encoders all \
  --cache-derivatives \
  -v
```

#### 🎨 Example 4: With Custom Trained Model

With your own trained model:

```bash
python archaeo_detect.py \
  --encoder resnet50 \
  --weights models/my_trained_model.pth \
  --th 0.65 \
  --enable-classic \
  --enable-fusion \
  --alpha 0.7
```

#### 📊 Example 5: Large Area Analysis (Optimized)

Optimized settings for a wide area:

```bash
python archaeo_detect.py \
  --tile 2048 \
  --overlap 512 \
  --half \
  --global-norm \
  --cache-derivatives \
  --enable-fusion \
  --min-area 100 \
  -v
```

### Command-Line Parameters (Full List)

```bash
python archaeo_detect.py --help
```

**Important Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input` | Input GeoTIFF file | `--input area.tif` |
| `--th` | DL threshold (0-1) | `--th 0.7` |
| `--tile` | Tile size (pixels) | `--tile 1024` |
| `--overlap` | Overlap amount | `--overlap 128` |
| `--encoder` | Single encoder selection | `--encoder resnet34` |
| `--encoders` | Multi-encoder mode | `--encoders all` |
| `--alpha` | Fusion weight | `--alpha 0.6` |
| `--enable-fusion` | Enable fusion | (flag) |
| `--cache-derivatives` | Use cache | (flag) |
| `-v` or `--verbose` | Detailed log | (flag) |

---

## ⚙️ Configuration

### config.yaml File

System behavior is controlled by the `config.yaml` file. This file is **richly documented** (including Turkish inline comments) with detailed explanations.

**Path resolution:** Relative paths inside YAML are resolved **relative to the directory containing `config.yaml`**, not necessarily the process working directory. Paths you pass on the CLI are resolved relative to the **current working directory**.

#### Main Sections:

1. **Input/Output**: File paths and band selection
2. **Method Selection**: `enable_deep_learning`, `enable_classic`, `enable_yolo`, `enable_fusion`
3. **DL task**: `dl_task` — `segmentation` (per-pixel) or `tile_classification` (tile score → risk map with overlap blending)
4. **Trained-only mode**: `trained_model_only` — when `true`, enforces a single checkpoint + metadata (`weights`, `training_metadata`); locks tile/overlap/bands from metadata
5. **Deep Learning**: Architecture, encoder, weights, `zero_shot_imagenet`, attention / band importance (`save_band_importance`, `band_importance_max_tiles`)
6. **Classical Methods**: RVT, Hessian, Morphology parameters
7. **Advanced Topographic Analysis (legacy / off in default preset)**: `enable_curvature`, `enable_tpi`, `tpi_radii` still exist in `config.yaml` and `archaeo_detect.py` for experimentation, but the **checked-in 5-channel DL schema does not add curvature/TPI to the model tensor** (see comments at the top of `config.yaml`).
8. **Fusion**: Hybrid combination settings (`alpha`, …) — requires both DL and classic enabled
9. **YOLO11** (optional): Separate RGB-only inventory / segmentation path; usually off for the tile-classification preset
10. **Tile Processing**: Memory and performance optimization (`tile` / `overlap` documented vs metadata-locked)
11. **Normalization**: Data preprocessing
12. **Masking**: Filtering tall structures (`mask_talls`, `rgb_only`)
13. **Vectorization**: GIS output (`vectorize`, `min_area`, `export_candidate_excel`, …)
14. **Performance**: Device, `half`, `seed`, `verbose`; automatic OOM guard for large rasters (full-raster derivative precompute is skipped if available RAM is insufficient)
15. **Cache**: `cache_derivatives`, `cache_derivatives_mode` (`auto` / `npz` / `raster`), raster cache tuning

#### Quick Configuration Scenarios:

**Scenario 1: Deep Learning Only**
```yaml
enable_deep_learning: true
enable_classic: false
enable_fusion: false
encoder: "resnet34"
zero_shot_imagenet: true
```

**Scenario 2: Classical Method Only**
```yaml
enable_deep_learning: false
enable_classic: true
enable_fusion: false
classic_modes: "combo"
cache_derivatives: true
```

**Scenario 3: Hybrid (Best Results)**
```yaml
enable_deep_learning: true
enable_classic: true
enable_fusion: true
alpha: 0.5
encoders: "all"
cache_derivatives: true
```

### Data Preparation

#### Input File Requirements:

✅ **GeoTIFF format** (.tif or .tiff)  
✅ **Multi-band** (at least 3 bands: RGB)  
✅ **Same grid** (all bands same resolution and extent)  
✅ **Geographic reference** (CRS/EPSG code)

#### Recommended Band Structure:

| Band # | Content | Description |
|--------|---------|-------------|
| 1 | Red | RGB's R component |
| 2 | Green | RGB's G component |
| 3 | Blue | RGB's B component |
| 4 | DSM | Digital Surface Model (elevation) |
| 5 | DTM | Digital Terrain Model (ground elevation) |

#### Data Creation Example (GDAL):

```bash
# Combine separate RGB and elevation files
gdal_merge.py -separate -o combined.tif \
  red.tif green.tif blue.tif dsm.tif dtm.tif

# Resampling (equalizing different resolutions)
gdalwarp -tr 1.0 1.0 -r bilinear input.tif output.tif

# Assign coordinate system
gdal_edit.py -a_srs EPSG:32635 output.tif
```

---

## 📂 Output Files

When the system runs, the following files are created:

All outputs are written under:

```
workspace/ciktilar/<session_folder>/<out_name>*
```

`<session_folder>` uses a compact format:
`<timestamp>_<input>_<methods>_t<tile>o<overlap>_m-<model>`
(example model tokens: `m-<checkpoint>`, `m-zs`, `m-<encoder>`).

Each session folder also includes:

```
run_params.txt
```

This file contains all effective run parameters (final config values, parsed bands, CLI args, and device).
If enabled (`save_band_importance: true`), DL runs also write
`*_band_importance.txt` and `*_band_importance.json`.

### 📊 Raster Outputs (GeoTIFF)

#### 1️⃣ Deep Learning Outputs

**Single Encoder:**
```
kesif_alani_prob.tif     → Probability map (continuous values 0.0-1.0)
kesif_alani_mask.tif     → Binary mask (0: not archaeological, 1: archaeological area)
```

**Multi-Encoder:**
```
kesif_alani_resnet34_prob.tif
kesif_alani_resnet34_mask.tif
kesif_alani_resnet50_prob.tif
kesif_alani_resnet50_mask.tif
kesif_alani_efficientnet-b3_prob.tif
kesif_alani_efficientnet-b3_mask.tif
```

#### 2️⃣ Classical Method Outputs

```
kesif_alani_classic_prob.tif     → Combined classical probability
kesif_alani_classic_mask.tif     → Classical binary mask
```

**Intermediate Files (classic_save_intermediate: true):**
```
kesif_alani_classic_rvtlog_prob.tif    → RVT method only
kesif_alani_classic_hessian_prob.tif   → Hessian method only
kesif_alani_classic_morph_prob.tif     → Morphology method only
```

#### 3️⃣ Fusion Outputs

```
kesif_alani_fused_resnet34_prob.tif
kesif_alani_fused_resnet34_mask.tif
```

### 📍 Vector Outputs (GeoPackage)

```
kesif_alani_mask.gpkg                → DL vector polygons
kesif_alani_classic_mask.gpkg        → Classical vector polygons
kesif_alani_fused_resnet34_mask.gpkg → Fusion vector polygons
```

When `export_candidate_excel: true` in `config.yaml`, companion `*_gps.xlsx` files are written next to the vector outputs (candidate centers / GPS-style tables for field checks).

**GeoPackage Features:**
- Polygon geometry
- Area information (in m²)
- CRS information preserved
- Can be opened directly in QGIS/ArcGIS

### 💾 Cache Files

**Cache Directory Structure:**
```
workspace/cache/
├── kesif_alani.a1b2c3d4e5f6.derivatives.npz    → RVT derivatives cache
└── karlik_vadi.f6e5d4c3b2a1.derivatives.npz   → RVT derivatives cache
```

**Cache System:**
- RVT calculations are cached in `.npz` format
- Cache files are stored in the `workspace/cache/` directory (configurable via `cache_dir` in config.yaml)
- Cache validation checks file name and modification time
- **Important:** Cache files are reusable even if the project folder is moved (file name-based validation)
- Provides 10-100x speedup on subsequent runs
- Cache files are typically 10-50 MB, but can be larger for high-resolution data

**Cache Configuration:**
```yaml
cache_derivatives: true      # Enable caching
cache_dir: "workspace/cache/"          # Cache directory (relative to project root)
recalculate_cache: false     # Don't recalculate if cache exists
```

### 📋 File Naming Logic

Output files are automatically named in the following format:

```
<prefix>_[method]_[encoder]_[params]_[type].ext
```

Example:
```
kesif_alani_fused_resnet34_th0.6_tile1024_alpha0.5_prob.tif
```

**Parameters:**
- `th`: Threshold value
- `tile`: Tile size
- `alpha`: Fusion ratio
- `minarea`: Minimum area
- And others...

---

## 🔬 How It Works

### Workflow Overview

```
┌─────────────────────┐
│  GeoTIFF Input      │
│ (RGB, DSM, DTM)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Preprocessing │
│  - Band reading     │
│  - Normalization    │
│  - Masking          │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Deep    │ │ Classical│
│ Learning│ │ Methods  │
│ (U-Net) │ │ (RVT)    │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           ▼
   ┌───────────────┐
   │    Fusion     │
   │  (Combine)    │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  Thresholding │
   │  (Prob → Mask)│
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ Vectorization │
   │  (GeoPackage) │
   └───────────────┘
```

### 1️⃣ Deep Learning Method

**Steps:**

1. **Build the 5-channel DL tensor**
   - Read **RGB** and **DSM/DTM** from the GeoTIFF (bands chosen in `config.yaml`)
   - On the filled DTM, compute **SVF** and **SLRM** with RVT (`compute_derivatives_with_rvt` in `archaeo_detect.py`)
   - Concatenate with `stack_channels(rgb, svf, slrm)` → shape `(5, H, W)` in channel order `MODEL_CHANNEL_NAMES`

   Other RVT products (openness, slope, etc.) may still be used on **classical** or experimental code paths; they are **not** separate planes in this DL stack.

2. **Normalization**
   - Global or local percentile-based
   - Scaling to 2%-98% range

3. **Tile-Based Processing**
   - Large image divided into small tiles
   - Each tile fed to U-Net
   - Probability map generated

4. **Feathering (Smoothing)**
   - Transitions between tiles smoothed
   - Seamless mosaic created

5. **Thresholding**
   - Probability > threshold → Mask = 1
   - Probability ≤ threshold → Mask = 0

### 2️⃣ Classical Image Processing

**Three Sub-Methods:**

**A) RVT (Relief Visualization)**
- SVF, Openness calculations
- Relief visualization
- Ideal for tumuli and mounds

**B) Hessian Matrix**
- Second derivative analysis
- Ridge and valley detection
- Effective for walls and ditches

**C) Morphological Operators**
- Opening, closing
- Top-hat transformations
- Local texture features

**Combination:**
- Each method produces 0-1 score
- Scores averaged (combo mode)
- Otsu or manual thresholding applied

### 3️⃣ Fusion (Hybrid Combination)

**Formula:**
```
P_fused = α × P_deep_learning + (1 - α) × P_classic
```

**Advantages:**
- Deep learning: Complex patterns
- Classical: Reliable elevation features
- Fusion: Strengths of both

**Example:**
- α = 0.5: Equal weight
- α = 0.7: Priority to DL
- α = 0.3: Priority to classical

---

## 💡 Use Cases

### 📍 Scenario 1: New Area Discovery

**Situation:** First scan of an unexplored area

**Recommended Settings:**
```bash
python archaeo_detect.py \
  --encoders all \
  --enable-classic \
  --enable-fusion \
  --th 0.5 \
  --classic-th null \
  --alpha 0.5 \
  --min-area 50 \
  --cache-derivatives \
  -v
```

**Why these settings?**
- Multi-encoder: Maximum detection sensitivity
- Low threshold: Catch all candidates
- Low min_area: Don't miss small structures
- Cache: Speedup for repeated analysis

### 🎯 Scenario 2: Detailed Analysis of Known Area

**Situation:** Detailed examination of a previously detected area

**Recommended Settings:**
```bash
python archaeo_detect.py \
  --encoder efficientnet-b3 \
  --weights models/my_tuned_model.pth \
  --th 0.65 \
  --enable-classic \
  --alpha 0.6 \
  --min-area 80 \
  --simplify 2.0 \
  -v
```

**Why these settings?**
- Custom model: Region-specific trained model
- High threshold: Only reliable detections
- Simplify: Clean polygons

### ⚡ Scenario 3: Quick Preliminary Assessment

**Situation:** To quickly get an idea

**Recommended Settings:**
```bash
python archaeo_detect.py \
  --no-enable-deep-learning \
  --enable-classic \
  --classic-modes rvtlog \
  --tile 512 \
  --no-vectorize \
  --cache-derivatives
```

**Why these settings?**
- Classical only: Fastest method
- Small tiles: Less memory
- No vector: Time saving

### 🔬 Scenario 4: Research and Comparison

**Situation:** Comparative analysis of different methods

**Recommended Settings:**
```bash
python archaeo_detect.py \
  --encoders all \
  --enable-classic \
  --classic-save-intermediate \
  --enable-fusion \
  --fuse-encoders all \
  --cache-derivatives \
  -v
```

**Why these settings?**
- All methods active
- Intermediate files: See each method's contribution
- All fusion: Try every combination

---

## 🎨 Visualization

### Viewing in QGIS

#### 1️⃣ Loading Probability Maps

```
Layer → Add Layer → Add Raster Layer
```

**Recommended Color Scheme:**
- 0.0-0.3: Blue (Low probability)
- 0.3-0.5: Yellow (Medium probability)
- 0.5-0.7: Orange (High probability)
- 0.7-1.0: Red (Very high probability)

#### 2️⃣ Viewing Vector Polygons

```
Layer → Add Layer → Add Vector Layer → Select GeoPackage
```

**Style Suggestions:**
- Fill: Semi-transparent red (opacity: 50%)
- Line: Thick red (2 pixels)
- Label: Area value (m²)

#### 3️⃣ Overlay with Base Map

```python
# QGIS Python Console
from qgis.core import QgsRasterLayer

# Add orthophoto
ortho = QgsRasterLayer('kesif_alani.tif', 'Orthophoto')
QgsProject.instance().addMapLayer(ortho)

# Add mask (semi-transparent)
mask = QgsRasterLayer('kesif_alani_mask.tif', 'Detection')
QgsProject.instance().addMapLayer(mask)
mask.renderer().setOpacity(0.6)
```

### Python Visualization

```python
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Read probability map
with rasterio.open('kesif_alani_prob.tif') as src:
    prob = src.read(1)

# Custom color palette
colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
cmap = LinearSegmentedColormap.from_list('archaeo', colors)

# Visualize
plt.figure(figsize=(12, 10))
plt.imshow(prob, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(label='Archaeological Site Probability')
plt.title('Archaeological Site Detection Results')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.tight_layout()
plt.savefig('result_visualization.png', dpi=300)
plt.show()
```

### Web-Based Visualization

```python
import folium
import geopandas as gpd

# Read vector
gdf = gpd.read_file('kesif_alani_mask.gpkg')

# Create map
m = folium.Map(
    location=[gdf.geometry.centroid.y.mean(), 
              gdf.geometry.centroid.x.mean()],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Add polygons
for idx, row in gdf.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 2,
            'fillOpacity': 0.5
        },
        tooltip=f"Area: {row.get('area', 0):.1f} m²"
    ).add_to(m)

# Save
m.save('interactive_map.html')
print("Map created: interactive_map.html")
```

---

## ⚡ Performance Optimization

### GPU Usage

#### CUDA Check
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

#### GPU Acceleration
```bash
# 2x speedup with mixed precision (FP16)
python archaeo_detect.py --half

# Fill GPU with large tiles
python archaeo_detect.py --tile 2048 --overlap 512
```

### Memory Optimization

#### Low Memory Situation
```bash
python archaeo_detect.py \
  --tile 512 \
  --overlap 64 \
  --no-enable-deep-learning \
  --enable-classic
```

#### High Memory Situation
```bash
python archaeo_detect.py \
  --tile 4096 \
  --overlap 1024 \
  --half \
  --encoders all
```

### Cache Strategy

```bash
# First run: Create cache
python archaeo_detect.py --cache-derivatives

# Subsequent runs: 10-100x faster!
# Cache is automatically used

# Recalculate cache when changing parameters
python archaeo_detect.py --recalculate-cache
```

**Cache Benefits:**
- RVT derivatives are calculated once and cached
- Cache files stored in `workspace/cache/` directory
- Cache validation is flexible: works even if project folder is moved
- File name and modification time are checked for validation
- Significant time savings on repeated runs

### Parallel Processing

For multiple areas, run in parallel:

```bash
# Bash script
for file in area1.tif area2.tif area3.tif; do
  python archaeo_detect.py --input $file &
done
wait
```

### Performance Comparison

| Configuration | Processing Time | Memory Usage | Quality |
|---------------|----------------|--------------|---------|
| **Minimum** (CPU, 512 tile) | ~30 min | 4 GB | Low |
| **Balanced** (GPU, 1024 tile) | ~5 min | 8 GB | Medium |
| **Maximum** (GPU, 2048 tile, ensemble) | ~15 min | 16 GB | High |

*Estimated times for 10 km² area (1m resolution)*

---

## 🐛 Troubleshooting

### Common Errors and Solutions

#### ❌ Error 1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
```bash
# Solution 1: Reduce tile size
python archaeo_detect.py --tile 512

# Solution 2: Use mixed precision
python archaeo_detect.py --half

# Solution 3: Use CPU
python archaeo_detect.py --device cpu
```

#### ❌ Error 2: RVT Import Error

```
ModuleNotFoundError: No module named 'rvt'
```

**Solution:**
```bash
# Python 3.10
pip install rvt-py

# Python 3.11+
pip install rvt

# Or via conda
conda install -c conda-forge rvt
```

#### ❌ Error 3: Empty Output

```
Warning: No detections found
```

**Solutions:**
1. Lower threshold value:
   ```bash
   python archaeo_detect.py --th 0.3 --classic-th 0.3
   ```

2. Lower minimum area:
   ```bash
   python archaeo_detect.py --min-area 20
   ```

3. Check in verbose mode:
   ```bash
   python archaeo_detect.py -v
   ```

#### ❌ Error 4: Classical Method Not Working

```
Error: DTM band not found
```

**Solution:**
Check bands in `config.yaml`:
```yaml
bands: "1,2,3,4,5"  # Band 5 should be DTM
# DTM is required (0 is invalid)
# If missing, generate/provide a valid DTM band first.
```

#### ❌ Error 5: Lines at Tile Boundaries

**Solution:**
```bash
# Increase overlap and enable feathering
python archaeo_detect.py --overlap 512 --feather
```

#### ❌ Error 6: Cache Not Being Used

**Symptoms:** System recalculates RVT derivatives even when cache files exist

**Solutions:**
1. Check cache directory path in `config.yaml`:
   ```yaml
   cache_dir: "workspace/cache/"  # Should match your cache directory
   ```

2. Verify cache file naming:
   - NPZ cache (default for small/medium rasters): `<input_name>.<cache_hash>.derivatives.npz`
   - Raster cache (block-based; used automatically for very large rasters or with `cache_derivatives_mode: "raster"`):
     - `<input_name>.<cache_hash>.derivatives_raster.tif`
     - `<input_name>.<cache_hash>.derivatives_raster.json`
   - Example for input `kesif_alani.tif`:
     - `kesif_alani.a1b2c3d4e5f6.derivatives.npz`
     - `kesif_alani.a1b2c3d4e5f6.derivatives_raster.tif`

3. Check cache validation:
   - Cache validation checks file name and modification time
   - If input file was moved, cache should still work (file name-based validation)
   - If input file was modified, cache will be recalculated

4. Enable verbose mode to see cache status:
   ```bash
   python archaeo_detect.py --cache-derivatives -v
   ```

#### ❌ Error 7: Training Script Import Errors

**Symptoms:**
```
HATA: segmentation-models-pytorch kurulu değil!
HATA: archaeo_detect.py'den attention modülleri import edilemedi.
```

**Solutions:**
1. **Install missing packages**:
   ```bash
   pip install segmentation-models-pytorch
   ```

2. **Check Python path**: Ensure `archaeo_detect.py` is in the same directory or in Python path

3. **Verify installation**: Run `python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"`

#### ❌ Error 8: Training Data Format Mismatch

**Symptoms:**
```
ValueError: Expected 5 channels but got X
```

**Solutions:**
1. **Regenerate training data**: Use `egitim_verisi_olusturma.py` with correct parameters
2. **Check metadata.json**: Verify `num_channels` matches actual data
3. **Verify file format**: Ensure `.npz` files contain an `image` key with shape `(5, H, W)` for the current schema

### Debug Mode

For detailed debugging:

```bash
python archaeo_detect.py --verbose 2 2>&1 | tee debug_log.txt
```

This command writes all debug messages to both screen and `debug_log.txt` file.

### Training Script Debugging

**Check training data:**
```bash
# Verify training data structure
ls -R workspace/training_data/
# Should show: train/images/, train/masks/, val/images/, val/masks/

# Check metadata
cat workspace/training_data/metadata.json | python -m json.tool
```

**Test data loading:**
```python
# Quick test script
import numpy as np
from pathlib import Path

data_dir = Path("workspace/training_data")
train_images = list((data_dir / "train" / "images").glob("*.npz"))
if train_images:
    sample = np.load(train_images[0])
    print(f"Keys: {sample.files}")
    if 'image' in sample.files:
        img = sample['image']
        print(f"Image shape: {img.shape}")
        print(f"Expected: (5, 256, 256), Got: {img.shape}")
```

**Monitor training in real-time:**
```bash
# Watch training history file
watch -n 5 'tail -20 workspace/checkpoints/training_history.json'
```

---

## ❓ FAQ

### 🤔 General Questions

**Q: I don't have a trained model, can I still use it?**  
A: Yes! Use `zero_shot_imagenet: true` to use ImageNet weights. Also, classical methods don't require models.

**Q: I don't have a GPU, will it work with CPU?**  
A: Yes, but it will be slower. Prefer classical methods or use small tile size.

**Q: Which method gives the best results?**  
A: Generally **fusion** (DL + Classical) gives the best results. However, it varies based on your data quality and region.

**Q: UAV vs satellite—which source does this work with?**  
A: The system is **primarily designed for UAV (drone) nadir imagery** (orthomosaic, DSM, DTM, and the derived channels this repo generates). **Satellite imagery is also supported**—provide a compatible multi-band GeoTIFF (RGB, plus DSM/DTM if available) on an aligned grid and the same pipeline runs. LiDAR-based surfaces and other sensors work the same way. What matters is consistent band structure and georeferencing, not the platform.

### 🔧 Technical Questions

**Q: How many bands are required?**  
A: Minimum 3 bands (RGB). For the current pipeline, use **5 bands** (RGB + DSM + DTM). The **model tensor** is **5 channels**: R, G, B plus **SVF** and **SLRM** computed from the DTM inside the code (see `stack_channels()` in `archaeo_detect.py`).

**Q: How much space do cache files take?**  
A: Typically 10-50 MB. Depends on input file size. Can be larger (several GB) for high-resolution data.

**Q: How can I improve results?**  
A: 
1. Use multiple encoders (ensemble)
2. Enable fusion
3. Optimize threshold values
4. Use high-quality data

**Q: How do I train my own model?**  
A: The project includes dedicated training scripts! See the [Model Training Guide](#-model-training-guide) section below for step-by-step instructions using `egitim_verisi_olusturma.py` and `training.py`.

**Q: Can I use the training scripts interactively?**  
A: There is no file-picker dialog. Either pass `--input`, `--mask`, and `--output` on the CLI, or set those keys (and defaults) in the `CONFIG` dict at the top of `egitim_verisi_olusturma.py` / `training.py` and run from your IDE.

**Q: What if I don't have ground truth masks?**  
A: You can still use the system with zero-shot ImageNet weights (`zero_shot_imagenet: true`) or classical methods only. However, for best results, train a custom model with your own labeled data.

### 📊 Data Questions

**Q: What is the minimum area resolution?**  
A: Recommended: 0.5-2 meters/pixel. At lower resolution, small structures may not be detected.

**Q: Is there a maximum file size?**  
A: No, thanks to tile system, very large files can be processed. Tested: 50 GB+

**Q: Are different CRS supported?**  
A: Yes, input CRS is preserved and transferred to output.

---

## 🎓 Model Training Guide

This guide walks you through training custom models with your own labeled data. Follow the steps below to go from raw data to a trained model.

---

### ⚡ Quick Start (TL;DR)

For experienced users, here's the minimal workflow:

```bash
# 1. Prepare your data (GeoTIFF + binary mask)
# 2. Generate training tiles
python egitim_verisi_olusturma.py --input data.tif --mask mask.tif --output workspace/training_data

# 3. Train the model
python training.py --data workspace/training_data --task tile_classification --epochs 50

# 4. Use your trained model
python archaeo_detect.py --input new_area.tif
```

---

### 📋 Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │  STEP 1      │      │  STEP 2      │      │  STEP 3      │              │
│   │  Prepare     │ ───► │  Generate    │ ───► │  Train       │              │
│   │  Masks       │      │  Tiles       │      │  Model       │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                     │                     │                        │
│         ▼                     ▼                     ▼                        │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │ GeoTIFF +    │      │ 5-channel    │      │ Trained      │              │
│   │ Binary Mask  │      │ NPZ tiles    │      │ .pth model   │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                      │                       │
│                                                      ▼                       │
│                                               ┌──────────────┐              │
│                                               │  STEP 4      │              │
│                                               │  Use Model   │              │
│                                               └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What you need:**
- GeoTIFF file with RGB + DSM + DTM bands
- Binary mask (GeoTIFF): archaeological areas = 1, background = 0
- Python environment with dependencies installed
- GPU recommended (CPU works but slower)

---

### 🛠️ Step 1: Prepare Ground Truth Masks

Create a binary mask where archaeological features are marked as **1** (white) and everything else as **0** (black).

#### Using QGIS (Free, open-source)

**What you'll do:** Draw polygons around archaeological features, then convert them to a raster image where archaeological areas = 1 and everything else = 0.

**Step 1: Open your orthophoto**
```
Menu: Layer → Add Layer → Add Raster Layer...
Navigate to your GeoTIFF file → Click "Add"
```
Your image should now appear on the map canvas. Use mouse wheel to zoom, hold middle button to pan.

**Step 2: Create a new polygon layer for digitizing**
```
Menu: Layer → Create Layer → New Shapefile Layer...
```
In the dialog:
- **File name:** Click "..." and choose where to save (e.g., `archaeological_mask.shp`)
- **Geometry type:** Select "Polygon"
- **CRS:** Click the globe icon → search for your raster's coordinate system (check raster properties if unsure)
- Click "OK"

A new empty layer appears in the Layers panel.

**Step 3: Start digitizing (drawing polygons)**
```
1. Select your new layer in the Layers panel (click on it)
2. Menu: Layer → Toggle Editing (or click the pencil icon)
3. Look for "Add Polygon Feature" button in the toolbar (polygon with + sign)
4. Click it, then start clicking on the map to draw vertices
5. Right-click to finish each polygon
```

**Tips for digitizing:**
- Zoom in close for accuracy (scroll wheel)
- Draw around tumuli, walls, ditches - anything archaeological
- If you make a mistake: Ctrl+Z to undo
- Each click adds a vertex; right-click closes the polygon
- Draw as many polygons as needed

**Step 4: Save your edits**
```
Menu: Layer → Toggle Editing → Click "Save" when prompted
Or: Click the floppy disk icon in the toolbar
```

**Step 5: Convert polygons to raster (the mask)**
```
Menu: Raster → Conversion → Rasterize (Vector to Raster)...
```
In the dialog:
- **Input layer:** Your polygon layer (`archaeological_mask`)
- **Field to use for a burn-in value:** Leave empty (we'll use fixed value)
- **A fixed value to burn:** Enter `1`
- **Output raster size units:** Georeferenced units
- **Width/Horizontal resolution:** Same as your input raster (e.g., `1.0` for 1m resolution)
- **Height/Vertical resolution:** Same value (e.g., `1.0`)
- **Output extent:** Click "..." → "Calculate from Layer" → Select your input raster
- **Rasterized:** Click "..." → Save to File → name it `ground_truth.tif`
- Click "Run"

**Step 6: Fill NoData with zeros**

The rasterize tool creates NoData where there are no polygons. We need those to be 0.
```
Menu: Raster → Raster Calculator...
```
Enter this expression (replace with your actual layer name):
```
("ground_truth@1" >= 1) * 1
```
Or use:
```
Menu: Processing → Toolbox → Search "Fill nodata"
Use "Fill NoData cells" tool with fill value = 0
```

**Verify your mask:**
- Values should be only 0 and 1
- Right-click layer → Properties → Symbology → check min/max values
- Dimensions should match your input raster exactly

---

#### Using ArcGIS Pro

**What you'll do:** Create a polygon feature class, digitize archaeological features, then convert to a raster mask.

**Step 1: Create a new project and add your data**
```
1. Open ArcGIS Pro → New Project → Map
2. Give it a name and location → OK
3. Map tab → Add Data → Browse to your GeoTIFF → Add
```
Your orthophoto should appear on the map. Use scroll wheel to zoom, hold wheel to pan.

**Step 2: Check your raster's properties (important for later)**
```
1. In Contents pane, right-click your raster → Properties
2. Go to "Source" tab → Note the:
   - Cell Size (e.g., 1.0 x 1.0)
   - Extent (Top, Left, Right, Bottom coordinates)
   - Spatial Reference (e.g., EPSG:32635)
```
Write these down - you'll need them to match your mask.

**Step 3: Create a new feature class for digitizing**
```
1. In Catalog pane, expand your project's geodatabase (.gdb)
2. Right-click the geodatabase → New → Feature Class
```
In the wizard:
- **Name:** `archaeological_features`
- **Alias:** Archaeological Features (optional)
- **Feature Class Type:** Polygon
- Click "Next"
- **Fields:** Skip (we'll add later) → Click "Next"
- **Spatial Reference:** Click the globe → Import → Select your raster
- Click "Finish"

The new empty layer appears in Contents.

**Step 4: Start digitizing**
```
1. In Contents, click on your new layer to select it
2. Edit tab → Create (opens Create Features pane)
3. Click on "archaeological_features" in the Create Features pane
4. Select "Polygon" tool
5. Click on the map to add vertices, double-click to finish
```

**Digitizing tips:**
- Press `Z` to zoom, `C` to pan while drawing
- Press `Ctrl+Z` to undo last vertex
- Double-click (or press `F2`) to finish each polygon
- Draw around all visible archaeological features
- Be as precise as possible - these become your training labels!

**Step 5: Save your edits**
```
Edit tab → Save → Save Edits
```

**Step 6: Add a field for the raster value**
```
1. In Contents, right-click your layer → Attribute Table
2. Click "Add Field" button (top of table)
3. Field Name: burn_value
4. Data Type: Short (Integer)
5. Click "Save" (in Fields tab)
```

**Step 7: Set all polygons to value 1**
```
1. In the attribute table, right-click the "burn_value" column header
2. Select "Calculate Field..."
3. In Expression box, simply type: 1
4. Click "OK"
```
All rows should now show `1` in the burn_value column.

**Step 8: Convert to raster**
```
Analysis tab → Tools → Search "Polygon to Raster"
```
In the tool dialog:
- **Input Features:** archaeological_features
- **Value field:** burn_value
- **Output Raster Dataset:** Browse → Save as `ground_truth.tif`
- **Cell assignment type:** CELL_CENTER
- **Priority field:** NONE
- **Cellsize:** Same as your input raster (e.g., `1`)

**Important - Set Environment:**
```
Click "Environments" tab at bottom of tool:
- Snap Raster: Select your input raster (ensures alignment!)
- Cell Size: Same as your input raster
- Extent: Same as your input raster
```
Click "Run"

**Step 9: Convert NoData to 0**

By default, areas outside polygons become NoData. We need them to be 0.
```
Analysis tab → Tools → Search "Reclassify"
```
Or use Raster Calculator:
```
Analysis tab → Tools → Search "Raster Calculator"
Expression: Con(IsNull("ground_truth.tif"), 0, "ground_truth.tif")
Output: ground_truth_final.tif
```

Alternative with Reclassify:
```
- Input raster: ground_truth.tif
- Reclass field: Value
- Reclassification:
  - Add row: Old = NoData, New = 0
  - Existing: Old = 1, New = 1
- Output: ground_truth_final.tif
```

**Step 10: Verify your mask**
```
1. Add the final mask to your map
2. Right-click → Properties → Source → Check:
   - Cell size matches input ✓
   - Extent matches input ✓
   - Values are only 0 and 1 ✓
```

**Common issues:**
- **Mask extent doesn't match:** Re-run Polygon to Raster with correct Environment settings
- **Mask has wrong cell size:** Set cell size explicitly in tool and Environment
- **Mask is all NoData:** Check that burn_value field has value 1

---

#### Using Python

```python
import rasterio
import numpy as np

# Create mask array (same dimensions as input)
mask = np.zeros((height, width), dtype=np.uint8)

# Mark archaeological areas (example: from coordinates or polygons)
mask[100:200, 150:250] = 1  # Replace with actual areas

# Save as GeoTIFF (must match input CRS and transform!)
with rasterio.open('mask.tif', 'w', driver='GTiff',
                   height=height, width=width, count=1, 
                   dtype='uint8', crs=input_crs, 
                   transform=input_transform) as dst:
    dst.write(mask, 1)
```

> **Important:** Mask dimensions, CRS, and resolution must exactly match your input GeoTIFF!

---

### 📦 Step 2: Generate Training Tiles

The script `egitim_verisi_olusturma.py` converts your GeoTIFF + mask into **5-channel** training tiles (R, G, B, SVF, SLRM).

#### Basic Command

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data
```

#### IDE / CLI Mode

The script is non-interactive. Use explicit arguments:

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data
```

#### What Happens Inside

```
Input GeoTIFF (5 bands)          Ground Truth Mask
       |                                |
       v                                |
+------------------+                    |
| Read RGB + DSM   |                    |
| + DTM bands      |                    |
+--------+---------+                    |
         |                              |
         v                              |
+------------------+                    |
| RVT on DTM       |                    |
| SVF + SLRM       |                    |
+--------+---------+                    |
         |                              |
         v                              |
+------------------+                    |
| stack_channels   |<-------------------+
| R,G,B,SVF,SLRM   |
| 256x256 tiles    |
+--------+---------+
         |
         v
   workspace/training_data/
   |-- train/images/*.npz  (5, 256, 256)
   |-- train/masks/*.npz   (256, 256)
   |-- val/images/*.npz
   |-- val/masks/*.npz
   `-- metadata.json
```

#### Key Parameters

Run `python egitim_verisi_olusturma.py --help` for the full list. Common options:

| Parameter | Default (see script `CONFIG`) | Description |
|-----------|------------------------------|-------------|
| `--input` / `-i` | _(required via CLI or `CONFIG`)_ | Multi-band GeoTIFF (RGB + DSM + DTM) |
| `--mask` / `-m` | _(required via CLI or `CONFIG`)_ | Ground-truth mask (values ≠ 0 treated as positive, then binarized) |
| `--output` / `-o` | `workspace/training_data` | Output root (`train/`, `val/`, `metadata.json`, …) |
| `--tile-size` / `-t` | `256` | Tile size in pixels |
| `--overlap` | `128` | Sliding-window overlap in pixels (must stay consistent with training/inference metadata) |
| `--bands` / `-b` | `1,2,3,4,5` | 1-based GeoTIFF band indices: R, G, B, DSM, DTM |
| `--min-positive` | `0.0` | Minimum fraction of positive pixels in a tile to keep it (0 = allow all-negative tiles subject to negative sampling) |
| `--tile-label-min-positive-ratio` | _(from `CONFIG`)_ | For tile-level labels: minimum positive ratio for the tile's class label (0 = any positive pixel) |
| `--max-nodata` | `0.3` | Maximum allowed NoData fraction per tile |
| `--train-ratio` | `0.8` | Train fraction |
| `--train-negative-keep-ratio` | `1.0` | Fraction of all-negative **train** tiles to retain (`0` = drop all, `1` = keep all) |
| `--train-negative-max` | `None` | Optional cap on kept negative train tiles |
| `--split-mode` | `spatial` | `spatial` (recommended) or `random` train/val split |
| `--no-normalize` | off | Skip `robust_norm` on the stacked channels |
| `--format` | `npz` | `npz` (compressed) or `npy` |
| `--num-workers` | CPU-based default | Worker processes for tile generation |
| `--tile-prefix` | `""` | Optional prefix for tile filenames (empty → auto prefix with tile/overlap/bands/timestamp) |
| `--append` / `--no-append` | clean rebuild | Append tiles without deleting existing outputs vs full regenerate |

#### Recommended Settings by Scenario

| Scenario | Command |
|----------|---------|
| **Standard** | `--tile-size 256 --overlap 64` |
| **Large structures** | `--tile-size 512 --overlap 64` |
| **Imbalanced data** (<5% archaeological) | `--train-negative-keep-ratio 0.2 --min-positive 0.01` |
| **Quick test** | `--tile-size 256 --train-ratio 0.9` |

#### Output: 5 channels (model tensor)

| # | Channel | Role |
|---|---------|------|
| 0–2 | R, G, B | Orthophoto color/texture |
| 3 | SVF | Sky-View Factor (relief visibility; mounds, local dominance) |
| 4 | SLRM | Simple Local Relief Model (local height anomalies on DTM) |

DSM/DTM **bands** are still required in the GeoTIFF for correct masking and for computing SVF/SLRM; only RGB + the two relief channels enter the saved `image` tensor.

---

### 🚀 Step 3: Train the Model

Use `training.py` to train a U-Net (SMP) model on your **5-channel** tiles.

#### Basic Training

```bash
python training.py --data workspace/training_data
```

This uses the checked-in `CONFIG` in `training.py` (currently: **U-Net**, **ResNet50**, **BCE** loss, **patience 20**, **CBAM off** via `no_attention: true`, **AMP on** unless you disable it). Successful runs **publish** the best weights to `workspace/checkpoints/active/` when `publish_active` is true.

#### Full Command with Options

```bash
python training.py \
  --data workspace/training_data \
  --arch Unet \
  --encoder resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --loss bce \
  --patience 20
```

#### Key Parameters

| Parameter | Default | Options / Notes |
|-----------|---------|-----------------|
| `--data` | `workspace/training_data` | Path to Step 2 output (paired or Positive/Negative layout) |
| `--task` | `tile_classification` | `segmentation` or `tile_classification` |
| `--arch` | `Unet` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN` |
| `--encoder` | `resnet50` | `resnet34`, `efficientnet-b3`, `densenet121` |
| `--epochs` | `50` | More = potentially better (with early stopping) |
| `--batch-size` | `16` | Increase if GPU memory allows |
| `--lr` | `1e-4` | Reduce if loss oscillates |
| `--loss` | `bce` | `bce`, `dice`, `combined`, `focal`; `tile_classification` only supports `bce`/`focal` |
| `--balance-mode` | `auto` | `auto` (computes pos_weight from train ratio), `manual`, `none` |
| `--pos-weight` | `1.0` | Manual BCE positive class weight (used when `--balance-mode manual`) |
| `--max-auto-pos-weight` | `100.0` | Clamp for auto-computed pos_weight to avoid destabilizing training |
| `--patience` | `20` | Early stopping after N epochs without improvement |
| `--metric-threshold` | `0.5` | Probability threshold used to compute IoU/F1/Precision/Recall metrics |
| `--val-threshold-sweep` | on | Sweep thresholds 0.1–0.9 on val and report best IoU + threshold |
| `--no-attention` | on | Matches `CONFIG["no_attention"]` (default **true** → CBAM **off**; set `no_attention: false` in `CONFIG` to enable attention) |
| `--no-amp` | Off | Disable mixed precision (FP16) |
| `--train-neg-to-pos-ratio` | `2` | Sub-sample negatives to this multiple of positives in train (`None` = keep all) |
| `--train-neg-sample-seed` | `42` | RNG seed for negative sub-sampling |
| `--val-keep-ratio` | `1.0` | Fraction of val tiles to keep relative to selected train count |
| `--val-sample-seed` | `42` | RNG seed for val sub-sampling |
| `--tile-label-min-positive-ratio` | `0.02` | Minimum positive-pixel ratio to label a tile as Positive (paired layout) |
| `--monitor-channel-importance` | on | Track and log per-channel gradient importance during training |
| `--channel-importance-max-batches` | `12` | Max batches for gradient-based channel importance (0 = all) |
| `--deterministic-rotate-step-deg` | `30.0` | Augmentation rotation step (0 = disabled; 30 → 12 views per sample) |
| `--allow-all-negative` | off | Continue training even if all labels are negative (guards against bad data) |
| `--save-every-epoch` | on | Save a per-epoch checkpoint under `workspace/checkpoints/epochs/` |

#### Choosing the Right Settings

**Model Architecture:**

| Architecture | Speed | Accuracy | When to Use |
|--------------|-------|----------|-------------|
| `Unet` | Fast | Good | **Start here** - reliable baseline |
| `UnetPlusPlus` | Medium | Excellent | Need higher accuracy |
| `DeepLabV3Plus` | Medium | Excellent | Multi-scale structures |

**Encoder:**

| Encoder | Speed | Accuracy | Memory |
|---------|-------|----------|--------|
| `resnet34` | Fast | Good | Low | **Recommended start** |
| `resnet50` | Medium | Better | Medium | Better accuracy |
| `efficientnet-b3` | Fast | Excellent | Low | Best efficiency |

**Loss Function:**

| Loss | When to Use |
|------|-------------|
| `bce` | **Default in `training.py` `CONFIG`**; required path for `tile_classification` together with `focal` |
| `focal` | Strong class imbalance in tile labels |
| `combined` / `dice` | Primarily for **segmentation** task (pixel masks) |

#### Training Output

```
workspace/checkpoints/
├── active/model.pth                         ← copy of best weights for inference (`publish_active` in training.py)
├── active/training_metadata.json           ← tile / overlap / bands (+ schema) for trained_model_only mode
├── active/published_from.json              ← manifest pointing at the source checkpoint used for the copy
├── epochs/                                  ← per-epoch checkpoints when `save_every_epoch` is true (default in CONFIG)
└── training_history.json                    ← training metrics
```

You may set `weights` in `config.yaml` to `workspace/checkpoints/active/model.pth` **or** to a specific run under `workspace/checkpoints/epochs/`; in both cases **`training_metadata.json` must describe the same architecture, channel count, tile size, overlap, and bands** as that run.

`channel_importance_history.json` may also appear under `workspace/checkpoints/` and stores per-epoch band importance rankings when enabled.

#### Monitoring Training

Watch the console output:

```
Epoch  1/50 | Train Loss: 0.45 | Val Loss: 0.39 | Val IoU: 0.62 | LR: 1e-04
  → Best model saved
Epoch  2/50 | Train Loss: 0.38 | Val Loss: 0.34 | Val IoU: 0.68 | LR: 1e-04
  → Best model saved
...
Early stopping: Best model at epoch 15 (Val IoU: 0.79)
```

**What the metrics mean:**
- **Val IoU** (Intersection over Union): Higher = better. Target: 0.6-0.8
- **Val Loss**: Lower = better. Should decrease over time
- **Train Loss**: Should be slightly lower than Val Loss (not much lower = overfitting)

---

### 📊 Step 4: Use Your Trained Model

#### Command Line

```bash
python archaeo_detect.py \
  --weights workspace/checkpoints/active/model.pth \
  --training-metadata workspace/checkpoints/active/training_metadata.json \
  --input new_area.tif \
  --th 0.6
```

#### Via config.yaml

For the IDE-first trained profile:

- keep `trained_model_only: true`
- keep `weights: "workspace/checkpoints/active/model.pth"`
- keep `training_metadata: "workspace/checkpoints/active/training_metadata.json"`
- treat `tile`, `overlap`, and `bands` as metadata-locked values during inference

```yaml
weights: "workspace/checkpoints/active/model.pth"
training_metadata: "workspace/checkpoints/active/training_metadata.json"
zero_shot_imagenet: false
trained_model_only: true
```

Then simply run:
```bash
python archaeo_detect.py
```

---

### 🔧 Troubleshooting

#### Data Preparation Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| "Mask dimensions don't match" | Different resolution/extent | Resample mask: `gdalwarp -tr 1.0 1.0 -r nearest mask.tif mask_fixed.tif` |
| "No valid tiles found" | `--min-positive` too high | Lower to `0.0` or `0.01` |
| "Memory error" | Large input file | Reduce `--tile-size` to 128 |

#### Training Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss not decreasing | Learning rate too high | Use `--lr 5e-5` or `1e-5` |
| GPU out of memory | Batch size too large | Use `--batch-size 4` or `--no-amp` |
| Overfitting (train << val loss) | Too little data | Add more training tiles or use `--loss focal` |
| All predictions = 0 | Class imbalance | Use `--loss focal` and reduce train negatives (e.g., `--train-negative-keep-ratio 0.2`) in data prep |
| Training too slow | No GPU / small batches | Use GPU, increase `--batch-size`, enable AMP |

#### Quick Diagnostic Commands

```bash
# Check training data structure
ls -R workspace/training_data/

# View metadata
cat workspace/training_data/metadata.json | python -m json.tool

# Test data loading
python -c "import numpy as np; d=np.load('workspace/training_data/train/images/tile_00000_00000.npz'); print(d['image'].shape)"
# Expected: (5, 256, 256)
```

---

### 💡 Best Practices

#### Data Quality Checklist

- [ ] Masks are accurate (precise boundaries)
- [ ] All archaeological features labeled consistently
- [ ] Balanced dataset (30-50% positive tiles)
- [ ] Diverse terrain types in negatives
- [ ] Minimum 1000 tiles (2000-5000 recommended)

#### Training Workflow

```
1. Quick test (5 epochs)     → Verify everything works
2. Baseline (50 epochs)      → Establish benchmark
3. Optimize (try better encoder/architecture)
4. Fine-tune (lower LR if needed)
```

#### Performance Expectations

| Dataset Size | Expected Val IoU | Training Time (GPU) |
|--------------|------------------|---------------------|
| 500-1000 tiles | 0.55-0.65 | 30-60 min |
| 1000-3000 tiles | 0.65-0.75 | 1-2 hours |
| 3000-5000 tiles | 0.70-0.80 | 2-4 hours |
| 5000+ tiles | 0.75-0.85 | 4+ hours |

---

### 📚 Complete Example: End-to-End

```bash
# 1. Generate training data (single balancing mechanism: train-negative filtering)
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data \
  --tile-size 256 \
  --train-negative-keep-ratio 0.3

# 2. Train model
python training.py \
  --data workspace/training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16 \
  --loss bce

# 3. Run inference on new area
python archaeo_detect.py \
  --weights workspace/checkpoints/active/model.pth \
  --training-metadata workspace/checkpoints/active/training_metadata.json \
  --input new_area.tif \
  --th 0.6 \
  --enable-fusion
```

**Expected results:**
- ~1000-2000 training tiles
- Val IoU: 0.65-0.75
- Training time: 1-2 hours (GPU)
- Model file: ~70 MB

---

## 🔬 Advanced Features

### Custom Model Training

> **📖 For detailed training guide, see [Model Training Guide](#-model-training-guide) section above.**

The project includes two dedicated scripts for training custom models:

- **`egitim_verisi_olusturma.py`**: Creates **5-channel** training tiles (R, G, B, SVF, SLRM) from GeoTIFF + ground truth masks
- **`training.py`**: Trains SMP U-Net (and related heads) on **5-channel** tiles; **CBAM** is optional (`no_attention` in `CONFIG`)

**Quick Start:**

```bash
# 1. Create training data
python egitim_verisi_olusturma.py --input area.tif --mask mask.tif --output workspace/training_data

# 2. Train model
python training.py --data workspace/training_data --task tile_classification --epochs 50

# 3. Use trained model
python archaeo_detect.py
```

**Key Features:**
- ✅ 5-channel input (R, G, B, SVF, SLRM) aligned with inference
- ✅ Optional CBAM attention on the SMP model (`training.py`)
- ✅ Losses: **BCE / Focal** for `tile_classification`; **BCE / Dice / Combined / Focal** for `segmentation`
- ✅ Mixed precision training
- ✅ Early stopping and checkpointing

For complete documentation, examples, and troubleshooting, see the [Model Training Guide](#-model-training-guide) section.

### Encoder selection

Encoders are **Segmentation Models PyTorch** backbone names (e.g. `resnet34`, `resnet50`, `efficientnet-b3`). Set `encoder` / `encoders` in `config.yaml` or pass the corresponding CLI flags. Use only encoders your installed `segmentation-models-pytorch` build supports and that match your checkpoint when loading weights.

### API Usage

Calling the script from Python code:

```python
import subprocess

result = subprocess.run([
    'python', 'archaeo_detect.py',
    '--input', 'my_area.tif',
    '--th', '0.7',
    '--enable-fusion'
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
```

### Batch Processing

Script for processing multiple files:

```python
import os
from pathlib import Path
import subprocess

input_dir = Path('input_files')
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

for tif_file in input_dir.glob('*.tif'):
    print(f"Processing: {tif_file.name}")
    
    subprocess.run([
        'python', 'archaeo_detect.py',
        '--input', str(tif_file),
        '--out-prefix', str(output_dir / tif_file.stem),
        '--enable-fusion',
        '--cache-derivatives',
        '-v'
    ])
    
print("All files processed!")
```

### Performance Profiling

Analyzing processing times:

```bash
python -m cProfile -o profile.stats archaeo_detect.py

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## 📚 Technical Details

### Project Structure

```text
arkeolojik_alan_tespit/
├── archaeo_detect.py                  # Main detection script
├── archeo_shared/                     # Shared channel schema and model helpers
├── egitim_verisi_olusturma.py         # Paired tile data generation
├── prepare_tile_classification_dataset.py
├── training.py                        # Model training script
├── evaluation.py                      # Evaluation metrics
├── veri_birlestir_rgb_dsm_dtm.py      # RGB + DSM + DTM merge tool
├── dtm_uret.py                        # DSM/LAS -> DTM conversion
├── ground_truth_kare_etiketleme_qt.py # Qt-based annotation tool
├── config.yaml                        # Shared repo config
├── configs/                           # Example YAML profiles
├── docs/                              # Extra documentation (YOLO, notes)
├── tests/                             # Test suite
├── tools/                             # Maintenance and inspection helpers
└── workspace/                         # Repo-local data and artifact area
    ├── on_veri/
    ├── training_data/
    ├── training_data_classification/
    ├── checkpoints/
    ├── cache/
    ├── ciktilar/
    └── assets/
```
### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework |
| SMP | 0.3.2+ | Segmentation models |
| Rasterio | 1.3+ | Raster data I/O |
| GeoPandas | 0.12+ | Vector data processing |
| OpenCV | 4.7+ | Image processing |
| scikit-image | 0.20+ | Advanced image processing |
| RVT-py | 1.2+ (Python < 3.11) or RVT 2.0+ (Python >= 3.11) | Relief visualization |
| NumPy | 1.24+ | Numerical operations |
| SciPy | 1.10+ | Scientific computing |

### Channel Architecture

**5-channel DL tensor** (canonical names in `archeo_shared/channels.py` → `MODEL_CHANNEL_NAMES`):

| Index | Name | Source |
|------:|------|--------|
| 0–2 | R, G, B | GeoTIFF bands selected as RGB |
| 3 | SVF | RVT Sky-View Factor on filled DTM |
| 4 | SLRM | RVT Simple Local Relief Model on DTM (with Gaussian fallback if needed) |

Training tiles and inference use the same ordering via `stack_channels(rgb, svf, slrm)` in `archaeo_detect.py`. GeoTIFF **DSM/DTM bands** are still read for masking and for building these relief channels.

**CBAM Attention (when enabled in training):**
- **Channel Attention**: Weights the five input-derived feature maps (e.g., emphasizing SVF vs. SLRM where useful).
- **Spatial Attention**: Highlights informative regions in the feature maps.
- **Note:** The checked-in `training.py` `CONFIG` keeps attention off (`no_attention: true`) unless you change it.

### Algorithm Details

#### RVT (Relief Visualization Toolbox)

**Sky-View Factor (SVF):**
```
SVF = (1/n) * Σ(max(0, cos(α_i)))
```
Where `α_i` is the horizon angle in each direction.

**Openness:**
```
Openness_positive = (1/n) * Σ(90° - α_i)
Openness_negative = (1/n) * Σ(α_i - 90°)
```

#### Hessian Matrix

Second derivative matrix:
```
H = [∂²f/∂x²    ∂²f/∂x∂y]
    [∂²f/∂y∂x   ∂²f/∂y²]
```

Ridge/valley detection via eigenvalue analysis.

#### Elevation Context (DSM / DTM / nDSM)

- **DSM** preserves surface-level height cues (vegetation, structures, terrain).
- **DTM** provides ground-only relief context.
- **nDSM = DSM - DTM** emphasizes above-ground height anomalies and supports tall-object masking.

#### TPI (Topographic Position Index) — optional, not in the 5-channel DL tensor

`archaeo_detect.py` still contains `compute_tpi_multiscale()` and `config.yaml` keys such as `enable_tpi` / `tpi_radii` for experiments, but the **current** `MODEL_CHANNEL_NAMES` stack does **not** include TPI. Relief cues for the neural network are **SVF + SLRM** only.

#### Fusion Algorithm

```python
def fusion(p_dl, p_classic, alpha):
    """
    p_dl: Deep learning probability (0-1)
    p_classic: Classical method probability (0-1)
    alpha: Weight factor (0-1)
    """
    p_fused = alpha * p_dl + (1 - alpha) * p_classic
    return np.clip(p_fused, 0, 1)
```

---

## 🤝 Contributing

To contribute to the project:

1. **Fork** the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'New feature: ...'`)
4. Push your branch (`git push origin feature/new-feature`)
5. Open a **Pull Request**

### Contribution Areas

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🌍 Translations (i18n)
- 🧪 Test scenarios
- 🎨 Visualization tools

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 [Ahmet Ertuğrul Arık]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📧 Contact and Support

- **Issues**: [GitHub Issues](https://github.com/elestirmen/archaeological-site-detection/issues)
- **Email**: ertugrularik@hotmail.com
- **Documentation**: [Wiki](https://github.com/elestirmen/archaeological-site-detection/wiki)

---

## 🙏 Acknowledgments

This project benefits from the following open-source projects:

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [RVT-py](https://github.com/EarthObservation/RVT_py)
- [PyTorch](https://pytorch.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [GeoPandas](https://geopandas.org/)

---

## 📖 Citation

If you use this project in your academic work, please cite:

```bibtex
@software{archaeological_site_detection,
  title = {Archaeological Site Detection: Deep Learning and Classical Image Processing},
  author = {Ahmet Ertuğrul Arık},
  year = {2025},
  url = {https://github.com/elestirmen/archaeological-site-detection}
}
```

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/elestirmen/archaeological-site-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/elestirmen/archaeological-site-detection?style=social)

---

<div align="center">

Developer: Ahmet Ertuğrul Arık  
Last update: April 2026

</div>



