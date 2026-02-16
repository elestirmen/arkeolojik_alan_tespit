# 🏛️ Archaeological Site Detection (Deep Learning + Classical Image Processing)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Advanced AI system for automatic detection of archaeological structures from LiDAR and multi-band satellite imagery**

This project combines **deep learning** and **classical image processing** methods to detect archaeological traces (tumuli, ditches, mounds, wall remains, etc.) from multi-band GeoTIFF data (RGB, DSM, DTM).

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🎯 What It Does](#-what-it-does)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [DSM to DTM Preprocessing (`on_isleme.py`)](#dsm-to-dtm-preprocessing-on_islemepy)
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
  - ⚠️ **Note:** Fine-tuning required for nadir (bird's-eye) imagery (see YOLO11_NADIR_TRAINING.md)
- **Classical Image Processing**: RVT (Relief Visualization Toolbox), Hessian matrix, Morphological operators
- **Hybrid Fusion**: Smart fusion combining strengths of each method

### 🎯 Smart Detection Features
- ✅ **Multi-Encoder Support**: ResNet, EfficientNet, VGG, DenseNet, MobileNet and more
- ✅ **Zero-Shot Learning**: Works even without trained models using ImageNet weights
- ✅ **Ensemble Learning**: Combines results from multiple encoders for more reliable detection
- ✅ **Multi-Scale Analysis**: Detects structures of different sizes
- ✅ **🆕 Labeled Object Detection**: Automatic labeling of 80 different object classes (trees, buildings, vehicles, etc.) with YOLO11
- ✅ **🆕 12-Channel Input**: Advanced topographic features including Curvature and TPI for enhanced detection
- ✅ **🆕 CBAM Attention**: Channel and spatial attention mechanism for dynamic feature weighting

### 🔧 Technical Features
- 🚀 **Tile-Based Processing**: Memory-efficient processing for large images
- 🎨 **Seamless Mosaicking**: No artifacts at tile boundaries with cosine feathering
- 📊 **Robust Normalization**: Global or local percentile-based normalization
- ⚡ **Cache System**: 10-100x speedup by caching RVT calculations
- 🎯 **Smart Masking**: Automatic filtering of tall structures (trees, buildings)
- 📐 **Vectorization**: Converts results to GIS-compatible polygons

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

### Run in 5 Minutes!

```bash
# 1. Clone the repository
git clone https://github.com/your-username/archaeological-site-detection.git
cd archaeological-site-detection

# 2. Install required packages
pip install -r requirements.txt

# 3. Prepare your data (a GeoTIFF named kesif_alani.tif)
# Should contain RGB, DSM, DTM bands in a single file

# 4. Run it!
python archaeo_detect.py
```

🎉 **Congratulations!** The system has started. Results will be created in the current directory.

### 🎓 Training Your Own Model (Optional)

If you have labeled data (ground truth masks), you can train a custom model:

```bash
# Step 1: Create training data from GeoTIFF + ground truth mask
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data

# Or use interactive mode (no arguments needed):
python egitim_verisi_olusturma.py
# Follow the prompts to enter file paths

# Step 2: Train the model
python training.py --data training_data --epochs 50

# Step 3: Use your trained model
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth
```

**💡 Tip:** The training data generation script (`egitim_verisi_olusturma.py`) supports interactive mode. If you run it without arguments, it will guide you through the process step-by-step.

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

**Note:** The project includes a `.venv310` directory. If you move the project folder, make sure to update the virtual environment paths in `.venv310/Scripts/activate.bat` and `.venv310/Scripts/activate`.

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

## DSM to DTM Preprocessing (`on_isleme.py`)

`on_isleme.py` converts a DSM GeoTIFF into a DTM GeoTIFF.

Current processing flow in code:
- `[1/4]` Read input raster metadata.
- `[2/4]` Run PDAL SMRF pipeline.
- `[3/4]` Snap output DTM to the exact source grid (resolution/transform/extent).
- `[4/4]` If SMRF fails and fallback is enabled, run morphological fallback DTM.

### Quick Run

```bash
python on_isleme.py \
  --input veri/karlik_dag_dsm.tif \
  --output veri/karlik_dag_dtm_smrf.tif \
  --progress
```

### Key CLI Parameters

- `--input`, `--output`
- `--cell`, `--slope`, `--threshold`, `--window`, `--scalar` (SMRF parameters)
- `--allow-fallback` / `--no-fallback`
- `--opening-meters`, `--smooth-sigma-px`, `--tile-size` (fallback tuning)
- `--nodata`, `--compression`, `--log-level`
- `--progress` / `--no-progress`

### Dependencies and Environment Notes

- SMRF requires Python PDAL module in the same environment:

```bash
conda install -n <env_name> -c conda-forge pdal python-pdal
```

- Fallback method requires `scipy`.
- On Windows, keep geospatial stack consistent in one environment (avoid mixing `pip` `gdal/rasterio` with conda GDAL libraries), otherwise GDAL plugin DLL errors may occur.
- Runtime defaults are defined in `on_isleme.py` (`CONFIG` dict) and can be overridden via CLI.

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
| `--overlap` | Overlap amount | `--overlap 256` |
| `--encoder` | Single encoder selection | `--encoder resnet34` |
| `--encoders` | Multi-encoder mode | `--encoders all` |
| `--alpha` | Fusion weight | `--alpha 0.6` |
| `--enable-fusion` | Enable fusion | (flag) |
| `--cache-derivatives` | Use cache | (flag) |
| `-v` or `--verbose` | Detailed log | (flag) |

---

## ⚙️ Configuration

### config.yaml File

System behavior is controlled by the `config.yaml` file. This file is **richly documented** with detailed explanations.

#### Main Sections:

1. **Input/Output**: File paths and band selection
2. **Method Selection**: Which methods to use
3. **Deep Learning**: Model architecture and encoder settings
4. **Classical Methods**: RVT, Hessian, Morphology parameters
5. **Fusion**: Hybrid combination settings
6. **Tile Processing**: Memory and performance optimization
7. **Normalization**: Data preprocessing
8. **Masking**: Filtering tall structures
9. **Vectorization**: GIS output format
10. **Performance**: Speed and memory optimization
11. **Cache**: Acceleration system

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

**GeoPackage Features:**
- Polygon geometry
- Area information (in m²)
- CRS information preserved
- Can be opened directly in QGIS/ArcGIS

### 💾 Cache Files

**Cache Directory Structure:**
```
cache/
├── kesif_alani.derivatives.npz    → RVT derivatives cache
└── karlik_vadi.derivatives.npz   → RVT derivatives cache
```

**Cache System:**
- RVT calculations are cached in `.npz` format
- Cache files are stored in the `cache/` directory (configurable via `cache_dir` in config.yaml)
- Cache validation checks file name and modification time
- **Important:** Cache files are reusable even if the project folder is moved (file name-based validation)
- Provides 10-100x speedup on subsequent runs
- Cache files are typically 10-50 MB, but can be larger for high-resolution data

**Cache Configuration:**
```yaml
cache_derivatives: true      # Enable caching
cache_dir: "cache/"          # Cache directory (relative to project root)
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

1. **RVT Derivatives Calculation**
   - Sky-View Factor (SVF)
   - Openness (Positive & Negative)
   - Local Relief Model (LRM)
   - Slope

2. **12-Channel Tensor Creation** (Updated!)
   - 3 x RGB
   - 1 x nDSM (DSM - DTM)
   - 5 x RVT derivatives (SVF, Pos/Neg Openness, LRM, Slope)
   - 2 x Curvature (Plan + Profile) - NEW!
   - 1 x TPI (Topographic Position Index) - NEW!

3. **Normalization**
   - Global or local percentile-based
   - Scaling to 2%-98% range

4. **Tile-Based Processing**
   - Large image divided into small tiles
   - Each tile fed to U-Net
   - Probability map generated

5. **Feathering (Smoothing)**
   - Transitions between tiles smoothed
   - Seamless mosaic created

6. **Thresholding**
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
- Cache files stored in `cache/` directory
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
   cache_dir: "cache/"  # Should match your cache directory
   ```

2. Verify cache file naming:
   - NPZ cache (default for small/medium rasters): `<input_name>.derivatives.npz`
   - Raster cache (block-based; used automatically for very large rasters or with `cache_derivatives_mode: "raster"`):
     - `<input_name>.derivatives_raster.tif`
     - `<input_name>.derivatives_raster.json`
   - Example for input `kesif_alani.tif`:
     - `kesif_alani.derivatives.npz`
     - `kesif_alani.derivatives_raster.tif`

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
ValueError: Expected 12 channels but got 9
```

**Solutions:**
1. **Regenerate training data**: Use `egitim_verisi_olusturma.py` with correct parameters
2. **Check metadata.json**: Verify `num_channels` matches actual data
3. **Verify file format**: Ensure `.npz` files contain `image` key with shape `(12, H, W)`

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
ls -R training_data/
# Should show: train/images/, train/masks/, val/images/, val/masks/

# Check metadata
cat training_data/metadata.json | python -m json.tool
```

**Test data loading:**
```python
# Quick test script
import numpy as np
from pathlib import Path

data_dir = Path("training_data")
train_images = list((data_dir / "train" / "images").glob("*.npz"))
if train_images:
    sample = np.load(train_images[0])
    print(f"Keys: {sample.files}")
    if 'image' in sample.files:
        img = sample['image']
        print(f"Image shape: {img.shape}")
        print(f"Expected: (12, 256, 256), Got: {img.shape}")
```

**Monitor training in real-time:**
```bash
# Watch training history file
watch -n 5 'tail -20 checkpoints/training_history.json'
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

**Q: Does it work with satellite imagery?**  
A: Yes, satellite imagery and LiDAR data are supported. Important thing is that it's in multi-band GeoTIFF format.

### 🔧 Technical Questions

**Q: How many bands are required?**  
A: Minimum 3 bands (RGB). Optimum 5 bands (RGB + DSM + DTM). **12 channels** are automatically created with RVT derivatives, Curvature, and TPI calculations.

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
A: Yes! `egitim_verisi_olusturma.py` supports interactive mode. Just run it without arguments: `python egitim_verisi_olusturma.py` and it will prompt you for inputs.

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
python egitim_verisi_olusturma.py --input data.tif --mask mask.tif --output training_data

# 3. Train the model
python training.py --data training_data --epochs 50

# 4. Use your trained model
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth --input new_area.tif
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
│   │ GeoTIFF +    │      │ 12-channel   │      │ Trained      │              │
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

The script `egitim_verisi_olusturma.py` converts your GeoTIFF + mask into 12-channel training tiles.

#### Basic Command

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data
```

#### Interactive Mode

Run without arguments for guided input:

```bash
python egitim_verisi_olusturma.py
# Prompts: Input file → Mask file → Output dir → Tile size
```

#### What Happens Inside

```
Input GeoTIFF (5 bands)          Ground Truth Mask
       │                                │
       ▼                                │
┌──────────────────┐                    │
│ Read RGB + DSM   │                    │
│ + DTM bands      │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ Calculate RVT    │                    │
│ derivatives:     │                    │
│ - SVF            │                    │
│ - Openness (+/-) │                    │
│ - LRM, Slope     │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ Calculate:       │                    │
│ - Curvatures     │                    │
│ - TPI            │                    │
│ - nDSM           │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ Stack 12 channels│◄───────────────────┘
│ + slice into     │
│ 256x256 tiles    │
└────────┬─────────┘
         │
         ▼
   training_data/
   ├── train/images/*.npz  (12, 256, 256)
   ├── train/masks/*.npz   (256, 256)
   ├── val/images/*.npz
   ├── val/masks/*.npz
   └── metadata.json
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Multi-band GeoTIFF (RGB+DSM+DTM) |
| `--mask` | Required | Binary mask GeoTIFF (0/1 values) |
| `--output` | `training_data` | Output directory |
| `--tile-size` | `256` | Tile dimensions in pixels |
| `--overlap` | `64` | Overlap between tiles |
| `--train-ratio` | `0.8` | 80% train, 20% validation |
| `--balance-ratio` | `None` | Balance positive/negative tiles (e.g., `0.4` = 40% positive) |
| `--min-positive` | `0.0` | Min positive pixel ratio to include tile |
| `--bands` | `1,2,3,4,5` | Band order: R,G,B,DSM,DTM |

#### Recommended Settings by Scenario

| Scenario | Command |
|----------|---------|
| **Standard** | `--tile-size 256 --overlap 64 --balance-ratio 0.4` |
| **Large structures** | `--tile-size 512 --overlap 128` |
| **Imbalanced data** (<5% archaeological) | `--balance-ratio 0.4 --min-positive 0.01` |
| **Quick test** | `--tile-size 256 --train-ratio 0.9` |

#### Output: 12 Channels Explained

| # | Channel | What it detects |
|---|---------|-----------------|
| 0-2 | RGB | Color/texture anomalies |
| 3 | SVF | Tumuli, mounds (horizon visibility) |
| 4 | Positive Openness | Raised structures |
| 5 | Negative Openness | Ditches, depressions |
| 6 | LRM | Local topographic anomalies |
| 7 | Slope | Terraces, walls |
| 8 | nDSM | Surface height above ground |
| 9 | Plan Curvature | Ridges vs valleys |
| 10 | Profile Curvature | Terraces, steps |
| 11 | TPI | Relative elevation (mounds/depressions) |

---

### 🚀 Step 3: Train the Model

Use `training.py` to train a U-Net model with CBAM attention on your 12-channel data.

#### Basic Training

```bash
python training.py --data training_data
```

This uses sensible defaults: U-Net + ResNet34 + 50 epochs + CBAM attention + mixed precision.

#### Full Command with Options

```bash
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --loss combined \
  --patience 10
```

#### Key Parameters

| Parameter | Default | Options / Notes |
|-----------|---------|-----------------|
| `--data` | `training_data` | Path to Step 2 output |
| `--arch` | `Unet` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN` |
| `--encoder` | `resnet34` | `resnet50`, `efficientnet-b3`, `densenet121` |
| `--epochs` | `50` | More = potentially better (with early stopping) |
| `--batch-size` | `8` | Increase if GPU memory allows |
| `--lr` | `1e-4` | Reduce if loss oscillates |
| `--loss` | `combined` | `bce`, `dice`, `combined`, `focal` |
| `--patience` | `10` | Early stopping after N epochs without improvement |
| `--no-attention` | Off | Disable CBAM attention |
| `--no-amp` | Off | Disable mixed precision (FP16) |

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
| `combined` | **Default** - works for most cases |
| `focal` | Imbalanced data (few archaeological pixels) |
| `dice` | Small objects, overlap-focused |

#### Training Output

```
checkpoints/
├── best_Unet_resnet34_12ch_attention.pth   ← Use this for inference
└── training_history.json                    ← Training metrics
```

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
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input new_area.tif \
  --th 0.6
```

#### Via config.yaml

```yaml
weights: "checkpoints/best_Unet_resnet34_12ch_attention.pth"
zero_shot_imagenet: false
encoder: "resnet34"
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
| All predictions = 0 | Class imbalance | Use `--loss focal` and `--balance-ratio 0.4` in data prep |
| Training too slow | No GPU / small batches | Use GPU, increase `--batch-size`, enable AMP |

#### Quick Diagnostic Commands

```bash
# Check training data structure
ls -R training_data/

# View metadata
cat training_data/metadata.json | python -m json.tool

# Test data loading
python -c "import numpy as np; d=np.load('training_data/train/images/tile_00000_00000.npz'); print(d['image'].shape)"
# Expected: (12, 256, 256)
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
# 1. Generate training data with balanced sampling
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data \
  --tile-size 256 \
  --balance-ratio 0.4

# 2. Train model
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16 \
  --loss combined

# 3. Run inference on new area
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
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

- **`egitim_verisi_olusturma.py`**: Creates 12-channel training tiles from GeoTIFF + ground truth masks
- **`training.py`**: Trains U-Net models with CBAM Attention support

**Quick Start:**

```bash
# 1. Create training data
python egitim_verisi_olusturma.py --input area.tif --mask mask.tif --output training_data

# 2. Train model
python training.py --data training_data --epochs 50

# 3. Use trained model
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth
```

**Key Features:**
- ✅ 12-channel input (RGB + RVT + Curvature + TPI)
- ✅ CBAM Attention (channel + spatial)
- ✅ Multiple loss functions (BCE, Dice, Combined, Focal)
- ✅ Mixed precision training
- ✅ Early stopping and checkpointing

For complete documentation, examples, and troubleshooting, see the [Model Training Guide](#-model-training-guide) section.

### Adding Custom Encoders

To add a new encoder:

```python
# In archaeo_detect.py
SUPPORTED_ENCODERS = [
    'resnet34', 'resnet50',
    'efficientnet-b3',
    'your_custom_encoder'  # Add new encoder
]
```

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

```
archaeo_detect_base/
├── archaeo_detect.py              # Main detection script
├── egitim_verisi_olusturma.py     # Training data generation
├── training.py                     # Model training script
├── evaluation.py                   # Evaluation metrics
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── training_data/                  # Generated training tiles
│   ├── train/
│   │   ├── images/                 # 12-channel image tiles (.npz)
│   │   └── masks/                  # Binary mask tiles (.npz)
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── metadata.json               # Dataset metadata
├── checkpoints/                    # Trained model weights
│   ├── best_Unet_resnet34_12ch_attention.pth
│   └── training_history.json
├── cache/                          # RVT derivatives cache
│   └── *.derivatives.npz
└── ciktilar/                       # Output detection results
    ├── *_prob.tif                  # Probability maps
    ├── *_mask.tif                  # Binary masks
    └── *_mask.gpkg                 # Vector polygons
```

### System Architecture

```
archaeo_detect.py
├── Data Loading (rasterio)
│   ├── Multi-band GeoTIFF reading
│   ├── Band selection and validation
│   └── CRS and transform preservation
├── Preprocessing
│   ├── Band reading (RGB, DSM, DTM)
│   ├── RVT derivatives (rvt-py)
│   │   ├── SVF (Sky-View Factor)
│   │   ├── Positive/Negative Openness
│   │   ├── LRM (Local Relief Model)
│   │   └── Slope
│   ├── Advanced features
│   │   ├── Curvature calculation (Plan + Profile)
│   │   ├── TPI calculation (multi-scale)
│   │   └── nDSM calculation (DSM - DTM)
│   └── Normalization (global/local percentile)
├── Detection Pipeline
│   ├── Deep Learning (PyTorch + SMP)
│   │   ├── U-Net / UnetPlusPlus / DeepLabV3+
│   │   ├── CBAM Attention (optional)
│   │   ├── Multi-encoder ensemble
│   │   └── Tile-based inference
│   ├── Classical Methods
│   │   ├── RVT visualization (SVF, Openness, LRM)
│   │   ├── Hessian Matrix (ridge/valley detection)
│   │   └── Morphology (opening, closing, top-hat)
│   ├── YOLO11 (optional)
│   │   ├── Object detection
│   │   ├── Segmentation
│   │   └── Labeled inventory
│   └── Fusion (Hybrid combination)
│       └── Weighted averaging (alpha blending)
├── Post-Processing
│   ├── Thresholding (probability → binary)
│   ├── Morphological operations
│   ├── Area filtering (min_area)
│   └── Tall object masking (nDSM-based)
└── Output Generation
    ├── Raster outputs (GeoTIFF)
    │   ├── Probability maps
    │   └── Binary masks
    └── Vector outputs (GeoPackage)
        ├── Polygon geometries
        ├── Area attributes
        └── CRS preservation
```

### Training Pipeline Architecture

```
egitim_verisi_olusturma.py
├── Input Validation
│   ├── GeoTIFF + mask file check
│   ├── Dimension matching
│   └── CRS validation
├── Feature Extraction
│   ├── RVT derivatives (SVF, Openness, LRM, Slope)
│   ├── Curvature (Plan, Profile)
│   ├── TPI (multi-scale)
│   └── nDSM calculation
├── Tile Generation
│   ├── Sliding window with overlap
│   ├── Quality filtering (nodata, positive ratio)
│   └── Balanced sampling (optional)
├── Normalization
│   └── Robust percentile scaling (2%-98%)
└── Data Export
    ├── Train/Val split
    ├── File saving (.npz or .npy)
    └── Metadata generation

training.py
├── Data Loading
│   ├── ArchaeologyDataset class
│   ├── Data augmentation (flip, rotate)
│   └── DataLoader with workers
├── Model Creation
│   ├── Architecture selection (U-Net, etc.)
│   ├── Encoder initialization
│   ├── CBAM Attention wrapper
│   └── Channel adaptation (12 channels)
├── Training Loop
│   ├── Forward pass
│   ├── Loss calculation (BCE, Dice, Combined, Focal)
│   ├── Backward pass (with AMP)
│   └── Optimizer step
├── Validation
│   ├── IoU calculation
│   ├── Loss monitoring
│   └── Best model tracking
└── Checkpointing
    ├── Best model saving
    └── Training history logging
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

**12-Channel Input Structure:**

| Channel | Feature | Description | Archaeological Use |
|---------|---------|-------------|-------------------|
| 0-2 | RGB | Red, Green, Blue | Color/texture anomalies |
| 3 | SVF | Sky-View Factor | Tumuli, mounds |
| 4 | Pos. Openness | Positive Openness | Raised structures |
| 5 | Neg. Openness | Negative Openness | Ditches, depressions |
| 6 | LRM | Local Relief Model | Local topographic anomalies |
| 7 | Slope | Terrain slope | Terraces, walls |
| 8 | nDSM | Normalized DSM | Surface height |
| 9 | Plan Curvature | Horizontal curvature | Ridge/ditch separation |
| 10 | Profile Curvature | Vertical curvature | Terraces, steps |
| 11 | TPI | Topographic Position Index | Mounds/depressions |

**CBAM Attention:**
- **Channel Attention**: Dynamically weights feature channels (e.g., SVF and TPI for tumuli, Curvature for ditches)
- **Spatial Attention**: Focuses on important regions (structure boundaries, centers)
- **Benefits**: Improves detection accuracy, reduces false positives, adapts to different structure types

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

#### Curvature Calculation

**Plan Curvature (Horizontal):**
- Measures curvature along contour lines
- Positive values: Ridges, mounds (convex)
- Negative values: Valleys, ditches (concave)
- Formula: `Kh = -[(fxx * q) - (2 * fxy * fx * fy) + (fyy * p)] / [(p + q)^1.5]`

**Profile Curvature (Vertical):**
- Measures curvature along slope direction
- Positive values: Convex surfaces (accelerating flow)
- Negative values: Concave surfaces (decelerating flow)
- Useful for detecting terraces and steps

#### TPI (Topographic Position Index)

Multi-scale TPI calculation:
```
TPI = center_elevation - mean(neighbor_elevations)
```

- **Positive TPI**: Higher than surroundings → Mounds, tumuli, hills
- **Negative TPI**: Lower than surroundings → Ditches, depressions, valleys
- **Near zero**: Flat areas or slopes

Multi-scale approach uses multiple radii (e.g., 5, 15, 30 pixels) and averages results to detect structures of different sizes.

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
  url = {https://github.com/your-username/archaeological-site-detection}
}
```

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/your-username/archaeological-site-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/archaeological-site-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/archaeological-site-detection?style=social)

---

<div align="center">

Developer: [Ahmet Ertuğrul Arık]  
Last Update: February 2026

</div>
