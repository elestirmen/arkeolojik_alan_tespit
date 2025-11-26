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

# Step 2: Train the model
python training.py --data training_data --epochs 50

# Step 3: Use your trained model
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth
```

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
# If no DTM:
bands: "1,2,3,4,0"  # Use 0 instead of DTM
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
   - Cache files should be named: `<input_name>.derivatives.npz`
   - Example: `kesif_alani.derivatives.npz` for input `kesif_alani.tif`

3. Check cache validation:
   - Cache validation checks file name and modification time
   - If input file was moved, cache should still work (file name-based validation)
   - If input file was modified, cache will be recalculated

4. Enable verbose mode to see cache status:
   ```bash
   python archaeo_detect.py --cache-derivatives -v
   ```

### Debug Mode

For detailed debugging:

```bash
python archaeo_detect.py --verbose 2 2>&1 | tee debug_log.txt
```

This command writes all debug messages to both screen and `debug_log.txt` file.

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
A: The project includes dedicated training scripts! See the [Custom Model Training](#custom-model-training) section below for step-by-step instructions using `egitim_verisi_olusturma.py` and `training.py`.

### 📊 Data Questions

**Q: What is the minimum area resolution?**  
A: Recommended: 0.5-2 meters/pixel. At lower resolution, small structures may not be detected.

**Q: Is there a maximum file size?**  
A: No, thanks to tile system, very large files can be processed. Tested: 50 GB+

**Q: Are different CRS supported?**  
A: Yes, input CRS is preserved and transferred to output.

---

## 🎓 Model Training Guide

This section provides a comprehensive guide for training custom models with your own labeled data.

### 📋 Prerequisites

Before training, you need:
- ✅ GeoTIFF files with RGB + DSM + DTM bands
- ✅ Ground truth mask files (GeoTIFF format)
  - Archaeological areas = 1 (white)
  - Background = 0 (black)
- ✅ Python environment with all dependencies installed
- ✅ GPU recommended (but CPU training is possible)

### 🛠️ Step 1: Prepare Ground Truth Masks

Create binary mask files where:
- **Value 1 (white)**: Archaeological sites/structures
- **Value 0 (black)**: Background/non-archaeological areas

**Example using QGIS:**
1. Load your RGB orthophoto
2. Create a new polygon layer
3. Digitize archaeological features
4. Export as GeoTIFF with single band (0/1 values)

**Example using Python:**
```python
import rasterio
import numpy as np
from rasterio.transform import from_bounds

# Create a simple binary mask
# (Replace with your actual digitization workflow)
mask = np.zeros((height, width), dtype=np.uint8)
# Set archaeological areas to 1
mask[archaeological_areas] = 1

# Save as GeoTIFF
with rasterio.open('ground_truth.tif', 'w',
                   driver='GTiff',
                   height=height, width=width,
                   count=1, dtype=mask.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(mask, 1)
```

### 📦 Step 2: Create Training Data

Use `egitim_verisi_olusturma.py` to generate 12-channel training tiles from your GeoTIFF files and ground truth masks.

#### Basic Usage

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data
```

#### Complete Example with All Options

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data \
  --tile-size 256 \
  --overlap 64 \
  --train-ratio 0.8 \
  --min-positive 0.01 \
  --max-nodata 0.3 \
  --balance-ratio 0.4 \
  --format npz \
  --bands 1,2,3,4,5 \
  --tpi-radii 5,15,30
```

#### All Parameters Explained

| Parameter | Default | Description | When to Use |
|-----------|---------|-------------|-------------|
| `--input`, `-i` | **Required** | Input GeoTIFF file path (RGB + DSM + DTM bands) | Always required |
| `--mask`, `-m` | **Required** | Ground truth mask file path (binary GeoTIFF: 0=background, 1=archaeological) | Always required |
| `--output`, `-o` | `training_data` | Output directory for training tiles | Change if you want different name |
| `--tile-size`, `-t` | `256` | Tile size in pixels (256, 512, etc.) | 256 for most cases, 512 for larger structures |
| `--overlap` | `64` | Overlap between tiles in pixels | Increase for better coverage (e.g., 128 for 512 tiles) |
| `--train-ratio` | `0.8` | Train/validation split ratio (0.0-1.0) | 0.8 = 80% train, 20% validation (standard) |
| `--min-positive` | `0.0` | Minimum positive pixel ratio to include tile (0.0-1.0) | 0.01 = filter tiles with <1% archaeological pixels |
| `--max-nodata` | `0.3` | Maximum nodata ratio to include tile (0.0-1.0) | 0.3 = exclude tiles with >30% nodata |
| `--balance-ratio` | `None` | Positive/negative balance ratio (0.0-1.0) | 0.4 = 40% positive, 60% negative (recommended for imbalanced data) |
| `--format` | `npz` | File format: `npy` (faster) or `npz` (compressed, smaller) | `npz` saves disk space (~50-70% smaller) |
| `--bands`, `-b` | `1,2,3,4,5` | Band order: R,G,B,DSM,DTM | Change if your bands are in different order |
| `--tpi-radii` | `5,15,30` | TPI radii in pixels (comma-separated) | Adjust for different structure sizes |
| `--no-normalize` | `False` | Disable normalization (not recommended) | Only if you want raw values |

#### Detailed Workflow

**Step-by-Step Process:**

1. **Input Validation**
   - Checks if input GeoTIFF and mask files exist
   - Verifies they have the same dimensions and CRS
   - Validates band count and data types

2. **Band Reading**
   - Reads RGB bands (typically bands 1-3)
   - Reads DSM (Digital Surface Model, band 4)
   - Reads DTM (Digital Terrain Model, band 5)
   - Handles nodata values and missing bands

3. **RVT Derivative Calculation**
   - **SVF (Sky-View Factor)**: Calculates horizon visibility (tumuli detection)
   - **Positive Openness**: Measures upward visibility (mounds)
   - **Negative Openness**: Measures downward visibility (ditches)
   - **LRM (Local Relief Model)**: Highlights local topographic anomalies
   - **Slope**: Calculates terrain steepness (terraces, walls)

4. **Advanced Feature Calculation**
   - **Plan Curvature**: Horizontal curvature (ridge/ditch separation)
   - **Profile Curvature**: Vertical curvature (terrace detection)
   - **TPI (Topographic Position Index)**: Multi-scale elevation comparison (mounds/depressions)

5. **nDSM Calculation**
   - Computes normalized DSM: `nDSM = DSM - DTM`
   - Used for masking tall objects (trees, buildings)

6. **Tile Generation**
   - Divides input image into overlapping tiles
   - Handles edge cases (partial tiles at boundaries)
   - Filters tiles based on `--min-positive` and `--max-nodata` criteria

7. **Balanced Sampling** (if `--balance-ratio` specified)
   - Separates tiles into positive (contains archaeological pixels) and negative (background only)
   - Samples negative tiles to achieve target ratio
   - Prevents class imbalance in training data

8. **Normalization**
   - Applies robust percentile-based normalization (2%-98% range)
   - Normalizes each channel independently
   - Handles outliers and extreme values

9. **Train/Validation Split**
   - Randomly splits tiles into train and validation sets
   - Maintains same positive/negative ratio in both sets
   - Uses seed for reproducibility

10. **File Saving**
    - Saves 12-channel image tiles (`.npz` or `.npy` format)
    - Saves corresponding binary masks
    - Creates directory structure: `train/images/`, `train/masks/`, `val/images/`, `val/masks/`

11. **Metadata Export**
    - Saves `metadata.json` with dataset statistics
    - Includes tile counts, positive ratios, channel information
    - Useful for tracking dataset characteristics

#### Output Structure

After running the script, you'll get the following directory structure:

```
training_data/
├── train/
│   ├── images/
│   │   ├── tile_00000_00000.npz  # 12-channel array (12, 256, 256)
│   │   ├── tile_00000_00192.npz   # Shape: (12, 256, 256)
│   │   ├── tile_00000_00384.npz
│   │   └── ...                    # More tiles
│   └── masks/
│       ├── tile_00000_00000.npz    # Binary mask (256, 256)
│       ├── tile_00000_00192.npz   # Values: 0 (background) or 1 (archaeological)
│       ├── tile_00000_00384.npz
│       └── ...                    # Corresponding masks
├── val/
│   ├── images/
│   │   ├── tile_01234_00000.npz   # Validation images
│   │   └── ...
│   └── masks/
│       ├── tile_01234_00000.npz   # Validation masks
│       └── ...
└── metadata.json                  # Dataset statistics and info
```

**File Format Details:**

- **`.npz` format (default)**: Compressed NumPy archive
  - Smaller file size (~50-70% reduction)
  - Slower read/write (still very fast)
  - Recommended for disk space savings
  
- **`.npy` format**: Uncompressed NumPy array
  - Faster read/write
  - Larger file size
  - Use if disk space is not a concern

**Loading Files:**

```python
import numpy as np

# Load .npz file
data = np.load('tile_00000_00000.npz')
image = data['image']  # Shape: (12, 256, 256)
mask = np.load('tile_00000_00000.npz')['mask']  # Shape: (256, 256)

# Load .npy file (if format was npy)
image = np.load('tile_00000_00000.npy')  # Shape: (12, 256, 256)
mask = np.load('tile_00000_00000.npy')   # Shape: (256, 256)
```

**Metadata.json Contents:**

```json
{
  "dataset_info": {
    "input_file": "kesif_alani.tif",
    "mask_file": "ground_truth.tif",
    "tile_size": 256,
    "overlap": 64,
    "train_ratio": 0.8,
    "format": "npz",
    "created_at": "2025-01-15T10:30:00"
  },
  "statistics": {
    "total_tiles": 1250,
    "train_tiles": 1000,
    "val_tiles": 250,
    "positive_tiles": 450,
    "negative_tiles": 800,
    "positive_ratio": 0.36,
    "actual_positive_ratio": 0.40
  },
  "channels": {
    "count": 12,
    "order": [
      "Red", "Green", "Blue", "SVF", "Positive Openness",
      "Negative Openness", "LRM", "Slope", "nDSM",
      "Plan Curvature", "Profile Curvature", "TPI"
    ]
  }
}
```

**Channel Order (12 channels):**

| Index | Channel | Description | Archaeological Use |
|-------|---------|-------------|---------------------|
| 0 | Red | RGB Red band | Color/texture anomalies |
| 1 | Green | RGB Green band | Vegetation patterns |
| 2 | Blue | RGB Blue band | Soil color variations |
| 3 | SVF | Sky-View Factor | Tumuli, mounds (horizon visibility) |
| 4 | Pos. Openness | Positive Openness | Raised structures (upward visibility) |
| 5 | Neg. Openness | Negative Openness | Ditches, depressions (downward visibility) |
| 6 | LRM | Local Relief Model | Local topographic anomalies |
| 7 | Slope | Terrain slope | Terraces, walls, steps |
| 8 | nDSM | Normalized DSM | Surface height (DSM - DTM) |
| 9 | Plan Curvature | Horizontal curvature | Ridge/ditch separation |
| 10 | Profile Curvature | Vertical curvature | Terraces, steps, flow direction |
| 11 | TPI | Topographic Position Index | Mounds/depressions relative to surroundings |

#### Practical Examples

**Example 1: Balanced Dataset for Imbalanced Data**

If your ground truth has very few archaeological pixels (<5%), use balanced sampling:

```bash
python egitim_verisi_olusturma.py \
  --input area.tif \
  --mask mask.tif \
  --output training_data \
  --balance-ratio 0.4 \
  --min-positive 0.01
```

This will:
- Keep all tiles with archaeological content (positive tiles)
- Sample negative tiles to achieve 40% positive, 60% negative ratio
- Filter out tiles with <1% archaeological pixels

**Example 2: Large Structures (512x512 tiles)**

For detecting large archaeological complexes:

```bash
python egitim_verisi_olusturma.py \
  --input area.tif \
  --mask mask.tif \
  --output training_data \
  --tile-size 512 \
  --overlap 128 \
  --min-positive 0.05
```

**Example 3: Multiple Areas**

Combine multiple areas into one dataset:

```bash
# Area 1
python egitim_verisi_olusturma.py \
  --input area1.tif \
  --mask area1_mask.tif \
  --output training_data_area1

# Area 2
python egitim_verisi_olusturma.py \
  --input area2.tif \
  --mask area2_mask.tif \
  --output training_data_area2

# Then manually combine or use separate datasets
```

**Example 4: Custom Band Order**

If your GeoTIFF has bands in different order (e.g., B,G,R,DSM,DTM):

```bash
python egitim_verisi_olusturma.py \
  --input area.tif \
  --mask mask.tif \
  --bands 3,2,1,4,5 \
  --output training_data
```

#### Troubleshooting Data Preparation

**Problem: "No valid tiles found"**
- **Cause**: `--min-positive` too high or `--max-nodata` too low
- **Solution**: Lower `--min-positive` (e.g., 0.0) or increase `--max-nodata` (e.g., 0.5)

**Problem: "Memory error"**
- **Cause**: Input file too large
- **Solution**: Process in smaller chunks or reduce tile size

**Problem: "Mask and input dimensions don't match"**
- **Cause**: Different resolutions or extents
- **Solution**: Resample mask to match input using GDAL:
  ```bash
  gdalwarp -tr 1.0 1.0 -r nearest mask.tif mask_resampled.tif
  ```

**Problem: "Too many negative tiles"**
- **Cause**: Imbalanced dataset (common in archaeological data)
- **Solution**: Use `--balance-ratio 0.4` to balance positive/negative tiles

**Problem: "RVT calculation too slow"**
- **Cause**: Large input files, complex terrain
- **Solution**: This is normal for first run. Subsequent runs with same input will be faster if you reuse the script (it uses caching internally)

### 🚀 Step 3: Train the Model

Use `training.py` to train your custom U-Net model with 12-channel input and CBAM Attention.

#### Basic Training

```bash
python training.py --data training_data
```

This will use default settings:
- Architecture: U-Net
- Encoder: ResNet34
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-4
- Loss: Combined (BCE + Dice)
- CBAM Attention: Enabled
- Mixed Precision: Enabled (FP16)

#### Complete Training Example

```bash
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --loss combined \
  --patience 10 \
  --workers 4 \
  --output checkpoints \
  --seed 42
```

#### All Parameters Explained

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `--data`, `-d` | `training_data` | Training data directory (from Step 2) | Path to your `training_data` folder |
| `--arch`, `-a` | `Unet` | Model architecture | `Unet` (fast, good), `UnetPlusPlus` (better accuracy), `DeepLabV3Plus` (multi-scale) |
| `--encoder`, `-e` | `resnet34` | Encoder backbone | `resnet34` (balanced), `resnet50` (better), `efficientnet-b3` (efficient) |
| `--epochs` | `50` | Number of training epochs | Start with 50, increase if loss still decreasing |
| `--batch-size`, `-b` | `8` | Batch size | Increase if GPU memory allows (16-32 better) |
| `--lr` | `1e-4` | Learning rate | Start with 1e-4, reduce if loss oscillates |
| `--loss` | `combined` | Loss function | `combined` (BCE+Dice), `focal` (imbalanced data), `dice` (small objects) |
| `--patience` | `10` | Early stopping patience | Stop if no improvement for N epochs |
| `--no-attention` | `False` | Disable CBAM Attention | Only disable if you want to test without attention |
| `--no-amp` | `False` | Disable mixed precision (FP16) | Disable only if you encounter numerical issues |
| `--workers` | `4` | DataLoader worker threads | Increase for faster data loading (4-8 typical) |
| `--output`, `-o` | `checkpoints` | Checkpoint directory | Where to save trained models |
| `--seed` | `42` | Random seed | For reproducibility |

#### Training Features

**1. CBAM Attention (Enabled by Default)**
- **Channel Attention**: Dynamically weights feature channels
  - Emphasizes important channels (e.g., SVF for tumuli, Curvature for ditches)
- **Spatial Attention**: Focuses on important spatial regions
  - Highlights structure boundaries and centers
- **Benefits**: Improves accuracy, reduces false positives, adapts to different structure types

**2. Multiple Loss Functions**

| Loss Function | Formula | Best For |
|--------------|---------|----------|
| `bce` | Binary Cross-Entropy | General purpose |
| `dice` | Dice Loss = 1 - (2\|A∩B\|)/(\|A\|+\|B\|) | Small objects, overlap-focused |
| `combined` | BCE + Dice | **Recommended** - balances both |
| `focal` | Focal Loss (α-balanced) | **Imbalanced datasets** - focuses on hard examples |

**3. Mixed Precision Training (FP16)**
- Uses half-precision (FP16) for faster training
- ~2x speedup on modern GPUs
- Automatic loss scaling prevents underflow
- Disable with `--no-amp` if you encounter issues

**4. Early Stopping**
- Monitors validation loss
- Stops training if no improvement for `--patience` epochs
- Prevents overfitting
- Saves best model automatically

**5. Data Augmentation**
- Random horizontal flip (50% probability)
- Random vertical flip (50% probability)
- Random 90° rotations (0°, 90°, 180°, 270°)
- Applied only to training set, not validation

**6. Model Checkpointing**
- Saves best model based on validation IoU
- Model name format: `best_{arch}_{encoder}_12ch_attention.pth`
- Example: `best_Unet_resnet34_12ch_attention.pth`

**7. Training History**
- Saves `training_history.json` with:
  - Loss values per epoch (train/val)
  - IoU scores per epoch
  - Learning rate schedule
  - Training time per epoch

#### Training Output Structure

```
checkpoints/
├── best_Unet_resnet34_12ch_attention.pth  # Best model checkpoint
└── training_history.json                  # Training metrics (JSON)
```

**Training History JSON Format:**

```json
{
  "config": {
    "arch": "Unet",
    "encoder": "resnet34",
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.0001,
    "loss": "combined",
    "attention": true
  },
  "history": {
    "train_loss": [0.4523, 0.3891, 0.3456, ...],
    "val_loss": [0.3891, 0.3456, 0.3123, ...],
    "val_iou": [0.6234, 0.6789, 0.7123, ...],
    "learning_rate": [0.0001, 0.0001, 0.0001, ...],
    "epoch_time": [45.2, 43.8, 44.1, ...]
  },
  "best_epoch": 15,
  "best_val_iou": 0.7891,
  "total_time": "2h 15m 30s"
}
```

#### Monitoring Training Progress

**Console Output:**

```
12 Kanallı Arkeolojik Tespit Modeli Eğitimi
==========================================
Veri dizini: training_data
Mimari: Unet
Encoder: resnet34
Epoch sayısı: 50
Batch boyutu: 8
Learning rate: 0.0001
Loss fonksiyonu: combined
CBAM Attention: Aktif
Mixed Precision: Aktif

Veri yükleme...
  Train: 1000 tile
  Val: 250 tile

Model oluşturuluyor...
  Giriş kanalları: 12
  Çıkış kanalları: 1
  Parametre sayısı: 21,234,567

Eğitim başlıyor...
Epoch   1/50 | Train Loss: 0.4523 | Val Loss: 0.3891 | Val IoU: 0.6234 | LR: 1.00e-04 | Süre: 45.2s
  → En iyi model kaydedildi: best_Unet_resnet34_12ch_attention.pth

Epoch   2/50 | Train Loss: 0.3891 | Val Loss: 0.3456 | Val IoU: 0.6789 | LR: 1.00e-04 | Süre: 43.8s
  → En iyi model kaydedildi: best_Unet_resnet34_12ch_attention.pth

...

Epoch  15/50 | Train Loss: 0.2345 | Val Loss: 0.2123 | Val IoU: 0.7891 | LR: 1.00e-04 | Süre: 44.5s
  → En iyi model kaydedildi: best_Unet_resnet34_12ch_attention.pth

...

Early stopping: En iyi model 15. epoch'ta (Val IoU: 0.7891)
Eğitim tamamlandı!
Toplam süre: 2h 15m 30s
```

**Key Metrics Explained:**

- **Train Loss**: Training loss (lower is better)
- **Val Loss**: Validation loss (lower is better, should track train loss)
- **Val IoU**: Validation Intersection over Union (higher is better, 0.6-0.8 typical)
- **LR**: Current learning rate
- **Süre**: Time per epoch

#### Training Scenarios

**Scenario 1: Quick Test Run**

```bash
python training.py \
  --data training_data \
  --epochs 5 \
  --batch-size 16
```

**Scenario 2: High Accuracy (More Epochs)**

```bash
python training.py \
  --data training_data \
  --arch UnetPlusPlus \
  --encoder resnet50 \
  --epochs 100 \
  --batch-size 8 \
  --patience 15
```

**Scenario 3: Imbalanced Dataset**

```bash
python training.py \
  --data training_data \
  --loss focal \
  --lr 5e-5 \
  --batch-size 16
```

**Scenario 4: Limited GPU Memory**

```bash
python training.py \
  --data training_data \
  --batch-size 4 \
  --workers 2 \
  --no-amp
```

**Scenario 5: Fast Training (Large Batch)**

```bash
python training.py \
  --data training_data \
  --batch-size 32 \
  --workers 8 \
  --encoder efficientnet-b3
```

### 📊 Step 4: Evaluate and Use Trained Model

#### Using Trained Model for Inference

```bash
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input new_area.tif \
  --th 0.6
```

#### Configure in config.yaml

```yaml
weights: "checkpoints/best_Unet_resnet34_12ch_attention.pth"
zero_shot_imagenet: false
encoder: "resnet34"
```

### 💡 Training Tips and Best Practices

#### 1. Data Quality Guidelines

**✅ High-Quality Masks:**
- **Accurate digitization**: Precise boundaries are crucial
- **Consistent labeling**: Same structure types labeled consistently
- **Complete coverage**: All visible archaeological features included
- **Avoid noise**: Don't include ambiguous areas

**✅ Balanced Dataset:**
- **Positive examples**: Archaeological structures (tumuli, ditches, walls, etc.)
- **Negative examples**: Background terrain (fields, forests, roads)
- **Recommended ratio**: 30-50% positive, 50-70% negative (use `--balance-ratio` if needed)
- **Diverse negatives**: Include various terrain types, not just empty fields

**✅ Diverse Examples:**
- **Different structure types**: Tumuli, ditches, walls, mounds, terraces
- **Various sizes**: Small (5-10m) to large (50-100m+) structures
- **Different terrain**: Flat, hilly, forested, agricultural areas
- **Different seasons**: If possible, include data from different times

**✅ Adequate Coverage:**
- **Minimum**: 500-1000 tiles (for initial testing)
- **Recommended**: 2000-5000 tiles (for production models)
- **Large datasets**: 10000+ tiles (for best accuracy)
- **Validation set**: Should be 15-25% of total data

#### 2. Hyperparameter Tuning Guide

**Learning Rate (`--lr`):**

| Scenario | Learning Rate | When to Use |
|----------|---------------|-------------|
| **Initial training** | `1e-4` | Default, good starting point |
| **Loss not decreasing** | `5e-5` or `1e-5` | Reduce if loss plateaus |
| **Loss oscillating** | `5e-5` | Reduce if loss jumps around |
| **Fine-tuning** | `1e-5` or `5e-6` | When continuing from checkpoint |
| **Large batch (>32)** | `2e-4` or `3e-4` | Can use higher LR with large batches |

**Batch Size (`--batch-size`):**

| GPU Memory | Batch Size | Notes |
|------------|------------|-------|
| **4 GB** | 4-8 | Minimum viable |
| **8 GB** | 8-16 | Comfortable |
| **12 GB** | 16-24 | Good performance |
| **16 GB+** | 24-32 | Optimal |

**Tips:**
- Larger batches → more stable gradients → better convergence
- If OOM error: reduce batch size or use `--no-amp`
- Batch size affects effective learning rate (larger batch = can use higher LR)

**Loss Function (`--loss`):**

| Loss Function | When to Use | Advantages |
|---------------|-------------|------------|
| `combined` | **Default** - Most cases | Balances pixel-level and overlap metrics |
| `focal` | Imbalanced datasets (<30% positive) | Focuses on hard examples, reduces class imbalance impact |
| `dice` | Small objects, overlap-focused | Emphasizes intersection over union |
| `bce` | Simple baseline | Standard binary classification loss |

**Epochs and Patience:**

| Dataset Size | Epochs | Patience | Notes |
|--------------|--------|----------|-------|
| **Small (<1000 tiles)** | 50-100 | 10-15 | May need more epochs |
| **Medium (1000-5000)** | 50-80 | 10-15 | Standard |
| **Large (>5000)** | 30-50 | 5-10 | Usually converges faster |

#### 3. Model Architecture Selection

| Architecture | Speed | Accuracy | Memory | Use Case |
|-------------|-------|----------|--------|----------|
| **Unet** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | **Recommended start** - General purpose |
| **UnetPlusPlus** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💾💾 Medium | High accuracy needed, dense connections |
| **DeepLabV3Plus** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💾💾 Medium | Multi-scale features, ASPP module |
| **FPN** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | Fast inference, feature pyramid |
| **PSPNet** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💾💾 Medium | Pyramid pooling, multi-scale |
| **MAnet** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💾💾 Medium | Multi-attention mechanism |
| **Linknet** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | Efficient decoder, fast |

**Recommendation:**
- **Start with Unet + ResNet34**: Fast, reliable, good baseline
- **Upgrade to UnetPlusPlus + ResNet50**: If you need better accuracy
- **Try DeepLabV3Plus**: If structures vary greatly in size

#### 4. Encoder Selection Guide

| Encoder | Parameters | Speed | Accuracy | Memory | Best For |
|---------|-----------|-------|----------|--------|----------|
| **resnet34** | ~21M | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | **Recommended start** - Balanced |
| **resnet50** | ~25M | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | 💾💾 Medium | Better accuracy, more parameters |
| **resnet101** | ~44M | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | 💾💾💾 High | Maximum accuracy, slower |
| **efficientnet-b0** | ~5M | ⚡⚡⚡⚡ Very Fast | ⭐⭐ Fair | 💾 Very Low | Mobile/edge devices |
| **efficientnet-b3** | ~12M | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | 💾 Low | **Recommended** - Efficient |
| **efficientnet-b5** | ~30M | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | 💾💾 Medium | High accuracy, efficient |
| **densenet121** | ~8M | ⚡⚡ Medium | ⭐⭐⭐ Good | 💾💾 Medium | Dense connections |
| **vgg16** | ~15M | ⚡⚡ Medium | ⭐⭐⭐ Good | 💾💾 Medium | Classic architecture |

**Recommendation:**
- **Start**: ResNet34 (balanced)
- **Better accuracy**: ResNet50 or EfficientNet-B3
- **Efficiency**: EfficientNet-B3 (best speed/accuracy tradeoff)
- **Maximum accuracy**: ResNet101 or EfficientNet-B5

#### 5. Common Issues and Solutions

**Problem: Loss Not Decreasing**

**Symptoms:**
- Loss stays constant or increases
- Val IoU doesn't improve

**Solutions:**
1. **Lower learning rate**: Try `5e-5` or `1e-5`
2. **Check data quality**: Verify masks are correct, check for label errors
3. **Verify data loading**: Ensure images and masks match correctly
4. **Check normalization**: Data should be normalized (default: yes)
5. **Try different loss**: Switch to `focal` if dataset is imbalanced
6. **Increase batch size**: Larger batches stabilize training

**Problem: Overfitting (Train Loss << Val Loss)**

**Symptoms:**
- Train loss decreases but val loss increases
- Train IoU >> Val IoU (e.g., 0.9 vs 0.6)

**Solutions:**
1. **More data**: Increase training dataset size
2. **Data augmentation**: Already enabled, but verify it's working
3. **Reduce model complexity**: Use ResNet34 instead of ResNet50
4. **Early stopping**: Reduce `--patience` to stop earlier
5. **Regularization**: Model already includes dropout, but you can add more
6. **Reduce learning rate**: Lower LR can help generalization

**Problem: GPU Out of Memory (OOM)**

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size**: Try `--batch-size 4` or `2`
2. **Disable mixed precision**: Use `--no-amp` (slower but uses less memory)
3. **Reduce workers**: Use `--workers 2` or `1`
4. **Use smaller encoder**: Switch to EfficientNet-B0 or ResNet18
5. **Process in chunks**: Not applicable here, but consider smaller tiles in data prep

**Problem: Training Too Slow**

**Symptoms:**
- Each epoch takes >5 minutes
- Total training time >10 hours

**Solutions:**
1. **Enable mixed precision**: Remove `--no-amp` flag (default enabled)
2. **Increase batch size**: Larger batches = fewer iterations per epoch
3. **Increase workers**: Use `--workers 8` for faster data loading
4. **Use GPU**: Ensure CUDA is available (`torch.cuda.is_available()`)
5. **Use efficient encoder**: Switch to EfficientNet-B3
6. **Reduce tile size**: Smaller tiles = faster processing (but may reduce accuracy)

**Problem: Validation Loss Oscillates**

**Symptoms:**
- Val loss jumps up and down
- No clear improvement trend

**Solutions:**
1. **Lower learning rate**: Try `5e-5` or `1e-5`
2. **Increase batch size**: More stable gradients
3. **Check validation set**: Ensure it's representative and not too small
4. **Reduce learning rate schedule**: Model uses constant LR, consider manual reduction

**Problem: Model Predicts Everything as Background**

**Symptoms:**
- All predictions are 0 (no archaeological sites detected)
- Very low positive predictions

**Solutions:**
1. **Check class imbalance**: Use `--loss focal` for imbalanced data
2. **Lower threshold**: In inference, use `--th 0.3` instead of 0.5
3. **Check masks**: Verify ground truth has positive pixels
4. **Use balanced sampling**: In data prep, use `--balance-ratio 0.4`
5. **Check normalization**: Ensure data is properly normalized

**Problem: Model Predicts Everything as Archaeological**

**Symptoms:**
- All predictions are 1 (everything detected as archaeological)
- Very high false positive rate

**Solutions:**
1. **Increase threshold**: In inference, use `--th 0.7` or higher
2. **More negative examples**: Add more background tiles to training data
3. **Check data quality**: Ensure masks don't have labeling errors
4. **Use balanced sampling**: In data prep, use `--balance-ratio 0.3` (more negatives)

#### 6. Training Workflow Best Practices

**Step 1: Start Small**
```bash
# Quick test run
python training.py --data training_data --epochs 5 --batch-size 8
```

**Step 2: Baseline Training**
```bash
# Standard training
python training.py --data training_data --epochs 50
```

**Step 3: Optimize**
```bash
# If results are good, try better architecture
python training.py --data training_data --arch UnetPlusPlus --encoder resnet50
```

**Step 4: Fine-tune**
```bash
# Continue from best checkpoint with lower LR
python training.py --data training_data --lr 5e-5 --epochs 30
```

**Step 5: Evaluate**
- Check `training_history.json` for trends
- Test on validation set
- Visualize predictions on sample tiles

### 📈 Expected Results and Performance

**With Good Quality Training Data:**

| Metric | Typical Range | Excellent | Notes |
|--------|---------------|-----------|-------|
| **Val IoU** | 0.6-0.8 | >0.8 | Intersection over Union |
| **Val F1 Score** | 0.7-0.9 | >0.9 | Harmonic mean of precision/recall |
| **Val Precision** | 0.7-0.9 | >0.9 | Low false positives |
| **Val Recall** | 0.6-0.8 | >0.8 | Low false negatives |
| **Training Time** | 2-5 hours | - | 50 epochs, GPU, 2000 tiles |
| **Model Size** | 50-200 MB | - | Depends on encoder |

**Performance by Architecture:**

| Architecture | Val IoU | Training Time | Inference Speed |
|-------------|---------|---------------|-----------------|
| Unet + ResNet34 | 0.65-0.75 | 2-3 hours | Fast |
| UnetPlusPlus + ResNet50 | 0.75-0.85 | 4-6 hours | Medium |
| DeepLabV3Plus + EfficientNet-B3 | 0.70-0.80 | 3-4 hours | Medium |

**Factors Affecting Results:**

- ✅ **Data quality**: Accurate masks → better results
- ✅ **Dataset size**: More tiles → better generalization
- ✅ **Class balance**: Balanced dataset → more stable training
- ✅ **Diversity**: Various terrain types → better generalization
- ✅ **Hyperparameters**: Proper tuning → optimal performance

### 🔄 Complete Training Workflow Example

**Full End-to-End Example:**

#### Step 1: Prepare Ground Truth Masks

Create binary masks in QGIS:
1. Load RGB orthophoto
2. Create polygon layer
3. Digitize archaeological features
4. Export as GeoTIFF (single band, 0/1 values)

#### Step 2: Generate Training Data

```bash
# Basic data generation
python egitim_verisi_olusturma.py \
  --input area1.tif \
  --mask area1_mask.tif \
  --output training_data \
  --tile-size 256 \
  --overlap 64 \
  --balance-ratio 0.4

# Expected output:
# training_data/
# ├── train/ (1000 tiles)
# ├── val/ (250 tiles)
# └── metadata.json
```

**Check metadata.json:**
```bash
cat training_data/metadata.json
```

#### Step 3: Train Model

```bash
# Initial training
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --loss combined

# Expected output:
# checkpoints/
# ├── best_Unet_resnet34_12ch_attention.pth
# └── training_history.json
```

**Monitor training:**
- Watch console output for loss and IoU trends
- Check `training_history.json` for detailed metrics
- Early stopping will save best model automatically

#### Step 4: Evaluate Model

**Check training history:**
```python
import json

with open('checkpoints/training_history.json') as f:
    history = json.load(f)

print(f"Best Val IoU: {history['history']['val_iou'][history['best_epoch']]:.4f}")
print(f"Best epoch: {history['best_epoch']}")
```

#### Step 5: Use Trained Model

```bash
# Inference with trained model
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input new_area.tif \
  --th 0.6 \
  --enable-fusion \
  --encoder resnet34

# Or configure in config.yaml
```

**config.yaml:**
```yaml
weights: "checkpoints/best_Unet_resnet34_12ch_attention.pth"
zero_shot_imagenet: false
encoder: "resnet34"
enable_attention: true
```

#### Step 6: Iterate and Improve

**If results are not satisfactory:**

1. **Add more training data**
   ```bash
   # Generate more tiles from additional areas
   python egitim_verisi_olusturma.py --input area2.tif --mask area2_mask.tif --output training_data_area2
   # Manually combine or retrain with more data
   ```

2. **Try different architecture**
   ```bash
   python training.py --data training_data --arch UnetPlusPlus --encoder resnet50
   ```

3. **Fine-tune hyperparameters**
   ```bash
   python training.py --data training_data --lr 5e-5 --loss focal --batch-size 32
   ```

### 📊 Real-World Example: Kapadokya Region

**Scenario:** Detecting old settlement remains in Kapadokya region

**Step 1: Data Preparation**
```bash
python egitim_verisi_olusturma.py \
  --input kapadokya_area.tif \
  --mask kapadokya_mask.tif \
  --output kapadokya_training \
  --tile-size 256 \
  --balance-ratio 0.4 \
  --min-positive 0.01
```

**Step 2: Training**
```bash
python training.py \
  --data kapadokya_training \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16 \
  --loss combined
```

**Step 3: Results**
- Val IoU: 0.72
- Val F1: 0.81
- Training time: 3h 15m
- Model: `best_Unet_resnet34_12ch_attention.pth` (67 MB)

**Step 4: Inference**
```bash
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input new_kapadokya_area.tif \
  --th 0.65 \
  --enable-fusion
```

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

### System Architecture

```
archaeo_detect.py
├── Data Loading (rasterio)
├── Preprocessing
│   ├── Band reading
│   ├── RVT derivatives (rvt-py)
│   ├── Curvature calculation (Plan + Profile)
│   ├── TPI calculation (multi-scale)
│   ├── nDSM calculation
│   └── Normalization
├── Detection
│   ├── Deep Learning (PyTorch + SMP)
│   │   ├── U-Net
│   │   ├── DeepLabV3+
│   │   ├── CBAM Attention (optional)
│   │   └── Other architectures
│   ├── Classical Methods
│   │   ├── RVT (SVF, Openness, LRM)
│   │   ├── Hessian Matrix
│   │   └── Morphology (scikit-image)
│   └── Fusion (Hybrid)
├── Post-Processing
│   ├── Thresholding
│   ├── Morphological post-processing
│   └── Area filtering
└── Output
    ├── Raster (GeoTIFF)
    └── Vector (GeoPackage)
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
Last Update: October 2025

</div>
