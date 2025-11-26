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

Use `egitim_verisi_olusturma.py` to generate 12-channel training tiles:

#### Basic Usage

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data
```

#### Advanced Options

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data \
  --tile-size 256 \
  --overlap 64 \
  --train-ratio 0.8 \
  --min-positive 0.01 \
  --max-nodata 0.3
```

**Parameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Input GeoTIFF (RGB + DSM + DTM) |
| `--mask` | Required | Ground truth mask (binary GeoTIFF) |
| `--output` | `training_data` | Output directory |
| `--tile-size` | `256` | Tile size in pixels |
| `--overlap` | `64` | Overlap between tiles |
| `--train-ratio` | `0.8` | Train/validation split ratio |
| `--min-positive` | `0.0` | Minimum positive pixel ratio (filter empty tiles) |
| `--max-nodata` | `0.3` | Maximum nodata ratio (filter invalid tiles) |
| `--format` | `npy` | File format (`npy` or `npz`) |

**What Happens:**

1. **Band Reading**: Reads RGB, DSM, DTM from input GeoTIFF
2. **RVT Calculation**: Computes SVF, Openness, LRM, Slope
3. **Advanced Features**: Calculates Curvature (Plan + Profile) and TPI
4. **Tile Generation**: Creates overlapping tiles with specified size
5. **Normalization**: Applies robust percentile-based normalization
6. **Train/Val Split**: Automatically splits tiles into train/validation sets
7. **Metadata Export**: Saves dataset information to `metadata.json`

**Output Structure:**

```
training_data/
├── train/
│   ├── images/
│   │   ├── tile_00000_00000.npy  # 12-channel array (12, 256, 256)
│   │   ├── tile_00000_00192.npy
│   │   └── ...
│   └── masks/
│       ├── tile_00000_00000.npy  # Binary mask (256, 256)
│       ├── tile_00000_00192.npy
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── metadata.json  # Dataset information
```

**Channel Order (12 channels):**

```
[0]  Red
[1]  Green
[2]  Blue
[3]  SVF (Sky-View Factor)
[4]  Positive Openness
[5]  Negative Openness
[6]  LRM (Local Relief Model)
[7]  Slope
[8]  nDSM (normalized DSM)
[9]  Plan Curvature
[10] Profile Curvature
[11] TPI (Topographic Position Index)
```

### 🚀 Step 3: Train the Model

Use `training.py` to train your custom model:

#### Basic Training

```bash
python training.py --data training_data
```

#### Advanced Training with Custom Settings

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

**Parameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `training_data` | Training data directory |
| `--arch` | `Unet` | Architecture (Unet, UnetPlusPlus, DeepLabV3Plus, etc.) |
| `--encoder` | `resnet34` | Encoder (resnet34, resnet50, efficientnet-b3, etc.) |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `8` | Batch size (adjust based on GPU memory) |
| `--lr` | `1e-4` | Learning rate |
| `--loss` | `combined` | Loss function (bce, dice, combined, focal) |
| `--patience` | `10` | Early stopping patience |
| `--no-attention` | - | Disable CBAM Attention |
| `--no-amp` | - | Disable mixed precision training |
| `--workers` | `4` | DataLoader worker threads |
| `--output` | `checkpoints` | Checkpoint directory |

**Training Features:**

- ✅ **CBAM Attention**: Enabled by default, dynamically weights channels
- ✅ **Multiple Loss Functions**: BCE, Dice, Combined, Focal
- ✅ **Mixed Precision**: FP16 training for faster training (GPU)
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Model Checkpointing**: Saves best model automatically
- ✅ **Training History**: JSON file with loss/metrics over epochs
- ✅ **Data Augmentation**: Random flips and rotations

**Training Output:**

```
checkpoints/
├── best_Unet_resnet34_12ch_attention.pth  # Best model checkpoint
└── training_history.json                  # Training metrics
```

**Monitoring Training:**

The script prints progress for each epoch:
```
Epoch   1/50 | Train Loss: 0.4523 | Val Loss: 0.3891 | Val IoU: 0.6234 | LR: 1.00e-04 | Süre: 45.2s
  → En iyi model kaydedildi: best_Unet_resnet34_12ch_attention.pth
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

### 💡 Training Tips

#### 1. Data Quality

- ✅ **High-quality masks**: Accurate ground truth is crucial
- ✅ **Balanced dataset**: Include both positive and negative examples
- ✅ **Diverse examples**: Cover different structure types and terrain conditions
- ✅ **Adequate coverage**: At least 1000+ tiles recommended

#### 2. Hyperparameter Tuning

**Learning Rate:**
- Start with `1e-4`
- If loss doesn't decrease: try `5e-5` or `1e-5`
- If loss oscillates: reduce learning rate

**Batch Size:**
- GPU memory permitting: larger batches (16-32) often better
- Limited GPU: reduce batch size, increase gradient accumulation

**Loss Function:**
- `combined`: Good starting point (BCE + Dice)
- `focal`: Useful for imbalanced datasets
- `dice`: Focuses on overlap, good for small structures

#### 3. Model Architecture Selection

| Architecture | Speed | Accuracy | Use Case |
|-------------|-------|----------|----------|
| **Unet** | Fast | Good | General purpose, recommended |
| **UnetPlusPlus** | Medium | Excellent | High accuracy needed |
| **DeepLabV3Plus** | Medium | Excellent | Multi-scale features |
| **FPN** | Fast | Good | Fast inference needed |

#### 4. Encoder Selection

| Encoder | Parameters | Speed | Accuracy |
|---------|-----------|-------|----------|
| **resnet34** | ~21M | Fast | Good (recommended start) |
| **resnet50** | ~25M | Medium | Better |
| **efficientnet-b3** | ~12M | Fast | Excellent |
| **densenet121** | ~8M | Medium | Good |

#### 5. Common Issues and Solutions

**Problem: Loss not decreasing**
- Solution: Lower learning rate, check data quality, verify masks

**Problem: Overfitting (train loss << val loss)**
- Solution: Increase dropout, use data augmentation, reduce model complexity

**Problem: GPU out of memory**
- Solution: Reduce batch size, use smaller tiles, enable mixed precision

**Problem: Training too slow**
- Solution: Enable mixed precision (`--no-amp` flag removed), use GPU, increase batch size

### 📈 Expected Results

With good quality training data:
- **IoU (Intersection over Union)**: 0.6-0.8 typical
- **F1 Score**: 0.7-0.9 typical
- **Training Time**: ~2-5 hours for 50 epochs (GPU)
- **Model Size**: 50-200 MB depending on encoder

### 🔄 Complete Training Workflow Example

```bash
# 1. Prepare your data
# (Create ground truth masks in QGIS or similar)

# 2. Generate training tiles
python egitim_verisi_olusturma.py \
  --input area1.tif \
  --mask area1_mask.tif \
  --output training_data \
  --tile-size 256

# 3. Train model
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16

# 4. Use trained model
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input new_area.tif \
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
