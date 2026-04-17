# ğŸ›ï¸ Arkeolojik Alan Tespiti (Derin Ã–ÄŸrenme + Klasik GÃ¶rÃ¼ntÃ¼ Ä°ÅŸleme)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Ä°ngilizce dokÃ¼mantasyon: [`README.md`](README.md).

> **Ã‡ok bantlÄ± GeoTIFF verilerinden arkeolojik yapÄ±larÄ±n otomatik tespiti iÃ§in geliÅŸmiÅŸ yapay zeka sistemi. Ã–ncelikli olarak Ä°HA (insansÄ±z hava aracÄ±) nadir gÃ¶rÃ¼ntÃ¼lerinden Ã¼retilen tÃ¼revler Ã¼zerinde Ã§alÄ±ÅŸÄ±râ€”ortofoto, DSM/DTM ve tÃ¼retilmiÅŸ kabartma kanallarÄ±. Uydu gÃ¶rÃ¼ntÃ¼leri ve diÄŸer hava/LiDAR kaynaklarÄ± da aynÄ± Ã§ok bantlÄ± GeoTIFF formatÄ±nda saÄŸlandÄ±ÄŸÄ±nda desteklenir.**

Bu proje, Ã§ok bantlÄ± GeoTIFF verilerinden (RGB, DSM, DTM) arkeolojik izleri (tÃ¼mÃ¼lÃ¼s, hendek, hÃ¶yÃ¼k, duvar kalÄ±ntÄ±larÄ± vb.) tespit etmek iÃ§in **derin Ã¶ÄŸrenme** ve **klasik gÃ¶rÃ¼ntÃ¼ iÅŸleme** yÃ¶ntemlerini birleÅŸtirir. Girdi verileri Ã§oÄŸunlukla **Ä°HA fotogrametrisinden** elde edilir; **uydu gÃ¶rÃ¼ntÃ¼leri veya diÄŸer hava Ã¼rÃ¼nleri** de bant yapÄ±sÄ± ve jeoreferans uyumlu olduÄŸu sÃ¼rece kullanÄ±labilir.

### Depodaki varsayilan is akisi (`config.yaml`)

Eger `config.local.yaml` varsa CLI otomatik olarak onu tercih eder; yoksa `config.yaml` kullanilir.

KayÄ±tlÄ± profil **karo dÃ¼zeyinde sÄ±nÄ±flandÄ±rma** (`dl_task: tile_classification`) ve **tek eÄŸitilmiÅŸ checkpoint** (`trained_model_only: true`) iÃ§in ayarlÄ±dÄ±r. Bu modda:

- **`weights`** (`.pth` dosyasÄ±) ve **`training_metadata`** (eÄŸitimden gelen JSON) kullanÄ±lÄ±r.
- **`tile`**, **`overlap`** ve **`bands`** Ã§Ä±karÄ±m sÄ±rasÄ±nda `training_metadata.json` iÃ§inden kilitlenir; YAMLâ€™da yalnÄ±zca `overlap` deÄŸerini artÄ±rarak uyumsuzluÄŸu gidermeyin â€” farklÄ± overlap iÃ§in veri Ã¼retimini ve eÄŸitimi o overlap ile yeniden yapÄ±n.
- BaÅŸarÄ±lÄ± bir `training.py` Ã§alÄ±ÅŸmasÄ±ndan sonra en iyi aÄŸÄ±rlÄ±klar `workspace/checkpoints/active/model.pth` dosyasÄ±na, metadata ise `workspace/checkpoints/active/training_metadata.json` dosyasÄ±na kopyalanÄ±r (`weights` yolunu `workspace/checkpoints/active/` altÄ±ndaki baÅŸka bir checkpointâ€™e de yÃ¶nlendirebilirsiniz).

**Model girdi kanallarÄ± (gÃ¼ncel kod):** derin Ã¶ÄŸrenme tensÃ¶rÃ¼ **5 kanaldÄ±r** â€” **R, G, B, SVF, SLRM** â€” sÄ±rasÄ±yla `archeo_shared/channels.py` iÃ§indeki `MODEL_CHANNEL_NAMES` ile tanÄ±mlÄ±dÄ±r. GeoTIFF **5 bantlÄ±** kalÄ±r (RGB + DSM + DTM). **SVF** (Sky-View Factor) ve **SLRM** (RVT ile DTM Ã¼zerinde hesaplanan Simple Local Relief Model) `archaeo_detect.py` ve veri hazÄ±rlÄ±k betikleri **iÃ§inde tÃ¼retilir**; ayrÄ± GeoTIFF bandÄ± deÄŸildirler. Eski belgelerde geÃ§en **12 kanallÄ±** tensÃ¶r (nDSM, Ã§ok Ã¶lÃ§ekli TPI, ek RVT aÃ§Ä±klÄ±k kanallarÄ± vb.) **Ã¶nceki bir ÅŸemayÄ±** anlatÄ±r; mevcut eÄŸitim ve Ã§Ä±karÄ±m yolu bu yapÄ±yÄ± kullanmaz.

---

## ğŸ“‘ Ä°Ã§indekiler

- [âœ¨ Ã–zellikler](#-Ã¶zellikler)
- [ğŸ¯ Ne Yapar](#-ne-yapar)
- [ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§](#-hÄ±zlÄ±-baÅŸlangÄ±Ã§)
- [ğŸ“¦ Kurulum](#-kurulum)
- [ğŸ·ï¸ Ground Truth Etiketleme AracÄ± (`ground_truth_kare_etiketleme_qt.py`)](#%EF%B8%8F-ground-truth-etiketleme-aracÄ±-ground_truth_kare_etiketleme_qtpy)
- [ğŸ® KullanÄ±m](#-kullanÄ±m)
- [âš™ï¸ YapÄ±landÄ±rma](#ï¸-yapÄ±landÄ±rma)
- [ğŸ“‚ Ã‡Ä±ktÄ± DosyalarÄ±](#-Ã§Ä±ktÄ±-dosyalarÄ±)
- [ğŸ”¬ NasÄ±l Ã‡alÄ±ÅŸÄ±r](#-nasÄ±l-Ã§alÄ±ÅŸÄ±r)
- [ğŸ’¡ KullanÄ±m SenaryolarÄ±](#-kullanÄ±m-senaryolarÄ±)
- [ğŸ¨ GÃ¶rselleÅŸtirme](#-gÃ¶rselleÅŸtirme)
- [âš¡ Performans Optimizasyonu](#-performans-optimizasyonu)
- [ğŸ› Sorun Giderme](#-sorun-giderme)
- [â“ SSS](#-sss)
- [ğŸ“ Model EÄŸitimi KÄ±lavuzu](#-model-eÄŸitimi-kÄ±lavuzu)
- [ğŸ”¬ GeliÅŸmiÅŸ Ã–zellikler](#-geliÅŸmiÅŸ-Ã¶zellikler)
- [ğŸ“š Teknik Detaylar](#-teknik-detaylar)
- [ğŸ¤ KatkÄ±da Bulunma](#-katkÄ±da-bulunma)
- [ğŸ“„ Lisans](#-lisans)

---

## âœ¨ Ã–zellikler

### ğŸ§  DÃ¶rt GÃ¼Ã§lÃ¼ YÃ¶ntem
- **Derin Ã–ÄŸrenme**: U-Net, DeepLabV3+ ve diÄŸer modern segmentasyon mimarileri
- **YOLO11 (YENÄ°!)**: Ultralytics YOLO11 ile hÄ±zlÄ± nesne tespiti ve segmentasyon + etiketli arazi envanteri ğŸ·ï¸
  - âš ï¸ **Not:** Nadir (kuÅŸbakÄ±ÅŸÄ±) gÃ¶rÃ¼ntÃ¼ler iÃ§in ince ayar gereklidir (bkz. docs/YOLO11_NADIR_TRAINING.md)
- **Klasik GÃ¶rÃ¼ntÃ¼ Ä°ÅŸleme**: RVT (Kabartma GÃ¶rselleÅŸtirme AraÃ§ Kutusu), Hessian matrisi, Morfolojik operatÃ¶rler
- **Hibrit FÃ¼zyon**: Her yÃ¶ntemin gÃ¼Ã§lÃ¼ yÃ¶nlerini birleÅŸtiren akÄ±llÄ± fÃ¼zyon

### ğŸ¯ AkÄ±llÄ± Tespit Ã–zellikleri
- âœ… **Ã‡oklu KodlayÄ±cÄ± DesteÄŸi**: ResNet, EfficientNet, VGG, DenseNet, MobileNet ve daha fazlasÄ±
- âœ… **SÄ±fÄ±r AtÄ±ÅŸ Ã–ÄŸrenme**: ImageNet aÄŸÄ±rlÄ±klarÄ±nÄ± kullanarak eÄŸitilmiÅŸ modeller olmadan bile Ã§alÄ±ÅŸÄ±r
- âœ… **Topluluk Ã–ÄŸrenme**: Daha gÃ¼venilir tespit iÃ§in birden fazla kodlayÄ±cÄ±nÄ±n sonuÃ§larÄ±nÄ± birleÅŸtirir
- âœ… **Ã‡ok Ã–lÃ§ekli Analiz**: FarklÄ± boyutlardaki yapÄ±larÄ± tespit eder
- âœ… **ğŸ†• Etiketli Nesne Tespiti**: YOLO11 ile 80 farklÄ± nesne sÄ±nÄ±fÄ±nÄ±n otomatik etiketlenmesi (aÄŸaÃ§lar, binalar, araÃ§lar vb.)
- âœ… **ğŸ†• 5 kanallÄ± DL yÄ±ÄŸÄ±nÄ±**: Rasterdan R, G, B; DTM Ã¼zerinden RVT ile **SVF** ve **SLRM** â€” kodda birleÅŸtirilir, ekstra GeoTIFF bandÄ± deÄŸildir
- âœ… **ğŸ†• CBAM dikkat (isteÄŸe baÄŸlÄ±)**: `training.py` iÃ§inde etkinleÅŸtirilebilir; kayÄ±tlÄ± `CONFIG` varsayÄ±lanÄ±nda genelde kapalÄ± (`no_attention: true`)

### ğŸ”§ Teknik Ã–zellikler
- ğŸš€ **Karo TabanlÄ± Ä°ÅŸleme**: BÃ¼yÃ¼k gÃ¶rÃ¼ntÃ¼ler iÃ§in bellek verimli iÅŸleme
- ğŸ¨ **Sorunsuz Mozaikleme**: KosinÃ¼s yumuÅŸatma ile karo sÄ±nÄ±rlarÄ±nda artefakt yok
- ğŸ“Š **SaÄŸlam Normalizasyon**: Global veya yerel yÃ¼zdelik tabanlÄ± normalizasyon
- âš¡ **Ã–nbellek Sistemi**: RVT hesaplamalarÄ±nÄ± Ã¶nbelleÄŸe alarak 10-100x hÄ±zlanma
- ğŸ¯ **AkÄ±llÄ± Maskeleme**: YÃ¼ksek yapÄ±larÄ±n (aÄŸaÃ§lar, binalar) otomatik filtrelenmesi
- ğŸ“ **VektÃ¶rleÅŸtirme**: SonuÃ§larÄ± CBS uyumlu Ã§okgenlere dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r
- ğŸ·ï¸ **Ground Truth Etiketleme**: Katman yÃ¶netimli interaktif Qt tabanlÄ± GeoTIFF etiketleme aracÄ±

### ğŸŒ CBS Entegrasyonu
- ğŸ“ GeoPackage (.gpkg) formatÄ±nda vektÃ¶r Ã§Ä±ktÄ±sÄ±
- ğŸ—ºï¸ CoÄŸrafi koordinat sistemi (CRS) korunur
- ğŸ“ Alan hesaplama ve filtreleme
- ğŸ¯ QGIS, ArcGIS ve benzeri yazÄ±lÄ±mlarla uyumlu

---

## ğŸ¯ Ne Yapar

Bu sistem aÅŸaÄŸÄ±daki arkeolojik Ã¶zellikleri tespit edebilir:

| YapÄ± Tipi | AÃ§Ä±klama | Tespit YÃ¶ntemi |
|-----------|----------|----------------|
| ğŸ”ï¸ **TÃ¼mÃ¼lÃ¼sler** | YÃ¼kseltilmiÅŸ mezar hÃ¶yÃ¼kleri | RVT + Hessian + DL |
| ğŸ›ï¸ **HÃ¶yÃ¼kler** | YerleÅŸim hÃ¶yÃ¼kleri | TÃ¼m yÃ¶ntemler |
| ğŸ§± **Duvar KalÄ±ntÄ±larÄ±** | DoÄŸrusal yapÄ± izleri | Hessian + DL |
| â­• **Halka Hendekler** | Dairesel savunma yapÄ±larÄ± | Morfolojik + DL |
| ğŸ° **Kale KalÄ±ntÄ±larÄ±** | BÃ¼yÃ¼k yapÄ± kompleksleri | FÃ¼zyon (en etkili) |
| ğŸº **YerleÅŸim Ä°zleri** | DÃ¼zensiz topografik anomaliler | Klasik + DL |
| ğŸ›¤ï¸ **Antik Yollar** | DoÄŸrusal yÃ¼kseklik deÄŸiÅŸimleri | Hessian + RVT |

---

## ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§

### UÃ§tan uca: etiket â†’ karo â†’ eÄŸitim â†’ tespit

```bash
pip install -r requirements.txt

# 1a) Eski paired dÃ¼zen (images + masks): segmentation veya geri uyumluluk iÃ§in
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data

# 1b) DoÄŸrudan Positive/Negative klasÃ¶rleri Ã¼reten tile-classification dataset
python prepare_tile_classification_dataset.py \
  --pair kesif_alani.tif ground_truth.tif \
  --output-dir workspace/training_data_classification \
  --sampling-mode selected_regions \
  --overwrite

# 2) EÄŸitim (training.py artÄ±k iki dÃ¼zeni de kabul ediyor)
python training.py --data workspace/training_data_classification --task tile_classification --epochs 50

# 3) Ã‡Ä±karÄ±m (config.yaml; workspace/checkpoints/active/ yayÄ±nÄ± eÄŸitim Ã§Ä±ktÄ±sÄ±nda Ã¶zetlenir)
python archaeo_detect.py
```

EÄŸitim sonrasÄ± artefactâ€™lar:

- `workspace/checkpoints/active/model.pth` â€” Ã§Ä±karÄ±m iÃ§in kopyalanan en iyi aÄŸÄ±rlÄ±klar
- `workspace/checkpoints/active/training_metadata.json` â€” `trained_model_only: true` iken **`tile` / `overlap` / `bands` iÃ§in kaynak**

**Ã–nemli:** `trained_model_only: true` iken YAMLâ€™da yalnÄ±zca `overlap` artÄ±rarak eÄŸitimle uyumu â€œsaÄŸlamayaâ€ Ã§alÄ±ÅŸmayÄ±n; metadata bu alanlarÄ± kilitler. FarklÄ± overlap gerekiyorsa veriyi o overlap ile Ã¼retin ve modeli yeniden eÄŸitin.

**HenÃ¼z eÄŸitilmiÅŸ model yoksa:** sÄ±fÄ±r atÄ±ÅŸ / klasik yollar iÃ§in [KullanÄ±m](#-kullanÄ±m) bÃ¶lÃ¼mÃ¼ne bakÄ±n; ana `config.yaml` yerine Ã¶rnek profil ile denemek iÃ§in:

```bash
python archaeo_detect.py --config configs/tile_classification_baseline.example.yaml
```

### YalnÄ±zca tespit (ortam hazÄ±r)

```bash
python archaeo_detect.py
```

`config.yaml` iÃ§indeki girdi rasterÄ±, yÃ¶ntemler ve eÅŸikler kullanÄ±lÄ±r. SonuÃ§lar `workspace/ciktilar/<oturum>/` altÄ±na yazÄ±lÄ±r.

### IDE / CLI ile veri hazÄ±rlÄ±ÄŸÄ±

`egitim_verisi_olusturma.py` dosyasÄ±nda bir `CONFIG` sÃ¶zlÃ¼ÄŸÃ¼ vardÄ±r (varsayÄ±lan `input`, `mask`, `output`, `tile_size`, `overlap`, `bands`, â€¦). `--input` / `--mask` vermeden Ã§alÄ±ÅŸtÄ±rÄ±yorsanÄ±z bu anahtarlarÄ±n `CONFIG` iÃ§inde dolu olmasÄ± gerekir â€” girdi iÃ§in etkileÅŸimli dosya penceresi yoktur.

---

## ğŸ“¦ Kurulum

### Sistem Gereksinimleri

| Gereksinim | Minimum | Ã–nerilen |
|------------|---------|----------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8 GB | 16 GB+ |
| **Disk AlanÄ±** | 2 GB | 5 GB+ |
| **GPU** | Yok (CPU ile Ã§alÄ±ÅŸÄ±r) | NVIDIA CUDA destekli GPU |

### AdÄ±m AdÄ±m Kurulum

#### 1ï¸âƒ£ Python ve Pip KontrolÃ¼

```bash
python --version  # Python 3.10 veya Ã¼stÃ¼ olmalÄ±
pip --version     # pip yÃ¼klÃ¼ olmalÄ±
```

#### 2ï¸âƒ£ Sanal Ortam OluÅŸturma (Ã–nerilir)

```bash
# Windows
python -m venv .venv310
.venv310\Scripts\activate

# Linux/Mac
python -m venv .venv310
source .venv310/bin/activate
```

**Not:** `.venv310` opsiyoneldir. Conda (`archeo`) kullanÄ±yorsanÄ±z `.venv310` oluÅŸturmanÄ±z gerekmez ve varsa silebilirsiniz.

#### 3ï¸âƒ£ Gerekli Paketleri YÃ¼kleme

```bash
pip install -r requirements.txt
```

**requirements.txt iÃ§eriÄŸi:**
- `torch>=2.0.0` - PyTorch (derin Ã¶ÄŸrenme)
- `torchvision>=0.15.0` - GÃ¶rÃ¼ntÃ¼ iÅŸleme
- `segmentation-models-pytorch>=0.3.2` - Segmentasyon modelleri
- `rasterio>=1.3.0` - Raster veri okuma/yazma
- `fiona>=1.9.0` - VektÃ¶r veri iÅŸleme
- `geopandas>=0.12.0` - CoÄŸrafi veri analizi
- `opencv-python>=4.7.0` - GÃ¶rÃ¼ntÃ¼ iÅŸleme
- `scikit-image>=0.20.0` - GeliÅŸmiÅŸ gÃ¶rÃ¼ntÃ¼ iÅŸleme
- `scipy>=1.10.0` - Bilimsel hesaplama
- `numpy>=1.24.0` - SayÄ±sal iÅŸlemler
- `rvt-py>=1.2.0` (Python < 3.11) veya `rvt>=2.0.0` (Python >= 3.11) - Kabartma GÃ¶rselleÅŸtirme AraÃ§ Kutusu
- `pyyaml>=6.0` - YAML yapÄ±landÄ±rma dosyalarÄ±

#### 4ï¸âƒ£ GDAL Kurulumu (Ä°steÄŸe BaÄŸlÄ± ama Ã–nerilir)

**Windows:**
```bash
# OSGeo4W veya Conda ile
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

#### 5ï¸âƒ£ GPU DesteÄŸi (Ä°steÄŸe BaÄŸlÄ±)

NVIDIA GPU'nuz varsa, CUDA'yÄ± yÃ¼kleyin:

```bash
# CUDA 11.8 iÃ§in
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 iÃ§in
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU kontrolÃ¼:
```python
import torch
print(torch.cuda.is_available())  # True olmalÄ±
```

---

## ğŸ·ï¸ Ground Truth Etiketleme AracÄ± (`ground_truth_kare_etiketleme_qt.py`)

GeoTIFF gÃ¶rÃ¼ntÃ¼leri Ã¼zerinde ikili (binary) ground truth maskeleri oluÅŸturmak iÃ§in interaktif Qt tabanlÄ± etiketleme aracÄ±. Raster verinizin Ã¶nizlemesi Ã¼zerinde dikdÃ¶rtgenler Ã§izerek model eÄŸitimi iÃ§in piksel dÃ¼zeyinde doÄŸru GeoTIFF maskeleri oluÅŸturabilirsiniz.

### âœ¨ Temel Ã–zellikler

| Ã–zellik | AÃ§Ä±klama |
|---------|----------|
| **ğŸ–±ï¸ DikdÃ¶rtgen Ã‡izim** | Sol tÄ±klama + sÃ¼rÃ¼kle ile etiketleme/silme dikdÃ¶rtgeni Ã§iz |
| **ğŸ” YakÄ±nlaÅŸtÄ±rma & KaydÄ±rma** | Fare tekerleÄŸi ile yakÄ±nlaÅŸtÄ±rma, saÄŸ tÄ±klama ile kaydÄ±rma |
| **ğŸ“ Kare Kilidi** | Ã‡izimi mÃ¼kemmel kareye sÄ±nÄ±rla |
| **â†©ï¸ Geri Al** | Tam geri alma geÃ§miÅŸi (Ctrl+Z) |
| **ğŸ¨ Bant SeÃ§imi** | Otomatik bant algÄ±lama; Ã§ok bantlÄ± dosyalarda seÃ§im dialogâ€™u (RGB, BGR, NIR Ã¶n ayarlarÄ±) |
| **ğŸ—‚ï¸ Katman Paneli** | GÃ¶rÃ¼nÃ¼rlÃ¼k, saydamlÄ±k ayarÄ±, sÃ¼rÃ¼kle-bÄ±rak ile sÄ±ralama |
| **â• Ek Katmanlar** | Ek GeoTIFF raster dosyalarÄ±nÄ± Ã¼st katman olarak yÃ¼kleyin |
| **ğŸ’¾ GeoTIFF Ã‡Ä±ktÄ±** | Kaynak CRS, dÃ¶nÃ¼ÅŸÃ¼m ve DEFLATE sÄ±kÄ±ÅŸtÄ±rma ile maske kaydet |
| **ğŸ–¼ï¸ SÃ¼rÃ¼kle & BÄ±rak** | `.tif` dosyalarÄ±nÄ± doÄŸrudan pencereye bÄ±rakÄ±n |
| **ğŸ¨ AÃ§Ä±k Tema** | Gradient araÃ§ Ã§ubuÄŸu ve stilize kontroller ile modern aÃ§Ä±k arayÃ¼z |
| **ğŸ”Œ Ã‡ift Backend** | PySide6 veya PyQt6 ile Ã§alÄ±ÅŸÄ±r |

### ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§

```bash
# ArgÃ¼mansÄ±z â€” dosya dialogâ€™u aÃ§Ä±lÄ±r
python ground_truth_kare_etiketleme_qt.py

# ArgÃ¼manlarla
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --output kesif_alani_ground_truth.tif

# Mevcut maskeyi dÃ¼zenlemeye devam
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --existing-mask kesif_alani_ground_truth.tif

# Tek bantlÄ± DEM, Ã¶nizleme kÃ¼Ã§Ã¼ltme ile
python ground_truth_kare_etiketleme_qt.py \
  --input karlik_dag_dsm.tif \
  --preview-max-size 4096
```

### âŒ¨ï¸ Klavye KÄ±sayollarÄ±

| KÄ±sayol | Eylem |
|---------|-------|
| `Ctrl+O` | GeoTIFF AÃ§ |
| `Ctrl+S` | Maskeyi Kaydet |
| `Ctrl+Shift+S` | FarklÄ± Kaydet |
| `Ctrl+Z` | Geri Al |
| `D` | Ã‡izim modu |
| `E` | Silme modu |
| `S` | Kare kilidi aÃ§/kapat |
| `F` | Pencereye sÄ±ÄŸdÄ±r |
| `W` | Fare tekerleÄŸi yÃ¶nÃ¼nÃ¼ ters Ã§evir |

### ğŸ“‹ Komut SatÄ±rÄ± Parametreleri

| Parametre | AÃ§Ä±klama | VarsayÄ±lan |
|-----------|----------|----------:|
| `--input`, `-i` | Girdi GeoTIFF yolu | _(dosya dialogâ€™u)_ |
| `--output`, `-o` | Ã‡Ä±ktÄ± maske yolu | `<girdi>_ground_truth.tif` |
| `--existing-mask` | DÃ¼zenlemeye devam edilecek mevcut maske | _(yok)_ |
| `--preview-max-size` | Maks Ã¶nizleme boyutu piksel (0 = tam Ã§Ã¶zÃ¼nÃ¼rlÃ¼k) | `0` |
| `--bands` | RGB gÃ¶rÃ¼ntÃ¼leme iÃ§in virgÃ¼lle ayrÄ±lmÄ±ÅŸ bant indeksleri | `1,2,3` |
| `--positive-value` | Pozitif sÄ±nÄ±f piksel deÄŸeri (1â€“255) | `1` |
| `--square-mode` | Kare kilidi aÃ§Ä±k baÅŸlat | `false` |

### ğŸµ Bant SeÃ§imi

Dosya aÃ§Ä±ldÄ±ÄŸÄ±nda araÃ§ bant sayÄ±sÄ±nÄ± otomatik algÄ±lar:

| Bant SayÄ±sÄ± | DavranÄ±ÅŸ |
|:----------:|----------|
| **1** | Otomatik gri tonlama â€” dialog yok |
| **2** | Bant 1,2 kullanÄ±lÄ±r â€” dialog yok |
| **3+** | Ã–n ayarlarla **Bant SeÃ§im Dialogâ€™u** gÃ¶sterilir |

**Mevcut Ã–n Ayarlar (3+ bant):**
- **RGB (1, 2, 3)** â€” standart gerÃ§ek renk
- **BGR (3, 2, 1)** â€” ters bant sÄ±rasÄ±
- **NIR (4, 3, 2)** â€” yakÄ±n kÄ±zÄ±lÃ¶tesi sahte renk (5+ bant)
- **Gri Tonlama (Bant 1)** â€” tek bant
- **Ã–zel** â€” R/G/B iÃ§in SpinBox ile herhangi bir bant seÃ§

### ğŸ—‚ï¸ Katman Paneli

Sol taraftaki panel gÃ¶rÃ¼ntÃ¼ katmanlarÄ±nÄ± yÃ¶netir:

- **â˜‘ï¸ GÃ¶rÃ¼nÃ¼rlÃ¼k** â€” her katman iÃ§in iÅŸaret kutusu
- **ğŸ”€ SÄ±ralama** â€” sÃ¼rÃ¼kle veya â¬†/â¬‡ butonlarÄ± (en Ã¼stteki Ã¶n planda)
- **ğŸšï¸ SaydamlÄ±k** â€” seÃ§ili katman iÃ§in sÃ¼rÃ¼gÃ¼ (%0â€“100)
- **â• Katman Ekle** â€” ek GeoTIFF dosyalarÄ±nÄ± gÃ¶rsel katman olarak yÃ¼kle
- **â– Katman Sil** â€” ekstra katmanlarÄ± kaldÄ±r (ana gÃ¶rÃ¼ntÃ¼ ve maske silinemez)

VarsayÄ±lan katmanlar:
1. ğŸ”´ **Maske** â€” etiketleme katmanÄ± (kÄ±rmÄ±zÄ±, yarÄ± saydam)
2. ğŸ–¼ï¸ **Ana GÃ¶rÃ¼ntÃ¼** â€” temel raster

### ğŸ”§ BaÄŸÄ±mlÄ±lÄ±klar

```bash
pip install rasterio opencv-python numpy
pip install PySide6   # veya: pip install PyQt6
```

---

## ğŸ® KullanÄ±m

### Temel KullanÄ±m

#### VarsayÄ±lan Ayarlarla Ã‡alÄ±ÅŸtÄ±rma

```bash
python archaeo_detect.py
```

Bu komut `config.yaml` dosyasÄ±ndaki ayarlarÄ± kullanÄ±r ve giriÅŸ olarak tanÄ±mlanan GeoTIFF dosyasÄ±nÄ± iÅŸler.

#### Komut SatÄ±rÄ± Parametreleriyle Ã‡alÄ±ÅŸtÄ±rma

```bash
# EÅŸik deÄŸerini deÄŸiÅŸtirme
python archaeo_detect.py --th 0.7

# Karo boyutunu ayarlama
python archaeo_detect.py --tile 512 --overlap 128

# AyrÄ±ntÄ±lÄ± modu etkinleÅŸtirme (detaylÄ± log)
python archaeo_detect.py -v

# FarklÄ± bir giriÅŸ dosyasÄ± kullanma
python archaeo_detect.py --input yeni_alan.tif

# Birden fazla parametre
python archaeo_detect.py --th 0.7 --tile 1024 --enable-fusion -v
```

### YaygÄ±n KullanÄ±m Ã–rnekleri

#### ğŸ”° Ã–rnek 1: Ä°lk Kez KullanÄ±m (SÄ±fÄ±r AtÄ±ÅŸ)

EÄŸitilmiÅŸ modeller olmadan, sadece ImageNet aÄŸÄ±rlÄ±klarÄ±nÄ± kullanarak:

```bash
python archaeo_detect.py \
  --encoder resnet34 \
  --zero-shot-imagenet \
  --enable-classic \
  --enable-fusion \
  -v
```

#### ğŸ¯ Ã–rnek 2: Sadece Klasik YÃ¶ntem (HÄ±zlÄ±)

GPU yoksa veya hÄ±zlÄ± test iÃ§in:

```bash
python archaeo_detect.py \
  --no-enable-deep-learning \
  --enable-classic \
  --classic-modes combo \
  --cache-derivatives
```

#### ğŸš€ Ã–rnek 3: Topluluk (Ã‡oklu KodlayÄ±cÄ±)

Birden fazla kodlayÄ±cÄ± ile en yÃ¼ksek doÄŸruluk iÃ§in:

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

#### ğŸ¨ Ã–rnek 4: Ã–zel EÄŸitilmiÅŸ Modelle

Kendi eÄŸitilmiÅŸ modelinizle:

```bash
python archaeo_detect.py \
  --encoder resnet50 \
  --weights models/benim_egitilmis_modelim.pth \
  --th 0.65 \
  --enable-classic \
  --enable-fusion \
  --alpha 0.7
```

#### ğŸ“Š Ã–rnek 5: GeniÅŸ Alan Analizi (Optimize EdilmiÅŸ)

GeniÅŸ bir alan iÃ§in optimize edilmiÅŸ ayarlar:

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

### Komut SatÄ±rÄ± Parametreleri (Tam Liste)

```bash
python archaeo_detect.py --help
```

**Ã–nemli Parametreler:**

| Parametre | AÃ§Ä±klama | Ã–rnek |
|-----------|----------|-------|
| `--input` | GiriÅŸ GeoTIFF dosyasÄ± | `--input alan.tif` |
| `--th` | DL eÅŸiÄŸi (0-1) | `--th 0.7` |
| `--tile` | Karo boyutu (piksel) | `--tile 1024` |
| `--overlap` | Ã–rtÃ¼ÅŸme miktarÄ± | `--overlap 128` |
| `--encoder` | Tek kodlayÄ±cÄ± seÃ§imi | `--encoder resnet34` |
| `--encoders` | Ã‡oklu kodlayÄ±cÄ± modu | `--encoders all` |
| `--alpha` | FÃ¼zyon aÄŸÄ±rlÄ±ÄŸÄ± | `--alpha 0.6` |
| `--enable-fusion` | FÃ¼zyonu etkinleÅŸtir | (bayrak) |
| `--cache-derivatives` | Ã–nbellek kullan | (bayrak) |
| `-v` veya `--verbose` | DetaylÄ± log | (bayrak) |

---

## âš™ï¸ YapÄ±landÄ±rma

### config.yaml DosyasÄ±

Sistem davranÄ±ÅŸÄ± `config.yaml` dosyasÄ± tarafÄ±ndan kontrol edilir. Bu dosya detaylÄ± aÃ§Ä±klamalarla **zengin bir ÅŸekilde belgelenmiÅŸtir** (satÄ±r iÃ§i TÃ¼rkÃ§e yorumlar dahil).

**Yol Ã§Ã¶zÃ¼mlemesi:** YAML iÃ§indeki gÃ¶reli yollar, **`config.yaml` dosyasÄ±nÄ±n bulunduÄŸu dizine** gÃ¶re Ã§Ã¶zÃ¼lÃ¼r; Ã§alÄ±ÅŸma dizininize gÃ¶re deÄŸil. Komut satÄ±rÄ±ndan verdiÄŸiniz yollar ise **o anki Ã§alÄ±ÅŸma dizinine** gÃ¶re Ã§Ã¶zÃ¼lÃ¼r.

#### Ana BÃ¶lÃ¼mler:

1. **GiriÅŸ/Ã‡Ä±kÄ±ÅŸ**: Dosya yollarÄ± ve bant seÃ§imi
2. **YÃ¶ntem SeÃ§imi**: `enable_deep_learning`, `enable_classic`, `enable_yolo`, `enable_fusion`
3. **DL gÃ¶revi**: `dl_task` â€” `segmentation` (piksel) veya `tile_classification` (karo skoru â†’ bindirme ile risk haritasÄ±)
4. **EÄŸitilmiÅŸ-tekil mod**: `trained_model_only` â€” `true` iken tek checkpoint + metadata (`weights`, `training_metadata`); `tile` / `overlap` / `bands` metadataâ€™dan kilitlenir
5. **Derin Ã¶ÄŸrenme**: Mimari, encoder, aÄŸÄ±rlÄ±klar, `zero_shot_imagenet`, dikkat / bant Ã¶nem raporu
6. **Klasik YÃ¶ntemler**: RVT, Hessian, Morfoloji parametreleri
7. **GeliÅŸmiÅŸ topografik analiz (legacy / varsayÄ±lan presetâ€™te kapalÄ±)**: `enable_curvature`, `enable_tpi`, `tpi_radii` â€” `config.yaml` ve `archaeo_detect.py` iÃ§inde deneysel kullanÄ±m iÃ§in durur; **kayÄ±tlÄ± 5 kanallÄ± DL ÅŸemasÄ±** eÄŸri/TPIâ€™yi model tensÃ¶rÃ¼ne eklemez (`config.yaml` Ã¼st yorumlarÄ±na bakÄ±n).
8. **FÃ¼zyon**: Hibrit kombinasyon (`alpha`, â€¦) â€” hem DL hem klasik aÃ§Ä±k olmalÄ±dÄ±r
9. **YOLO11** (isteÄŸe baÄŸlÄ±): YalnÄ±zca RGB; genelde tile sÄ±nÄ±flandÄ±rma Ã¶n ayarÄ±nda kapalÄ±dÄ±r
10. **Karo Ä°ÅŸleme**: Bellek ve performans; `tile` / `overlap` belge ile metadata kilitlenmesi
11. **Normalizasyon**: Veri Ã¶n iÅŸleme
12. **Maskeleme**: YÃ¼ksek yapÄ±lar (`mask_talls`, `rgb_only`)
13. **VektÃ¶rleÅŸtirme**: CBS Ã§Ä±ktÄ±sÄ± (`vectorize`, `min_area`, `export_candidate_excel`, â€¦)
14. **Performans**: Cihaz, `half`, `seed`, `verbose`
15. **Ã–nbellek**: `cache_derivatives`, `cache_derivatives_mode` (`auto` / `npz` / `raster`), raster Ã¶nbellek ayarlarÄ±

#### HÄ±zlÄ± YapÄ±landÄ±rma SenaryolarÄ±:

**Senaryo 1: Sadece Derin Ã–ÄŸrenme**
```yaml
enable_deep_learning: true
enable_classic: false
enable_fusion: false
encoder: "resnet34"
zero_shot_imagenet: true
```

**Senaryo 2: Sadece Klasik YÃ¶ntem**
```yaml
enable_deep_learning: false
enable_classic: true
enable_fusion: false
classic_modes: "combo"
cache_derivatives: true
```

**Senaryo 3: Hibrit (En Ä°yi SonuÃ§lar)**
```yaml
enable_deep_learning: true
enable_classic: true
enable_fusion: true
alpha: 0.5
encoders: "all"
cache_derivatives: true
```

### Veri HazÄ±rlama

#### GiriÅŸ DosyasÄ± Gereksinimleri:

âœ… **GeoTIFF formatÄ±** (.tif veya .tiff)  
âœ… **Ã‡ok bantlÄ±** (en az 3 bant: RGB)  
âœ… **AynÄ± grid** (tÃ¼m bantlar aynÄ± Ã§Ã¶zÃ¼nÃ¼rlÃ¼k ve kapsam)  
âœ… **CoÄŸrafi referans** (CRS/EPSG kodu)

#### Ã–nerilen Bant YapÄ±sÄ±:

| Bant # | Ä°Ã§erik | AÃ§Ä±klama |
|--------|--------|----------|
| 1 | KÄ±rmÄ±zÄ± | RGB'nin R bileÅŸeni |
| 2 | YeÅŸil | RGB'nin G bileÅŸeni |
| 3 | Mavi | RGB'nin B bileÅŸeni |
| 4 | DSM | SayÄ±sal YÃ¼zey Modeli (yÃ¼kseklik) |
| 5 | DTM | SayÄ±sal Arazi Modeli (zemin yÃ¼ksekliÄŸi) |

#### Veri OluÅŸturma Ã–rneÄŸi (GDAL):

```bash
# AyrÄ± RGB ve yÃ¼kseklik dosyalarÄ±nÄ± birleÅŸtirme
gdal_merge.py -separate -o birlesik.tif \
  kirmizi.tif yesil.tif mavi.tif dsm.tif dtm.tif

# Yeniden Ã¶rnekleme (farklÄ± Ã§Ã¶zÃ¼nÃ¼rlÃ¼kleri eÅŸitleme)
gdalwarp -tr 1.0 1.0 -r bilinear giris.tif cikis.tif

# Koordinat sistemi atama
gdal_edit.py -a_srs EPSG:32635 cikis.tif
```

---

## ğŸ“‚ Ã‡Ä±ktÄ± DosyalarÄ±

Sistem Ã§alÄ±ÅŸtÄ±ÄŸÄ±nda aÅŸaÄŸÄ±daki dosyalar oluÅŸturulur:

TÃ¼m Ã§Ä±ktÄ±lar aÅŸaÄŸÄ±daki kÃ¶ke yazÄ±lÄ±r:

```
workspace/ciktilar/<oturum_klasoru>/<cikti_adi>*
```

`<oturum_klasoru>` kÄ±sa bir format kullanÄ±r:
`<zaman>_<girdi>_<yontemler>_t<tile>o<overlap>_m-<model>`
(Ã¶rnek model belirteÃ§leri: `m-<checkpoint>`, `m-zs`, `m-<encoder>`).

Her oturum klasÃ¶rÃ¼nde ayrÄ±ca ÅŸunlar bulunur:

```
run_params.txt
```

Bu dosya, etkin parametrelerin tamamÄ±nÄ± iÃ§erir (nihai config deÄŸerleri, parse edilen bantlar, CLI argÃ¼manlarÄ± ve cihaz).

Etkinse (`save_band_importance: true`), DL koÅŸularÄ± ayrÄ±ca `*_band_importance.txt` ve `*_band_importance.json` dosyalarÄ±nÄ± Ã¼retir.

### ğŸ“Š Raster Ã‡Ä±ktÄ±lar (GeoTIFF)

#### 1ï¸âƒ£ Derin Ã–ÄŸrenme Ã‡Ä±ktÄ±larÄ±

**Tek KodlayÄ±cÄ±:**
```
kesif_alani_prob.tif     â†’ OlasÄ±lÄ±k haritasÄ± (sÃ¼rekli deÄŸerler 0.0-1.0)
kesif_alani_mask.tif     â†’ Ä°kili maske (0: arkeolojik deÄŸil, 1: arkeolojik alan)
```

**Ã‡oklu KodlayÄ±cÄ±:**
```
kesif_alani_resnet34_prob.tif
kesif_alani_resnet34_mask.tif
kesif_alani_resnet50_prob.tif
kesif_alani_resnet50_mask.tif
kesif_alani_efficientnet-b3_prob.tif
kesif_alani_efficientnet-b3_mask.tif
```

#### 2ï¸âƒ£ Klasik YÃ¶ntem Ã‡Ä±ktÄ±larÄ±

```
kesif_alani_classic_prob.tif     â†’ BirleÅŸik klasik olasÄ±lÄ±k
kesif_alani_classic_mask.tif     â†’ Klasik ikili maske
```

**Ara Dosyalar (classic_save_intermediate: true):**
```
kesif_alani_classic_rvtlog_prob.tif    â†’ Sadece RVT yÃ¶ntemi
kesif_alani_classic_hessian_prob.tif   â†’ Sadece Hessian yÃ¶ntemi
kesif_alani_classic_morph_prob.tif     â†’ Sadece Morfoloji yÃ¶ntemi
```

#### 3ï¸âƒ£ FÃ¼zyon Ã‡Ä±ktÄ±larÄ±

```
kesif_alani_fused_resnet34_prob.tif
kesif_alani_fused_resnet34_mask.tif
```

### ğŸ“ VektÃ¶r Ã‡Ä±ktÄ±lar (GeoPackage)

```
kesif_alani_mask.gpkg                â†’ DL vektÃ¶r Ã§okgenleri
kesif_alani_classic_mask.gpkg        â†’ Klasik vektÃ¶r Ã§okgenleri
kesif_alani_fused_resnet34_mask.gpkg â†’ FÃ¼zyon vektÃ¶r Ã§okgenleri
```

`config.yaml` iÃ§inde `export_candidate_excel: true` ise, vektÃ¶r Ã§Ä±ktÄ±larÄ±na eÅŸlik eden aday merkezleri / GPS tarzÄ± tablolar `*_gps.xlsx` dosyalarÄ± olarak Ã¼retilir.

**GeoPackage Ã–zellikleri:**
- Ã‡okgen geometrisi
- Alan bilgisi (mÂ² cinsinden)
- CRS bilgisi korunur
- QGIS/ArcGIS'te doÄŸrudan aÃ§Ä±labilir

### ğŸ’¾ Ã–nbellek DosyalarÄ±

**Ã–nbellek Dizin YapÄ±sÄ±:**
```
workspace/cache/
â”œâ”€â”€ kesif_alani.a1b2c3d4e5f6.derivatives.npz    â†’ RVT tÃ¼revleri Ã¶nbelleÄŸi
â””â”€â”€ karlik_vadi.f6e5d4c3b2a1.derivatives.npz   â†’ RVT tÃ¼revleri Ã¶nbelleÄŸi
```

**Ã–nbellek Sistemi:**
- RVT hesaplamalarÄ± `.npz` formatÄ±nda Ã¶nbelleÄŸe alÄ±nÄ±r
- Ã–nbellek dosyalarÄ± `workspace/cache/` dizininde saklanÄ±r (config.yaml'daki `cache_dir` ile yapÄ±landÄ±rÄ±labilir)
- Ã–nbellek doÄŸrulamasÄ± dosya adÄ± ve deÄŸiÅŸiklik zamanÄ±nÄ± kontrol eder
- **Ã–nemli:** Proje klasÃ¶rÃ¼ taÅŸÄ±nsa bile Ã¶nbellek dosyalarÄ± yeniden kullanÄ±labilir (dosya adÄ± tabanlÄ± doÄŸrulama)
- Sonraki Ã§alÄ±ÅŸtÄ±rmalarda 10-100x hÄ±zlanma saÄŸlar
- Ã–nbellek dosyalarÄ± tipik olarak 10-50 MB'dÄ±r, ancak yÃ¼ksek Ã§Ã¶zÃ¼nÃ¼rlÃ¼klÃ¼ veriler iÃ§in daha bÃ¼yÃ¼k olabilir

**Ã–nbellek YapÄ±landÄ±rmasÄ±:**
```yaml
cache_derivatives: true      # Ã–nbelleÄŸi etkinleÅŸtir
cache_dir: "workspace/cache/"          # Ã–nbellek dizini (proje kÃ¶kÃ¼ne gÃ¶re)
recalculate_cache: false     # Ã–nbellek varsa yeniden hesaplama
```

### ğŸ“‹ Dosya AdlandÄ±rma MantÄ±ÄŸÄ±

Ã‡Ä±ktÄ± dosyalarÄ± aÅŸaÄŸÄ±daki formatta otomatik olarak adlandÄ±rÄ±lÄ±r:

```
<Ã¶nek>_[yÃ¶ntem]_[kodlayÄ±cÄ±]_[parametreler]_[tip].ext
```

Ã–rnek:
```
kesif_alani_fused_resnet34_th0.6_tile1024_alpha0.5_prob.tif
```

**Parametreler:**
- `th`: EÅŸik deÄŸeri
- `tile`: Karo boyutu
- `alpha`: FÃ¼zyon oranÄ±
- `minarea`: Minimum alan
- Ve diÄŸerleri...

---

## ğŸ”¬ NasÄ±l Ã‡alÄ±ÅŸÄ±r

### Ä°ÅŸ AkÄ±ÅŸÄ± Genel BakÄ±ÅŸ

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  GeoTIFF GiriÅŸi     â”‚
â”‚ (RGB, DSM, DTM)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Veri Ã–n Ä°ÅŸleme     â”‚
â”‚  - Bant okuma       â”‚
â”‚  - Normalizasyon    â”‚
â”‚  - Maskeleme        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
     â”Œâ”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”
     â–¼           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Derin   â”‚ â”‚ Klasik   â”‚
â”‚ Ã–ÄŸrenme â”‚ â”‚ YÃ¶ntemlerâ”‚
â”‚ (U-Net) â”‚ â”‚ (RVT)    â”‚
â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
     â”‚           â”‚
     â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
           â–¼
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚    FÃ¼zyon     â”‚
   â”‚  (BirleÅŸtir)  â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚  EÅŸikleme     â”‚
   â”‚  (Olas â†’ Mask)â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚ VektÃ¶rleÅŸtirmeâ”‚
   â”‚  (GeoPackage) â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 1ï¸âƒ£ Derin Ã–ÄŸrenme YÃ¶ntemi

**AdÄ±mlar:**

1. **5 kanallÄ± DL tensÃ¶rÃ¼nÃ¼ oluÅŸturma**
   - GeoTIFFâ€™ten **RGB** ve **DSM/DTM** okunur (`config.yaml`â€™daki bant seÃ§imine gÃ¶re)
   - DoldurulmuÅŸ DTM Ã¼zerinde RVT ile **SVF** ve **SLRM** hesaplanÄ±r (`archaeo_detect.py` iÃ§indeki `compute_derivatives_with_rvt` vb.)
   - `stack_channels(rgb, svf, slrm)` ile `(5, H, W)` tensÃ¶rÃ¼ Ã¼retilir; kanal sÄ±rasÄ± `MODEL_CHANNEL_NAMES`

   AÃ§Ä±klÄ±k, eÄŸim vb. diÄŸer RVT Ã§Ä±ktÄ±larÄ± **klasik** veya deneysel yollarda kullanÄ±labilir; bu DL yÄ±ÄŸÄ±nÄ±nda ayrÄ± dÃ¼zlem olarak yer almazlar.

2. **Normalizasyon**
   - Global veya yerel yÃ¼zdelik tabanlÄ±
   - %2-%98 aralÄ±ÄŸÄ±na Ã¶lÃ§ekleme

3. **Karo TabanlÄ± Ä°ÅŸleme**
   - BÃ¼yÃ¼k gÃ¶rÃ¼ntÃ¼ kÃ¼Ã§Ã¼k karolara bÃ¶lÃ¼nÃ¼r
   - Her karo U-Net'e beslenir
   - OlasÄ±lÄ±k haritasÄ± oluÅŸturulur

4. **YumuÅŸatma (Feathering)**
   - Karolar arasÄ±ndaki geÃ§iÅŸler yumuÅŸatÄ±lÄ±r
   - Sorunsuz mozaik oluÅŸturulur

5. **EÅŸikleme**
   - OlasÄ±lÄ±k > eÅŸik â†’ Maske = 1
   - OlasÄ±lÄ±k â‰¤ eÅŸik â†’ Maske = 0

### 2ï¸âƒ£ Klasik GÃ¶rÃ¼ntÃ¼ Ä°ÅŸleme

**ÃœÃ§ Alt YÃ¶ntem:**

**A) RVT (Kabartma GÃ¶rselleÅŸtirme)**
- SVF, AÃ§Ä±klÄ±k hesaplamalarÄ±
- Kabartma gÃ¶rselleÅŸtirme
- TÃ¼mÃ¼lÃ¼s ve hÃ¶yÃ¼kler iÃ§in ideal

**B) Hessian Matrisi**
- Ä°kinci tÃ¼rev analizi
- SÄ±rt ve vadi tespiti
- Duvarlar ve hendekler iÃ§in etkili

**C) Morfolojik OperatÃ¶rler**
- AÃ§ma, kapama
- Ãœst-ÅŸapka dÃ¶nÃ¼ÅŸÃ¼mleri
- Yerel doku Ã¶zellikleri

**Kombinasyon:**
- Her yÃ¶ntem 0-1 puan Ã¼retir
- Puanlar ortalamasÄ± alÄ±nÄ±r (combo modu)
- Otsu veya manuel eÅŸikleme uygulanÄ±r

### 3ï¸âƒ£ FÃ¼zyon (Hibrit Kombinasyon)

**FormÃ¼l:**
```
P_fused = Î± Ã— P_derin_Ã¶ÄŸrenme + (1 - Î±) Ã— P_klasik
```

**Avantajlar:**
- Derin Ã¶ÄŸrenme: KarmaÅŸÄ±k desenler
- Klasik: GÃ¼venilir yÃ¼kseklik Ã¶zellikleri
- FÃ¼zyon: Her ikisinin gÃ¼Ã§lÃ¼ yÃ¶nleri

**Ã–rnek:**
- Î± = 0.5: EÅŸit aÄŸÄ±rlÄ±k
- Î± = 0.7: DL'ye Ã¶ncelik
- Î± = 0.3: KlasiÄŸe Ã¶ncelik

---

## ğŸ’¡ KullanÄ±m SenaryolarÄ±

### ğŸ“ Senaryo 1: Yeni Alan KeÅŸfi

**Durum:** KeÅŸfedilmemiÅŸ bir alanÄ±n ilk taramasÄ±

**Ã–nerilen Ayarlar:**
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

**Neden bu ayarlar?**
- Ã‡oklu kodlayÄ±cÄ±: Maksimum tespit hassasiyeti
- DÃ¼ÅŸÃ¼k eÅŸik: TÃ¼m adaylarÄ± yakala
- DÃ¼ÅŸÃ¼k min_area: KÃ¼Ã§Ã¼k yapÄ±larÄ± kaÃ§Ä±rma
- Ã–nbellek: Tekrarlanan analiz iÃ§in hÄ±zlanma

### ğŸ¯ Senaryo 2: Bilinen AlanÄ±n DetaylÄ± Analizi

**Durum:** Daha Ã¶nce tespit edilen bir alanÄ±n detaylÄ± incelemesi

**Ã–nerilen Ayarlar:**
```bash
python archaeo_detect.py \
  --encoder efficientnet-b3 \
  --weights models/benim_ayarli_modelim.pth \
  --th 0.65 \
  --enable-classic \
  --alpha 0.6 \
  --min-area 80 \
  --simplify 2.0 \
  -v
```

**Neden bu ayarlar?**
- Ã–zel model: BÃ¶lgeye Ã¶zgÃ¼ eÄŸitilmiÅŸ model
- YÃ¼ksek eÅŸik: Sadece gÃ¼venilir tespitler
- Simplify: Temiz Ã§okgenler

### âš¡ Senaryo 3: HÄ±zlÄ± Ã–n DeÄŸerlendirme

**Durum:** HÄ±zlÄ±ca fikir edinmek iÃ§in

**Ã–nerilen Ayarlar:**
```bash
python archaeo_detect.py \
  --no-enable-deep-learning \
  --enable-classic \
  --classic-modes rvtlog \
  --tile 512 \
  --no-vectorize \
  --cache-derivatives
```

**Neden bu ayarlar?**
- Sadece klasik: En hÄ±zlÄ± yÃ¶ntem
- KÃ¼Ã§Ã¼k karolar: Daha az bellek
- VektÃ¶r yok: Zaman tasarrufu

### ğŸ”¬ Senaryo 4: AraÅŸtÄ±rma ve KarÅŸÄ±laÅŸtÄ±rma

**Durum:** FarklÄ± yÃ¶ntemlerin karÅŸÄ±laÅŸtÄ±rmalÄ± analizi

**Ã–nerilen Ayarlar:**
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

**Neden bu ayarlar?**
- TÃ¼m yÃ¶ntemler aktif
- Ara dosyalar: Her yÃ¶ntemin katkÄ±sÄ±nÄ± gÃ¶rme
- TÃ¼m fÃ¼zyon: Her kombinasyonu deneme

---

## ğŸ¨ GÃ¶rselleÅŸtirme

### QGIS'te GÃ¶rÃ¼ntÃ¼leme

#### 1ï¸âƒ£ OlasÄ±lÄ±k HaritalarÄ±nÄ± YÃ¼kleme

```
Katman â†’ Katman Ekle â†’ Raster Katman Ekle
```

**Ã–nerilen Renk ÅemasÄ±:**
- 0.0-0.3: Mavi (DÃ¼ÅŸÃ¼k olasÄ±lÄ±k)
- 0.3-0.5: SarÄ± (Orta olasÄ±lÄ±k)
- 0.5-0.7: Turuncu (YÃ¼ksek olasÄ±lÄ±k)
- 0.7-1.0: KÄ±rmÄ±zÄ± (Ã‡ok yÃ¼ksek olasÄ±lÄ±k)

#### 2ï¸âƒ£ VektÃ¶r Ã‡okgenleri GÃ¶rÃ¼ntÃ¼leme

```
Katman â†’ Katman Ekle â†’ VektÃ¶r Katman Ekle â†’ GeoPackage SeÃ§
```

**Stil Ã–nerileri:**
- Dolgu: YarÄ± saydam kÄ±rmÄ±zÄ± (opaklÄ±k: %50)
- Ã‡izgi: KalÄ±n kÄ±rmÄ±zÄ± (2 piksel)
- Etiket: Alan deÄŸeri (mÂ²)

#### 3ï¸âƒ£ Temel Haritayla Bindirme

```python
# QGIS Python Konsolu
from qgis.core import QgsRasterLayer

# Ortofoto ekle
ortho = QgsRasterLayer('kesif_alani.tif', 'Ortofoto')
QgsProject.instance().addMapLayer(ortho)

# Maske ekle (yarÄ± saydam)
mask = QgsRasterLayer('kesif_alani_mask.tif', 'Tespit')
QgsProject.instance().addMapLayer(mask)
mask.renderer().setOpacity(0.6)
```

### Python GÃ¶rselleÅŸtirme

```python
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# OlasÄ±lÄ±k haritasÄ±nÄ± oku
with rasterio.open('kesif_alani_prob.tif') as src:
    prob = src.read(1)

# Ã–zel renk paleti
colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
cmap = LinearSegmentedColormap.from_list('archaeo', colors)

# GÃ¶rselleÅŸtir
plt.figure(figsize=(12, 10))
plt.imshow(prob, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(label='Arkeolojik Alan OlasÄ±lÄ±ÄŸÄ±')
plt.title('Arkeolojik Alan Tespit SonuÃ§larÄ±')
plt.xlabel('X (piksel)')
plt.ylabel('Y (piksel)')
plt.tight_layout()
plt.savefig('sonuc_gorsellestirme.png', dpi=300)
plt.show()
```

### Web TabanlÄ± GÃ¶rselleÅŸtirme

```python
import folium
import geopandas as gpd

# VektÃ¶r oku
gdf = gpd.read_file('kesif_alani_mask.gpkg')

# Harita oluÅŸtur
m = folium.Map(
    location=[gdf.geometry.centroid.y.mean(), 
              gdf.geometry.centroid.x.mean()],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Ã‡okgenleri ekle
for idx, row in gdf.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 2,
            'fillOpacity': 0.5
        },
        tooltip=f"Alan: {row.get('area', 0):.1f} mÂ²"
    ).add_to(m)

# Kaydet
m.save('interaktif_harita.html')
print("Harita oluÅŸturuldu: interaktif_harita.html")
```

---

## âš¡ Performans Optimizasyonu

### GPU KullanÄ±mÄ±

#### CUDA KontrolÃ¼
```python
import torch
print(f"CUDA KullanÄ±labilir: {torch.cuda.is_available()}")
print(f"CUDA Versiyonu: {torch.version.cuda}")
print(f"GPU SayÄ±sÄ±: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU AdÄ±: {torch.cuda.get_device_name(0)}")
```

#### GPU HÄ±zlandÄ±rma
```bash
# Karma hassasiyet (FP16) ile 2x hÄ±zlanma
python archaeo_detect.py --half

# GPU'yu bÃ¼yÃ¼k karolarla doldur
python archaeo_detect.py --tile 2048 --overlap 512
```

### Bellek Optimizasyonu

#### DÃ¼ÅŸÃ¼k Bellek Durumu
```bash
python archaeo_detect.py \
  --tile 512 \
  --overlap 64 \
  --no-enable-deep-learning \
  --enable-classic
```

#### YÃ¼ksek Bellek Durumu
```bash
python archaeo_detect.py \
  --tile 4096 \
  --overlap 1024 \
  --half \
  --encoders all
```

### Ã–nbellek Stratejisi

```bash
# Ä°lk Ã§alÄ±ÅŸtÄ±rma: Ã–nbellek oluÅŸtur
python archaeo_detect.py --cache-derivatives

# Sonraki Ã§alÄ±ÅŸtÄ±rmalar: 10-100x daha hÄ±zlÄ±!
# Ã–nbellek otomatik olarak kullanÄ±lÄ±r

# Parametreler deÄŸiÅŸtiÄŸinde Ã¶nbelleÄŸi yeniden hesapla
python archaeo_detect.py --recalculate-cache
```

**Ã–nbellek FaydalarÄ±:**
- RVT tÃ¼revleri bir kez hesaplanÄ±r ve Ã¶nbelleÄŸe alÄ±nÄ±r
- Ã–nbellek dosyalarÄ± `workspace/cache/` dizininde saklanÄ±r
- Ã–nbellek doÄŸrulamasÄ± esnektir: proje klasÃ¶rÃ¼ taÅŸÄ±nsa bile Ã§alÄ±ÅŸÄ±r
- Dosya adÄ± ve deÄŸiÅŸiklik zamanÄ± doÄŸrulama iÃ§in kontrol edilir
- Tekrarlanan Ã§alÄ±ÅŸtÄ±rmalarda Ã¶nemli zaman tasarrufu

### Paralel Ä°ÅŸleme

Birden fazla alan iÃ§in paralel Ã§alÄ±ÅŸtÄ±rma:

```bash
# Bash betiÄŸi
for file in alan1.tif alan2.tif alan3.tif; do
  python archaeo_detect.py --input $file &
done
wait
```

### Performans KarÅŸÄ±laÅŸtÄ±rmasÄ±

| YapÄ±landÄ±rma | Ä°ÅŸleme SÃ¼resi | Bellek KullanÄ±mÄ± | Kalite |
|--------------|---------------|------------------|--------|
| **Minimum** (CPU, 512 karo) | ~30 dk | 4 GB | DÃ¼ÅŸÃ¼k |
| **Dengeli** (GPU, 1024 karo) | ~5 dk | 8 GB | Orta |
| **Maksimum** (GPU, 2048 karo, topluluk) | ~15 dk | 16 GB | YÃ¼ksek |

*10 kmÂ² alan iÃ§in tahmini sÃ¼reler (1m Ã§Ã¶zÃ¼nÃ¼rlÃ¼k)*

---

## ğŸ› Sorun Giderme

### YaygÄ±n Hatalar ve Ã‡Ã¶zÃ¼mler

#### âŒ Hata 1: CUDA Bellek YetersizliÄŸi

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Ã‡Ã¶zÃ¼mler:**
```bash
# Ã‡Ã¶zÃ¼m 1: Karo boyutunu kÃ¼Ã§Ã¼lt
python archaeo_detect.py --tile 512

# Ã‡Ã¶zÃ¼m 2: Karma hassasiyet kullan
python archaeo_detect.py --half

# Ã‡Ã¶zÃ¼m 3: CPU kullan
python archaeo_detect.py --device cpu
```

#### âŒ Hata 2: RVT Ä°Ã§e Aktarma HatasÄ±

```
ModuleNotFoundError: No module named 'rvt'
```

**Ã‡Ã¶zÃ¼m:**
```bash
# Python 3.10
pip install rvt-py

# Python 3.11+
pip install rvt

# Veya conda ile
conda install -c conda-forge rvt
```

#### âŒ Hata 3: BoÅŸ Ã‡Ä±ktÄ±

```
Warning: No detections found
```

**Ã‡Ã¶zÃ¼mler:**
1. EÅŸik deÄŸerini dÃ¼ÅŸÃ¼r:
   ```bash
   python archaeo_detect.py --th 0.3 --classic-th 0.3
   ```

2. Minimum alanÄ± dÃ¼ÅŸÃ¼r:
   ```bash
   python archaeo_detect.py --min-area 20
   ```

3. AyrÄ±ntÄ±lÄ± modda kontrol et:
   ```bash
   python archaeo_detect.py -v
   ```

#### âŒ Hata 4: Klasik YÃ¶ntem Ã‡alÄ±ÅŸmÄ±yor

```
Error: DTM band not found
```

**Ã‡Ã¶zÃ¼m:**
`config.yaml`'da bantlarÄ± kontrol edin:
```yaml
bands: "1,2,3,4,5"  # Bant 5 DTM olmalÄ±
# DTM zorunlu (0 geÃ§ersizdir)
# DTM eksikse Ã¶nce geÃ§erli bir DTM bandÄ± Ã¼retin/saÄŸlayÄ±n.
```

#### âŒ Hata 5: Karo SÄ±nÄ±rlarÄ±nda Ã‡izgiler

**Ã‡Ã¶zÃ¼m:**
```bash
# Ã–rtÃ¼ÅŸmeyi artÄ±r ve yumuÅŸatmayÄ± etkinleÅŸtir
python archaeo_detect.py --overlap 512 --feather
```

#### âŒ Hata 6: Ã–nbellek KullanÄ±lmÄ±yor

**Belirtiler:** Ã–nbellek dosyalarÄ± varken bile sistem RVT tÃ¼revlerini yeniden hesaplÄ±yor

**Ã‡Ã¶zÃ¼mler:**
1. `config.yaml`'da Ã¶nbellek dizini yolunu kontrol edin:
   ```yaml
   cache_dir: "workspace/cache/"  # Ã–nbellek dizininizle eÅŸleÅŸmeli
   ```

2. Ã–nbellek dosya adlandÄ±rmasÄ±nÄ± doÄŸrulayÄ±n:
   - NPZ Ã¶nbellek (kÃ¼Ã§Ã¼k/orta rasterlar iÃ§in varsayÄ±lan): `<giriÅŸ_adÄ±>.<cache_hash>.derivatives.npz`
   - Raster Ã¶nbellek (blok tabanlÄ±; Ã§ok bÃ¼yÃ¼k rasterlar veya `cache_derivatives_mode: "raster"` ile otomatik kullanÄ±lÄ±r):
     - `<giriÅŸ_adÄ±>.<cache_hash>.derivatives_raster.tif`
     - `<giriÅŸ_adÄ±>.<cache_hash>.derivatives_raster.json`
   - `kesif_alani.tif` giriÅŸi iÃ§in Ã¶rnek:
     - `kesif_alani.a1b2c3d4e5f6.derivatives.npz`
     - `kesif_alani.a1b2c3d4e5f6.derivatives_raster.tif`

3. Ã–nbellek doÄŸrulamasÄ±nÄ± kontrol edin:
   - Ã–nbellek doÄŸrulamasÄ± dosya adÄ± ve deÄŸiÅŸiklik zamanÄ±nÄ± kontrol eder
   - GiriÅŸ dosyasÄ± taÅŸÄ±nmÄ±ÅŸsa, Ã¶nbellek yine de Ã§alÄ±ÅŸmalÄ±dÄ±r (dosya adÄ± tabanlÄ± doÄŸrulama)
   - GiriÅŸ dosyasÄ± deÄŸiÅŸtirilmiÅŸse, Ã¶nbellek yeniden hesaplanÄ±r

4. Ã–nbellek durumunu gÃ¶rmek iÃ§in ayrÄ±ntÄ±lÄ± modu etkinleÅŸtirin:
   ```bash
   python archaeo_detect.py --cache-derivatives -v
   ```

#### âŒ Hata 7: EÄŸitim BetiÄŸi Ä°Ã§e Aktarma HatalarÄ±

**Belirtiler:**
```
HATA: segmentation-models-pytorch kurulu deÄŸil!
HATA: archaeo_detect.py'den attention modÃ¼lleri import edilemedi.
```

**Ã‡Ã¶zÃ¼mler:**
1. **Eksik paketleri yÃ¼kleyin**:
   ```bash
   pip install segmentation-models-pytorch
   ```

2. **Python yolunu kontrol edin**: `archaeo_detect.py`'nin aynÄ± dizinde veya Python yolunda olduÄŸundan emin olun

3. **Kurulumu doÄŸrulayÄ±n**: `python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"` Ã§alÄ±ÅŸtÄ±rÄ±n

#### âŒ Hata 8: EÄŸitim Verisi Format UyumsuzluÄŸu

**Belirtiler:**
```
ValueError: Expected 5 channels but got X
```

**Ã‡Ã¶zÃ¼mler:**
1. **EÄŸitim verisini yeniden oluÅŸturun**: `egitim_verisi_olusturma.py`'yi doÄŸru parametrelerle kullanÄ±n
2. **metadata.json'u kontrol edin**: `num_channels`'Ä±n gerÃ§ek veriyle eÅŸleÅŸtiÄŸini doÄŸrulayÄ±n
3. **Dosya formatÄ±nÄ± doÄŸrulayÄ±n**: `.npz` dosyalarÄ±nÄ±n gÃ¼ncel ÅŸemada `(5, H, W)` ÅŸeklinde `image` anahtarÄ± iÃ§erdiÄŸinden emin olun

### Hata AyÄ±klama Modu

DetaylÄ± hata ayÄ±klama iÃ§in:

```bash
python archaeo_detect.py --verbose 2 2>&1 | tee debug_log.txt
```

Bu komut tÃ¼m hata ayÄ±klama mesajlarÄ±nÄ± hem ekrana hem de `debug_log.txt` dosyasÄ±na yazar.

### EÄŸitim BetiÄŸi Hata AyÄ±klama

**EÄŸitim verisini kontrol edin:**
```bash
# EÄŸitim verisi yapÄ±sÄ±nÄ± doÄŸrulayÄ±n
ls -R workspace/training_data/
# GÃ¶stermelidir: train/images/, train/masks/, val/images/, val/masks/

# Metadata'yÄ± kontrol edin
cat workspace/training_data/metadata.json | python -m json.tool
```

**Veri yÃ¼klemesini test edin:**
```python
# HÄ±zlÄ± test betiÄŸi
import numpy as np
from pathlib import Path

data_dir = Path("workspace/training_data")
train_images = list((data_dir / "train" / "images").glob("*.npz"))
if train_images:
    sample = np.load(train_images[0])
    print(f"Anahtarlar: {sample.files}")
    if 'image' in sample.files:
        img = sample['image']
        print(f"GÃ¶rÃ¼ntÃ¼ ÅŸekli: {img.shape}")
        print(f"Beklenen: (5, 256, 256), AlÄ±nan: {img.shape}")
```

**EÄŸitimi gerÃ§ek zamanlÄ± izleyin:**
```bash
# EÄŸitim geÃ§miÅŸi dosyasÄ±nÄ± izleyin
watch -n 5 'tail -20 workspace/checkpoints/training_history.json'
```

---

## â“ SSS

### ğŸ¤” Genel Sorular

**S: EÄŸitilmiÅŸ modelim yok, yine de kullanabilir miyim?**  
C: Evet! ImageNet aÄŸÄ±rlÄ±klarÄ±nÄ± kullanmak iÃ§in `zero_shot_imagenet: true` kullanÄ±n. AyrÄ±ca, klasik yÃ¶ntemler model gerektirmez.

**S: GPU'm yok, CPU ile Ã§alÄ±ÅŸÄ±r mÄ±?**  
C: Evet, ama daha yavaÅŸ olacaktÄ±r. Klasik yÃ¶ntemleri tercih edin veya kÃ¼Ã§Ã¼k karo boyutu kullanÄ±n.

**S: Hangi yÃ¶ntem en iyi sonuÃ§larÄ± verir?**  
C: Genellikle **fÃ¼zyon** (DL + Klasik) en iyi sonuÃ§larÄ± verir. Ancak, veri kalitenize ve bÃ¶lgenize gÃ¶re deÄŸiÅŸir.

**S: Ä°HA mÄ± uydu muâ€”hangi kaynakla Ã§alÄ±ÅŸÄ±r?**  
C: Sistem **Ã¶ncelikli olarak Ä°HA (drone) nadir gÃ¶rÃ¼ntÃ¼leri** iÃ§in tasarlanmÄ±ÅŸtÄ±r (ortofoto, DSM, DTM ve bu depoda Ã¼retilen tÃ¼rev kanallar). **Uydu gÃ¶rÃ¼ntÃ¼leri de desteklenir**â€”uyumlu Ã§ok bantlÄ± bir GeoTIFF (RGB, varsa DSM/DTM) hizalÄ± bir Ä±zgarada saÄŸlandÄ±ÄŸÄ±nda aynÄ± sÃ¼reÃ§ Ã§alÄ±ÅŸÄ±r. LiDAR tabanlÄ± yÃ¼zeyler ve diÄŸer sensÃ¶rler de aynÄ± ÅŸekilde kullanÄ±labilir. Ã–nemli olan platform deÄŸil, tutarlÄ± bant yapÄ±sÄ± ve jeoreferanstÄ±r.

### ğŸ”§ Teknik Sorular

**S: KaÃ§ bant gerekli?**  
C: Minimum 3 bant (RGB). GÃ¼ncel sÃ¼reÃ§te **5 bant** (RGB + DSM + DTM) kullanÄ±n. **Model tensÃ¶rÃ¼** ise **5 kanaldÄ±r**: R, G, B ve DTM Ã¼zerinden kod iÃ§inde Ã¼retilen **SVF** ile **SLRM** (`archaeo_detect.py` iÃ§indeki `stack_channels`).

**S: Ã–nbellek dosyalarÄ± ne kadar yer kaplar?**  
C: Tipik olarak 10-50 MB. GiriÅŸ dosya boyutuna baÄŸlÄ±dÄ±r. YÃ¼ksek Ã§Ã¶zÃ¼nÃ¼rlÃ¼klÃ¼ veriler iÃ§in daha bÃ¼yÃ¼k (birkaÃ§ GB) olabilir.

**S: SonuÃ§larÄ± nasÄ±l iyileÅŸtirebilirim?**  
C: 
1. Birden fazla kodlayÄ±cÄ± kullanÄ±n (topluluk)
2. FÃ¼zyonu etkinleÅŸtirin
3. EÅŸik deÄŸerlerini optimize edin
4. YÃ¼ksek kaliteli veri kullanÄ±n

**S: Kendi modelimi nasÄ±l eÄŸitirim?**  
C: Proje Ã¶zel eÄŸitim betikleri iÃ§erir! `egitim_verisi_olusturma.py` ve `training.py` kullanarak adÄ±m adÄ±m talimatlar iÃ§in aÅŸaÄŸÄ±daki [Model EÄŸitimi KÄ±lavuzu](#-model-eÄŸitimi-kÄ±lavuzu) bÃ¶lÃ¼mÃ¼ne bakÄ±n.

**S: EÄŸitim betiklerini etkileÅŸimli (dosya seÃ§me penceresiyle) kullanabilir miyim?**  
C: HayÄ±r. Ya `--input`, `--mask` ve `--output` parametrelerini komut satÄ±rÄ±ndan verin ya da `egitim_verisi_olusturma.py` / `training.py` dosyasÄ±ndaki `CONFIG` sÃ¶zlÃ¼ÄŸÃ¼nde varsayÄ±lan yollarÄ± tanÄ±mlayÄ±p IDEâ€™den Ã§alÄ±ÅŸtÄ±rÄ±n.

**S: Ground truth maskelerim yoksa ne olur?**  
C: Yine de sÄ±fÄ±r atÄ±ÅŸ ImageNet aÄŸÄ±rlÄ±klarÄ±yla (`zero_shot_imagenet: true`) veya sadece klasik yÃ¶ntemlerle sistemi kullanabilirsiniz. Ancak, en iyi sonuÃ§lar iÃ§in kendi etiketli verilerinizle Ã¶zel bir model eÄŸitin.

### ğŸ“Š Veri SorularÄ±

**S: Minimum alan Ã§Ã¶zÃ¼nÃ¼rlÃ¼ÄŸÃ¼ nedir?**  
C: Ã–nerilen: 0.5-2 metre/piksel. Daha dÃ¼ÅŸÃ¼k Ã§Ã¶zÃ¼nÃ¼rlÃ¼kte kÃ¼Ã§Ã¼k yapÄ±lar tespit edilemeyebilir.

**S: Maksimum dosya boyutu var mÄ±?**  
C: HayÄ±r, karo sistemi sayesinde Ã§ok bÃ¼yÃ¼k dosyalar iÅŸlenebilir. Test edilmiÅŸ: 50 GB+

**S: FarklÄ± CRS'ler destekleniyor mu?**  
C: Evet, giriÅŸ CRS'i korunur ve Ã§Ä±ktÄ±ya aktarÄ±lÄ±r.

---

## ğŸ“ Model EÄŸitimi KÄ±lavuzu

Bu kÄ±lavuz, kendi etiketli verilerinizle Ã¶zel model eÄŸitme sÃ¼recini adÄ±m adÄ±m aÃ§Ä±klar. Ham veriden eÄŸitilmiÅŸ modele kadar tÃ¼m sÃ¼reci kapsar.

---

### âš¡ HÄ±zlÄ± BaÅŸlangÄ±Ã§ (Ã–zet)

Deneyimli kullanÄ±cÄ±lar iÃ§in minimal iÅŸ akÄ±ÅŸÄ±:

```bash
# 1. Verilerinizi hazÄ±rlayÄ±n (GeoTIFF + ikili maske)
# 2. EÄŸitim karolarÄ±nÄ± oluÅŸturun
python egitim_verisi_olusturma.py --input veri.tif --mask maske.tif --output workspace/training_data

# 3. Modeli eÄŸitin
python training.py --data workspace/training_data --task tile_classification --epochs 50

# 4. EÄŸitilmiÅŸ modeli kullanÄ±n
python archaeo_detect.py --input yeni_alan.tif
```

---

### ğŸ“‹ Genel BakÄ±ÅŸ

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                         MODEL EÄÄ°TÄ°M Ä°Å AKIÅI                                â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                              â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”              â”‚
â”‚   â”‚  ADIM 1      â”‚      â”‚  ADIM 2      â”‚      â”‚  ADIM 3      â”‚              â”‚
â”‚   â”‚  Maske       â”‚ â”€â”€â”€â–º â”‚  Karo        â”‚ â”€â”€â”€â–º â”‚  Model       â”‚              â”‚
â”‚   â”‚  HazÄ±rlama   â”‚      â”‚  OluÅŸturma   â”‚      â”‚  EÄŸitimi     â”‚              â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â”‚
â”‚         â”‚                     â”‚                     â”‚                        â”‚
â”‚         â–¼                     â–¼                     â–¼                        â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”              â”‚
â”‚   â”‚ GeoTIFF +    â”‚      â”‚ 5 kanallÄ±    â”‚      â”‚ EÄŸitilmiÅŸ    â”‚              â”‚
â”‚   â”‚ Ä°kili Maske  â”‚      â”‚ NPZ karolar  â”‚      â”‚ .pth model   â”‚              â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â”‚
â”‚                                                      â”‚                       â”‚
â”‚                                                      â–¼                       â”‚
â”‚                                               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”              â”‚
â”‚                                               â”‚  ADIM 4      â”‚              â”‚
â”‚                                               â”‚  Modeli      â”‚              â”‚
â”‚                                               â”‚  Kullan      â”‚              â”‚
â”‚                                               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Ä°htiyacÄ±nÄ±z olanlar:**
- RGB + DSM + DTM bantlarÄ± iÃ§eren GeoTIFF dosyasÄ±
- Ä°kili maske (GeoTIFF): arkeolojik alanlar = 1, arka plan = 0
- BaÄŸÄ±mlÄ±lÄ±klarÄ± yÃ¼klÃ¼ Python ortamÄ±
- GPU Ã¶nerilir (CPU da Ã§alÄ±ÅŸÄ±r ama yavaÅŸtÄ±r)

---

### ğŸ› ï¸ AdÄ±m 1: Ground Truth Maskeleri HazÄ±rlama

Arkeolojik Ã¶zelliklerin **1** (beyaz), diÄŸer her ÅŸeyin **0** (siyah) olarak iÅŸaretlendiÄŸi ikili bir maske oluÅŸturun.

#### QGIS Kullanarak (Ãœcretsiz, aÃ§Ä±k kaynak)

**Ne yapacaksÄ±nÄ±z:** Arkeolojik Ã¶zelliklerin etrafÄ±na Ã§okgenler Ã§izecek, sonra bunlarÄ± arkeolojik alanlar = 1, diÄŸer her yer = 0 olan bir raster gÃ¶rÃ¼ntÃ¼ye dÃ¶nÃ¼ÅŸtÃ¼receksiniz.

**AdÄ±m 1: Ortofotoyu aÃ§Ä±n**
```
MenÃ¼: Katman â†’ Katman Ekle â†’ Raster Katman Ekle...
GeoTIFF dosyanÄ±za gidin â†’ "Ekle"ye tÄ±klayÄ±n
```
GÃ¶rÃ¼ntÃ¼nÃ¼z harita tuvalinde gÃ¶rÃ¼nmelidir. YakÄ±nlaÅŸtÄ±rmak iÃ§in fare tekerleÄŸini, kaydÄ±rmak iÃ§in orta tuÅŸu basÄ±lÄ± tutun.

**AdÄ±m 2: SayÄ±sallaÅŸtÄ±rma iÃ§in yeni Ã§okgen katmanÄ± oluÅŸturun**
```
MenÃ¼: Katman â†’ Katman OluÅŸtur â†’ Yeni Shapefile KatmanÄ±...
```
AÃ§Ä±lan pencerede:
- **Dosya adÄ±:** "..." butonuna tÄ±klayÄ±p kayÄ±t yerini seÃ§in (Ã¶rn. `arkeolojik_maske.shp`)
- **Geometri tipi:** "Ã‡okgen" seÃ§in
- **KRS (Koordinat Referans Sistemi):** KÃ¼re ikonuna tÄ±klayÄ±n â†’ rasterinizin koordinat sistemini arayÄ±n (emin deÄŸilseniz raster Ã¶zelliklerinden bakÄ±n)
- "Tamam"a tÄ±klayÄ±n

Katmanlar panelinde yeni boÅŸ bir katman gÃ¶rÃ¼nÃ¼r.

**AdÄ±m 3: SayÄ±sallaÅŸtÄ±rmaya baÅŸlayÄ±n (Ã§okgen Ã§izimi)**
```
1. Katmanlar panelinde yeni katmanÄ±nÄ±zÄ± seÃ§in (Ã¼zerine tÄ±klayÄ±n)
2. MenÃ¼: Katman â†’ Sketching'e GeÃ§ (veya kalem ikonuna tÄ±klayÄ±n)
3. AraÃ§ Ã§ubuÄŸunda "Ã‡okgen Objesi Ekle" butonunu bulun (+ iÅŸaretli Ã§okgen)
4. Butona tÄ±klayÄ±n, sonra haritada kÃ¶ÅŸe noktalarÄ± eklemek iÃ§in tÄ±klamaya baÅŸlayÄ±n
5. Her Ã§okgeni bitirmek iÃ§in saÄŸ tÄ±klayÄ±n
```

**SayÄ±sallaÅŸtÄ±rma ipuÃ§larÄ±:**
- Hassasiyet iÃ§in yakÄ±nlaÅŸtÄ±rÄ±n (fare tekerleÄŸi)
- TÃ¼mÃ¼lÃ¼slerin, duvarlarÄ±n, hendeklerin etrafÄ±nÄ± Ã§izin - arkeolojik olan her ÅŸey
- Hata yaparsanÄ±z: Ctrl+Z ile geri alÄ±n
- Her tÄ±klama bir kÃ¶ÅŸe noktasÄ± ekler; saÄŸ tÄ±klama Ã§okgeni kapatÄ±r
- Ä°htiyaÃ§ kadar Ã§okgen Ã§izin

**AdÄ±m 4: DÃ¼zenlemelerinizi kaydedin**
```
MenÃ¼: Katman â†’ Sketching'e GeÃ§ â†’ SorulduÄŸunda "Kaydet"e tÄ±klayÄ±n
Veya: AraÃ§ Ã§ubuÄŸundaki disket ikonuna tÄ±klayÄ±n
```

**AdÄ±m 5: Ã‡okgenleri rastera dÃ¶nÃ¼ÅŸtÃ¼rÃ¼n (maske)**
```
MenÃ¼: Raster â†’ DÃ¶nÃ¼ÅŸtÃ¼rme â†’ RasterleÅŸtir (VektÃ¶rÃ¼ Rastera)...
```
AÃ§Ä±lan pencerede:
- **GiriÅŸ katmanÄ±:** Ã‡okgen katmanÄ±nÄ±z (`arkeolojik_maske`)
- **Yakma deÄŸeri iÃ§in kullanÄ±lacak alan:** BoÅŸ bÄ±rakÄ±n (sabit deÄŸer kullanacaÄŸÄ±z)
- **Yakmak iÃ§in sabit deÄŸer:** `1` girin
- **Ã‡Ä±ktÄ± raster boyut birimi:** CoÄŸrafi birimler
- **GeniÅŸlik/Yatay Ã§Ã¶zÃ¼nÃ¼rlÃ¼k:** GiriÅŸ rasterinizle aynÄ± (Ã¶rn. 1m Ã§Ã¶zÃ¼nÃ¼rlÃ¼k iÃ§in `1.0`)
- **YÃ¼kseklik/Dikey Ã§Ã¶zÃ¼nÃ¼rlÃ¼k:** AynÄ± deÄŸer (Ã¶rn. `1.0`)
- **Ã‡Ä±ktÄ± kapsamÄ±:** "..." â†’ "Katmandan Hesapla" â†’ GiriÅŸ rasterinizi seÃ§in
- **RasterleÅŸtirilmiÅŸ:** "..." â†’ Dosyaya Kaydet â†’ `ground_truth.tif` olarak adlandÄ±rÄ±n
- "Ã‡alÄ±ÅŸtÄ±r"a tÄ±klayÄ±n

**AdÄ±m 6: NoData alanlarÄ±nÄ± sÄ±fÄ±rla doldurun**

RasterleÅŸtirme aracÄ± Ã§okgen olmayan yerlerde NoData oluÅŸturur. BunlarÄ±n 0 olmasÄ± gerekiyor.
```
MenÃ¼: Raster â†’ Raster Hesap Makinesi...
```
Bu ifadeyi girin (gerÃ§ek katman adÄ±nÄ±zla deÄŸiÅŸtirin):
```
("ground_truth@1" >= 1) * 1
```
Veya ÅŸunu kullanÄ±n:
```
MenÃ¼: Ä°ÅŸleme â†’ AraÃ§ Kutusu â†’ "Fill nodata" arayÄ±n
"Fill NoData cells" aracÄ±nÄ± dolgu deÄŸeri = 0 ile kullanÄ±n
```

**Maskenizi doÄŸrulayÄ±n:**
- DeÄŸerler sadece 0 ve 1 olmalÄ±
- Katmana saÄŸ tÄ±klayÄ±n â†’ Ã–zellikler â†’ Sembolloji â†’ min/max deÄŸerlerini kontrol edin
- Boyutlar giriÅŸ rasterinizle tam olarak eÅŸleÅŸmeli

---

#### ArcGIS Pro Kullanarak

**Ne yapacaksÄ±nÄ±z:** Ã‡okgen feature class oluÅŸturacak, arkeolojik Ã¶zellikleri sayÄ±sallaÅŸtÄ±racak, sonra raster maskeye dÃ¶nÃ¼ÅŸtÃ¼receksiniz.

**AdÄ±m 1: Yeni proje oluÅŸturun ve verilerinizi ekleyin**
```
1. ArcGIS Pro'yu aÃ§Ä±n â†’ New Project â†’ Map
2. Ä°sim ve konum verin â†’ OK
3. Map sekmesi â†’ Add Data â†’ GeoTIFF'inize gÃ¶z atÄ±n â†’ Add
```
Ortofotunuz haritada gÃ¶rÃ¼nmelidir. YakÄ±nlaÅŸtÄ±rmak iÃ§in fare tekerleÄŸi, kaydÄ±rmak iÃ§in tekerleÄŸi basÄ±lÄ± tutun.

**AdÄ±m 2: Rasterinizin Ã¶zelliklerini kontrol edin (sonrasÄ± iÃ§in Ã¶nemli)**
```
1. Contents panelinde rasterinize saÄŸ tÄ±klayÄ±n â†’ Properties
2. "Source" sekmesine gidin â†’ ÅunlarÄ± not edin:
   - Cell Size (HÃ¼cre Boyutu, Ã¶rn. 1.0 x 1.0)
   - Extent (Kapsam - Top, Left, Right, Bottom koordinatlarÄ±)
   - Spatial Reference (Mekansal Referans, Ã¶rn. EPSG:32635)
```
BunlarÄ± yazÄ±n - maskenizi eÅŸleÅŸtirmek iÃ§in gerekecek.

**AdÄ±m 3: SayÄ±sallaÅŸtÄ±rma iÃ§in yeni feature class oluÅŸturun**
```
1. Catalog panelinde projenizin geodatabase'ini (.gdb) geniÅŸletin
2. Geodatabase'e saÄŸ tÄ±klayÄ±n â†’ New â†’ Feature Class
```
Sihirbazda:
- **Name (Ad):** `arkeolojik_ozellikler`
- **Alias (Takma Ad):** Arkeolojik Ã–zellikler (isteÄŸe baÄŸlÄ±)
- **Feature Class Type:** Polygon
- "Next"e tÄ±klayÄ±n
- **Fields (Alanlar):** AtlayÄ±n (sonra ekleyeceÄŸiz) â†’ "Next"e tÄ±klayÄ±n
- **Spatial Reference:** KÃ¼reye tÄ±klayÄ±n â†’ Import â†’ Rasterinizi seÃ§in
- "Finish"e tÄ±klayÄ±n

Yeni boÅŸ katman Contents'te gÃ¶rÃ¼nÃ¼r.

**AdÄ±m 4: SayÄ±sallaÅŸtÄ±rmaya baÅŸlayÄ±n**
```
1. Contents'te seÃ§mek iÃ§in yeni katmanÄ±nÄ±za tÄ±klayÄ±n
2. Edit sekmesi â†’ Create (Create Features panelini aÃ§ar)
3. Create Features panelinde "arkeolojik_ozellikler"e tÄ±klayÄ±n
4. "Polygon" aracÄ±nÄ± seÃ§in
5. KÃ¶ÅŸe noktalarÄ± eklemek iÃ§in haritada tÄ±klayÄ±n, bitirmek iÃ§in Ã§ift tÄ±klayÄ±n
```

**SayÄ±sallaÅŸtÄ±rma ipuÃ§larÄ±:**
- Ã‡izerken yakÄ±nlaÅŸtÄ±rmak iÃ§in `Z`, kaydÄ±rmak iÃ§in `C` tuÅŸuna basÄ±n
- Son kÃ¶ÅŸe noktasÄ±nÄ± geri almak iÃ§in `Ctrl+Z`
- Her Ã§okgeni bitirmek iÃ§in Ã§ift tÄ±klayÄ±n (veya `F2`)
- GÃ¶rÃ¼nen tÃ¼m arkeolojik Ã¶zelliklerin etrafÄ±nÄ± Ã§izin
- MÃ¼mkÃ¼n olduÄŸunca hassas olun - bunlar eÄŸitim etiketleriniz olacak!

**AdÄ±m 5: DÃ¼zenlemelerinizi kaydedin**
```
Edit sekmesi â†’ Save â†’ Save Edits
```

**AdÄ±m 6: Raster deÄŸeri iÃ§in alan ekleyin**
```
1. Contents'te katmanÄ±nÄ±za saÄŸ tÄ±klayÄ±n â†’ Attribute Table
2. "Add Field" butonuna tÄ±klayÄ±n (tablonun Ã¼stÃ¼nde)
3. Field Name: yakma_degeri
4. Data Type: Short (Integer)
5. Fields sekmesinde "Save"e tÄ±klayÄ±n
```

**AdÄ±m 7: TÃ¼m Ã§okgenlere deÄŸer 1 atayÄ±n**
```
1. Ã–znitelik tablosunda "yakma_degeri" sÃ¼tun baÅŸlÄ±ÄŸÄ±na saÄŸ tÄ±klayÄ±n
2. "Calculate Field..." seÃ§in
3. Expression kutusuna sadece ÅŸunu yazÄ±n: 1
4. "OK"e tÄ±klayÄ±n
```
TÃ¼m satÄ±rlarda yakma_degeri sÃ¼tununda `1` gÃ¶rÃ¼nmelidir.

**AdÄ±m 8: Rastera dÃ¶nÃ¼ÅŸtÃ¼rÃ¼n**
```
Analysis sekmesi â†’ Tools â†’ "Polygon to Raster" arayÄ±n
```
AraÃ§ penceresinde:
- **Input Features:** arkeolojik_ozellikler
- **Value field:** yakma_degeri
- **Output Raster Dataset:** GÃ¶z at â†’ `ground_truth.tif` olarak kaydedin
- **Cell assignment type:** CELL_CENTER
- **Priority field:** NONE
- **Cellsize:** GiriÅŸ rasterinizle aynÄ± (Ã¶rn. `1`)

**Ã–nemli - Environment AyarlarÄ±:**
```
AracÄ±n altÄ±ndaki "Environments" sekmesine tÄ±klayÄ±n:
- Snap Raster: GiriÅŸ rasterinizi seÃ§in (hizalamayÄ± garantiler!)
- Cell Size: GiriÅŸ rasterinizle aynÄ±
- Extent: GiriÅŸ rasterinizle aynÄ±
```
"Run"a tÄ±klayÄ±n

**AdÄ±m 9: NoData'yÄ± 0'a dÃ¶nÃ¼ÅŸtÃ¼rÃ¼n**

VarsayÄ±lan olarak, Ã§okgenlerin dÄ±ÅŸÄ±ndaki alanlar NoData olur. BunlarÄ±n 0 olmasÄ± gerekiyor.
```
Analysis sekmesi â†’ Tools â†’ "Reclassify" arayÄ±n
```
Veya Raster Calculator kullanÄ±n:
```
Analysis sekmesi â†’ Tools â†’ "Raster Calculator" arayÄ±n
Expression: Con(IsNull("ground_truth.tif"), 0, "ground_truth.tif")
Output: ground_truth_final.tif
```

Reclassify ile alternatif:
```
- Input raster: ground_truth.tif
- Reclass field: Value
- Reclassification (Yeniden sÄ±nÄ±flandÄ±rma):
  - SatÄ±r ekle: Old = NoData, New = 0
  - Mevcut: Old = 1, New = 1
- Output: ground_truth_final.tif
```

**AdÄ±m 10: Maskenizi doÄŸrulayÄ±n**
```
1. Final maskeyi haritanÄ±za ekleyin
2. SaÄŸ tÄ±klayÄ±n â†’ Properties â†’ Source â†’ Kontrol edin:
   - HÃ¼cre boyutu giriÅŸle eÅŸleÅŸiyor âœ“
   - Kapsam giriÅŸle eÅŸleÅŸiyor âœ“
   - DeÄŸerler sadece 0 ve 1 âœ“
```

**YaygÄ±n sorunlar:**
- **Maske kapsamÄ± eÅŸleÅŸmiyor:** Polygon to Raster'Ä± doÄŸru Environment ayarlarÄ±yla yeniden Ã§alÄ±ÅŸtÄ±rÄ±n
- **Maske yanlÄ±ÅŸ hÃ¼cre boyutunda:** AraÃ§ta ve Environment'ta hÃ¼cre boyutunu aÃ§Ä±kÃ§a ayarlayÄ±n
- **Maske tamamen NoData:** yakma_degeri alanÄ±nÄ±n 1 deÄŸerine sahip olduÄŸunu kontrol edin

---

#### Python Kullanarak

```python
import rasterio
import numpy as np

# Maske dizisi oluÅŸtur (giriÅŸle aynÄ± boyutlarda)
mask = np.zeros((height, width), dtype=np.uint8)

# Arkeolojik alanlarÄ± iÅŸaretle (Ã¶rnek: koordinatlardan veya Ã§okgenlerden)
mask[100:200, 150:250] = 1  # GerÃ§ek alanlarla deÄŸiÅŸtirin

# GeoTIFF olarak kaydet (giriÅŸ CRS ve transform ile eÅŸleÅŸmeli!)
with rasterio.open('maske.tif', 'w', driver='GTiff',
                   height=height, width=width, count=1, 
                   dtype='uint8', crs=giris_crs, 
                   transform=giris_transform) as dst:
    dst.write(mask, 1)
```

> **Ã–nemli:** Maske boyutlarÄ±, CRS ve Ã§Ã¶zÃ¼nÃ¼rlÃ¼k giriÅŸ GeoTIFF'inizle tam olarak eÅŸleÅŸmelidir!

---

### ğŸ“¦ AdÄ±m 2: EÄŸitim KarolarÄ± OluÅŸturma

`egitim_verisi_olusturma.py` betiÄŸi GeoTIFF + maskenizi **5 kanallÄ±** (R, G, B, SVF, SLRM) eÄŸitim karolarÄ±na dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.

#### Temel Komut

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data
```

#### IDE / CLI

Betik etkileÅŸimli dosya penceresi aÃ§maz. Ya `--input` / `--mask` / `--output` verin ya da betikteki `CONFIG` sÃ¶zlÃ¼ÄŸÃ¼nÃ¼ doldurup IDEâ€™den Ã§alÄ±ÅŸtÄ±rÄ±n.

#### Ä°Ã§eride Ne Olur

```
GiriÅŸ GeoTIFF (5 bant)           Ground Truth Maske
       |                                |
       v                                |
+------------------+                    |
| RGB + DSM + DTM  |                    |
| bantlarÄ±nÄ± oku   |                    |
+--------+---------+                    |
         |                              |
         v                              |
+------------------+                    |
| DTM Ã¼zerinde RVT |                    |
| SVF + SLRM       |                    |
+--------+---------+                    |
         |                              |
         v                              |
+------------------+                    |
| stack_channels   |<-------------------+
| R,G,B,SVF,SLRM   |
| 256x256 karolar  |
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

#### Temel Parametreler

Tam liste iÃ§in: `python egitim_verisi_olusturma.py --help`. SÄ±k kullanÄ±lanlar:

| Parametre | VarsayÄ±lan (betikteki `CONFIG`) | AÃ§Ä±klama |
|-----------|--------------------------------|----------|
| `--input` / `-i` | CLI veya `CONFIG` ile zorunlu | Ã‡ok bantlÄ± GeoTIFF (RGB + DSM + DTM) |
| `--mask` / `-m` | CLI veya `CONFIG` ile zorunlu | Ground truth maske (0 dÄ±ÅŸÄ±ndaki deÄŸerler pozitif sayÄ±lÄ±p ikiliye Ã§evrilir) |
| `--output` / `-o` | `workspace/training_data` | Ã‡Ä±ktÄ± kÃ¶kÃ¼ (`train/`, `val/`, `metadata.json`, â€¦) |
| `--tile-size` / `-t` | `256` | Karo boyutu (piksel) |
| `--overlap` | `128` | KaydÄ±rmalÄ± pencere Ã¶rtÃ¼ÅŸmesi (eÄŸitim/Ã§Ä±karÄ±mda metadata ile tutarlÄ± kalmalÄ±) |
| `--bands` / `-b` | `1,2,3,4,5` | 1 tabanlÄ± GeoTIFF bant indeksleri: R, G, B, DSM, DTM |
| `--min-positive` | `0.0` | Karoda minimum pozitif piksel oranÄ± |
| `--tile-label-min-positive-ratio` | `CONFIG`â€™ten | Karo sÄ±nÄ±fÄ± etiketi iÃ§in minimum pozitif oran (0 = en az bir pozitif piksel yeter) |
| `--max-nodata` | `0.3` | Karo baÅŸÄ±na izin verilen maksimum NoData oranÄ± |
| `--train-ratio` | `0.8` | EÄŸitim oranÄ± |
| `--train-negative-keep-ratio` | `1.0` | Tamamen negatif **eÄŸitim** karolarÄ±nÄ±n tutulacak kesri (`0` = hepsini at, `1` = hepsini tut) |
| `--train-negative-max` | `None` | Tutulan negatif eÄŸitim karosu iÃ§in Ã¼st sÄ±nÄ±r |
| `--split-mode` | `spatial` | `spatial` (Ã¶nerilir) veya `random` train/val bÃ¶lmesi |
| `--no-normalize` | kapalÄ± | `robust_norm` atlanÄ±r |
| `--format` | `npz` | `npz` veya `npy` |
| `--num-workers` | CPUâ€™ya gÃ¶re | Paralel iÅŸÃ§i sayÄ±sÄ± |
| `--tile-prefix` | `""` | Ä°steÄŸe baÄŸlÄ± dosya adÄ± Ã¶neki (boÅŸsa otomatik Ã¶nek) |
| `--append` / `--no-append` | temiz yeniden Ã¼ret | Mevcut karolara ekleme vs tam yeniden oluÅŸturma |

#### Senaryoya GÃ¶re Ã–nerilen Ayarlar

| Senaryo | Komut |
|---------|-------|
| **Standart** | `--tile-size 256 --overlap 64` |
| **BÃ¼yÃ¼k yapÄ±lar** | `--tile-size 512 --overlap 64` |
| **Dengesiz veri** (<%5 arkeolojik) | `--train-negative-keep-ratio 0.2 --min-positive 0.01` |
| **HÄ±zlÄ± test** | `--tile-size 256 --train-ratio 0.9` |

#### Ã‡Ä±ktÄ±: model tensÃ¶rÃ¼ (5 kanal)

Kanonik sÄ±ra `archeo_shared/channels.py` â†’ `MODEL_CHANNEL_NAMES`; `archaeo_detect.py` iÃ§indeki `stack_channels()` ile aynÄ±dÄ±r.

| # | Kanal | Kaynak |
|---|-------|--------|
| 0â€“2 | R, G, B | GeoTIFFâ€™te seÃ§ilen RGB bantlarÄ± |
| 3 | SVF | DTM Ã¼zerinde RVT Sky-View Factor |
| 4 | SLRM | DTM Ã¼zerinde RVT Simple Local Relief Model (gerekirse Gaussian yedek) |

DSM/DTM **bantlarÄ±** maskeleme ve tÃ¼rev hesabÄ± iÃ§in hÃ¢lÃ¢ gereklidir; kaydedilen `image` tensÃ¶rÃ¼nde yalnÄ±zca RGB + iki kabartma kanalÄ± bulunur.

---

### ğŸš€ AdÄ±m 3: Modeli EÄŸitme

**5 kanallÄ±** karolarÄ±nÄ±z Ã¼zerinde SMP tabanlÄ± U-Net (ve ilgili baÅŸlÄ±klar) eÄŸitmek iÃ§in `training.py` kullanÄ±n; **CBAM** isteÄŸe baÄŸlÄ±dÄ±r (`CONFIG` iÃ§inde `no_attention`).

#### Temel EÄŸitim

```bash
python training.py --data workspace/training_data
```

KayÄ±tlÄ± `training.py` `CONFIG` deÄŸerlerini kullanÄ±r (Ã¶r. **U-Net**, **ResNet50**, **BCE** kaybÄ±, **patience 20**, **CBAM kapalÄ±** â€” `no_attention: true`, **AMP** genelde aÃ§Ä±k). `publish_active: true` iken en iyi aÄŸÄ±rlÄ±klar `workspace/checkpoints/active/` altÄ±na kopyalanÄ±r.

#### TÃ¼m SeÃ§eneklerle Tam Komut

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

#### Temel Parametreler

| Parametre | VarsayÄ±lan | SeÃ§enekler / Notlar |
|-----------|------------|---------------------|
| `--data` | `workspace/training_data` | AdÄ±m 2 Ã§Ä±ktÄ±sÄ±nÄ±n yolu (eÅŸleÅŸtirilmiÅŸ veya Positive/Negative dÃ¼zeni) |
| `--task` | `tile_classification` | `segmentation` veya `tile_classification` |
| `--arch` | `Unet` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN` |
| `--encoder` | `resnet50` | `resnet34`, `efficientnet-b3`, `densenet121` |
| `--epochs` | `50` | Daha fazla = potansiyel olarak daha iyi (erken durdurma ile) |
| `--batch-size` | `16` | GPU belleÄŸi izin veriyorsa artÄ±rÄ±n |
| `--lr` | `1e-4` | KayÄ±p salÄ±nÄ±yorsa azaltÄ±n |
| `--loss` | `bce` | `bce` / `focal` (`tile_classification`); `segmentation` iÃ§in ayrÄ±ca `dice` / `combined` |
| `--balance-mode` | `auto` | `auto`, `manual`, `none` (BCE tarafÄ±) |
| `--patience` | `20` | N epoch iyileÅŸme yoksa erken durdurma |
| `--no-attention` | aÃ§Ä±k (`CONFIG` ile) | VarsayÄ±lan **true** â†’ CBAM **kapalÄ±**; aÃ§mak iÃ§in `CONFIG`â€™ta `no_attention: false` |
| `--no-amp` | kapalÄ± | Karma hassasiyeti (FP16) kapatÄ±r |

#### DoÄŸru AyarlarÄ± SeÃ§me

**Model Mimarisi:**

| Mimari | HÄ±z | DoÄŸruluk | Ne Zaman KullanÄ±lÄ±r |
|--------|-----|----------|---------------------|
| `Unet` | HÄ±zlÄ± | Ä°yi | **Buradan baÅŸlayÄ±n** - gÃ¼venilir temel |
| `UnetPlusPlus` | Orta | MÃ¼kemmel | Daha yÃ¼ksek doÄŸruluk gerektiÄŸinde |
| `DeepLabV3Plus` | Orta | MÃ¼kemmel | Ã‡ok Ã¶lÃ§ekli yapÄ±lar |

**KodlayÄ±cÄ±:**

| KodlayÄ±cÄ± | HÄ±z | DoÄŸruluk | Bellek |
|-----------|-----|----------|--------|
| `resnet34` | HÄ±zlÄ± | Ä°yi | DÃ¼ÅŸÃ¼k | **Ã–nerilen baÅŸlangÄ±Ã§** |
| `resnet50` | Orta | Daha iyi | Orta | Daha iyi doÄŸruluk |
| `efficientnet-b3` | HÄ±zlÄ± | MÃ¼kemmel | DÃ¼ÅŸÃ¼k | En iyi verimlilik |

**KayÄ±p Fonksiyonu:**

| KayÄ±p | Ne Zaman KullanÄ±lÄ±r |
|-------|---------------------|
| `bce` | **`training.py` CONFIG varsayÄ±lanÄ±**; `tile_classification` iÃ§in `focal` ile birlikte uygun seÃ§enekler |
| `focal` | Karo etiketlerinde gÃ¼Ã§lÃ¼ sÄ±nÄ±f dengesizliÄŸi |
| `combined` / `dice` | Ã–ncelikle **segmentation** (piksel maskesi) gÃ¶revi iÃ§in |

#### EÄŸitim Ã‡Ä±ktÄ±sÄ±

```
workspace/checkpoints/
â”œâ”€â”€ active/model.pth                         â† en iyi aÄŸÄ±rlÄ±klarÄ±n kopyasÄ± (training.py iÃ§indeki publish_active)
â”œâ”€â”€ active/training_metadata.json           â† trained_model_only iÃ§in tile / overlap / bands (+ ÅŸema)
â”œâ”€â”€ active/published_from.json              â† kopyanÄ±n kaynak checkpointâ€™e iÅŸaret eden manifest
â”œâ”€â”€ epochs/                                  â† save_every_epoch aÃ§Ä±ksa (CONFIGâ€™te varsayÄ±lan genelde aÃ§Ä±k) epoch checkpointâ€™leri
â””â”€â”€ training_history.json                   â† EÄŸitim metrikleri
```

`weights` olarak `workspace/checkpoints/active/model.pth` veya `workspace/checkpoints/epochs/` altÄ±ndaki belirli bir Ã§alÄ±ÅŸtÄ±rmayÄ± seÃ§ebilirsiniz; her durumda **`training_metadata.json` aynÄ± mimari, kanal sayÄ±sÄ±, karo boyutu, overlap ve bantlarÄ±** tanÄ±mlamalÄ±dÄ±r.

`channel_importance_history.json` dosyasÄ± da (etkinse) `workspace/checkpoints/` altÄ±nda Ã¼retilir; epoch bazlÄ± bant Ã¶nem sÄ±ralarÄ±nÄ± iÃ§erir.

#### EÄŸitimi Ä°zleme

Konsol Ã§Ä±ktÄ±sÄ±nÄ± izleyin:

```
Epoch  1/50 | Train Loss: 0.45 | Val Loss: 0.39 | Val IoU: 0.62 | LR: 1e-04
  â†’ En iyi model kaydedildi
Epoch  2/50 | Train Loss: 0.38 | Val Loss: 0.34 | Val IoU: 0.68 | LR: 1e-04
  â†’ En iyi model kaydedildi
...
Erken durdurma: En iyi model 15. epoch'ta (Val IoU: 0.79)
```

**Metriklerin anlamÄ±:**
- **Val IoU** (KesiÅŸim/BirleÅŸim): YÃ¼ksek = daha iyi. Hedef: 0.6-0.8
- **Val Loss**: DÃ¼ÅŸÃ¼k = daha iyi. Zamanla azalmalÄ±
- **Train Loss**: Val Loss'tan biraz dÃ¼ÅŸÃ¼k olmalÄ± (Ã§ok dÃ¼ÅŸÃ¼kse = aÅŸÄ±rÄ± Ã¶ÄŸrenme)

---

### ğŸ“Š AdÄ±m 4: EÄŸitilmiÅŸ Modeli Kullanma

#### Komut SatÄ±rÄ±ndan

```bash
python archaeo_detect.py \
  --weights workspace/checkpoints/active/model.pth \
  --training-metadata workspace/checkpoints/active/training_metadata.json \
  --input yeni_alan.tif \
  --th 0.6
```

#### config.yaml Ãœzerinden

```yaml
weights: "workspace/checkpoints/active/model.pth"
training_metadata: "workspace/checkpoints/active/training_metadata.json"
zero_shot_imagenet: false
trained_model_only: true
```

Sonra sadece Ã§alÄ±ÅŸtÄ±rÄ±n:
```bash
python archaeo_detect.py
```

---

### ğŸ”§ Sorun Giderme

#### Veri HazÄ±rlama SorunlarÄ±

| Sorun | Neden | Ã‡Ã¶zÃ¼m |
|-------|-------|-------|
| "Maske boyutlarÄ± eÅŸleÅŸmiyor" | FarklÄ± Ã§Ã¶zÃ¼nÃ¼rlÃ¼k/kapsam | Maskeyi yeniden Ã¶rnekle: `gdalwarp -tr 1.0 1.0 -r nearest maske.tif maske_duzeltilmis.tif` |
| "GeÃ§erli karo bulunamadÄ±" | `--min-positive` Ã§ok yÃ¼ksek | `0.0` veya `0.01`'e dÃ¼ÅŸÃ¼rÃ¼n |
| "Bellek hatasÄ±" | BÃ¼yÃ¼k giriÅŸ dosyasÄ± | `--tile-size`'Ä± 128'e dÃ¼ÅŸÃ¼rÃ¼n |

#### EÄŸitim SorunlarÄ±

| Sorun | Neden | Ã‡Ã¶zÃ¼m |
|-------|-------|-------|
| KayÄ±p dÃ¼ÅŸmÃ¼yor | Ã–ÄŸrenme oranÄ± Ã§ok yÃ¼ksek | `--lr 5e-5` veya `1e-5` kullanÄ±n |
| GPU bellek yetersiz | Batch boyutu Ã§ok bÃ¼yÃ¼k | `--batch-size 4` veya `--no-amp` kullanÄ±n |
| AÅŸÄ±rÄ± Ã¶ÄŸrenme (train << val loss) | Ã‡ok az veri | Daha fazla karo ekleyin veya `--loss focal` kullanÄ±n |
| TÃ¼m tahminler = 0 | SÄ±nÄ±f dengesizliÄŸi | `--loss focal` kullanÄ±n, veri hazÄ±rlamada negatif eÄŸitim karolarÄ±nÄ± azaltÄ±n (Ã¶rn. `--train-negative-keep-ratio 0.2`) |
| EÄŸitim Ã§ok yavaÅŸ | GPU yok / kÃ¼Ã§Ã¼k batch | GPU kullanÄ±n, `--batch-size` artÄ±rÄ±n, AMP etkinleÅŸtirin |

#### HÄ±zlÄ± TanÄ± KomutlarÄ±

```bash
# EÄŸitim verisi yapÄ±sÄ±nÄ± kontrol et
ls -R workspace/training_data/

# Metadata'yÄ± gÃ¶rÃ¼ntÃ¼le
cat workspace/training_data/metadata.json | python -m json.tool

# Veri yÃ¼klemeyi test et
python -c "import numpy as np; d=np.load('workspace/training_data/train/images/tile_00000_00000.npz'); print(d['image'].shape)"
# Beklenen: (5, 256, 256)
```

---

### ğŸ’¡ En Ä°yi Uygulamalar

#### Veri Kalitesi Kontrol Listesi

- [ ] Maskeler doÄŸru (kesin sÄ±nÄ±rlar)
- [ ] TÃ¼m arkeolojik Ã¶zellikler tutarlÄ± ÅŸekilde etiketlenmiÅŸ
- [ ] Dengeli veri kÃ¼mesi (%30-50 pozitif karo)
- [ ] Negatiflerde Ã§eÅŸitli arazi tÃ¼rleri
- [ ] Minimum 1000 karo (2000-5000 Ã¶nerilir)

#### EÄŸitim Ä°ÅŸ AkÄ±ÅŸÄ±

```
1. HÄ±zlÄ± test (5 epoch)      â†’ Her ÅŸeyin Ã§alÄ±ÅŸtÄ±ÄŸÄ±nÄ± doÄŸrula
2. Temel (50 epoch)          â†’ BaÅŸlangÄ±Ã§ noktasÄ± belirle
3. Optimize et (daha iyi kodlayÄ±cÄ±/mimari dene)
4. Ä°nce ayar (gerekirse LR dÃ¼ÅŸÃ¼r)
```

#### Performans Beklentileri

| Veri KÃ¼mesi Boyutu | Beklenen Val IoU | EÄŸitim SÃ¼resi (GPU) |
|--------------------|------------------|---------------------|
| 500-1000 karo | 0.55-0.65 | 30-60 dk |
| 1000-3000 karo | 0.65-0.75 | 1-2 saat |
| 3000-5000 karo | 0.70-0.80 | 2-4 saat |
| 5000+ karo | 0.75-0.85 | 4+ saat |

---

### ğŸ“š Tam Ã–rnek: UÃ§tan Uca

```bash
# 1. EÄŸitim verisini oluÅŸtur (tek dengeleme mekanizmasÄ±: train-negatif filtreleme)
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output workspace/training_data \
  --tile-size 256 \
  --train-negative-keep-ratio 0.3

# 2. Modeli eÄŸit
python training.py \
  --data workspace/training_data \
  --arch Unet \
  --encoder resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --loss bce

# 3. Yeni alanda Ã§Ä±karÄ±m yap
python archaeo_detect.py \
  --weights workspace/checkpoints/active/model.pth \
  --training-metadata workspace/checkpoints/active/training_metadata.json \
  --input yeni_alan.tif \
  --th 0.6 \
  --enable-fusion
```

**Beklenen sonuÃ§lar:**
- ~1000-2000 eÄŸitim karosu
- Val IoU: 0.65-0.75
- EÄŸitim sÃ¼resi: 1-2 saat (GPU)
- Model dosyasÄ±: ~70 MB

---

## ğŸ”¬ GeliÅŸmiÅŸ Ã–zellikler

### Ã–zel Model EÄŸitimi

> **ğŸ“– DetaylÄ± eÄŸitim kÄ±lavuzu iÃ§in yukarÄ±daki [Model EÄŸitimi KÄ±lavuzu](#-model-eÄŸitimi-kÄ±lavuzu) bÃ¶lÃ¼mÃ¼ne bakÄ±n.**

Proje, Ã¶zel modeller eÄŸitmek iÃ§in iki Ã¶zel betik iÃ§erir:

- **`egitim_verisi_olusturma.py`**: GeoTIFF + ground truth maskelerinden **5 kanallÄ±** eÄŸitim karolarÄ± oluÅŸturur
- **`training.py`**: SMP U-Net ailesinde eÄŸitim; **CBAM** isteÄŸe baÄŸlÄ± (`no_attention` / `CONFIG`)

**HÄ±zlÄ± BaÅŸlangÄ±Ã§:**

```bash
# 1. EÄŸitim verisi oluÅŸtur
python egitim_verisi_olusturma.py --input alan.tif --mask maske.tif --output workspace/training_data

# 2. Model eÄŸit
python training.py --data workspace/training_data --task tile_classification --epochs 50

# 3. EÄŸitilmiÅŸ modeli kullan
python archaeo_detect.py
```

**Temel Ã–zellikler:**
- âœ… 5 kanallÄ± giriÅŸ (R, G, B, SVF, SLRM) â€” Ã§Ä±karÄ±mla uyumlu
- âœ… Ä°steÄŸe baÄŸlÄ± CBAM dikkat (`training.py`)
- âœ… KayÄ±plar: **BCE / Focal** (`tile_classification`); **BCE / Dice / Combined / Focal** (`segmentation`)
- âœ… Karma hassasiyet eÄŸitimi
- âœ… Erken durdurma ve checkpoint kaydetme

Tam dokÃ¼mantasyon, Ã¶rnekler ve sorun giderme iÃ§in [Model EÄŸitimi KÄ±lavuzu](#-model-eÄŸitimi-kÄ±lavuzu) bÃ¶lÃ¼mÃ¼ne bakÄ±n.

### KodlayÄ±cÄ± seÃ§imi

KodlayÄ±cÄ±lar **Segmentation Models PyTorch** omurga adlarÄ±dÄ±r (`resnet34`, `resnet50`, `efficientnet-b3`, â€¦). `config.yaml` iÃ§indeki `encoder` / `encoders` veya CLI bayraklarÄ±yla seÃ§in. YÃ¼klÃ¼ `segmentation-models-pytorch` sÃ¼rÃ¼mÃ¼nÃ¼zÃ¼n desteklediÄŸi ve checkpointâ€™inizle eÅŸleÅŸen adlarÄ± kullanÄ±n.

### API KullanÄ±mÄ±

Python kodundan betiÄŸi Ã§aÄŸÄ±rma:

```python
import subprocess

result = subprocess.run([
    'python', 'archaeo_detect.py',
    '--input', 'benim_alanim.tif',
    '--th', '0.7',
    '--enable-fusion'
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("Hata:", result.stderr)
```

### Toplu Ä°ÅŸleme

Birden fazla dosyayÄ± iÅŸlemek iÃ§in betik:

```python
import os
from pathlib import Path
import subprocess

input_dir = Path('giris_dosyalari')
output_dir = Path('sonuclar')
output_dir.mkdir(exist_ok=True)

for tif_file in input_dir.glob('*.tif'):
    print(f"Ä°ÅŸleniyor: {tif_file.name}")
    
    subprocess.run([
        'python', 'archaeo_detect.py',
        '--input', str(tif_file),
        '--out-prefix', str(output_dir / tif_file.stem),
        '--enable-fusion',
        '--cache-derivatives',
        '-v'
    ])
    
print("TÃ¼m dosyalar iÅŸlendi!")
```

### Performans Profili

Ä°ÅŸleme sÃ¼relerini analiz etme:

```bash
python -m cProfile -o profile.stats archaeo_detect.py

# SonuÃ§larÄ± gÃ¶rÃ¼ntÃ¼le
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## ğŸ“š Teknik Detaylar

### Proje Yapisi

```text
arkeolojik_alan_tespit/
├── archaeo_detect.py                  # Ana tespit akisi
├── archeo_shared/                     # Ortak kanal ve model yardimcilari
├── egitim_verisi_olusturma.py         # Egitim verisi olusturma
├── prepare_tile_classification_dataset.py
├── training.py                        # Model egitimi
├── evaluation.py                      # Degerlendirme metrikleri
├── config.yaml                        # Paylasilan repo konfigu
├── configs/                           # Ornek profiller
├── docs/                              # Ek dokumantasyon
├── tests/                             # Testler
├── tools/                             # Bakim ve inceleme yardimcilari
└── workspace/                         # Repo-ici veri ve artifact alani
    ├── on_veri/
    ├── training_data/
    ├── training_data_classification/
    ├── checkpoints/
    ├── cache/
    ├── ciktilar/
    └── assets/
```
### Kullanilan Kutuphaneler

| KÃ¼tÃ¼phane | Versiyon | AmaÃ§ |
|-----------|----------|------|
| PyTorch | 2.0+ | Derin Ã¶ÄŸrenme Ã§erÃ§evesi |
| SMP | 0.3.2+ | Segmentasyon modelleri |
| Rasterio | 1.3+ | Raster veri I/O |
| GeoPandas | 0.12+ | VektÃ¶r veri iÅŸleme |
| OpenCV | 4.7+ | GÃ¶rÃ¼ntÃ¼ iÅŸleme |
| scikit-image | 0.20+ | GeliÅŸmiÅŸ gÃ¶rÃ¼ntÃ¼ iÅŸleme |
| RVT-py | 1.2+ (Python < 3.11) veya RVT 2.0+ (Python >= 3.11) | Kabartma gÃ¶rselleÅŸtirme |
| NumPy | 1.24+ | SayÄ±sal iÅŸlemler |
| SciPy | 1.10+ | Bilimsel hesaplama |

---

## ğŸ¤ KatkÄ±da Bulunma

Projeye katkÄ±da bulunmak iÃ§in:

1. Depoyu **fork** edin
2. Ã–zellik dalÄ± oluÅŸturun (`git checkout -b feature/yeni-ozellik`)
3. DeÄŸiÅŸikliklerinizi commit edin (`git commit -m 'Yeni Ã¶zellik: ...'`)
4. DalÄ±nÄ±zÄ± push edin (`git push origin feature/yeni-ozellik`)
5. **Pull Request** aÃ§Ä±n

### KatkÄ± AlanlarÄ±

- ğŸ› Hata dÃ¼zeltmeleri
- âœ¨ Yeni Ã¶zellikler
- ğŸ“ DokÃ¼mantasyon iyileÅŸtirmeleri
- ğŸŒ Ã‡eviriler (i18n)
- ğŸ§ª Test senaryolarÄ±
- ğŸ¨ GÃ¶rselleÅŸtirme araÃ§larÄ±

---

## ğŸ“„ Lisans

Bu proje [MIT LisansÄ±](LICENSE) altÄ±nda lisanslanmÄ±ÅŸtÄ±r.

```
MIT LisansÄ±

Telif HakkÄ± (c) 2025 [Ahmet ErtuÄŸrul ArÄ±k]

Bu yazÄ±lÄ±mÄ±n ve iliÅŸkili dokÃ¼mantasyon dosyalarÄ±nÄ±n ("YazÄ±lÄ±m") bir kopyasÄ±nÄ± 
alan herhangi bir kiÅŸiye, YazÄ±lÄ±mÄ± kÄ±sÄ±tlama olmaksÄ±zÄ±n kullanma, kopyalama, 
deÄŸiÅŸtirme, birleÅŸtirme, yayÄ±nlama, daÄŸÄ±tma, alt lisanslama ve/veya satma 
haklarÄ±nÄ± Ã¼cretsiz olarak verilir...
```

---

## ğŸ“§ Ä°letiÅŸim ve Destek

- **Sorunlar**: [GitHub Issues](https://github.com/elestirmen/archaeological-site-detection/issues)
- **E-posta**: ertugrularik@hotmail.com
- **DokÃ¼mantasyon**: [Wiki](https://github.com/elestirmen/archaeological-site-detection/wiki)

---

## ğŸ™ TeÅŸekkÃ¼rler

Bu proje aÅŸaÄŸÄ±daki aÃ§Ä±k kaynak projelerden faydalanmaktadÄ±r:

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [RVT-py](https://github.com/EarthObservation/RVT_py)
- [PyTorch](https://pytorch.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [GeoPandas](https://geopandas.org/)

---

## ğŸ“– AtÄ±f

Bu projeyi akademik Ã§alÄ±ÅŸmanÄ±zda kullanÄ±rsanÄ±z, lÃ¼tfen ÅŸu ÅŸekilde atÄ±f yapÄ±n:

```bibtex
@software{archaeological_site_detection,
  title = {Arkeolojik Alan Tespiti: Derin Ã–ÄŸrenme ve Klasik GÃ¶rÃ¼ntÃ¼ Ä°ÅŸleme},
  author = {Ahmet ErtuÄŸrul ArÄ±k},
  year = {2025},
  url = {https://github.com/elestirmen/archaeological-site-detection}
}
```

---

## ğŸ“Š Proje Ä°statistikleri

![GitHub stars](https://img.shields.io/github/stars/elestirmen/archaeological-site-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/elestirmen/archaeological-site-detection?style=social)

---

<div align="center">

GeliÅŸtirici: Ahmet ErtuÄŸrul ArÄ±k  
Son gÃ¼ncelleme: Nisan 2026

</div>



