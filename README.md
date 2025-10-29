# 🏛️ Arkeolojik Alan Tespiti (Derin Öğrenme + Klasik Görüntü İşleme)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **LiDAR ve çok bantlı uydu görüntülerinden arkeolojik yapıları otomatik tespit eden gelişmiş bir yapay zeka sistemi**

Bu proje, çok bantlı GeoTIFF verilerinden (RGB, DSM, DTM) arkeolojik izleri (tümülüs, hendek, höyük, duvar kalıntıları vb.) tespit etmek için **derin öğrenme** ve **klasik görüntü işleme** yöntemlerini birleştirir.

---

## 📑 İçindekiler

- [✨ Özellikler](#-özellikler)
- [🎯 Ne İşe Yarar?](#-ne-işe-yarar)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [📦 Kurulum](#-kurulum)
- [🎮 Kullanım](#-kullanım)
- [⚙️ Yapılandırma](#️-yapılandırma)
- [📂 Çıktı Dosyaları](#-çıktı-dosyaları)
- [🔬 Nasıl Çalışır?](#-nasıl-çalışır)
- [💡 Kullanım Senaryoları](#-kullanım-senaryoları)
- [🎨 Sonuçları Görselleştirme](#-sonuçları-görselleştirme)
- [⚡ Performans Optimizasyonu](#-performans-optimizasyonu)
- [🐛 Sorun Giderme](#-sorun-giderme)
- [❓ Sık Sorulan Sorular (SSS)](#-sık-sorulan-sorular-sss)
- [🔬 İleri Düzey Özellikler](#-i̇leri-düzey-özellikler)
- [📚 Teknik Detaylar](#-teknik-detaylar)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)
- [📄 Lisans](#-lisans)

---

## ✨ Özellikler

### 🧠 Dört Güçlü Yöntem
- **Derin Öğrenme (Deep Learning)**: U-Net, DeepLabV3+ ve diğer modern segmentasyon mimarileri
- **YOLO11 (YENİ!)**: Ultralytics YOLO11 ile hızlı nesne tespit ve segmentasyon + etiketli arazi envanteri 🏷️
  - ⚠️ **Not:** Kuş bakışı görüntüler için fine-tuning gerekir (YOLO11_NADIR_TRAINING.md)
- **Klasik Görüntü İşleme**: RVT (Relief Visualization Toolbox), Hessian matrisi, Morfolojik operatörler
- **Hibrit Fusion**: Her yöntemin güçlü yönlerini birleştiren akıllı füzyon

### 🎯 Akıllı Tespit Özellikleri
- ✅ **Çoklu Encoder Desteği**: ResNet, EfficientNet, VGG, DenseNet, MobileNet ve daha fazlası
- ✅ **Zero-Shot Öğrenme**: Eğitilmiş model olmadan bile ImageNet ağırlıkları ile çalışabilir
- ✅ **Ensemble Learning**: Birden fazla encoder'ın sonuçlarını birleştirerek daha güvenilir tespit
- ✅ **Çok Ölçekli Analiz**: Farklı boyutlardaki yapıları tespit edebilme
- ✅ **🆕 Etiketli Nesne Tespiti**: YOLO11 ile 80 farklı nesne sınıfını (ağaç, bina, araç, vb.) otomatik etiketleme

### 🔧 Teknik Özellikler
- 🚀 **Karo Tabanlı İşleme**: Büyük görüntüler için bellek verimli işleme
- 🎨 **Dikişsiz Mozaikleme**: Cosine feathering ile karo sınırlarında görüntü bozulması yok
- 📊 **Robust Normalizasyon**: Global veya lokal persentil tabanlı normalizasyon
- ⚡ **Önbellek Sistemi**: RVT hesaplamalarını önbelleğe alarak 10-100x hızlanma
- 🎯 **Akıllı Maskeleme**: Yüksek yapıları (ağaç, bina) otomatik filtreleme
- 📐 **Vektörleştirme**: Sonuçları GIS uyumlu poligonlara dönüştürme

### 🌐 GIS Entegrasyonu
- 📁 GeoPackage (.gpkg) formatında vektör çıktı
- 🗺️ Coğrafi koordinat sistemi (CRS) korunur
- 📏 Alan hesaplama ve filtreleme
- 🎯 QGIS, ArcGIS gibi yazılımlarla uyumlu

---

## 🎯 Ne İşe Yarar?

Bu sistem aşağıdaki arkeolojik özellikleri tespit edebilir:

| Yapı Tipi | Açıklama | Tespit Yöntemi |
|-----------|----------|----------------|
| 🏔️ **Tümülüsler** | Yükseltilmiş mezar höyükleri | RVT + Hessian + DL |
| 🏛️ **Höyükler** | Yerleşim höyükleri | Tüm yöntemler |
| 🧱 **Duvar Kalıntıları** | Çizgisel yapı izleri | Hessian + DL |
| ⭕ **Halka Hendekler** | Dairesel savunma yapıları | Morfolojik + DL |
| 🏰 **Kale Kalıntıları** | Büyük yapı kompleksleri | Fusion (en etkili) |
| 🏺 **Yerleşim İzleri** | Düzensiz topografik anomaliler | Klasik + DL |
| 🛤️ **Antik Yollar** | Çizgisel yükseklik değişimleri | Hessian + RVT |

---

## 🚀 Hızlı Başlangıç

### 5 Dakikada Çalıştırın!

```bash
# 1. Depoyu klonlayın
git clone https://github.com/your-username/arkeolojik_alan_tespit.git
cd arkeolojik_alan_tespit

# 2. Gerekli paketleri yükleyin
pip install -r requirements.txt

# 3. Verilerinizi hazırlayın (kesif_alani.tif adında bir GeoTIFF)
# RGB, DSM, DTM bantlarını içeren tek bir dosya olmalı

# 4. Çalıştırın!
python archaeo_detect.py
```

🎉 **Tebrikler!** Sistem çalışmaya başladı. Sonuçlar mevcut dizinde oluşturulacaktır.

---

## 📦 Kurulum

### Sistem Gereksinimleri

| Gereksinim | Minimum | Önerilen |
|------------|---------|----------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8 GB | 16 GB+ |
| **Disk Alanı** | 2 GB | 5 GB+ |
| **GPU** | Yok (CPU ile çalışır) | NVIDIA CUDA destekli GPU |

### Adım Adım Kurulum

#### 1️⃣ Python ve Pip Kontrolü

```bash
python --version  # Python 3.10 veya üstü olmalı
pip --version     # pip kurulu olmalı
```

#### 2️⃣ Sanal Ortam Oluşturma (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### 3️⃣ Gerekli Paketlerin Yüklenmesi

```bash
pip install -r requirements.txt
```

**requirements.txt içeriği:**
- `torch>=2.0.0` - PyTorch (derin öğrenme)
- `torchvision>=0.15.0` - Görüntü işleme
- `segmentation-models-pytorch>=0.3.2` - Segmentasyon modelleri
- `rasterio>=1.3.0` - Raster veri okuma/yazma
- `fiona>=1.9.0` - Vektör veri işleme
- `geopandas>=0.12.0` - Coğrafi veri analizi
- `opencv-python>=4.7.0` - Görüntü işleme
- `scikit-image>=0.20.0` - İleri düzey görüntü işleme
- `scipy>=1.10.0` - Bilimsel hesaplama
- `numpy>=1.24.0` - Sayısal işlemler
- `rvt-py>=1.2.0` - Relief Visualization Toolbox
- `pyyaml>=6.0` - YAML yapılandırma dosyaları

#### 4️⃣ GDAL Kurulumu (Opsiyonel ama Önerilen)

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

#### 5️⃣ GPU Desteği (Opsiyonel)

NVIDIA GPU'nuz varsa CUDA kurulumu:

```bash
# CUDA 11.8 için
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 için
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU kontrolü:
```python
import torch
print(torch.cuda.is_available())  # True olmalı
```

---

## 🎮 Kullanım

### Temel Kullanım

#### Varsayılan Ayarlarla Çalıştırma

```bash
python archaeo_detect.py
```

Bu komut `config.yaml` dosyasındaki ayarları kullanır ve girdi olarak tanımlanmış GeoTIFF dosyasını işler.

#### Komut Satırı Parametreleri ile Çalıştırma

```bash
# Eşik değerini değiştir
python archaeo_detect.py --th 0.7

# Karo boyutunu ayarla
python archaeo_detect.py --tile 512 --overlap 128

# Verbose modu aç (detaylı log)
python archaeo_detect.py -v

# Farklı bir girdi dosyası kullan
python archaeo_detect.py --input yeni_alan.tif

# Birden fazla parametre
python archaeo_detect.py --th 0.7 --tile 1024 --enable-fusion -v
```

### Yaygın Kullanım Örnekleri

#### 🔰 Örnek 1: İlk Defa Kullanım (Zero-Shot)

Eğitilmiş model olmadan, sadece ImageNet ağırlıkları ile:

```bash
python archaeo_detect.py \
  --encoder resnet34 \
  --zero-shot-imagenet \
  --enable-classic \
  --enable-fusion \
  -v
```

#### 🎯 Örnek 2: Sadece Klasik Yöntem (Hızlı)

GPU yoksa veya hızlı test için:

```bash
python archaeo_detect.py \
  --no-enable-deep-learning \
  --enable-classic \
  --classic-modes combo \
  --cache-derivatives
```

#### 🚀 Örnek 3: Ensemble (Çoklu Encoder)

En yüksek doğruluk için birden fazla encoder:

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

#### 🎨 Örnek 4: Özel Eğitilmiş Model ile

Kendi eğittiğiniz model ile:

```bash
python archaeo_detect.py \
  --encoder resnet50 \
  --weights models/my_trained_model.pth \
  --th 0.65 \
  --enable-classic \
  --enable-fusion \
  --alpha 0.7
```

#### 📊 Örnek 5: Büyük Alan Analizi (Optimize)

Geniş bir bölge için optimize edilmiş ayarlar:

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

### Komut Satırı Parametreleri (Tam Liste)

```bash
python archaeo_detect.py --help
```

**Önemli Parametreler:**

| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `--input` | Girdi GeoTIFF dosyası | `--input alan.tif` |
| `--th` | DL eşik değeri (0-1) | `--th 0.7` |
| `--tile` | Karo boyutu (piksel) | `--tile 1024` |
| `--overlap` | Bindirme miktarı | `--overlap 256` |
| `--encoder` | Tek encoder seçimi | `--encoder resnet34` |
| `--encoders` | Çoklu encoder modu | `--encoders all` |
| `--alpha` | Fusion ağırlığı | `--alpha 0.6` |
| `--enable-fusion` | Fusion'ı etkinleştir | (bayrak) |
| `--cache-derivatives` | Önbelleği kullan | (bayrak) |
| `-v` veya `--verbose` | Detaylı log | (bayrak) |

---

## ⚙️ Yapılandırma

### config.yaml Dosyası

Sistem davranışı `config.yaml` dosyası ile kontrol edilir. Bu dosya **çok detaylı açıklamalarla** zenginleştirilmiştir.

#### Ana Bölümler:

1. **Girdi/Çıktı**: Dosya yolları ve bant seçimi
2. **Yöntem Seçimi**: Hangi yöntemlerin kullanılacağı
3. **Derin Öğrenme**: Model mimarisi ve encoder ayarları
4. **Klasik Yöntemler**: RVT, Hessian, Morfoloji parametreleri
5. **Fusion**: Hibrit birleştirme ayarları
6. **Karo İşleme**: Bellek ve performans optimizasyonu
7. **Normalizasyon**: Veri ön işleme
8. **Maskeleme**: Yüksek yapıları filtreleme
9. **Vektörleştirme**: GIS çıktı formatı
10. **Performans**: Hız ve bellek optimizasyonu
11. **Önbellek**: Hızlandırma sistemi

#### Hızlı Yapılandırma Senaryoları:

**Senaryo 1: Sadece Derin Öğrenme**
```yaml
enable_deep_learning: true
enable_classic: false
enable_fusion: false
encoder: "resnet34"
zero_shot_imagenet: true
```

**Senaryo 2: Sadece Klasik Yöntem**
```yaml
enable_deep_learning: false
enable_classic: true
enable_fusion: false
classic_modes: "combo"
cache_derivatives: true
```

**Senaryo 3: Hibrit (En İyi Sonuç)**
```yaml
enable_deep_learning: true
enable_classic: true
enable_fusion: true
alpha: 0.5
encoders: "all"
cache_derivatives: true
```

### Veri Hazırlama

#### Girdi Dosyası Gereksinimleri:

✅ **GeoTIFF formatı** (.tif veya .tiff)  
✅ **Çok bantlı** (en az 3 bant: RGB)  
✅ **Aynı grid** (tüm bantlar aynı çözünürlük ve kapsam)  
✅ **Coğrafi referans** (CRS/EPSG kodu)

#### Önerilen Bant Yapısı:

| Bant # | İçerik | Açıklama |
|--------|--------|----------|
| 1 | Red (Kırmızı) | RGB'nin R bileşeni |
| 2 | Green (Yeşil) | RGB'nin G bileşeni |
| 3 | Blue (Mavi) | RGB'nin B bileşeni |
| 4 | DSM | Sayısal Yüzey Modeli (yükseklik) |
| 5 | DTM | Sayısal Arazi Modeli (zemin yüksekliği) |

#### Veri Oluşturma Örneği (GDAL):

```bash
# Ayrı RGB ve yükseklik dosyalarını birleştirme
gdal_merge.py -separate -o combined.tif \
  red.tif green.tif blue.tif dsm.tif dtm.tif

# Yeniden örnekleme (farklı çözünürlükleri eşitleme)
gdalwarp -tr 1.0 1.0 -r bilinear input.tif output.tif

# Koordinat sistemi atama
gdal_edit.py -a_srs EPSG:32635 output.tif
```

---

## 📂 Çıktı Dosyaları

Sistem çalıştırıldığında aşağıdaki dosyalar oluşturulur:

### 📊 Raster Çıktılar (GeoTIFF)

#### 1️⃣ Derin Öğrenme Çıktıları

**Tek Encoder:**
```
kesif_alani_prob.tif     → Olasılık haritası (0.0-1.0 arası sürekli değerler)
kesif_alani_mask.tif     → İkili maske (0: arkeolojik değil, 1: arkeolojik alan)
```

**Çoklu Encoder:**
```
kesif_alani_resnet34_prob.tif
kesif_alani_resnet34_mask.tif
kesif_alani_resnet50_prob.tif
kesif_alani_resnet50_mask.tif
kesif_alani_efficientnet-b3_prob.tif
kesif_alani_efficientnet-b3_mask.tif
```

#### 2️⃣ Klasik Yöntem Çıktıları

```
kesif_alani_classic_prob.tif     → Birleştirilmiş klasik olasılık
kesif_alani_classic_mask.tif     → Klasik ikili maske
```

**Ara Dosyalar (classic_save_intermediate: true):**
```
kesif_alani_classic_rvtlog_prob.tif    → Sadece RVT yöntemi
kesif_alani_classic_hessian_prob.tif   → Sadece Hessian yöntemi
kesif_alani_classic_morph_prob.tif     → Sadece Morfoloji yöntemi
```

#### 3️⃣ Fusion Çıktıları

```
kesif_alani_fused_resnet34_prob.tif
kesif_alani_fused_resnet34_mask.tif
```

### 📍 Vektör Çıktılar (GeoPackage)

```
kesif_alani_mask.gpkg                → DL vektör poligonlar
kesif_alani_classic_mask.gpkg        → Klasik vektör poligonlar
kesif_alani_fused_resnet34_mask.gpkg → Fusion vektör poligonlar
```

**GeoPackage Özellikleri:**
- Poligon geometrisi
- Alan bilgisi (m² cinsinden)
- CRS bilgisi korunur
- QGIS/ArcGIS'te doğrudan açılabilir

### 💾 Önbellek Dosyaları

```
kesif_alani.derivatives.npz    → RVT türevleri önbelleği
```

Bu dosya RVT hesaplamalarını saklar ve sonraki çalıştırmalarda 10-100x hızlanma sağlar.

### 📋 Dosya Adlandırma Mantığı

Çıktı dosyaları otomatik olarak şu formatta adlandırılır:

```
<prefix>_[method]_[encoder]_[params]_[type].ext
```

Örnek:
```
kesif_alani_fused_resnet34_th0.6_tile1024_alpha0.5_prob.tif
```

**Parametreler:**
- `th`: Eşik değeri
- `tile`: Karo boyutu
- `alpha`: Fusion oranı
- `minarea`: Minimum alan
- Ve diğerleri...

---

## 🔬 Nasıl Çalışır?

### İş Akışı Genel Bakış

```
┌─────────────────────┐
│  GeoTIFF Girdi      │
│ (RGB, DSM, DTM)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Veri Ön İşleme     │
│  - Bant okuma       │
│  - Normalizasyon    │
│  - Maskeleme        │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Derin   │ │ Klasik   │
│ Öğrenme │ │ Yöntemler│
│ (U-Net) │ │ (RVT)    │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           ▼
   ┌───────────────┐
   │    Fusion     │
   │  (Birleştir)  │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  Eşikleme     │
   │  (Prob → Mask)│
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ Vektörleştirme│
   │  (GeoPackage) │
   └───────────────┘
```

### 1️⃣ Derin Öğrenme Yöntemi

**Adımlar:**

1. **RVT Türevleri Hesaplama**
   - Sky-View Factor (SVF)
   - Openness (Pozitif & Negatif)
   - Local Relief Model (LRM)
   - Slope (Eğim)

2. **9 Kanallı Tensör Oluşturma**
   - 3 x RGB
   - 1 x nDSM (DSM - DTM)
   - 5 x RVT türevleri

3. **Normalizasyon**
   - Global veya lokal persentil bazlı
   - 2%-98% aralığına ölçekleme

4. **Karo Bazlı İşleme**
   - Büyük görüntü küçük karolara bölünür
   - Her karo U-Net'e verilir
   - Olasılık haritası üretilir

5. **Feathering (Yumuşatma)**
   - Karolar arası geçişler yumuşatılır
   - Dikişsiz mozaik oluşturulur

6. **Eşikleme**
   - Olasılık > eşik → Maske = 1
   - Olasılık ≤ eşik → Maske = 0

### 2️⃣ Klasik Görüntü İşleme

**Üç Alt Yöntem:**

**A) RVT (Relief Visualization)**
- SVF, Openness hesaplamaları
- Kabartı görselleştirme
- Tümülüs ve höyükler için ideal

**B) Hessian Matrisi**
- İkinci türev analizi
- Ridge (çıkıntı) ve valley (çukur) tespiti
- Duvar ve hendek izleri için etkili

**C) Morfolojik Operatörler**
- Açma (opening), kapatma (closing)
- Top-hat dönüşümleri
- Yerel doku özellikleri

**Birleştirme:**
- Her yöntem 0-1 arası skor üretir
- Skorlar ortalaması alınır (combo modu)
- Otsu veya manuel eşikleme uygulanır

### 3️⃣ Fusion (Hibrit Birleştirme)

**Formül:**
```
P_fused = α × P_deep_learning + (1 - α) × P_classic
```

**Avantajlar:**
- Derin öğrenme: Karmaşık paternler
- Klasik: Güvenilir yükseklik özellikleri
- Fusion: Her ikisinin güçlü yönleri

**Örnek:**
- α = 0.5: Eşit ağırlık
- α = 0.7: DL'ye öncelik
- α = 0.3: Klasik'e öncelik

---

## 💡 Kullanım Senaryoları

### 📍 Senaryo 1: Yeni Bir Bölge Keşfi

**Durum:** Hiç araştırılmamış bir bölgede ilk tarama

**Önerilen Ayarlar:**
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
- Çoklu encoder: Maksimum tespit hassasiyeti
- Düşük eşik: Tüm adayları yakala
- Düşük min_area: Küçük yapıları kaçırma
- Cache: Tekrar analiz için hızlandırma

### 🎯 Senaryo 2: Bilinen Alan Detaylı Analiz

**Durum:** Daha önce tespit edilmiş bir alanın detaylı incelenmesi

**Önerilen Ayarlar:**
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

**Neden bu ayarlar?**
- Özel model: Bölgeye özgü eğitilmiş model
- Yüksek eşik: Sadece güvenilir tespitler
- Simplify: Temiz poligonlar

### ⚡ Senaryo 3: Hızlı Ön Değerlendirme

**Durum:** Hızlıca bir fikir edinmek için

**Önerilen Ayarlar:**
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
- Sadece klasik: En hızlı yöntem
- Küçük karo: Daha az bellek
- Vektör yok: Zaman tasarrufu

### 🔬 Senaryo 4: Araştırma ve Karşılaştırma

**Durum:** Farklı yöntemleri karşılaştırmalı analiz

**Önerilen Ayarlar:**
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
- Tüm yöntemler aktif
- Ara dosyalar: Her yöntemin katkısını gör
- Tüm fusion: Her kombinasyonu dene

---

## 🎨 Sonuçları Görselleştirme

### QGIS'te Görüntüleme

#### 1️⃣ Olasılık Haritalarını Yükleme

```
Katman → Katman Ekle → Raster Katman
```

**Önerilen Renk Şeması:**
- 0.0-0.3: Mavi (Düşük olasılık)
- 0.3-0.5: Sarı (Orta olasılık)
- 0.5-0.7: Turuncu (Yüksek olasılık)
- 0.7-1.0: Kırmızı (Çok yüksek olasılık)

#### 2️⃣ Vektör Poligonları Görüntüleme

```
Katman → Katman Ekle → Vektör Katman → GeoPackage seç
```

**Stil Önerileri:**
- Dolgu: Yarı şeffaf kırmızı (opacity: 50%)
- Çizgi: Kalın kırmızı (2 piksel)
- Etiket: Alan değeri (m²)

#### 3️⃣ Temel Harita ile Overlay

```python
# QGIS Python Console
from qgis.core import QgsRasterLayer

# Ortofoto ekle
ortho = QgsRasterLayer('kesif_alani.tif', 'Ortofoto')
QgsProject.instance().addMapLayer(ortho)

# Maske ekle (yarı şeffaf)
mask = QgsRasterLayer('kesif_alani_mask.tif', 'Tespit')
QgsProject.instance().addMapLayer(mask)
mask.renderer().setOpacity(0.6)
```

### Python ile Görselleştirme

```python
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Olasılık haritasını oku
with rasterio.open('kesif_alani_prob.tif') as src:
    prob = src.read(1)

# Özel renk paleti
colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
cmap = LinearSegmentedColormap.from_list('archaeo', colors)

# Görselleştir
plt.figure(figsize=(12, 10))
plt.imshow(prob, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(label='Arkeolojik Alan Olasılığı')
plt.title('Arkeolojik Alan Tespit Sonuçları')
plt.xlabel('X (piksel)')
plt.ylabel('Y (piksel)')
plt.tight_layout()
plt.savefig('sonuc_gorsel.png', dpi=300)
plt.show()
```

### Web Tabanlı Görselleştirme

```python
import folium
import geopandas as gpd

# Vektörü oku
gdf = gpd.read_file('kesif_alani_mask.gpkg')

# Harita oluştur
m = folium.Map(
    location=[gdf.geometry.centroid.y.mean(), 
              gdf.geometry.centroid.x.mean()],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Poligonları ekle
for idx, row in gdf.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 2,
            'fillOpacity': 0.5
        },
        tooltip=f"Alan: {row.get('area', 0):.1f} m²"
    ).add_to(m)

# Kaydet
m.save('interaktif_harita.html')
print("Harita oluşturuldu: interaktif_harita.html")
```

---

## ⚡ Performans Optimizasyonu

### GPU Kullanımı

#### CUDA Kontrolü
```python
import torch
print(f"CUDA Kullanılabilir: {torch.cuda.is_available()}")
print(f"CUDA Versiyon: {torch.version.cuda}")
print(f"GPU Sayısı: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
```

#### GPU ile Hızlandırma
```bash
# Mixed precision (FP16) ile 2x hızlanma
python archaeo_detect.py --half

# Büyük karolarla GPU'yu doldur
python archaeo_detect.py --tile 2048 --overlap 512
```

### Bellek Optimizasyonu

#### Düşük Bellek Durumu
```bash
python archaeo_detect.py \
  --tile 512 \
  --overlap 64 \
  --no-enable-deep-learning \
  --enable-classic
```

#### Yüksek Bellek Durumu
```bash
python archaeo_detect.py \
  --tile 4096 \
  --overlap 1024 \
  --half \
  --encoders all
```

### Önbellek Stratejisi

```bash
# İlk çalıştırma: Önbellek oluştur
python archaeo_detect.py --cache-derivatives

# Sonraki çalıştırmalar: 10-100x daha hızlı!
# Önbellek otomatik kullanılır

# Parametreleri değiştirirken önbelleği yeniden hesapla
python archaeo_detect.py --recalculate-cache
```

### Paralel İşleme

Birden fazla alan varsa paralel çalıştırma:

```bash
# Bash script
for file in alan1.tif alan2.tif alan3.tif; do
  python archaeo_detect.py --input $file &
done
wait
```

### Performans Karşılaştırması

| Yapılandırma | İşlem Süresi | Bellek Kullanımı | Kalite |
|--------------|--------------|------------------|--------|
| **Minimum** (CPU, 512 tile) | ~30 dk | 4 GB | Düşük |
| **Dengeli** (GPU, 1024 tile) | ~5 dk | 8 GB | Orta |
| **Maksimum** (GPU, 2048 tile, ensemble) | ~15 dk | 16 GB | Yüksek |

*10 km² alan için tahmini süreler (1m çözünürlük)*

---

## 🐛 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### ❌ Hata 1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Çözümler:**
```bash
# Çözüm 1: Karo boyutunu küçült
python archaeo_detect.py --tile 512

# Çözüm 2: Mixed precision kullan
python archaeo_detect.py --half

# Çözüm 3: CPU kullan
python archaeo_detect.py --device cpu
```

#### ❌ Hata 2: RVT Import Hatası

```
ModuleNotFoundError: No module named 'rvt'
```

**Çözüm:**
```bash
# Python 3.10
pip install rvt-py

# Python 3.11+
pip install rvt

# Veya conda
conda install -c conda-forge rvt
```

#### ❌ Hata 3: Boş Çıktı

```
Warning: No detections found
```

**Çözümler:**
1. Eşik değerini düşür:
   ```bash
   python archaeo_detect.py --th 0.3 --classic-th 0.3
   ```

2. Minimum alanı düşür:
   ```bash
   python archaeo_detect.py --min-area 20
   ```

3. Verbose modda kontrol et:
   ```bash
   python archaeo_detect.py -v
   ```

#### ❌ Hata 4: Klasik Yöntem Çalışmıyor

```
Error: DTM band not found
```

**Çözüm:**
`config.yaml` dosyasında bantları kontrol edin:
```yaml
bands: "1,2,3,4,5"  # 5. bant DTM olmalı
# DTM yoksa:
bands: "1,2,3,4,0"  # DTM yerine 0 kullanın
```

#### ❌ Hata 5: Karo Sınırlarında Çizgiler

**Çözüm:**
```bash
# Overlap artır ve feathering etkinleştir
python archaeo_detect.py --overlap 512 --feather
```

### Debug Modu

Detaylı hata ayıklama için:

```bash
python archaeo_detect.py --verbose 2 2>&1 | tee debug_log.txt
```

Bu komut tüm debug mesajlarını hem ekrana hem de `debug_log.txt` dosyasına yazar.

---

## ❓ Sık Sorulan Sorular (SSS)

### 🤔 Genel Sorular

**S: Eğitilmiş bir modelim yok, yine de kullanabilir miyim?**  
C: Evet! `zero_shot_imagenet: true` ayarı ile ImageNet ağırlıklarını kullanabilirsiniz. Ayrıca klasik yöntemler model gerektirmez.

**S: GPU'um yok, CPU ile çalışır mı?**  
C: Evet, ancak daha yavaş olur. Klasik yöntemleri tercih edin veya küçük karo boyutu kullanın.

**S: Hangi yöntem en iyi sonucu verir?**  
C: Genellikle **fusion** (DL + Klasik) en iyi sonuçları verir. Ancak veri kalitenize ve bölgenize göre değişir.

**S: Uydu görüntüleri ile çalışır mı?**  
C: Evet, uydu görüntüleri ve LiDAR verileri desteklenir. Önemli olan çok bantlı GeoTIFF formatında olması.

### 🔧 Teknik Sorular

**S: Kaç bant gerekli?**  
C: Minimum 3 bant (RGB). Optimum 5 bant (RGB + DSM + DTM). 9 kanal RVT türevleri ile otomatik oluşturulur.

**S: Önbellek dosyası ne kadar yer kaplar?**  
C: Genellikle 10-50 MB. Girdi dosyası boyutuna bağlıdır.

**S: Sonuçları nasıl iyileştirebilirim?**  
C: 
1. Çoklu encoder kullanın (ensemble)
2. Fusion'ı etkinleştirin
3. Eşik değerlerini optimize edin
4. Yüksek kaliteli veri kullanın

**S: Kendi modelimi nasıl eğitirim?**  
C: PyTorch ve segmentation_models_pytorch kullanarak eğitebilirsiniz. Eğitilmiş modeli `--weights` parametresi ile kullanın.

### 📊 Veri Soruları

**S: Minimum alan çözünürlüğü nedir?**  
C: Önerilen: 0.5-2 metre/piksel. Daha düşük çözünürlükte küçük yapılar tespit edilemeyebilir.

**S: Maksimum dosya boyutu var mı?**  
C: Hayır, karo sistemi sayesinde çok büyük dosyalar işlenebilir. Test edilmiş: 50 GB+

**S: Farklı CRS'ler destekleniyor mu?**  
C: Evet, girdi CRS'i korunur ve çıktıya aktarılır.

---

## 🔬 İleri Düzey Özellikler

### Özel Model Eğitimi

Kendi verilerinizle model eğitimi:

```python
import torch
import segmentation_models_pytorch as smp

# Model oluştur
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=9,  # RGB + nDSM + 5 RVT türevi
    classes=1,      # Binary segmentation
    activation='sigmoid'
)

# Eğitim döngüsü
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        images, masks = batch
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Modeli kaydet
torch.save(model.state_dict(), 'my_trained_model.pth')
```

### Özel Encoder Ekleme

Yeni bir encoder eklemek için:

```python
# archaeo_detect.py içinde
SUPPORTED_ENCODERS = [
    'resnet34', 'resnet50',
    'efficientnet-b3',
    'your_custom_encoder'  # Yeni encoder ekle
]
```

### API Kullanımı

Script'i Python kodundan çağırma:

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
    print("Hata:", result.stderr)
```

### Batch İşleme

Çoklu dosya işleme scripti:

```python
import os
from pathlib import Path
import subprocess

input_dir = Path('input_files')
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

for tif_file in input_dir.glob('*.tif'):
    print(f"İşleniyor: {tif_file.name}")
    
    subprocess.run([
        'python', 'archaeo_detect.py',
        '--input', str(tif_file),
        '--out-prefix', str(output_dir / tif_file.stem),
        '--enable-fusion',
        '--cache-derivatives',
        '-v'
    ])
    
print("Tüm dosyalar işlendi!")
```

### Performans Profilleme

İşlem sürelerini analiz etme:

```bash
python -m cProfile -o profile.stats archaeo_detect.py

# Sonuçları görüntüle
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## 📚 Teknik Detaylar

### Sistem Mimarisi

```
archaeo_detect.py
├── Veri Yükleme (rasterio)
├── Ön İşleme
│   ├── Bant okuma
│   ├── RVT türevleri (rvt-py)
│   ├── nDSM hesaplama
│   └── Normalizasyon
├── Tespit
│   ├── Derin Öğrenme (PyTorch + SMP)
│   │   ├── U-Net
│   │   ├── DeepLabV3+
│   │   └── Diğer mimariler
│   ├── Klasik Yöntemler
│   │   ├── RVT (SVF, Openness, LRM)
│   │   ├── Hessian Matrisi
│   │   └── Morfoloji (scikit-image)
│   └── Fusion (Hibrit)
├── Son İşleme
│   ├── Eşikleme
│   ├── Morfological post-processing
│   └── Alan filtreleme
└── Çıktı
    ├── Raster (GeoTIFF)
    └── Vektör (GeoPackage)
```

### Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Amaç |
|-----------|----------|------|
| PyTorch | 2.0+ | Derin öğrenme framework |
| SMP | 0.3.2+ | Segmentasyon modelleri |
| Rasterio | 1.3+ | Raster veri I/O |
| GeoPandas | 0.12+ | Vektör veri işleme |
| OpenCV | 4.7+ | Görüntü işleme |
| scikit-image | 0.20+ | İleri görüntü işleme |
| RVT-py | 1.2+ | Relief visualization |
| NumPy | 1.24+ | Sayısal işlemler |
| SciPy | 1.10+ | Bilimsel hesaplama |

### Algoritma Detayları

#### RVT (Relief Visualization Toolbox)

**Sky-View Factor (SVF):**
```
SVF = (1/n) * Σ(max(0, cos(α_i)))
```
Burada `α_i` her yöndeki horizon açısıdır.

**Openness:**
```
Openness_positive = (1/n) * Σ(90° - α_i)
Openness_negative = (1/n) * Σ(α_i - 90°)
```

#### Hessian Matrisi

İkinci türev matrisi:
```
H = [∂²f/∂x²    ∂²f/∂x∂y]
    [∂²f/∂y∂x   ∂²f/∂y²]
```

Eigenvalue analizi ile ridge/valley tespiti.

#### Fusion Algoritması

```python
def fusion(p_dl, p_classic, alpha):
    """
    p_dl: Derin öğrenme olasılığı (0-1)
    p_classic: Klasik yöntem olasılığı (0-1)
    alpha: Ağırlık faktörü (0-1)
    """
    p_fused = alpha * p_dl + (1 - alpha) * p_classic
    return np.clip(p_fused, 0, 1)
```

---

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak isterseniz:

1. **Fork** edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: ...'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. **Pull Request** açın

### Katkı Alanları

- 🐛 Bug düzeltmeleri
- ✨ Yeni özellikler
- 📝 Dokümantasyon iyileştirmeleri
- 🌍 Çeviri (i18n)
- 🧪 Test senaryoları
- 🎨 Görselleştirme araçları

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📧 İletişim ve Destek

- **Issues**: [GitHub Issues](https://github.com/your-username/arkeolojik_alan_tespit/issues)
- **Email**: your.email@example.com
- **Dokümantasyon**: [Wiki](https://github.com/your-username/arkeolojik_alan_tespit/wiki)

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden yararlanmıştır:

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [RVT-py](https://github.com/EarthObservation/RVT_py)
- [PyTorch](https://pytorch.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [GeoPandas](https://geopandas.org/)

---

## 📖 Alıntı (Citation)

Bu projeyi akademik çalışmanızda kullanırsanız lütfen alıntı yapın:

```bibtex
@software{arkeolojik_alan_tespit,
  title = {Arkeolojik Alan Tespiti: Derin Öğrenme ve Klasik Görüntü İşleme},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/arkeolojik_alan_tespit}
}
```

---

## 📊 Proje İstatistikleri

![GitHub stars](https://img.shields.io/github/stars/your-username/arkeolojik_alan_tespit?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/arkeolojik_alan_tespit?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/arkeolojik_alan_tespit?style=social)

---

## 🗺️ Yol Haritası (Roadmap)

### Versiyon 2.0 (Planlanıyor)
- [ ] Web tabanlı GUI
- [ ] Real-time processing
- [ ] 3D görselleştirme
- [ ] Otomatik model eğitimi
- [ ] Cloud processing desteği

### Versiyon 1.5 (Yakında)
- [ ] Docker container
- [ ] REST API
- [ ] Batch processing GUI
- [ ] Çoklu dil desteği

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

Geliştirici: [Your Name]  
Son Güncelleme: Ekim 2025

</div>
