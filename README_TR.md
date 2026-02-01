# 🏛️ Arkeolojik Alan Tespiti (Derin Öğrenme + Klasik Görüntü İşleme)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **LiDAR ve çok bantlı uydu görüntülerinden arkeolojik yapıların otomatik tespiti için gelişmiş yapay zeka sistemi**

Bu proje, çok bantlı GeoTIFF verilerinden (RGB, DSM, DTM) arkeolojik izleri (tümülüs, hendek, höyük, duvar kalıntıları vb.) tespit etmek için **derin öğrenme** ve **klasik görüntü işleme** yöntemlerini birleştirir.

---

## 📑 İçindekiler

- [✨ Özellikler](#-özellikler)
- [🎯 Ne Yapar](#-ne-yapar)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [📦 Kurulum](#-kurulum)
- [🎮 Kullanım](#-kullanım)
- [⚙️ Yapılandırma](#️-yapılandırma)
- [📂 Çıktı Dosyaları](#-çıktı-dosyaları)
- [🔬 Nasıl Çalışır](#-nasıl-çalışır)
- [💡 Kullanım Senaryoları](#-kullanım-senaryoları)
- [🎨 Görselleştirme](#-görselleştirme)
- [⚡ Performans Optimizasyonu](#-performans-optimizasyonu)
- [🐛 Sorun Giderme](#-sorun-giderme)
- [❓ SSS](#-sss)
- [🎓 Model Eğitimi Kılavuzu](#-model-eğitimi-kılavuzu)
- [🔬 Gelişmiş Özellikler](#-gelişmiş-özellikler)
- [📚 Teknik Detaylar](#-teknik-detaylar)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)
- [📄 Lisans](#-lisans)

---

## ✨ Özellikler

### 🧠 Dört Güçlü Yöntem
- **Derin Öğrenme**: U-Net, DeepLabV3+ ve diğer modern segmentasyon mimarileri
- **YOLO11 (YENİ!)**: Ultralytics YOLO11 ile hızlı nesne tespiti ve segmentasyon + etiketli arazi envanteri 🏷️
  - ⚠️ **Not:** Nadir (kuşbakışı) görüntüler için ince ayar gereklidir (bkz. YOLO11_NADIR_TRAINING.md)
- **Klasik Görüntü İşleme**: RVT (Kabartma Görselleştirme Araç Kutusu), Hessian matrisi, Morfolojik operatörler
- **Hibrit Füzyon**: Her yöntemin güçlü yönlerini birleştiren akıllı füzyon

### 🎯 Akıllı Tespit Özellikleri
- ✅ **Çoklu Kodlayıcı Desteği**: ResNet, EfficientNet, VGG, DenseNet, MobileNet ve daha fazlası
- ✅ **Sıfır Atış Öğrenme**: ImageNet ağırlıklarını kullanarak eğitilmiş modeller olmadan bile çalışır
- ✅ **Topluluk Öğrenme**: Daha güvenilir tespit için birden fazla kodlayıcının sonuçlarını birleştirir
- ✅ **Çok Ölçekli Analiz**: Farklı boyutlardaki yapıları tespit eder
- ✅ **🆕 Etiketli Nesne Tespiti**: YOLO11 ile 80 farklı nesne sınıfının otomatik etiketlenmesi (ağaçlar, binalar, araçlar vb.)
- ✅ **🆕 12 Kanallı Giriş**: Gelişmiş tespit için Eğrilik ve TPI dahil ileri düzey topografik özellikler
- ✅ **🆕 CBAM Dikkat**: Dinamik özellik ağırlıklandırma için kanal ve uzamsal dikkat mekanizması

### 🔧 Teknik Özellikler
- 🚀 **Karo Tabanlı İşleme**: Büyük görüntüler için bellek verimli işleme
- 🎨 **Sorunsuz Mozaikleme**: Kosinüs yumuşatma ile karo sınırlarında artefakt yok
- 📊 **Sağlam Normalizasyon**: Global veya yerel yüzdelik tabanlı normalizasyon
- ⚡ **Önbellek Sistemi**: RVT hesaplamalarını önbelleğe alarak 10-100x hızlanma
- 🎯 **Akıllı Maskeleme**: Yüksek yapıların (ağaçlar, binalar) otomatik filtrelenmesi
- 📐 **Vektörleştirme**: Sonuçları CBS uyumlu çokgenlere dönüştürür

### 🌐 CBS Entegrasyonu
- 📁 GeoPackage (.gpkg) formatında vektör çıktısı
- 🗺️ Coğrafi koordinat sistemi (CRS) korunur
- 📏 Alan hesaplama ve filtreleme
- 🎯 QGIS, ArcGIS ve benzeri yazılımlarla uyumlu

---

## 🎯 Ne Yapar

Bu sistem aşağıdaki arkeolojik özellikleri tespit edebilir:

| Yapı Tipi | Açıklama | Tespit Yöntemi |
|-----------|----------|----------------|
| 🏔️ **Tümülüsler** | Yükseltilmiş mezar höyükleri | RVT + Hessian + DL |
| 🏛️ **Höyükler** | Yerleşim höyükleri | Tüm yöntemler |
| 🧱 **Duvar Kalıntıları** | Doğrusal yapı izleri | Hessian + DL |
| ⭕ **Halka Hendekler** | Dairesel savunma yapıları | Morfolojik + DL |
| 🏰 **Kale Kalıntıları** | Büyük yapı kompleksleri | Füzyon (en etkili) |
| 🏺 **Yerleşim İzleri** | Düzensiz topografik anomaliler | Klasik + DL |
| 🛤️ **Antik Yollar** | Doğrusal yükseklik değişimleri | Hessian + RVT |

---

## 🚀 Hızlı Başlangıç

### 5 Dakikada Çalıştırın!

```bash
# 1. Depoyu klonlayın
git clone https://github.com/your-username/archaeological-site-detection.git
cd archaeological-site-detection

# 2. Gerekli paketleri yükleyin
pip install -r requirements.txt

# 3. Verilerinizi hazırlayın (kesif_alani.tif adında bir GeoTIFF)
# Tek bir dosyada RGB, DSM, DTM bantları içermelidir

# 4. Çalıştırın!
python archaeo_detect.py
```

🎉 **Tebrikler!** Sistem başladı. Sonuçlar mevcut dizinde oluşturulacak.

### 🎓 Kendi Modelinizi Eğitme (İsteğe Bağlı)

Etiketli verileriniz (ground truth maskeleri) varsa, özel bir model eğitebilirsiniz:

```bash
# Adım 1: GeoTIFF + ground truth maskesinden eğitim verisi oluşturun
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data

# Veya interaktif modu kullanın (argüman gerekmez):
python egitim_verisi_olusturma.py
# Dosya yollarını girmek için yönergeleri izleyin

# Adım 2: Modeli eğitin
python training.py --data training_data --epochs 50

# Adım 3: Eğitilmiş modelinizi kullanın
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth
```

**💡 İpucu:** Eğitim verisi oluşturma betiği (`egitim_verisi_olusturma.py`) interaktif modu destekler. Argüman olmadan çalıştırırsanız, adım adım size rehberlik eder.

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
pip --version     # pip yüklü olmalı
```

#### 2️⃣ Sanal Ortam Oluşturma (Önerilir)

```bash
# Windows
python -m venv .venv310
.venv310\Scripts\activate

# Linux/Mac
python -m venv .venv310
source .venv310/bin/activate
```

**Not:** Proje `.venv310` dizini içerir. Proje klasörünü taşırsanız, `.venv310/Scripts/activate.bat` ve `.venv310/Scripts/activate` dosyalarındaki sanal ortam yollarını güncellediğinizden emin olun.

#### 3️⃣ Gerekli Paketleri Yükleme

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
- `scikit-image>=0.20.0` - Gelişmiş görüntü işleme
- `scipy>=1.10.0` - Bilimsel hesaplama
- `numpy>=1.24.0` - Sayısal işlemler
- `rvt-py>=1.2.0` (Python < 3.11) veya `rvt>=2.0.0` (Python >= 3.11) - Kabartma Görselleştirme Araç Kutusu
- `pyyaml>=6.0` - YAML yapılandırma dosyaları

#### 4️⃣ GDAL Kurulumu (İsteğe Bağlı ama Önerilir)

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

#### 5️⃣ GPU Desteği (İsteğe Bağlı)

NVIDIA GPU'nuz varsa, CUDA'yı yükleyin:

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

Bu komut `config.yaml` dosyasındaki ayarları kullanır ve giriş olarak tanımlanan GeoTIFF dosyasını işler.

#### Komut Satırı Parametreleriyle Çalıştırma

```bash
# Eşik değerini değiştirme
python archaeo_detect.py --th 0.7

# Karo boyutunu ayarlama
python archaeo_detect.py --tile 512 --overlap 128

# Ayrıntılı modu etkinleştirme (detaylı log)
python archaeo_detect.py -v

# Farklı bir giriş dosyası kullanma
python archaeo_detect.py --input yeni_alan.tif

# Birden fazla parametre
python archaeo_detect.py --th 0.7 --tile 1024 --enable-fusion -v
```

### Yaygın Kullanım Örnekleri

#### 🔰 Örnek 1: İlk Kez Kullanım (Sıfır Atış)

Eğitilmiş modeller olmadan, sadece ImageNet ağırlıklarını kullanarak:

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

#### 🚀 Örnek 3: Topluluk (Çoklu Kodlayıcı)

Birden fazla kodlayıcı ile en yüksek doğruluk için:

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

#### 🎨 Örnek 4: Özel Eğitilmiş Modelle

Kendi eğitilmiş modelinizle:

```bash
python archaeo_detect.py \
  --encoder resnet50 \
  --weights models/benim_egitilmis_modelim.pth \
  --th 0.65 \
  --enable-classic \
  --enable-fusion \
  --alpha 0.7
```

#### 📊 Örnek 5: Geniş Alan Analizi (Optimize Edilmiş)

Geniş bir alan için optimize edilmiş ayarlar:

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
| `--input` | Giriş GeoTIFF dosyası | `--input alan.tif` |
| `--th` | DL eşiği (0-1) | `--th 0.7` |
| `--tile` | Karo boyutu (piksel) | `--tile 1024` |
| `--overlap` | Örtüşme miktarı | `--overlap 256` |
| `--encoder` | Tek kodlayıcı seçimi | `--encoder resnet34` |
| `--encoders` | Çoklu kodlayıcı modu | `--encoders all` |
| `--alpha` | Füzyon ağırlığı | `--alpha 0.6` |
| `--enable-fusion` | Füzyonu etkinleştir | (bayrak) |
| `--cache-derivatives` | Önbellek kullan | (bayrak) |
| `-v` veya `--verbose` | Detaylı log | (bayrak) |

---

## ⚙️ Yapılandırma

### config.yaml Dosyası

Sistem davranışı `config.yaml` dosyası tarafından kontrol edilir. Bu dosya detaylı açıklamalarla **zengin bir şekilde belgelenmiştir**.

#### Ana Bölümler:

1. **Giriş/Çıkış**: Dosya yolları ve bant seçimi
2. **Yöntem Seçimi**: Hangi yöntemlerin kullanılacağı
3. **Derin Öğrenme**: Model mimarisi ve kodlayıcı ayarları
4. **Klasik Yöntemler**: RVT, Hessian, Morfoloji parametreleri
5. **Füzyon**: Hibrit kombinasyon ayarları
6. **Karo İşleme**: Bellek ve performans optimizasyonu
7. **Normalizasyon**: Veri ön işleme
8. **Maskeleme**: Yüksek yapıları filtreleme
9. **Vektörleştirme**: CBS çıktı formatı
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

**Senaryo 3: Hibrit (En İyi Sonuçlar)**
```yaml
enable_deep_learning: true
enable_classic: true
enable_fusion: true
alpha: 0.5
encoders: "all"
cache_derivatives: true
```

### Veri Hazırlama

#### Giriş Dosyası Gereksinimleri:

✅ **GeoTIFF formatı** (.tif veya .tiff)  
✅ **Çok bantlı** (en az 3 bant: RGB)  
✅ **Aynı grid** (tüm bantlar aynı çözünürlük ve kapsam)  
✅ **Coğrafi referans** (CRS/EPSG kodu)

#### Önerilen Bant Yapısı:

| Bant # | İçerik | Açıklama |
|--------|--------|----------|
| 1 | Kırmızı | RGB'nin R bileşeni |
| 2 | Yeşil | RGB'nin G bileşeni |
| 3 | Mavi | RGB'nin B bileşeni |
| 4 | DSM | Sayısal Yüzey Modeli (yükseklik) |
| 5 | DTM | Sayısal Arazi Modeli (zemin yüksekliği) |

#### Veri Oluşturma Örneği (GDAL):

```bash
# Ayrı RGB ve yükseklik dosyalarını birleştirme
gdal_merge.py -separate -o birlesik.tif \
  kirmizi.tif yesil.tif mavi.tif dsm.tif dtm.tif

# Yeniden örnekleme (farklı çözünürlükleri eşitleme)
gdalwarp -tr 1.0 1.0 -r bilinear giris.tif cikis.tif

# Koordinat sistemi atama
gdal_edit.py -a_srs EPSG:32635 cikis.tif
```

---

## 📂 Çıktı Dosyaları

Sistem çalıştığında aşağıdaki dosyalar oluşturulur:

### 📊 Raster Çıktılar (GeoTIFF)

#### 1️⃣ Derin Öğrenme Çıktıları

**Tek Kodlayıcı:**
```
kesif_alani_prob.tif     → Olasılık haritası (sürekli değerler 0.0-1.0)
kesif_alani_mask.tif     → İkili maske (0: arkeolojik değil, 1: arkeolojik alan)
```

**Çoklu Kodlayıcı:**
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
kesif_alani_classic_prob.tif     → Birleşik klasik olasılık
kesif_alani_classic_mask.tif     → Klasik ikili maske
```

**Ara Dosyalar (classic_save_intermediate: true):**
```
kesif_alani_classic_rvtlog_prob.tif    → Sadece RVT yöntemi
kesif_alani_classic_hessian_prob.tif   → Sadece Hessian yöntemi
kesif_alani_classic_morph_prob.tif     → Sadece Morfoloji yöntemi
```

#### 3️⃣ Füzyon Çıktıları

```
kesif_alani_fused_resnet34_prob.tif
kesif_alani_fused_resnet34_mask.tif
```

### 📍 Vektör Çıktılar (GeoPackage)

```
kesif_alani_mask.gpkg                → DL vektör çokgenleri
kesif_alani_classic_mask.gpkg        → Klasik vektör çokgenleri
kesif_alani_fused_resnet34_mask.gpkg → Füzyon vektör çokgenleri
```

**GeoPackage Özellikleri:**
- Çokgen geometrisi
- Alan bilgisi (m² cinsinden)
- CRS bilgisi korunur
- QGIS/ArcGIS'te doğrudan açılabilir

### 💾 Önbellek Dosyaları

**Önbellek Dizin Yapısı:**
```
cache/
├── kesif_alani.derivatives.npz    → RVT türevleri önbelleği
└── karlik_vadi.derivatives.npz   → RVT türevleri önbelleği
```

**Önbellek Sistemi:**
- RVT hesaplamaları `.npz` formatında önbelleğe alınır
- Önbellek dosyaları `cache/` dizininde saklanır (config.yaml'daki `cache_dir` ile yapılandırılabilir)
- Önbellek doğrulaması dosya adı ve değişiklik zamanını kontrol eder
- **Önemli:** Proje klasörü taşınsa bile önbellek dosyaları yeniden kullanılabilir (dosya adı tabanlı doğrulama)
- Sonraki çalıştırmalarda 10-100x hızlanma sağlar
- Önbellek dosyaları tipik olarak 10-50 MB'dır, ancak yüksek çözünürlüklü veriler için daha büyük olabilir

**Önbellek Yapılandırması:**
```yaml
cache_derivatives: true      # Önbelleği etkinleştir
cache_dir: "cache/"          # Önbellek dizini (proje köküne göre)
recalculate_cache: false     # Önbellek varsa yeniden hesaplama
```

### 📋 Dosya Adlandırma Mantığı

Çıktı dosyaları aşağıdaki formatta otomatik olarak adlandırılır:

```
<önek>_[yöntem]_[kodlayıcı]_[parametreler]_[tip].ext
```

Örnek:
```
kesif_alani_fused_resnet34_th0.6_tile1024_alpha0.5_prob.tif
```

**Parametreler:**
- `th`: Eşik değeri
- `tile`: Karo boyutu
- `alpha`: Füzyon oranı
- `minarea`: Minimum alan
- Ve diğerleri...

---

## 🔬 Nasıl Çalışır

### İş Akışı Genel Bakış

```
┌─────────────────────┐
│  GeoTIFF Girişi     │
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
   │    Füzyon     │
   │  (Birleştir)  │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  Eşikleme     │
   │  (Olas → Mask)│
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
   - Gökyüzü Görünürlük Faktörü (SVF)
   - Açıklık (Pozitif & Negatif)
   - Yerel Kabartma Modeli (LRM)
   - Eğim

2. **12 Kanallı Tensör Oluşturma** (Güncellenmiş!)
   - 3 x RGB
   - 1 x nDSM (DSM - DTM)
   - 5 x RVT türevleri (SVF, Poz/Neg Açıklık, LRM, Eğim)
   - 2 x Eğrilik (Plan + Profil) - YENİ!
   - 1 x TPI (Topografik Konum İndeksi) - YENİ!

3. **Normalizasyon**
   - Global veya yerel yüzdelik tabanlı
   - %2-%98 aralığına ölçekleme

4. **Karo Tabanlı İşleme**
   - Büyük görüntü küçük karolara bölünür
   - Her karo U-Net'e beslenir
   - Olasılık haritası oluşturulur

5. **Yumuşatma (Feathering)**
   - Karolar arasındaki geçişler yumuşatılır
   - Sorunsuz mozaik oluşturulur

6. **Eşikleme**
   - Olasılık > eşik → Maske = 1
   - Olasılık ≤ eşik → Maske = 0

### 2️⃣ Klasik Görüntü İşleme

**Üç Alt Yöntem:**

**A) RVT (Kabartma Görselleştirme)**
- SVF, Açıklık hesaplamaları
- Kabartma görselleştirme
- Tümülüs ve höyükler için ideal

**B) Hessian Matrisi**
- İkinci türev analizi
- Sırt ve vadi tespiti
- Duvarlar ve hendekler için etkili

**C) Morfolojik Operatörler**
- Açma, kapama
- Üst-şapka dönüşümleri
- Yerel doku özellikleri

**Kombinasyon:**
- Her yöntem 0-1 puan üretir
- Puanlar ortalaması alınır (combo modu)
- Otsu veya manuel eşikleme uygulanır

### 3️⃣ Füzyon (Hibrit Kombinasyon)

**Formül:**
```
P_fused = α × P_derin_öğrenme + (1 - α) × P_klasik
```

**Avantajlar:**
- Derin öğrenme: Karmaşık desenler
- Klasik: Güvenilir yükseklik özellikleri
- Füzyon: Her ikisinin güçlü yönleri

**Örnek:**
- α = 0.5: Eşit ağırlık
- α = 0.7: DL'ye öncelik
- α = 0.3: Klasiğe öncelik

---

## 💡 Kullanım Senaryoları

### 📍 Senaryo 1: Yeni Alan Keşfi

**Durum:** Keşfedilmemiş bir alanın ilk taraması

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
- Çoklu kodlayıcı: Maksimum tespit hassasiyeti
- Düşük eşik: Tüm adayları yakala
- Düşük min_area: Küçük yapıları kaçırma
- Önbellek: Tekrarlanan analiz için hızlanma

### 🎯 Senaryo 2: Bilinen Alanın Detaylı Analizi

**Durum:** Daha önce tespit edilen bir alanın detaylı incelemesi

**Önerilen Ayarlar:**
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
- Özel model: Bölgeye özgü eğitilmiş model
- Yüksek eşik: Sadece güvenilir tespitler
- Simplify: Temiz çokgenler

### ⚡ Senaryo 3: Hızlı Ön Değerlendirme

**Durum:** Hızlıca fikir edinmek için

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
- Küçük karolar: Daha az bellek
- Vektör yok: Zaman tasarrufu

### 🔬 Senaryo 4: Araştırma ve Karşılaştırma

**Durum:** Farklı yöntemlerin karşılaştırmalı analizi

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
- Ara dosyalar: Her yöntemin katkısını görme
- Tüm füzyon: Her kombinasyonu deneme

---

## 🎨 Görselleştirme

### QGIS'te Görüntüleme

#### 1️⃣ Olasılık Haritalarını Yükleme

```
Katman → Katman Ekle → Raster Katman Ekle
```

**Önerilen Renk Şeması:**
- 0.0-0.3: Mavi (Düşük olasılık)
- 0.3-0.5: Sarı (Orta olasılık)
- 0.5-0.7: Turuncu (Yüksek olasılık)
- 0.7-1.0: Kırmızı (Çok yüksek olasılık)

#### 2️⃣ Vektör Çokgenleri Görüntüleme

```
Katman → Katman Ekle → Vektör Katman Ekle → GeoPackage Seç
```

**Stil Önerileri:**
- Dolgu: Yarı saydam kırmızı (opaklık: %50)
- Çizgi: Kalın kırmızı (2 piksel)
- Etiket: Alan değeri (m²)

#### 3️⃣ Temel Haritayla Bindirme

```python
# QGIS Python Konsolu
from qgis.core import QgsRasterLayer

# Ortofoto ekle
ortho = QgsRasterLayer('kesif_alani.tif', 'Ortofoto')
QgsProject.instance().addMapLayer(ortho)

# Maske ekle (yarı saydam)
mask = QgsRasterLayer('kesif_alani_mask.tif', 'Tespit')
QgsProject.instance().addMapLayer(mask)
mask.renderer().setOpacity(0.6)
```

### Python Görselleştirme

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
plt.savefig('sonuc_gorsellestirme.png', dpi=300)
plt.show()
```

### Web Tabanlı Görselleştirme

```python
import folium
import geopandas as gpd

# Vektör oku
gdf = gpd.read_file('kesif_alani_mask.gpkg')

# Harita oluştur
m = folium.Map(
    location=[gdf.geometry.centroid.y.mean(), 
              gdf.geometry.centroid.x.mean()],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Çokgenleri ekle
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
print(f"CUDA Versiyonu: {torch.version.cuda}")
print(f"GPU Sayısı: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
```

#### GPU Hızlandırma
```bash
# Karma hassasiyet (FP16) ile 2x hızlanma
python archaeo_detect.py --half

# GPU'yu büyük karolarla doldur
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
# Önbellek otomatik olarak kullanılır

# Parametreler değiştiğinde önbelleği yeniden hesapla
python archaeo_detect.py --recalculate-cache
```

**Önbellek Faydaları:**
- RVT türevleri bir kez hesaplanır ve önbelleğe alınır
- Önbellek dosyaları `cache/` dizininde saklanır
- Önbellek doğrulaması esnektir: proje klasörü taşınsa bile çalışır
- Dosya adı ve değişiklik zamanı doğrulama için kontrol edilir
- Tekrarlanan çalıştırmalarda önemli zaman tasarrufu

### Paralel İşleme

Birden fazla alan için paralel çalıştırma:

```bash
# Bash betiği
for file in alan1.tif alan2.tif alan3.tif; do
  python archaeo_detect.py --input $file &
done
wait
```

### Performans Karşılaştırması

| Yapılandırma | İşleme Süresi | Bellek Kullanımı | Kalite |
|--------------|---------------|------------------|--------|
| **Minimum** (CPU, 512 karo) | ~30 dk | 4 GB | Düşük |
| **Dengeli** (GPU, 1024 karo) | ~5 dk | 8 GB | Orta |
| **Maksimum** (GPU, 2048 karo, topluluk) | ~15 dk | 16 GB | Yüksek |

*10 km² alan için tahmini süreler (1m çözünürlük)*

---

## 🐛 Sorun Giderme

### Yaygın Hatalar ve Çözümler

#### ❌ Hata 1: CUDA Bellek Yetersizliği

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Çözümler:**
```bash
# Çözüm 1: Karo boyutunu küçült
python archaeo_detect.py --tile 512

# Çözüm 2: Karma hassasiyet kullan
python archaeo_detect.py --half

# Çözüm 3: CPU kullan
python archaeo_detect.py --device cpu
```

#### ❌ Hata 2: RVT İçe Aktarma Hatası

```
ModuleNotFoundError: No module named 'rvt'
```

**Çözüm:**
```bash
# Python 3.10
pip install rvt-py

# Python 3.11+
pip install rvt

# Veya conda ile
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

3. Ayrıntılı modda kontrol et:
   ```bash
   python archaeo_detect.py -v
   ```

#### ❌ Hata 4: Klasik Yöntem Çalışmıyor

```
Error: DTM band not found
```

**Çözüm:**
`config.yaml`'da bantları kontrol edin:
```yaml
bands: "1,2,3,4,5"  # Bant 5 DTM olmalı
# DTM yoksa:
bands: "1,2,3,4,0"  # DTM yerine 0 kullan
```

#### ❌ Hata 5: Karo Sınırlarında Çizgiler

**Çözüm:**
```bash
# Örtüşmeyi artır ve yumuşatmayı etkinleştir
python archaeo_detect.py --overlap 512 --feather
```

#### ❌ Hata 6: Önbellek Kullanılmıyor

**Belirtiler:** Önbellek dosyaları varken bile sistem RVT türevlerini yeniden hesaplıyor

**Çözümler:**
1. `config.yaml`'da önbellek dizini yolunu kontrol edin:
   ```yaml
   cache_dir: "cache/"  # Önbellek dizininizle eşleşmeli
   ```

2. Önbellek dosya adlandırmasını doğrulayın:
   - NPZ önbellek (küçük/orta rasterlar için varsayılan): `<giriş_adı>.derivatives.npz`
   - Raster önbellek (blok tabanlı; çok büyük rasterlar veya `cache_derivatives_mode: "raster"` ile otomatik kullanılır):
     - `<giriş_adı>.derivatives_raster.tif`
     - `<giriş_adı>.derivatives_raster.json`
   - `kesif_alani.tif` girişi için örnek:
     - `kesif_alani.derivatives.npz`
     - `kesif_alani.derivatives_raster.tif`

3. Önbellek doğrulamasını kontrol edin:
   - Önbellek doğrulaması dosya adı ve değişiklik zamanını kontrol eder
   - Giriş dosyası taşınmışsa, önbellek yine de çalışmalıdır (dosya adı tabanlı doğrulama)
   - Giriş dosyası değiştirilmişse, önbellek yeniden hesaplanır

4. Önbellek durumunu görmek için ayrıntılı modu etkinleştirin:
   ```bash
   python archaeo_detect.py --cache-derivatives -v
   ```

#### ❌ Hata 7: Eğitim Betiği İçe Aktarma Hataları

**Belirtiler:**
```
HATA: segmentation-models-pytorch kurulu değil!
HATA: archaeo_detect.py'den attention modülleri import edilemedi.
```

**Çözümler:**
1. **Eksik paketleri yükleyin**:
   ```bash
   pip install segmentation-models-pytorch
   ```

2. **Python yolunu kontrol edin**: `archaeo_detect.py`'nin aynı dizinde veya Python yolunda olduğundan emin olun

3. **Kurulumu doğrulayın**: `python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"` çalıştırın

#### ❌ Hata 8: Eğitim Verisi Format Uyumsuzluğu

**Belirtiler:**
```
ValueError: Expected 12 channels but got 9
```

**Çözümler:**
1. **Eğitim verisini yeniden oluşturun**: `egitim_verisi_olusturma.py`'yi doğru parametrelerle kullanın
2. **metadata.json'u kontrol edin**: `num_channels`'ın gerçek veriyle eşleştiğini doğrulayın
3. **Dosya formatını doğrulayın**: `.npz` dosyalarının `(12, H, W)` şeklinde `image` anahtarı içerdiğinden emin olun

### Hata Ayıklama Modu

Detaylı hata ayıklama için:

```bash
python archaeo_detect.py --verbose 2 2>&1 | tee debug_log.txt
```

Bu komut tüm hata ayıklama mesajlarını hem ekrana hem de `debug_log.txt` dosyasına yazar.

### Eğitim Betiği Hata Ayıklama

**Eğitim verisini kontrol edin:**
```bash
# Eğitim verisi yapısını doğrulayın
ls -R training_data/
# Göstermelidir: train/images/, train/masks/, val/images/, val/masks/

# Metadata'yı kontrol edin
cat training_data/metadata.json | python -m json.tool
```

**Veri yüklemesini test edin:**
```python
# Hızlı test betiği
import numpy as np
from pathlib import Path

data_dir = Path("training_data")
train_images = list((data_dir / "train" / "images").glob("*.npz"))
if train_images:
    sample = np.load(train_images[0])
    print(f"Anahtarlar: {sample.files}")
    if 'image' in sample.files:
        img = sample['image']
        print(f"Görüntü şekli: {img.shape}")
        print(f"Beklenen: (12, 256, 256), Alınan: {img.shape}")
```

**Eğitimi gerçek zamanlı izleyin:**
```bash
# Eğitim geçmişi dosyasını izleyin
watch -n 5 'tail -20 checkpoints/training_history.json'
```

---

## ❓ SSS

### 🤔 Genel Sorular

**S: Eğitilmiş modelim yok, yine de kullanabilir miyim?**  
C: Evet! ImageNet ağırlıklarını kullanmak için `zero_shot_imagenet: true` kullanın. Ayrıca, klasik yöntemler model gerektirmez.

**S: GPU'm yok, CPU ile çalışır mı?**  
C: Evet, ama daha yavaş olacaktır. Klasik yöntemleri tercih edin veya küçük karo boyutu kullanın.

**S: Hangi yöntem en iyi sonuçları verir?**  
C: Genellikle **füzyon** (DL + Klasik) en iyi sonuçları verir. Ancak, veri kalitenize ve bölgenize göre değişir.

**S: Uydu görüntüleriyle çalışır mı?**  
C: Evet, uydu görüntüleri ve LiDAR verileri desteklenir. Önemli olan çok bantlı GeoTIFF formatında olmasıdır.

### 🔧 Teknik Sorular

**S: Kaç bant gerekli?**  
C: Minimum 3 bant (RGB). Optimum 5 bant (RGB + DSM + DTM). **12 kanal** RVT türevleri, Eğrilik ve TPI hesaplamalarıyla otomatik olarak oluşturulur.

**S: Önbellek dosyaları ne kadar yer kaplar?**  
C: Tipik olarak 10-50 MB. Giriş dosya boyutuna bağlıdır. Yüksek çözünürlüklü veriler için daha büyük (birkaç GB) olabilir.

**S: Sonuçları nasıl iyileştirebilirim?**  
C: 
1. Birden fazla kodlayıcı kullanın (topluluk)
2. Füzyonu etkinleştirin
3. Eşik değerlerini optimize edin
4. Yüksek kaliteli veri kullanın

**S: Kendi modelimi nasıl eğitirim?**  
C: Proje özel eğitim betikleri içerir! `egitim_verisi_olusturma.py` ve `training.py` kullanarak adım adım talimatlar için aşağıdaki [Model Eğitimi Kılavuzu](#-model-eğitimi-kılavuzu) bölümüne bakın.

**S: Eğitim betiklerini interaktif olarak kullanabilir miyim?**  
C: Evet! `egitim_verisi_olusturma.py` interaktif modu destekler. Argüman olmadan çalıştırın: `python egitim_verisi_olusturma.py` ve size girişler için yönergeler verir.

**S: Ground truth maskelerim yoksa ne olur?**  
C: Yine de sıfır atış ImageNet ağırlıklarıyla (`zero_shot_imagenet: true`) veya sadece klasik yöntemlerle sistemi kullanabilirsiniz. Ancak, en iyi sonuçlar için kendi etiketli verilerinizle özel bir model eğitin.

### 📊 Veri Soruları

**S: Minimum alan çözünürlüğü nedir?**  
C: Önerilen: 0.5-2 metre/piksel. Daha düşük çözünürlükte küçük yapılar tespit edilemeyebilir.

**S: Maksimum dosya boyutu var mı?**  
C: Hayır, karo sistemi sayesinde çok büyük dosyalar işlenebilir. Test edilmiş: 50 GB+

**S: Farklı CRS'ler destekleniyor mu?**  
C: Evet, giriş CRS'i korunur ve çıktıya aktarılır.

---

## 🎓 Model Eğitimi Kılavuzu

Bu bölüm, kendi etiketli verilerinizle özel modeller eğitmek için kapsamlı bir kılavuz sağlar.

### 📋 Ön Koşullar

Eğitimden önce şunlara ihtiyacınız var:
- ✅ RGB + DSM + DTM bantları içeren GeoTIFF dosyaları
- ✅ Ground truth maske dosyaları (GeoTIFF formatı)
  - Arkeolojik alanlar = 1 (beyaz)
  - Arka plan = 0 (siyah)
- ✅ Tüm bağımlılıkların yüklü olduğu Python ortamı
- ✅ GPU önerilir (ama CPU eğitimi mümkündür)

### 🛠️ Adım 1: Ground Truth Maskeleri Hazırlama

İkili maske dosyaları oluşturun:
- **Değer 1 (beyaz)**: Arkeolojik alanlar/yapılar
- **Değer 0 (siyah)**: Arka plan/arkeolojik olmayan alanlar

**QGIS kullanarak örnek:**
1. RGB ortofotoyu yükleyin
2. Yeni çokgen katmanı oluşturun
3. Arkeolojik özellikleri sayısallaştırın
4. Tek bantlı GeoTIFF olarak dışa aktarın (0/1 değerleri)

**Python kullanarak örnek:**
```python
import rasterio
import numpy as np
from rasterio.transform import from_bounds

# Basit ikili maske oluştur
# (Kendi sayısallaştırma iş akışınızla değiştirin)
mask = np.zeros((height, width), dtype=np.uint8)
# Arkeolojik alanları 1 olarak ayarla
mask[arkeolojik_alanlar] = 1

# GeoTIFF olarak kaydet
with rasterio.open('ground_truth.tif', 'w',
                   driver='GTiff',
                   height=height, width=width,
                   count=1, dtype=mask.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(mask, 1)
```

### 📦 Adım 2: Eğitim Verisi Oluşturma

GeoTIFF dosyalarınızdan ve ground truth maskelerinden 12 kanallı eğitim karoları oluşturmak için `egitim_verisi_olusturma.py` kullanın.

#### Temel Kullanım

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data
```

#### İnteraktif Mod

Betiği argüman olmadan çalıştırırsanız, interaktif olarak size yönergeler verir:

```bash
python egitim_verisi_olusturma.py
```

**İnteraktif yönergeler:**
- Giriş GeoTIFF dosyası (varsayılan: `kesif_alani.tif`)
- Ground truth maske dosyası (gerekli)
- Çıktı dizini (varsayılan: `training_data`)
- Karo boyutu (varsayılan: `256`)

Bu, hızlı test veya interaktif giriş tercih ettiğinizde kullanışlıdır.

#### Tüm Seçeneklerle Tam Örnek

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

#### Tüm Parametreler Açıklaması

| Parametre | Varsayılan | Açıklama | Ne Zaman Kullanılır |
|-----------|------------|----------|---------------------|
| `--input`, `-i` | **Gerekli** | Giriş GeoTIFF dosya yolu (RGB + DSM + DTM bantları) | Her zaman gerekli |
| `--mask`, `-m` | **Gerekli** | Ground truth maske dosya yolu (ikili GeoTIFF: 0=arka plan, 1=arkeolojik) | Her zaman gerekli |
| `--output`, `-o` | `training_data` | Eğitim karoları için çıktı dizini | Farklı ad istiyorsanız değiştirin |
| `--tile-size`, `-t` | `256` | Piksel cinsinden karo boyutu (256, 512, vb.) | Çoğu durum için 256, büyük yapılar için 512 |
| `--overlap` | `64` | Karolar arasındaki örtüşme piksel cinsinden | Daha iyi kapsam için artırın (örn. 512 karolar için 128) |
| `--train-ratio` | `0.8` | Eğitim/doğrulama bölme oranı (0.0-1.0) | 0.8 = %80 eğitim, %20 doğrulama (standart) |
| `--min-positive` | `0.0` | Karo dahil etmek için minimum pozitif piksel oranı (0.0-1.0) | 0.01 = <%1 arkeolojik piksel içeren karoları filtrele |
| `--max-nodata` | `0.3` | Karo dahil etmek için maksimum nodata oranı (0.0-1.0) | 0.3 = >%30 nodata içeren karoları hariç tut |
| `--balance-ratio` | `None` | Pozitif/negatif denge oranı (0.0-1.0) | 0.4 = %40 pozitif, %60 negatif (dengesiz veri için önerilir) |
| `--format` | `npz` | Dosya formatı: `npy` (daha hızlı) veya `npz` (sıkıştırılmış, daha küçük) | `npz` disk alanı tasarrufu sağlar (~%50-70 daha küçük) |
| `--bands`, `-b` | `1,2,3,4,5` | Bant sırası: R,G,B,DSM,DTM | Bantlarınız farklı sıradaysa değiştirin |
| `--tpi-radii` | `5,15,30` | Piksel cinsinden TPI yarıçapları (virgülle ayrılmış) | Farklı yapı boyutları için ayarlayın |
| `--no-normalize` | `False` | Normalizasyonu devre dışı bırak (önerilmez) | Sadece ham değerler istiyorsanız |

#### Detaylı İş Akışı

**Adım Adım Süreç:**

1. **Giriş Doğrulama**
   - Giriş GeoTIFF ve maske dosyalarının varlığını kontrol eder
   - Aynı boyutlara ve CRS'e sahip olduklarını doğrular
   - Bant sayısını ve veri tiplerini doğrular

2. **Bant Okuma**
   - RGB bantlarını okur (tipik olarak 1-3 bantlar)
   - DSM'yi okur (Sayısal Yüzey Modeli, bant 4)
   - DTM'yi okur (Sayısal Arazi Modeli, bant 5)
   - Nodata değerlerini ve eksik bantları işler

3. **RVT Türev Hesaplama**
   - **SVF (Gökyüzü Görünürlük Faktörü)**: Ufuk görünürlüğünü hesaplar (tümülüs tespiti)
   - **Pozitif Açıklık**: Yukarı doğru görünürlüğü ölçer (höyükler)
   - **Negatif Açıklık**: Aşağı doğru görünürlüğü ölçer (hendekler)
   - **LRM (Yerel Kabartma Modeli)**: Yerel topografik anomalileri vurgular
   - **Eğim**: Arazi dikliğini hesaplar (teraslar, duvarlar)

4. **Gelişmiş Özellik Hesaplama**
   - **Plan Eğriliği**: Yatay eğrilik (sırt/hendek ayrımı)
   - **Profil Eğriliği**: Dikey eğrilik (teras tespiti)
   - **TPI (Topografik Konum İndeksi)**: Çok ölçekli yükseklik karşılaştırması (höyükler/çöküntüler)

5. **nDSM Hesaplama**
   - Normalize edilmiş DSM hesaplar: `nDSM = DSM - DTM`
   - Yüksek nesneleri (ağaçlar, binalar) maskelemek için kullanılır

6. **Karo Oluşturma**
   - Giriş görüntüsünü örtüşen karolara böler
   - Kenar durumlarını işler (sınırlardaki kısmi karolar)
   - `--min-positive` ve `--max-nodata` kriterlerine göre karoları filtreler

7. **Dengeli Örnekleme** (eğer `--balance-ratio` belirtilmişse)
   - Karoları pozitif (arkeolojik piksel içeren) ve negatif (sadece arka plan) olarak ayırır
   - Hedef oranı elde etmek için negatif karoları örnekler
   - Eğitim verisinde sınıf dengesizliğini önler

8. **Normalizasyon**
   - Sağlam yüzdelik tabanlı normalizasyon uygular (%2-%98 aralığı)
   - Her kanalı bağımsız olarak normalize eder
   - Aykırı değerleri ve ekstrem değerleri işler

9. **Eğitim/Doğrulama Bölme**
   - Karoları rastgele eğitim ve doğrulama setlerine böler
   - Her iki sette aynı pozitif/negatif oranını korur
   - Tekrarlanabilirlik için tohum kullanır

10. **Dosya Kaydetme**
    - 12 kanallı görüntü karolarını kaydeder (`.npz` veya `.npy` formatı)
    - Karşılık gelen ikili maskeleri kaydeder
    - Dizin yapısı oluşturur: `train/images/`, `train/masks/`, `val/images/`, `val/masks/`

11. **Metadata Dışa Aktarma**
    - Veri kümesi istatistikleriyle `metadata.json` kaydeder
    - Karo sayıları, pozitif oranlar, kanal bilgisi içerir
    - Veri kümesi özelliklerini izlemek için kullanışlıdır

#### Çıktı Yapısı

Betiği çalıştırdıktan sonra aşağıdaki dizin yapısını alırsınız:

```
training_data/
├── train/
│   ├── images/
│   │   ├── tile_00000_00000.npz  # 12 kanallı dizi (12, 256, 256)
│   │   ├── tile_00000_00192.npz   # Şekil: (12, 256, 256)
│   │   ├── tile_00000_00384.npz
│   │   └── ...                    # Daha fazla karo
│   └── masks/
│       ├── tile_00000_00000.npz    # İkili maske (256, 256)
│       ├── tile_00000_00192.npz   # Değerler: 0 (arka plan) veya 1 (arkeolojik)
│       ├── tile_00000_00384.npz
│       └── ...                    # Karşılık gelen maskeler
├── val/
│   ├── images/
│   │   ├── tile_01234_00000.npz   # Doğrulama görüntüleri
│   │   └── ...
│   └── masks/
│       ├── tile_01234_00000.npz   # Doğrulama maskeleri
│       └── ...
└── metadata.json                  # Veri kümesi istatistikleri ve bilgisi
```

**Dosya Formatı Detayları:**

- **`.npz` formatı (varsayılan)**: Sıkıştırılmış NumPy arşivi
  - Daha küçük dosya boyutu (~%50-70 azalma)
  - Daha yavaş okuma/yazma (yine de çok hızlı)
  - Disk alanı tasarrufu için önerilir
  
- **`.npy` formatı**: Sıkıştırılmamış NumPy dizisi
  - Daha hızlı okuma/yazma
  - Daha büyük dosya boyutu
  - Disk alanı sorun değilse kullanın

**Dosyaları Yükleme:**

```python
import numpy as np

# .npz dosyasını yükle
data = np.load('tile_00000_00000.npz')
image = data['image']  # Şekil: (12, 256, 256)
mask = np.load('tile_00000_00000.npz')['mask']  # Şekil: (256, 256)

# .npy dosyasını yükle (format npy ise)
image = np.load('tile_00000_00000.npy')  # Şekil: (12, 256, 256)
mask = np.load('tile_00000_00000.npy')   # Şekil: (256, 256)
```

**Metadata.json İçeriği:**

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
      "Kırmızı", "Yeşil", "Mavi", "SVF", "Pozitif Açıklık",
      "Negatif Açıklık", "LRM", "Eğim", "nDSM",
      "Plan Eğriliği", "Profil Eğriliği", "TPI"
    ]
  }
}
```

**Kanal Sırası (12 kanal):**

| İndeks | Kanal | Açıklama | Arkeolojik Kullanım |
|--------|-------|----------|---------------------|
| 0 | Kırmızı | RGB Kırmızı bant | Renk/doku anomalileri |
| 1 | Yeşil | RGB Yeşil bant | Bitki örtüsü desenleri |
| 2 | Mavi | RGB Mavi bant | Toprak renk varyasyonları |
| 3 | SVF | Gökyüzü Görünürlük Faktörü | Tümülüsler, höyükler (ufuk görünürlüğü) |
| 4 | Poz. Açıklık | Pozitif Açıklık | Yükseltilmiş yapılar (yukarı görünürlük) |
| 5 | Neg. Açıklık | Negatif Açıklık | Hendekler, çöküntüler (aşağı görünürlük) |
| 6 | LRM | Yerel Kabartma Modeli | Yerel topografik anomaliler |
| 7 | Eğim | Arazi eğimi | Teraslar, duvarlar, basamaklar |
| 8 | nDSM | Normalize DSM | Yüzey yüksekliği (DSM - DTM) |
| 9 | Plan Eğriliği | Yatay eğrilik | Sırt/hendek ayrımı |
| 10 | Profil Eğriliği | Dikey eğrilik | Teraslar, basamaklar, akış yönü |
| 11 | TPI | Topografik Konum İndeksi | Çevreye göre höyükler/çöküntüler |

### 🚀 Adım 3: Modeli Eğitme

12 kanallı giriş ve CBAM Dikkat ile özel U-Net modelinizi eğitmek için `training.py` kullanın.

#### Temel Eğitim

```bash
python training.py --data training_data
```

**Not:** Betik, eğitim verisi dizinindeki `metadata.json`'dan kanal sayısını otomatik olarak okur. Metadata eksikse, varsayılan olarak 12 kanal kullanır.

Bu, varsayılan ayarları kullanır:
- Mimari: U-Net
- Kodlayıcı: ResNet34
- Epoch: 50
- Batch boyutu: 8
- Öğrenme oranı: 1e-4
- Kayıp: Birleşik (BCE + Dice)
- CBAM Dikkat: Etkin
- Karma Hassasiyet: Etkin (FP16)

#### Tam Eğitim Örneği

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

#### Tüm Parametreler Açıklaması

| Parametre | Varsayılan | Açıklama | Öneriler |
|-----------|------------|----------|----------|
| `--data`, `-d` | `training_data` | Eğitim verisi dizini (Adım 2'den) | `training_data` klasörünüzün yolu |
| `--arch`, `-a` | `Unet` | Model mimarisi | `Unet` (hızlı, iyi), `UnetPlusPlus` (daha iyi doğruluk), `DeepLabV3Plus` (çok ölçekli) |
| `--encoder`, `-e` | `resnet34` | Kodlayıcı omurga | `resnet34` (dengeli), `resnet50` (daha iyi), `efficientnet-b3` (verimli) |
| `--epochs` | `50` | Eğitim epoch sayısı | 50 ile başlayın, kayıp hala düşüyorsa artırın |
| `--batch-size`, `-b` | `8` | Batch boyutu | GPU belleği izin veriyorsa artırın (16-32 daha iyi) |
| `--lr` | `1e-4` | Öğrenme oranı | 1e-4 ile başlayın, kayıp salınıyorsa azaltın |
| `--loss` | `combined` | Kayıp fonksiyonu | `combined` (BCE+Dice), `focal` (dengesiz veri), `dice` (küçük nesneler) |
| `--patience` | `10` | Erken durdurma sabrı | N epoch boyunca iyileşme yoksa dur |
| `--no-attention` | `False` | CBAM Dikkat'i devre dışı bırak | Sadece dikkat olmadan test etmek istiyorsanız devre dışı bırakın |
| `--no-amp` | `False` | Karma hassasiyeti (FP16) devre dışı bırak | Sadece sayısal sorunlarla karşılaşırsanız devre dışı bırakın |
| `--workers` | `4` | DataLoader işçi iş parçacıkları | Daha hızlı veri yükleme için artırın (4-8 tipik) |
| `--output`, `-o` | `checkpoints` | Checkpoint dizini | Eğitilmiş modellerin kaydedileceği yer |
| `--seed` | `42` | Rastgele tohum | Tekrarlanabilirlik için |

#### Eğitim Çıktı Yapısı

```
checkpoints/
├── best_Unet_resnet34_12ch_attention.pth  # En iyi model checkpoint'i
└── training_history.json                  # Eğitim metrikleri (JSON)
```

### 📊 Adım 4: Eğitilmiş Modeli Değerlendirme ve Kullanma

#### Çıkarım için Eğitilmiş Modeli Kullanma

```bash
python archaeo_detect.py \
  --weights checkpoints/best_Unet_resnet34_12ch_attention.pth \
  --input yeni_alan.tif \
  --th 0.6
```

#### config.yaml'da Yapılandırma

```yaml
weights: "checkpoints/best_Unet_resnet34_12ch_attention.pth"
zero_shot_imagenet: false
encoder: "resnet34"
```

---

## 🔬 Gelişmiş Özellikler

### Özel Model Eğitimi

> **📖 Detaylı eğitim kılavuzu için yukarıdaki [Model Eğitimi Kılavuzu](#-model-eğitimi-kılavuzu) bölümüne bakın.**

Proje, özel modeller eğitmek için iki özel betik içerir:

- **`egitim_verisi_olusturma.py`**: GeoTIFF + ground truth maskelerinden 12 kanallı eğitim karoları oluşturur
- **`training.py`**: CBAM Dikkat desteğiyle U-Net modelleri eğitir

**Hızlı Başlangıç:**

```bash
# 1. Eğitim verisi oluştur
python egitim_verisi_olusturma.py --input alan.tif --mask maske.tif --output training_data

# 2. Model eğit
python training.py --data training_data --epochs 50

# 3. Eğitilmiş modeli kullan
python archaeo_detect.py --weights checkpoints/best_Unet_resnet34_12ch_attention.pth
```

**Temel Özellikler:**
- ✅ 12 kanallı giriş (RGB + RVT + Eğrilik + TPI)
- ✅ CBAM Dikkat (kanal + uzamsal)
- ✅ Birden fazla kayıp fonksiyonu (BCE, Dice, Birleşik, Focal)
- ✅ Karma hassasiyet eğitimi
- ✅ Erken durdurma ve checkpoint kaydetme

Tam dokümantasyon, örnekler ve sorun giderme için [Model Eğitimi Kılavuzu](#-model-eğitimi-kılavuzu) bölümüne bakın.

### Özel Kodlayıcı Ekleme

Yeni bir kodlayıcı eklemek için:

```python
# archaeo_detect.py içinde
SUPPORTED_ENCODERS = [
    'resnet34', 'resnet50',
    'efficientnet-b3',
    'sizin_ozel_kodlayiciniz'  # Yeni kodlayıcı ekle
]
```

### API Kullanımı

Python kodundan betiği çağırma:

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

### Toplu İşleme

Birden fazla dosyayı işlemek için betik:

```python
import os
from pathlib import Path
import subprocess

input_dir = Path('giris_dosyalari')
output_dir = Path('sonuclar')
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

### Performans Profili

İşleme sürelerini analiz etme:

```bash
python -m cProfile -o profile.stats archaeo_detect.py

# Sonuçları görüntüle
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## 📚 Teknik Detaylar

### Proje Yapısı

```
archaeo_detect_base/
├── archaeo_detect.py              # Ana tespit betiği
├── egitim_verisi_olusturma.py     # Eğitim verisi oluşturma
├── training.py                     # Model eğitim betiği
├── evaluation.py                   # Değerlendirme metrikleri
├── config.yaml                     # Yapılandırma dosyası
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # İngilizce dokümantasyon
├── README_TR.md                    # Türkçe dokümantasyon (bu dosya)
├── training_data/                  # Oluşturulan eğitim karoları
│   ├── train/
│   │   ├── images/                 # 12 kanallı görüntü karoları (.npz)
│   │   └── masks/                  # İkili maske karoları (.npz)
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── metadata.json               # Veri kümesi metadatası
├── checkpoints/                    # Eğitilmiş model ağırlıkları
│   ├── best_Unet_resnet34_12ch_attention.pth
│   └── training_history.json
├── cache/                          # RVT türevleri önbelleği
│   └── *.derivatives.npz
└── ciktilar/                       # Çıktı tespit sonuçları
    ├── *_prob.tif                  # Olasılık haritaları
    ├── *_mask.tif                  # İkili maskeler
    └── *_mask.gpkg                 # Vektör çokgenler
```

### Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Amaç |
|-----------|----------|------|
| PyTorch | 2.0+ | Derin öğrenme çerçevesi |
| SMP | 0.3.2+ | Segmentasyon modelleri |
| Rasterio | 1.3+ | Raster veri I/O |
| GeoPandas | 0.12+ | Vektör veri işleme |
| OpenCV | 4.7+ | Görüntü işleme |
| scikit-image | 0.20+ | Gelişmiş görüntü işleme |
| RVT-py | 1.2+ (Python < 3.11) veya RVT 2.0+ (Python >= 3.11) | Kabartma görselleştirme |
| NumPy | 1.24+ | Sayısal işlemler |
| SciPy | 1.10+ | Bilimsel hesaplama |

---

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak için:

1. Depoyu **fork** edin
2. Özellik dalı oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: ...'`)
4. Dalınızı push edin (`git push origin feature/yeni-ozellik`)
5. **Pull Request** açın

### Katkı Alanları

- 🐛 Hata düzeltmeleri
- ✨ Yeni özellikler
- 📝 Dokümantasyon iyileştirmeleri
- 🌍 Çeviriler (i18n)
- 🧪 Test senaryoları
- 🎨 Görselleştirme araçları

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

```
MIT Lisansı

Telif Hakkı (c) 2025 [Ahmet Ertuğrul Arık]

Bu yazılımın ve ilişkili dokümantasyon dosyalarının ("Yazılım") bir kopyasını 
alan herhangi bir kişiye, Yazılımı kısıtlama olmaksızın kullanma, kopyalama, 
değiştirme, birleştirme, yayınlama, dağıtma, alt lisanslama ve/veya satma 
haklarını ücretsiz olarak verilir...
```

---

## 📧 İletişim ve Destek

- **Sorunlar**: [GitHub Issues](https://github.com/elestirmen/archaeological-site-detection/issues)
- **E-posta**: ertugrularik@hotmail.com
- **Dokümantasyon**: [Wiki](https://github.com/elestirmen/archaeological-site-detection/wiki)

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden faydalanmaktadır:

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [RVT-py](https://github.com/EarthObservation/RVT_py)
- [PyTorch](https://pytorch.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [GeoPandas](https://geopandas.org/)

---

## 📖 Atıf

Bu projeyi akademik çalışmanızda kullanırsanız, lütfen şu şekilde atıf yapın:

```bibtex
@software{archaeological_site_detection,
  title = {Arkeolojik Alan Tespiti: Derin Öğrenme ve Klasik Görüntü İşleme},
  author = {Ahmet Ertuğrul Arık},
  year = {2025},
  url = {https://github.com/your-username/archaeological-site-detection}
}
```

---

## 📊 Proje İstatistikleri

![GitHub stars](https://img.shields.io/github/stars/your-username/archaeological-site-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/archaeological-site-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/archaeological-site-detection?style=social)

---

<div align="center">

Geliştirici: [Ahmet Ertuğrul Arık]  
Son Güncelleme: Ekim 2025

</div>
