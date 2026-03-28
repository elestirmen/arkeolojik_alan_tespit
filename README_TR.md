# 🏛️ Arkeolojik Alan Tespiti (Derin Öğrenme + Klasik Görüntü İşleme)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

İngilizce dokümantasyon: [`README.md`](README.md).

> **Çok bantlı GeoTIFF verilerinden arkeolojik yapıların otomatik tespiti için gelişmiş yapay zeka sistemi. Öncelikli olarak İHA (insansız hava aracı) nadir görüntülerinden üretilen türevler üzerinde çalışır—ortofoto, DSM/DTM ve türetilmiş kabartma kanalları. Uydu görüntüleri ve diğer hava/LiDAR kaynakları da aynı çok bantlı GeoTIFF formatında sağlandığında desteklenir.**

Bu proje, çok bantlı GeoTIFF verilerinden (RGB, DSM, DTM) arkeolojik izleri (tümülüs, hendek, höyük, duvar kalıntıları vb.) tespit etmek için **derin öğrenme** ve **klasik görüntü işleme** yöntemlerini birleştirir. Girdi verileri çoğunlukla **İHA fotogrametrisinden** elde edilir; **uydu görüntüleri veya diğer hava ürünleri** de bant yapısı ve jeoreferans uyumlu olduğu sürece kullanılabilir.

### Depodaki varsayılan iş akışı (`config.yaml`)

Kayıtlı profil **karo düzeyinde sınıflandırma** (`dl_task: tile_classification`) ve **tek eğitilmiş checkpoint** (`trained_model_only: true`) için ayarlıdır. Bu modda:

- **`weights`** (`.pth` dosyası) ve **`training_metadata`** (eğitimden gelen JSON) kullanılır.
- **`tile`**, **`overlap`** ve **`bands`** çıkarım sırasında `training_metadata.json` içinden kilitlenir; YAML’da yalnızca `overlap` değerini artırarak uyumsuzluğu gidermeyin — farklı overlap için veri üretimini ve eğitimi o overlap ile yeniden yapın.
- Başarılı bir `training.py` çalışmasından sonra en iyi ağırlıklar `checkpoints/active/model.pth` dosyasına, metadata ise `checkpoints/active/training_metadata.json` dosyasına kopyalanır (`weights` yolunu `checkpoints/active/` altındaki başka bir checkpoint’e de yönlendirebilirsiniz).

---

## 📑 İçindekiler

- [✨ Özellikler](#-özellikler)
- [🎯 Ne Yapar](#-ne-yapar)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [📦 Kurulum](#-kurulum)
- [🏷️ Ground Truth Etiketleme Aracı (`ground_truth_kare_etiketleme_qt.py`)](#%EF%B8%8F-ground-truth-etiketleme-aracı-ground_truth_kare_etiketleme_qtpy)
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
- ✅ **🆕 12 Kanallı Giriş**: Gelişmiş tespit için RGB + DSM + DTM + RVT türevleri + nDSM + TPI
- ✅ **🆕 CBAM Dikkat**: Dinamik özellik ağırlıklandırma için kanal ve uzamsal dikkat mekanizması

### 🔧 Teknik Özellikler
- 🚀 **Karo Tabanlı İşleme**: Büyük görüntüler için bellek verimli işleme
- 🎨 **Sorunsuz Mozaikleme**: Kosinüs yumuşatma ile karo sınırlarında artefakt yok
- 📊 **Sağlam Normalizasyon**: Global veya yerel yüzdelik tabanlı normalizasyon
- ⚡ **Önbellek Sistemi**: RVT hesaplamalarını önbelleğe alarak 10-100x hızlanma
- 🎯 **Akıllı Maskeleme**: Yüksek yapıların (ağaçlar, binalar) otomatik filtrelenmesi
- 📐 **Vektörleştirme**: Sonuçları CBS uyumlu çokgenlere dönüştürür
- 🏷️ **Ground Truth Etiketleme**: Katman yönetimli interaktif Qt tabanlı GeoTIFF etiketleme aracı

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

### Uçtan uca: etiket → karo → eğitim → tespit

```bash
pip install -r requirements.txt

# 1) 12 kanallı eğitim karoları (CLI), veya egitim_verisi_olusturma.py içindeki CONFIG ile IDE’den çalıştırın
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data

# 2) Eğitim (training.py CONFIG’te varsayılan görev genelde tile_classification)
python training.py --data training_data --task tile_classification --epochs 50

# 3) Çıkarım (config.yaml; checkpoints/active/ yayını eğitim çıktısında özetlenir)
python archaeo_detect.py
```

Eğitim sonrası artefact’lar:

- `checkpoints/active/model.pth` — çıkarım için kopyalanan en iyi ağırlıklar
- `checkpoints/active/training_metadata.json` — `trained_model_only: true` iken **`tile` / `overlap` / `bands` için kaynak**

**Önemli:** `trained_model_only: true` iken YAML’da yalnızca `overlap` artırarak eğitimle uyumu “sağlamaya” çalışmayın; metadata bu alanları kilitler. Farklı overlap gerekiyorsa veriyi o overlap ile üretin ve modeli yeniden eğitin.

**Henüz eğitilmiş model yoksa:** sıfır atış / klasik yollar için [Kullanım](#-kullanım) bölümüne bakın; ana `config.yaml` yerine örnek profil ile denemek için:

```bash
python archaeo_detect.py --config configs/tile_classification_baseline.example.yaml
```

### Yalnızca tespit (ortam hazır)

```bash
python archaeo_detect.py
```

`config.yaml` içindeki girdi rasterı, yöntemler ve eşikler kullanılır. Sonuçlar `ciktilar/<oturum>/` altına yazılır.

### IDE / CLI ile veri hazırlığı

`egitim_verisi_olusturma.py` dosyasında bir `CONFIG` sözlüğü vardır (varsayılan `input`, `mask`, `output`, `tile_size`, `overlap`, `bands`, …). `--input` / `--mask` vermeden çalıştırıyorsanız bu anahtarların `CONFIG` içinde dolu olması gerekir — girdi için etkileşimli dosya penceresi yoktur.

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

**Not:** `.venv310` opsiyoneldir. Conda (`archeo`) kullanıyorsanız `.venv310` oluşturmanız gerekmez ve varsa silebilirsiniz.

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

## 🏷️ Ground Truth Etiketleme Aracı (`ground_truth_kare_etiketleme_qt.py`)

GeoTIFF görüntüleri üzerinde ikili (binary) ground truth maskeleri oluşturmak için interaktif Qt tabanlı etiketleme aracı. Raster verinizin önizlemesi üzerinde dikdörtgenler çizerek model eğitimi için piksel düzeyinde doğru GeoTIFF maskeleri oluşturabilirsiniz.

### ✨ Temel Özellikler

| Özellik | Açıklama |
|---------|----------|
| **🖱️ Dikdörtgen Çizim** | Sol tıklama + sürükle ile etiketleme/silme dikdörtgeni çiz |
| **🔍 Yakınlaştırma & Kaydırma** | Fare tekerleği ile yakınlaştırma, sağ tıklama ile kaydırma |
| **📐 Kare Kilidi** | Çizimi mükemmel kareye sınırla |
| **↩️ Geri Al** | Tam geri alma geçmişi (Ctrl+Z) |
| **🎨 Bant Seçimi** | Otomatik bant algılama; çok bantlı dosyalarda seçim dialog’u (RGB, BGR, NIR ön ayarları) |
| **🗂️ Katman Paneli** | Görünürlük, saydamlık ayarı, sürükle-bırak ile sıralama |
| **➕ Ek Katmanlar** | Ek GeoTIFF raster dosyalarını üst katman olarak yükleyin |
| **💾 GeoTIFF Çıktı** | Kaynak CRS, dönüşüm ve DEFLATE sıkıştırma ile maske kaydet |
| **🖼️ Sürükle & Bırak** | `.tif` dosyalarını doğrudan pencereye bırakın |
| **🎨 Açık Tema** | Gradient araç çubuğu ve stilize kontroller ile modern açık arayüz |
| **🔌 Çift Backend** | PySide6 veya PyQt6 ile çalışır |

### 🚀 Hızlı Başlangıç

```bash
# Argümansız — dosya dialog’u açılır
python ground_truth_kare_etiketleme_qt.py

# Argümanlarla
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --output kesif_alani_ground_truth.tif

# Mevcut maskeyi düzenlemeye devam
python ground_truth_kare_etiketleme_qt.py \
  --input kesif_alani.tif \
  --existing-mask kesif_alani_ground_truth.tif

# Tek bantlı DEM, önizleme küçültme ile
python ground_truth_kare_etiketleme_qt.py \
  --input karlik_dag_dsm.tif \
  --preview-max-size 4096
```

### ⌨️ Klavye Kısayolları

| Kısayol | Eylem |
|---------|-------|
| `Ctrl+O` | GeoTIFF Aç |
| `Ctrl+S` | Maskeyi Kaydet |
| `Ctrl+Shift+S` | Farklı Kaydet |
| `Ctrl+Z` | Geri Al |
| `D` | Çizim modu |
| `E` | Silme modu |
| `S` | Kare kilidi aç/kapat |
| `F` | Pencereye sığdır |
| `W` | Fare tekerleği yönünü ters çevir |

### 📋 Komut Satırı Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|----------:|
| `--input`, `-i` | Girdi GeoTIFF yolu | _(dosya dialog’u)_ |
| `--output`, `-o` | Çıktı maske yolu | `<girdi>_ground_truth.tif` |
| `--existing-mask` | Düzenlemeye devam edilecek mevcut maske | _(yok)_ |
| `--preview-max-size` | Maks önizleme boyutu piksel (0 = tam çözünürlük) | `0` |
| `--bands` | RGB görüntüleme için virgülle ayrılmış bant indeksleri | `1,2,3` |
| `--positive-value` | Pozitif sınıf piksel değeri (1–255) | `1` |
| `--square-mode` | Kare kilidi açık başlat | `false` |

### 🎵 Bant Seçimi

Dosya açıldığında araç bant sayısını otomatik algılar:

| Bant Sayısı | Davranış |
|:----------:|----------|
| **1** | Otomatik gri tonlama — dialog yok |
| **2** | Bant 1,2 kullanılır — dialog yok |
| **3+** | Ön ayarlarla **Bant Seçim Dialog’u** gösterilir |

**Mevcut Ön Ayarlar (3+ bant):**
- **RGB (1, 2, 3)** — standart gerçek renk
- **BGR (3, 2, 1)** — ters bant sırası
- **NIR (4, 3, 2)** — yakın kızılötesi sahte renk (5+ bant)
- **Gri Tonlama (Bant 1)** — tek bant
- **Özel** — R/G/B için SpinBox ile herhangi bir bant seç

### 🗂️ Katman Paneli

Sol taraftaki panel görüntü katmanlarını yönetir:

- **☑️ Görünürlük** — her katman için işaret kutusu
- **🔀 Sıralama** — sürükle veya ⬆/⬇ butonları (en üstteki ön planda)
- **🎚️ Saydamlık** — seçili katman için sürügü (%0–100)
- **➕ Katman Ekle** — ek GeoTIFF dosyalarını görsel katman olarak yükle
- **➖ Katman Sil** — ekstra katmanları kaldır (ana görüntü ve maske silinemez)

Varsayılan katmanlar:
1. 🔴 **Maske** — etiketleme katmanı (kırmızı, yarı saydam)
2. 🖼️ **Ana Görüntü** — temel raster

### 🔧 Bağımlılıklar

```bash
pip install rasterio opencv-python numpy
pip install PySide6   # veya: pip install PyQt6
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
| `--overlap` | Örtüşme miktarı | `--overlap 128` |
| `--encoder` | Tek kodlayıcı seçimi | `--encoder resnet34` |
| `--encoders` | Çoklu kodlayıcı modu | `--encoders all` |
| `--alpha` | Füzyon ağırlığı | `--alpha 0.6` |
| `--enable-fusion` | Füzyonu etkinleştir | (bayrak) |
| `--cache-derivatives` | Önbellek kullan | (bayrak) |
| `-v` veya `--verbose` | Detaylı log | (bayrak) |

---

## ⚙️ Yapılandırma

### config.yaml Dosyası

Sistem davranışı `config.yaml` dosyası tarafından kontrol edilir. Bu dosya detaylı açıklamalarla **zengin bir şekilde belgelenmiştir** (satır içi Türkçe yorumlar dahil).

**Yol çözümlemesi:** YAML içindeki göreli yollar, **`config.yaml` dosyasının bulunduğu dizine** göre çözülür; çalışma dizininize göre değil. Komut satırından verdiğiniz yollar ise **o anki çalışma dizinine** göre çözülür.

#### Ana Bölümler:

1. **Giriş/Çıkış**: Dosya yolları ve bant seçimi
2. **Yöntem Seçimi**: `enable_deep_learning`, `enable_classic`, `enable_yolo`, `enable_fusion`
3. **DL görevi**: `dl_task` — `segmentation` (piksel) veya `tile_classification` (karo skoru → bindirme ile risk haritası)
4. **Eğitilmiş-tekil mod**: `trained_model_only` — `true` iken tek checkpoint + metadata (`weights`, `training_metadata`); `tile` / `overlap` / `bands` metadata’dan kilitlenir
5. **Derin öğrenme**: Mimari, encoder, ağırlıklar, `zero_shot_imagenet`, dikkat / bant önem raporu
6. **Klasik Yöntemler**: RVT, Hessian, Morfoloji parametreleri
7. **Füzyon**: Hibrit kombinasyon (`alpha`, …) — hem DL hem klasik açık olmalıdır
8. **YOLO11** (isteğe bağlı): Yalnızca RGB; genelde tile sınıflandırma ön ayarında kapalıdır
9. **Karo İşleme**: Bellek ve performans; `tile` / `overlap` belge ile metadata kilitlenmesi
10. **Normalizasyon**: Veri ön işleme
11. **Maskeleme**: Yüksek yapılar (`mask_talls`, `rgb_only`)
12. **Vektörleştirme**: CBS çıktısı (`vectorize`, `min_area`, `export_candidate_excel`, …)
13. **Performans**: Cihaz, `half`, `seed`, `verbose`
14. **Önbellek**: `cache_derivatives`, `cache_derivatives_mode` (`auto` / `npz` / `raster`), raster önbellek ayarları

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

Tüm çıktılar aşağıdaki köke yazılır:

```
ciktilar/<oturum_klasoru>/<cikti_adi>*
```

`<oturum_klasoru>` kısa bir format kullanır:
`<zaman>_<girdi>_<yontemler>_t<tile>o<overlap>_m-<model>`
(örnek model belirteçleri: `m-<checkpoint>`, `m-zs`, `m-<encoder>`).

Her oturum klasöründe ayrıca şunlar bulunur:

```
run_params.txt
```

Bu dosya, etkin parametrelerin tamamını içerir (nihai config değerleri, parse edilen bantlar, CLI argümanları ve cihaz).

Etkinse (`save_band_importance: true`), DL koşuları ayrıca `*_band_importance.txt` ve `*_band_importance.json` dosyalarını üretir.

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

`config.yaml` içinde `export_candidate_excel: true` ise, vektör çıktılarına eşlik eden aday merkezleri / GPS tarzı tablolar `*_gps.xlsx` dosyaları olarak üretilir.

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
   - 1 x DSM (ham)
   - 1 x DTM (ham)
   - 5 x RVT türevleri (SVF, Poz/Neg Açıklık, LRM, Eğim)
   - 1 x nDSM (DSM - DTM)
   - 1 x TPI (Topografik Konum İndeksi)

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
# DTM zorunlu (0 geçersizdir)
# DTM eksikse önce geçerli bir DTM bandı üretin/sağlayın.
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

**S: İHA mı uydu mu—hangi kaynakla çalışır?**  
C: Sistem **öncelikli olarak İHA (drone) nadir görüntüleri** için tasarlanmıştır (ortofoto, DSM, DTM ve bu depoda üretilen türev kanallar). **Uydu görüntüleri de desteklenir**—uyumlu çok bantlı bir GeoTIFF (RGB, varsa DSM/DTM) hizalı bir ızgarada sağlandığında aynı süreç çalışır. LiDAR tabanlı yüzeyler ve diğer sensörler de aynı şekilde kullanılabilir. Önemli olan platform değil, tutarlı bant yapısı ve jeoreferanstır.

### 🔧 Teknik Sorular

**S: Kaç bant gerekli?**  
C: Minimum 3 bant (RGB). Optimum 5 bant (RGB + DSM + DTM). **12 kanal** ham DSM/DTM + RVT türevleri + nDSM + TPI ile otomatik oluşturulur.

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

**S: Eğitim betiklerini etkileşimli (dosya seçme penceresiyle) kullanabilir miyim?**  
C: Hayır. Ya `--input`, `--mask` ve `--output` parametrelerini komut satırından verin ya da `egitim_verisi_olusturma.py` / `training.py` dosyasındaki `CONFIG` sözlüğünde varsayılan yolları tanımlayıp IDE’den çalıştırın.

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

Bu kılavuz, kendi etiketli verilerinizle özel model eğitme sürecini adım adım açıklar. Ham veriden eğitilmiş modele kadar tüm süreci kapsar.

---

### ⚡ Hızlı Başlangıç (Özet)

Deneyimli kullanıcılar için minimal iş akışı:

```bash
# 1. Verilerinizi hazırlayın (GeoTIFF + ikili maske)
# 2. Eğitim karolarını oluşturun
python egitim_verisi_olusturma.py --input veri.tif --mask maske.tif --output training_data

# 3. Modeli eğitin
python training.py --data training_data --task tile_classification --epochs 50

# 4. Eğitilmiş modeli kullanın
python archaeo_detect.py --input yeni_alan.tif
```

---

### 📋 Genel Bakış

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL EĞİTİM İŞ AKIŞI                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │  ADIM 1      │      │  ADIM 2      │      │  ADIM 3      │              │
│   │  Maske       │ ───► │  Karo        │ ───► │  Model       │              │
│   │  Hazırlama   │      │  Oluşturma   │      │  Eğitimi     │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                     │                     │                        │
│         ▼                     ▼                     ▼                        │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │ GeoTIFF +    │      │ 12 kanallı   │      │ Eğitilmiş    │              │
│   │ İkili Maske  │      │ NPZ karolar  │      │ .pth model   │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                      │                       │
│                                                      ▼                       │
│                                               ┌──────────────┐              │
│                                               │  ADIM 4      │              │
│                                               │  Modeli      │              │
│                                               │  Kullan      │              │
│                                               └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**İhtiyacınız olanlar:**
- RGB + DSM + DTM bantları içeren GeoTIFF dosyası
- İkili maske (GeoTIFF): arkeolojik alanlar = 1, arka plan = 0
- Bağımlılıkları yüklü Python ortamı
- GPU önerilir (CPU da çalışır ama yavaştır)

---

### 🛠️ Adım 1: Ground Truth Maskeleri Hazırlama

Arkeolojik özelliklerin **1** (beyaz), diğer her şeyin **0** (siyah) olarak işaretlendiği ikili bir maske oluşturun.

#### QGIS Kullanarak (Ücretsiz, açık kaynak)

**Ne yapacaksınız:** Arkeolojik özelliklerin etrafına çokgenler çizecek, sonra bunları arkeolojik alanlar = 1, diğer her yer = 0 olan bir raster görüntüye dönüştüreceksiniz.

**Adım 1: Ortofotoyu açın**
```
Menü: Katman → Katman Ekle → Raster Katman Ekle...
GeoTIFF dosyanıza gidin → "Ekle"ye tıklayın
```
Görüntünüz harita tuvalinde görünmelidir. Yakınlaştırmak için fare tekerleğini, kaydırmak için orta tuşu basılı tutun.

**Adım 2: Sayısallaştırma için yeni çokgen katmanı oluşturun**
```
Menü: Katman → Katman Oluştur → Yeni Shapefile Katmanı...
```
Açılan pencerede:
- **Dosya adı:** "..." butonuna tıklayıp kayıt yerini seçin (örn. `arkeolojik_maske.shp`)
- **Geometri tipi:** "Çokgen" seçin
- **KRS (Koordinat Referans Sistemi):** Küre ikonuna tıklayın → rasterinizin koordinat sistemini arayın (emin değilseniz raster özelliklerinden bakın)
- "Tamam"a tıklayın

Katmanlar panelinde yeni boş bir katman görünür.

**Adım 3: Sayısallaştırmaya başlayın (çokgen çizimi)**
```
1. Katmanlar panelinde yeni katmanınızı seçin (üzerine tıklayın)
2. Menü: Katman → Sketching'e Geç (veya kalem ikonuna tıklayın)
3. Araç çubuğunda "Çokgen Objesi Ekle" butonunu bulun (+ işaretli çokgen)
4. Butona tıklayın, sonra haritada köşe noktaları eklemek için tıklamaya başlayın
5. Her çokgeni bitirmek için sağ tıklayın
```

**Sayısallaştırma ipuçları:**
- Hassasiyet için yakınlaştırın (fare tekerleği)
- Tümülüslerin, duvarların, hendeklerin etrafını çizin - arkeolojik olan her şey
- Hata yaparsanız: Ctrl+Z ile geri alın
- Her tıklama bir köşe noktası ekler; sağ tıklama çokgeni kapatır
- İhtiyaç kadar çokgen çizin

**Adım 4: Düzenlemelerinizi kaydedin**
```
Menü: Katman → Sketching'e Geç → Sorulduğunda "Kaydet"e tıklayın
Veya: Araç çubuğundaki disket ikonuna tıklayın
```

**Adım 5: Çokgenleri rastera dönüştürün (maske)**
```
Menü: Raster → Dönüştürme → Rasterleştir (Vektörü Rastera)...
```
Açılan pencerede:
- **Giriş katmanı:** Çokgen katmanınız (`arkeolojik_maske`)
- **Yakma değeri için kullanılacak alan:** Boş bırakın (sabit değer kullanacağız)
- **Yakmak için sabit değer:** `1` girin
- **Çıktı raster boyut birimi:** Coğrafi birimler
- **Genişlik/Yatay çözünürlük:** Giriş rasterinizle aynı (örn. 1m çözünürlük için `1.0`)
- **Yükseklik/Dikey çözünürlük:** Aynı değer (örn. `1.0`)
- **Çıktı kapsamı:** "..." → "Katmandan Hesapla" → Giriş rasterinizi seçin
- **Rasterleştirilmiş:** "..." → Dosyaya Kaydet → `ground_truth.tif` olarak adlandırın
- "Çalıştır"a tıklayın

**Adım 6: NoData alanlarını sıfırla doldurun**

Rasterleştirme aracı çokgen olmayan yerlerde NoData oluşturur. Bunların 0 olması gerekiyor.
```
Menü: Raster → Raster Hesap Makinesi...
```
Bu ifadeyi girin (gerçek katman adınızla değiştirin):
```
("ground_truth@1" >= 1) * 1
```
Veya şunu kullanın:
```
Menü: İşleme → Araç Kutusu → "Fill nodata" arayın
"Fill NoData cells" aracını dolgu değeri = 0 ile kullanın
```

**Maskenizi doğrulayın:**
- Değerler sadece 0 ve 1 olmalı
- Katmana sağ tıklayın → Özellikler → Sembolloji → min/max değerlerini kontrol edin
- Boyutlar giriş rasterinizle tam olarak eşleşmeli

---

#### ArcGIS Pro Kullanarak

**Ne yapacaksınız:** Çokgen feature class oluşturacak, arkeolojik özellikleri sayısallaştıracak, sonra raster maskeye dönüştüreceksiniz.

**Adım 1: Yeni proje oluşturun ve verilerinizi ekleyin**
```
1. ArcGIS Pro'yu açın → New Project → Map
2. İsim ve konum verin → OK
3. Map sekmesi → Add Data → GeoTIFF'inize göz atın → Add
```
Ortofotunuz haritada görünmelidir. Yakınlaştırmak için fare tekerleği, kaydırmak için tekerleği basılı tutun.

**Adım 2: Rasterinizin özelliklerini kontrol edin (sonrası için önemli)**
```
1. Contents panelinde rasterinize sağ tıklayın → Properties
2. "Source" sekmesine gidin → Şunları not edin:
   - Cell Size (Hücre Boyutu, örn. 1.0 x 1.0)
   - Extent (Kapsam - Top, Left, Right, Bottom koordinatları)
   - Spatial Reference (Mekansal Referans, örn. EPSG:32635)
```
Bunları yazın - maskenizi eşleştirmek için gerekecek.

**Adım 3: Sayısallaştırma için yeni feature class oluşturun**
```
1. Catalog panelinde projenizin geodatabase'ini (.gdb) genişletin
2. Geodatabase'e sağ tıklayın → New → Feature Class
```
Sihirbazda:
- **Name (Ad):** `arkeolojik_ozellikler`
- **Alias (Takma Ad):** Arkeolojik Özellikler (isteğe bağlı)
- **Feature Class Type:** Polygon
- "Next"e tıklayın
- **Fields (Alanlar):** Atlayın (sonra ekleyeceğiz) → "Next"e tıklayın
- **Spatial Reference:** Küreye tıklayın → Import → Rasterinizi seçin
- "Finish"e tıklayın

Yeni boş katman Contents'te görünür.

**Adım 4: Sayısallaştırmaya başlayın**
```
1. Contents'te seçmek için yeni katmanınıza tıklayın
2. Edit sekmesi → Create (Create Features panelini açar)
3. Create Features panelinde "arkeolojik_ozellikler"e tıklayın
4. "Polygon" aracını seçin
5. Köşe noktaları eklemek için haritada tıklayın, bitirmek için çift tıklayın
```

**Sayısallaştırma ipuçları:**
- Çizerken yakınlaştırmak için `Z`, kaydırmak için `C` tuşuna basın
- Son köşe noktasını geri almak için `Ctrl+Z`
- Her çokgeni bitirmek için çift tıklayın (veya `F2`)
- Görünen tüm arkeolojik özelliklerin etrafını çizin
- Mümkün olduğunca hassas olun - bunlar eğitim etiketleriniz olacak!

**Adım 5: Düzenlemelerinizi kaydedin**
```
Edit sekmesi → Save → Save Edits
```

**Adım 6: Raster değeri için alan ekleyin**
```
1. Contents'te katmanınıza sağ tıklayın → Attribute Table
2. "Add Field" butonuna tıklayın (tablonun üstünde)
3. Field Name: yakma_degeri
4. Data Type: Short (Integer)
5. Fields sekmesinde "Save"e tıklayın
```

**Adım 7: Tüm çokgenlere değer 1 atayın**
```
1. Öznitelik tablosunda "yakma_degeri" sütun başlığına sağ tıklayın
2. "Calculate Field..." seçin
3. Expression kutusuna sadece şunu yazın: 1
4. "OK"e tıklayın
```
Tüm satırlarda yakma_degeri sütununda `1` görünmelidir.

**Adım 8: Rastera dönüştürün**
```
Analysis sekmesi → Tools → "Polygon to Raster" arayın
```
Araç penceresinde:
- **Input Features:** arkeolojik_ozellikler
- **Value field:** yakma_degeri
- **Output Raster Dataset:** Göz at → `ground_truth.tif` olarak kaydedin
- **Cell assignment type:** CELL_CENTER
- **Priority field:** NONE
- **Cellsize:** Giriş rasterinizle aynı (örn. `1`)

**Önemli - Environment Ayarları:**
```
Aracın altındaki "Environments" sekmesine tıklayın:
- Snap Raster: Giriş rasterinizi seçin (hizalamayı garantiler!)
- Cell Size: Giriş rasterinizle aynı
- Extent: Giriş rasterinizle aynı
```
"Run"a tıklayın

**Adım 9: NoData'yı 0'a dönüştürün**

Varsayılan olarak, çokgenlerin dışındaki alanlar NoData olur. Bunların 0 olması gerekiyor.
```
Analysis sekmesi → Tools → "Reclassify" arayın
```
Veya Raster Calculator kullanın:
```
Analysis sekmesi → Tools → "Raster Calculator" arayın
Expression: Con(IsNull("ground_truth.tif"), 0, "ground_truth.tif")
Output: ground_truth_final.tif
```

Reclassify ile alternatif:
```
- Input raster: ground_truth.tif
- Reclass field: Value
- Reclassification (Yeniden sınıflandırma):
  - Satır ekle: Old = NoData, New = 0
  - Mevcut: Old = 1, New = 1
- Output: ground_truth_final.tif
```

**Adım 10: Maskenizi doğrulayın**
```
1. Final maskeyi haritanıza ekleyin
2. Sağ tıklayın → Properties → Source → Kontrol edin:
   - Hücre boyutu girişle eşleşiyor ✓
   - Kapsam girişle eşleşiyor ✓
   - Değerler sadece 0 ve 1 ✓
```

**Yaygın sorunlar:**
- **Maske kapsamı eşleşmiyor:** Polygon to Raster'ı doğru Environment ayarlarıyla yeniden çalıştırın
- **Maske yanlış hücre boyutunda:** Araçta ve Environment'ta hücre boyutunu açıkça ayarlayın
- **Maske tamamen NoData:** yakma_degeri alanının 1 değerine sahip olduğunu kontrol edin

---

#### Python Kullanarak

```python
import rasterio
import numpy as np

# Maske dizisi oluştur (girişle aynı boyutlarda)
mask = np.zeros((height, width), dtype=np.uint8)

# Arkeolojik alanları işaretle (örnek: koordinatlardan veya çokgenlerden)
mask[100:200, 150:250] = 1  # Gerçek alanlarla değiştirin

# GeoTIFF olarak kaydet (giriş CRS ve transform ile eşleşmeli!)
with rasterio.open('maske.tif', 'w', driver='GTiff',
                   height=height, width=width, count=1, 
                   dtype='uint8', crs=giris_crs, 
                   transform=giris_transform) as dst:
    dst.write(mask, 1)
```

> **Önemli:** Maske boyutları, CRS ve çözünürlük giriş GeoTIFF'inizle tam olarak eşleşmelidir!

---

### 📦 Adım 2: Eğitim Karoları Oluşturma

`egitim_verisi_olusturma.py` betiği GeoTIFF + maskenizi 12 kanallı eğitim karolarına dönüştürür.

#### Temel Komut

```bash
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data
```

#### İnteraktif Mod

Yönlendirmeli giriş için argümansız çalıştırın:

```bash
python egitim_verisi_olusturma.py
# Sırayla sorar: Giriş dosyası → Maske dosyası → Çıktı dizini → Karo boyutu
```

#### İçeride Ne Olur

```
Giriş GeoTIFF (5 bant)           Ground Truth Maske
       │                                │
       ▼                                │
┌──────────────────┐                    │
│ RGB + DSM + DTM  │                    │
│ bantlarını oku   │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ RVT türevlerini  │                    │
│ hesapla:         │                    │
│ - SVF            │                    │
│ - Açıklık (+/-)  │                    │
│ - LRM, Eğim      │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ Hesapla:         │                    │
│ - DSM/DTM (ham)  │                    │
│ - TPI            │                    │
│ - nDSM           │                    │
└────────┬─────────┘                    │
         │                              │
         ▼                              │
┌──────────────────┐                    │
│ 12 kanalı yığınla│◄───────────────────┘
│ + 256x256        │
│ karolara böl     │
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

#### Temel Parametreler

Tam liste için: `python egitim_verisi_olusturma.py --help`. Sık kullanılanlar:

| Parametre | Varsayılan (betikteki `CONFIG`) | Açıklama |
|-----------|--------------------------------|----------|
| `--input` / `-i` | CLI veya `CONFIG` ile zorunlu | Çok bantlı GeoTIFF (RGB + DSM + DTM) |
| `--mask` / `-m` | CLI veya `CONFIG` ile zorunlu | Ground truth maske (0 dışındaki değerler pozitif sayılıp ikiliye çevrilir) |
| `--output` / `-o` | `training_data` | Çıktı kökü (`train/`, `val/`, `metadata.json`, …) |
| `--tile-size` / `-t` | `256` | Karo boyutu (piksel) |
| `--overlap` | `128` | Kaydırmalı pencere örtüşmesi (eğitim/çıkarımda metadata ile tutarlı kalmalı) |
| `--bands` / `-b` | `1,2,3,4,5` | 1 tabanlı GeoTIFF bant indeksleri: R, G, B, DSM, DTM |
| `--tpi-radii` | `5,15,30` | Virgülle ayrılmış TPI yarıçapları (piksel) |
| `--min-positive` | `0.0` | Karoda minimum pozitif piksel oranı |
| `--tile-label-min-positive-ratio` | `CONFIG`’ten | Karo sınıfı etiketi için minimum pozitif oran (0 = en az bir pozitif piksel yeter) |
| `--max-nodata` | `0.3` | Karo başına izin verilen maksimum NoData oranı |
| `--train-ratio` | `0.8` | Eğitim oranı |
| `--train-negative-keep-ratio` | `1.0` | Tamamen negatif **eğitim** karolarının tutulacak kesri (`0` = hepsini at, `1` = hepsini tut) |
| `--train-negative-max` | `None` | Tutulan negatif eğitim karosu için üst sınır |
| `--split-mode` | `spatial` | `spatial` (önerilir) veya `random` train/val bölmesi |
| `--no-normalize` | kapalı | `robust_norm` atlanır |
| `--format` | `npz` | `npz` veya `npy` |
| `--num-workers` | CPU’ya göre | Paralel işçi sayısı |
| `--tile-prefix` | `""` | İsteğe bağlı dosya adı öneki (boşsa otomatik önek) |
| `--append` / `--no-append` | temiz yeniden üret | Mevcut karolara ekleme vs tam yeniden oluşturma |

#### Senaryoya Göre Önerilen Ayarlar

| Senaryo | Komut |
|---------|-------|
| **Standart** | `--tile-size 256 --overlap 64` |
| **Büyük yapılar** | `--tile-size 512 --overlap 64` |
| **Dengesiz veri** (<%5 arkeolojik) | `--train-negative-keep-ratio 0.2 --min-positive 0.01` |
| **Hızlı test** | `--tile-size 256 --train-ratio 0.9` |

#### Çıktı: 12 Kanal Açıklaması

Kanonik sıra `archeo_shared/channels.py` içindeki `MODEL_CHANNEL_NAMES` ile tanımlıdır; eğitim ve çıkarım `archaeo_detect.py` içindeki `stack_channels()` ile aynı düzeni kullanır.

| # | Kanal | Ne Tespit Eder |
|---|-------|----------------|
| 0-2 | RGB | Renk/doku anomalileri |
| 3 | DSM | Yüzey yükseklik bağlamı |
| 4 | DTM | Zemin yükseklik bağlamı |
| 5 | SVF | Tümülüsler, höyükler (ufuk görünürlüğü) |
| 6 | Pozitif Açıklık | Yükseltilmiş yapılar |
| 7 | Negatif Açıklık | Hendekler, çöküntüler |
| 8 | LRM | Yerel topografik anomaliler |
| 9 | Eğim | Teraslar, duvarlar |
| 10 | nDSM | Zemin üstü yüzey yüksekliği |
| 11 | TPI | Göreceli yükseklik (höyükler/çöküntüler) |

---

### 🚀 Adım 3: Modeli Eğitme

12 kanallı verileriniz üzerinde CBAM dikkat mekanizmalı U-Net modeli eğitmek için `training.py` kullanın.

#### Temel Eğitim

```bash
python training.py --data training_data
```

Bu mantıklı varsayılanları kullanır: U-Net + ResNet34 + 50 epoch + CBAM dikkat + karma hassasiyet.

#### Tüm Seçeneklerle Tam Komut

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

#### Temel Parametreler

| Parametre | Varsayılan | Seçenekler / Notlar |
|-----------|------------|---------------------|
| `--data` | `training_data` | Adım 2 çıktısının yolu |
| `--arch` | `Unet` | `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN` |
| `--encoder` | `resnet34` | `resnet50`, `efficientnet-b3`, `densenet121` |
| `--epochs` | `50` | Daha fazla = potansiyel olarak daha iyi (erken durdurma ile) |
| `--batch-size` | `8` | GPU belleği izin veriyorsa artırın |
| `--lr` | `1e-4` | Kayıp salınıyorsa azaltın |
| `--loss` | `combined` | `bce`, `dice`, `combined`, `focal` |
| `--patience` | `10` | N epoch iyileşme yoksa erken durdurma |
| `--no-attention` | Kapalı | CBAM dikkatini devre dışı bırak |
| `--no-amp` | Kapalı | Karma hassasiyeti (FP16) devre dışı bırak |

#### Doğru Ayarları Seçme

**Model Mimarisi:**

| Mimari | Hız | Doğruluk | Ne Zaman Kullanılır |
|--------|-----|----------|---------------------|
| `Unet` | Hızlı | İyi | **Buradan başlayın** - güvenilir temel |
| `UnetPlusPlus` | Orta | Mükemmel | Daha yüksek doğruluk gerektiğinde |
| `DeepLabV3Plus` | Orta | Mükemmel | Çok ölçekli yapılar |

**Kodlayıcı:**

| Kodlayıcı | Hız | Doğruluk | Bellek |
|-----------|-----|----------|--------|
| `resnet34` | Hızlı | İyi | Düşük | **Önerilen başlangıç** |
| `resnet50` | Orta | Daha iyi | Orta | Daha iyi doğruluk |
| `efficientnet-b3` | Hızlı | Mükemmel | Düşük | En iyi verimlilik |

**Kayıp Fonksiyonu:**

| Kayıp | Ne Zaman Kullanılır |
|-------|---------------------|
| `combined` | **Varsayılan** - çoğu durum için çalışır |
| `focal` | Dengesiz veri (az arkeolojik piksel) |
| `dice` | Küçük nesneler, örtüşme odaklı |

#### Eğitim Çıktısı

```
checkpoints/
├── active/model.pth                         ← en iyi ağırlıkların kopyası (training.py içindeki publish_active)
├── active/training_metadata.json           ← trained_model_only için tile / overlap / bands (+ şema)
├── active/published_from.json              ← kopyanın kaynak checkpoint’e işaret eden manifest
├── epochs/                                  ← save_every_epoch açıksa (CONFIG’te varsayılan genelde açık) epoch checkpoint’leri
└── training_history.json                   ← Eğitim metrikleri
```

`weights` olarak `checkpoints/active/model.pth` veya `checkpoints/epochs/` altındaki belirli bir çalıştırmayı seçebilirsiniz; her durumda **`training_metadata.json` aynı mimari, kanal sayısı, karo boyutu, overlap ve bantları** tanımlamalıdır.

`channel_importance_history.json` dosyası da (etkinse) `checkpoints/` altında üretilir; epoch bazlı bant önem sıralarını içerir.

#### Eğitimi İzleme

Konsol çıktısını izleyin:

```
Epoch  1/50 | Train Loss: 0.45 | Val Loss: 0.39 | Val IoU: 0.62 | LR: 1e-04
  → En iyi model kaydedildi
Epoch  2/50 | Train Loss: 0.38 | Val Loss: 0.34 | Val IoU: 0.68 | LR: 1e-04
  → En iyi model kaydedildi
...
Erken durdurma: En iyi model 15. epoch'ta (Val IoU: 0.79)
```

**Metriklerin anlamı:**
- **Val IoU** (Kesişim/Birleşim): Yüksek = daha iyi. Hedef: 0.6-0.8
- **Val Loss**: Düşük = daha iyi. Zamanla azalmalı
- **Train Loss**: Val Loss'tan biraz düşük olmalı (çok düşükse = aşırı öğrenme)

---

### 📊 Adım 4: Eğitilmiş Modeli Kullanma

#### Komut Satırından

```bash
python archaeo_detect.py \
  --weights checkpoints/active/model.pth \
  --training-metadata checkpoints/active/training_metadata.json \
  --input yeni_alan.tif \
  --th 0.6
```

#### config.yaml Üzerinden

```yaml
weights: "checkpoints/active/model.pth"
training_metadata: "checkpoints/active/training_metadata.json"
zero_shot_imagenet: false
trained_model_only: true
```

Sonra sadece çalıştırın:
```bash
python archaeo_detect.py
```

---

### 🔧 Sorun Giderme

#### Veri Hazırlama Sorunları

| Sorun | Neden | Çözüm |
|-------|-------|-------|
| "Maske boyutları eşleşmiyor" | Farklı çözünürlük/kapsam | Maskeyi yeniden örnekle: `gdalwarp -tr 1.0 1.0 -r nearest maske.tif maske_duzeltilmis.tif` |
| "Geçerli karo bulunamadı" | `--min-positive` çok yüksek | `0.0` veya `0.01`'e düşürün |
| "Bellek hatası" | Büyük giriş dosyası | `--tile-size`'ı 128'e düşürün |

#### Eğitim Sorunları

| Sorun | Neden | Çözüm |
|-------|-------|-------|
| Kayıp düşmüyor | Öğrenme oranı çok yüksek | `--lr 5e-5` veya `1e-5` kullanın |
| GPU bellek yetersiz | Batch boyutu çok büyük | `--batch-size 4` veya `--no-amp` kullanın |
| Aşırı öğrenme (train << val loss) | Çok az veri | Daha fazla karo ekleyin veya `--loss focal` kullanın |
| Tüm tahminler = 0 | Sınıf dengesizliği | `--loss focal` kullanın, veri hazırlamada negatif eğitim karolarını azaltın (örn. `--train-negative-keep-ratio 0.2`) |
| Eğitim çok yavaş | GPU yok / küçük batch | GPU kullanın, `--batch-size` artırın, AMP etkinleştirin |

#### Hızlı Tanı Komutları

```bash
# Eğitim verisi yapısını kontrol et
ls -R training_data/

# Metadata'yı görüntüle
cat training_data/metadata.json | python -m json.tool

# Veri yüklemeyi test et
python -c "import numpy as np; d=np.load('training_data/train/images/tile_00000_00000.npz'); print(d['image'].shape)"
# Beklenen: (12, 256, 256)
```

---

### 💡 En İyi Uygulamalar

#### Veri Kalitesi Kontrol Listesi

- [ ] Maskeler doğru (kesin sınırlar)
- [ ] Tüm arkeolojik özellikler tutarlı şekilde etiketlenmiş
- [ ] Dengeli veri kümesi (%30-50 pozitif karo)
- [ ] Negatiflerde çeşitli arazi türleri
- [ ] Minimum 1000 karo (2000-5000 önerilir)

#### Eğitim İş Akışı

```
1. Hızlı test (5 epoch)      → Her şeyin çalıştığını doğrula
2. Temel (50 epoch)          → Başlangıç noktası belirle
3. Optimize et (daha iyi kodlayıcı/mimari dene)
4. İnce ayar (gerekirse LR düşür)
```

#### Performans Beklentileri

| Veri Kümesi Boyutu | Beklenen Val IoU | Eğitim Süresi (GPU) |
|--------------------|------------------|---------------------|
| 500-1000 karo | 0.55-0.65 | 30-60 dk |
| 1000-3000 karo | 0.65-0.75 | 1-2 saat |
| 3000-5000 karo | 0.70-0.80 | 2-4 saat |
| 5000+ karo | 0.75-0.85 | 4+ saat |

---

### 📚 Tam Örnek: Uçtan Uca

```bash
# 1. Eğitim verisini oluştur (tek dengeleme mekanizması: train-negatif filtreleme)
python egitim_verisi_olusturma.py \
  --input kesif_alani.tif \
  --mask ground_truth.tif \
  --output training_data \
  --tile-size 256 \
  --train-negative-keep-ratio 0.3

# 2. Modeli eğit
python training.py \
  --data training_data \
  --arch Unet \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 16 \
  --loss combined

# 3. Yeni alanda çıkarım yap
python archaeo_detect.py \
  --weights checkpoints/active/model.pth \
  --training-metadata checkpoints/active/training_metadata.json \
  --input yeni_alan.tif \
  --th 0.6 \
  --enable-fusion
```

**Beklenen sonuçlar:**
- ~1000-2000 eğitim karosu
- Val IoU: 0.65-0.75
- Eğitim süresi: 1-2 saat (GPU)
- Model dosyası: ~70 MB

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
python training.py --data training_data --task tile_classification --epochs 50

# 3. Eğitilmiş modeli kullan
python archaeo_detect.py
```

**Temel Özellikler:**
- ✅ 12 kanallı giriş (RGB + DSM + DTM + RVT + nDSM + TPI)
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
arkeolojik_alan_tespit/            # proje kökü (örnek ad)
├── archaeo_detect.py              # Ana tespit betiği
├── archeo_shared/                 # Ortak kanal şeması ve model yardımcıları
│   └── channels.py                # MODEL_CHANNEL_NAMES, metadata şema sürümü
├── egitim_verisi_olusturma.py     # Eğitim verisi oluşturma
├── training.py                    # Model eğitim betiği
├── evaluation.py                  # Değerlendirme metrikleri
├── config.yaml                    # Yapılandırma dosyası
├── configs/                       # Örnek profiller (ör. tile classification)
├── requirements.txt               # Python bağımlılıkları
├── README.md                      # İngilizce dokümantasyon
├── README_TR.md                   # Türkçe dokümantasyon (bu dosya)
├── training_data/                  # Oluşturulan eğitim karoları
│   ├── train/
│   │   ├── images/                 # 12 kanallı görüntü karoları (.npz)
│   │   └── masks/                  # İkili maske karoları (.npz)
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── metadata.json               # Veri kümesi metadatası
├── checkpoints/                    # Eğitilmiş model ağırlıkları
│   ├── active/model.pth
│   ├── active/training_metadata.json
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
  url = {https://github.com/elestirmen/archaeological-site-detection}
}
```

---

## 📊 Proje İstatistikleri

![GitHub stars](https://img.shields.io/github/stars/elestirmen/archaeological-site-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/elestirmen/archaeological-site-detection?style=social)

---

<div align="center">

Geliştirici: Ahmet Ertuğrul Arık  
Son güncelleme: Mart 2026

</div>
