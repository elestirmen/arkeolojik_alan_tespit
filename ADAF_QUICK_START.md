# 🚀 ADAF Hızlı Başlangıç - 5 Dakikada Kurulum

## ✅ ADAF ile Hemen Başlayın

ADAF = **Hazır, arkeolojik özellikler için eğitilmiş, LiDAR/kuş bakışı optimize** modeller!

---

## 📦 Kurulum (5 Dakika)

### 1. ADAF Repository'yi Klonla

```bash
# Projenizin ana dizininde
cd C:\d_surucusu\arkeolojik_alan_tespit

# ADAF'i klonla
git clone https://github.com/elestirmen/adaf.git
```

### 2. Modelleri İndir (~5GB)

**Link:** https://github.com/elestirmen/adaf

README'deki download linkinden modelleri indirin.

**İndirilenler:**
- 4 Segmentation modeli (TAR dosyaları)
- 4 Object detection modeli (TAR dosyaları)
- Toplam ~5GB

### 3. Modelleri Yerleştir

```bash
# TAR dosyalarını adaf/ml_models/ dizinine taşıyın
# ⚠️ UYARI: TAR dosyalarını AÇMAYIN! Olduğu gibi bırakın!

# Dizin yapısı:
adaf/
└── ml_models/
    ├── segmentation_model_resnet34.tar
    ├── segmentation_model_resnet50.tar
    ├── segmentation_model_efficientnet.tar
    ├── segmentation_model_densenet.tar
    ├── detection_model_resnet34.tar
    ├── detection_model_resnet50.tar
    ├── detection_model_efficientnet.tar
    └── detection_model_densenet.tar
```

### 4. AiTLAS Yükle

```bash
# Conda environment'ınızı aktifleştirin
conda activate archaeo_detect  # veya sizin env'iniz

# AiTLAS wheel'i yükleyin
pip install adaf/installation/aitlas-0.0.1-py3-none-any.whl

# Kontrol edin
python -c "import aitlas; print('✓ AiTLAS yüklü')"
```

### 5. Etkinleştir

```yaml
# config.yaml
enable_adaf: true
```

### 6. Çalıştır!

```bash
python archaeo_detect.py
```

**Çıktı:**
```
ciktilar/
├── kesif_alani_adaf_th0.5_tile1024_minarea80_prob.tif   # Olasılık
├── kesif_alani_adaf_th0.5_tile1024_minarea80_mask.tif   # Maske
└── kesif_alani_adaf_th0.5_tile1024_minarea80_mask.gpkg  # Vektör
```

---

## 🎯 Kullanım Örnekleri

### Örnek 1: Sadece ADAF (En Hızlı)

```yaml
# config.yaml
enable_deep_learning: false
enable_classic: false
enable_yolo: false
enable_adaf: true
cache_derivatives: true
```

```bash
python archaeo_detect.py
# 10-15 dakika (cache ile daha hızlı!)
```

### Örnek 2: ADAF + YOLO (Çifte Güç)

```yaml
# config.yaml
enable_adaf: true   # Arkeolojik özellikler (barrows, ringforts)
enable_yolo: true   # Genel envanter (ağaç, bina, araç)
```

**Sonuç:**
- ADAF: Tümülüs, ringfort, çevrili alan
- YOLO: Ağaç, bina, araç, insan
- **İki dosya:** `*_adaf_mask.gpkg` + `*_yolo11_labels.gpkg`

### Örnek 3: Tüm Yöntemler (Maksimum Tespit)

```yaml
# config.yaml - Tüm güçleri birleştir
enable_deep_learning: true
enable_classic: true
enable_yolo: true
enable_adaf: true
enable_fusion: true
cache_derivatives: true
```

**Çıktılar:**
- DL: U-Net tespitleri
- Classic: RVT tabanlı
- YOLO: Genel envanter
- **ADAF: Arkeolojik özellikler** ⭐
- Fusion: Hepsinin birleşimi

---

## 🎨 QGIS'te Görselleştirme

```bash
# 1. QGIS'i açın
qgis

# 2. Base raster'ı yükle
Layer → Add Raster Layer → kesif_alani.tif

# 3. ADAF tespitlerini yükle
Layer → Add Vector Layer → ciktilar/*_adaf_mask.gpkg

# 4. Stil ayarla
Properties → Symbology
  → Single Symbol
  → Renk: Kırmızı/Turuncu
  → Transparency: 30%
  → Outline: Beyaz, 1px

# 5. YOLO etiketlerini de yükle
Layer → Add Vector Layer → ciktilar/*_yolo11_labels.gpkg
  → Symbology → Categorized → class_name

# 6. Karşılaştır!
```

---

## 📊 Performans

| İşlem | Cache YOK | Cache VAR |
|-------|-----------|-----------|
| RVT hesaplama | 15 dakika | 0 saniye ✅ |
| ADAF inference | 10 dakika | 10 dakika |
| **Toplam** | **25 dakika** | **10 dakika** |

**Öneri:** İlk çalıştırmada `cache_derivatives: true` yapın!

---

## 🔧 Sorun Giderme Hızlı Referans

### ❌ "aitlas bulunamadı"

```bash
pip install adaf/installation/aitlas-0.0.1-py3-none-any.whl
```

### ❌ "Model dosyası bulunamadı"

```bash
# Dizini kontrol et
ls adaf/ml_models/*.tar

# Yoksa modelleri GitHub'dan indir
# https://github.com/elestirmen/adaf
```

### ❌ "TAR dosyası bozuk"

- TAR dosyalarını AÇMAYIN
- TAR dosyalarını yeniden indirin
- MD5 checksum kontrol edin

### ⚠️ "DTM band gerekli"

ADAF RVT derivatives kullanır, DTM zorunlu:
```yaml
bands: "1,2,3,4,5"  # Son band DTM olmalı
```

---

## 🎯 Beklenen Sonuçlar

### Başarılı Çalıştırma

```
======================================================================
ADAF (AiTLAS) BAŞLATILIYOR
======================================================================
ADAF modelleri yükleniyor (4 model)...
  → segmentation_model_resnet34.tar
  → segmentation_model_resnet50.tar
  → segmentation_model_efficientnet.tar
  → segmentation_model_densenet.tar
✓ 4 ADAF modeli yüklendi
ADAF ensemble inference başlıyor (4 model)...
  [████████████████████████] 100% Inference
✓ ADAF olasılık haritası: ciktilar/kesif_alani_adaf_prob.tif
✓ ADAF ikili maske: ciktilar/kesif_alani_adaf_mask.tif
✓ Tespit edilen sınıflar: barrows, ringforts, enclosures
```

### Tespit Örnekleri

**Barrows (Tümülüs):**
- Yuvarlak yükseltiler
- 10-30m çap
- score_mean > 0.7: Çok güvenilir
- score_mean > 0.5: Olası
- area_m2: 100-2000 m²

**Ringforts:**
- Dairesel tahkimatlar
- 20-50m çap
- İçi çukur, dışı yüksek
- area_m2: 500-5000 m²

**Enclosures:**
- Çevrili alanlar
- Dikdörtgen/dairesel
- area_m2: değişken (500-50000 m²)

---

## ✅ Checklist

- [ ] ADAF klonlandı (`adaf/` dizini var)
- [ ] Modeller indirildi (~5GB)
- [ ] TAR dosyaları `adaf/ml_models/` dizininde (AÇILMAMŞ!)
- [ ] AiTLAS yüklü (`import aitlas` çalışıyor)
- [ ] `config.yaml`: `enable_adaf: true`
- [ ] DTM band tanımlı (`bands: "1,2,3,4,5"`)
- [ ] `python archaeo_detect.py` çalışıyor
- [ ] Çıktı dosyaları oluştu (`ciktilar/*_adaf_*`)

---

**5 dakika kurulum, ömür boyu arkeolojik tespit! 🏛️🚀**

