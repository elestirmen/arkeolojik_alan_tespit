# 🏛️ ADAF Entegrasyonu - Arkeolojik Özellikler için Özel Modeller

## 🎯 ADAF Nedir?

[ADAF (Automatic Detection of Archaeological Features)](https://github.com/elestirmen/adaf), İrlanda'daki geniş ALS (Airborne Laser Scanning) verilerinden eğitilmiş, **arkeolojik özellikler için özel** derin öğrenme modelleri sunar.

**Geliştiriciler:** ZRC SAZU, Bias Variance Labs, The Discovery Programme

---

## ✨ ADAF'in Avantajları

### ✅ Arkeolojik Özellikler için Özel Eğitilmiş

YOLO veya genel modellerin aksine, **tamamen arkeolojik yapılar** için eğitilmiş:

| Model Tipi | Eğitim Verisi | Tespit Ettiği |
|-----------|---------------|---------------|
| **ADAF** | İrlanda ALS (arkeolojik) | Barrows, Ringforts, Enclosures |
| YOLO11 | COCO (genel) | Person, car, tree (genel) |
| U-Net (sizinki) | Sizin verileriniz | Sizin etiketleriniz |

### ✅ Kuş Bakışı (LiDAR) için Optimize

- **ADAF:** LiDAR/ALS verilerinden eğitilmiş ✅
- **YOLO:** Yatay perspektiften eğitilmiş ❌
- **Sonuç:** ADAF kuş bakışı görüntülerde doğrudan yüksek performans!

### ✅ Ensemble Öğrenme

- 4 Segmentation modeli (farklı encoder'lar)
- 4 Object detection modeli
- Otomatik ensemble: tüm modellerin ortalaması
- Tek modelden daha güvenilir

### ✅ Hazır Modeller

- Eğitim gerektirmez ✅
- ~5GB indirme ile hemen kullanılabilir
- Fine-tuning opsiyonel

---

## 📦 Kurulum

### Adım 1: ADAF Repository'yi Klonla

```bash
# Projenizin yanına klonlayın
cd C:\d_surucusu\arkeolojik_alan_tespit
git clone https://github.com/elestirmen/adaf.git
```

**Dizin yapısı:**
```
arkeolojik_alan_tespit/
├── archaeo_detect.py
├── config.yaml
├── adaf/                    # ← Klonlanan ADAF
│   ├── ml_models/           # ← Modeller buraya gelecek
│   ├── installation/        # ← AiTLAS wheel burada
│   ├── ADAF_main.ipynb
│   └── README.md
└── ...
```

### Adım 2: ADAF Modellerini İndir

ADAF GitHub sayfasından modelleri indirin (~5GB):
- https://github.com/elestirmen/adaf

**İndirme linki** README'de belirtilmiş.

**Modeller:**
- 4 Segmentation modeli (TAR formatı)
- 4 Object detection modeli (TAR formatı)

### Adım 3: Modelleri Yerleştir

```bash
# İndirilen TAR dosyalarını adaf/ml_models/ dizinine taşıyın
# Örnek:
adaf/
└── ml_models/
    ├── segmentation_model_resnet34.tar
    ├── segmentation_model_resnet50.tar
    ├── segmentation_model_efficientnet.tar
    ├── segmentation_model_densenet.tar
    ├── detection_model_resnet34.tar
    └── ...
```

**⚠️ ÖNEMLİ:**
- TAR dosyalarını **AÇMAYIN**! (expand etmeyin)
- Dosya adlarını **DEĞİŞTİRMEYİN**!
- Tam yolu: `adaf/ml_models/*.tar`

### Adım 4: AiTLAS Kütüphanesini Yükle

```bash
# Conda environment'ınızda
conda activate archaeo_detect  # veya sizin env'iniz

# AiTLAS wheel'i yükle
pip install adaf/installation/aitlas-0.0.1-py3-none-any.whl

# Kontrol et
python -c "import aitlas; print('AiTLAS yüklü ✓')"
```

### Adım 5: Sistemi Yapılandır

```yaml
# config.yaml
enable_adaf: true
adaf_models_dir: "adaf/ml_models"
adaf_model_type: "segmentation"  # veya "detection"
adaf_threshold: 0.5
adaf_classes: "barrows,ringforts,enclosures"
```

---

## 🚀 Kullanım

### Temel Kullanım

```bash
python archaeo_detect.py
```

**Ne olur:**
1. ADAF modelleri yüklenir (4 segmentation modeli)
2. RVT derivatives hesaplanır (SVF, openness, LRM, slope)
3. Ensemble inference yapılır
4. Çıktılar kaydedilir:
   - `*_adaf_prob.tif` - Olasılık haritası
   - `*_adaf_mask.tif` - İkili maske
   - `*_adaf_mask.gpkg` - Vektör poligonlar

### Cache ile Hızlandırma

```yaml
# config.yaml
enable_adaf: true
cache_derivatives: true  # RVT cache'i kullan
```

```bash
# İlk çalıştırma (yavaş)
python archaeo_detect.py
# RVT türevleri hesaplanır ve cache'lenir

# İkinci çalıştırma (ÇOK HIZLI!)
python archaeo_detect.py
# RVT cache'den okunur, sadece ADAF inference yapılır
```

### Sadece ADAF Kullan

```yaml
# config.yaml
enable_deep_learning: false
enable_classic: false
enable_yolo: false
enable_adaf: true
```

### Tüm Yöntemleri Birleştir

```yaml
# config.yaml - Maksimum tespit için
enable_deep_learning: true
enable_classic: true
enable_yolo: true
enable_adaf: true
enable_fusion: true
```

---

## ⚙️ Parametreler

### adaf_models_dir (str)
ADAF model dosyalarının dizini
- Varsayılan: `"adaf/ml_models"`
- TAR dosyaları burada olmalı

### adaf_model_type (str)
- `"segmentation"`: Semantic segmentation (önerilen) 
- `"detection"`: Object detection

### adaf_model_name (str veya null)
- `null`: Ensemble (tüm modeller) - ÖNERİLEN
- `"model_resnet34"`: Sadece belirtilen model

### adaf_threshold (float, 0-1)
Olasılık eşiği
- `0.3-0.4`: Hassas, daha fazla tespit
- `0.5`: Dengeli (varsayılan)
- `0.6-0.7`: Seçici, daha az tespit

### adaf_classes (str)
Tespit edilecek sınıflar (virgülle ayrılmış)
- Varsayılan: `"barrows,ringforts,enclosures"`

**Sınıflar:**
- **barrows**: Tümülüs, höyük (yuvarlak/yükseltilmiş mezar yapıları)
- **ringforts**: Dairesel tahkimatlar (erken orta çağ yerleşimleri)
- **enclosures**: Çevrili alanlar (duvar/hendek/kazık ile çevrili)

---

## 📊 Çıktı Dosyaları

```
ciktilar/
├── kesif_alani_adaf_th0.5_tile1024_minarea80_prob.tif   # Olasılık haritası
├── kesif_alani_adaf_th0.5_tile1024_minarea80_mask.tif   # İkili maske
└── kesif_alani_adaf_th0.5_tile1024_minarea80_mask.gpkg  # Vektör poligonlar
```

**Attribute table (GPKG):**
- `id`: Poligon numarası
- `area_m2`: Alan (metrekare)
- `score_mean`: Ortalama güven skoru
- `geometry`: Polygon geometrisi

---

## 🔬 Teknik Detaylar

### ADAF Input Pipeline

```
DTM
  ↓
RVT Derivatives
  ├── SVF (Sky View Factor)
  ├── Positive Openness
  ├── Negative Openness
  ├── LRM (Local Relief Model)
  └── Slope
  ↓
Normalize (2-98 percentile)
  ↓
AiTLAS Ensemble (4 models)
  ↓
Average Predictions
  ↓
Probability Map
```

### Ensemble Stratejisi

```python
# Her model için prediction al
predictions = []
for model in adaf_models:
    pred = model(rvt_derivatives)
    predictions.append(pred)

# Ortalamasını al
final_prob = np.mean(predictions, axis=0)
```

Bu, tek modelden daha güvenilir ve robust sonuç verir.

---

## 📈 Performans

### Cache ile Hızlandırma

| Durum | RVT Hesaplama | ADAF Inference | Toplam |
|-------|---------------|----------------|--------|
| Cache YOK | ~15 dakika | ~10 dakika | ~25 dakika |
| Cache VAR | ~0 saniye | ~10 dakika | ~10 dakika |

**Öneri:** `cache_derivatives: true` kullanın!

### GPU Kullanımı

ADAF modelleri PyTorch tabanlı, GPU'yu destekler:
```yaml
# Otomatik GPU kullanımı (device parametresi sistemde var)
enable_adaf: true
```

---

## 🎨 QGIS'te Görselleştirme

1. **ADAF çıktısını yükle:**
   ```
   Layer → Add Vector Layer → *_adaf_mask.gpkg
   ```

2. **Renklendir:**
   ```
   Properties → Symbology
   Single Symbol → Kırmızı/Turuncu (arkeolojik özellikler)
   ```

3. **Olasılık haritasını yükle:**
   ```
   Layer → Add Raster Layer → *_adaf_prob.tif
   Symbology → Singleband pseudocolor
   ```

4. **Diğer yöntemlerle karşılaştır:**
   ```
   - *_dl_mask.gpkg (Derin öğrenme)
   - *_classic_mask.gpkg (Klasik)
   - *_adaf_mask.gpkg (ADAF) ← Arkeolojik özellikler için en iyi
   - *_yolo11_labels.gpkg (Genel envanter)
   ```

---

## 💡 Kullanım Senaryoları

### Senaryo 1: Sadece ADAF (Hızlı, Arkeolojik)

```yaml
enable_deep_learning: false
enable_classic: false
enable_yolo: false
enable_adaf: true
```

**Kullanım:** Sadece arkeolojik özellikler (barrows, ringforts, enclosures)

### Senaryo 2: ADAF + Klasik (Fusion)

```yaml
enable_deep_learning: false
enable_classic: true
enable_yolo: false
enable_adaf: true
enable_fusion: true
```

**Kullanım:** ADAF arkeolojik tespiti + RVT klasik yöntem doğrulaması

### Senaryo 3: Tüm Yöntemler (Maksimum)

```yaml
enable_deep_learning: true
enable_classic: true
enable_yolo: true
enable_adaf: true
enable_fusion: true
```

**Kullanım:**
- ADAF: Arkeolojik özellikler
- YOLO: Genel envanter (ağaç, bina, araç)
- U-Net: Sizin özel modeliniz
- Classic: Doğrulama ve fusion

---

## 🔍 ADAF vs Diğer Yöntemler

| Özellik | ADAF | U-Net (Sizinki) | YOLO | Klasik |
|---------|------|-----------------|------|--------|
| **Eğitim Verisi** | İrlanda ALS (arkeo) | Sizin verileriniz | COCO (genel) | - |
| **Arkeolojik Odak** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Kuş Bakışı** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Kurulum** | Orta (5GB) | Kolay | Kolay | Çok kolay |
| **Eğitim Gerekli** | ❌ Hayır | ✅ Evet | ⚠️ Fine-tune | ❌ Hayır |
| **Sınıflar** | 3 (arkeo özel) | Sizin sınıflarınız | 80 (genel) | - |

---

## 🐛 Sorun Giderme

### Problem 1: AiTLAS yüklenemiyor

```bash
# GDAL ve diğer geospatial bağımlılıkları önce yükleyin
pip install rasterio fiona shapely

# Sonra AiTLAS
pip install adaf/installation/aitlas-0.0.1-py3-none-any.whl
```

### Problem 2: Model dosyaları bulunamıyor

```bash
# Dizin yapısını kontrol edin
ls adaf/ml_models/*.tar

# Çıktı şöyle olmalı:
# segmentation_model_*.tar (4 adet)
# detection_model_*.tar (4 adet)
```

**Düzeltme:**
```yaml
# config.yaml - doğru yolu belirtin
adaf_models_dir: "C:/d_surucusu/arkeolojik_alan_tespit/adaf/ml_models"
```

### Problem 3: TAR dosyaları hatalı

**Hata:** `TorchModel.load_from_file() failed`

**Çözüm:**
- TAR dosyalarını AÇMAYIN (extract etmeyin)
- TAR dosyalarını DEĞİŞTİRMEYİN
- Yeniden indirin (bozulmuş olabilir)

### Problem 4: GPU kullanılmıyor

ADAF otomatik olarak sisteminizin GPU'sunu kullanır (PyTorch):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ADAF modelleri bu device'a yüklenir
```

### Problem 5: Çok yavaş

**Çözüm:** Cache kullanın!
```yaml
cache_derivatives: true  # RVT'yi cache'le
enable_adaf: true
```

İkinci çalıştırmada 2-3x daha hızlı!

---

## 📚 Detaylı Karşılaştırma

### ADAF Tespit Edebileceği Yapılar

**Barrows (Tümülüs/Höyük):**
- Yuvarlak veya oval şekilli
- Hafif yükseltilmiş (0.5-2m)
- Çap: 10-30m
- Prehistorik mezar yapıları
- Örnekler: Bronz Çağı tümülüsleri

**Ringforts (Dairesel Tahkimatlar):**
- Dairesel savunma yapıları
- İçi çukur, dışı yükseltilmiş
- Çap: 20-50m
- Erken Orta Çağ yerleşimleri
- İrlanda'da yaygın

**Enclosures (Çevrili Alanlar):**
- Duvar/hendek ile çevrili
- Dikdörtgen veya dairesel
- Boyut: değişken (20-200m)
- Çeşitli dönemler
- Yerleşim/tarım/ritüel alanlar

### Türkiye'deki Karşılıklar

ADAF İrlanda'da eğitilmiş ama benzer yapılar Türkiye'de de var:

| ADAF Sınıfı | Türkiye Karşılığı |
|-------------|-------------------|
| Barrows | Kurganer, tümülüsler, tepeler |
| Ringforts | Hisar kalıntıları, sur duvarları |
| Enclosures | Antik yerleşim alanları, kale çevreleri |

**Beklenen Performans:**
- ✅ Benzer yapılarda iyi performans
- ⚠️ Farklı yapılarda fine-tuning gerekebilir

---

## 🎓 İleri Seviye: Fine-Tuning

ADAF modellerini kendi verilerinizle fine-tune edebilirsiniz:

```python
from aitlas.models import TorchModel

# ADAF modelini yükle
model = TorchModel.load_from_file("adaf/ml_models/segmentation_model_resnet34.tar")

# Kendi verilerinizle fine-tune
# (AiTLAS API kullanarak)
model.train(your_dataset, epochs=50, lr=0.001)

# Kaydet
model.save("models/adaf_finetuned_turkey.tar")
```

```yaml
# config.yaml - fine-tuned modeli kullan
adaf_model_name: "adaf_finetuned_turkey"
```

---

## 📊 Örnek Sonuçlar

### Çıktı Karşılaştırması

**Girdi:** LiDAR DSM/DTM
**Yöntemler:**

1. **U-Net (DL):**
   ```
   ├── Genel arkeolojik yapılar
   ├── Sizin eğittiğiniz sınıflar
   └── Performans: Eğitime bağlı
   ```

2. **Klasik (RVT):**
   ```
   ├── Yükseklik anomalileri
   ├── Genel kabartı özellikleri
   └── Performans: İyi (sınıf ayrımı yok)
   ```

3. **YOLO:**
   ```
   ├── Ağaç, bina, araç, insan
   ├── Genel envanter
   └── Performans: Yatay perspektif için iyi
   ```

4. **ADAF (YENİ!):**
   ```
   ├── Barrows (tümülüs/höyük)
   ├── Ringforts (dairesel tahkimat)
   ├── Enclosures (çevrili alan)
   └── Performans: Arkeolojik için MÜKEMMEL ⭐
   ```

---

## ✅ Özet

### Avantajlar

✅ Arkeolojik özellikler için **özel eğitilmiş**
✅ Kuş bakışı (LiDAR) için **optimize**
✅ Ensemble öğrenme (4 model birlikte)
✅ Hazır modeller (~5GB indirme)
✅ Eğitim gerektirmez
✅ Cache sistemi ile **uyumlu**
✅ Vektörleştirme ve fusion desteği

### Dezavantajlar

⚠️ ~5GB model indirmesi gerekli
⚠️ AiTLAS kurulumu (ek bağımlılık)
⚠️ İrlanda'ya özgü (Türkiye için fine-tune gerekebilir)
⚠️ Sadece 3 sınıf (barrows, ringforts, enclosures)

### Ne Zaman Kullanılır?

**ADAF kullanın:**
- ✅ Barrows/tümülüs arıyorsanız
- ✅ Dairesel yapılar arıyorsanız
- ✅ Çevrili alanlar arıyorsanız
- ✅ Hazır model istiyorsanız
- ✅ LiDAR/kuş bakışı veriniz varsa

**U-Net kullanın:**
- ✅ Özel sınıflarınız varsa
- ✅ Kendi bölgenize özel eğitim yaptıysanız

**YOLO kullanın:**
- ✅ Genel envanter istiyorsanız (ağaç, bina, araç)
- ✅ RGB görüntüleriniz varsa

**Klasik kullanın:**
- ✅ Eğitim yoksa
- ✅ Doğrulama/fusion için

**Hepsini kullanın:**
- ✅ Maksimum tespit için
- ✅ Karşılaştırma için

---

## 🚀 Hızlı Başlangıç

```bash
# 1. ADAF klonla
git clone https://github.com/elestirmen/adaf.git

# 2. Modelleri indir (~5GB)
# GitHub'dan indirin ve adaf/ml_models/ dizinine koyun

# 3. AiTLAS yükle
pip install adaf/installation/aitlas-0.0.1-py3-none-any.whl

# 4. Config düzenle
# config.yaml: enable_adaf: true

# 5. Çalıştır
python archaeo_detect.py

# 6. Sonuçları kontrol et
qgis ciktilar/*_adaf_mask.gpkg
```

---

## 📝 Kaynaklar

- **ADAF GitHub:** https://github.com/elestirmen/adaf
- **AiTLAS Docs:** https://aitlas.readthedocs.io/
- **Paper:** (ADAF README'de belirtilmiş)

---

**ADAF ile arkeolojik tespitleriniz artık çok daha güçlü! 🏛️✨**

