# YOLO11 Entegrasyonu - Arkeolojik Alan Tespit Sistemi

## 🎯 Genel Bakış

YOLO11 (Ultralytics), arkeolojik alan tespit sistemine başarıyla entegre edilmiştir. Artık sistem üç farklı tespit yöntemi sunmaktadır:

1. **Derin Öğrenme (U-Net)** - Segmentation Models PyTorch ile
2. **Klasik Görüntü İşleme** - RVT, Hessian, Morfoloji
3. **YOLO11 (YENİ!)** - Ultralytics nesne tespit/segmentasyon

## 📦 Kurulum

YOLO11 kullanmak için Ultralytics kütüphanesini yüklemelisiniz:

```bash
pip install ultralytics>=8.1.0
```

## 🚀 Kullanım

### Temel Kullanım (config.yaml ile)

1. `config.yaml` dosyasını düzenleyin:

```yaml
# YOLO11'i etkinleştir
enable_yolo: true

# YOLO11 ayarları
yolo_weights: null  # Otomatik olarak yolo11n-seg.pt indirilir
yolo_conf: 0.25     # Güven eşiği
yolo_iou: 0.45      # NMS IoU eşiği
yolo_imgsz: 640     # Model girdi boyutu
```

2. Çalıştırın:

```bash
python archaeo_detect.py
```

### Komut Satırı ile

```bash
# YOLO11'i etkinleştir
python archaeo_detect.py --enable-yolo

# Özel model kullan
python archaeo_detect.py --enable-yolo --yolo-weights path/to/your/best.pt

# Güven eşiğini ayarla
python archaeo_detect.py --enable-yolo --yolo-conf 0.3

# Sadece YOLO11 kullan (diğer yöntemler kapalı)
python archaeo_detect.py --enable-yolo --no-enable-deep-learning --no-enable-classic
```

## ⚠️ ÖNEMLİ: Kuş Bakışı (Nadir) Görüntüler Hakkında

### YOLO11 ve Perspektif Sorunu

**YOLO11'in varsayılan COCO ağırlıkları YATAY perspektiften eğitilmiştir!**

| Özellik | COCO (Varsayılan) | Arkeolojik Alan |
|---------|-------------------|-----------------|
| Perspektif | Yatay (yan/önden) | Kuş bakışı (üstten) |
| Görüntü kaynağı | Sokak kameraları | LiDAR/İHA/Uydu |
| Nesne görünümü | Yan profil | Üst görünüş |
| Performans | ✅ Yüksek | ❌ Düşük |

**Sonuç:** Varsayılan YOLO11 modeli kuş bakışı görüntülerde **kötü performans** gösterir veya **yanlış tespitler** yapar.

### 🛠️ Çözüm

**Seçenek 1: Fine-Tuning (ÖNERİLEN)**
```bash
# Kendi kuş bakışı verilerinizle eğitin
yolo segment train \
    data=nadir_dataset/data.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=1280
```

Detaylı rehber: **YOLO11_NADIR_TRAINING.md**

**Seçenek 2: Hazır Nadir Modeller**
- DroneVision veri setiyle eğitilmiş modeller
- Aerial Object Detection modelleri
- Kendinizin eğittiği modeller

**Seçenek 3: Test Amaçlı Kullanım**
- Varsayılan modeli genel envanter için kullanabilirsiniz
- Ama sonuçlar düşük doğrulukta olacaktır
- Arkeolojik yapı tespiti için **güvenmeyin**!

---

## 🎨 YOLO11 Model Seçenekleri

### Segmentasyon Modelleri (Önerilen)

Piksel seviyesinde maske üretir:

**Varsayılan (COCO) - YATAY Perspektif:**
- **yolo11n-seg.pt** - Nano (3.4M parametre) - Hızlı, hafif
- **yolo11s-seg.pt** - Small (11M parametre) - Dengeli
- **yolo11m-seg.pt** - Medium (27M parametre) - Yüksek doğruluk
- **yolo11l-seg.pt** - Large (46M parametre) - Çok yüksek doğruluk
- **yolo11x-seg.pt** - Extra Large (71M parametre) - Maksimum doğruluk

**Özel Eğitilmiş - KUŞ BAKIŞI (Nadir):**
- **models/yolo11_nadir_best.pt** - Kendi eğittiğiniz nadir model
- **models/yolo11_archaeological.pt** - Arkeolojik özel model
- **yolo11-aerial.pt** - Hazır aerial detection modeli (varsa)

### Tespit Modelleri (Detection)

Bounding box üretir (maske yerine dikdörtgen):

- **yolo11n.pt** - Nano detection
- **yolo11s.pt** - Small detection
- **yolo11m.pt** - Medium detection

**Not:** Arkeolojik yapı tespiti için segmentasyon modellerini kullanmanız önerilir.

## ⚙️ Parametreler

### yolo_weights (str veya null)
- **null**: Otomatik olarak `yolo11n-seg.pt` indirilir
- **"yolo11s-seg.pt"**: Daha iyi sonuç için
- **"path/to/model.pt"**: Özel eğitilmiş modeliniz

### yolo_conf (float, 0.0-1.0)
- Tespit güven eşiği
- **0.15-0.20**: Çok hassas (daha fazla tespit)
- **0.25**: Dengeli (varsayılan)
- **0.35-0.50**: Seçici (daha az tespit)

### yolo_iou (float, 0.0-1.0)
- NMS IoU eşiği (çakışan tespitler için)
- **0.3-0.4**: Agresif NMS
- **0.45**: Dengeli (varsayılan)
- **0.5-0.6**: Yumuşak NMS

### yolo_tile (int veya null)
- YOLO için özel tile boyutu
- **null**: Genel `--tile` parametresini kullanır
- **640**: Küçük yapılar için iyi
- **1280**: Orta boy yapılar için

### yolo_imgsz (int)
- YOLO model girdi boyutu
- **640**: Dengeli (varsayılan)
- **1280**: Yüksek doğruluk (yavaş)

### yolo_device (str veya null)
- Hesaplama cihazı
- **null**: Otomatik (GPU varsa GPU)
- **"0"**: İlk GPU
- **"cpu"**: CPU'yu zorla
- **"mps"**: Apple Silicon (M1/M2)

## 📊 Çıktı Dosyaları

YOLO11 etkinse aşağıdaki dosyalar üretilir:

```
ciktilar/
├── kesif_alani_yolo11_th0.25_tile1024_minarea80_prob.tif    # Olasılık haritası
├── kesif_alani_yolo11_th0.25_tile1024_minarea80_mask.tif    # İkili maske
├── kesif_alani_yolo11_th0.25_tile1024_minarea80_mask.gpkg   # Vektör (opsiyonel)
└── kesif_alani_yolo11_th0.25_tile1024_minarea80_labels.gpkg # 🆕 Etiketli tespitler
```

### 🏷️ Etiketli Tespitler (YENİ!)

`*_labels.gpkg` dosyası, YOLO'nun tespit ettiği **tüm nesneleri** sınıf etiketleriyle birlikte içerir:

**Attribute Table (Öznitelikler):**
- **id**: Tespit numarası
- **class_id**: COCO sınıf ID'si (0-79)
- **class_name**: Sınıf adı (person, car, tree, building, vb.)
- **confidence**: Güven skoru (0-1)
- **area_m2**: Alan (metrekare)
- **center_x, center_y**: Merkez koordinatları
- **bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax**: Bounding box koordinatları
- **tile_row, tile_col**: Hangi tile'da tespit edildi
- **geometry**: Polygon (segmentasyon) veya bbox (detection)

**COCO Sınıfları (Örnekler):**
- 0: person (insan)
- 1: bicycle (bisiklet)
- 2: car (araba)
- 3: motorcycle (motosiklet)
- 5: bus (otobüs)
- 7: truck (kamyon)
- 14: bird (kuş)
- 15: cat (kedi)
- 16: dog (köpek)
- 24: backpack (sırt çantası)
- ...ve 80 sınıf daha

**Kullanım Senaryoları:**
- 🌳 Arazideki ağaçları haritalama
- 🏗️ Binaları tespit etme
- 🚗 Araçları sayma
- 👥 İnsan aktivitelerini belirleme
- 📊 Arazi kullanım analizi
- 🗺️ Genel arazi envanteri

## 🔧 Teknik Detaylar

### YOLO11'in Çalışma Mantığı

1. **RGB Okuma**: Sadece RGB bantları kullanılır (DSM/DTM görmez)
2. **Tile-based Inference**: Görüntü karolara bölünür
3. **RGB Stretch**: Görüntü 8-bit'e normalize edilir (percentile stretch)
4. **YOLO Inference**: Her tile üzerinde tespit yapılır
5. **Maske Üretimi**:
   - Segmentasyon modeli → Piksel maskesi
   - Detection modeli → Bounding box'tan maske
6. **Birleştirme**: Tile'lar feathering ile birleştirilir
7. **Vektörleştirme**: Maske poligonlara dönüştürülür

### Cache Desteği

YOLO11, cache sistemi ile uyumludur:

```yaml
cache_derivatives: true  # RVT türevlerini önbellekle
```

Cache aktifse, RGB bantları cache'den okunur ve YOLO çok daha hızlı çalışır.

## 📈 Performans İpuçları

1. **GPU Kullanın**: CPU'ya göre 10-50x daha hızlı
2. **Küçük Model**: İlk denemeler için `yolo11n-seg.pt` kullanın
3. **Cache Aktif**: `cache_derivatives: true` ile RVT cache'i kullanın
4. **Tile Boyutu**: 640-1280 arası optimal

## 🎯 Kullanım Senaryoları

### Senaryo 1: Sadece YOLO11
Hızlı RGB tabanlı tespit:
```yaml
enable_deep_learning: false
enable_classic: false
enable_yolo: true
```

### Senaryo 2: YOLO11 + Klasik (Fusion)
RGB ve yükseklik verisi birleşimi:
```yaml
enable_deep_learning: false
enable_classic: true
enable_yolo: true
enable_fusion: true
```

### Senaryo 3: Tüm Yöntemler
Maksimum doğruluk için:
```yaml
enable_deep_learning: true
enable_classic: true
enable_yolo: true
enable_fusion: true
```

## 🔍 Karşılaştırma

| Yöntem | Girdi | Hız | Doğruluk | Model Gereksinimi |
|--------|-------|-----|----------|-------------------|
| U-Net | RGB+DSM+DTM | Orta | Yüksek | Eğitilmiş model |
| Klasik | DSM+DTM | Hızlı | Orta | Yok |
| YOLO11 | RGB | Çok Hızlı | Orta-Yüksek | Opsiyonel |

## 🐛 Sorun Giderme

### YOLO kurulu değil
```bash
pip install ultralytics>=8.1.0
```

### GPU kullanılmıyor
```yaml
yolo_device: "0"  # İlk GPU'yu zorla
```

### Çok az tespit
```yaml
yolo_conf: 0.15  # Eşiği düşür
```

### Çok fazla yanlış tespit
```yaml
yolo_conf: 0.4   # Eşiği yükselt
```

### Segmentasyon maskesi boş
- Segmentasyon modeli kullanıyor musunuz? (`-seg.pt`)
- Detection modeli kullanıyorsanız, bounding box'lar maskye dönüştürülür

## 📝 Örnek Çalıştırmalar

### Zero-shot ile Hızlı Test
```bash
python archaeo_detect.py --enable-yolo --no-enable-deep-learning --no-enable-classic
```

### Özel Model ile Yüksek Doğruluk
```bash
python archaeo_detect.py \
  --enable-yolo \
  --yolo-weights my_trained_model.pt \
  --yolo-conf 0.3 \
  --yolo-imgsz 1280
```

### Cache ile Hızlandırılmış Çalıştırma
```bash
python archaeo_detect.py \
  --enable-yolo \
  --cache-derivatives \
  --tile 1024 \
  --yolo-tile 640
```

## 🎓 İleri Seviye: Özel Model Eğitimi

YOLO11'i kendi verilerinizle eğitebilirsiniz:

```python
from ultralytics import YOLO

# Model oluştur
model = YOLO("yolo11n-seg.pt")

# Eğit
model.train(
    data="your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="archaeological_sites"
)

# Kullan
# --yolo-weights runs/segment/archaeological_sites/weights/best.pt
```

## 📚 Kaynaklar

- [Ultralytics YOLO11 Dökümantasyonu](https://docs.ultralytics.com/)
- [YOLO11 GitHub](https://github.com/ultralytics/ultralytics)
- [Model Zoo](https://docs.ultralytics.com/models/yolo11/)

## 🎨 QGIS'te Görselleştirme

Etiketli tespitleri QGIS'te görselleştirmek için:

1. **GeoPackage'i Yükle:**
   - Layer → Add Layer → Add Vector Layer
   - `*_labels.gpkg` dosyasını seç

2. **Sınıflara Göre Renklendirme:**
   - Layer Properties → Symbology
   - "Categorized" seç
   - Column: `class_name`
   - Classify düğmesine bas
   - Her sınıf farklı renk alır

3. **Etiketleri Göster:**
   - Layer Properties → Labels
   - "Single Labels" seç
   - Value: `class_name` veya `concat(class_name, ' (', round(confidence, 2), ')')`

4. **Güven Skoruna Göre Filtrele:**
   - Layer Properties → Source → Query Builder
   - `confidence > 0.5` (sadece yüksek güvenli tespitler)

## 📊 Örnek Analiz

```python
import geopandas as gpd

# Etiketli tespitleri oku
gdf = gpd.read_file("ciktilar/kesif_alani_yolo11_labels.gpkg")

# Sınıf bazlı özet
print(gdf['class_name'].value_counts())

# Yüksek güvenli tespitler
high_conf = gdf[gdf['confidence'] > 0.7]

# Büyük nesneler (100m² üzeri)
large_objects = gdf[gdf['area_m2'] > 100]

# Belirli sınıfları filtrele (örn: araçlar)
vehicles = gdf[gdf['class_name'].isin(['car', 'truck', 'bus'])]

# CSV'ye aktar
gdf.to_csv("detections_summary.csv", index=False)
```

## ✅ Özet

YOLO11 entegrasyonu ile sistem artık:
- ✅ RGB görüntülerden direkt tespit yapabilir
- ✅ Çok hızlı çalışır (özellikle GPU ile)
- ✅ Zero-shot modda kullanılabilir
- ✅ Özel modeller ile genişletilebilir
- ✅ Cache sistemi ile uyumlu
- ✅ Mevcut pipeline ile tam entegre
- ✅ **Tüm nesneleri etiketli olarak GeoPackage'e kaydeder** 🎯

**Mutlu tespitler! 🏛️🔍**

