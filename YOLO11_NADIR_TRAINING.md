# 🚁 YOLO11 Kuş Bakışı (Nadir) Görüntüler için Eğitim Rehberi

## ⚠️ Önemli Not

YOLO11'in varsayılan COCO ağırlıkları **yatay perspektiften** (sokak, ev içi görüntüler) eğitilmiştir. Arkeolojik alan tespitinde kullanılan LiDAR/İHA görüntüleri ise **kuş bakışı (nadir/vertical)** perspektiftedir.

### Neden Özel Eğitim Gerekli?

| Özellik | Yatay Perspektif (COCO) | Kuş Bakışı (Nadir) |
|---------|------------------------|-------------------|
| Görüş açısı | Yandan/önden | Üstten/dik |
| Nesne görünümü | Yan profil | Üst görünüş |
| Ölçek | Sabit mesafe | Yükseklikle değişir |
| Gölgeler | Yan/önde | Altında/uzun |
| Örnek | Sokaktaki araba | Havadan bakış araba |

**Sonuç:** Varsayılan YOLO11 kuş bakışı görüntülerde **kötü performans** gösterir veya **yanlış tespitler** yapar.

---

## 🎯 Çözüm: Fine-Tuning (İnce Ayar)

YOLO11'i kendi kuş bakışı verilerinizle yeniden eğitmeniz gerekir.

---

## 📦 Gereksinimler

```bash
pip install ultralytics>=8.1.0
pip install roboflow  # Veri seti yönetimi için (opsiyonel)
```

---

## 🗂️ Veri Seti Hazırlama

### 1. Veri Toplama

Kuş bakışı görüntülerinizi toplayın:
- İHA/drone görüntüleri
- Uydu görüntüleri
- LiDAR RGB ortofotoları
- Helikopter/uçak görüntüleri

**Minimum gereksinim:**
- Eğitim: 500-1000 görüntü
- Validasyon: 100-200 görüntü
- Test: 100-200 görüntü

### 2. Etiketleme (Annotation)

Görüntülerinizi etiketleyin. Önerilen araçlar:

**a) LabelImg (Basit)**
```bash
pip install labelImg
labelImg
```

**b) Roboflow (Bulut tabanlı, önerilen)**
- https://roboflow.com
- Drag-drop ile yükle
- Online etiketleme arayüzü
- Otomatik train/val/test split
- YOLO formatında export

**c) CVAT (Profesyonel)**
- https://cvat.org
- Takım çalışması için ideal
- Birden fazla etiketleyici

### 3. Veri Seti Yapısı

YOLO formatında organize edin:

```
nadir_dataset/
├── data.yaml           # Veri seti konfigürasyonu
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_0001.txt
│       ├── img_0002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 4. data.yaml Dosyası

```yaml
# nadir_dataset/data.yaml

# Veri yolları
path: /path/to/nadir_dataset  # Dataset kök dizini
train: train/images            # Eğitim görüntüleri
val: val/images                # Validasyon görüntüleri
test: test/images              # Test görüntüleri (opsiyonel)

# Sınıflar
names:
  0: vehicle       # Araç (araba, kamyon, vb.)
  1: building      # Bina
  2: tree          # Ağaç
  3: person        # İnsan
  4: excavation    # Kazı alanı
  5: mound         # Höyük/tümülüs
  6: wall          # Duvar kalıntısı
  # İhtiyacınıza göre ekleyin

# Sınıf sayısı
nc: 7
```

---

## 🎓 Model Eğitimi

### Seçenek 1: Scratch'ten Eğitim (Önerilmez)

```python
from ultralytics import YOLO

# Yeni model oluştur (ImageNet ağırlıklarıyla)
model = YOLO("yolo11n-seg.yaml")

# Eğit
results = model.train(
    data="nadir_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo11_nadir_scratch"
)
```

### Seçenek 2: Fine-Tuning (ÖNERİLEN)

```python
from ultralytics import YOLO

# Önceden eğitilmiş COCO modelini yükle
model = YOLO("yolo11n-seg.pt")  # veya yolo11s-seg.pt

# Fine-tune yap
results = model.train(
    data="nadir_dataset/data.yaml",
    epochs=50,                    # Fine-tuning için daha az epoch
    imgsz=640,
    batch=16,
    lr0=0.001,                    # Daha küçük learning rate
    patience=10,                  # Early stopping
    name="yolo11_nadir_finetuned",
    pretrained=True,              # COCO ağırlıklarını kullan
    # Veri artırma (data augmentation)
    hsv_h=0.015,                  # Renk tonu değişimi
    hsv_s=0.7,                    # Doygunluk
    hsv_v=0.4,                    # Parlaklık
    degrees=0.0,                  # Rotasyon (nadir için 0 tutun)
    translate=0.1,                # Kaydırma
    scale=0.5,                    # Ölçekleme
    shear=0.0,                    # Eğim (nadir için 0)
    perspective=0.0,              # Perspektif (nadir için 0)
    flipud=0.5,                   # Dikey flip (nadir için önemli!)
    fliplr=0.5,                   # Yatay flip
    mosaic=1.0,                   # Mosaic augmentation
    mixup=0.1,                    # Mixup
)
```

### Seçenek 3: Komut Satırı

```bash
# Fine-tuning
yolo segment train \
    data=nadir_dataset/data.yaml \
    model=yolo11n-seg.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    lr0=0.001 \
    patience=10 \
    name=yolo11_nadir \
    pretrained=True \
    flipud=0.5 \
    fliplr=0.5

# Daha büyük model (daha iyi sonuç)
yolo segment train \
    data=nadir_dataset/data.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=1280 \
    batch=8 \
    name=yolo11s_nadir_large
```

---

## 📊 Eğitim İzleme

Eğitim sırasında TensorBoard kullanın:

```bash
tensorboard --logdir runs/segment
```

Tarayıcıda açın: http://localhost:6006

**İzlenecek metrikler:**
- `box_loss`: Bounding box loss (düşmeli)
- `seg_loss`: Segmentasyon loss (düşmeli)
- `cls_loss`: Sınıflandırma loss (düşmeli)
- `mAP50`: Mean Average Precision @ IoU=0.5 (artmalı)
- `mAP50-95`: mAP @ IoU=0.5:0.95 (artmalı)

---

## ✅ Model Değerlendirme

```python
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/segment/yolo11_nadir/weights/best.pt")

# Test setinde değerlendir
metrics = model.val(data="nadir_dataset/data.yaml")

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

---

## 🚀 Modeli Kullanma

### archaeo_detect.py ile

```yaml
# config.yaml
enable_yolo: true
yolo_weights: "runs/segment/yolo11_nadir/weights/best.pt"  # Eğittiğiniz model
yolo_conf: 0.25
```

```bash
python archaeo_detect.py
```

### Tek Başına Test

```python
from ultralytics import YOLO

# Modelinizi yükle
model = YOLO("runs/segment/yolo11_nadir/weights/best.pt")

# Tahmin yap
results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)

# Sonuçları göster
for r in results:
    print(f"Tespit edilen: {len(r.boxes)} nesne")
    print(f"Sınıflar: {r.boxes.cls}")
```

---

## 🎯 İpuçları ve En İyi Uygulamalar

### 1. Veri Artırma (Data Augmentation)

Kuş bakışı görüntüler için:
```python
# ✅ ÖNERİLEN
flipud=0.5      # Dikey flip (kuş bakışında önemli)
fliplr=0.5      # Yatay flip
mosaic=1.0      # Mosaic augmentation
scale=0.5       # Ölçekleme
translate=0.1   # Kaydırma

# ❌ KULLANMAYIN
degrees=0.0     # Rotasyon (nadir için gerekli değil, hatta zararlı)
shear=0.0       # Eğim (nadir perspektif bozar)
perspective=0.0 # Perspektif (zaten üstten bakış)
```

### 2. Hiperparametre Optimizasyonu

```python
# Otomatik hiperparametre araması
model.tune(
    data="nadir_dataset/data.yaml",
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    plots=True,
    save=True
)
```

### 3. Model Karşılaştırma

| Model | Hız | Doğruluk | RAM | Kullanım |
|-------|-----|----------|-----|----------|
| yolo11n-seg | ⚡⚡⚡ | ⭐⭐ | 💾 | İlk test |
| yolo11s-seg | ⚡⚡ | ⭐⭐⭐ | 💾💾 | Önerilen |
| yolo11m-seg | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 | Yüksek doğruluk |
| yolo11l-seg | 🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Maksimum |

### 4. Transfer Learning Stratejisi

```python
# Freeze backbone, sadece head'i eğit (hızlı)
model = YOLO("yolo11n-seg.pt")
model.model.freeze(10)  # İlk 10 layer'ı dondur
results = model.train(data="nadir_dataset/data.yaml", epochs=20)

# Sonra tüm modeli fine-tune et
model.model.unfreeze()
results = model.train(data="nadir_dataset/data.yaml", epochs=30, lr0=0.0001)
```

---

## 📚 Hazır Veri Setleri

Kendi veriniz yoksa, bu veri setleriyle başlayabilirsiniz:

### 1. DOTA (Dataset for Object Detection in Aerial Images)
- **Link:** https://captain-whu.github.io/DOTA/
- **İçerik:** 2,806 aerial görüntü, 188,282 nesne
- **Sınıflar:** plane, ship, vehicle, bridge, vb.

### 2. xView
- **Link:** http://xviewdataset.org/
- **İçerik:** 1 million+ nesne, uydu görüntüleri
- **Sınıflar:** 60 sınıf (araç, bina, gemi, vb.)

### 3. VisDrone
- **Link:** https://github.com/VisDrone/VisDrone-Dataset
- **İçerik:** Drone görüntüleri, video + image
- **Sınıflar:** pedestrian, car, van, truck, vb.

### 4. UAVDT
- **Link:** https://sites.google.com/view/grli-uavdt/
- **İçerik:** UAV görüntüleri, araç tespiti

---

## 🔄 Örnek İş Akışı

### Başlangıçtan Üretime Kadar

```bash
# 1. Veri toplama (100-1000 görüntü)
# Manuel toplama veya mevcut veri setlerinden

# 2. Etiketleme
# Roboflow veya LabelImg ile

# 3. Veri seti hazırlama
python prepare_dataset.py --input raw_images/ --output nadir_dataset/

# 4. Quick test (5 epoch)
yolo segment train data=nadir_dataset/data.yaml model=yolo11n-seg.pt epochs=5 imgsz=640

# 5. Sonuç kontrolü
# runs/segment/train/results.png dosyasını incele

# 6. Full eğitim (50-100 epoch)
yolo segment train \
    data=nadir_dataset/data.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=1280 \
    batch=8 \
    patience=15 \
    name=yolo11s_nadir_final

# 7. En iyi modeli kopyala
cp runs/segment/yolo11s_nadir_final/weights/best.pt models/yolo11_nadir_best.pt

# 8. archaeo_detect.py ile kullan
# config.yaml: yolo_weights: "models/yolo11_nadir_best.pt"
python archaeo_detect.py
```

---

## 🐛 Sorun Giderme

### Problem 1: Düşük mAP (< 0.3)

**Çözümler:**
- Daha fazla veri toplayın (minimum 500 görüntü)
- Etiketleme kalitesini kontrol edin
- Daha büyük model kullanın (yolo11s veya yolo11m)
- Daha fazla epoch eğitin (100+)

### Problem 2: Overfitting (train loss düşüyor, val loss artıyor)

**Çözümler:**
- Daha fazla data augmentation kullanın
- Dropout artırın: `dropout=0.2`
- Early stopping: `patience=10`
- Daha küçük model kullanın

### Problem 3: GPU Belleği Yetersiz

**Çözümler:**
- Batch size düşürün: `batch=4` veya `batch=2`
- Image size küçültün: `imgsz=640` yerine `imgsz=512`
- Daha küçük model: `yolo11n-seg` yerine `yolo11n`
- Mixed precision: otomatik aktif

---

## 📝 Özet

1. ✅ Kuş bakışı görüntüler için **mutlaka fine-tuning** yapın
2. ✅ En az **500-1000 etiketli görüntü** toplayın
3. ✅ **yolo11s-seg.pt** ile başlayın (dengeli)
4. ✅ **flipud=0.5, fliplr=0.5** kullanın (nadir için önemli)
5. ✅ **50-100 epoch** eğitin
6. ✅ **mAP50 > 0.5** hedefleyin
7. ✅ Eğitilmiş modeli `config.yaml`'da belirtin

**Başarılar! 🚁📸**

---

## 📚 Ek Kaynaklar

- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/)
- [YOLO Transfer Learning Guide](https://docs.ultralytics.com/modes/train/)
- [Roboflow Blog - Aerial Object Detection](https://blog.roboflow.com/aerial-object-detection/)
- [Papers with Code - Aerial Detection](https://paperswithcode.com/task/aerial-object-detection)

---

**Not:** Bu rehber, YOLO11'i kuş bakışı görüntüler için nasıl eğiteceğinizi adım adım gösterir. Sorularınız için: [Issues](https://github.com/your-repo/issues)

