# 🚀 YOLO Hızlı Başlangıç - Kuş Bakışı Görüntüler

## ✅ YOLO11 ile Hemen Başlayın (Varsayılan)

Sistem hem YOLO11 hem YOLOv8 modellerini destekler. Varsayılan akışta YOLO11 kullanılır.

---

## 📦 1. Seçenek: Varsayılan (Test Amaçlı)

Hiçbir şey yapmadan direkt çalıştırın:

```bash
python archaeo_detect.py
```

**Sonuç:**
- ✅ YOLO11 otomatik indirilir
- ⚠️ Uyarı mesajı görünür (kuş bakışı için optimize değil)
- 📊 Genel envanter çıktısı (düşük doğruluk)

---

## 🎯 2. Seçenek: VisDrone ile Eğitim (1-2 Gün)

### Adım 1: VisDrone İndir

```bash
# Train set (2.5GB)
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-train.zip

# Validation set (370MB)
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-val.zip
```

### Adım 2: YOLO Formatına Çevir

```bash
# Conversion script'i çalıştır
python scripts/convert_visdrone_to_yolo.py \
    --input VisDrone2019-DET \
    --output visdrone_yolo
```

**Çıktı:**
```
visdrone_yolo/
├── data.yaml
├── images/
│   ├── train/    (6,471 görüntü)
│   └── val/      (548 görüntü)
└── labels/
    ├── train/
    └── val/
```

### Adım 3: Quick Test (30 dakika)

```bash
# 5 epoch ile hızlı test
yolo segment train \
    data=visdrone_yolo/data.yaml \
    model=yolov8s-seg.pt \
    epochs=5 \
    imgsz=640 \
    batch=8 \
    name=visdrone_test

# Sonuçları kontrol et
firefox runs/segment/visdrone_test/results.png
```

**Beklenen Sonuç:**
- mAP50 > 0.2 (5 epoch için makul)
- Loss düşüyor mu? ✅

### Adım 4: Full Eğitim (1-2 gün)

```bash
# 100 epoch ile tam eğitim
yolo segment train \
    data=visdrone_yolo/data.yaml \
    model=yolov8s-seg.pt \
    epochs=100 \
    imgsz=1280 \
    batch=4 \
    lr0=0.001 \
    patience=15 \
    flipud=0.5 \
    fliplr=0.5 \
    mosaic=1.0 \
    name=visdrone_full \
    device=0  # GPU kullan

# TensorBoard ile izle
tensorboard --logdir runs/segment
# http://localhost:6006
```

**Beklenen Sonuç:**
- mAP50 > 0.5 (iyi)
- mAP50-95 > 0.3 (iyi)

### Adım 5: Modeli Kullan

```bash
# En iyi modeli kopyala
mkdir -p models
cp runs/segment/visdrone_full/weights/best.pt models/yolov8_nadir_visdrone.pt
```

**config.yaml:**
```yaml
enable_yolo: true
yolo_weights: "models/yolov8_nadir_visdrone.pt"
yolo_conf: 0.25
```

**Çalıştır:**
```bash
python archaeo_detect.py
# Artık uyarı YOK, yüksek doğruluk VAR! ✅
```

---

## 🎨 3. Seçenek: Farklı Modeller Dene

### YOLOv8 Model Karşılaştırması

```bash
# Nano (en hızlı)
yolo segment train data=visdrone_yolo/data.yaml model=yolov8n-seg.pt epochs=100

# Small (dengeli) ⭐ ÖNERİLEN
yolo segment train data=visdrone_yolo/data.yaml model=yolov8s-seg.pt epochs=100

# Medium (yüksek doğruluk)
yolo segment train data=visdrone_yolo/data.yaml model=yolov8m-seg.pt epochs=100

# Large (en iyi doğruluk, en yavaş)
yolo segment train data=visdrone_yolo/data.yaml model=yolov8l-seg.pt epochs=100
```

| Model | Hız | Doğruluk | GPU RAM | Eğitim Süresi |
|-------|-----|----------|---------|---------------|
| yolov8n-seg | ⚡⚡⚡ | ⭐⭐ | 2GB | 12 saat |
| yolov8s-seg | ⚡⚡ | ⭐⭐⭐ | 4GB | 24 saat |
| yolov8m-seg | ⚡ | ⭐⭐⭐⭐ | 8GB | 48 saat |
| yolov8l-seg | 🐌 | ⭐⭐⭐⭐⭐ | 12GB | 72 saat |

---

## 🔄 4. Seçenek: YOLO11 de Deneyin

YOLO11 daha yeni ama daha iyi performans:

```bash
# YOLO11 ile eğitim
yolo segment train \
    data=visdrone_yolo/data.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=1280 \
    batch=4

# Model kullanımı
# config.yaml: yolo_weights: "runs/segment/train/weights/best.pt"
```

**Karşılaştırma:**
- **YOLOv8:** Daha olgun, daha fazla topluluk desteği
- **YOLO11:** Daha hızlı, biraz daha yüksek doğruluk

---

## 💡 5. İpuçları ve Optimizasyon

### GPU Belleği Yetersizse

```bash
# Batch size düşür
yolo segment train ... batch=2

# Image size küçült
yolo segment train ... imgsz=640

# Daha küçük model
yolo segment train ... model=yolov8n-seg.pt
```

### Overfitting Oluyorsa

```bash
# Daha fazla augmentation
yolo segment train \
    ... \
    hsv_h=0.02 \
    hsv_s=0.8 \
    hsv_v=0.5 \
    scale=0.9 \
    mosaic=1.0 \
    mixup=0.15

# Early stopping
yolo segment train ... patience=10
```

### Transfer Learning

```bash
# İlk 10 layer'ı dondur (hızlı)
from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt")
model.model.freeze(10)
model.train(data="visdrone_yolo/data.yaml", epochs=20)

# Sonra tüm modeli fine-tune
model.model.unfreeze()
model.train(data="visdrone_yolo/data.yaml", epochs=50, lr0=0.0001)
```

---

## 📊 6. Sonuçları Değerlendirme

### Eğitim İstatistikleri

```bash
# TensorBoard
tensorboard --logdir runs/segment

# Metriklere bakın:
# - box_loss ↓ (düşmeli)
# - seg_loss ↓ (düşmeli)
# - cls_loss ↓ (düşmeli)
# - mAP50 ↑ (artmalı)
# - mAP50-95 ↑ (artmalı)
```

### Test Sonuçları

```python
from ultralytics import YOLO

# Modeli yükle
model = YOLO("models/yolov8_nadir_visdrone.pt")

# Validasyon seti
metrics = model.val(data="visdrone_yolo/data.yaml")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Tek görüntü
results = model.predict("test_image.jpg", conf=0.25, save=True)
```

### Başarı Kriterleri

| Metrik | Hedef | İyi | Mükemmel |
|--------|-------|-----|----------|
| mAP50 | > 0.3 | > 0.5 | > 0.7 |
| mAP50-95 | > 0.2 | > 0.3 | > 0.5 |
| Inference Hızı | < 100ms | < 50ms | < 20ms |

---

## 🎯 Hızlı Karar Ağacı

```
Hangi yöntemi seçmeliyim?
│
├─ Hemen test etmek istiyorum
│  └─ Seçenek 1: Varsayılan (python archaeo_detect.py)
│
├─ 1-2 gün ayırabiliyorum
│  └─ Seçenek 2: VisDrone ile eğitim (ÖNERİLEN)
│
├─ Kendi verilerim var (100+ görüntü)
│  └─ Kendi verilerinle eğit (en iyi sonuç)
│
└─ GPU/zaman yok
   └─ Bulut servisleri (Colab, Paperspace, Lambda Labs)
```

---

## 🚀 Tek Komutla Başlangıç

```bash
# Tüm işlemi otomatikleştirin
bash scripts/train_yolo_nadir.sh
```

**train_yolo_nadir.sh:**
```bash
#!/bin/bash
set -e

echo "VisDrone indiriliyor..."
wget -q https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-train.zip
wget -q https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-val.zip
unzip -q VisDrone2019-DET-train.zip
unzip -q VisDrone2019-DET-val.zip

echo "YOLO formatına çevriliyor..."
python scripts/convert_visdrone_to_yolo.py

echo "Quick test (5 epoch)..."
yolo segment train data=visdrone_yolo/data.yaml model=yolov8s-seg.pt epochs=5 batch=8 name=test

echo "Sonuçlar:"
cat runs/segment/test/results.txt

echo "✅ Test tamamlandı!"
echo "Full eğitim için:"
echo "  yolo segment train data=visdrone_yolo/data.yaml model=yolov8s-seg.pt epochs=100"
```

---

## ✅ Özet

1. **Test için:** Direkt `python archaeo_detect.py` ⚡ (5 dakika)
2. **Üretim için:** VisDrone ile eğitim ⭐ (1-2 gün)
3. **En iyi için:** Kendi verilerinle eğitim 🏆 (1 hafta)

**Başarılar! 🚁📸**

