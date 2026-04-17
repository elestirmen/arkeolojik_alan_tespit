#!/bin/bash
# Kuş bakışı (nadir) görüntüler için YOLO eğitim scripti
# VisDrone veri setini indirir, dönüştürür ve eğitir

set -e  # Hata durumunda dur

echo "=============================================================================="
echo "🚁 YOLO Kuş Bakışı (Nadir) Model Eğitim Scripti"
echo "=============================================================================="
echo ""

# Parametreler
EPOCHS_TEST=${1:-5}        # Test için epoch sayısı (varsayılan: 5)
EPOCHS_FULL=${2:-100}      # Full eğitim için epoch (varsayılan: 100)
MODEL=${3:-yolov8s-seg.pt} # Model (varsayılan: yolov8s-seg.pt)
IMGSZ=${4:-1280}           # Image size (varsayılan: 1280)

echo "Parametreler:"
echo "  Test Epochs: $EPOCHS_TEST"
echo "  Full Epochs: $EPOCHS_FULL"
echo "  Model: $MODEL"
echo "  Image Size: $IMGSZ"
echo ""

# Adım 1: VisDrone İndir
echo "=============================================================================="
echo "📥 Adım 1/5: VisDrone Veri Seti İndiriliyor..."
echo "=============================================================================="

if [ ! -d "VisDrone2019-DET" ]; then
    echo "  → Train set indiriliyor (2.5GB)..."
    if [ ! -f "VisDrone2019-DET-train.zip" ]; then
        wget -q --show-progress https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-train.zip
    fi
    
    echo "  → Validation set indiriliyor (370MB)..."
    if [ ! -f "VisDrone2019-DET-val.zip" ]; then
        wget -q --show-progress https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-val.zip
    fi
    
    echo "  → Dosyalar açılıyor..."
    unzip -q VisDrone2019-DET-train.zip -d VisDrone2019-DET
    unzip -q VisDrone2019-DET-val.zip -d VisDrone2019-DET
    
    echo "  ✅ VisDrone indirildi!"
else
    echo "  ⏩ VisDrone zaten mevcut, atlıyor..."
fi

echo ""

# Adım 2: YOLO Formatına Çevir
echo "=============================================================================="
echo "🔄 Adım 2/5: YOLO Formatına Dönüştürülüyor..."
echo "=============================================================================="

if [ ! -d "visdrone_yolo" ]; then
    echo "  → Conversion script çalıştırılıyor..."
    python scripts/convert_visdrone_to_yolo.py \
        --input VisDrone2019-DET \
        --output visdrone_yolo
    
    echo "  ✅ Dönüştürme tamamlandı!"
else
    echo "  ⏩ visdrone_yolo zaten mevcut, atlıyor..."
fi

echo ""

# Adım 3: Quick Test
echo "=============================================================================="
echo "⚡ Adım 3/5: Quick Test ($EPOCHS_TEST epoch)..."
echo "=============================================================================="

echo "  → Test eğitimi başlıyor (30-60 dakika)..."
yolo segment train \
    data=visdrone_yolo/data.yaml \
    model=$MODEL \
    epochs=$EPOCHS_TEST \
    imgsz=640 \
    batch=8 \
    name=visdrone_test \
    verbose=True

echo ""
echo "  ✅ Test tamamlandı!"
echo "  📊 Sonuçlar: runs/segment/visdrone_test/"
echo ""

# Sonuçları göster
if [ -f "runs/segment/visdrone_test/results.csv" ]; then
    echo "  Son epoch metrikleri:"
    tail -n 1 runs/segment/visdrone_test/results.csv
fi

echo ""
echo "=============================================================================="
echo "❓ Full eğitim yapılsın mı?"
echo "=============================================================================="
echo ""
echo "Test sonuçlarını kontrol edin:"
echo "  - runs/segment/visdrone_test/results.png"
echo "  - runs/segment/visdrone_test/val_batch0_pred.jpg"
echo ""
read -p "Full eğitim yapılsın mı? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Adım 4: Full Eğitim
    echo ""
    echo "=============================================================================="
    echo "🎓 Adım 4/5: Full Eğitim ($EPOCHS_FULL epoch)..."
    echo "=============================================================================="
    echo "  → Full eğitim başlıyor (1-2 gün)..."
    echo "  💡 TensorBoard: tensorboard --logdir runs/segment"
    echo ""
    
    yolo segment train \
        data=visdrone_yolo/data.yaml \
        model=$MODEL \
        epochs=$EPOCHS_FULL \
        imgsz=$IMGSZ \
        batch=4 \
        lr0=0.001 \
        patience=15 \
        flipud=0.5 \
        fliplr=0.5 \
        mosaic=1.0 \
        name=visdrone_full \
        verbose=True \
        device=0
    
    echo ""
    echo "  ✅ Full eğitim tamamlandı!"
    echo ""
    
    # Adım 5: Modeli Kopyala
    echo "=============================================================================="
    echo "💾 Adım 5/5: Model Kaydediliyor..."
    echo "=============================================================================="
    
    mkdir -p models
    cp runs/segment/visdrone_full/weights/best.pt models/yolov8_nadir_visdrone.pt
    
    echo "  ✅ Model kaydedildi: models/yolov8_nadir_visdrone.pt"
    echo ""
    
    # Final metrikleri göster
    if [ -f "runs/segment/visdrone_full/results.csv" ]; then
        echo "  📊 Final Metrikler:"
        tail -n 1 runs/segment/visdrone_full/results.csv
    fi
    
    echo ""
    echo "=============================================================================="
    echo "✅ TÜM İŞLEMLER TAMAMLANDI!"
    echo "=============================================================================="
    echo ""
    echo "Modeli kullanmak için config.yaml'ı güncelleyin:"
    echo "  yolo_weights: \"models/yolov8_nadir_visdrone.pt\""
    echo ""
    echo "Sonra çalıştırın:"
    echo "  python archaeo_detect.py"
    echo ""
else
    echo ""
    echo "✅ Test tamamlandı, full eğitim atlandı."
    echo ""
    echo "Full eğitim için manuel olarak:"
    echo "  yolo segment train data=visdrone_yolo/data.yaml model=$MODEL epochs=$EPOCHS_FULL"
fi

echo ""
echo "🎉 Script tamamlandı!"

