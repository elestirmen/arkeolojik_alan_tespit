@echo off
REM Kuş bakışı (nadir) görüntüler için YOLO eğitim scripti (Windows)
REM VisDrone veri setini indirir, dönüştürür ve eğitir

setlocal EnableDelayedExpansion

echo ==============================================================================
echo 🚁 YOLO Kuş Bakışı (Nadir) Model Eğitim Scripti (Windows)
echo ==============================================================================
echo.

REM Parametreler
set EPOCHS_TEST=5
set EPOCHS_FULL=100
set MODEL=yolov8s-seg.pt
set IMGSZ=1280

if not "%1"=="" set EPOCHS_TEST=%1
if not "%2"=="" set EPOCHS_FULL=%2
if not "%3"=="" set MODEL=%3
if not "%4"=="" set IMGSZ=%4

echo Parametreler:
echo   Test Epochs: %EPOCHS_TEST%
echo   Full Epochs: %EPOCHS_FULL%
echo   Model: %MODEL%
echo   Image Size: %IMGSZ%
echo.

REM Adım 1: VisDrone İndir
echo ==============================================================================
echo 📥 Adım 1/5: VisDrone Veri Seti İndiriliyor...
echo ==============================================================================

if not exist "VisDrone2019-DET" (
    echo   → Train set indiriliyor ^(2.5GB^)...
    if not exist "VisDrone2019-DET-train.zip" (
        curl -L -o VisDrone2019-DET-train.zip https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-train.zip
    )
    
    echo   → Validation set indiriliyor ^(370MB^)...
    if not exist "VisDrone2019-DET-val.zip" (
        curl -L -o VisDrone2019-DET-val.zip https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.0.1/VisDrone2019-DET-val.zip
    )
    
    echo   → Dosyalar açılıyor...
    mkdir VisDrone2019-DET
    tar -xf VisDrone2019-DET-train.zip -C VisDrone2019-DET
    tar -xf VisDrone2019-DET-val.zip -C VisDrone2019-DET
    
    echo   ✅ VisDrone indirildi!
) else (
    echo   ⏩ VisDrone zaten mevcut, atlıyor...
)

echo.

REM Adım 2: YOLO Formatına Çevir
echo ==============================================================================
echo 🔄 Adım 2/5: YOLO Formatına Dönüştürülüyor...
echo ==============================================================================

if not exist "visdrone_yolo" (
    echo   → Conversion script çalıştırılıyor...
    python scripts\convert_visdrone_to_yolo.py --input VisDrone2019-DET --output visdrone_yolo
    echo   ✅ Dönüştürme tamamlandı!
) else (
    echo   ⏩ visdrone_yolo zaten mevcut, atlıyor...
)

echo.

REM Adım 3: Quick Test
echo ==============================================================================
echo ⚡ Adım 3/5: Quick Test ^(%EPOCHS_TEST% epoch^)...
echo ==============================================================================

echo   → Test eğitimi başlıyor ^(30-60 dakika^)...
yolo segment train data=visdrone_yolo/data.yaml model=%MODEL% epochs=%EPOCHS_TEST% imgsz=640 batch=8 name=visdrone_test verbose=True

echo.
echo   ✅ Test tamamlandı!
echo   📊 Sonuçlar: runs\segment\visdrone_test\
echo.

REM Sonuçları göster
if exist "runs\segment\visdrone_test\results.csv" (
    echo   Son epoch metrikleri:
    powershell -Command "Get-Content runs\segment\visdrone_test\results.csv | Select-Object -Last 1"
)

echo.
echo ==============================================================================
echo ❓ Full eğitim yapılsın mı?
echo ==============================================================================
echo.
echo Test sonuçlarını kontrol edin:
echo   - runs\segment\visdrone_test\results.png
echo   - runs\segment\visdrone_test\val_batch0_pred.jpg
echo.

set /p CHOICE="Full eğitim yapılsın mı? (y/N): "
if /i "%CHOICE%"=="y" goto FULL_TRAINING
if /i "%CHOICE%"=="Y" goto FULL_TRAINING
goto SKIP_FULL

:FULL_TRAINING
REM Adım 4: Full Eğitim
echo.
echo ==============================================================================
echo 🎓 Adım 4/5: Full Eğitim ^(%EPOCHS_FULL% epoch^)...
echo ==============================================================================
echo   → Full eğitim başlıyor ^(1-2 gün^)...
echo   💡 TensorBoard: tensorboard --logdir runs\segment
echo.

yolo segment train data=visdrone_yolo/data.yaml model=%MODEL% epochs=%EPOCHS_FULL% imgsz=%IMGSZ% batch=4 lr0=0.001 patience=15 flipud=0.5 fliplr=0.5 mosaic=1.0 name=visdrone_full verbose=True device=0

echo.
echo   ✅ Full eğitim tamamlandı!
echo.

REM Adım 5: Modeli Kopyala
echo ==============================================================================
echo 💾 Adım 5/5: Model Kaydediliyor...
echo ==============================================================================

if not exist "models" mkdir models
copy runs\segment\visdrone_full\weights\best.pt models\yolov8_nadir_visdrone.pt

echo   ✅ Model kaydedildi: models\yolov8_nadir_visdrone.pt
echo.

REM Final metrikleri göster
if exist "runs\segment\visdrone_full\results.csv" (
    echo   📊 Final Metrikler:
    powershell -Command "Get-Content runs\segment\visdrone_full\results.csv | Select-Object -Last 1"
)

echo.
echo ==============================================================================
echo ✅ TÜM İŞLEMLER TAMAMLANDI!
echo ==============================================================================
echo.
echo Modeli kullanmak için config.yaml'ı güncelleyin:
echo   yolo_weights: "models/yolov8_nadir_visdrone.pt"
echo.
echo Sonra çalıştırın:
echo   python archaeo_detect.py
echo.

goto END

:SKIP_FULL
echo.
echo ✅ Test tamamlandı, full eğitim atlandı.
echo.
echo Full eğitim için manuel olarak:
echo   yolo segment train data=visdrone_yolo/data.yaml model=%MODEL% epochs=%EPOCHS_FULL%
echo.

:END
echo 🎉 Script tamamlandı!
pause

