# 🏛️ Arkeolojik Alan Tespiti - Proje Yapısı

Bu repository üç bağımsız arkeolojik alan tespit yöntemi içerir. Her biri kendi başına çalışabilir ve birbirinden bağımsızdır.

## 📁 Proje Yapısı

```
arkeolojik_alan_tespit/
├── archaeo_detect_base/      # 12 kanallı derin öğrenme yöntemi
├── archaeo_detect_enki/      # Enki yöntemi
├── archaeo_detect_adaf/      # ADAF yöntemi
└── README_PROJECTS.md        # Bu dosya
```

## 🎯 Projeler

### 1. archaeo_detect_base
**12 Kanallı Derin Öğrenme Yöntemi**

- **Girdi**: RGB + DSM + DTM + SVF + Openness + LRM + Slope + nDSM + TPI (12 kanal)
- **Model**: U-Net, DeepLabV3+ (Segmentation Models PyTorch)
- **Encoder**: ResNet, EfficientNet, DenseNet, vb.
- **Özellikler**: 
  - Klasik görüntü işleme desteği (RVT, Hessian, Morfoloji)
  - Fusion (DL + Klasik birleştirme)
  - YOLO11 desteği
  - Çoklu encoder ensemble

**Kullanım:**
```bash
cd archaeo_detect_base
pip install -r requirements.txt
python archaeo_detect.py
```

**Dokümantasyon:** `archaeo_detect_base/README.md`

---

### 2. archaeo_detect_enki
**Enki Yöntemi**

- **Girdi**: Uydu görüntüleri, GIS verileri
- **Model**: TensorFlow/Keras tabanlı derin öğrenme modelleri
- **Özellikler**:
  - Tell, yerleşim ve tarihi yapı tespiti
  - Google Satellite, Bing Maps entegrasyonu
  - Yüksek çözünürlüklü veri setleri ile eğitilmiş

**Kullanım:**
```bash
cd archaeo_detect_enki
pip install -r requirements.txt
# Enki'nin kendi scriptlerini kullanın
```

**Dokümantasyon:** `archaeo_detect_enki/README.md` ve `archaeo_detect_enki/enki/README.md`

---

### 3. archaeo_detect_adaf
**ADAF (Automatic Detection of Archaeological Features)**

- **Girdi**: ALS (Airborne Laser Scanning) verileri
- **Model**: İrlanda ALS verilerinden eğitilmiş özel modeller
- **Özellikler**:
  - Barrows, Ringforts, Enclosures tespiti
  - Kuş bakışı (nadir) görüntüler için optimize
  - Ensemble öğrenme (4 segmentation + 4 detection modeli)
  - Hazır modeller (eğitim gerektirmez)

**Kullanım:**
```bash
cd archaeo_detect_adaf
pip install -r requirements.txt
# AiTLAS wheel dosyalarını yükle
pip install installation/*.whl
# Modelleri ml_models/ klasörüne yerleştirin
python adaf_inference.py
```

**Dokümantasyon:** `archaeo_detect_adaf/README.md` ve `archaeo_detect_adaf/adaf/README.md`

---

## 🔄 Projeler Arası Farklar

| Özellik | Base | Enki | ADAF |
|---------|------|------|------|
| **Girdi Kanalları** | 12 (RGB + DSM/DTM + RVT + nDSM + TPI) | Değişken | ALS tabanlı |
| **Framework** | PyTorch | TensorFlow/Keras | PyTorch (AiTLAS) |
| **Eğitim Verisi** | Genel | Tell verileri | İrlanda ALS |
| **Tespit Türleri** | Genel arkeolojik | Tell, yerleşim | Barrows, Ringforts, Enclosures |
| **Klasik Yöntemler** | ✅ RVT, Hessian | ❌ | ✅ RVT |
| **Fusion** | ✅ | ❌ | ❌ |

## 📦 Bağımsız Kullanım

Her proje tamamen bağımsızdır:
- ✅ Kendi `requirements.txt` dosyası
- ✅ Kendi `README.md` dosyası
- ✅ Kendi konfigürasyon dosyaları
- ✅ Birbirine bağımlı değil

## 🤝 Katkıda Bulunma

Her proje kendi geliştirme sürecine sahiptir. İlgili projenin dizinine giderek katkıda bulunabilirsiniz.

## 📄 Lisans

Her proje kendi lisansına sahip olabilir. İlgili projenin dizinindeki LICENSE dosyasına bakın.

