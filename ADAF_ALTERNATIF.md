# 🔧 ADAF Kurulum Sorunu ve Alternatif Çözümler

## ⚠️ Sorun

AiTLAS kütüphanesi eski bağımlılıklar gerektiriyor:
- h5py < 3.2.1
- numpy == 1.19.3
- Python 3.9

Sisteminiz Python 3.10/3.12 kullanıyor ve bu eski paketler uyumsuz.

---

## 🛠️ Çözüm Seçenekleri

### Seçenek 1: Basitleştirilmiş ADAF Wrapper (ÖNERİLEN)

ADAF modellerini kullanabilmek için basit bir wrapper oluşturabiliriz.

**Adımlar:**

1. ADAF modellerini PyTorch formatına çevirin
2. Sisteminizle uyumlu hale getirin
3. Doğrudan kullanın

**Implementasyon:** Aşağıdaki script'i kullanın.

### Seçenek 2: Docker Container

```bash
# ADAF için Docker container
docker pull earthobservation/adaf:latest

# Container'da çalıştır
docker run -v $(pwd):/data earthobservation/adaf \
    python ADAF_main.ipynb
```

### Seçenek 3: Ayrı Python 3.9 Environment

```bash
# Conda ile ayrı environment
conda create -n adaf_py39 python=3.9
conda activate adaf_py39

# ADAF kurulumu
cd adaf
pip install installation/aitlas-0.0.1-py3-none-any.whl

# ADAF'i ayrı çalıştır, sonra sonuçları ana sistemle birleştir
```

### Seçenek 4: ADAF Modellerini Manuel Yükle (BASIT)

ADAF TAR dosyalarını manuel olarak PyTorch formatına çevirebiliriz.

---

## 🚀 ÖNERİLEN: Basit PyTorch Wrapper

Sisteminize ADAF'i eklemek yerine, ADAF modellerini PyTorch'a çevirebilirim:

```python
# adaf_pytorch_wrapper.py
import torch
import tarfile
from pathlib import Path

def extract_adaf_model_to_pytorch(tar_path: Path) -> torch.nn.Module:
    """
    ADAF TAR dosyasından PyTorch modelini çıkar.
    """
    with tarfile.open(tar_path, 'r') as tar:
        # Model state_dict'i çıkar
        tar.extractall('temp_adaf/')
    
    # PyTorch modelini yükle
    state_dict = torch.load('temp_adaf/model.pth')
    
    # U-Net benzeri mimari oluştur
    model = build_unet_like_adaf()
    model.load_state_dict(state_dict)
    
    return model
```

Bu yaklaşımla AiTLAS'a ihtiyaç duymadan ADAF modellerini kullanabilirsiniz.

---

## 💡 Size Önerim

**Kısa vadeli (Şimdi):**
1. ADAF'i şimdilik devre dışı bırakın
2. U-Net + YOLO + Klasik yöntemlerle devam edin
3. Sonuçlar zaten çok iyi!

```yaml
# config.yaml - şimdilik
enable_adaf: false  # AiTLAS sorunu çözülene kadar
enable_deep_learning: true
enable_classic: true
enable_yolo: true
```

**Orta vadeli (1-2 hafta):**
- ADAF modellerini PyTorch formatına çevireyim
- Sisteminize entegre edeyim
- AiTLAS'a ihtiyaç kalmaz

**Uzun vadeli:**
- Kendi arkeolojik verilerinizle model eğitin
- ADAF'ten daha iyi performans (bölgenize özel)

---

## 🎯 Hemen Yapılabilecekler

### Şu An İçin: YOLO + Klasik

```yaml
# config.yaml
enable_deep_learning: false  # Model yoksa kapalı
enable_classic: true
enable_yolo: true  # Genel envanter için
enable_adaf: false  # AiTLAS sorunu çözülene kadar
```

```bash
python archaeo_detect.py
```

**Çıktı:**
- Klasik RVT tespitleri (arkeolojik anomaliler)
- YOLO genel envanter (ağaç, bina, araç)

---

## ✅ Ne Yapalım?

Size üç seçenek sunuyorum:

**A) Basitleştirilmiş ADAF wrapper yapalım**
- ADAF modellerini PyTorch'a çevirelim
- AiTLAS'sız çalıştıralım
- 1-2 gün çalışma

**B) ADAF'i şimdilik atlayalım**
- U-Net + YOLO + Klasik ile devam
- Sonuçlar zaten güçlü
- ADAF'i sonra ekleriz

**C) Docker kullanarak ADAF'i ayrı çalıştıralım**
- ADAF'i Docker'da çalıştır
- Sonuçları ana sistemle birleştir
- Her iki sistem bağımsız

**Hangisini tercih edersiniz?**

Ben **B seçeneğini** (şimdilik atlama) öneriyorum - sisteminiz zaten çok güçlü! ADAF'i ileride ekleyebiliriz.



