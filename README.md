Arkeolojik Alan Tespiti (Derin Öğrenme + Klasik)
===============================================

Bu proje, çok bantlı GeoTIFF (RGB, DSM, DTM) verilerinden arkeolojik izleri (tümsek, hendek, halka vb.) çıkarmak için iki yaklaşımı birleştirir:
- Derin Öğrenme: U‑Net (segmentation_models_pytorch) ile 9 kanallı tensör üzerinde olasılık üretimi
- Klasik Yöntemler: RVT (SVF, openness, LRM, slope) + SciPy tabanlı görüntü işleme skorları

Öne Çıkanlar
- Büyük sahalarda karo bazlı (tile/overlap) çıkarım ve cosine feathering ile dikişsiz mozaik
- 2–98 persentil ile robust normalizasyon (karo bazlı veya global)
- Sıfırdan kurulumda “zero‑shot” seçeneği (ImageNet encoder 3→9 kanal inflate)
- Klasik skorlar: rvtlog, hessian, morph (ve “combo” ortalaması)
- DL + Klasik “fusion” ve isteğe bağlı poligonlaştırma (GeoPackage)

Kurulum (hızlı)
1) Python 3.10+ tavsiye edilir. Paketleri yükleyin:
- pip install -r requirements.txt
2) Veri: RGB, DSM, DTM bantlarını aynı gridde içeren tek GeoTIFF hazırlayın.
3) config.yaml içindeki input yolunu doğrulayın. Çalıştırın:
- python archaeo_detect.py

config.yaml (temel ayarlar)
- enable_deep_learning: true/false — DL yolunu aç/kapat
- enable_classic: true/false — Klasik skorları aç/kapat
- enable_fusion: true/false — DL ve Klasik olasılıkları birleştir
- arch / encoder / encoders — Mimari ve encoder(lar)
- weights / weights_template / zero_shot_imagenet — Ağırlıklar ve zero‑shot
- classic_modes: combo veya rvtlog,hessian,morph
- classic_save_intermediate: true ise her klasik modu ayrı raster olarak yazar
- classic_th: 0..1 (null: Otsu)
- alpha: 0..1 (fusion karışım oranı)
- fuse_encoders: "all" veya CSV (resnet34,resnet50,efficientnet-b3)
- tile / overlap / feather — Karo işleme ve dikiş yumuşatma
- global_norm / norm_sample_tiles / percentile_low / percentile_high — Normalizasyon
- mask_talls — nDSM temelli yüksek obje maskeleme (metre)
- vectorize / min_area / simplify — Poligonlaştırma ve alan eşiği
- half / seed / verbose — Performans ve günlükleme
- cache_derivatives / cache_dir / recalculate_cache — RVT cache yönetimi

Çalıştırma Örnekleri
- Varsayılan (config.yaml’dan okur):
  - python archaeo_detect.py
- Eşiği ve karo ayarlarını override et:
  - python archaeo_detect.py --th 0.7 --tile 512 --overlap 128
- Tek encoder (zero‑shot):
  - python archaeo_detect.py --encoders none --encoder resnet34 --zero-shot-imagenet -v
- Çoklu encoder (hepsi) + fusion + cache:
  - python archaeo_detect.py --encoders all --enable-classic --enable-fusion --cache-derivatives -v
- Çoklu encoder ama sadece seçili encoder’larda fused yaz:
  - python archaeo_detect.py --encoders resnet34,resnet50,efficientnet-b3 --fuse-encoders resnet34,resnet50 --enable-fusion -v

Üretilen Çıktılar ve İsimlendirme
- Olasılık rasterı: *_prob.tif (float32, 0..1; nodata=NaN)
- Maske rasterı: *_mask.tif (uint8, {0,1}; nodata=0)
- Poligon vektör: *_mask.gpkg (isteğe bağlı)

Tek Encoder (DL):
- <prefix>_prob.tif, <prefix>_mask.tif

Klasik (combo veya modlar):
- <prefix>_classic_prob.tif, <prefix>_classic_mask.tif
- (opsiyonel mod başına): <prefix>_classic_rvtlog_prob.tif, …

Fusion (DL + Klasik):
- Tek encoder: <prefix>_fused_<encoder>_prob.tif ve _mask.tif
- Çoklu encoder: her encoder için ayrıca yazılır:
  - <prefix>_fused_resnet34_..._prob.tif, <prefix>_fused_resnet34_..._mask.tif
  - <prefix>_fused_resnet50_..._prob.tif, …
  - <prefix>_fused_efficientnet-b3_..._prob.tif, …
- İsimdeki parametreler: th0.5, tile1024, alpha0.5, minarea80 vb.

Nasıl Çalışır (özet)
- DL: Her karoda RGB/DSM/DTM okunur; RVT türevleri (SVF, openness, LRM, slope) + nDSM ile 9 kanal oluşturulur; normalize edilir; U‑Net ile olasılık üretilir; feather ile birleştirilir; eşiklenir.
- Klasik: DTM’den rvtlog/hessian/morph skorları; 0..1’e ölçek; feather ile birleştir; Otsu veya classic_th ile eşik.
- Fusion: p_fused = alpha*p_dl + (1-alpha)*p_classic; 0..1’e kırp; eşiklenir.
- Vectorize: Bağlı bileşenler -> poligon; min_area ile filtre; opsiyonel simplify; GeoPackage çıktı.

Performans İpuçları
- GPU ve half=true ile mixed precision hız sağlar.
- Büyük mozaiklerde tile’ı artırıp overlap’i %10–25 seçin; feather=true bırakın.
- global_norm=true görünüm tutarlılığı sağlar.
- cache_derivatives=true ile ilk çalıştırmadan sonra 10–100x hızlanma.

Sorun Giderme
- Klasik raster yok: DTM band index (5.) > 0 olmalı; kapatmak için --no-enable-classic.
- Fusion yok: enable_deep_learning, enable_classic, enable_fusion hepsi true olmalı; çoklu encoder’da fused dosyaları per‑encoder isimleriyle arayın veya --fuse-encoders ile filtreyi doğru verin.
- Boş vektör çıktı: th / classic_th çok yüksek olabilir; min_area da yüksek olabilir.
- Kenar dikişleri: overlap’i artırın ve feather=true kullanın.
- RVT kurulum: Python 3.10’da rvt-py; yeni sürümlerde rvt (uygun wheel gerekir).

Teknik Notlar
- Kod: archaeo_detect.py (tüm CLI/iş akışı burada)
- Nodata: Ara değerlerde NaN korunur; prob.tif’te nodata=NaN, mask.tif’te nodata=0
- CRS/alan: Gerekirse EPSG:6933’e proje edilerek alan hesaplanır

