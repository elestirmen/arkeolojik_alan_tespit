"""
Çok bantlı GeoTIFF'lerden arkeolojik alan tespiti için komut satırı aracı.

Pipeline aşağıdaki adımları gerçekleştirir:
1. Tek bir raster'dan RGB, DSM ve DTM bantlarını okur ve karo bazlı iş yüklerini hazırlar.
2. DTM'den kabartma görselleştirme katmanlarını (gökyüzü görüş faktörü, açıklık, yerel 
   kabartma modeli, eğim) Relief Visualization Toolbox (rvt-py) ile türetir.
3. 9 kanallı tensör yığını [R, G, B, SVF, PosOpen, NegOpen, LRM, Slope, nDSM] oluşturur,
   kanalları robust 2-98 persentil ölçekleme ile normalize eder ve nodata değerlerini güvenli şekilde işler.
4. Önceden eğitilmiş U-Net tipi model (segmentation_models_pytorch) ile karolara bölünmüş
   çıkarım yapar, örtüşen karoları harmanlayarak kesintisiz olasılık haritası oluşturur.
5. İsteğe bağlı yüksek obje maskeleme uygular, eşikleme ile ikili maske oluşturur ve hem
   olasılık hem de maske GeoTIFF çıktılarını sıkıştırılmış olarak, jeoreferansı koruyarak yazar.
6. İsteğe bağlı olarak tespit edilen özellikleri alan/skor öznitelikleriyle poligonlara dönüştürür 
   ve GIS iş akışları için GeoPackage formatında dışa aktarır.

Varsayımlar:
- Girdi mozaiği RGB, DSM ve DTM bantlarını aynı CRS/kapsam/piksel-ızgarada içerir.
- Önceden eğitilmiş model belirtilen sırada dokuz kanal bekler.
- rvt-py ve torch uyumlu hesaplama ortamı önceden kurulmuştur.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import uuid
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, TextIO, Tuple, TypeVar

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    gaussian_gradient_magnitude,
    gaussian_laplace,
    grey_closing,
    grey_opening,
    uniform_filter,
)
import torch
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # segmentation_models_pytorch is optional; required only for DL inference

try:
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform
except ImportError:
    mapping = None  # type: ignore[assignment]
    shape = None  # type: ignore[assignment]
    shapely_transform = None  # type: ignore[assignment]

try:
    from pyproj import CRS, Transformer
except ImportError:
    CRS = None  # type: ignore[assignment]
    Transformer = None  # type: ignore[assignment]

try:
    from rvt import vis as rvt_vis
except ImportError:
    rvt_vis = None  # rvt is required only when RVT-based features are used

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # geopandas is optional; fallback writer is used otherwise

try:
    import fiona
except ImportError:
    fiona = None

try:
    from osgeo import gdal
except ImportError:  # pragma: no cover - optional, used to tame GDAL warnings
    gdal = None

try:
    import yaml
except ImportError:
    yaml = None  # YAML support is optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # YOLO11 support is optional

LOGGER = logging.getLogger("archaeo_detect")

T = TypeVar("T")

# ==== Sabit Değerler ====
# Bu değerler PipelineDefaults içinde yapılandırılabilir hale getirildi.
# Eski kodlarla uyumluluk için burada referanslar var.

# Histogram ve eşikleme parametreleri (algoritmanın çalışması için sabit kalmalı)
HISTOGRAM_BINS = 256
HISTOGRAM_RANGE = (0.0, 1.0)

# Karo işleme sabitleri
TILE_SAMPLE_CROP_SIZE = 256
LABEL_CONNECTIVITY_STRUCTURE = (3, 3)

_GDAL_HANDLER_INSTALLED = False
_SUPPRESSED_GDAL_MESSAGES = (
    "TIFFReadDirectory:Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel",
)


def available_memory_bytes() -> Optional[int]:
    """Best-effort available system memory in bytes (None if unknown)."""
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def estimate_precompute_derivatives_bytes(
    pixels: int,
    *,
    has_dsm: bool,
    enable_curvature: bool,
    enable_tpi: bool,
) -> int:
    """
    Rough upper-bound estimate of RAM needed by `precompute_derivatives`.

    The function currently materializes full-raster arrays for:
    - RGB (3), DTM (1), optional DSM (1)
    - RVT outputs (SVF, pos/neg openness, LRM, slope) (5)
    - nDSM (1)
    - optional plan/profile curvature (2)
    - optional TPI (1)
    """
    float_arrays = 3 + 1 + (1 if has_dsm else 0) + 5 + 1
    if enable_curvature:
        float_arrays += 2
    if enable_tpi:
        float_arrays += 1
    return int(float_arrays * pixels * np.dtype(np.float32).itemsize)


def full_raster_cache_precompute_ok(
    input_path: Path,
    band_idx: Sequence[int],
    *,
    enable_curvature: bool,
    enable_tpi: bool,
) -> Tuple[bool, str]:
    """
    Decide whether full-raster derivative precompute/cache is feasible.

    This avoids out-of-memory crashes on very large rasters by falling back to
    per-tile derivative computation.
    """
    try:
        with rasterio.open(input_path) as src:
            height = int(src.height)
            width = int(src.width)
    except Exception as e:
        return False, f"Could not read raster size ({e}); skipping full-raster derivative cache precompute."

    pixels = height * width
    if pixels <= 0:
        return False, "Invalid raster size; skipping full-raster derivative cache precompute."

    # Heuristic: above this, full-raster materialization (many float32 layers) is rarely practical.
    if pixels >= 200_000_000:
        return (
            False,
            f"Raster is very large ({width}x{height}); skipping full-raster derivative cache precompute (using per-tile derivatives).",
        )

    avail = available_memory_bytes()
    has_dsm = bool(len(band_idx) >= 4 and band_idx[3] > 0)
    est = estimate_precompute_derivatives_bytes(
        pixels,
        has_dsm=has_dsm,
        enable_curvature=enable_curvature,
        enable_tpi=enable_tpi,
    )
    est_with_overhead = int(est * 1.35)
    if avail is None or avail <= 0:
        return True, ""

    if est_with_overhead > int(avail * 0.75):
        est_gib = est_with_overhead / (1024**3)
        avail_gib = avail / (1024**3)
        return (
            False,
            f"Full-raster derivative cache precompute would likely OOM (raster {width}x{height}; estimated ~{est_gib:.1f} GiB, available ~{avail_gib:.1f} GiB). Falling back to per-tile derivatives.",
        )

    return True, ""


def make_scratch_dir(tag: str) -> Path:
    """Create a per-run scratch directory under `checkpoints/scratch/`."""
    base = Path("checkpoints") / "scratch"
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"{tag}_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def alloc_array(
    shape: Tuple[int, int],
    dtype: Any,
    *,
    fill_value: Any,
    use_memmap: bool,
    scratch_dir: Optional[Path],
    name: str,
) -> Tuple[np.ndarray, Optional[Path]]:
    """Allocate an array in RAM or as a disk-backed memmap (for huge rasters)."""
    if not use_memmap:
        if fill_value == 0 or fill_value is False:
            return np.zeros(shape, dtype=dtype), None
        return np.full(shape, fill_value, dtype=dtype), None

    scratch_dir = scratch_dir or make_scratch_dir("arrays")
    path = scratch_dir / f"{name}_{uuid.uuid4().hex}.mmap"
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    # Ensure deterministic initialization for downstream logic.
    arr.fill(fill_value)
    return arr, path


def install_gdal_warning_filter() -> None:
    """Suppress noisy but harmless GDAL warnings about mis-declared TIFF extrasamples."""
    global _GDAL_HANDLER_INSTALLED
    if _GDAL_HANDLER_INSTALLED or gdal is None:
        return

    def _handler(err_class: int, err_num: int, err_msg: str) -> None:
        if any(err_msg.startswith(prefix) for prefix in _SUPPRESSED_GDAL_MESSAGES):
            LOGGER.debug("GDAL uyarısı bastırıldı: %s", err_msg)
            return
        try:
            gdal.CPLDefaultErrorHandler(err_class, err_num, err_msg)
        except AttributeError:  # pragma: no cover - safety for stripped bindings
            LOGGER.warning("GDAL uyarısı: %s", err_msg)

    gdal.PushErrorHandler(_handler)
    _GDAL_HANDLER_INSTALLED = True


def progress_bar(iterable: Iterable[T], **tqdm_kwargs) -> tqdm:
    """Wrapper around tqdm that prefers stdout so CLI runners can render progress."""
    stream: Optional[TextIO]
    if sys.stdout:
        stream = sys.stdout
    elif sys.stderr:
        stream = sys.stderr
    else:  # pragma: no cover - exceptionally rare in CLI usage
        stream = None
    if stream is not None:
        tqdm_kwargs.setdefault("file", stream)
    tqdm_kwargs.setdefault("dynamic_ncols", True)
    tqdm_kwargs.setdefault("mininterval", 0.5)
    tqdm_kwargs.setdefault("smoothing", 0.1)
    tqdm_kwargs.setdefault("leave", False)
    return tqdm(iterable, **tqdm_kwargs)

# ==== PIPELINE AYARLARI (TÜM PARAMETRELERİ BURADAN KONTROl EDİN) ====
# Bu dataclass tüm parametreleri tek merkezde toplar ve her alanın üzerindeki açıklama 
# parametrenin ne yaptığını gösterir. İstediğiniz ayarları buradan değiştirebilirsiniz.


@dataclass
class PipelineDefaults:
    """
    Arkeolojik alan tespit pipeline'ının tüm ayarlarını içeren merkezi yapılandırma sınıfı.
    Tüm parametreleri buradan kontrol edebilir, hangi yöntemlerin çalışacağını belirleyebilirsiniz.
    """
    
    # ===== GİRDİ/ÇIKTI AYARLARI =====
    input: str = field(
        default="kesif_alani.tif",
        metadata={"help": "Çok bantlı GeoTIFF dosyasının yolu (RGB + DSM + DTM önerilir)"},
    )
    out_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Çıktı dosyaları için ön-ek veya dizin; None ise girdi adından otomatik üretilir"},
    )
    bands: str = field(
        default="1,2,3,4,5",
        metadata={"help": "R,G,B,DSM,DTM bantlarının 1-tabanlı sırası; eksik bant için 0 yazın (örn: 1,2,3,4,5)"},
    )
    
    # ===== YÖNTEM SEÇİMİ (HANGİ YÖNTEMLERİN ÇALIŞACAĞINI BURADAN KONTROL EDİN) =====
    enable_deep_learning: bool = field(
        default=True,
        metadata={"help": "Derin öğrenme (U-Net) modelini çalıştır"},
    )
    enable_classic: bool = field(
        default=True,
        metadata={"help": "Klasik kabartma tabanlı yöntemleri (RVT, Hessian, morfoloji) çalıştır"},
    )
    enable_yolo: bool = field(
        default=False,
        metadata={"help": "YOLO11 nesne tespit/segmentasyon modelini çalıştır"},
    )
    enable_fusion: bool = field(
        default=True,
        metadata={"help": "Derin öğrenme ve klasik yöntem sonuçlarını birleştir (fusion)"},
    )
    
    # ===== DERİN ÖĞRENME MODEL AYARLARI =====
    arch: str = field(
        default="Unet",
        metadata={"help": "Segmentation model mimarisi (örn: Unet, UnetPlusPlus, DeepLabV3Plus)"},
    )
    encoder: str = field(
        default="resnet34",
        metadata={"help": "Tek model çalışırken kullanılacak varsayılan encoder (resnet34, resnet50, efficientnet-b3)"},
    )
    encoders: str = field(
        default="none",
        metadata={"help": "Çalıştırılacak encoder listesi; 'all'=hepsi, 'none'=sadece encoder parametresi, veya 'resnet34,resnet50'"},
    )
    weights: Optional[str] = field(
        default=None,
        metadata={"help": "Eğitilmiş model ağırlıkları (.pth dosyası); None ise zero-shot modu kullanılır"},
    )
    weights_template: Optional[str] = field(
        default=None,
        metadata={"help": "Encoder bazlı ağırlık şablonu (örn: models/unet_{encoder}_9ch.pth); {encoder} yerine encoder adı konur"},
    )
    zero_shot_imagenet: bool = field(
        default=True,
        metadata={"help": "Ağırlık dosyası yoksa ImageNet encoder'ını 9 kanala genişletip zero-shot çalıştır"},
    )
    th: float = field(
        default=0.6,
        metadata={"help": "DL olasılık haritasını maskeye çevirmek için eşik (0-1 arası); düşük değer daha fazla tespit üretir"},
    )
    
    # ===== KLASİK YÖNTEM AYARLARI =====
    classic_modes: str = field(
        default="combo",
        metadata={"help": "Klasik yöntemler: 'combo' (hepsini çalıştır), veya 'rvtlog,hessian,morph' (seçili olanları çalıştır)"},
    )
    classic_save_intermediate: bool = field(
        default=True,
        metadata={"help": "Her klasik modu ayrı dosya olarak kaydet (karşılaştırma için kullanışlı)"},
    )
    classic_th: Optional[float] = field(
        default=None,
        metadata={"help": "Klasik yöntem için sabit maske eşiği (0-1); None ise otomatik Otsu eşiği kullanılır"},
    )
    
    # ===== KLASİK YÖNTEM ALGORİTMA PARAMETRELERİ =====
    sigma_scales: Tuple[float, ...] = field(
        default=(1.0, 2.0, 4.0, 8.0),
        metadata={"help": "Çok ölçekli filtreleme için sigma değerleri (LoG, Hessian için)"},
    )
    morphology_radii: Tuple[int, ...] = field(
        default=(3, 5, 9, 15),
        metadata={"help": "Morfolojik filtreleme için yapısal eleman yarıçapları"},
    )
    rvt_radii: Tuple[float, ...] = field(
        default=(5.0, 10.0, 20.0, 30.0, 50.0),
        metadata={"help": "RVT hesaplamaları (SVF, openness) için arama yarıçapları (metre)"},
    )
    local_variance_window: int = field(
        default=7,
        metadata={"help": "Yerel varyans hesaplama için pencere boyutu"},
    )
    gaussian_gradient_sigma: float = field(
        default=1.5,
        metadata={"help": "Gradyan büyüklüğü hesaplama için Gaussian sigma değeri"},
    )
    gaussian_lrm_sigma: float = field(
        default=6.0,
        metadata={"help": "LRM fallback hesaplama için Gaussian sigma değeri"},
    )
    
    # ===== GELİŞMİŞ TOPOGRAFİK ANALİZ AYARLARI (YENİ KANALLAR) =====
    enable_curvature: bool = field(
        default=True,
        metadata={"help": "Plan ve Profile Curvature kanallarını hesapla (hendek/tepe ayrımı için)"},
    )
    enable_tpi: bool = field(
        default=True,
        metadata={"help": "Multi-scale TPI (Topographic Position Index) kanalını hesapla (höyük/çukur tespiti için)"},
    )
    tpi_radii: Tuple[int, ...] = field(
        default=(5, 15, 30),
        metadata={"help": "TPI hesaplama yarıçapları (piksel); farklı boyutlardaki yapılar için"},
    )
    
    # ===== ATTENTION MEKANİZMASI AYARLARI =====
    enable_attention: bool = field(
        default=True,
        metadata={"help": "CBAM (Convolutional Block Attention Module) kullan; kanal önemini dinamik öğrenir"},
    )
    attention_reduction: int = field(
        default=4,
        metadata={"help": "Attention modülü için kanal azaltma oranı (4=1/4 oranında sıkıştırma)"},
    )
    
    # ===== YOLO11 MODEL AYARLARI =====
    yolo_weights: Optional[str] = field(
        default=None,
        metadata={"help": "YOLO11 ağırlık dosyası (örn: yolo11n.pt, yolo11n-seg.pt); None ise varsayılan yolo11n-seg.pt indirilir"},
    )
    yolo_conf: float = field(
        default=0.25,
        metadata={"help": "YOLO11 tespit güven eşiği (0-1 arası)"},
    )
    yolo_iou: float = field(
        default=0.45,
        metadata={"help": "YOLO11 NMS IoU eşiği (0-1 arası)"},
    )
    yolo_tile: Optional[int] = field(
        default=None,
        metadata={"help": "YOLO11 için özel tile boyutu; None ise --tile değerini kullanır"},
    )
    yolo_imgsz: int = field(
        default=640,
        metadata={"help": "YOLO11 model girdi boyutu (piksel); model bu boyuta ölçekler"},
    )
    yolo_device: Optional[str] = field(
        default=None,
        metadata={"help": "YOLO11 için cihaz ('0', 'cpu', vb.); None ise otomatik seçilir"},
    )
    
    # ===== BİRLEŞTİRME (FUSION) AYARLARI =====
    alpha: float = field(
        default=0.5,
        metadata={"help": "Fusion karışım ağırlığı (0-1); 1.0=sadece DL, 0.0=sadece klasik, 0.5=eşit ağırlık"},
    )
    fuse_encoders: str = field(
        default="all",
        metadata={"help": "Füzyon yapılacak encoder listesi: 'all' veya 'resnet34,resnet50,efficientnet-b3'"},
    )
    
    # ===== KARO İŞLEME AYARLARI =====
    tile: int = field(
        default=1024,
        metadata={"help": "Karo boyutu (piksel); büyük değer daha fazla GPU belleği kullanır ama daha az karo demektir"},
    )
    overlap: int = field(
        default=256,
        metadata={"help": "Komşu karolar arası bindirme (piksel); dikiş hatlarını azaltır ama işlem süresini artırır"},
    )
    feather: bool = field(
        default=True,
        metadata={"help": "Karo sınırlarında yumuşatma uygula; dikiş izlerini azaltır"},
    )
    
    # ===== NORMALİZASYON AYARLARI =====
    global_norm: bool = field(
        default=True,
        metadata={"help": "Tüm karolar için tek bir global persentil normalizasyonu kullan (tutarlılık için önerilir)"},
    )
    norm_sample_tiles: int = field(
        default=32,
        metadata={"help": "Global normalizasyon için örneklenecek karo sayısı"},
    )
    percentile_low: float = field(
        default=2.0,
        metadata={"help": "Normalizasyon alt persentil değeri"},
    )
    percentile_high: float = field(
        default=98.0,
        metadata={"help": "Normalizasyon üst persentil değeri"},
    )
    
    # ===== MASKELEME VE FİLTRELEME AYARLARI =====
    mask_talls: Optional[float] = field(
        default=2.5,
        metadata={"help": "Bu yüksekliği (metre) aşan nDSM piksellerini maskele (yüksek bina/ağaçları eleme); None=kapalı"},
    )
    rgb_only: bool = field(
        default=False,
        metadata={"help": "If True, ignore derivative channels and run effective RGB-only inference by zero-filling SVF/Openness/LRM/Slope/nDSM."},
    )
    
    # ===== VEKTÖRLEŞTİRME AYARLARI =====
    vectorize: bool = field(
        default=True,
        metadata={"help": "Tespit maskelerini poligonlara dönüştürüp GeoPackage (.gpkg) olarak dışa aktar"},
    )
    min_area: float = field(
        default=80.0,
        metadata={"help": "Vektörleştirmede kabul edilecek minimum poligon alanı (metrekare)"},
    )
    simplify: Optional[float] = field(
        default=None,
        metadata={"help": "Poligon basitleştirme toleransı (metre); None ise basitleştirme yapılmaz"},
    )
    vector_opening_size: int = field(
        default=3,
        metadata={"help": "Vektörleştirme için morfolojik açma (opening) boyutu (piksel)"},
    )
    label_connectivity: int = field(
        default=8,
        metadata={"help": "Bileşen etiketlemede bağlanırlık (4 veya 8 komşuluk)"},
    )
    
    # ===== PERFORMANS VE TEKNİK AYARLAR =====
    half: bool = field(
        default=True,
        metadata={"help": "CUDA varsa float16 (yarı hassasiyet) kullan; bellek tasarrufu ve hız kazancı sağlar"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Rastgelelik tohum değeri (tekrarlanabilirlik için)"},
    )
    verbose: int = field(
        default=1,
        metadata={"help": "Log detay seviyesi: 0=WARNING, 1=INFO, 2=DEBUG"},
    )
    
    # ===== CACHE YÖNETİMİ (ÇOK HIZLI TEKRAR ÇALIŞTIRMALAR İÇİN) =====
    cache_derivatives: bool = field(
        default=False,
        metadata={"help": "RVT türevlerini (SVF, openness, LRM, slope) diske kaydet/oku; tekrar çalıştırmalarda çok hızlı"},
    )
    cache_derivatives_mode: str = field(
        default="auto",
        metadata={
            "help": "Cache formatı: 'auto' (küçükte .npz, büyükte blok-bazlı raster-cache), 'npz' veya 'raster'."
        },
    )
    deriv_cache_chunk: int = field(
        default=2048,
        metadata={"help": "Raster-cache üretiminde blok boyutu (piksel). Büyük değer daha hızlı ama daha çok RAM kullanır."},
    )
    deriv_cache_halo: Optional[int] = field(
        default=None,
        metadata={"help": "Raster-cache için halo (piksel). None ise rvt_radii/tpi_radii/sigma'dan otomatik hesaplanır."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache dosyalarının kaydedileceği dizin; None ise girdi dosyasının yanına yazar"},
    )
    recalculate_cache: bool = field(
        default=False,
        metadata={"help": "Mevcut cache'i yoksay ve RVT türevlerini yeniden hesapla"},
    )
    
    # ===== CİHAZ SEÇİMİ =====
    device: Optional[str] = field(
        default=None,
        metadata={"help": "Hesaplama cihazı: 'cuda', 'cpu', 'cuda:0', vb. None ise otomatik seçilir (CUDA varsa GPU)"},
    )
    
    def __post_init__(self) -> None:
        """Konfigürasyon değerlerini doğrula ve geçersiz değerlerde hata fırlat."""
        errors: List[str] = []
        
        # Eşik değerleri kontrolü (0-1 aralığı)
        if not 0.0 <= self.th <= 1.0:
            errors.append(f"th değeri 0-1 arasında olmalı, verilen: {self.th}")
        
        if self.classic_th is not None and not 0.0 <= self.classic_th <= 1.0:
            errors.append(f"classic_th değeri 0-1 arasında olmalı, verilen: {self.classic_th}")
        
        if not 0.0 <= self.alpha <= 1.0:
            errors.append(f"alpha değeri 0-1 arasında olmalı, verilen: {self.alpha}")
        
        if not 0.0 <= self.yolo_conf <= 1.0:
            errors.append(f"yolo_conf değeri 0-1 arasında olmalı, verilen: {self.yolo_conf}")
        
        if not 0.0 <= self.yolo_iou <= 1.0:
            errors.append(f"yolo_iou değeri 0-1 arasında olmalı, verilen: {self.yolo_iou}")
        
        # Pozitif değer kontrolleri
        if self.tile <= 0:
            errors.append(f"tile pozitif olmalı, verilen: {self.tile}")
        
        if self.overlap < 0:
            errors.append(f"overlap negatif olamaz, verilen: {self.overlap}")
        
        if self.overlap >= self.tile:
            errors.append(f"overlap ({self.overlap}) tile'dan ({self.tile}) küçük olmalı")
        
        if self.yolo_imgsz <= 0:
            errors.append(f"yolo_imgsz pozitif olmalı, verilen: {self.yolo_imgsz}")
        
        if self.yolo_tile is not None and self.yolo_tile <= 0:
            errors.append(f"yolo_tile pozitif olmalı, verilen: {self.yolo_tile}")
        
        if self.norm_sample_tiles <= 0:
            errors.append(f"norm_sample_tiles pozitif olmalı, verilen: {self.norm_sample_tiles}")
        
        if self.min_area < 0:
            errors.append(f"min_area negatif olamaz, verilen: {self.min_area}")
        
        if self.simplify is not None and self.simplify < 0:
            errors.append(f"simplify negatif olamaz, verilen: {self.simplify}")
        
        # Persentil değerleri kontrolü
        if not 0.0 <= self.percentile_low <= 100.0:
            errors.append(f"percentile_low 0-100 arasında olmalı, verilen: {self.percentile_low}")
        
        if not 0.0 <= self.percentile_high <= 100.0:
            errors.append(f"percentile_high 0-100 arasında olmalı, verilen: {self.percentile_high}")
        
        if self.percentile_low >= self.percentile_high:
            errors.append(f"percentile_low ({self.percentile_low}) percentile_high'dan ({self.percentile_high}) küçük olmalı")
        
        # Connectivity kontrolü
        if self.label_connectivity not in (4, 8):
            errors.append(f"label_connectivity 4 veya 8 olmalı, verilen: {self.label_connectivity}")
        
        # Verbose seviyesi
        if self.verbose < 0 or self.verbose > 2:
            errors.append(f"verbose 0-2 arasında olmalı, verilen: {self.verbose}")
        
        # Cihaz kontrolü
        if self.device is not None:
            valid_devices = {"cpu", "cuda", "mps"}
            device_base = self.device.split(":")[0].lower()
            if device_base not in valid_devices and not device_base.startswith("cuda"):
                errors.append(f"device geçersiz: {self.device}. Geçerli değerler: 'cpu', 'cuda', 'cuda:0', 'mps'")
        
        # Sigma ve radii kontrolleri
        if len(self.sigma_scales) == 0:
            errors.append("sigma_scales en az bir değer içermeli")
        if any(s <= 0 for s in self.sigma_scales):
            errors.append("sigma_scales tüm değerleri pozitif olmalı")
        
        if len(self.morphology_radii) == 0:
            errors.append("morphology_radii en az bir değer içermeli")
        if any(r <= 0 for r in self.morphology_radii):
            errors.append("morphology_radii tüm değerleri pozitif olmalı")
        
        if len(self.rvt_radii) == 0:
            errors.append("rvt_radii en az bir değer içermeli")
        if any(r <= 0 for r in self.rvt_radii):
            errors.append("rvt_radii tüm değerleri pozitif olmalı")

        # Cache (NPZ vs raster) kontrolleri
        mode = str(self.cache_derivatives_mode).strip().lower()
        if mode not in ("auto", "npz", "raster"):
            errors.append(f"cache_derivatives_mode geçersiz: {self.cache_derivatives_mode} (auto/npz/raster)")
        if self.deriv_cache_chunk <= 0:
            errors.append(f"deriv_cache_chunk pozitif olmalı, verilen: {self.deriv_cache_chunk}")
        if self.deriv_cache_halo is not None and self.deriv_cache_halo < 0:
            errors.append(f"deriv_cache_halo negatif olamaz, verilen: {self.deriv_cache_halo}")
        
        # Tüm hataları topla ve fırlat
        if errors:
            error_msg = "Konfigürasyon doğrulama hataları:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)


DEFAULTS = PipelineDefaults()


def default_for(name: str) -> Any:
    """PipelineDefaults'tan verilen isimli varsayılan değeri döndür."""
    return getattr(DEFAULTS, name)


def help_for(name: str) -> str:
    for f in fields(PipelineDefaults):
        if f.name == name:
            return f.metadata.get("help", "")
    raise KeyError(name)


def cli_help(name: str, extra: Optional[str] = None) -> str:
    base = help_for(name)
    if extra:
        return f"{base} {extra}".strip()
    return base


def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """YAML config dosyasını oku ve dictionary olarak döndür."""
    if yaml is None:
        raise ImportError(
            "YAML desteği için PyYAML yükleyin: pip install pyyaml"
        )
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return config_dict


def build_config_from_args(args: argparse.Namespace) -> PipelineDefaults:
    """Komut satırı argümanlarından ve opsiyonel YAML config'den PipelineDefaults oluştur."""
    # Önce YAML config'i yükle (varsa)
    base_config = {}
    if hasattr(args, 'config') and args.config:
        yaml_path = Path(args.config)
        try:
            base_config = load_config_from_yaml(yaml_path)
            LOGGER.info(f"Config dosyası yüklendi: {yaml_path}")
        except Exception as e:
            LOGGER.warning(f"Config dosyası yüklenemedi ({yaml_path}): {e}")
    
    # YAML'dan gelen listeleri tuple'a çevir
    list_to_tuple_fields = {'sigma_scales', 'morphology_radii', 'rvt_radii', 'tpi_radii'}
    for field_name in list_to_tuple_fields:
        if field_name in base_config and isinstance(base_config[field_name], list):
            base_config[field_name] = tuple(base_config[field_name])
    
    # Komut satırı argümanlarıyla override et
    # Sadece komut satırında açıkça belirtilen argümanları kullan
    values = {}
    for f in fields(PipelineDefaults):
        # Önce YAML'dan al (varsa)
        if f.name in base_config:
            values[f.name] = base_config[f.name]
        # Sonra komut satırından al (varsa ve default değilse)
        if hasattr(args, f.name):
            arg_value = getattr(args, f.name)
            # Eğer argüman default değilse, YAML'ı override et
            default_value = default_for(f.name)
            if arg_value != default_value:
                values[f.name] = arg_value
            # Eğer YAML'da da yoksa, default'u kullan
            elif f.name not in values:
                values[f.name] = arg_value
    
    return PipelineDefaults(**values)



def configure_logging(verbosity: int) -> None:
    """Configure global logging level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seeds(seed: int) -> None:
    """Enforce deterministic behaviour across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_windowed(
    raster: rasterio.io.DatasetReader | str | Path,
    window: Window,
    indexes: Optional[Sequence[int]] = None,
    boundless: bool = True,
) -> np.ndarray:
    """Read a raster window from either an open dataset or a file path."""
    if hasattr(raster, "read"):
        dataset = raster
        return dataset.read(indexes=indexes, window=window, boundless=boundless)
    with rasterio.open(raster) as dataset:
        return dataset.read(indexes=indexes, window=window, boundless=boundless)


def compute_ndsm(dsm: Optional[np.ndarray], dtm: np.ndarray) -> np.ndarray:
    """Compute normalised DSM (nDSM). Returns zeros if DSM is missing."""
    ndsm = np.full_like(dtm, np.nan, dtype=np.float32)
    dtm_valid = np.isfinite(dtm)
    if dsm is None:
        ndsm[dtm_valid] = 0.0
        return ndsm
    dsm_valid = np.isfinite(dsm)
    mask = dtm_valid & dsm_valid
    ndsm[mask] = (dsm - dtm)[mask]
    return ndsm


def percentile_clip(
    arr: np.ndarray, low: Optional[float] = None, high: Optional[float] = None
) -> np.ndarray:
    """Dizi değerlerini verilen persentiller arasında kırp, NaN değerlerine saygı göster."""
    if low is None:
        low = DEFAULTS.percentile_low
    if high is None:
        high = DEFAULTS.percentile_high
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lower = np.percentile(arr[valid], low)
    upper = np.percentile(arr[valid], high)
    if upper - lower <= 1e-6:
        clipped = np.clip(arr - lower, 0.0, 1.0)
    else:
        clipped = np.clip((arr - lower) / (upper - lower), 0.0, 1.0)
    clipped[~valid] = 0.0
    return clipped.astype(np.float32)


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Persentil tabanlı 0-1 ölçekleme için kolaylık sarmalayıcısı."""
    return percentile_clip(arr)


def _otsu_threshold_0to1(arr: np.ndarray, valid: np.ndarray) -> float:
    """0..1 arası veride geçerlilik maskesini dikkate alarak Otsu eşiği hesapla."""
    data = arr[valid]
    if data.size == 0:
        return 0.5
    clipped = np.clip(data, 0.0, 1.0)
    hist, bin_edges = np.histogram(clipped, bins=HISTOGRAM_BINS, range=HISTOGRAM_RANGE)
    if not np.any(hist):
        return float(np.nanmean(clipped)) if np.any(np.isfinite(clipped)) else 0.5
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    cumulative = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    global_mean = cumulative_mean[-1]
    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (1.0 - cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = numerator / np.maximum(denominator, 1e-12)
    sigma_b[cumulative == 0.0] = 0.0
    sigma_b[cumulative == 1.0] = 0.0
    # If there is a plateau of equally-good thresholds (common when there is a wide empty gap),
    # pick the midpoint instead of the first bin. This yields more intuitive thresholds.
    best_val = float(np.max(sigma_b))
    best_idxs = np.flatnonzero(np.isclose(sigma_b, best_val, rtol=0.0, atol=1e-12))
    best = int(best_idxs[len(best_idxs) // 2]) if best_idxs.size else int(np.argmax(sigma_b))
    threshold = (bin_edges[best] + bin_edges[best + 1]) * 0.5
    return float(np.clip(threshold, 0.0, 1.0))


def otsu_threshold_from_hist(hist: np.ndarray, bin_edges: np.ndarray) -> float:
    """Compute Otsu threshold from a histogram (0..1 range)."""
    if hist.size == 0 or not np.any(hist):
        return 0.5
    hist_f = hist.astype(np.float64, copy=False)
    prob = hist_f / hist_f.sum()
    cumulative = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    cumulative_mean = np.cumsum(prob * bin_centers)
    global_mean = cumulative_mean[-1]
    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (1.0 - cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = numerator / np.maximum(denominator, 1e-12)
    sigma_b[cumulative == 0.0] = 0.0
    sigma_b[cumulative == 1.0] = 0.0
    best_val = float(np.max(sigma_b))
    best_idxs = np.flatnonzero(np.isclose(sigma_b, best_val, rtol=0.0, atol=1e-12))
    best = int(best_idxs[len(best_idxs) // 2]) if best_idxs.size else int(np.argmax(sigma_b))
    threshold = (bin_edges[best] + bin_edges[best + 1]) * 0.5
    return float(np.clip(threshold, 0.0, 1.0))


def otsu_threshold_streaming(
    prob_map: np.ndarray,
    valid_global: np.ndarray,
    *,
    block_rows: int = 1024,
) -> float:
    """Compute Otsu threshold without materializing all valid pixels at once."""
    bin_edges = np.linspace(HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], HISTOGRAM_BINS + 1)
    hist = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    height = int(prob_map.shape[0])
    for row0 in range(0, height, int(block_rows)):
        row1 = min(height, row0 + int(block_rows))
        block = prob_map[row0:row1, :]
        valid_block = valid_global[row0:row1, :] & np.isfinite(block)
        if not np.any(valid_block):
            continue
        values = block[valid_block]
        if values.size == 0:
            continue
        values = np.clip(values, HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1])
        h, _ = np.histogram(values, bins=HISTOGRAM_BINS, range=HISTOGRAM_RANGE)
        hist += h.astype(np.int64, copy=False)
    return otsu_threshold_from_hist(hist, bin_edges)


def _local_variance(arr: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Uniform filtreleme kullanarak yerel varyans tahmini yap."""
    if size is None:
        size = DEFAULTS.local_variance_window
    if size <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    arr_f = arr.astype(np.float32, copy=False)
    mean = uniform_filter(arr_f, size=size, mode="nearest")
    mean_sq = uniform_filter(arr_f * arr_f, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return var.astype(np.float32)


def _hessian_response(im: np.ndarray, sigma: float) -> np.ndarray:
    """Hessian matrisinden normalize edilmiş ikinci özdeğer büyüklüğünü hesapla."""
    im_f = im.astype(np.float32, copy=False)
    Hxx = gaussian_filter(im_f, sigma=sigma, order=(2, 0), mode="nearest")
    Hxy = gaussian_filter(im_f, sigma=sigma, order=(1, 1), mode="nearest")
    Hyy = gaussian_filter(im_f, sigma=sigma, order=(0, 2), mode="nearest")
    tr = Hxx + Hyy
    det = Hxx * Hyy - Hxy * Hxy
    tmp = np.sqrt(np.maximum((tr * tr) / 4.0 - det, 0.0))
    l2 = tr / 2.0 - tmp
    return _norm01(np.abs(l2))


def robust_norm(
    arr: np.ndarray, p_low: Optional[float] = None, p_high: Optional[float] = None
) -> np.ndarray:
    """Persentil kırpma ile kanal başına robust min-max normalizasyonu."""
    if p_low is None:
        p_low = DEFAULTS.percentile_low
    if p_high is None:
        p_high = DEFAULTS.percentile_high
    if arr.ndim != 3:
        raise ValueError("robust_norm expects a (C, H, W) array.")
    normed = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        channel = arr[idx]
        normed[idx] = percentile_clip(channel, low=p_low, high=p_high)
    return normed


def robust_norm_fixed(arr: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Normalize (C,H,W) with fixed per-channel lows/highs; NaNsâ†’0."""
    assert arr.ndim == 3
    out = np.zeros_like(arr, dtype=np.float32)
    C = arr.shape[0]
    for c in range(C):
        ch = arr[c]
        low = float(lows[c])
        high = float(highs[c])
        if high - low <= 1e-6:
            norm = np.clip(ch - low, 0.0, 1.0)
        else:
            norm = np.clip((ch - low) / (high - low), 0.0, 1.0)
        norm[~np.isfinite(ch)] = 0.0
        out[c] = norm.astype(np.float32)
    return out


def fill_nodata(
    arr: np.ndarray, fill_value: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaNs in an array with a constant (fallback median if not provided)."""
    filled = arr.astype(np.float32, copy=True)
    invalid_mask = ~np.isfinite(filled)
    if not np.any(invalid_mask):
        return filled, ~invalid_mask
    if fill_value is None:
        valid_vals = filled[~invalid_mask]
        fill_value = float(np.nanmedian(valid_vals)) if valid_vals.size else 0.0
    filled[invalid_mask] = fill_value
    return filled, ~invalid_mask


# ==============================================================================
# GELİŞMİŞ TOPOGRAFİK ANALİZ FONKSİYONLARI (Curvature + TPI)
# ==============================================================================

def compute_curvatures(
    dtm: np.ndarray, 
    pixel_size: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DTM'den Plan ve Profile Curvature hesaplar.
    
    Plan Curvature (Yatay Eğrilik):
        - Kontür çizgileri boyunca eğrilik
        - Pozitif değerler: Sırt/tepe (dışbükey)
        - Negatif değerler: Vadi/hendek (içbükey)
        - Arkeolojik yapılar: Tümülüsler pozitif, hendekler negatif
    
    Profile Curvature (Dikey Eğrilik):
        - Eğim yönündeki eğrilik
        - Pozitif değerler: Dışbükey yüzeyler (ivmelenen akış)
        - Negatif değerler: İçbükey yüzeyler (yavaşlayan akış)
        - Arkeolojik yapılar: Teraslar, basamaklar tespit edilir
    
    Args:
        dtm: Sayısal Arazi Modeli (Digital Terrain Model)
        pixel_size: Piksel boyutu (metre cinsinden)
        
    Returns:
        (plan_curvature, profile_curvature) - Her ikisi de (H, W) boyutunda
    """
    # NaN değerleri doldur
    dtm_filled, valid_mask = fill_nodata(dtm)
    
    # Birinci türevler (gradient) - x ve y yönlerinde
    fy, fx = np.gradient(dtm_filled, pixel_size)
    
    # İkinci türevler
    fyy, fyx = np.gradient(fy, pixel_size)
    fxy, fxx = np.gradient(fx, pixel_size)
    
    # Gradyan büyüklüğünün kareleri
    p = fx ** 2  # (df/dx)^2
    q = fy ** 2  # (df/dy)^2
    
    # Sıfıra bölme koruması
    denominator = p + q
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    # Plan Curvature (horizontal/contour curvature)
    # Formül: Kh = -[(fxx * q) - (2 * fxy * fx * fy) + (fyy * p)] / [(p + q)^1.5]
    plan_curv = -((fxx * q) - (2 * fxy * fx * fy) + (fyy * p)) / (denominator ** 1.5)
    
    # Profile Curvature (vertical/slope curvature)
    # Formül: Kv = -[(fxx * p) + (2 * fxy * fx * fy) + (fyy * q)] / [(p + q)^1.5]
    profile_curv = -((fxx * p) + (2 * fxy * fx * fy) + (fyy * q)) / (denominator ** 1.5)
    
    # NaN ve sonsuz değerleri temizle
    plan_curv = np.nan_to_num(plan_curv, nan=0.0, posinf=0.0, neginf=0.0)
    profile_curv = np.nan_to_num(profile_curv, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Aşırı değerleri kırp (robust normalizasyon için)
    # Tipik curvature değerleri -0.1 ile 0.1 arasında
    plan_curv = np.clip(plan_curv, -0.1, 0.1)
    profile_curv = np.clip(profile_curv, -0.1, 0.1)
    
    # Orijinal geçersiz bölgeleri işaretle
    plan_curv[~valid_mask] = np.nan
    profile_curv[~valid_mask] = np.nan
    
    return plan_curv.astype(np.float32), profile_curv.astype(np.float32)


def compute_tpi_multiscale(
    dtm: np.ndarray, 
    radii: Tuple[int, ...] = (5, 15, 30)
) -> np.ndarray:
    """
    Çok ölçekli Topographic Position Index (TPI) hesaplar.
    
    TPI, bir pikselin çevresine göre göreli yüksekliğini ölçer:
        TPI = merkez_yükseklik - ortalama(komşu_yükseklikleri)
    
    Arkeolojik Yapı Tespitinde Kullanımı:
        - Pozitif TPI: Çevreden yüksek yapılar (tümülüs, höyük, tepe)
        - Negatif TPI: Çevreden alçak yapılar (hendek, vadi, çukur)
        - Sıfıra yakın TPI: Düz alanlar veya yamaçlar
    
    Çok Ölçekli Yaklaşım:
        - Küçük yarıçap (5px): Küçük yapılar, duvar kalıntıları
        - Orta yarıçap (15px): Tipik tümülüsler, küçük höyükler
        - Büyük yarıçap (30px): Geniş höyükler, yerleşim tepeleri
    
    Args:
        dtm: Sayısal Arazi Modeli
        radii: Farklı ölçekler için yarıçaplar (piksel cinsinden)
        
    Returns:
        tpi_combined: Çok ölçekli TPI ortalaması, normalize edilmiş [-1, 1] arasında
    """
    # NaN değerleri doldur
    dtm_filled, valid_mask = fill_nodata(dtm)
    
    tpi_stack = []
    
    for radius in radii:
        # Kare pencere boyutu (2*r + 1)
        window_size = 2 * radius + 1
        
        # Uniform filter ile komşuluk ortalaması hesapla
        # reflect mode: kenar piksellerini yansıtarak sınır etkilerini azalt
        mean_elevation = uniform_filter(dtm_filled, size=window_size, mode='reflect')
        
        # TPI = merkez piksel değeri - ortalama değer
        tpi = dtm_filled - mean_elevation
        
        # Z-score benzeri normalizasyon (-1 ile 1 arası)
        tpi_std = np.nanstd(tpi[valid_mask]) if np.any(valid_mask) else 1.0
        if tpi_std > 0:
            tpi = tpi / (3 * tpi_std)  # 3-sigma normalizasyon (±3σ → ±1)
        
        tpi = np.clip(tpi, -1, 1)
        tpi_stack.append(tpi)
    
    # Ölçeklerin ortalamasını al (ensemble yaklaşım)
    tpi_combined = np.nanmean(np.stack(tpi_stack, axis=0), axis=0)
    
    # Orijinal geçersiz bölgeleri işaretle
    tpi_combined[~valid_mask] = np.nan
    
    return tpi_combined.astype(np.float32)


def compute_derivatives_with_rvt(
    dtm: np.ndarray,
    pixel_size: float,
    radii: Optional[Sequence[float]] = None,
    gaussian_lrm_sigma: Optional[float] = None,
    *,
    show_progress: bool = True,
    log_steps: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RVT rutinlerini kullanarak SVF, açıklık, LRM ve eğim hesapla."""
    if rvt_vis is None:
        raise ImportError("Install rvt-py: pip install rvt")
    if radii is None:
        radii = DEFAULTS.rvt_radii
    dtm_filled, valid_mask = fill_nodata(dtm)
    log_fn = LOGGER.info if log_steps else LOGGER.debug
    # Compatibility helpers for rvt-py API differences across versions
    def _as_float32_array(result_obj, name: str) -> np.ndarray:
        """Best-effort conversion of various RVT return types to float32 ndarray."""
        # Direct ndarray
        if isinstance(result_obj, np.ndarray):
            return result_obj.astype(np.float32)
        # Masked array
        if isinstance(result_obj, np.ma.MaskedArray):
            return np.ma.filled(result_obj, np.nan).astype(np.float32)
        # Dict of outputs (pick most relevant or first ndarray)
        if isinstance(result_obj, dict):
            preferred_keys = (
                "svf",
                "SVF",
                "positive_openness",
                "pos_open",
                "openness_positive",
                "negative_openness",
                "neg_open",
                "openness_negative",
                "lrm",
                "LRM",
                "slope",
                "Slope",
            )
            for k in preferred_keys:
                if k in result_obj and isinstance(result_obj[k], (np.ndarray, np.ma.MaskedArray)):
                    return _as_float32_array(result_obj[k], name)
            # fallback: first ndarray-looking value
            for v in result_obj.values():
                try:
                    arr = _as_float32_array(v, name)
                    if isinstance(arr, np.ndarray):
                        return arr
                except Exception:
                    continue
            raise TypeError(f"RVT '{name}' returned dict without ndarray values.")
        # Sequence of outputs: take first array-like
        if isinstance(result_obj, (list, tuple)):
            for v in result_obj:
                try:
                    arr = _as_float32_array(v, name)
                    if isinstance(arr, np.ndarray):
                        return arr
                except Exception:
                    continue
            raise TypeError(f"RVT '{name}' returned a sequence without ndarray values.")
        # Generic coercion
        try:
            arr = np.asarray(result_obj)
            if isinstance(arr, np.ndarray):
                return arr.astype(np.float32)
        except Exception:
            pass
        raise TypeError(f"RVT '{name}' returned unsupported type: {type(result_obj)}")

    def _call_with_radius(func, radius_val: float) -> np.ndarray:
        # try different keyword names for radius and nodata
        radius_keys = ("max_radius", "radius", "r_max", "search_radius", "max_search_radius")
        nodata_keys = ("no_data", "nodata")
        last_err: Optional[Exception] = None
        for rk in radius_keys:
            for nk in nodata_keys:
                try:
                    return func(dem=dtm_filled, resolution=pixel_size, **{rk: float(radius_val), nk: None})
                except TypeError as e:
                    last_err = e
                    continue
        # try without nodata keyword
        for rk in radius_keys:
            try:
                return func(dem=dtm_filled, resolution=pixel_size, **{rk: float(radius_val)})
            except TypeError as e:
                last_err = e
                continue
        # final fallback: minimal signature
        try:
            return func(dem=dtm_filled, resolution=pixel_size)
        except Exception as e:  # pragma: no cover - defensive
            if last_err is not None:
                raise last_err
            raise e

    svf_layers: List[np.ndarray] = []
    log_fn("  → SVF (Sky View Factor) hesaplanıyor...")
    radii_iter: Iterable[float]
    if show_progress:
        radii_iter = progress_bar(
            radii,
            desc="SVF hesaplama",
            unit="yarıçap",
            total=len(radii),
        )
    else:
        radii_iter = radii
    for radius in radii_iter:
        svf_res = _call_with_radius(rvt_vis.sky_view_factor, float(radius))
        svf = _as_float32_array(svf_res, "sky_view_factor")
        svf_layers.append(svf)
    svf_avg = np.mean(np.stack(svf_layers, axis=0), axis=0)
    # Openness variants differ across rvt versions; compute both pos/neg robustly
    def _compute_openness(radius_val: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(rvt_vis, "positive_openness") and hasattr(rvt_vis, "negative_openness"):
            pos = _as_float32_array(
                _call_with_radius(rvt_vis.positive_openness, radius_val),
                "positive_openness",
            )
            neg = _as_float32_array(
                _call_with_radius(rvt_vis.negative_openness, radius_val),
                "negative_openness",
            )
            return pos, neg
        # Single 'openness' function that returns both
        if hasattr(rvt_vis, "openness"):
            res = _call_with_radius(rvt_vis.openness, radius_val)
            # dict case
            if isinstance(res, dict):
                # try common keys
                pos_keys = ("positive_openness", "openness_positive", "pos_open", "pos")
                neg_keys = ("negative_openness", "openness_negative", "neg_open", "neg")
                pos_arr = None
                neg_arr = None
                for k in pos_keys:
                    if k in res:
                        pos_arr = _as_float32_array(res[k], "positive_openness")
                        break
                for k in neg_keys:
                    if k in res:
                        neg_arr = _as_float32_array(res[k], "negative_openness")
                        break
                if pos_arr is not None and neg_arr is not None:
                    return pos_arr, neg_arr
                # fallback: try to pick two ndarray-like values in order
                arrays = []
                for v in res.values():
                    try:
                        arrays.append(_as_float32_array(v, "openness"))
                    except Exception:
                        continue
                if len(arrays) >= 2:
                    return arrays[0], arrays[1]
                if len(arrays) == 1:
                    return arrays[0], arrays[0]
                LOGGER.warning("RVT openness dict değeri array içermiyor; sıfır kullanılıyor.")
            # sequence case
            if isinstance(res, (list, tuple)):
                seq = list(res)
                if len(seq) >= 2:
                    return (
                        _as_float32_array(seq[0], "positive_openness"),
                        _as_float32_array(seq[1], "negative_openness"),
                    )
                if len(seq) == 1:
                    arr = _as_float32_array(seq[0], "openness")
                    return arr, arr
                LOGGER.warning("RVT openness dizisi boş; sıfır kullanılıyor.")
            # single array case
            try:
                arr = _as_float32_array(res, "openness")
                return arr, arr
            except Exception:
                LOGGER.warning("RVT openness desteklenmeyen tip döndürdü; sıfır kullanılıyor.")
        # fallback if no openness available
        zeros = np.zeros_like(dtm_filled, dtype=np.float32)
        return zeros, zeros

    log_fn("  → Openness (pozitif/negatif) hesaplanıyor...")
    pos_open, neg_open = _compute_openness(float(max(radii)))

    log_fn("  → LRM (Local Relief Model) hesaplanıyor...")
    try:
        # LRM may also have different radius/nodata keywords
        def _call_lrm() -> np.ndarray:
            radius_keys = ("search_radius", "radius", "r_max", "max_radius", "max_search_radius")
            nodata_keys = ("no_data", "nodata")
            last_err: Optional[Exception] = None
            for rk in radius_keys:
                for nk in nodata_keys:
                    try:
                        return rvt_vis.local_relief_model(
                            dem=dtm_filled,
                            resolution=pixel_size,
                            **{rk: float(max(radii)), nk: None},
                        )
                    except TypeError as e:
                        last_err = e
                        continue
            for rk in radius_keys:
                try:
                    return rvt_vis.local_relief_model(
                        dem=dtm_filled,
                        resolution=pixel_size,
                        **{rk: float(max(radii))},
                    )
                except TypeError as e:
                    last_err = e
                    continue
            try:
                return rvt_vis.local_relief_model(dem=dtm_filled, resolution=pixel_size)
            except Exception as e:
                if last_err is not None:
                    raise last_err
                raise e

        lrm = _as_float32_array(_call_lrm(), "local_relief_model")
    except AttributeError:
        LOGGER.warning("rvt.vis.local_relief_model eksik; Gaussian fallback kullanılıyor.")
        _lrm_sigma = DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else float(gaussian_lrm_sigma)
        low_pass = gaussian_filter(dtm_filled, sigma=_lrm_sigma)
        lrm = (dtm_filled - low_pass).astype(np.float32)

    # slope signature also varies across versions; try with/without keywords.
    # If slope is unavailable in rvt.vis, fall back to numpy gradient.
    log_fn("  → Slope (eğim) hesaplanıyor...")
    slope: np.ndarray
    if hasattr(rvt_vis, "slope"):
        try:
            slope_res = rvt_vis.slope(
                dem=dtm_filled,
                resolution=pixel_size,
                output_units="degree",
                no_data=None,
            )
            slope = _as_float32_array(slope_res, "slope")
        except TypeError:
            try:
                slope = _as_float32_array(
                    rvt_vis.slope(dem=dtm_filled, resolution=pixel_size),
                    "slope",
                )
            except Exception as e:  # pragma: no cover - defensive
                raise e
    else:
        LOGGER.warning("rvt.vis.slope eksik; gradyan tabanlı fallback kullanılıyor.")
        # Compute slope in degrees: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        # Use filled DTM to avoid NaNs breaking gradient
        gy, gx = np.gradient(dtm_filled.astype(np.float32), pixel_size, pixel_size)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    for derivative in (svf_avg, pos_open, neg_open, lrm, slope):
        derivative[~valid_mask] = np.nan
    return svf_avg, pos_open, neg_open, lrm, slope


def _score_rvtlog(
    dtm: np.ndarray,
    pixel_size: float,
    pre_svf: Optional[np.ndarray] = None,
    pre_neg_open: Optional[np.ndarray] = None,
    pre_lrm: Optional[np.ndarray] = None,
    pre_slope: Optional[np.ndarray] = None,
    sigmas: Optional[Sequence[float]] = None,
    gaussian_gradient_sigma: Optional[float] = None,
    local_variance_window: Optional[int] = None,
    rvt_radii: Optional[Sequence[float]] = None,
    gaussian_lrm_sigma: Optional[float] = None,
) -> np.ndarray:
    """Birleşik RVT + LoG + gradyan klasik skor hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    if pre_svf is None or pre_neg_open is None or pre_lrm is None:
        svf, _, neg, lrm, _ = compute_derivatives_with_rvt(
            dtm,
            pixel_size=pixel_size,
            radii=rvt_radii,
            gaussian_lrm_sigma=gaussian_lrm_sigma,
            show_progress=False,
            log_steps=False,
        )
    else:
        svf, neg, lrm = pre_svf, pre_neg_open, pre_lrm
    _sigmas = DEFAULTS.sigma_scales if sigmas is None else sigmas
    log_responses = [
        np.abs(gaussian_laplace(dtm_filled, sigma=s, mode="nearest"))
        for s in _sigmas
    ]
    blob = np.maximum.reduce(log_responses)
    if pre_slope is None:
        _grad_sigma = (
            DEFAULTS.gaussian_gradient_sigma if gaussian_gradient_sigma is None else float(gaussian_gradient_sigma)
        )
        grad = gaussian_gradient_magnitude(dtm_filled, sigma=_grad_sigma, mode="nearest")
        slope_term = _norm01(grad)
    else:
        slope_term = _norm01(pre_slope)
    svf_c = 1.0 - _norm01(svf)
    neg_n = _norm01(neg)
    lrm_n = _norm01(lrm)
    _var_win = DEFAULTS.local_variance_window if local_variance_window is None else int(local_variance_window)
    var_n = _norm01(_local_variance(dtm_filled, size=_var_win))
    score = (
        0.30 * _norm01(blob)
        + 0.20 * lrm_n
        + 0.15 * _norm01(svf_c)
        + 0.15 * slope_term
        + 0.10 * neg_n
        + 0.10 * var_n
    )
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_hessian(dtm: np.ndarray, sigmas: Optional[Sequence[float]] = None) -> np.ndarray:
    """Çok ölçekli Hessian sırt/vadi yanıt hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    _sigmas = DEFAULTS.sigma_scales if sigmas is None else sigmas
    responses = [_hessian_response(dtm_filled, sigma=s) for s in _sigmas]
    score = np.maximum.reduce(responses)
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_morph(dtm: np.ndarray, radii: Optional[Sequence[int]] = None) -> np.ndarray:
    """Morfolojik beyaz/siyah top-hat belirginlik skoru hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    radii = DEFAULTS.morphology_radii if radii is None else radii
    wth_list = []
    bth_list = []
    for r in radii:
        size = (int(r), int(r))
        opening = grey_opening(dtm_filled, size=size, mode="nearest")
        closing = grey_closing(dtm_filled, size=size, mode="nearest")
        wth = dtm_filled - opening
        bth = closing - dtm_filled
        wth_list.append(_norm01(wth))
        bth_list.append(_norm01(bth))
    hill = np.maximum.reduce(wth_list)
    moat = np.maximum.reduce(bth_list)
    score = np.maximum(hill, moat)
    score[~valid] = np.nan
    return score.astype(np.float32)


def stack_channels(
    rgb: np.ndarray,
    svf: np.ndarray,
    pos_open: np.ndarray,
    neg_open: np.ndarray,
    lrm: np.ndarray,
    slope: np.ndarray,
    ndsm: np.ndarray,
    plan_curv: Optional[np.ndarray] = None,
    profile_curv: Optional[np.ndarray] = None,
    tpi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stack channels in the required order.
    
    Varsayılan 9 kanal (geriye uyumlu):
        [0-2]: RGB
        [3]: SVF (Sky-View Factor)
        [4]: Positive Openness
        [5]: Negative Openness
        [6]: LRM (Local Relief Model)
        [7]: Slope
        [8]: nDSM
    
    Gelişmiş 12 kanal (enable_curvature ve enable_tpi aktifse):
        [9]: Plan Curvature
        [10]: Profile Curvature
        [11]: TPI (Topographic Position Index)
    
    Args:
        rgb: RGB bantları (3, H, W)
        svf: Sky-View Factor
        pos_open: Positive Openness
        neg_open: Negative Openness
        lrm: Local Relief Model
        slope: Eğim
        ndsm: Normalize edilmiş DSM (DSM - DTM)
        plan_curv: Plan Curvature (opsiyonel)
        profile_curv: Profile Curvature (opsiyonel)
        tpi: Topographic Position Index (opsiyonel)
    
    Returns:
        Stacked tensor (C, H, W) where C is 9 or 12
    """
    if rgb.shape[0] != 3:
        raise ValueError("RGB input must have three channels.")
    
    # Temel 9 kanal
    channels = [
        rgb[0],
        rgb[1],
        rgb[2],
        svf,
        pos_open,
        neg_open,
        lrm,
        slope,
        ndsm,
    ]
    
    # Gelişmiş kanalları ekle (varsa)
    if plan_curv is not None:
        channels.append(plan_curv)
    if profile_curv is not None:
        channels.append(profile_curv)
    if tpi is not None:
        channels.append(tpi)
    
    stacked = np.stack([ch.astype(np.float32) for ch in channels], axis=0)
    return stacked


# ==============================================================================
# CBAM (Convolutional Block Attention Module) - DİKKAT MEKANİZMASI
# ==============================================================================

class ChannelAttention(torch.nn.Module):
    """
    Squeeze-and-Excitation tarzı Kanal Dikkat Modülü.
    
    Her kanalın (RGB, SVF, Curvature vb.) önemini dinamik olarak ağırlıklandırır.
    Model, hangi kanalların tespit için daha önemli olduğunu öğrenir.
    
    Örnek: Tümülüs tespitinde SVF ve TPI kanalları daha yüksek ağırlık alabilir,
    hendek tespitinde ise Profile Curvature ve Negative Openness ön plana çıkabilir.
    
    Args:
        in_channels: Giriş kanal sayısı (9 veya 12)
        reduction: Kanal azaltma oranı (default: 4 → 12 kanal için 3 kanala sıkıştırır)
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        # Global pooling
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        
        # Paylaşımlı MLP (iki yol için ortak)
        reduced_channels = max(in_channels // reduction, 1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling yolu
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling yolu
        max_out = self.fc(self.max_pool(x))
        # Birleştir ve sigmoid ile [0, 1] aralığına normalize et
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(torch.nn.Module):
    """
    Mekansal Dikkat Modülü.
    
    Görüntünün hangi bölgelerinin (hangi piksellerin) önemli olduğunu öğrenir.
    Arkeolojik yapı sınırlarına ve merkez bölgelere dikkat çeker.
    
    Args:
        kernel_size: Konvolüsyon çekirdek boyutu (tek sayı, default: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kanal boyunca average ve max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Birleştir
        combined = torch.cat([avg_out, max_out], dim=1)
        # Sigmoid ile [0, 1] aralığına normalize et
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class CBAM(torch.nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Hem kanal hem de mekansal dikkat uygular:
    1. Önce kanal dikkat: Hangi özellik kanalları önemli?
    2. Sonra mekansal dikkat: Hangi bölgeler önemli?
    
    Referans: Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    
    Arkeolojik tespit için avantajları:
    - Farklı yapı tipleri için farklı kanalları ön plana çıkarır
    - Yapı sınırlarına ve merkezlerine odaklanır
    - Gürültülü bölgeleri otomatik olarak azaltır
    
    Args:
        in_channels: Giriş kanal sayısı
        reduction: Kanal dikkat için azaltma oranı
        kernel_size: Mekansal dikkat için çekirdek boyutu
    """
    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionWrapper(torch.nn.Module):
    """
    Mevcut modele CBAM attention ekleyen sarmalayıcı.
    
    Giriş katmanından önce CBAM uygulayarak modelin hangi kanallara
    ve bölgelere dikkat etmesi gerektiğini öğrenmesini sağlar.
    
    Args:
        base_model: Sarmalanacak temel model (U-Net vb.)
        in_channels: Giriş kanal sayısı
        reduction: CBAM için kanal azaltma oranı
    """
    def __init__(self, base_model: torch.nn.Module, in_channels: int, reduction: int = 4):
        super().__init__()
        self.input_attention = CBAM(in_channels, reduction=reduction)
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Önce attention uygula
        x = self.input_attention(x)
        # Sonra base model'i çalıştır
        return self.base_model(x)


def get_num_channels(enable_curvature: bool = True, enable_tpi: bool = True) -> int:
    """
    Aktif kanalların toplam sayısını döndürür.
    
    Temel: 9 kanal (RGB + SVF + Openness + LRM + Slope + nDSM)
    + Curvature: +2 kanal (Plan + Profile)
    + TPI: +1 kanal
    
    Returns:
        9, 11, 10, veya 12 (aktif özelliklere bağlı)
    """
    base_channels = 9
    if enable_curvature:
        base_channels += 2  # Plan + Profile Curvature
    if enable_tpi:
        base_channels += 1  # TPI
    return base_channels


def build_model(
    arch: str = "Unet",
    encoder: str = "resnet34",
    in_ch: int = 12,
    enable_attention: bool = True,
    attention_reduction: int = 4,
) -> torch.nn.Module:
    """
    Segmentation model oluşturur (opsiyonel CBAM attention ile).
    
    Args:
        arch: Model mimarisi (Unet, UnetPlusPlus, DeepLabV3Plus, vb.)
        encoder: Encoder ismi (resnet34, resnet50, efficientnet-b3, vb.)
        in_ch: Giriş kanal sayısı (9 veya 12)
        enable_attention: CBAM attention modülünü etkinleştir
        attention_reduction: Attention için kanal azaltma oranı
        
    Returns:
        PyTorch modeli (attention sarmalayıcılı veya değil)
    """
    if smp is None:
        raise ImportError("Install segmentation_models_pytorch: pip install segmentation-models-pytorch")
    if not hasattr(smp, arch):
        raise ValueError(f"Architecture '{arch}' not found in segmentation_models_pytorch.")
    model_cls = getattr(smp, arch)
    base_model = model_cls(
        encoder_name=encoder,
        in_channels=in_ch,
        classes=1,
        activation=None,
    )
    
    # Attention modülü ekle (istenirse)
    if enable_attention:
        model = AttentionWrapper(base_model, in_ch, reduction=attention_reduction)
        LOGGER.debug(f"CBAM Attention modülü eklendi (reduction={attention_reduction})")
    else:
        model = base_model
    
    return model


def inflate_conv1_to_n(conv_w: torch.Tensor, in_ch: int = 9, mode: str = "avg") -> torch.Tensor:
    """
    Inflate a first-conv weight tensor (out,c,kH,kW) to match the requested in_ch.
    - If channels already match, the tensor is returned unchanged.
    - If fewer channels are present, extra channels are filled using the average (or first) channel.
    - If more channels exist than requested, the tensor is truncated.
    """
    if conv_w.ndim != 4:
        raise ValueError("inflate_conv1_to_n expects a 4D convolution weight tensor.")
    out_ch, c_in, kH, kW = conv_w.shape
    if c_in == in_ch:
        return conv_w.clone()
    if in_ch < c_in:
        return conv_w[:, :in_ch].clone()

    if mode == "avg":
        base = conv_w.mean(dim=1, keepdim=True)
    elif mode == "first":
        base = conv_w[:, :1]
    else:
        raise ValueError(f"Unsupported inflate mode: {mode}")

    extra = base.repeat(1, in_ch - c_in, 1, 1)
    return torch.cat([conv_w, extra], dim=1).clone()


def build_model_with_imagenet_inflated(
    arch: str = "Unet", 
    encoder: str = "resnet34", 
    in_ch: int = 12,
    enable_attention: bool = True,
    attention_reduction: int = 4,
) -> torch.nn.Module:
    """
    Build a model that uses an ImageNet-pretrained 3-ch encoder and inflate its first conv to in_ch.
    
    ImageNet encoder'ları 3 kanal (RGB) için eğitilmiştir. Bu fonksiyon:
    1. ImageNet ağırlıklarını yükler
    2. İlk konvolüsyon katmanını 3 kanaldan in_ch kanala genişletir ("inflate")
    3. Opsiyonel olarak CBAM attention ekler
    
    Args:
        arch: Model mimarisi
        encoder: Encoder ismi
        in_ch: Hedef kanal sayısı (9 veya 12)
        enable_attention: CBAM attention ekle
        attention_reduction: Attention için kanal azaltma oranı
    
    Decoder weights come from the 3-ch SMP model where shapes match; others remain default-initialized.
    """
    if smp is None:
        raise ImportError("Install segmentation_models_pytorch: pip install segmentation-models-pytorch")
    if not hasattr(smp, arch):
        raise ValueError(f"Architecture '{arch}' not found in segmentation_models_pytorch.")

    model_3 = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1, activation=None
    )
    state_3 = model_3.state_dict()

    model_9 = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights=None, in_channels=in_ch, classes=1, activation=None
    )
    state_9 = model_9.state_dict()

    for k, v in state_3.items():
        if k in state_9 and state_9[k].shape == v.shape:
            state_9[k] = v.clone()

    def _first_conv_key(state_dict: dict) -> Optional[str]:
        convs: List[Tuple[str, torch.Tensor]] = []
        for key, val in state_dict.items():
            if not key.endswith(".weight"):
                continue
            if getattr(val, "ndim", 0) != 4:
                continue
            convs.append((key, val))
        if not convs:
            return None

        def priority(item: Tuple[str, torch.Tensor]) -> Tuple[int, str]:
            key, tensor = item
            score = 0
            if "conv1" in key:
                score -= 20
            if "stem" in key:
                score -= 10
            if key.startswith("encoder."):
                score -= 5
            # prefer smaller input channel counts (likely RGB)
            score += int(tensor.shape[1])
            return (score, key)

        # Prefer weights with exactly three input channels, then any other.
        for target_channels in (3, 1, None):
            filtered = [
                item for item in convs if target_channels is None or item[1].shape[1] == target_channels
            ]
            if not filtered:
                continue
            filtered.sort(key=priority)
            return filtered[0][0]

        convs.sort(key=priority)
        return convs[0][0]

    conv1_key_src = _first_conv_key(state_3)
    conv1_key_tgt = _first_conv_key(state_9)
    if conv1_key_src is None or conv1_key_tgt is None:
        raise RuntimeError("Could not locate a 4D first-conv weight to inflate.")

    w3 = state_3[conv1_key_src]  # (out,3,kH,kW)
    state_9[conv1_key_tgt] = inflate_conv1_to_n(w3, in_ch=in_ch, mode="avg")

    model_9.load_state_dict(state_9, strict=False)
    
    # Attention modülü ekle (istenirse)
    if enable_attention:
        model = AttentionWrapper(model_9, in_ch, reduction=attention_reduction)
        LOGGER.debug(f"CBAM Attention modülü eklendi (ImageNet inflated, reduction={attention_reduction})")
    else:
        model = model_9
    
    return model


def load_weights(
    model: torch.nn.Module,
    weights_path: Path,
    map_location: torch.device,
) -> None:
    """Load model weights with helpful diagnostics on mismatch."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Weights file did not contain a valid state dict.")

    expected_in_ch = model.encoder.conv1.weight.shape[1]
    first_conv_key = None
    for key, tensor in state.items():
        if tensor.ndim == 4:
            first_conv_key = key
            break
    if first_conv_key:
        in_channels = state[first_conv_key].shape[1]
        if in_channels != expected_in_ch:
            raise RuntimeError(
                f"Encoder expects {expected_in_ch} channels but weights provide {in_channels}. "
                "Ensure the model was trained with nine-channel input."
            )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        stripped_state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped_state, strict=True)


def generate_windows(
    width: int, height: int, tile: int, overlap: int
) -> Generator[Tuple[Window, int, int], None, None]:
    """Yield raster windows with the desired tile size and overlap."""
    if overlap >= tile:
        raise ValueError("--overlap must be smaller than --tile.")
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("Stride must be positive; reduce overlap or increase tile size.")
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            win_height = min(tile, height - row)
            win_width = min(tile, width - col)
            yield Window(col, row, win_width, win_height), row, col


def make_feather_weights(h: int, w: int, tile: int, overlap: int) -> np.ndarray:
    """Cosine feathering weights for a tile of size (h,w)."""
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)
    ramp = max(1, min(overlap, tile // 2))

    def _ramp(n: int) -> np.ndarray:
        win = np.ones(n, dtype=np.float32)
        # Border tiles can be smaller; clamp ramp length
        local_ramp = max(1, min(ramp, n // 2 if n > 1 else 1))
        # half-cosine ramps
        t = np.linspace(0, np.pi, local_ramp, endpoint=False, dtype=np.float32)
        up = (1 - np.cos(t)) * 0.5  # 0â†’1
        win[:local_ramp] = up
        win[-local_ramp:] = up[::-1]
        return win

    wr = _ramp(h)
    wc = _ramp(w)
    return np.outer(wr, wc).astype(np.float32)


@dataclass
class InferenceOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    transform: Affine
    crs: Optional[RasterioCRS]


@dataclass
class ClassicModeOutput:
    prob_path: Path
    mask_path: Path
    prob_map: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    threshold: float


@dataclass
class ClassicOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float
    per_mode: Dict[str, ClassicModeOutput]


@dataclass
class FusionOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float


@dataclass
class YoloOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float


def infer_tiled(
    model: torch.nn.Module,
    input_path: Path,
    band_idx: Sequence[int],
    tile: int,
    overlap: int,
    device: torch.device,
    use_half: bool,
    threshold: float,
    mask_talls: Optional[float],
    out_prefix: Path,
    global_norm: bool,
    norm_sample_tiles: int,
    feather: bool,
    precomputed_deriv: Optional[PrecomputedDerivatives] = None,
    derivative_cache_tif: Optional[Path] = None,
    derivative_cache_meta: Optional[Path] = None,
    enable_curvature: bool = True,
    enable_tpi: bool = True,
    encoder: Optional[str] = None,
    min_area: Optional[float] = None,
    percentile_low: Optional[float] = None,
    percentile_high: Optional[float] = None,
    rvt_radii: Optional[Sequence[float]] = None,
    gaussian_lrm_sigma: Optional[float] = None,
    rgb_only: bool = False,
    tpi_radii: Tuple[int, ...] = (5, 15, 30),
) -> InferenceOutputs:
    """
    Run tiled inference and save outputs.
    
    Args:
        precomputed_deriv: Önceden hesaplanmış RVT türevleri (cache kullanımı için).
                          None ise her tile için RVT yeniden hesaplanır.
    """
    model.eval()
    model.to(device)

    rgb_only_log_emitted = False

    def log_rgb_only_once() -> None:
        nonlocal rgb_only_log_emitted
        if rgb_only and not rgb_only_log_emitted:
            LOGGER.info("RGB-only modu aktif: türev kanalları sıfırla dolduruldu; efektif girdi sadece RGB.")
            rgb_only_log_emitted = True

    if mask_talls is not None and band_idx[3] <= 0:
        LOGGER.warning("Yüksek obje maskeleme istendi ama DSM bandı eksik; devre dışı bırakılıyor.")
        mask_talls = None

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_path))
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        deriv_ds: Optional[rasterio.io.DatasetReader] = None
        deriv_band_map: Optional[Dict[str, int]] = None
        if (
            precomputed_deriv is None
            and derivative_cache_tif is not None
            and derivative_cache_meta is not None
            and not rgb_only
        ):
            info = load_derivative_raster_cache_info(
                derivative_cache_tif,
                derivative_cache_meta,
                input_path,
                band_idx,
                rvt_radii=rvt_radii,
                gaussian_lrm_sigma=gaussian_lrm_sigma,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tpi_radii,
            )
            if info is not None:
                deriv_band_map = {str(k): int(v) for k, v in dict(info.get("band_map", {})).items()}
                deriv_ds = stack.enter_context(rasterio.open(derivative_cache_tif))
                LOGGER.info("Derivatives raster-cache kullanılıyor: %s", derivative_cache_tif)

        pixels = int(height) * int(width)
        one_float_bytes = pixels * np.dtype(np.float32).itemsize
        est_bytes = one_float_bytes * 3 + pixels  # prob_acc + weight_acc + ndsm_max + valid_global(bool)
        avail = available_memory_bytes()
        use_memmap = False
        if avail is not None and avail > 0:
            # Prefer RAM if it comfortably fits; fall back to disk when it likely risks OOM.
            use_memmap = est_bytes > int(avail * 0.75)
        else:
            # If we cannot detect available RAM, be conservative only for very large accumulators.
            use_memmap = est_bytes >= int(24 * 1024**3)
        scratch_dir: Optional[Path] = make_scratch_dir("infer") if use_memmap else None
        if use_memmap:
            LOGGER.info(
                "Large raster detected (%dx%d). Using disk-backed accumulators under %s",
                width,
                height,
                scratch_dir,
            )

        prob_acc, _prob_acc_path = alloc_array(
            (height, width),
            np.float32,
            fill_value=0.0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="dl_prob_acc",
        )
        weight_acc, _weight_acc_path = alloc_array(
            (height, width),
            np.float32,
            fill_value=0.0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="dl_weight_acc",
        )
        ndsm_max, _ndsm_max_path = alloc_array(
            (height, width),
            np.float32,
            fill_value=-np.inf,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="dl_ndsm_max",
        )
        valid_global, _valid_global_path = alloc_array(
            (height, width),
            bool,
            fill_value=False,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="dl_valid_global",
        )

        total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
            width / max(tile - overlap, 1)
        )

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_half and device.type == "cuda"
            else nullcontext()
        )

        # ---- Global normalization pre-pass (optional) ----
        fixed_lows = None
        fixed_highs = None
        if global_norm:
            lows_list = []
            highs_list = []
            sampled = 0
            for window, row, col in generate_windows(width, height, tile, overlap):
                if precomputed_deriv is not None:
                    # Use precomputed full-raster derivatives; slice to window
                    row_start = int(window.row_off)
                    col_start = int(window.col_off)
                    row_end = row_start + int(window.height)
                    col_end = col_start + int(window.width)

                    rgb_s = precomputed_deriv.rgb[:, row_start:row_end, col_start:col_end].copy()
                    svf_s = precomputed_deriv.svf[row_start:row_end, col_start:col_end].copy()
                    pos_s = precomputed_deriv.pos_open[row_start:row_end, col_start:col_end].copy()
                    neg_s = precomputed_deriv.neg_open[row_start:row_end, col_start:col_end].copy()
                    lrm_s = precomputed_deriv.lrm[row_start:row_end, col_start:col_end].copy()
                    slope_s = precomputed_deriv.slope[row_start:row_end, col_start:col_end].copy()
                    ndsm_s = precomputed_deriv.ndsm[row_start:row_end, col_start:col_end].copy()
                    # Yeni kanallar
                    plan_curv_s = (
                        precomputed_deriv.plan_curv[row_start:row_end, col_start:col_end].copy()
                        if (enable_curvature and precomputed_deriv.plan_curv is not None)
                        else None
                    )
                    profile_curv_s = (
                        precomputed_deriv.profile_curv[row_start:row_end, col_start:col_end].copy()
                        if (enable_curvature and precomputed_deriv.profile_curv is not None)
                        else None
                    )
                    tpi_s = (
                        precomputed_deriv.tpi[row_start:row_end, col_start:col_end].copy()
                        if (enable_tpi and precomputed_deriv.tpi is not None)
                        else None
                    )
                    if rgb_only:
                        log_rgb_only_once()
                        Hs, Ws = rgb_s.shape[1], rgb_s.shape[2]
                        Z = np.zeros((Hs, Ws), dtype=np.float32)
                        svf_s = Z
                        pos_s = Z
                        neg_s = Z
                        lrm_s = Z
                        slope_s = Z
                        ndsm_s = Z
                        plan_curv_s = None
                        profile_curv_s = None
                        tpi_s = None
                    stack_s = stack_channels(rgb_s, svf_s, pos_s, neg_s, lrm_s, slope_s, ndsm_s, plan_curv_s, profile_curv_s, tpi_s)
                elif deriv_ds is not None and deriv_band_map is not None:
                    def read_band(idx: int) -> Optional[np.ndarray]:
                        if idx <= 0:
                            return None
                        data = src.read(idx, window=window, boundless=True, masked=True)
                        return np.ma.filled(data.astype(np.float32), np.nan)

                    rgb_s = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)

                    if rgb_only:
                        log_rgb_only_once()
                        Hs, Ws = rgb_s.shape[1], rgb_s.shape[2]
                        Z = np.zeros((Hs, Ws), dtype=np.float32)
                        svf_s = Z
                        pos_s = Z
                        neg_s = Z
                        lrm_s = Z
                        slope_s = Z
                        ndsm_s = Z
                        plan_curv_s = None
                        profile_curv_s = None
                        tpi_s = None
                    else:
                        band_names: List[str] = ["svf", "pos_open", "neg_open", "lrm", "slope", "ndsm"]
                        if enable_curvature:
                            band_names += ["plan_curv", "profile_curv"]
                        if enable_tpi:
                            band_names += ["tpi"]
                        indexes = [int(deriv_band_map[name]) for name in band_names]
                        deriv_stack = deriv_ds.read(indexes=indexes, window=window, boundless=True, masked=True)
                        deriv_stack = np.ma.filled(deriv_stack.astype(np.float32), np.nan)
                        svf_s, pos_s, neg_s, lrm_s, slope_s, ndsm_s = deriv_stack[:6]
                        off = 6
                        plan_curv_s = None
                        profile_curv_s = None
                        tpi_s = None
                        if enable_curvature:
                            plan_curv_s = deriv_stack[off]
                            profile_curv_s = deriv_stack[off + 1]
                            off += 2
                        if enable_tpi:
                            tpi_s = deriv_stack[off]

                    stack_s = stack_channels(
                        rgb_s,
                        svf_s,
                        pos_s,
                        neg_s,
                        lrm_s,
                        slope_s,
                        ndsm_s,
                        plan_curv_s,
                        profile_curv_s,
                        tpi_s,
                    )
                else:
                    def read_band(idx: int) -> Optional[np.ndarray]:
                        if idx <= 0:
                            return None
                        data = src.read(idx, window=window, boundless=True, masked=True)
                        return np.ma.filled(data.astype(np.float32), np.nan)

                    rgb_s = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
                    dsm_s = read_band(band_idx[3])
                    dtm_s = read_band(band_idx[4])
                    if dtm_s is None:
                        break

                    if rgb_only:
                        log_rgb_only_once()
                        Hs, Ws = rgb_s.shape[1], rgb_s.shape[2]
                        Z = np.zeros((Hs, Ws), dtype=np.float32)
                        svf_s = Z
                        pos_s = Z
                        neg_s = Z
                        lrm_s = Z
                        slope_s = Z
                        ndsm_s = Z
                        plan_curv_s = None
                        profile_curv_s = None
                        tpi_s = None
                    else:
                        ndsm_s = compute_ndsm(dsm_s, dtm_s)
                        svf_s, pos_s, neg_s, lrm_s, slope_s = compute_derivatives_with_rvt(
                            dtm_s,
                            pixel_size=pixel_size,
                            radii=rvt_radii,
                            gaussian_lrm_sigma=gaussian_lrm_sigma,
                            show_progress=False,
                            log_steps=False,
                        )
                        plan_curv_s = None
                        profile_curv_s = None
                        tpi_s = None
                        if enable_curvature:
                            plan_curv_s, profile_curv_s = compute_curvatures(dtm_s, pixel_size=pixel_size)
                        if enable_tpi:
                            tpi_s = compute_tpi_multiscale(dtm_s, radii=tpi_radii)
                    stack_s = stack_channels(rgb_s, svf_s, pos_s, neg_s, lrm_s, slope_s, ndsm_s, plan_curv_s, profile_curv_s, tpi_s)

                Hs, Ws = stack_s.shape[1], stack_s.shape[2]
                ch = min(TILE_SAMPLE_CROP_SIZE, Hs)
                cw = min(TILE_SAMPLE_CROP_SIZE, Ws)
                r0 = max(0, (Hs - ch) // 2)
                c0 = max(0, (Ws - cw) // 2)
                crop = stack_s[:, r0 : r0 + ch, c0 : c0 + cw]

                Cc = crop.shape[0]
                lows = np.zeros(Cc, dtype=np.float32)
                highs = np.zeros(Cc, dtype=np.float32)
                for ci in range(Cc):
                    v = crop[ci].ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        lows[ci] = 0.0
                        highs[ci] = 1.0
                    else:
                        _p_low = percentile_low if percentile_low is not None else DEFAULTS.percentile_low
                        _p_high = percentile_high if percentile_high is not None else DEFAULTS.percentile_high
                        lows[ci] = float(np.percentile(v, _p_low))
                        highs[ci] = float(np.percentile(v, _p_high))
                lows_list.append(lows)
                highs_list.append(highs)

                sampled += 1
                if sampled >= norm_sample_tiles:
                    break

            if lows_list and highs_list:
                fixed_lows = np.median(np.stack(lows_list, axis=0), axis=0)
                fixed_highs = np.median(np.stack(highs_list, axis=0), axis=0)
            if global_norm and (fixed_lows is None or fixed_highs is None):
                LOGGER.info("Global normalizasyon atlandı; karo başına normalizasyona geçiliyor.")

        for window, row, col in progress_bar(
            generate_windows(width, height, tile, overlap),
            total=total_tiles,
            desc="Inference",
            unit="tile",
        ):
            win_height = int(window.height)
            win_width = int(window.width)
            pad_h = max(0, tile - win_height)
            pad_w = max(0, tile - win_width)
            
            # Precomputed derivatives kullan (cache modunda)
            if precomputed_deriv is not None:
                # Tüm raster'dan tile'ı kes
                row_start = int(window.row_off)
                col_start = int(window.col_off)
                row_end = row_start + win_height
                col_end = col_start + win_width
                
                rgb = precomputed_deriv.rgb[:, row_start:row_end, col_start:col_end].copy()
                dsm = precomputed_deriv.dsm[row_start:row_end, col_start:col_end].copy() if precomputed_deriv.dsm is not None else None
                dtm = precomputed_deriv.dtm[row_start:row_end, col_start:col_end].copy()
                svf = precomputed_deriv.svf[row_start:row_end, col_start:col_end].copy()
                pos_open = precomputed_deriv.pos_open[row_start:row_end, col_start:col_end].copy()
                neg_open = precomputed_deriv.neg_open[row_start:row_end, col_start:col_end].copy()
                lrm = precomputed_deriv.lrm[row_start:row_end, col_start:col_end].copy()
                slope = precomputed_deriv.slope[row_start:row_end, col_start:col_end].copy()
                ndsm_tile = precomputed_deriv.ndsm[row_start:row_end, col_start:col_end].copy()
                
                # Yeni kanallar (varsa)
                plan_curv_tile = (
                    precomputed_deriv.plan_curv[row_start:row_end, col_start:col_end].copy()
                    if (enable_curvature and precomputed_deriv.plan_curv is not None)
                    else None
                )
                profile_curv_tile = (
                    precomputed_deriv.profile_curv[row_start:row_end, col_start:col_end].copy()
                    if (enable_curvature and precomputed_deriv.profile_curv is not None)
                    else None
                )
                tpi_tile = (
                    precomputed_deriv.tpi[row_start:row_end, col_start:col_end].copy()
                    if (enable_tpi and precomputed_deriv.tpi is not None)
                    else None
                )
                
                # Padding ekle (gerekirse)
                if pad_h or pad_w:
                    rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    dtm = np.pad(dtm, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    if dsm is not None:
                        dsm = np.pad(dsm, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    svf = np.pad(svf, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    pos_open = np.pad(pos_open, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    neg_open = np.pad(neg_open, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    lrm = np.pad(lrm, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    slope = np.pad(slope, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    ndsm_tile = np.pad(ndsm_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    # Yeni kanallar için padding
                    if plan_curv_tile is not None:
                        plan_curv_tile = np.pad(plan_curv_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    if profile_curv_tile is not None:
                        profile_curv_tile = np.pad(profile_curv_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    if tpi_tile is not None:
                        tpi_tile = np.pad(tpi_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)

                if rgb_only:
                    log_rgb_only_once()
                    Ht, Wt = rgb.shape[1], rgb.shape[2]
                    Z = np.zeros((Ht, Wt), dtype=np.float32)
                    svf = Z
                    pos_open = Z
                    neg_open = Z
                    lrm = Z
                    slope = Z
                    ndsm_tile = Z
                    plan_curv_tile = None
                    profile_curv_tile = None
                    tpi_tile = None

                valid_mask = np.isfinite(dtm)
            elif deriv_ds is not None and deriv_band_map is not None:
                def read_band(idx: int) -> Optional[np.ndarray]:
                    if idx <= 0:
                        return None
                    data = src.read(idx, window=window, boundless=True, masked=True)
                    return np.ma.filled(data.astype(np.float32), np.nan)

                rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)

                band_names: List[str] = ["svf", "pos_open", "neg_open", "lrm", "slope", "ndsm"]
                if enable_curvature:
                    band_names += ["plan_curv", "profile_curv"]
                if enable_tpi:
                    band_names += ["tpi"]
                indexes = [int(deriv_band_map[name]) for name in band_names]
                deriv_stack = deriv_ds.read(indexes=indexes, window=window, boundless=True, masked=True)
                deriv_stack = np.ma.filled(deriv_stack.astype(np.float32), np.nan)

                svf, pos_open, neg_open, lrm, slope, ndsm_tile = deriv_stack[:6]
                off = 6
                plan_curv_tile = None
                profile_curv_tile = None
                tpi_tile = None
                if enable_curvature:
                    plan_curv_tile = deriv_stack[off]
                    profile_curv_tile = deriv_stack[off + 1]
                    off += 2
                if enable_tpi:
                    tpi_tile = deriv_stack[off]

                if pad_h or pad_w:
                    rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    svf = np.pad(svf, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    pos_open = np.pad(pos_open, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    neg_open = np.pad(neg_open, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    lrm = np.pad(lrm, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    slope = np.pad(slope, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    ndsm_tile = np.pad(ndsm_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
                    if plan_curv_tile is not None:
                        plan_curv_tile = np.pad(
                            plan_curv_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan
                        )
                    if profile_curv_tile is not None:
                        profile_curv_tile = np.pad(
                            profile_curv_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan
                        )
                    if tpi_tile is not None:
                        tpi_tile = np.pad(tpi_tile, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)

                valid_mask = np.isfinite(ndsm_tile)
            else:
                # Normal mod - Her tile için RVT hesapla
                def read_band(idx: int) -> Optional[np.ndarray]:
                    if idx <= 0:
                        return None
                    data = src.read(idx, window=window, boundless=True, masked=True)
                    return np.ma.filled(data.astype(np.float32), np.nan)

                rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
                dsm = read_band(band_idx[3])
                dtm = read_band(band_idx[4])
                if dtm is None:
                    raise ValueError("DTM band is required (five-band input expected).")

                if pad_h or pad_w:
                    rgb = np.pad(
                        rgb,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    dtm = np.pad(
                        dtm,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    if dsm is not None:
                        dsm = np.pad(
                            dsm,
                            ((0, pad_h), (0, pad_w)),
                            mode="constant",
                            constant_values=np.nan,
                        )

                valid_mask = np.isfinite(dtm)

                if rgb_only:
                    log_rgb_only_once()
                    Ht, Wt = rgb.shape[1], rgb.shape[2]
                    Z = np.zeros((Ht, Wt), dtype=np.float32)
                    svf = Z
                    pos_open = Z
                    neg_open = Z
                    lrm = Z
                    slope = Z
                    ndsm_tile = Z
                    plan_curv_tile = None
                    profile_curv_tile = None
                    tpi_tile = None
                else:
                    ndsm_tile = compute_ndsm(dsm, dtm)
                    svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
                        dtm,
                        pixel_size=pixel_size,
                        radii=rvt_radii,
                        gaussian_lrm_sigma=gaussian_lrm_sigma,
                        show_progress=False,
                        log_steps=False,
                    )
                    plan_curv_tile = None
                    profile_curv_tile = None
                    tpi_tile = None
                    if enable_curvature:
                        plan_curv_tile, profile_curv_tile = compute_curvatures(dtm, pixel_size=pixel_size)
                    if enable_tpi:
                        tpi_tile = compute_tpi_multiscale(dtm, radii=tpi_radii)

            stacked = stack_channels(
                rgb=rgb,
                svf=svf,
                pos_open=pos_open,
                neg_open=neg_open,
                lrm=lrm,
                slope=slope,
                ndsm=ndsm_tile,
                plan_curv=plan_curv_tile,
                profile_curv=profile_curv_tile,
                tpi=tpi_tile,
            )
            if global_norm and fixed_lows is not None and fixed_highs is not None:
                normed = robust_norm_fixed(stacked, fixed_lows, fixed_highs)
            else:
                normed = robust_norm(
                    stacked,
                    p_low=percentile_low if percentile_low is not None else DEFAULTS.percentile_low,
                    p_high=percentile_high if percentile_high is not None else DEFAULTS.percentile_high,
                )

            tensor = torch.from_numpy(normed).unsqueeze(0).to(device)
            with torch.no_grad(), autocast_ctx:
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            row_slice = slice(row, row + win_height)
            col_slice = slice(col, col + win_width)
            tile_rows = row_slice.stop - row_slice.start
            tile_cols = col_slice.stop - col_slice.start
            prob_tile = probs.astype(np.float32)[:tile_rows, :tile_cols]
            valid_tile = valid_mask[:tile_rows, :tile_cols]

            if feather:
                W = make_feather_weights(tile_rows, tile_cols, tile, overlap)
            else:
                W = np.ones((tile_rows, tile_cols), dtype=np.float32)

            prob_acc[row_slice, col_slice] += prob_tile * W * valid_tile
            weight_acc[row_slice, col_slice] += W * valid_tile.astype(np.float32)
            valid_global[row_slice, col_slice] |= valid_tile
            ndsm_chunk = ndsm_tile.astype(np.float32)[:tile_rows, :tile_cols]
            existing = ndsm_max[row_slice, col_slice]
            ndsm_chunk[~valid_tile] = existing[~valid_tile]
            ndsm_max[row_slice, col_slice] = np.maximum(existing, ndsm_chunk)

        binary_mask, _binary_mask_path = alloc_array(
            (height, width),
            np.uint8,
            fill_value=0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="dl_binary_mask",
        )

        # Finalize in blocks to avoid huge temporary allocations (especially on large rasters).
        block_rows = 1024 if use_memmap else min(8192, int(height))
        total_blocks = math.ceil(int(height) / block_rows)
        for idx in progress_bar(
            range(total_blocks),
            desc="DL finalize",
            unit="block",
            total=total_blocks,
        ):
            row0 = idx * block_rows
            row1 = min(int(height), row0 + block_rows)
            acc_block = prob_acc[row0:row1, :]
            w_block = weight_acc[row0:row1, :]
            valid_block = valid_global[row0:row1, :]

            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(acc_block, w_block, out=acc_block, where=w_block > 0)
            acc_block[~valid_block] = np.nan

            if mask_talls is not None:
                tall_block = ndsm_max[row0:row1, :] > float(mask_talls)
                acc_block[tall_block] = 0.0

            binary_mask[row0:row1, :] = ((acc_block >= threshold) & valid_block).astype(np.uint8)

        prob_map = prob_acc

        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Parametreli dosya adı oluştur
        filename = build_filename_with_params(
            base_name=base_prefix.name,
            encoder=encoder,
            threshold=threshold,
            tile=tile,
            min_area=min_area,
        )
        
        prob_path = base_prefix.parent / f"{filename}_prob.tif"
        mask_path = base_prefix.parent / f"{filename}_mask.tif"

        write_prob_and_mask_rasters(
            prob_map=prob_map,
            mask=binary_mask,
            transform=transform,
            crs=crs,
            prob_path=prob_path,
            mask_path=mask_path,
        )

    # GPU belleğini temizle (bellek sızıntısını önle)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        LOGGER.debug("GPU belleği temizlendi")

    return InferenceOutputs(
        prob_path=prob_path,
        mask_path=mask_path,
        prob_map=prob_map,
        mask=binary_mask,
        transform=transform,
        crs=crs,
    )


def stretch_to_uint8(rgb: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Stretch float/int bands to uint8 using percentile bounds."""
    stretched = np.zeros_like(rgb, dtype=np.uint8)
    for idx in range(rgb.shape[0]):
        band = rgb[idx].astype(np.float32)
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        lo = float(np.nanpercentile(band, low))
        hi = float(np.nanpercentile(band, high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(band[valid]))
            hi = float(np.nanmax(band[valid]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        scaled = np.zeros_like(band, dtype=np.float32)
        scaled[valid] = (band[valid] - lo) / (hi - lo)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        stretched[idx] = (scaled * 255.0).astype(np.uint8)
    return stretched


def infer_yolo_tiled(
    input_path: Path,
    band_idx: Sequence[int],
    tile: int,
    overlap: int,
    out_prefix: Path,
    yolo_weights: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
    min_area: Optional[float] = None,
    precomputed_deriv: Optional[PrecomputedDerivatives] = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    save_labels: bool = True,
) -> YoloOutputs:
    """
    Run YOLO11 segmentation inference on RGB tiles.
    
    Args:
        input_path: Path to multi-band GeoTIFF
        band_idx: Band indices (RGB bands used: band_idx[0:3])
        tile: Tile size for sliding window
        overlap: Overlap between tiles
        out_prefix: Output file prefix
        yolo_weights: Path to YOLO weights file (e.g. 'yolo11n-seg.pt')
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
        imgsz: YOLO model input size
        device: Device for inference ('0', 'cpu', etc.)
        min_area: Minimum polygon area filter
        precomputed_deriv: Precomputed derivatives (for RGB extraction)
        percentile_low: Lower percentile for RGB stretch
        percentile_high: Upper percentile for RGB stretch
    
    Returns:
        YoloOutputs with probability map, mask, and metadata
    """
    if YOLO is None:
        raise ImportError(
            "Ultralytics YOLO gerekli. Yüklemek için: pip install ultralytics>=8.1.0"
        )
    
    # Load YOLO model
    if yolo_weights is None:
        # YOLOv8 daha olgun ve daha fazla topluluk desteği var
        yolo_weights = "yolov8s-seg.pt"
        LOGGER.info("YOLO ağırlık dosyası belirtilmedi, varsayılan kullanılıyor: %s", yolo_weights)
        LOGGER.info("Not: YOLOv8 (daha olgun) kullanılıyor. YOLO11 için: yolo_weights='yolo11s-seg.pt'")
    
    LOGGER.info("YOLO modeli yükleniyor: %s", yolo_weights)
    
    # Kuş bakışı (nadir) görüntüler için uyarı
    # Varsayılan COCO modelleri (yolov8, yolo11) için uyar, özel nadir modeller için uyarma
    weights_str = str(yolo_weights).lower()
    is_default_coco = ("yolov8" in weights_str or "yolo11" in weights_str) and \
                      "nadir" not in weights_str and \
                      "aerial" not in weights_str and \
                      "drone" not in weights_str and \
                      "visdrone" not in weights_str
    
    if is_default_coco:
        model_version = "YOLOv8" if "yolov8" in weights_str else "YOLO11"
        LOGGER.warning("")
        LOGGER.warning("=" * 70)
        LOGGER.warning("⚠️  KUŞ BAKIŞI (NADIR) GÖRÜNTÜ UYARISI")
        LOGGER.warning("=" * 70)
        LOGGER.warning("Varsayılan %s COCO modeli YATAY perspektiften eğitilmiştir.", model_version)
        LOGGER.warning("Arkeolojik LiDAR/İHA görüntüleri KUŞ BAKIŞI perspektiftedir.")
        LOGGER.warning("")
        LOGGER.warning("ÖNERİ:")
        LOGGER.warning("  1. VisDrone ile %s'i fine-tune edin (1-2 gün)", model_version)
        LOGGER.warning("     yolo segment train data=visdrone_yolo/data.yaml model=%s epochs=100", 
                      "yolov8s-seg.pt" if "yolov8" in weights_str else "yolo11s-seg.pt")
        LOGGER.warning("  2. Detaylar için: YOLO11_NADIR_TRAINING.md dosyasına bakın")
        LOGGER.warning("  3. Hazır nadir model: yolo_weights: 'models/yolov8_nadir_visdrone.pt'")
        LOGGER.warning("")
        LOGGER.warning("ŞUANKI SONUÇLAR: Genel envanter amaçlı, düşük doğruluk beklenir")
        LOGGER.warning("=" * 70)
        LOGGER.warning("")
    
    model = YOLO(yolo_weights)
    
    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        
        pixels = int(height) * int(width)
        one_float_bytes = pixels * np.dtype(np.float32).itemsize
        est_bytes = one_float_bytes * 2 + pixels  # prob_acc + weight_acc + valid_global(bool)
        avail = available_memory_bytes()
        use_memmap = False
        if avail is not None and avail > 0:
            use_memmap = est_bytes > int(avail * 0.75)
        else:
            use_memmap = est_bytes >= int(16 * 1024**3)
        scratch_dir: Optional[Path] = make_scratch_dir("yolo") if use_memmap else None
        if use_memmap:
            LOGGER.info(
                "Large raster detected (%dx%d). Using disk-backed accumulators under %s",
                width,
                height,
                scratch_dir,
            )

        # Initialize accumulator arrays
        prob_acc, _prob_acc_path = alloc_array(
            (height, width),
            np.float32,
            fill_value=0.0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="yolo_prob_acc",
        )
        weight_acc, _weight_acc_path = alloc_array(
            (height, width),
            np.float32,
            fill_value=0.0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="yolo_weight_acc",
        )
        valid_global, _valid_global_path = alloc_array(
            (height, width),
            bool,
            fill_value=False,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="yolo_valid_global",
        )
        
        # Initialize label detection storage
        all_detections: List[Dict[str, Any]] = []
        detection_id = 0
        
        total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
            width / max(tile - overlap, 1)
        )
        
        LOGGER.info("YOLO11 inference başlıyor: %dx%d piksel, %d tile", width, height, total_tiles)
        
        for window, row, col in progress_bar(
            generate_windows(width, height, tile, overlap),
            total=total_tiles,
            desc="YOLO11",
            unit="tile",
        ):
            win_height = int(window.height)
            win_width = int(window.width)
            
            # Read RGB bands
            if precomputed_deriv is not None:
                row_start = int(window.row_off)
                col_start = int(window.col_off)
                row_end = row_start + win_height
                col_end = col_start + win_width
                rgb = precomputed_deriv.rgb[:, row_start:row_end, col_start:col_end].copy()
            else:
                def read_band(idx: int) -> Optional[np.ndarray]:
                    if idx <= 0:
                        return None
                    data = src.read(idx, window=window, boundless=True, masked=True)
                    return np.ma.filled(data.astype(np.float32), np.nan)
                
                rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
            
            # Convert to uint8 for YOLO
            rgb_uint8 = stretch_to_uint8(rgb, low=percentile_low, high=percentile_high)
            
            # Transpose to HWC format and convert to BGR
            image_rgb = np.transpose(rgb_uint8, (1, 2, 0))
            image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
            
            # Check if tile is valid (not all NaN/zero)
            valid_mask = np.any(rgb_uint8 > 0, axis=0)
            if not np.any(valid_mask):
                continue
            
            # Run YOLO inference
            predict_kwargs = {
                'conf': conf_threshold,
                'iou': iou_threshold,
                'imgsz': imgsz,
                'verbose': False,
                'save': False,
            }
            if device is not None:
                predict_kwargs['device'] = device
            
            try:
                results = model.predict(image_bgr, **predict_kwargs)
                result = results[0]
                
                # Create probability map from masks
                tile_prob = np.zeros((win_height, win_width), dtype=np.float32)
                
                # Get class names from model
                class_names = result.names if hasattr(result, 'names') else {}
                
                if hasattr(result, 'masks') and result.masks is not None:
                    # Segmentation masks available
                    masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
                    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
                    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else None
                    
                    for i, mask in enumerate(masks_data):
                        # Resize mask to tile size if needed
                        if mask.shape != (win_height, win_width):
                            from scipy.ndimage import zoom
                            zy = win_height / mask.shape[0]
                            zx = win_width / mask.shape[1]
                            mask = zoom(mask, (zy, zx), order=1)
                        
                        # Weight by confidence
                        conf = float(confidences[i]) if confidences is not None and i < len(confidences) else 1.0
                        tile_prob = np.maximum(tile_prob, mask * conf)
                        
                        # Store detection info for label export
                        if save_labels and boxes is not None and i < len(boxes):
                            bbox = boxes[i]
                            xmin_tile, ymin_tile, xmax_tile, ymax_tile = bbox
                            
                            # Convert to global pixel coordinates
                            xmin_global = float(xmin_tile + col)
                            xmax_global = float(xmax_tile + col)
                            ymin_global = float(ymin_tile + row)
                            ymax_global = float(ymax_tile + row)
                            
                            # Get class info
                            class_id = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else -1
                            class_name = class_names.get(class_id, f"class_{class_id}")
                            
                            # Create polygon from mask in geographic coordinates
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            if np.any(mask_binary):
                                # Apply proper transformation: tile offset + main transform
                                # Window transform for this tile
                                tile_transform = transform * Affine.translation(col, row)
                                
                                # Convert mask to polygon using rasterio.features.shapes
                                geoms = list(shapes(mask_binary, mask=mask_binary, transform=tile_transform))
                                if geoms:
                                    geom_dict, _ = geoms[0]
                                    from shapely.geometry import shape as shapely_shape
                                    polygon = shapely_shape(geom_dict)
                                    
                                    # Calculate center in geographic coordinates from polygon
                                    center_geom = polygon.centroid
                                    center_x = float(center_geom.x)
                                    center_y = float(center_geom.y)
                                    
                                    all_detections.append({
                                        'id': detection_id,
                                        'class_id': class_id,
                                        'class_name': class_name,
                                        'confidence': conf,
                                        'bbox_xmin': xmin_global,
                                        'bbox_ymin': ymin_global,
                                        'bbox_xmax': xmax_global,
                                        'bbox_ymax': ymax_global,
                                        'center_x': center_x,
                                        'center_y': center_y,
                                        'tile_row': int(row),
                                        'tile_col': int(col),
                                        'geometry': polygon,
                                    })
                                    detection_id += 1
                
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # Only bounding boxes available - create masks from boxes
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (bbox, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        xmin, ymin, xmax, ymax = bbox.astype(int)
                        xmin = max(0, min(xmin, win_width - 1))
                        xmax = max(0, min(xmax, win_width))
                        ymin = max(0, min(ymin, win_height - 1))
                        ymax = max(0, min(ymax, win_height))
                        
                        if xmax > xmin and ymax > ymin:
                            tile_prob[ymin:ymax, xmin:xmax] = np.maximum(
                                tile_prob[ymin:ymax, xmin:xmax],
                                float(conf)
                            )
                            
                            # Store detection info for label export
                            if save_labels:
                                # Convert to global pixel coordinates
                                xmin_global = float(xmin + col)
                                xmax_global = float(xmax + col)
                                ymin_global = float(ymin + row)
                                ymax_global = float(ymax + row)
                                
                                # Get class name
                                class_name = class_names.get(int(class_id), f"class_{class_id}")
                                
                                # Create bounding box polygon in geographic coordinates
                                # Use Affine transform to convert pixel coords to geo coords
                                from shapely.geometry import Polygon
                                
                                # Get corners of bounding box in geo coordinates
                                # Affine transform * (col, row) -> (x_geo, y_geo)
                                corners = []
                                for px, py in [(xmin_global, ymin_global), 
                                              (xmax_global, ymin_global),
                                              (xmax_global, ymax_global),
                                              (xmin_global, ymax_global),
                                              (xmin_global, ymin_global)]:
                                    geo_x, geo_y = transform * (px, py)
                                    corners.append((geo_x, geo_y))
                                
                                polygon = Polygon(corners)
                                
                                # Calculate center from polygon
                                center_geom = polygon.centroid
                                center_x = float(center_geom.x)
                                center_y = float(center_geom.y)
                                
                                all_detections.append({
                                    'id': detection_id,
                                    'class_id': int(class_id),
                                    'class_name': class_name,
                                    'confidence': float(conf),
                                    'bbox_xmin': xmin_global,
                                    'bbox_ymin': ymin_global,
                                    'bbox_xmax': xmax_global,
                                    'bbox_ymax': ymax_global,
                                    'center_x': center_x,
                                    'center_y': center_y,
                                    'tile_row': int(row),
                                    'tile_col': int(col),
                                    'geometry': polygon,
                                })
                                detection_id += 1
                
                # Accumulate results
                row_slice = slice(row, row + win_height)
                col_slice = slice(col, col + win_width)
                
                valid_tile = valid_mask & np.isfinite(tile_prob)
                
                # Apply feathering weights
                weights = make_feather_weights(win_height, win_width, tile, overlap)
                
                prob_acc[row_slice, col_slice] += tile_prob * weights * valid_tile
                weight_acc[row_slice, col_slice] += weights * valid_tile.astype(np.float32)
                valid_global[row_slice, col_slice] |= valid_tile
                
            except Exception as e:
                LOGGER.warning("YOLO inference başarısız oldu (satır=%d, sütun=%d): %s", row, col, e)
                continue
        
        binary_mask, _binary_mask_path = alloc_array(
            (height, width),
            np.uint8,
            fill_value=0,
            use_memmap=use_memmap,
            scratch_dir=scratch_dir,
            name="yolo_binary_mask",
        )

        # Normalize accumulated probabilities in blocks to avoid huge temporaries.
        block_rows = 1024 if use_memmap else min(8192, int(height))
        total_blocks = math.ceil(int(height) / block_rows)
        for idx in progress_bar(
            range(total_blocks),
            desc="YOLO finalize",
            unit="block",
            total=total_blocks,
        ):
            row0 = idx * block_rows
            row1 = min(int(height), row0 + block_rows)
            acc_block = prob_acc[row0:row1, :]
            w_block = weight_acc[row0:row1, :]
            valid_block = valid_global[row0:row1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(acc_block, w_block, out=acc_block, where=w_block > 0)
            acc_block[~valid_block] = np.nan
            binary_mask[row0:row1, :] = ((acc_block >= conf_threshold) & valid_block).astype(np.uint8)

        prob_map = prob_acc
        
        # Build output paths
        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        filename = build_filename_with_params(
            base_name=base_prefix.name,
            mode_suffix="yolo11",
            threshold=conf_threshold,
            tile=tile,
            min_area=min_area,
        )
        
        prob_path = base_prefix.parent / f"{filename}_prob.tif"
        mask_path = base_prefix.parent / f"{filename}_mask.tif"
        
        # Write outputs
        write_prob_and_mask_rasters(
            prob_map=prob_map,
            mask=binary_mask,
            transform=transform,
            crs=crs,
            prob_path=prob_path,
            mask_path=mask_path,
        )
        
        # Export labeled detections to GeoPackage
        labels_path = None
        if save_labels and all_detections:
            LOGGER.info("YOLO11 etiketli tespitler GeoPackage'e yazılıyor: %d nesne", len(all_detections))
            labels_path = base_prefix.parent / f"{filename}_labels.gpkg"
            
            try:
                if gpd is not None:
                    # Create GeoDataFrame with all detections
                    gdf = gpd.GeoDataFrame(all_detections, geometry='geometry', crs=crs)
                    
                    # Calculate area for each detection (in square meters)
                    if crs and not crs.is_projected:
                        # If geographic CRS, project to calculate area
                        from pyproj import CRS as PyProjCRS, Transformer
                        area_crs = PyProjCRS.from_epsg(6933)  # Equal Area projection
                        transformer = Transformer.from_crs(crs, area_crs, always_xy=True)
                        gdf['area_m2'] = gdf.geometry.to_crs(area_crs).area
                    else:
                        gdf['area_m2'] = gdf.geometry.area
                    
                    # Reorder columns for better readability
                    column_order = [
                        'id', 'class_id', 'class_name', 'confidence', 'area_m2',
                        'center_x', 'center_y',
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                        'tile_row', 'tile_col', 'geometry'
                    ]
                    gdf = gdf[[col for col in column_order if col in gdf.columns]]
                    
                    # Save to GeoPackage
                    gdf.to_file(labels_path, driver="GPKG", layer="detections")
                    LOGGER.info("✓ YOLO11 etiketli tespitler kaydedildi: %s", labels_path)
                    
                    # Print summary statistics
                    class_counts = gdf['class_name'].value_counts()
                    LOGGER.info("Tespit edilen sınıflar:")
                    for class_name, count in class_counts.items():
                        LOGGER.info("  - %s: %d adet", class_name, count)
                
                else:
                    LOGGER.warning("GeoPandas yüklü değil, etiketli tespitler kaydedilemedi")
                    
            except Exception as e:
                LOGGER.error("YOLO11 etiketli tespitler kaydedilirken hata: %s", e)
        
        elif save_labels and not all_detections:
            LOGGER.info("YOLO11 hiç tespit bulamadı, etiketli çıktı yok")
        
        # GPU belleğini temizle (bellek sızıntısını önle)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            LOGGER.debug("GPU belleği temizlendi (YOLO)")

        return YoloOutputs(
            prob_path=prob_path,
            mask_path=mask_path,
            prob_map=prob_map,
            mask=binary_mask,
            transform=transform,
            crs=crs,
            threshold=conf_threshold,
        )


def infer_classic_tiled(
    input_path: Path,
    band_idx: Sequence[int],
    tile: int,
    overlap: int,
    out_prefix: Path,
    classic_th: Optional[float] = None,
    modes: Sequence[str] = ("rvtlog",),
    feather: bool = True,
    save_intermediate: bool = False,
    min_area: Optional[float] = None,
    precomputed_deriv: Optional[PrecomputedDerivatives] = None,
    derivative_cache_tif: Optional[Path] = None,
    derivative_cache_meta: Optional[Path] = None,
    enable_curvature: bool = True,
    enable_tpi: bool = True,
    tpi_radii: Tuple[int, ...] = (5, 15, 30),
    sigma_scales: Optional[Sequence[float]] = None,
    morphology_radii: Optional[Sequence[int]] = None,
    rvt_radii: Optional[Sequence[float]] = None,
    gaussian_gradient_sigma: Optional[float] = None,
    local_variance_window: Optional[int] = None,
    gaussian_lrm_sigma: Optional[float] = None,
) -> ClassicOutputs:
    """Run classical raster scoring methods in a tiled fashion."""
    if len(band_idx) < 5 or band_idx[4] <= 0:
        raise ValueError("DTM band index must be provided for classic inference.")
    base_modes = []
    for mode in modes:
        key = mode.strip().lower()
        if key not in base_modes:
            base_modes.append(key)
    if not base_modes:
        raise ValueError("At least one classic mode must be specified.")
    allowed = {"rvtlog", "hessian", "morph"}
    unknown = [m for m in base_modes if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported classic mode(s): {', '.join(unknown)}")
    base_prefix = out_prefix.with_suffix("")
    base_prefix.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_path))
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        deriv_ds: Optional[rasterio.io.DatasetReader] = None
        deriv_band_map: Optional[Dict[str, int]] = None
        if precomputed_deriv is None and derivative_cache_tif is not None and derivative_cache_meta is not None:
            info = load_derivative_raster_cache_info(
                derivative_cache_tif,
                derivative_cache_meta,
                input_path,
                band_idx,
                rvt_radii=rvt_radii,
                gaussian_lrm_sigma=gaussian_lrm_sigma,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tpi_radii,
            )
            if info is not None:
                deriv_band_map = {str(k): int(v) for k, v in dict(info.get("band_map", {})).items()}
                deriv_ds = stack.enter_context(rasterio.open(derivative_cache_tif))
                LOGGER.info("Classic iin derivatives raster-cache kullanlyor: %s", derivative_cache_tif)

        # Large-raster fast path: avoid per-mode full-size buffers + global percentile ops.
        pixels = int(height) * int(width)
        one_float_bytes = pixels * np.dtype(np.float32).itemsize
        est_bytes = one_float_bytes * 2 + pixels  # combined prob + weights + valid_global(bool)
        avail = available_memory_bytes()
        use_memmap = False
        if avail is not None and avail > 0:
            use_memmap = est_bytes > int(avail * 0.75)
        else:
            use_memmap = est_bytes >= int(16 * 1024**3)
        large_raster = pixels >= 200_000_000 or use_memmap

        if large_raster:
            scratch_dir: Optional[Path] = make_scratch_dir("classic") if use_memmap else None
            if use_memmap:
                LOGGER.info(
                    "Large raster detected (%dx%d). Using disk-backed classic accumulators under %s",
                    width,
                    height,
                    scratch_dir,
                )
            elif pixels >= 200_000_000:
                LOGGER.info(
                    "Large raster detected (%dx%d). Using streaming classic aggregation path.",
                    width,
                    height,
                )

            prob_acc_big, _prob_acc_path = alloc_array(
                (height, width),
                np.float32,
                fill_value=0.0,
                use_memmap=use_memmap,
                scratch_dir=scratch_dir,
                name="classic_prob_acc",
            )
            weight_acc_big, _weight_acc_path = alloc_array(
                (height, width),
                np.float32,
                fill_value=0.0,
                use_memmap=use_memmap,
                scratch_dir=scratch_dir,
                name="classic_weight_acc",
            )
            valid_global_big, _valid_global_path = alloc_array(
                (height, width),
                bool,
                fill_value=False,
                use_memmap=use_memmap,
                scratch_dir=scratch_dir,
                name="classic_valid_global",
            )

            dtm_idx = band_idx[4]
            total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
                width / max(tile - overlap, 1)
            )

            for window, row, col in progress_bar(
                generate_windows(width, height, tile, overlap),
                total=total_tiles,
                desc="Classic",
                unit="tile",
            ):
                dtm_window = src.read(dtm_idx, window=window, boundless=True, masked=True)
                dtm_tile = np.ma.filled(dtm_window.astype(np.float32), np.nan)

                win_height = int(window.height)
                win_width = int(window.width)
                row_slice = slice(row, row + win_height)
                col_slice = slice(col, col + win_width)
                valid_tile = np.isfinite(dtm_tile)
                valid_global_big[row_slice, col_slice] |= valid_tile
                if not np.any(valid_tile):
                    continue

                if feather:
                    weights = make_feather_weights(win_height, win_width, tile, overlap)
                else:
                    weights = np.ones((win_height, win_width), dtype=np.float32)

                prob_tiles: List[np.ndarray] = []
                for mode in base_modes:
                    if mode == "rvtlog":
                        if precomputed_deriv is not None:
                            row_start = int(window.row_off)
                            col_start = int(window.col_off)
                            row_end = row_start + win_height
                            col_end = col_start + win_width
                            svf_tile = precomputed_deriv.svf[row_start:row_end, col_start:col_end]
                            neg_tile = precomputed_deriv.neg_open[row_start:row_end, col_start:col_end]
                            lrm_tile = precomputed_deriv.lrm[row_start:row_end, col_start:col_end]
                            slope_tile = precomputed_deriv.slope[row_start:row_end, col_start:col_end]
                            score_tile = _score_rvtlog(
                                dtm_tile,
                                pixel_size=pixel_size,
                                pre_svf=svf_tile,
                                pre_neg_open=neg_tile,
                                pre_lrm=lrm_tile,
                                pre_slope=slope_tile,
                                sigmas=sigma_scales,
                                gaussian_gradient_sigma=gaussian_gradient_sigma,
                                local_variance_window=local_variance_window,
                                rvt_radii=rvt_radii,
                                gaussian_lrm_sigma=gaussian_lrm_sigma,
                            )
                        elif deriv_ds is not None and deriv_band_map is not None:
                            idxs = [
                                int(deriv_band_map["svf"]),
                                int(deriv_band_map["neg_open"]),
                                int(deriv_band_map["lrm"]),
                                int(deriv_band_map["slope"]),
                            ]
                            deriv_tile = deriv_ds.read(indexes=idxs, window=window, boundless=True, masked=True)
                            deriv_tile = np.ma.filled(deriv_tile.astype(np.float32), np.nan)
                            svf_tile, neg_tile, lrm_tile, slope_tile = (
                                deriv_tile[0],
                                deriv_tile[1],
                                deriv_tile[2],
                                deriv_tile[3],
                            )
                            score_tile = _score_rvtlog(
                                dtm_tile,
                                pixel_size=pixel_size,
                                pre_svf=svf_tile,
                                pre_neg_open=neg_tile,
                                pre_lrm=lrm_tile,
                                pre_slope=slope_tile,
                                sigmas=sigma_scales,
                                gaussian_gradient_sigma=gaussian_gradient_sigma,
                                local_variance_window=local_variance_window,
                                rvt_radii=rvt_radii,
                                gaussian_lrm_sigma=gaussian_lrm_sigma,
                            )
                        else:
                            score_tile = _score_rvtlog(
                                dtm_tile,
                                pixel_size=pixel_size,
                                sigmas=sigma_scales,
                                gaussian_gradient_sigma=gaussian_gradient_sigma,
                                local_variance_window=local_variance_window,
                                rvt_radii=rvt_radii,
                                gaussian_lrm_sigma=gaussian_lrm_sigma,
                            )
                    elif mode == "hessian":
                        score_tile = _score_hessian(dtm_tile, sigmas=sigma_scales)
                    elif mode == "morph":
                        score_tile = _score_morph(dtm_tile, radii=morphology_radii)
                    else:  # pragma: no cover - guarded above
                        continue

                    valid_mode = valid_tile & np.isfinite(score_tile)
                    if not np.any(valid_mode):
                        continue
                    score_tile_masked = np.where(valid_mode, score_tile, np.nan).astype(np.float32)
                    prob_tile = _norm01(score_tile_masked)
                    prob_tile[~valid_mode] = np.nan
                    prob_tiles.append(prob_tile.astype(np.float32, copy=False))

                if not prob_tiles:
                    continue

                if len(prob_tiles) == 1:
                    mean_prob = prob_tiles[0]
                else:
                    with np.errstate(all="ignore"):
                        mean_prob = np.nanmean(np.stack(prob_tiles, axis=0), axis=0).astype(np.float32)
                mean_prob[~valid_tile] = np.nan

                valid_combined = np.isfinite(mean_prob)
                if not np.any(valid_combined):
                    continue

                weight_tile = weights * valid_combined.astype(np.float32)
                prob_fill = np.nan_to_num(mean_prob, nan=0.0).astype(np.float32, copy=False)

                prob_acc_big[row_slice, col_slice] += prob_fill * weight_tile
                weight_acc_big[row_slice, col_slice] += weight_tile

            combined_prob = prob_acc_big
            combined_mask, _combined_mask_path = alloc_array(
                (height, width),
                np.uint8,
                fill_value=0,
                use_memmap=use_memmap,
                scratch_dir=scratch_dir,
                name="classic_binary_mask",
            )

            block_rows = 1024 if use_memmap else min(8192, int(height))
            total_blocks = math.ceil(int(height) / block_rows)
            for idx in progress_bar(
                range(total_blocks),
                desc="Classic finalize",
                unit="block",
                total=total_blocks,
            ):
                row0 = idx * block_rows
                row1 = min(int(height), row0 + block_rows)
                acc_block = combined_prob[row0:row1, :]
                w_block = weight_acc_big[row0:row1, :]
                valid_block = valid_global_big[row0:row1, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.divide(acc_block, w_block, out=acc_block, where=w_block > 0)
                acc_block[(w_block <= 0) | (~valid_block)] = np.nan
                np.clip(acc_block, 0.0, 1.0, out=acc_block)

            if classic_th is not None:
                combined_threshold = float(classic_th)
            else:
                combined_threshold = otsu_threshold_streaming(
                    combined_prob,
                    valid_global_big,
                    block_rows=block_rows,
                )

            for idx in progress_bar(
                range(total_blocks),
                desc="Classic mask",
                unit="block",
                total=total_blocks,
            ):
                row0 = idx * block_rows
                row1 = min(int(height), row0 + block_rows)
                prob_block = combined_prob[row0:row1, :]
                valid_block = np.isfinite(prob_block)
                combined_mask[row0:row1, :] = (valid_block & (prob_block >= combined_threshold)).astype(
                    np.uint8
                )

            classic_filename = build_filename_with_params(
                base_name=base_prefix.name,
                mode_suffix="classic",
                threshold=combined_threshold,
                tile=tile,
                min_area=min_area,
            )
            classic_prob_path = base_prefix.parent / f"{classic_filename}_prob.tif"
            classic_mask_path = base_prefix.parent / f"{classic_filename}_mask.tif"

            write_prob_and_mask_rasters(
                prob_map=combined_prob,
                mask=combined_mask,
                transform=transform,
                crs=crs,
                prob_path=classic_prob_path,
                mask_path=classic_mask_path,
            )

            return ClassicOutputs(
                prob_path=classic_prob_path,
                mask_path=classic_mask_path,
                prob_map=combined_prob,
                mask=combined_mask,
                transform=transform,
                crs=crs,
                threshold=combined_threshold,
                per_mode={},
            )

        prob_acc: Dict[str, np.ndarray] = {
            mode: np.zeros((height, width), dtype=np.float32) for mode in base_modes
        }
        weight_acc: Dict[str, np.ndarray] = {
            mode: np.zeros((height, width), dtype=np.float32) for mode in base_modes
        }
        valid_global = np.zeros((height, width), dtype=bool)
        dtm_idx = band_idx[4]

        total_tiles = math.ceil(height / max(tile - overlap, 1)) * math.ceil(
            width / max(tile - overlap, 1)
        )

        for window, row, col in progress_bar(
            generate_windows(width, height, tile, overlap),
            total=total_tiles,
            desc="Classic",
            unit="tile",
        ):
            dtm_window = src.read(dtm_idx, window=window, boundless=True, masked=True)
            dtm_tile = np.ma.filled(dtm_window.astype(np.float32), np.nan)

            win_height = int(window.height)
            win_width = int(window.width)
            row_slice = slice(row, row + win_height)
            col_slice = slice(col, col + win_width)
            valid_tile = np.isfinite(dtm_tile)
            valid_global[row_slice, col_slice] |= valid_tile
            if not np.any(valid_tile):
                continue

            if feather:
                weights = make_feather_weights(win_height, win_width, tile, overlap)
            else:
                weights = np.ones((win_height, win_width), dtype=np.float32)

            for mode in base_modes:
                if mode == "rvtlog":
                    # Use precomputed derivatives if available
                    if precomputed_deriv is not None:
                        row_start = int(window.row_off)
                        col_start = int(window.col_off)
                        row_end = row_start + win_height
                        col_end = col_start + win_width
                        svf_tile = precomputed_deriv.svf[row_start:row_end, col_start:col_end]
                        neg_tile = precomputed_deriv.neg_open[row_start:row_end, col_start:col_end]
                        lrm_tile = precomputed_deriv.lrm[row_start:row_end, col_start:col_end]
                        slope_tile = precomputed_deriv.slope[row_start:row_end, col_start:col_end]
                        score_tile = _score_rvtlog(
                            dtm_tile, pixel_size=pixel_size,
                            pre_svf=svf_tile, pre_neg_open=neg_tile, pre_lrm=lrm_tile,
                            pre_slope=slope_tile,
                            sigmas=sigma_scales,
                            gaussian_gradient_sigma=gaussian_gradient_sigma,
                            local_variance_window=local_variance_window,
                            rvt_radii=rvt_radii,
                            gaussian_lrm_sigma=gaussian_lrm_sigma,
                        )
                    elif deriv_ds is not None and deriv_band_map is not None:
                        idxs = [
                            int(deriv_band_map["svf"]),
                            int(deriv_band_map["neg_open"]),
                            int(deriv_band_map["lrm"]),
                            int(deriv_band_map["slope"]),
                        ]
                        deriv_tile = deriv_ds.read(indexes=idxs, window=window, boundless=True, masked=True)
                        deriv_tile = np.ma.filled(deriv_tile.astype(np.float32), np.nan)
                        svf_tile, neg_tile, lrm_tile, slope_tile = (
                            deriv_tile[0],
                            deriv_tile[1],
                            deriv_tile[2],
                            deriv_tile[3],
                        )
                        score_tile = _score_rvtlog(
                            dtm_tile,
                            pixel_size=pixel_size,
                            pre_svf=svf_tile,
                            pre_neg_open=neg_tile,
                            pre_lrm=lrm_tile,
                            pre_slope=slope_tile,
                            sigmas=sigma_scales,
                            gaussian_gradient_sigma=gaussian_gradient_sigma,
                            local_variance_window=local_variance_window,
                            rvt_radii=rvt_radii,
                            gaussian_lrm_sigma=gaussian_lrm_sigma,
                        )
                    else:
                        score_tile = _score_rvtlog(
                            dtm_tile, pixel_size=pixel_size,
                            sigmas=sigma_scales,
                            gaussian_gradient_sigma=gaussian_gradient_sigma,
                            local_variance_window=local_variance_window,
                            rvt_radii=rvt_radii,
                            gaussian_lrm_sigma=gaussian_lrm_sigma,
                        )
                elif mode == "hessian":
                    score_tile = _score_hessian(dtm_tile, sigmas=sigma_scales)
                elif mode == "morph":
                    score_tile = _score_morph(dtm_tile, radii=morphology_radii)
                else:  # pragma: no cover - guarded above
                    continue

                valid_mode = valid_tile & np.isfinite(score_tile)
                if not np.any(valid_mode):
                    continue
                score_tile_masked = np.where(valid_mode, score_tile, np.nan).astype(np.float32)
                prob_tile = _norm01(score_tile_masked)
                prob_tile[~valid_mode] = np.nan
                weight_tile = weights * valid_mode.astype(np.float32)
                prob_fill = np.nan_to_num(prob_tile, nan=0.0)

                prob_acc[mode][row_slice, col_slice] += prob_fill * weight_tile
                weight_acc[mode][row_slice, col_slice] += weight_tile

    per_mode_prob: Dict[str, np.ndarray] = {}
    per_mode_mask: Dict[str, np.ndarray] = {}
    per_mode_thresholds: Dict[str, float] = {}

    for mode in base_modes:
        acc = prob_acc[mode]
        weights = weight_acc[mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.divide(acc, weights, out=np.zeros_like(acc), where=weights > 0)
        prob_map = prob_map.astype(np.float32)
        prob_map[weights <= 0] = np.nan
        prob_map[~valid_global] = np.nan
        valid_mode = np.isfinite(prob_map)
        if np.any(valid_mode):
            prob_norm = _norm01(np.where(valid_mode, prob_map, np.nan))
            prob_norm[~valid_mode] = np.nan
        else:
            prob_norm = np.full_like(prob_map, np.nan, dtype=np.float32)
        if classic_th is not None:
            threshold = float(classic_th)
        else:
            threshold = _otsu_threshold_0to1(prob_norm, valid_mode)
        mask = np.zeros_like(prob_norm, dtype=np.uint8)
        mask[valid_mode & (prob_norm >= threshold)] = 1
        per_mode_prob[mode] = prob_norm.astype(np.float32)
        per_mode_mask[mode] = mask
        per_mode_thresholds[mode] = threshold

    if per_mode_prob:
        stack = np.stack([per_mode_prob[m] for m in base_modes], axis=0)
        combined_prob = np.nanmean(stack, axis=0).astype(np.float32)
    else:
        combined_prob = np.full((height, width), np.nan, dtype=np.float32)
    combined_prob[~valid_global] = np.nan
    combined_valid = np.isfinite(combined_prob)
    combined_prob = np.clip(combined_prob, 0.0, 1.0, out=combined_prob)

    if classic_th is not None:
        combined_threshold = float(classic_th)
    else:
        combined_threshold = _otsu_threshold_0to1(combined_prob, combined_valid)
    combined_mask = np.zeros_like(combined_prob, dtype=np.uint8)
    combined_mask[combined_valid & (combined_prob >= combined_threshold)] = 1

    # Parametreli dosya adı oluştur (classic için)
    classic_filename = build_filename_with_params(
        base_name=base_prefix.name,
        mode_suffix="classic",
        threshold=combined_threshold,
        tile=tile,
        min_area=min_area,
    )
    
    classic_prob_path = base_prefix.parent / f"{classic_filename}_prob.tif"
    classic_mask_path = base_prefix.parent / f"{classic_filename}_mask.tif"
    
    write_prob_and_mask_rasters(
        prob_map=combined_prob,
        mask=combined_mask,
        transform=transform,
        crs=crs,
        prob_path=classic_prob_path,
        mask_path=classic_mask_path,
    )

    per_mode_outputs: Dict[str, ClassicModeOutput] = {}
    write_individual = save_intermediate or len(base_modes) == 1
    if write_individual:
        for mode in base_modes:
            prob_map = per_mode_prob.get(mode)
            mask_map = per_mode_mask.get(mode)
            if prob_map is None or mask_map is None:
                continue
            
            # Her mode için ayrı parametreli dosya adı
            mode_filename = build_filename_with_params(
                base_name=base_prefix.name,
                mode_suffix=f"classic_{mode}",
                threshold=combined_threshold,
                tile=tile,
                min_area=min_area,
            )
            
            mode_prob_path = base_prefix.parent / f"{mode_filename}_prob.tif"
            mode_mask_path = base_prefix.parent / f"{mode_filename}_mask.tif"
            
            write_prob_and_mask_rasters(
                prob_map=prob_map,
                mask=mask_map,
                transform=transform,
                crs=crs,
                prob_path=mode_prob_path,
                mask_path=mode_mask_path,
            )
            
            per_mode_outputs[mode] = ClassicModeOutput(
                prob_path=mode_prob_path,
                mask_path=mode_mask_path,
                prob_map=prob_map,
                mask=mask_map,
                threshold=per_mode_thresholds.get(mode, combined_threshold),
            )

    return ClassicOutputs(
        prob_path=classic_prob_path,
        mask_path=classic_mask_path,
        prob_map=combined_prob,
        mask=combined_mask,
        transform=transform,
        crs=crs,
        threshold=combined_threshold,
        per_mode=per_mode_outputs,
    )


def vectorize_predictions(
    mask: np.ndarray,
    prob_map: np.ndarray,
    transform: Affine,
    crs: Optional[RasterioCRS],
    out_path: Path,
    min_area: float,
    simplify_tol: Optional[float],
    opening_size: int,
    label_connectivity: int,
) -> Optional[Path]:
    """Convert binary mask into polygons and write to GeoPackage."""
    if fiona is None and gpd is None:
        LOGGER.warning("Vektör çıktısı atlandı; vektörleştirme için geopandas veya fiona yükleyin.")
        return None
    if mapping is None or shape is None or shapely_transform is None or CRS is None or Transformer is None:
        LOGGER.warning("Vektör çıktısı atlandı; shapely/pyproj kurulu değil.")
        return None

    # Küçük gürültüleri temizlemek için binary opening (hızlı!)
    LOGGER.info("Küçük gürültüler temizleniyor...")
    k = max(1, int(opening_size))
    cleaned_mask = grey_opening(mask.astype(np.uint8), size=(k, k))
    
    LOGGER.info("Etiketleme yapılıyor...")
    # Connectivity: 4-neighbour (cross) or 8-neighbour (3x3 ones)
    if int(label_connectivity) == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        structure = np.ones((3, 3), dtype=int)
    labels, num_features = ndimage.label(cleaned_mask.astype(bool), structure=structure)
    if num_features == 0:
        LOGGER.info("Eşik üstünde özellik bulunamadı; vektörleştirme atlanıyor.")
        return None

    LOGGER.info("Tespit edilen özellik sayısı: %d", num_features)
    
    # Performans optimizasyonu: Çok küçük poligonları erken filtrelemek için piksel sayılarını hesapla
    label_ids = np.arange(1, num_features + 1)
    pixel_counts = ndimage.sum(mask.astype(np.uint8), labels, index=label_ids)
    prob_sums = ndimage.sum(prob_map.astype(np.float32), labels, index=label_ids)
    
    # Piksel tabanlı ön filtreleme: min_area'yı piksel sayısına çevir
    # Transform'dan piksel boyutunu hesapla (m²)
    pixel_width = abs(transform[0])  # x yönünde piksel boyutu
    pixel_height = abs(transform[4])  # y yönünde piksel boyutu
    pixel_area = pixel_width * pixel_height  # piksel alanı (m²)
    
    from math import ceil
    min_pixels = max(1, int(ceil(min_area / pixel_area)))
    valid_labels = pixel_counts >= min_pixels
    
    if not valid_labels.any():
        LOGGER.info("Tüm poligonlar minimum piksel sayısının altında; vektörleştirme atlanıyor.")
        return None
    
    filtered_count = valid_labels.sum()
    LOGGER.info("Piksel filtresinden sonra kalan poligon sayısı: %d (elenen: %d)", 
                filtered_count, num_features - filtered_count)
    
    # YENİ ETİKETLEME: Geçerli label'ları ardışık numaralarla yeniden etiketle
    # Bu, shapes() fonksiyonunun çok daha hızlı çalışmasını sağlar
    LOGGER.info("Label'lar yeniden numaralandırılıyor...")
    filtered_labels = np.zeros_like(labels)
    new_label_mapping = {}  # eski_id -> yeni_id
    new_id = 1
    filtered_pixel_counts = []
    filtered_prob_sums = []
    
    for old_id in range(1, num_features + 1):
        if valid_labels[old_id - 1]:
            filtered_labels[labels == old_id] = new_id
            new_label_mapping[new_id] = old_id
            filtered_pixel_counts.append(pixel_counts[old_id - 1])
            filtered_prob_sums.append(prob_sums[old_id - 1])
            new_id += 1
    
    # Dizileri numpy array'e çevir
    pixel_counts = np.array(filtered_pixel_counts)
    prob_sums = np.array(filtered_prob_sums)

    crs_obj: Optional[CRS] = None
    if crs:
        try:
            crs_obj = CRS.from_wkt(crs.to_wkt())
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("Unable to parse CRS; assuming projected coordinates for area.")

    to_area: Optional[Transformer] = None
    to_native: Optional[Transformer] = None
    if crs_obj and not crs_obj.is_projected:
        area_crs = CRS.from_epsg(6933)
        to_area = Transformer.from_crs(crs_obj, area_crs, always_xy=True)
        to_native = Transformer.from_crs(area_crs, crs_obj, always_xy=True)
    elif not crs_obj:
        LOGGER.warning(
            "Input CRS unknown; polygon areas assume coordinate units are meters."
        )

    records = []
    LOGGER.info("Poligonlar oluşturuluyor...")
    # Filtrelenmiş label'lardan poligon oluştur (artık ardışık numaralı)
    shape_generator = shapes(filtered_labels.astype(np.int32), mask=None, transform=transform)
    
    # Progress bar ile poligon işleme
    for geom, value in progress_bar(
        shape_generator,
        total=filtered_count + 1,  # +1 for background (0)
        desc="Vektörleştirme",
        unit="poligon"
    ):
        new_label_id = int(value)
        if new_label_id == 0:
            continue
        geom_shape = shape(geom)
        if to_area:
            geom_area_space = shapely_transform(to_area.transform, geom_shape)
            area_m2 = geom_area_space.area
        else:
            geom_area_space = geom_shape
            area_m2 = geom_shape.area
        if area_m2 < float(min_area):
            continue
        pixels = float(pixel_counts[new_label_id - 1])
        if pixels <= 0:
            continue
        mean_score = float(prob_sums[new_label_id - 1]) / pixels
        if simplify_tol and simplify_tol > 0:
            if to_area and to_native:
                simplified_area_geom = geom_area_space.simplify(
                    simplify_tol, preserve_topology=True
                )
                geom_shape = shapely_transform(to_native.transform, simplified_area_geom)
            else:
                geom_shape = geom_shape.simplify(simplify_tol, preserve_topology=True)
        records.append(
            {
                "id": int(new_label_id),
                "area_m2": float(area_m2),
                "score_mean": float(mean_score),
                "geometry": geom_shape,
            }
        )

    if not records:
        LOGGER.info("No polygons passed the min-area filter; skipping vector output.")
        return None

    out_path = out_path.with_suffix(".gpkg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("GeoPackage dosyası yazılıyor (%d poligon)...", len(records))
    if gpd is not None:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf.to_file(out_path, driver="GPKG")
    else:
        if fiona is None:
            LOGGER.warning("Vector output skipped; fiona not available.")
            return None
        schema = {
            "geometry": "MultiPolygon"
            if any(rec["geometry"].geom_type == "MultiPolygon" for rec in records)
            else "Polygon",
            "properties": {
                "id": "int",
                "area_m2": "float",
                "score_mean": "float",
            },
        }
        crs_wkt = crs.to_wkt() if crs else None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with fiona.open(
            out_path,
            mode="w",
            driver="GPKG",
            schema=schema,
            crs_wkt=crs_wkt,
        ) as dst:
            for rec in records:
                dst.write(
                    {
                        "geometry": mapping(rec["geometry"]),
                        "properties": {
                            "id": rec["id"],
                            "area_m2": rec["area_m2"],
                            "score_mean": rec["score_mean"],
                        },
                    }
                )
    return out_path



def parse_band_indexes(band_string: str) -> Tuple[int, int, int, int, int]:
    """Parse CSV band specification to integer tuple."""
    parts = [int(val.strip()) for val in band_string.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("--bands must specify exactly five entries.")
    if any(idx <= 0 for idx in parts[:3]):
        raise argparse.ArgumentTypeError("RGB band indices must be positive (1-based).")
    if parts[4] <= 0:
        raise argparse.ArgumentTypeError("DTM band index must be provided (>=1).")
    return tuple(parts)  # type: ignore[return-value]


def create_raster_metadata(
    height: int,
    width: int,
    transform: Affine,
    crs: Optional[RasterioCRS],
    dtype: str,
    nodata: Optional[float] = None,
) -> Dict[str, Any]:
    """Create standardized raster metadata dictionary."""
    # Choose predictor: 3 for floating point, 2 otherwise
    _dtype = str(dtype).lower()
    predictor_val = 3 if _dtype.startswith("float") else 2
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "transform": transform,
        "crs": crs,
        "dtype": dtype,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": predictor_val,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    return meta


def write_prob_and_mask_rasters(
    prob_map: np.ndarray,
    mask: np.ndarray,
    transform: Affine,
    crs: Optional[RasterioCRS],
    prob_path: Path,
    mask_path: Path,
) -> None:
    """Write probability and mask rasters with standard metadata."""
    height, width = prob_map.shape
    
    # Write probability raster
    prob_meta = create_raster_metadata(
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        dtype="float32",
        nodata=np.nan,
    )
    with rasterio.open(prob_path, "w", **prob_meta) as dst:
        dst.write(prob_map[np.newaxis, :, :])
    
    # Write mask raster
    mask_meta = create_raster_metadata(
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        dtype="uint8",
        nodata=0,
    )
    with rasterio.open(mask_path, "w", **mask_meta) as dst:
        dst.write(mask[np.newaxis, :, :])


def compute_fused_probability(
    dl_prob: np.ndarray,
    classic_prob: np.ndarray,
    alpha: float,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fused probability map and mask from DL and classic outputs.
    
    Args:
        dl_prob: Deep learning probability map
        classic_prob: Classical method probability map
        alpha: Mixing weight (1.0 = pure DL, 0.0 = pure classical)
        threshold: Probability threshold for binary mask
    
    Returns:
        Tuple of (fused_prob, fused_mask)
    """
    dl_filled = np.nan_to_num(dl_prob, nan=0.0)
    classic_filled = np.nan_to_num(classic_prob, nan=0.0)
    fused_prob = alpha * dl_filled + (1.0 - alpha) * classic_filled
    fused_prob = fused_prob.astype(np.float32)
    fused_valid = np.isfinite(dl_prob) | np.isfinite(classic_prob)
    fused_prob = np.clip(fused_prob, 0.0, 1.0, out=fused_prob)
    fused_prob[~fused_valid] = np.nan
    fused_mask = np.zeros_like(fused_prob, dtype=np.uint8)
    fused_mask[fused_valid & (fused_prob >= threshold)] = 1
    return fused_prob, fused_mask


def write_fused_probability_rasters_from_paths(
    dl_prob_path: Path,
    classic_prob_path: Path,
    *,
    alpha: float,
    threshold: float,
    out_prob_path: Path,
    out_mask_path: Path,
) -> Tuple[Affine, Optional[RasterioCRS]]:
    """Fuse two probability rasters in a streaming fashion (memory-safe for huge rasters)."""
    with rasterio.open(dl_prob_path) as dl_ds, rasterio.open(classic_prob_path) as cl_ds:
        if dl_ds.width != cl_ds.width or dl_ds.height != cl_ds.height:
            raise ValueError(
                f"Fusion raster sizes differ: dl={dl_ds.width}x{dl_ds.height}, classic={cl_ds.width}x{cl_ds.height}"
            )
        height = int(dl_ds.height)
        width = int(dl_ds.width)
        transform = dl_ds.transform
        crs = dl_ds.crs

        out_prob_path.parent.mkdir(parents=True, exist_ok=True)
        prob_meta = create_raster_metadata(
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            dtype="float32",
            nodata=np.nan,
        )
        mask_meta = create_raster_metadata(
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            dtype="uint8",
            nodata=0,
        )

        # Estimate block count for progress reporting.
        block_h = int(dl_ds.block_shapes[0][0]) if getattr(dl_ds, "block_shapes", None) else 256
        block_w = int(dl_ds.block_shapes[0][1]) if getattr(dl_ds, "block_shapes", None) else 256
        total_blocks = math.ceil(height / block_h) * math.ceil(width / block_w)

        with rasterio.open(out_prob_path, "w", **prob_meta) as out_prob, rasterio.open(
            out_mask_path, "w", **mask_meta
        ) as out_mask:
            for _, window in progress_bar(
                dl_ds.block_windows(1),
                total=total_blocks,
                desc="Fusion",
                unit="block",
            ):
                dl_block = dl_ds.read(1, window=window, masked=False).astype(np.float32, copy=False)
                cl_block = cl_ds.read(1, window=window, masked=False).astype(np.float32, copy=False)

                valid = np.isfinite(dl_block) | np.isfinite(cl_block)
                dl_block = np.nan_to_num(dl_block, nan=0.0, copy=False)
                cl_block = np.nan_to_num(cl_block, nan=0.0, copy=False)

                fused = alpha * dl_block + (1.0 - alpha) * cl_block
                fused = fused.astype(np.float32, copy=False)
                np.clip(fused, 0.0, 1.0, out=fused)
                fused[~valid] = np.nan

                mask = np.zeros(fused.shape, dtype=np.uint8)
                mask[valid & (fused >= threshold)] = 1

                out_prob.write(fused[np.newaxis, :, :], window=window)
                out_mask.write(mask[np.newaxis, :, :], window=window)

    return transform, crs


def resolve_out_prefix(input_path: Path, prefix: Optional[str]) -> Path:
    """Resolve output prefix path."""
    if prefix:
        out_path = Path(prefix)
        if out_path.is_dir():
            out_path = out_path / input_path.stem
    else:
        out_path = input_path.with_suffix("")

    # Route all raster/vector outputs under a dedicated subfolder: 'ciktilar'
    # Downstream code writes to base_prefix.parent; make that parent the 'ciktilar' folder
    return out_path.parent / "ciktilar" / out_path.name


def build_filename_with_params(
    base_name: str,
    encoder: Optional[str] = None,
    threshold: Optional[float] = None,
    tile: Optional[int] = None,
    alpha: Optional[float] = None,
    min_area: Optional[float] = None,
    mode_suffix: Optional[str] = None,
) -> str:
    """
    Parametreleri içeren dosya adı oluştur.
    
    Örnek: kesif_alani_resnet34_th0.5_tile1024_minarea80
    """
    parts = [base_name]
    
    if mode_suffix:
        parts.append(mode_suffix)
    
    if encoder:
        parts.append(encoder)
    
    if threshold is not None:
        # Threshold'u kısa formatta (0.5 -> th0.5)
        parts.append(f"th{threshold:.2f}".rstrip('0').rstrip('.'))
    
    if tile is not None:
        parts.append(f"tile{tile}")
    
    if alpha is not None:
        # Alpha değeri (fusion için)
        parts.append(f"alpha{alpha:.2f}".rstrip('0').rstrip('.'))
    
    if min_area is not None and min_area > 0:
        # Min area (80.0 -> minarea80)
        parts.append(f"minarea{int(min_area)}")
    
    return "_".join(parts)


def get_cache_path(input_path: Path, cache_dir: Optional[str] = None) -> Path:
    """RVT türevleri için cache dosya yolunu oluştur."""
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        # Dosya adını uzantısız al ve .derivatives.npz ekle
        # Örn: kesif_alani.tif -> kesif_alani.derivatives.npz
        return cache_path / f"{input_path.stem}.derivatives.npz"
    return input_path.with_suffix(".derivatives.npz")


def get_derivative_raster_cache_paths(
    input_path: Path, cache_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """Blok bazl RVT trev raster-cache (GeoTIFF) yolu + metadata JSON yolu."""
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        tif_path = cache_root / f"{input_path.stem}.derivatives_raster.tif"
        meta_path = cache_root / f"{input_path.stem}.derivatives_raster.json"
        return tif_path, meta_path
    return (
        input_path.with_suffix(".derivatives_raster.tif"),
        input_path.with_suffix(".derivatives_raster.json"),
    )


def _json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def _json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        LOGGER.warning("Raster-cache metadata okunamad (%s): %s", path, e)
        return None


def validate_derivative_raster_cache_metadata(
    metadata: Dict[str, Any],
    input_path: Path,
    band_idx: Sequence[int],
    *,
    rvt_radii: Optional[Sequence[float]],
    gaussian_lrm_sigma: Optional[float],
    enable_curvature: bool,
    enable_tpi: bool,
    tpi_radii: Sequence[int],
) -> bool:
    """Raster-cache metadata dogrula (girdi + parametre uyumu)."""
    required_keys = [
        "complete",
        "input_name",
        "input_mtime",
        "input_size",
        "bands",
        "shape",
        "pixel_size",
        "rvt_radii",
        "gaussian_lrm_sigma",
        "enable_curvature",
        "enable_tpi",
        "tpi_radii",
        "band_map",
    ]
    if not all(k in metadata for k in required_keys):
        LOGGER.warning("Raster-cache metadata eksik anahtarlar var; cache gecersiz")
        return False

    if not bool(metadata.get("complete", False)):
        LOGGER.warning("Raster-cache tamamlanmamis gorunuyor; cache gecersiz")
        return False

    if str(metadata.get("input_name", "")) != input_path.name:
        LOGGER.warning(
            "Raster-cache dosya adi uyusmuyor: cache'deki=%s, mevcut=%s",
            metadata.get("input_name"),
            input_path.name,
        )
        return False

    try:
        stat = input_path.stat()
        current_mtime = float(stat.st_mtime)
        current_size = int(stat.st_size)
    except (OSError, FileNotFoundError) as e:
        LOGGER.warning("Girdi dosyasi kontrol edilemedi: %s", e)
        return False

    cached_mtime = float(metadata.get("input_mtime", -1))
    cached_size = int(metadata.get("input_size", -1))
    if cached_size != current_size:
        LOGGER.warning("Girdi dosyasi boyutu degismis; cache gecersiz")
        return False
    if abs(cached_mtime - current_mtime) > 2.0:
        LOGGER.warning("Girdi dosyasi mtime degismis; cache gecersiz")
        return False

    if list(metadata.get("bands", [])) != list(band_idx):
        LOGGER.warning("Raster-cache bant sirasi degismis; cache gecersiz")
        return False

    wanted_radii = list(DEFAULTS.rvt_radii if rvt_radii is None else rvt_radii)
    cached_radii = list(metadata.get("rvt_radii", []))
    if len(cached_radii) != len(wanted_radii) or not np.allclose(
        np.array(cached_radii, dtype=np.float32),
        np.array(wanted_radii, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        LOGGER.warning("Raster-cache RVT radii parametreleri degismis; cache gecersiz")
        return False

    wanted_sigma = float(DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else gaussian_lrm_sigma)
    cached_sigma = float(metadata.get("gaussian_lrm_sigma", float("nan")))
    if not math.isfinite(cached_sigma) or abs(cached_sigma - wanted_sigma) > 1e-6:
        LOGGER.warning("Raster-cache gaussian_lrm_sigma degismis; cache gecersiz")
        return False

    if bool(metadata.get("enable_curvature", False)) != bool(enable_curvature):
        LOGGER.warning("Raster-cache enable_curvature degismis; cache gecersiz")
        return False
    if bool(metadata.get("enable_tpi", False)) != bool(enable_tpi):
        LOGGER.warning("Raster-cache enable_tpi degismis; cache gecersiz")
        return False

    wanted_tpi = [int(x) for x in (tpi_radii if enable_tpi else [])]
    cached_tpi = [int(x) for x in metadata.get("tpi_radii", [])]
    if wanted_tpi != cached_tpi:
        LOGGER.warning("Raster-cache tpi_radii degismis; cache gecersiz")
        return False

    band_map = metadata.get("band_map")
    if not isinstance(band_map, dict) or not band_map:
        LOGGER.warning("Raster-cache band_map eksik; cache gecersiz")
        return False

    required_bands = ["svf", "pos_open", "neg_open", "lrm", "slope", "ndsm"]
    if enable_curvature:
        required_bands += ["plan_curv", "profile_curv"]
    if enable_tpi:
        required_bands += ["tpi"]
    missing = [b for b in required_bands if b not in band_map]
    if missing:
        LOGGER.warning("Raster-cache band_map eksik kanallar: %s", ", ".join(missing))
        return False

    return True


def load_derivative_raster_cache_info(
    cache_tif_path: Path,
    cache_meta_path: Path,
    input_path: Path,
    band_idx: Sequence[int],
    *,
    rvt_radii: Optional[Sequence[float]],
    gaussian_lrm_sigma: Optional[float],
    enable_curvature: bool,
    enable_tpi: bool,
    tpi_radii: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Raster-cache mevcut ve uyumluysa metadata d”ndr."""
    if not cache_tif_path.exists() or not cache_meta_path.exists():
        return None
    metadata = _json_load(cache_meta_path)
    if metadata is None:
        return None
    if not validate_derivative_raster_cache_metadata(
        metadata,
        input_path,
        band_idx,
        rvt_radii=rvt_radii,
        gaussian_lrm_sigma=gaussian_lrm_sigma,
        enable_curvature=enable_curvature,
        enable_tpi=enable_tpi,
        tpi_radii=tpi_radii,
    ):
        return None

    # Quick sanity-check of the GeoTIFF container (shape + band count).
    try:
        with rasterio.open(cache_tif_path) as ds:
            shape = metadata.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                exp_h = int(shape[0])
                exp_w = int(shape[1])
                if int(ds.height) != exp_h or int(ds.width) != exp_w:
                    LOGGER.warning("Raster-cache boyutu uyumsuz; cache gecersiz")
                    return None
            band_map = metadata.get("band_map", {})
            if isinstance(band_map, dict) and band_map:
                max_band = max(int(v) for v in band_map.values())
                if int(ds.count) < max_band:
                    LOGGER.warning("Raster-cache band says yetersiz; cache gecersiz")
                    return None
    except Exception as e:
        LOGGER.warning("Raster-cache GeoTIFF a‡lamad (%s): %s", cache_tif_path, e)
        return None
    return metadata


def _estimate_derivative_cache_halo_px(
    pixel_size: float,
    *,
    rvt_radii: Optional[Sequence[float]],
    gaussian_lrm_sigma: Optional[float],
    enable_curvature: bool,
    enable_tpi: bool,
    tpi_radii: Sequence[int],
) -> int:
    # RVT radii are in meters; convert to pixels using resolution.
    radii_m = DEFAULTS.rvt_radii if rvt_radii is None else rvt_radii
    max_radius_m = float(max(radii_m)) if radii_m else 0.0
    px = max(pixel_size, 1e-9)
    halo_rvt = int(math.ceil(max_radius_m / px)) if max_radius_m > 0 else 0

    # Gaussian fallback uses sigma in pixels; use ~4*sigma support.
    sigma = float(DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else gaussian_lrm_sigma)
    halo_gauss = int(math.ceil(4.0 * sigma)) if sigma > 0 else 0

    halo_curv = 2 if enable_curvature else 0
    halo_tpi = int(max(tpi_radii)) if (enable_tpi and tpi_radii) else 0

    return max(2, halo_rvt, halo_gauss, halo_curv, halo_tpi)


def build_derivative_raster_cache(
    *,
    input_path: Path,
    band_idx: Sequence[int],
    cache_tif_path: Path,
    cache_meta_path: Path,
    recalculate: bool,
    rvt_radii: Optional[Sequence[float]],
    gaussian_lrm_sigma: Optional[float],
    enable_curvature: bool,
    enable_tpi: bool,
    tpi_radii: Sequence[int],
    chunk_size: int = 2048,
    halo_px: Optional[int] = None,
) -> Dict[str, Any]:
    """RVT trevlerini blok blok hesaplayp GeoTIFF olarak cache'ler."""
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmal")
    if halo_px is not None and halo_px < 0:
        raise ValueError("halo_px negatif olamaz")

    if recalculate:
        for p in (cache_tif_path, cache_meta_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    with rasterio.open(input_path) as src:
        height = int(src.height)
        width = int(src.width)
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        dtm_idx = int(band_idx[4])
        dsm_idx = int(band_idx[3]) if len(band_idx) >= 4 else 0
        if dtm_idx <= 0:
            raise ValueError("DTM band gerekli (band_idx[4])")

        halo = (
            int(halo_px)
            if halo_px is not None
            else _estimate_derivative_cache_halo_px(
                pixel_size,
                rvt_radii=rvt_radii,
                gaussian_lrm_sigma=gaussian_lrm_sigma,
                enable_curvature=enable_curvature,
                enable_tpi=enable_tpi,
                tpi_radii=tpi_radii,
            )
        )

        radii_used = list(DEFAULTS.rvt_radii if rvt_radii is None else rvt_radii)
        sigma_used = float(DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else gaussian_lrm_sigma)

        band_map: Dict[str, int] = {
            "svf": 1,
            "pos_open": 2,
            "neg_open": 3,
            "lrm": 4,
            "slope": 5,
            "ndsm": 6,
        }
        next_band = 7
        if enable_curvature:
            band_map["plan_curv"] = next_band
            band_map["profile_curv"] = next_band + 1
            next_band += 2
        if enable_tpi:
            band_map["tpi"] = next_band
            next_band += 1
        band_count = next_band - 1

        stat = input_path.stat()
        metadata: Dict[str, Any] = {
            "version": 1,
            "complete": False,
            "input_path": str(input_path),
            "input_name": input_path.name,
            "input_mtime": float(stat.st_mtime),
            "input_size": int(stat.st_size),
            "bands": list(band_idx),
            "shape": [height, width],
            "pixel_size": pixel_size,
            "rvt_radii": [float(x) for x in radii_used],
            "gaussian_lrm_sigma": float(sigma_used),
            "enable_curvature": bool(enable_curvature),
            "enable_tpi": bool(enable_tpi),
            "tpi_radii": [int(x) for x in (tpi_radii if enable_tpi else [])],
            "chunk_size": int(chunk_size),
            "halo_px": int(halo),
            "band_map": {k: int(v) for k, v in band_map.items()},
        }

        _json_dump(cache_meta_path, metadata)

        out_meta = create_raster_metadata(
            height=height,
            width=width,
            transform=transform,
            crs=crs,
            dtype="float32",
            nodata=np.nan,
        )
        out_meta["count"] = int(band_count)
        out_meta["BIGTIFF"] = "IF_SAFER"
        out_meta["interleave"] = "pixel"

        total_blocks = math.ceil(height / chunk_size) * math.ceil(width / chunk_size)
        LOGGER.info(
            "Raster-cache oluŸturuluyor: %s (%d band, block=%d, halo=%d px)",
            cache_tif_path,
            band_count,
            chunk_size,
            halo,
        )

        with rasterio.open(cache_tif_path, "w", **out_meta) as dst:
            for window, row, col in progress_bar(
                generate_windows(width, height, chunk_size, overlap=0),
                total=total_blocks,
                desc="DerivCache",
                unit="block",
            ):
                win_h = int(window.height)
                win_w = int(window.width)

                row0 = max(0, int(row) - halo)
                col0 = max(0, int(col) - halo)
                row1 = min(height, int(row) + win_h + halo)
                col1 = min(width, int(col) + win_w + halo)
                padded = Window(col0, row0, col1 - col0, row1 - row0)

                dtm_ma = src.read(dtm_idx, window=padded, boundless=False, masked=True)
                dtm = np.ma.filled(dtm_ma.astype(np.float32), np.nan)

                dsm: Optional[np.ndarray]
                if dsm_idx > 0:
                    dsm_ma = src.read(dsm_idx, window=padded, boundless=False, masked=True)
                    dsm = np.ma.filled(dsm_ma.astype(np.float32), np.nan)
                else:
                    dsm = None

                # Derivatives on padded window
                ndsm = compute_ndsm(dsm, dtm)
                svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
                    dtm,
                    pixel_size=pixel_size,
                    radii=rvt_radii,
                    gaussian_lrm_sigma=gaussian_lrm_sigma,
                    show_progress=False,
                    log_steps=False,
                )

                plan_curv: Optional[np.ndarray] = None
                profile_curv: Optional[np.ndarray] = None
                tpi: Optional[np.ndarray] = None
                if enable_curvature:
                    plan_curv, profile_curv = compute_curvatures(dtm, pixel_size=pixel_size)
                if enable_tpi:
                    tpi = compute_tpi_multiscale(dtm, radii=tuple(int(x) for x in tpi_radii))

                # Crop to the original (non-halo) window
                roff = int(row) - row0
                coff = int(col) - col0
                rs = slice(roff, roff + win_h)
                cs = slice(coff, coff + win_w)

                layers: List[np.ndarray] = [
                    svf[rs, cs],
                    pos_open[rs, cs],
                    neg_open[rs, cs],
                    lrm[rs, cs],
                    slope[rs, cs],
                    ndsm[rs, cs],
                ]
                if enable_curvature and plan_curv is not None and profile_curv is not None:
                    layers.append(plan_curv[rs, cs])
                    layers.append(profile_curv[rs, cs])
                if enable_tpi and tpi is not None:
                    layers.append(tpi[rs, cs])

                data = np.stack(layers, axis=0).astype(np.float32, copy=False)
                dst.write(data, window=window)

        metadata["complete"] = True
        _json_dump(cache_meta_path, metadata)

        LOGGER.info("Raster-cache hazr: %s", cache_tif_path)
        return metadata


def save_derivatives_cache(
    cache_path: Path,
    derivatives_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """RVT türevlerini compressed numpy dosyası olarak kaydet."""
    LOGGER.info(f"RVT türevleri kaydediliyor: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Tüm verileri birleştir
    save_dict = {**derivatives_data, **{"_metadata": metadata}}
    
    # Compressed format ile kaydet
    np.savez_compressed(cache_path, **save_dict)
    
    # Boyut bilgisi
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    LOGGER.info(f"Cache kaydedildi: {size_mb:.1f} MB")


def load_derivatives_cache(cache_path: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """RVT türevlerini cache'den yükle."""
    if not cache_path.exists():
        return None
    
    try:
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        LOGGER.info(f"Cache yükleniyor: {cache_path} ({size_mb:.1f} MB)")
        
        # Dosyayı aç
        with np.load(cache_path, allow_pickle=True) as data:
            # Metadata'yı ayır
            if "_metadata" not in data:
                LOGGER.warning("Cache dosyası geçersiz (metadata eksik)")
                return None
            
            metadata = data["_metadata"].item()
            
            # Diğer verileri progress bar ile yükle
            derivatives_data = {}
            layer_names = [key for key in data.files if key != "_metadata"]
            
            for key in progress_bar(
                layer_names,
                desc="Cache katmanları yükleniyor",
                unit="katman",
                total=len(layer_names)
            ):
                derivatives_data[key] = data[key].copy()
        
        LOGGER.info(f"✓ Cache yüklendi: {len(derivatives_data)} katman")
        
        return derivatives_data, metadata
        
    except Exception as e:
        LOGGER.error(f"Cache yüklenirken hata: {e}")
        return None


def validate_cache(
    metadata: Dict[str, Any],
    input_path: Path,
    bands: Sequence[int],
    *,
    rvt_radii: Optional[Sequence[float]] = None,
    gaussian_lrm_sigma: Optional[float] = None,
    enable_curvature: bool = True,
    enable_tpi: bool = True,
    tpi_radii: Optional[Sequence[int]] = None,
) -> bool:
    """
    Cache'in mevcut parametrelerle uyumlu olup olmadini kontrol et.
    
    NOT: tile ve overlap parametreleri cache gecerliligini ETKILEMEZ!
    RVT turevleri tum raster icin bir kez hesaplanir, tile/overlap sadece
    model inference sirasinda kullanilir.
    
    NOT: Dosya yolu degisse bile, dosya adi ve mtime ayniysa cache gecerlidir.
    Bu, dosyalarin farkli dizinlere tasinmasi durumunda cache'in kullanilabilmesini saglar.
    
    Args:
        metadata: Cache metadata sozlugu
        input_path: Girdi dosyasi yolu
        bands: Bant indeksleri
        rvt_radii: RVT radii parametreleri
        gaussian_lrm_sigma: LRM sigma parametresi
        enable_curvature: Curvature kanallari aktif mi
        enable_tpi: TPI kanali aktif mi
        tpi_radii: TPI radii parametreleri (enable_tpi=True ise kontrol edilir)
    
    Returns:
        bool: Cache gecerli mi
    """
    required_keys = ["input_path", "input_mtime", "input_size", "bands", "shape", "rvt_radii", "gaussian_lrm_sigma"]
    
    if not all(key in metadata for key in required_keys):
        LOGGER.warning("Cache metadata eksik")
        return False
    
    # Dosya adi kontrolu (yol degisse bile dosya adi ayni olmali)
    cached_path = Path(metadata["input_path"])
    if cached_path.name != input_path.name:
        LOGGER.warning(f"Cache dosya adi uyusmuyor: cache'deki={cached_path.name}, mevcut={input_path.name}")
        return False
    
    # Dosya degismis mi? (mtime kontrolu - dosya tasinsa bile mtime genelde ayni kalir)
    try:
        stat = input_path.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        cached_mtime = metadata["input_mtime"]
        cached_size = metadata["input_size"]
        # 2 saniye tolerans (dosya sistemi zamanlama farklari icin)
        if abs(cached_mtime - current_mtime) > 2.0:
            LOGGER.warning(f"Girdi dosyasi degismis, cache gecersiz (mtime farki: {abs(cached_mtime - current_mtime):.1f} saniye)")
            return False
        if int(cached_size) != int(current_size):
            LOGGER.warning("Girdi dosyasi boyutu degismis, cache gecersiz")
            return False
    except (OSError, FileNotFoundError) as e:
        LOGGER.warning(f"Girdi dosyasi kontrol edilemedi: {e}")
        return False

    # Raster boyutu (H,W) kontrolu
    try:
        with rasterio.open(input_path) as src:
            h = int(src.height)
            w = int(src.width)
        cached_shape = metadata["shape"]
        if tuple(cached_shape) != (h, w):
            LOGGER.warning("Raster boyutu degismis, cache gecersiz")
            return False
    except Exception as e:
        LOGGER.warning(f"Raster boyutu kontrol edilemedi: {e}")
        return False
    
    # Bant sirasi ayni mi?
    if metadata["bands"] != list(bands):
        LOGGER.warning(f"Bant sirasi degismis, cache gecersiz (cache: {metadata['bands']}, mevcut: {list(bands)})")
        return False

    # RVT parametreleri (radii/sigma) ayni mi?
    wanted_radii = list(DEFAULTS.rvt_radii if rvt_radii is None else rvt_radii)
    cached_radii = list(metadata.get("rvt_radii", []))
    if len(wanted_radii) != len(cached_radii) or not np.allclose(
        np.array(wanted_radii, dtype=np.float32),
        np.array(cached_radii, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        LOGGER.warning("RVT radii degismis, cache gecersiz")
        return False
    wanted_sigma = float(DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else gaussian_lrm_sigma)
    cached_sigma = float(metadata.get("gaussian_lrm_sigma", float("nan")))
    if not math.isfinite(cached_sigma) or abs(cached_sigma - wanted_sigma) > 1e-6:
        LOGGER.warning("gaussian_lrm_sigma degismis, cache gecersiz")
        return False
    
    # enable_curvature parametresi kontrolu
    cached_enable_curvature = metadata.get("enable_curvature", False)
    if enable_curvature != cached_enable_curvature:
        LOGGER.warning(f"enable_curvature degismis (cache: {cached_enable_curvature}, mevcut: {enable_curvature}), cache gecersiz")
        return False
    
    # enable_tpi parametresi kontrolu
    cached_enable_tpi = metadata.get("enable_tpi", False)
    if enable_tpi != cached_enable_tpi:
        LOGGER.warning(f"enable_tpi degismis (cache: {cached_enable_tpi}, mevcut: {enable_tpi}), cache gecersiz")
        return False
    
    # TPI radii kontrolu (sadece TPI aktifse)
    if enable_tpi:
        wanted_tpi_radii = list(DEFAULTS.tpi_radii if tpi_radii is None else tpi_radii)
        cached_tpi_radii = list(metadata.get("tpi_radii", []))
        if len(wanted_tpi_radii) != len(cached_tpi_radii) or wanted_tpi_radii != cached_tpi_radii:
            LOGGER.warning(f"tpi_radii degismis (cache: {cached_tpi_radii}, mevcut: {wanted_tpi_radii}), cache gecersiz")
            return False
    
    LOGGER.info("Cache gecerli")
    return True


@dataclass
class PrecomputedDerivatives:
    """
    Önceden hesaplanmış RVT türevlerini ve gelişmiş topografik özellikleri tutan sınıf.
    
    Temel Kanallar (9 kanal - geriye uyumlu):
        rgb: RGB bantları (3, H, W)
        svf: Sky-View Factor (H, W)
        pos_open: Positive Openness (H, W)
        neg_open: Negative Openness (H, W)
        lrm: Local Relief Model (H, W)
        slope: Eğim (H, W)
        ndsm: Normalize edilmiş DSM (H, W)
    
    Gelişmiş Kanallar (3 ek kanal):
        plan_curv: Plan Curvature - hendek/sırt ayrımı (H, W)
        profile_curv: Profile Curvature - teraslar (H, W)
        tpi: Topographic Position Index - höyük/çukur (H, W)
    """
    rgb: np.ndarray          # (3, H, W)
    dsm: Optional[np.ndarray]  # (H, W) or None
    dtm: np.ndarray          # (H, W)
    svf: np.ndarray          # (H, W)
    pos_open: np.ndarray     # (H, W)
    neg_open: np.ndarray     # (H, W)
    lrm: np.ndarray          # (H, W)
    slope: np.ndarray        # (H, W)
    ndsm: np.ndarray         # (H, W)
    # Yeni gelişmiş kanallar
    plan_curv: Optional[np.ndarray] = None    # (H, W) - Plan Curvature
    profile_curv: Optional[np.ndarray] = None # (H, W) - Profile Curvature
    tpi: Optional[np.ndarray] = None          # (H, W) - Topographic Position Index
    # Metadata
    transform: Affine = field(default=None)  # type: ignore
    crs: Optional[RasterioCRS] = None
    pixel_size: float = 1.0
    

def precompute_derivatives(
    input_path: Path,
    band_idx: Sequence[int],
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
    recalculate: bool = False,
    rvt_radii: Optional[Sequence[float]] = None,
    gaussian_lrm_sigma: Optional[float] = None,
    enable_curvature: bool = True,
    enable_tpi: bool = True,
    tpi_radii: Tuple[int, ...] = (5, 15, 30),
) -> Optional[PrecomputedDerivatives]:
    """
    Tüm raster için RVT türevlerini ve gelişmiş topografik özellikleri önceden hesapla.
    
    Bu fonksiyon:
    1. Cache varsa ve geçerliyse yükler
    2. Yoksa tüm raster'ı okur ve türevleri hesaplar
    3. İstenirse cache'e kaydeder
    
    Hesaplanan kanallar:
        - Temel 9 kanal: RGB, SVF, Openness (±), LRM, Slope, nDSM
        - Curvature (enable_curvature=True): Plan + Profile Curvature
        - TPI (enable_tpi=True): Multi-scale Topographic Position Index
    
    Args:
        input_path: GeoTIFF dosya yolu
        band_idx: Bant indeksleri [R, G, B, DSM, DTM]
        use_cache: Cache kullan
        cache_path: Cache dosya yolu
        recalculate: Cache'i yoksay, yeniden hesapla
        rvt_radii: RVT yarıçapları (metre)
        gaussian_lrm_sigma: LRM için Gaussian sigma
        enable_curvature: Curvature kanallarını hesapla
        enable_tpi: TPI kanalını hesapla
        tpi_radii: TPI yarıçapları (piksel)
    
    Returns:
        PrecomputedDerivatives veya None
    """
    stage_pbar: Optional[tqdm] = None
    if use_cache and cache_path is not None:
        # High-level cache progress (in addition to per-band progress bars).
        stage_pbar = tqdm(
            total=5,
            desc="Cache",
            unit="adım",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            leave=False,
        )

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)
        
        # Cache kullanılacak mı kontrol et
        if use_cache and cache_path and not recalculate:
            cache_result = load_derivatives_cache(cache_path)
            if cache_result:
                derivatives_data, metadata = cache_result
                # Cache geçerli mi?
                if validate_cache(
                    metadata,
                    input_path,
                    band_idx,
                    rvt_radii=rvt_radii,
                    gaussian_lrm_sigma=gaussian_lrm_sigma,
                    enable_curvature=enable_curvature,
                    enable_tpi=enable_tpi,
                    tpi_radii=tpi_radii,
                ):
                    # Cache'den yükle
                    dsm_cached = derivatives_data.get("dsm")
                    if dsm_cached is not None and getattr(dsm_cached, "size", 0) == 0:
                        dsm_cached = None
                    
                    # Yeni kanalları cache'den al (yoksa None)
                    plan_curv_cached = derivatives_data.get("plan_curv")
                    profile_curv_cached = derivatives_data.get("profile_curv")
                    tpi_cached = derivatives_data.get("tpi")
                    
                    # Eski cache formatından gelen boş array'leri None yap
                    if plan_curv_cached is not None and getattr(plan_curv_cached, "size", 0) == 0:
                        plan_curv_cached = None
                    if profile_curv_cached is not None and getattr(profile_curv_cached, "size", 0) == 0:
                        profile_curv_cached = None
                    if tpi_cached is not None and getattr(tpi_cached, "size", 0) == 0:
                        tpi_cached = None
                    
                    # Eğer yeni kanallar isteniyor ama cache'de yoksa, yeniden hesapla
                    need_recalc = False
                    if enable_curvature and (plan_curv_cached is None or profile_curv_cached is None):
                        LOGGER.info("Cache'de Curvature yok, yeniden hesaplanacak")
                        need_recalc = True
                    if enable_tpi and tpi_cached is None:
                        LOGGER.info("Cache'de TPI yok, yeniden hesaplanacak")
                        need_recalc = True
                    
                    if not need_recalc:
                        if stage_pbar is not None:
                            stage_pbar.update(stage_pbar.total - stage_pbar.n)
                            stage_pbar.close()
                        return PrecomputedDerivatives(
                            rgb=derivatives_data["rgb"],
                            dsm=dsm_cached,
                            dtm=derivatives_data["dtm"],
                            svf=derivatives_data["svf"],
                            pos_open=derivatives_data["pos_open"],
                            neg_open=derivatives_data["neg_open"],
                            lrm=derivatives_data["lrm"],
                            slope=derivatives_data["slope"],
                            ndsm=derivatives_data["ndsm"],
                            plan_curv=plan_curv_cached,
                            profile_curv=profile_curv_cached,
                            tpi=tpi_cached,
                            transform=transform,
                            crs=crs,
                            pixel_size=pixel_size,
                        )
                else:
                    LOGGER.info("Cache geçersiz, yeniden hesaplanacak")
        
        # Cache kullanılmıyor veya geçersiz - yeniden hesapla
        LOGGER.info(f"RVT türevleri hesaplanıyor: {width}x{height} piksel")
        LOGGER.info("Bu işlem birkaç dakika sürebilir...")
        
        # Tüm raster'ı oku
        LOGGER.info("Raster bandları okunuyor...")
        def read_band(idx: int) -> Optional[np.ndarray]:
            if idx <= 0:
                return None
            data = src.read(idx, masked=True)
            return np.ma.filled(data.astype(np.float32), np.nan)
        
        # Progress bar ile bandları oku
        bands_to_read = [("RGB-R", band_idx[0]), ("RGB-G", band_idx[1]), ("RGB-B", band_idx[2]), 
                        ("DSM", band_idx[3]), ("DTM", band_idx[4])]
        band_data = {}
        for band_name, band_id in progress_bar(
            bands_to_read,
            desc="Band okuma",
            unit="band",
            total=len(bands_to_read)
        ):
            if band_id > 0:
                band_data[band_name] = read_band(band_id)
        if stage_pbar is not None:
            stage_pbar.update(1)
        
        rgb = np.stack([band_data["RGB-R"], band_data["RGB-G"], band_data["RGB-B"]], axis=0)
        dsm = band_data.get("DSM")
        dtm = band_data["DTM"]
        
        if dtm is None:
            raise ValueError("DTM band gerekli")
        
        # nDSM hesapla
        LOGGER.info("nDSM hesaplanıyor...")
        ndsm = compute_ndsm(dsm, dtm)
        if stage_pbar is not None:
            stage_pbar.update(1)
        
        # RVT türevlerini hesapla
        LOGGER.info("RVT türevleri hesaplanıyor (SVF, openness, LRM, slope)...")
        svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
            dtm, pixel_size=pixel_size, radii=rvt_radii, gaussian_lrm_sigma=gaussian_lrm_sigma
        )
        if stage_pbar is not None:
            stage_pbar.update(1)
        
        # Gelişmiş topografik özellikler
        plan_curv = None
        profile_curv = None
        tpi = None
        
        if enable_curvature:
            LOGGER.info("Curvature kanalları hesaplanıyor (Plan + Profile)...")
            plan_curv, profile_curv = compute_curvatures(dtm, pixel_size=pixel_size)
            LOGGER.info("  → Plan Curvature: hendek/sırt ayrımı için")
            LOGGER.info("  → Profile Curvature: teras/basamak tespiti için")
        
        if enable_tpi:
            LOGGER.info(f"TPI (Topographic Position Index) hesaplanıyor (yarıçaplar: {tpi_radii})...")
            tpi = compute_tpi_multiscale(dtm, radii=tpi_radii)
            LOGGER.info("  → TPI: höyük/çukur tespiti için")

        if stage_pbar is not None:
            stage_pbar.update(1)
        
        derivatives = PrecomputedDerivatives(
            rgb=rgb,
            dsm=dsm,
            dtm=dtm,
            svf=svf,
            pos_open=pos_open,
            neg_open=neg_open,
            lrm=lrm,
            slope=slope,
            ndsm=ndsm,
            plan_curv=plan_curv,
            profile_curv=profile_curv,
            tpi=tpi,
            transform=transform,
            crs=crs,
            pixel_size=pixel_size,
        )
        
        # Cache'e kaydet (istenirse)
        if use_cache and cache_path:
            derivatives_data = {
                "rgb": rgb,
                "dsm": dsm if dsm is not None else np.array([]),  # None yerine boş array
                "dtm": dtm,
                "svf": svf,
                "pos_open": pos_open,
                "neg_open": neg_open,
                "lrm": lrm,
                "slope": slope,
                "ndsm": ndsm,
                # Yeni kanallar
                "plan_curv": plan_curv if plan_curv is not None else np.array([]),
                "profile_curv": profile_curv if profile_curv is not None else np.array([]),
                "tpi": tpi if tpi is not None else np.array([]),
            }
            metadata = {
                "input_path": str(input_path),
                "input_mtime": input_path.stat().st_mtime,
                "input_size": input_path.stat().st_size,
                "bands": list(band_idx),
                "shape": (height, width),
                "pixel_size": pixel_size,
                "rvt_radii": list(DEFAULTS.rvt_radii if rvt_radii is None else rvt_radii),
                "gaussian_lrm_sigma": float(
                    DEFAULTS.gaussian_lrm_sigma if gaussian_lrm_sigma is None else gaussian_lrm_sigma
                ),
                "enable_curvature": enable_curvature,
                "enable_tpi": enable_tpi,
                "tpi_radii": list(tpi_radii) if enable_tpi else [],
                # NOT: tile ve overlap cache'e KAYDEDİLMEZ!
                # RVT türevleri tüm raster için hesaplanır, tile/overlap
                # sadece model inference sırasında kullanılır.
            }
            save_derivatives_cache(cache_path, derivatives_data, metadata)
            if stage_pbar is not None:
                stage_pbar.update(1)
                stage_pbar.close()
                stage_pbar = None
        
        LOGGER.info("RVT türevleri hazır ✓")
        return derivatives


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Arkeolojik alan tespiti için önceden eğitilmiş U-Net modeli.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="YAML konfigürasyon dosyası yolu. Tüm ayarları buradan kontrol edebilirsiniz.",
    )
    parser.add_argument(
        "--input",
        default=default_for("input"),
        help=cli_help("input", "(PipelineDefaults.input veya config.yaml'dan değiştirin)."),
    )
    parser.add_argument(
        "--weights",
        default=default_for("weights"),
        help=cli_help("weights"),
    )
    parser.add_argument(
        "--bands",
        default=default_for("bands"),
        help=cli_help("bands"),
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=default_for("tile"),
        help=cli_help("tile"),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=default_for("overlap"),
        help=cli_help("overlap"),
    )
    parser.add_argument(
        "--th",
        type=float,
        default=default_for("th"),
        help=cli_help("th"),
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=default_for("half"),
        help=cli_help("half", "(use --no-half to force float32 even when CUDA is present)."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_for("device"),
        help=cli_help("device", "Örnek: 'cuda', 'cpu', 'cuda:0'. None ise otomatik seçilir."),
    )
    parser.add_argument(
        "--global-norm",
        action=argparse.BooleanOptionalAction,
        default=default_for("global_norm"),
        help=cli_help("global_norm", "(toggle with --no-global-norm for per-tile scaling)."),
    )
    parser.add_argument(
        "--norm-sample-tiles",
        type=int,
        default=default_for("norm_sample_tiles"),
        help=cli_help("norm_sample_tiles"),
    )
    parser.add_argument(
        "--feather",
        action=argparse.BooleanOptionalAction,
        default=default_for("feather"),
        help=cli_help("feather", "(switch off with --no-feather for raw tile edges)."),
    )
    parser.add_argument(
        "--enable-deep-learning",
        action=argparse.BooleanOptionalAction,
        default=default_for("enable_deep_learning"),
        dest="enable_deep_learning",
        help=cli_help("enable_deep_learning", "(--no-enable-deep-learning ile sadece klasik yöntemleri çalıştırabilirsiniz)."),
    )
    parser.add_argument(
        "--enable-classic",
        action=argparse.BooleanOptionalAction,
        default=default_for("enable_classic"),
        dest="enable_classic",
        help=cli_help("enable_classic", "(--no-enable-classic ile sadece DL çalıştırabilirsiniz)."),
    )
    parser.add_argument(
        "--enable-fusion",
        action=argparse.BooleanOptionalAction,
        default=default_for("enable_fusion"),
        dest="enable_fusion",
        help=cli_help("enable_fusion", "(--no-enable-fusion ile DL ve klasik sonuçları ayrı tutabilirsiniz)."),
    )
    parser.add_argument(
        "--classic-modes",
        default=default_for("classic_modes"),
        help=cli_help("classic_modes"),
    )
    parser.add_argument(
        "--classic-save-intermediate",
        action=argparse.BooleanOptionalAction,
        default=default_for("classic_save_intermediate"),
        help=cli_help("classic_save_intermediate", "(use --no-classic-save-intermediate to skip per-mode rasters)."),
    )
    parser.add_argument(
        "--classic-th",
        type=float,
        default=default_for("classic_th"),
        help=cli_help("classic_th"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_for("alpha"),
        help=cli_help("alpha"),
    )
    parser.add_argument(
        "--fuse-encoders",
        default=default_for("fuse_encoders"),
        help=cli_help("fuse_encoders", "('all' or CSV of encoder suffixes)")
    )
    parser.add_argument(
        "--mask-talls",
        type=float,
        default=default_for("mask_talls"),
        help=cli_help("mask_talls"),
    )
    parser.add_argument(
        "--rgb-only",
        action=argparse.BooleanOptionalAction,
        default=default_for("rgb_only"),
        dest="rgb_only",
        help=cli_help("rgb_only", "Zero out SVF/Openness/LRM/Slope/nDSM so the model effectively runs on RGB only."),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=default_for("min_area"),
        help=cli_help("min_area"),
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=default_for("simplify"),
        help=cli_help("simplify"),
    )
    parser.add_argument(
        "--vectorize",
        action=argparse.BooleanOptionalAction,
        default=default_for("vectorize"),
        help=cli_help("vectorize", "(set --no-vectorize to keep only raster outputs)."),
    )
    parser.add_argument(
        "--arch",
        default=default_for("arch"),
        help=cli_help("arch"),
    )
    parser.add_argument(
        "--encoder",
        default=default_for("encoder"),
        help=cli_help("encoder"),
    )
    parser.add_argument(
        "--encoders",
        default=default_for("encoders"),
        help=cli_help(
            "encoders",
            "(examples: 'resnet34,resnet50', 'all', 'none').",
        ),
    )
    parser.add_argument(
        "--weights-template",
        default=default_for("weights_template"),
        help=cli_help(
            "weights_template",
            "(uses {encoder} placeholder; ignored if file missing).",
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=default_for("out_prefix"),
        help=cli_help("out_prefix"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_for("seed"),
        help=cli_help("seed"),
    )
    parser.add_argument(
        "--zero-shot-imagenet",
        action=argparse.BooleanOptionalAction,
        default=default_for("zero_shot_imagenet"),
        help=cli_help("zero_shot_imagenet", "(toggle with --no-zero-shot-imagenet when supplying weights)."),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=default_for("verbose"),
        help=cli_help("verbose"),
    )
    parser.add_argument(
        "--cache-derivatives",
        action=argparse.BooleanOptionalAction,
        default=default_for("cache_derivatives"),
        dest="cache_derivatives",
        help=cli_help("cache_derivatives", "(tekrar çalıştırmalarda çok hızlı!)"),
    )
    parser.add_argument(
        "--cache-derivatives-mode",
        type=str,
        default=default_for("cache_derivatives_mode"),
        dest="cache_derivatives_mode",
        help=cli_help("cache_derivatives_mode"),
    )
    parser.add_argument(
        "--deriv-cache-chunk",
        type=int,
        default=default_for("deriv_cache_chunk"),
        dest="deriv_cache_chunk",
        help=cli_help("deriv_cache_chunk"),
    )
    parser.add_argument(
        "--deriv-cache-halo",
        type=int,
        default=default_for("deriv_cache_halo"),
        dest="deriv_cache_halo",
        help=cli_help("deriv_cache_halo"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=default_for("cache_dir"),
        dest="cache_dir",
        help=cli_help("cache_dir"),
    )
    parser.add_argument(
        "--recalculate-cache",
        action=argparse.BooleanOptionalAction,
        default=default_for("recalculate_cache"),
        dest="recalculate_cache",
        help=cli_help("recalculate_cache"),
    )
    parser.add_argument(
        "--enable-yolo",
        action=argparse.BooleanOptionalAction,
        default=default_for("enable_yolo"),
        dest="enable_yolo",
        help=cli_help("enable_yolo", "(YOLO11 nesne tespit/segmentasyon modelini etkinleştir)"),
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=default_for("yolo_weights"),
        dest="yolo_weights",
        help=cli_help("yolo_weights"),
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=default_for("yolo_conf"),
        dest="yolo_conf",
        help=cli_help("yolo_conf"),
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=default_for("yolo_iou"),
        dest="yolo_iou",
        help=cli_help("yolo_iou"),
    )
    parser.add_argument(
        "--yolo-tile",
        type=int,
        default=default_for("yolo_tile"),
        dest="yolo_tile",
        help=cli_help("yolo_tile"),
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=default_for("yolo_imgsz"),
        dest="yolo_imgsz",
        help=cli_help("yolo_imgsz"),
    )
    parser.add_argument(
        "--yolo-device",
        type=str,
        default=default_for("yolo_device"),
        dest="yolo_device",
        help=cli_help("yolo_device"),
    )

    args = parser.parse_args(argv)
    config = build_config_from_args(args)

    ENCODER_ALIASES = {
        "efficientnet-b3": "timm-efficientnet-b3",
        "effnet-b3": "timm-efficientnet-b3",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
    }

    def normalize_encoder(name: str) -> Tuple[str, str]:
        key = (name or "").strip().lower()
        smp_name = ENCODER_ALIASES.get(key, key)
        suffix = "efficientnet-b3" if smp_name == "timm-efficientnet-b3" else smp_name
        return smp_name, suffix

    def available_encoders_list() -> List[str]:
        return ["resnet34", "resnet50", "efficientnet-b3"]

    if config.classic_th is not None and not (0.0 <= config.classic_th <= 1.0):
        parser.error("--classic-th must be between 0 and 1.")
    if not (0.0 <= config.alpha <= 1.0):
        parser.error("--alpha must be between 0 and 1.")

    configure_logging(config.verbose)
    install_gdal_warning_filter()
    set_random_seeds(config.seed)

    if config.rgb_only:
        LOGGER.info("RGB-only mode active: derivative channels are zero-filled; effective input is RGB.")

    input_path = Path(config.input)
    if not input_path.exists():
        parser.error(f"Input raster not found: {input_path}")

    enc_mode = (config.encoders or "").strip().lower()
    ran_multi = enc_mode not in ("", "none")
    if enc_mode in ("", "none"):
        if not config.zero_shot_imagenet and not config.weights:
            parser.error("Either provide --weights or use --zero-shot-imagenet for zero-shot inference.")

    weights_path: Optional[Path] = None
    if config.weights:
        weights_path = Path(config.weights)
        if not weights_path.exists():
            parser.error(f"Weights file not found: {weights_path}")

    bands = parse_band_indexes(config.bands)
    out_prefix = resolve_out_prefix(input_path, config.out_prefix)

    # Heuristic: treat very large rasters differently to avoid OOM in downstream steps (fusion/vectorization).
    raster_pixels = 0
    raster_height = 0
    raster_width = 0
    try:
        with rasterio.open(input_path) as _src:
            raster_height = int(_src.height)
            raster_width = int(_src.width)
            raster_pixels = raster_height * raster_width
    except Exception as e:
        LOGGER.warning("Could not read raster size for large-raster heuristics: %s", e)
    one_float_bytes = raster_pixels * np.dtype(np.float32).itemsize
    is_large_raster = raster_pixels >= 200_000_000 or one_float_bytes >= int(1.5 * 1024**3)
    if is_large_raster and config.vectorize:
        LOGGER.warning(
            "Vectorization disabled for very large rasters (%dx%d). Run on a smaller subset or vectorize in GIS.",
            raster_width,
            raster_height,
        )
        config.vectorize = False

    # Yöntem etkinleştirme kontrolleri
    if not config.enable_deep_learning and not config.enable_classic and not config.enable_yolo:
        parser.error("En az bir yöntem etkin olmalı (deep learning, classic veya YOLO11).")
    
    classic_outputs: Optional[ClassicOutputs] = None
    yolo_outputs: Optional[YoloOutputs] = None
    fusion_outputs: Optional[FusionOutputs] = None
    fusion_encoder_label: Optional[str] = None
    dl_runs: List[Tuple[str, Path]] = []
    multi_fused_results: List[Tuple[str, FusionOutputs]] = []
    resolved_classic_modes: Optional[Tuple[str, ...]] = None
    if config.enable_classic:
        raw_modes = [mode.strip() for mode in config.classic_modes.split(",") if mode.strip()]
        if not raw_modes:
            parser.error("--classic-modes en az bir mod içermelidir.")
        valid_modes = {"rvtlog", "hessian", "morph", "combo"}
        invalid = [m for m in raw_modes if m.lower() not in valid_modes]
        if invalid:
            parser.error(f"Desteklenmeyen klasik mod(lar): {', '.join(invalid)}")
        if len(raw_modes) == 1 and raw_modes[0].lower() == "combo":
            resolved_classic_modes = ("rvtlog", "hessian", "morph")
        elif any(m.lower() == "combo" for m in raw_modes):
            parser.error("'combo' diğer klasik modlarla birleştirilemez.")
        else:
            resolved_classic_modes = tuple(m.lower() for m in raw_modes)

    # Cihaz seçimi: CLI/config'den alınır veya otomatik belirlenir
    if config.device:
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA istendi ama kullanılamıyor; CPU'ya geçiliyor.")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Kullanılan cihaz: %s", device)
    if config.half and device.type != "cuda":
        LOGGER.warning("--half istendi ama CUDA kullanılamıyor; float32 ile çalışılacak.")

    # Cache yönetimi - RVT türevlerini bir kez hesaplayıp tekrar kullan
    precomputed_deriv: Optional[PrecomputedDerivatives] = None
    derivative_cache_tif: Optional[Path] = None
    derivative_cache_meta: Optional[Path] = None

    cache_mode = str(getattr(config, "cache_derivatives_mode", "auto")).strip().lower()
    if cache_mode not in ("auto", "npz", "raster"):
        LOGGER.warning("cache_derivatives_mode geçersiz (%s); 'auto' kullanılacak.", cache_mode)
        cache_mode = "auto"

    cache_precompute_ok = True
    cache_precompute_msg = ""
    if (config.enable_deep_learning or config.enable_classic) and config.cache_derivatives and cache_mode in ("auto", "npz"):
        cache_precompute_ok, cache_precompute_msg = full_raster_cache_precompute_ok(
            input_path,
            bands,
            enable_curvature=config.enable_curvature,
            enable_tpi=config.enable_tpi,
        )
        if not cache_precompute_ok and cache_precompute_msg:
            LOGGER.warning(cache_precompute_msg)

    if (config.enable_deep_learning or config.enable_classic) and config.cache_derivatives:
        if cache_mode in ("auto", "npz") and cache_precompute_ok:
            # Full-raster NPZ cache (küçük/orta rasterlar için en hızlı okuma)
            LOGGER.info("=" * 70)
            LOGGER.info("CACHE MODU (npz): RVT türevleri bir kez hesaplanacak ve diskten okunacak")
            LOGGER.info("=" * 70)
            cache_path = get_cache_path(input_path, config.cache_dir)
            precomputed_deriv = precompute_derivatives(
                input_path=input_path,
                band_idx=bands,
                use_cache=True,
                cache_path=cache_path,
                recalculate=config.recalculate_cache,
                rvt_radii=config.rvt_radii,
                gaussian_lrm_sigma=config.gaussian_lrm_sigma,
                enable_curvature=config.enable_curvature,
                enable_tpi=config.enable_tpi,
                tpi_radii=config.tpi_radii,
            )
            if precomputed_deriv:
                num_ch = get_num_channels(config.enable_curvature, config.enable_tpi)
                LOGGER.info(f"✓ Türevler hazır ({num_ch} kanal) - Her encoder çok daha hızlı çalışacak!")
            else:
                LOGGER.warning("RVT türevleri hesaplanamadı, normal moda devam ediliyor")
        else:
            # Block-based raster cache (büyük rasterlar için bellek güvenli)
            if cache_mode == "npz" and not cache_precompute_ok:
                LOGGER.warning("Tam-raster NPZ cache bu veri için uygun değil; raster-cache kullanılacak.")

            LOGGER.info("=" * 70)
            LOGGER.info("CACHE MODU (raster): RVT türevleri blok blok diske yazılacak ve tekrar çalıştırmalarda buradan okunacak")
            LOGGER.info("=" * 70)

            derivative_cache_tif, derivative_cache_meta = get_derivative_raster_cache_paths(input_path, config.cache_dir)
            raster_info = None
            if not config.recalculate_cache:
                raster_info = load_derivative_raster_cache_info(
                    derivative_cache_tif,
                    derivative_cache_meta,
                    input_path,
                    bands,
                    rvt_radii=config.rvt_radii,
                    gaussian_lrm_sigma=config.gaussian_lrm_sigma,
                    enable_curvature=config.enable_curvature,
                    enable_tpi=config.enable_tpi,
                    tpi_radii=config.tpi_radii,
                )

            if raster_info is None:
                chunk_size = int(getattr(config, "deriv_cache_chunk", 2048))
                halo_px = getattr(config, "deriv_cache_halo", None)
                if halo_px is not None:
                    halo_px = int(halo_px)
                build_derivative_raster_cache(
                    input_path=input_path,
                    band_idx=bands,
                    cache_tif_path=derivative_cache_tif,
                    cache_meta_path=derivative_cache_meta,
                    recalculate=config.recalculate_cache,
                    rvt_radii=config.rvt_radii,
                    gaussian_lrm_sigma=config.gaussian_lrm_sigma,
                    enable_curvature=config.enable_curvature,
                    enable_tpi=config.enable_tpi,
                    tpi_radii=config.tpi_radii,
                    chunk_size=chunk_size,
                    halo_px=halo_px,
                )
                raster_info = load_derivative_raster_cache_info(
                    derivative_cache_tif,
                    derivative_cache_meta,
                    input_path,
                    bands,
                    rvt_radii=config.rvt_radii,
                    gaussian_lrm_sigma=config.gaussian_lrm_sigma,
                    enable_curvature=config.enable_curvature,
                    enable_tpi=config.enable_tpi,
                    tpi_radii=config.tpi_radii,
                )

            if raster_info is None:
                LOGGER.warning("Raster-cache doğrulanamadı; normal moda devam ediliyor")
                derivative_cache_tif = None
                derivative_cache_meta = None
            else:
                deriv_band_count = len(dict(raster_info.get("band_map", {})))
                model_ch = get_num_channels(config.enable_curvature, config.enable_tpi)
                LOGGER.info("✓ Raster-cache hazır (deriv=%d band, model=%d kanal): %s", deriv_band_count, model_ch, derivative_cache_tif)

    # Derin öğrenme etkinse modelleri çalıştır
    if config.enable_deep_learning and ran_multi:
        enc_list = (
            available_encoders_list()
            if enc_mode == "all"
            else [enc.strip() for enc in config.encoders.split(",") if enc.strip()]
        )
        
        # Cache kullanılıyorsa ilerleme mesajı
        if precomputed_deriv is not None or derivative_cache_tif is not None:
            LOGGER.info(f"Toplam {len(enc_list)} encoder çalıştırılacak (RVT türevleri cache'ten kullanılacak)")
        else:
            LOGGER.info(f"Toplam {len(enc_list)} encoder çalıştırılacak (her biri için RVT hesaplanacak)")
        
        for idx, enc in enumerate(enc_list, 1):
            LOGGER.info("=" * 70)
            LOGGER.info(f"ENCODER {idx}/{len(enc_list)}: {enc.upper()}")
            LOGGER.info("=" * 70)
            
            smp_name, suffix = normalize_encoder(enc)
            try:
                _ = smp.encoders.get_encoder(smp_name, in_channels=3)
            except Exception:
                avail = ", ".join(available_encoders_list())
                LOGGER.error("Unknown encoder '%s'. Supported: %s", enc, avail)
                continue

            per_weights: Optional[Path] = None
            if config.weights_template:
                cand = Path(config.weights_template.format(encoder=suffix))
                if cand.exists():
                    per_weights = cand
                else:
                    LOGGER.info("[%s] No weights at %s; falling back to zero-shot.", suffix, cand)

            # Kanal sayısını hesapla (9, 10, 11 veya 12)
            num_channels = get_num_channels(
                enable_curvature=config.enable_curvature,
                enable_tpi=config.enable_tpi
            )
            
            if per_weights is not None:
                LOGGER.info("[%s] Eğitilmiş ağırlıklar yükleniyor: %s", suffix, per_weights)
                model = build_model(
                    arch=config.arch, 
                    encoder=smp_name, 
                    in_ch=num_channels,
                    enable_attention=config.enable_attention,
                    attention_reduction=config.attention_reduction,
                )
                load_weights(model, per_weights, map_location=device)
            else:
                LOGGER.info("[%s] Zero-shot modunda başlatılıyor (ImageNet 3->%d genişletme)", suffix, num_channels)
                model = build_model_with_imagenet_inflated(
                    arch=config.arch, 
                    encoder=smp_name, 
                    in_ch=num_channels,
                    enable_attention=config.enable_attention,
                    attention_reduction=config.attention_reduction,
                )

            enc_prefix = out_prefix.with_suffix("")
            enc_prefix = enc_prefix.parent / f"{enc_prefix.name}_{suffix}"

            outputs = infer_tiled(
                model=model,
                input_path=input_path,
                band_idx=bands,
                tile=config.tile,
                overlap=config.overlap,
                device=device,
                use_half=config.half,
                threshold=config.th,
                mask_talls=config.mask_talls,
                out_prefix=enc_prefix,
                global_norm=config.global_norm,
                norm_sample_tiles=config.norm_sample_tiles,
                feather=config.feather,
                precomputed_deriv=precomputed_deriv,  # NPZ cache (uygunsa)
                derivative_cache_tif=derivative_cache_tif,  # Raster cache (büyük rasterlarda)
                derivative_cache_meta=derivative_cache_meta,
                enable_curvature=config.enable_curvature,
                enable_tpi=config.enable_tpi,
                # encoder adı base prefix'te zaten var; tekrar eklemeyelim
                encoder=None,
                min_area=config.min_area,
                percentile_low=config.percentile_low,
                percentile_high=config.percentile_high,
                rvt_radii=config.rvt_radii,
                gaussian_lrm_sigma=config.gaussian_lrm_sigma,
                rgb_only=config.rgb_only,
                tpi_radii=config.tpi_radii,
            )
            LOGGER.info("[%s] ✓ Olasılık haritası: %s", suffix, outputs.prob_path)
            LOGGER.info("[%s] ✓ İkili maske: %s", suffix, outputs.mask_path)

            # Store last encoder label for fusion filenames
            fusion_encoder_label = suffix
            dl_runs.append((suffix, outputs.prob_path))
            if config.vectorize:
                LOGGER.info("[%s] Vektörleştirme başlatılıyor...", suffix)
                vector_base = outputs.mask_path.with_suffix("")
                gpkg = vectorize_predictions(
                    mask=outputs.mask,
                    prob_map=outputs.prob_map,
                    transform=outputs.transform,
                    crs=outputs.crs,
                    out_path=vector_base,
                    min_area=config.min_area,
                    simplify_tol=config.simplify,
                    opening_size=config.vector_opening_size,
                    label_connectivity=config.label_connectivity,
                )
                if gpkg:
                    LOGGER.info("[%s] ✓ Vektör dosyası: %s", suffix, gpkg)
        
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("TÜM ENCODERLAR TAMAMLANDI!")
        LOGGER.info("=" * 70)
        
        # Multi-encoder loop tamamlandı
        # Klasik yöntemler kapalıysa burada çık, açıksa devam et
        if not config.enable_classic:
            LOGGER.info("Klasik yöntemler kapalı, işlem tamamlandı.")
            return
    
    # Tek model çalıştır (sadece encoders boş/none ise)
    # Kanal sayısını hesapla (9, 10, 11 veya 12)
    num_channels = get_num_channels(
        enable_curvature=config.enable_curvature,
        enable_tpi=config.enable_tpi
    )
    
    if enc_mode in ("", "none") and config.enable_deep_learning and config.zero_shot_imagenet:
        LOGGER.info(f"Zero-shot mode: using ImageNet-pretrained encoder inflated to {num_channels} channels.")
        if config.enable_attention:
            LOGGER.info("CBAM Attention modülü aktif.")
        model = build_model_with_imagenet_inflated(
            arch=config.arch, 
            encoder=config.encoder, 
            in_ch=num_channels,
            enable_attention=config.enable_attention,
            attention_reduction=config.attention_reduction,
        )
    elif enc_mode in ("", "none") and config.enable_deep_learning:
        model = build_model(
            arch=config.arch, 
            encoder=config.encoder, 
            in_ch=num_channels,
            enable_attention=config.enable_attention,
            attention_reduction=config.attention_reduction,
        )
        # weights_path is guaranteed to be set in this branch
        if weights_path is not None:
            load_weights(model, weights_path, map_location=device)
    else:
        model = None

    if bands[3] <= 0:
        LOGGER.warning(
            "DSM band not provided; nDSM channel will be zero and tall-object masking will be disabled."
        )

    # Derin öğrenme çıkarımını çalıştır (etkinse)
    outputs = None
    if enc_mode in ("", "none") and config.enable_deep_learning and model is not None:
        outputs = infer_tiled(
            model=model,
            input_path=input_path,
            band_idx=bands,
            tile=config.tile,
            overlap=config.overlap,
            device=device,
            use_half=config.half,
            threshold=config.th,
            mask_talls=config.mask_talls,
            out_prefix=out_prefix,
            global_norm=config.global_norm,
            norm_sample_tiles=config.norm_sample_tiles,
            feather=config.feather,
            precomputed_deriv=precomputed_deriv,
            derivative_cache_tif=derivative_cache_tif,
            derivative_cache_meta=derivative_cache_meta,
            enable_curvature=config.enable_curvature,
            enable_tpi=config.enable_tpi,
            encoder=config.encoder,
            min_area=config.min_area,
            percentile_low=config.percentile_low,
            percentile_high=config.percentile_high,
            rvt_radii=config.rvt_radii,
            gaussian_lrm_sigma=config.gaussian_lrm_sigma,
            rgb_only=config.rgb_only,
            tpi_radii=config.tpi_radii,
        )
        # Record encoder for fusion filenames (single-encoder path)
        fusion_encoder_label = config.encoder

        LOGGER.info("Olasılık haritası yazıldı: %s", outputs.prob_path)
        LOGGER.info("İkili maske yazıldı: %s", outputs.mask_path)

    fuse_enabled = config.enable_classic and config.enable_fusion and config.enable_deep_learning
    if config.enable_fusion and not (config.enable_classic and config.enable_deep_learning):
        LOGGER.warning("Fusion için hem deep learning hem de classic yöntemlerin etkin olması gerekir.")

    # Klasik pipeline'ı çalıştır (etkinse)
    if config.enable_classic and resolved_classic_modes is not None:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("KLASİK YÖNTEMLER BAŞLATILIYOR")
        LOGGER.info(f"Modlar: {', '.join(resolved_classic_modes)}")
        LOGGER.info("=" * 70)
        
        classic_outputs = infer_classic_tiled(
            input_path=input_path,
            band_idx=bands,
            tile=config.tile,
            overlap=config.overlap,
            out_prefix=out_prefix,
            classic_th=config.classic_th,
            modes=resolved_classic_modes,
            feather=config.feather,
            save_intermediate=config.classic_save_intermediate,
            min_area=config.min_area,
            precomputed_deriv=precomputed_deriv,
            derivative_cache_tif=derivative_cache_tif,
            derivative_cache_meta=derivative_cache_meta,
            enable_curvature=config.enable_curvature,
            enable_tpi=config.enable_tpi,
            tpi_radii=config.tpi_radii,
            sigma_scales=config.sigma_scales,
            morphology_radii=config.morphology_radii,
            rvt_radii=config.rvt_radii,
            gaussian_gradient_sigma=config.gaussian_gradient_sigma,
            local_variance_window=config.local_variance_window,
            gaussian_lrm_sigma=config.gaussian_lrm_sigma,
        )
        LOGGER.info("✓ Klasik yöntem birleşik çıktı: %s, %s", classic_outputs.prob_path, classic_outputs.mask_path)
        if classic_outputs.per_mode:
            for mode_name, mode_out in classic_outputs.per_mode.items():
                LOGGER.info(
                    "  ✓ Klasik mod '%s': %s, %s",
                    mode_name,
                    mode_out.prob_path,
                    mode_out.mask_path,
                )

    # YOLO11 pipeline'ı çalıştır (etkinse)
    if config.enable_yolo:
        if YOLO is None:
            LOGGER.error("YOLO11 etkinleştirildi ama ultralytics yüklü değil. Yüklemek için: pip install ultralytics>=8.1.0")
        else:
            LOGGER.info("")
            LOGGER.info("=" * 70)
            LOGGER.info("YOLO11 BAŞLATILIYOR")
            LOGGER.info("=" * 70)
            
            yolo_tile_size = config.yolo_tile if config.yolo_tile is not None else config.tile
            
            yolo_outputs = infer_yolo_tiled(
                input_path=input_path,
                band_idx=bands,
                tile=yolo_tile_size,
                overlap=config.overlap,
                out_prefix=out_prefix,
                yolo_weights=config.yolo_weights,
                conf_threshold=config.yolo_conf,
                iou_threshold=config.yolo_iou,
                imgsz=config.yolo_imgsz,
                device=config.yolo_device,
                min_area=config.min_area,
                precomputed_deriv=precomputed_deriv,
                percentile_low=config.percentile_low,
                percentile_high=config.percentile_high,
                save_labels=True,  # Etiketli tespitleri GeoPackage olarak kaydet
            )
            
            LOGGER.info("✓ YOLO11 olasılık haritası: %s", yolo_outputs.prob_path)
            LOGGER.info("✓ YOLO11 ikili maske: %s", yolo_outputs.mask_path)

    # Fusion (birleştirme) çalıştır (etkinse ve her iki yöntem de çalıştıysa)
    if fuse_enabled and classic_outputs is not None:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info(f"FUSION (BİRLEŞTİRME) BAŞLATILIYOR (alpha={config.alpha})")
        LOGGER.info("=" * 70)
        alpha = float(config.alpha)
        fuse_threshold = config.th if config.th is not None else 0.5
        
        # Per-encoder fusion (multi-encoder runs)
        if ran_multi and dl_runs:
            fuse_filter_raw = (config.fuse_encoders or "all").strip().lower()
            fuse_filter = None if fuse_filter_raw == "all" else {s.strip().lower() for s in fuse_filter_raw.split(",") if s.strip()}
            base_prefix = out_prefix.with_suffix("")
            base_prefix.parent.mkdir(parents=True, exist_ok=True)
            for enc_suffix, dl_prob_path in dl_runs:
                if fuse_filter is not None and enc_suffix.lower() not in fuse_filter:
                    continue
                _fname = build_filename_with_params(
                    base_name=base_prefix.name,
                    mode_suffix="fused",
                    encoder=enc_suffix,
                    threshold=fuse_threshold,
                    tile=config.tile,
                    alpha=alpha,
                    min_area=config.min_area,
                )
                _pp = base_prefix.parent / f"{_fname}_prob.tif"
                _mp = base_prefix.parent / f"{_fname}_mask.tif"
                _tr, _crs = write_fused_probability_rasters_from_paths(
                    dl_prob_path=dl_prob_path,
                    classic_prob_path=classic_outputs.prob_path,
                    alpha=alpha,
                    threshold=fuse_threshold,
                    out_prob_path=_pp,
                    out_mask_path=_mp,
                )
                multi_fused_results.append(
                    (
                        enc_suffix,
                        FusionOutputs(
                            prob_path=_pp,
                            mask_path=_mp,
                            prob_map=None,
                            mask=None,
                            transform=_tr,
                            crs=_crs,
                            threshold=fuse_threshold,
                        ),
                    )
                )
                LOGGER.info("? Fused (%s): %s, %s", enc_suffix, _pp, _mp)
        
        dl_prob_for_fusion: Optional[Path] = None
        if outputs is not None and getattr(outputs, "prob_path", None) is not None:
            dl_prob_for_fusion = outputs.prob_path
        elif dl_runs:
            # Fallback to last encoder run
            fusion_encoder_label = fusion_encoder_label or dl_runs[-1][0]
            dl_prob_for_fusion = dl_runs[-1][1]

        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Parametreli dosya adı oluştur (fusion için)
        fused_filename = build_filename_with_params(
            base_name=base_prefix.name,
            mode_suffix="fused",
            encoder=fusion_encoder_label,
            threshold=fuse_threshold,
            tile=config.tile,
            alpha=alpha,
            min_area=config.min_area,
        )
        
        fused_prob_path = base_prefix.parent / f"{fused_filename}_prob.tif"
        fused_mask_path = base_prefix.parent / f"{fused_filename}_mask.tif"
        if dl_prob_for_fusion is None:
            LOGGER.warning("Fusion skipped: no DL probability raster available.")
        else:
            _tr, _crs = write_fused_probability_rasters_from_paths(
                dl_prob_path=dl_prob_for_fusion,
                classic_prob_path=classic_outputs.prob_path,
                alpha=alpha,
                threshold=fuse_threshold,
                out_prob_path=fused_prob_path,
                out_mask_path=fused_mask_path,
            )
            fusion_outputs = FusionOutputs(
                prob_path=fused_prob_path,
                mask_path=fused_mask_path,
                prob_map=None,
                mask=None,
                transform=_tr,
                crs=_crs,
                threshold=fuse_threshold,
            )
            LOGGER.info("✓ Birleştirilmiş (fused) çıktılar: %s, %s", fused_prob_path, fused_mask_path)

    # Vektörleştirme (etkinse)
    if config.vectorize:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("VEKTÖRLEŞTİRME BAŞLATILIYOR")
        LOGGER.info("=" * 70)
        vector_jobs: List[
            Tuple[
                str,
                Optional[np.ndarray],
                Optional[np.ndarray],
                Affine,
                Optional[RasterioCRS],
                Path,
                Path,
                Path,
            ]
        ] = []
        
        # Çoklu-encoder modunda DL çıktıları döngü sırasında vektörleştirildi; tekrar etmeyelim
        if outputs is not None and not ran_multi:
            vector_jobs.append((
                "dl",
                outputs.mask,
                outputs.prob_map,
                outputs.transform,
                outputs.crs,
                outputs.mask_path.with_suffix(""),
                outputs.prob_path,
                outputs.mask_path,
            ))
        if yolo_outputs is not None:
            vector_jobs.append((
                "yolo11",
                yolo_outputs.mask,
                yolo_outputs.prob_map,
                yolo_outputs.transform,
                yolo_outputs.crs,
                yolo_outputs.mask_path.with_suffix(""),
                yolo_outputs.prob_path,
                yolo_outputs.mask_path,
            ))
        if classic_outputs is not None:
            vector_jobs.append(
                (
                    "classic",
                    classic_outputs.mask,
                    classic_outputs.prob_map,
                    classic_outputs.transform,
                    classic_outputs.crs,
                    classic_outputs.mask_path.with_suffix(""),
                    classic_outputs.prob_path,
                    classic_outputs.mask_path,
                )
            )
            if config.classic_save_intermediate:
                for mode_name, mode_out in classic_outputs.per_mode.items():
                    vector_jobs.append(
                        (
                            f"classic_{mode_name}",
                            mode_out.mask,
                            mode_out.prob_map,
                            classic_outputs.transform,
                            classic_outputs.crs,
                            mode_out.mask_path.with_suffix(""),
                            mode_out.prob_path,
                            mode_out.mask_path,
                        )
                    )
        if fusion_outputs is not None:
            vector_jobs.append(
                (
                    "fused",
                    fusion_outputs.mask,
                    fusion_outputs.prob_map,
                    fusion_outputs.transform,
                    fusion_outputs.crs,
                    fusion_outputs.mask_path.with_suffix(""),
                    fusion_outputs.prob_path,
                    fusion_outputs.mask_path,
                )
            )
        if multi_fused_results:
            for enc_label, fused in multi_fused_results:
                vector_jobs.append(
                    (
                        f"fused_{enc_label}",
                        fused.mask,
                        fused.prob_map,
                        fused.transform,
                        fused.crs,
                        fused.mask_path.with_suffix(""),
                        fused.prob_path,
                        fused.mask_path,
                    )
                )
        # 'fused_multi' tekrarını engelle: zaten her encoder için 'fused_{enc_label}' eklendi

        for label, mask_arr, prob_arr, transform, crs_obj, out_base, prob_path, mask_path in vector_jobs:
            if mask_arr is None or prob_arr is None:
                if is_large_raster:
                    LOGGER.warning("Skipping vectorization for '%s' (large raster; arrays not kept in memory).", label)
                    continue
                try:
                    with rasterio.open(prob_path) as _src:
                        prob_arr = _src.read(1).astype(np.float32, copy=False)
                    with rasterio.open(mask_path) as _src:
                        mask_arr = _src.read(1).astype(np.uint8, copy=False)
                except Exception as e:
                    LOGGER.warning("Could not load rasters for vectorization (%s): %s", label, e)
                    continue

            LOGGER.info(f"  → {label} poligonlaştırılıyor...")
            vector_file = vectorize_predictions(
                mask=mask_arr,
                prob_map=prob_arr,
                transform=transform,
                crs=crs_obj,
                out_path=out_base,
                min_area=config.min_area,
                simplify_tol=config.simplify,
                opening_size=config.vector_opening_size,
                label_connectivity=config.label_connectivity,
            )
            if vector_file:
                LOGGER.info("    ✓ Vektör çıktısı (%s): %s", label, vector_file)
        
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("✅ TÜM İŞLEMLER TAMAMLANDI!")
        LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()

# ============================================================================
# HIZLI BAŞLANGIÇ
# ============================================================================
#
# EN KOLAY YÖNTEM: Tek config.yaml dosyası! ⭐
# --------------------------------------------
# 1. config.yaml dosyasını düzenleyin:
#    - enable_deep_learning: true/false
#    - enable_classic: true/false
#    - enable_fusion: true/false
#    - th, tile, alpha ve diğer tüm ayarlar
#
# 2. Çalıştırın:
#    python archaeo_detect.py
#
# Bu kadar! Tüm ayarlar ve detaylı açıklamalar config.yaml içinde.
#
#
# KOMUT SATIRI ÖRNEKLERİ (config.yaml'ı override eder):
# -------------------------------------------------------
#
# Temel kullanım (config.yaml'dan okur):
# python archaeo_detect.py
#
# Eşik değerini değiştir:
# python archaeo_detect.py --th 0.7
#
# Karo boyutunu ve overlap'i değiştir:
# python archaeo_detect.py --tile 512 --overlap 128
#
# Fusion karışım oranını değiştir:
# python archaeo_detect.py --alpha 0.7
#
# Eğitilmiş ağırlıklarla çalıştır:
# python archaeo_detect.py --weights unet_resnet34_arch.pth
#
#
# SENARYOLAR (config.yaml içinde):
# ----------------------------------
# Senaryo 1: Sadece DL (U-Net)
#   enable_deep_learning: true, enable_classic: false, enable_yolo: false, enable_fusion: false
#
# Senaryo 2: Sadece Klasik
#   enable_deep_learning: false, enable_classic: true, enable_yolo: false, enable_fusion: false
#
# Senaryo 3: Sadece YOLO11
#   enable_deep_learning: false, enable_classic: false, enable_yolo: true, enable_fusion: false
#
# Senaryo 4: DL + Klasik + Fusion (ÖNERİLİR)
#   enable_deep_learning: true, enable_classic: true, enable_yolo: false, enable_fusion: true
#
# Senaryo 5: Tüm Yöntemler (DL + Klasik + YOLO11)
#   enable_deep_learning: true, enable_classic: true, enable_yolo: true, enable_fusion: true
#
# YOLO11 KULLANIMI:
# -----------------
# YOLO11 için segmentasyon modelini kullanmanız önerilir:
#   yolo_weights: "yolo11n-seg.pt"  # nano segmentation model
#   yolo_weights: "yolo11s-seg.pt"  # small segmentation model
#   yolo_weights: "yolo11m-seg.pt"  # medium segmentation model
#
# Tespit (detection) modeli kullanırsanız, bounding box'lardan maske oluşturulur:
#   yolo_weights: "yolo11n.pt"      # nano detection model
#
# Özel eğitilmiş modelinizi kullanabilirsiniz:
#   yolo_weights: "path/to/your/best.pt"
#
# ============================================================================
