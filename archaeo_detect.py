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
import logging
import math
import os
import sys
from contextlib import nullcontext
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
except ImportError as exc:  # pragma: no cover - dependency required at runtime
    raise ImportError(
        "Install segmentation_models_pytorch: pip install segmentation-models-pytorch"
    ) from exc

try:
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform
except ImportError as exc:  # pragma: no cover - dependency required for vectorisation
    raise ImportError("Install shapely: pip install shapely") from exc

try:
    from pyproj import CRS, Transformer
except ImportError as exc:  # pragma: no cover - dependency required for vectorisation
    raise ImportError("Install pyproj: pip install pyproj") from exc

try:
    from rvt import vis as rvt_vis
except ImportError as exc:  # pragma: no cover - rvt is a hard requirement
    raise ImportError("Install rvt-py: pip install rvt") from exc

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


def install_gdal_warning_filter() -> None:
    """Suppress noisy but harmless GDAL warnings about mis-declared TIFF extrasamples."""
    global _GDAL_HANDLER_INSTALLED
    if _GDAL_HANDLER_INSTALLED or gdal is None:
        return

    def _handler(err_class: int, err_num: int, err_msg: str) -> None:
        if any(err_msg.startswith(prefix) for prefix in _SUPPRESSED_GDAL_MESSAGES):
            LOGGER.debug("Suppressed GDAL warning: %s", err_msg)
            return
        try:
            gdal.CPLDefaultErrorHandler(err_class, err_num, err_msg)
        except AttributeError:  # pragma: no cover - safety for stripped bindings
            LOGGER.warning("GDAL warning: %s", err_msg)

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
        default=r"C:\Users\ertug\Nextcloud\arkeolojik_alan_tespit\kesif_alani.tif",
        metadata={"help": "Çok bantlı GeoTIFF dosyasının tam yolu (RGB + DSM + DTM içermeli)"},
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
    
    # ===== BİRLEŞTİRME (FUSION) AYARLARI =====
    alpha: float = field(
        default=0.5,
        metadata={"help": "Fusion karışım ağırlığı (0-1); 1.0=sadece DL, 0.0=sadece klasik, 0.5=eşit ağırlık"},
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache dosyalarının kaydedileceği dizin; None ise girdi dosyasının yanına yazar"},
    )
    recalculate_cache: bool = field(
        default=False,
        metadata={"help": "Mevcut cache'i yoksay ve RVT türevlerini yeniden hesapla"},
    )


DEFAULTS = PipelineDefaults()


def default_for(name: str):
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
    list_to_tuple_fields = {'sigma_scales', 'morphology_radii', 'rvt_radii'}
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
    best = np.argmax(sigma_b)
    threshold = (bin_edges[best] + bin_edges[best + 1]) * 0.5
    return float(np.clip(threshold, 0.0, 1.0))


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


def compute_derivatives_with_rvt(
    dtm: np.ndarray,
    pixel_size: float,
    radii: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RVT rutinlerini kullanarak SVF, açıklık, LRM ve eğim hesapla."""
    if radii is None:
        radii = DEFAULTS.rvt_radii
    dtm_filled, valid_mask = fill_nodata(dtm)
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
    for radius in radii:
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
                LOGGER.warning("RVT openness dict lacked array values; using zeros.")
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
                LOGGER.warning("RVT openness sequence empty; using zeros.")
            # single array case
            try:
                arr = _as_float32_array(res, "openness")
                return arr, arr
            except Exception:
                LOGGER.warning("RVT openness returned unsupported type; using zeros.")
        # fallback if no openness available
        zeros = np.zeros_like(dtm_filled, dtype=np.float32)
        return zeros, zeros

    pos_open, neg_open = _compute_openness(float(max(radii)))

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
        low_pass = gaussian_filter(dtm_filled, sigma=DEFAULTS.gaussian_lrm_sigma)
        lrm = (dtm_filled - low_pass).astype(np.float32)

    # slope signature also varies across versions; try with/without keywords.
    # If slope is unavailable in rvt.vis, fall back to numpy gradient.
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
        LOGGER.warning("rvt.vis.slope missing; using gradient-based fallback.")
        # Compute slope in degrees: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        # Use filled DTM to avoid NaNs breaking gradient
        gy, gx = np.gradient(dtm_filled.astype(np.float32), pixel_size, pixel_size)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    for derivative in (svf_avg, pos_open, neg_open, lrm, slope):
        derivative[~valid_mask] = np.nan
    return svf_avg, pos_open, neg_open, lrm, slope


def _score_rvtlog(dtm: np.ndarray, pixel_size: float) -> np.ndarray:
    """Birleşik RVT + LoG + gradyan klasik skor hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    svf, _, neg, lrm, _ = compute_derivatives_with_rvt(dtm, pixel_size=pixel_size)
    sigmas = DEFAULTS.sigma_scales
    log_responses = [
        np.abs(gaussian_laplace(dtm_filled, sigma=s, mode="nearest"))
        for s in sigmas
    ]
    blob = np.maximum.reduce(log_responses)
    grad = gaussian_gradient_magnitude(dtm_filled, sigma=DEFAULTS.gaussian_gradient_sigma, mode="nearest")
    svf_c = 1.0 - _norm01(svf)
    neg_n = _norm01(neg)
    lrm_n = _norm01(lrm)
    var_n = _norm01(_local_variance(dtm_filled))
    score = (
        0.30 * _norm01(blob)
        + 0.20 * lrm_n
        + 0.15 * _norm01(svf_c)
        + 0.15 * _norm01(grad)
        + 0.10 * neg_n
        + 0.10 * var_n
    )
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_hessian(dtm: np.ndarray) -> np.ndarray:
    """Çok ölçekli Hessian sırt/vadi yanıt hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    sigmas = DEFAULTS.sigma_scales
    responses = [_hessian_response(dtm_filled, sigma=s) for s in sigmas]
    score = np.maximum.reduce(responses)
    score[~valid] = np.nan
    return score.astype(np.float32)


def _score_morph(dtm: np.ndarray) -> np.ndarray:
    """Morfolojik beyaz/siyah top-hat belirginlik skoru hesaplama."""
    dtm_filled, valid = fill_nodata(dtm)
    radii = DEFAULTS.morphology_radii
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
) -> np.ndarray:
    """Stack nine channels in the required order."""
    if rgb.shape[0] != 3:
        raise ValueError("RGB input must have three channels.")
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
    stacked = np.stack([ch.astype(np.float32) for ch in channels], axis=0)
    return stacked


def build_model(
    arch: str = "Unet",
    encoder: str = "resnet34",
    in_ch: int = 9,
) -> torch.nn.Module:
    """Instantiate a segmentation model from segmentation_models_pytorch."""
    if not hasattr(smp, arch):
        raise ValueError(f"Architecture '{arch}' not found in segmentation_models_pytorch.")
    model_cls = getattr(smp, arch)
    model = model_cls(
        encoder_name=encoder,
        in_channels=in_ch,
        classes=1,
        activation=None,
    )
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
    arch: str = "Unet", encoder: str = "resnet34", in_ch: int = 9
) -> torch.nn.Module:
    """
    Build a model that uses an ImageNet-pretrained 3-ch encoder and inflate its first conv to in_ch (default 9).
    Decoder weights come from the 3-ch SMP model where shapes match; others remain default-initialized.
    """
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
    return model_9


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
        # half-cosine ramps
        t = np.linspace(0, np.pi, ramp, endpoint=False, dtype=np.float32)
        up = (1 - np.cos(t)) * 0.5  # 0â†’1
        win[:ramp] = up
        win[-ramp:] = up[::-1]
        return win

    wr = _ramp(h)
    wc = _ramp(w)
    return np.outer(wr, wc).astype(np.float32)


@dataclass
class InferenceOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    transform: Affine
    crs: Optional[RasterioCRS]


@dataclass
class ClassicModeOutput:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    threshold: float


@dataclass
class ClassicOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
    transform: Affine
    crs: Optional[RasterioCRS]
    threshold: float
    per_mode: Dict[str, ClassicModeOutput]


@dataclass
class FusionOutputs:
    prob_path: Path
    mask_path: Path
    prob_map: np.ndarray
    mask: np.ndarray
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
    encoder: Optional[str] = None,
    min_area: Optional[float] = None,
) -> InferenceOutputs:
    """
    Run tiled inference and save outputs.
    
    Args:
        precomputed_deriv: Önceden hesaplanmış RVT türevleri (cache kullanımı için).
                          None ise her tile için RVT yeniden hesaplanır.
    """
    model.eval()
    model.to(device)

    if mask_talls is not None and band_idx[3] <= 0:
        LOGGER.warning("Tall-object masking requested but DSM band missing; disabling.")
        mask_talls = None

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

        prob_acc = np.zeros((height, width), dtype=np.float32)
        weight_acc = np.zeros((height, width), dtype=np.float32)
        ndsm_max = np.full((height, width), -np.inf, dtype=np.float32)
        valid_global = np.zeros((height, width), dtype=bool)

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

                ndsm_s = compute_ndsm(dsm_s, dtm_s)
                svf_s, pos_s, neg_s, lrm_s, slope_s = compute_derivatives_with_rvt(
                    dtm_s, pixel_size=pixel_size
                )
                stack_s = stack_channels(rgb_s, svf_s, pos_s, neg_s, lrm_s, slope_s, ndsm_s)

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
                        lows[ci] = float(np.percentile(v, 2.0))
                        highs[ci] = float(np.percentile(v, 98.0))
                lows_list.append(lows)
                highs_list.append(highs)

                sampled += 1
                if sampled >= norm_sample_tiles:
                    break

            if lows_list and highs_list:
                fixed_lows = np.median(np.stack(lows_list, axis=0), axis=0)
                fixed_highs = np.median(np.stack(highs_list, axis=0), axis=0)

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
                
                valid_mask = np.isfinite(dtm)
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

                ndsm_tile = compute_ndsm(dsm, dtm)
                svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
                    dtm, pixel_size=pixel_size
                )

            stacked = stack_channels(
                rgb=rgb,
                svf=svf,
                pos_open=pos_open,
                neg_open=neg_open,
                lrm=lrm,
                slope=slope,
                ndsm=ndsm_tile,
            )
            if global_norm and fixed_lows is not None and fixed_highs is not None:
                normed = robust_norm_fixed(stacked, fixed_lows, fixed_highs)
            else:
                normed = robust_norm(stacked)

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

        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.divide(prob_acc, weight_acc, out=np.zeros_like(prob_acc), where=weight_acc > 0)
        prob_map[~valid_global] = np.nan

        if mask_talls is not None:
            tall_mask = ndsm_max > float(mask_talls)
            prob_map[tall_mask] = 0.0

        prob_map = prob_map.astype(np.float32)
        binary_mask = (prob_map >= threshold).astype(np.uint8)
        binary_mask[~valid_global] = 0

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

    return InferenceOutputs(
        prob_path=prob_path,
        mask_path=mask_path,
        prob_map=prob_map,
        mask=binary_mask,
        transform=transform,
        crs=crs,
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

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        pixel_size = float((abs(transform.a) + abs(transform.e)) / 2.0)

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
                    score_tile = _score_rvtlog(dtm_tile, pixel_size=pixel_size)
                elif mode == "hessian":
                    score_tile = _score_hessian(dtm_tile)
                elif mode == "morph":
                    score_tile = _score_morph(dtm_tile)
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
) -> Optional[Path]:
    """Convert binary mask into polygons and write to GeoPackage."""
    if fiona is None and gpd is None:
        LOGGER.warning("Vector output skipped; install geopandas or fiona for vectorisation.")
        return None

    # Küçük gürültüleri temizlemek için binary opening (hızlı!)
    LOGGER.info("Küçük gürültüler temizleniyor...")
    cleaned_mask = grey_opening(mask.astype(np.uint8), size=(3, 3))
    
    LOGGER.info("Etiketleme yapılıyor...")
    structure = np.ones(LABEL_CONNECTIVITY_STRUCTURE, dtype=int)
    labels, num_features = ndimage.label(cleaned_mask.astype(bool), structure=structure)
    if num_features == 0:
        LOGGER.info("No features above threshold; skipping vectorisation.")
        return None

    LOGGER.info("Tespit edilen özellik sayısı: %d", num_features)
    
    # Performans optimizasyonu: Çok küçük poligonları erken filtrelemek için piksel sayılarını hesapla
    label_ids = np.arange(1, num_features + 1)
    pixel_counts = ndimage.sum(mask.astype(np.uint8), labels, index=label_ids)
    prob_sums = ndimage.sum(prob_map.astype(np.float32), labels, index=label_ids)
    
    # Piksel tabanlı ön filtreleme: min_area'yı yaklaşık piksel sayısına çevir
    # Transform'dan piksel boyutunu hesapla (m²)
    pixel_width = abs(transform[0])  # x yönünde piksel boyutu
    pixel_height = abs(transform[4])  # y yönünde piksel boyutu
    pixel_area = pixel_width * pixel_height  # piksel alanı (m²)
    
    min_pixels = max(1, int(min_area / pixel_area * 0.5))  # Güvenli marj ile
    valid_labels = pixel_counts >= min_pixels
    
    # Sadece geçerli label'ları tut - maskeyi filtrele
    filtered_labels = labels.copy()
    for idx, is_valid in enumerate(valid_labels, start=1):
        if not is_valid:
            filtered_labels[labels == idx] = 0
    
    if not valid_labels.any():
        LOGGER.info("Tüm poligonlar minimum piksel sayısının altında; vektörleştirme atlanıyor.")
        return None
    
    filtered_count = valid_labels.sum()
    LOGGER.info("Piksel filtresinden sonra kalan poligon sayısı: %d (elenen: %d)", 
                filtered_count, num_features - filtered_count)

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
    # Filtrelenmiş label'lardan poligon oluştur
    shape_generator = shapes(filtered_labels.astype(np.int32), mask=None, transform=transform)
    
    # Progress bar ile poligon işleme
    for geom, value in progress_bar(
        shape_generator,
        total=filtered_count + 1,  # +1 for background (0)
        desc="Vektörleştirme",
        unit="poligon"
    ):
        label_id = int(value)
        if label_id == 0:
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
        pixels = float(pixel_counts[label_id - 1])
        if pixels <= 0:
            continue
        mean_score = float(prob_sums[label_id - 1]) / pixels
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
                "id": int(label_id),
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
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "transform": transform,
        "crs": crs,
        "dtype": dtype,
        "compress": "deflate",
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


def resolve_out_prefix(input_path: Path, prefix: Optional[str]) -> Path:
    """Resolve output prefix path."""
    if prefix:
        out_path = Path(prefix)
        if out_path.is_dir():
            out_path = out_path / input_path.stem
        return out_path
    return input_path.with_suffix("")


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
        return cache_path / f"{input_path.stem}_derivatives.npz"
    return input_path.with_suffix(".derivatives.npz")


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
        LOGGER.info(f"Cache yükleniyor: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        
        # Metadata'yı ayır
        if "_metadata" not in data:
            LOGGER.warning("Cache dosyası geçersiz (metadata eksik)")
            return None
        
        metadata = data["_metadata"].item()
        
        # Diğer verileri al
        derivatives_data = {key: data[key] for key in data.files if key != "_metadata"}
        
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        LOGGER.info(f"Cache yüklendi: {size_mb:.1f} MB, {len(derivatives_data)} layer")
        
        return derivatives_data, metadata
        
    except Exception as e:
        LOGGER.error(f"Cache yüklenirken hata: {e}")
        return None


def validate_cache(
    metadata: Dict[str, Any],
    input_path: Path,
    bands: Sequence[int],
) -> bool:
    """
    Cache'in mevcut parametrelerle uyumlu olup olmadığını kontrol et.
    
    NOT: tile ve overlap parametreleri cache geçerliliğini ETKİLEMEZ!
    RVT türevleri tüm raster için bir kez hesaplanır, tile/overlap sadece
    model inference sırasında kullanılır.
    """
    required_keys = ["input_path", "input_mtime", "bands", "shape"]
    
    if not all(key in metadata for key in required_keys):
        LOGGER.warning("Cache metadata eksik")
        return False
    
    # Dosya değişmiş mi?
    current_mtime = input_path.stat().st_mtime
    if abs(metadata["input_mtime"] - current_mtime) > 1.0:
        LOGGER.warning("Girdi dosyası değişmiş, cache geçersiz")
        return False
    
    # Bant sırası aynı mı?
    if metadata["bands"] != list(bands):
        LOGGER.warning("Bant sırası değişmiş, cache geçersiz")
        return False
    
    LOGGER.info("Cache geçerli ✓")
    return True


@dataclass
class PrecomputedDerivatives:
    """Önceden hesaplanmış RVT türevlerini tutan sınıf."""
    rgb: np.ndarray          # (3, H, W)
    dsm: Optional[np.ndarray]  # (H, W) or None
    dtm: np.ndarray          # (H, W)
    svf: np.ndarray          # (H, W)
    pos_open: np.ndarray     # (H, W)
    neg_open: np.ndarray     # (H, W)
    lrm: np.ndarray          # (H, W)
    slope: np.ndarray        # (H, W)
    ndsm: np.ndarray         # (H, W)
    transform: Affine
    crs: Optional[RasterioCRS]
    pixel_size: float
    

def precompute_derivatives(
    input_path: Path,
    band_idx: Sequence[int],
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
    recalculate: bool = False,
) -> Optional[PrecomputedDerivatives]:
    """
    Tüm raster için RVT türevlerini önceden hesapla.
    
    Bu fonksiyon:
    1. Cache varsa ve geçerliyse yükler
    2. Yoksa tüm raster'ı okur ve RVT türevlerini hesaplar
    3. İstenirse cache'e kaydeder
    
    Returns:
        PrecomputedDerivatives or None
    """
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
                if validate_cache(metadata, input_path, band_idx):
                    # Cache'den yükle
                    return PrecomputedDerivatives(
                        rgb=derivatives_data["rgb"],
                        dsm=derivatives_data.get("dsm"),
                        dtm=derivatives_data["dtm"],
                        svf=derivatives_data["svf"],
                        pos_open=derivatives_data["pos_open"],
                        neg_open=derivatives_data["neg_open"],
                        lrm=derivatives_data["lrm"],
                        slope=derivatives_data["slope"],
                        ndsm=derivatives_data["ndsm"],
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
        def read_band(idx: int) -> Optional[np.ndarray]:
            if idx <= 0:
                return None
            data = src.read(idx, masked=True)
            return np.ma.filled(data.astype(np.float32), np.nan)
        
        rgb = np.stack([read_band(band_idx[i]) for i in range(3)], axis=0)
        dsm = read_band(band_idx[3])
        dtm = read_band(band_idx[4])
        
        if dtm is None:
            raise ValueError("DTM band gerekli")
        
        # nDSM hesapla
        LOGGER.info("nDSM hesaplanıyor...")
        ndsm = compute_ndsm(dsm, dtm)
        
        # RVT türevlerini hesapla
        LOGGER.info("RVT türevleri hesaplanıyor (SVF, openness, LRM, slope)...")
        svf, pos_open, neg_open, lrm, slope = compute_derivatives_with_rvt(
            dtm, pixel_size=pixel_size
        )
        
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
            }
            metadata = {
                "input_path": str(input_path),
                "input_mtime": input_path.stat().st_mtime,
                "bands": list(band_idx),
                "shape": (height, width),
                "pixel_size": pixel_size,
                # NOT: tile ve overlap cache'e KAYDEDİLMEZ!
                # RVT türevleri tüm raster için hesaplanır, tile/overlap
                # sadece model inference sırasında kullanılır.
            }
            save_derivatives_cache(cache_path, derivatives_data, metadata)
        
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
        "--mask-talls",
        type=float,
        default=default_for("mask_talls"),
        help=cli_help("mask_talls"),
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

    input_path = Path(config.input)
    if not input_path.exists():
        parser.error(f"Input raster not found: {input_path}")

    enc_mode = (config.encoders or "").strip().lower()
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
    # Yöntem etkinleştirme kontrolleri
    if not config.enable_deep_learning and not config.enable_classic:
        parser.error("En az bir yöntem etkin olmalı (deep learning veya classic).")
    
    classic_outputs: Optional[ClassicOutputs] = None
    fusion_outputs: Optional[FusionOutputs] = None
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if config.half and device.type != "cuda":
        LOGGER.warning("--half requested but CUDA not available; running in float32.")

    # Cache yönetimi - Multi-encoder için RVT türevlerini önceden hesapla
    precomputed_deriv: Optional[PrecomputedDerivatives] = None
    if config.enable_deep_learning and enc_mode not in ("", "none") and config.cache_derivatives:
        # Multi-encoder modu + cache etkin
        LOGGER.info("=" * 70)
        LOGGER.info("CACHE MODU: RVT türevleri bir kez hesaplanacak, tüm encoderlar kullanacak")
        LOGGER.info("=" * 70)
        cache_path = get_cache_path(input_path, config.cache_dir)
        precomputed_deriv = precompute_derivatives(
            input_path=input_path,
            band_idx=bands,
            use_cache=True,
            cache_path=cache_path,
            recalculate=config.recalculate_cache,
        )
        if precomputed_deriv:
            LOGGER.info("✓ RVT türevleri hazır - Her encoder çok daha hızlı çalışacak!")
        else:
            LOGGER.warning("RVT türevleri hesaplanamadı, normal moda devam ediliyor")

    # Derin öğrenme etkinse modelleri çalıştır
    if config.enable_deep_learning and enc_mode not in ("", "none"):
        enc_list = (
            available_encoders_list()
            if enc_mode == "all"
            else [enc.strip() for enc in config.encoders.split(",") if enc.strip()]
        )
        
        # Cache kullanılıyorsa ilerleme mesajı
        if precomputed_deriv:
            LOGGER.info(f"Toplam {len(enc_list)} encoder çalıştırılacak (RVT hesaplaması atlanacak)")
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

            if per_weights is not None:
                LOGGER.info("[%s] Eğitilmiş ağırlıklar yükleniyor: %s", suffix, per_weights)
                model = build_model(arch=config.arch, encoder=smp_name, in_ch=9)
                load_weights(model, per_weights, map_location=device)
            else:
                LOGGER.info("[%s] Zero-shot modunda başlatılıyor (ImageNet 3->9 genişletme)", suffix)
                model = build_model_with_imagenet_inflated(arch=config.arch, encoder=smp_name, in_ch=9)

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
                precomputed_deriv=precomputed_deriv,  # Cache kullanımı
                encoder=suffix,
                min_area=config.min_area,
            )
            LOGGER.info("[%s] ✓ Olasılık haritası: %s", suffix, outputs.prob_path)
            LOGGER.info("[%s] ✓ İkili maske: %s", suffix, outputs.mask_path)

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
    
    # Tek model çalıştır (enable_deep_learning aktifse ve enc_mode none ise)
    if config.enable_deep_learning and config.zero_shot_imagenet:
        LOGGER.info("Zero-shot mode: using ImageNet-pretrained encoder inflated to 9 channels.")
        model = build_model_with_imagenet_inflated(
            arch=config.arch, encoder=config.encoder, in_ch=9
        )
    elif config.enable_deep_learning:
        model = build_model(arch=config.arch, encoder=config.encoder, in_ch=9)
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
    if config.enable_deep_learning and model is not None:
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
            encoder=config.encoder,
            min_area=config.min_area,
        )

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

    # Fusion (birleştirme) çalıştır (etkinse ve her iki yöntem de çalıştıysa)
    if fuse_enabled and classic_outputs is not None and outputs is not None:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info(f"FUSION (BİRLEŞTİRME) BAŞLATILIYOR (alpha={config.alpha})")
        LOGGER.info("=" * 70)
        alpha = float(config.alpha)
        fuse_threshold = config.th if config.th is not None else 0.5
        
        fused_prob, fused_mask = compute_fused_probability(
            dl_prob=outputs.prob_map.astype(np.float32),
            classic_prob=classic_outputs.prob_map.astype(np.float32),
            alpha=alpha,
            threshold=fuse_threshold,
        )

        base_prefix = out_prefix.with_suffix("")
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Parametreli dosya adı oluştur (fusion için)
        fused_filename = build_filename_with_params(
            base_name=base_prefix.name,
            mode_suffix="fused",
            threshold=fuse_threshold,
            tile=config.tile,
            alpha=alpha,
            min_area=config.min_area,
        )
        
        fused_prob_path = base_prefix.parent / f"{fused_filename}_prob.tif"
        fused_mask_path = base_prefix.parent / f"{fused_filename}_mask.tif"

        write_prob_and_mask_rasters(
            prob_map=fused_prob,
            mask=fused_mask,
            transform=outputs.transform,
            crs=outputs.crs,
            prob_path=fused_prob_path,
            mask_path=fused_mask_path,
        )

        fusion_outputs = FusionOutputs(
            prob_path=fused_prob_path,
            mask_path=fused_mask_path,
            prob_map=fused_prob,
            mask=fused_mask,
            transform=outputs.transform,
            crs=outputs.crs,
            threshold=fuse_threshold,
        )
        LOGGER.info("✓ Birleştirilmiş (fused) çıktılar: %s, %s", fused_prob_path, fused_mask_path)

    # Vektörleştirme (etkinse)
    if config.vectorize:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("VEKTÖRLEŞTİRME BAŞLATILIYOR")
        LOGGER.info("=" * 70)
        vector_jobs: List[Tuple[str, np.ndarray, np.ndarray, Affine, Optional[RasterioCRS], Path]] = []
        
        if outputs is not None:
            vector_jobs.append((
                "dl",
                outputs.mask,
                outputs.prob_map,
                outputs.transform,
                outputs.crs,
                outputs.mask_path.with_suffix(""),
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
                )
            )

        for label, mask_arr, prob_arr, transform, crs_obj, out_base in vector_jobs:
            LOGGER.info(f"  → {label} poligonlaştırılıyor...")
            vector_file = vectorize_predictions(
                mask=mask_arr,
                prob_map=prob_arr,
                transform=transform,
                crs=crs_obj,
                out_path=out_base,
                min_area=config.min_area,
                simplify_tol=config.simplify,
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
# Senaryo 1: Sadece DL
#   enable_deep_learning: true, enable_classic: false, enable_fusion: false
#
# Senaryo 2: Sadece Klasik
#   enable_deep_learning: false, enable_classic: true, enable_fusion: false
#
# Senaryo 3: Her İkisi + Fusion (ÖNERİLİR)
#   enable_deep_learning: true, enable_classic: true, enable_fusion: true
#
# ============================================================================
