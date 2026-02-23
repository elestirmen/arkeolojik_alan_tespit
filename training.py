#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arkeolojik Alan Tespiti - 12 Kanallı U-Net Eğitim Scripti

Bu script, egitim_verisi_olusturma.py ile oluşturulan 12 kanallı tile'ları
kullanarak U-Net modelini eğitir.

Özellikler:
    - 12 kanallı girdi desteği (RGB + RVT türevleri + Curvature + TPI)
    - CBAM Attention modülü (opsiyonel)
    - Mixed precision training (AMP)
    - Erken durdurma (Early stopping)
    - Model checkpoint'leri
    - Detaylı eğitim logları
    - TensorBoard desteği (opsiyonel)

Kullanım:
    python training.py --data training_data --epochs 50 --encoder resnet34
    
    Veya varsayılan ayarlarla:
    python training.py

Gereksinimler:
    - PyTorch >= 2.0
    - segmentation-models-pytorch
    - numpy
    - tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Segmentation Models PyTorch
smp = None


def _require_smp():
    """`segmentation-models-pytorch` bağımlılığını yalnızca eğitimde yükle."""
    global smp
    if smp is None:
        try:
            import segmentation_models_pytorch as _smp
        except ImportError as exc:
            raise ImportError(
                "segmentation-models-pytorch kurulu değil. Kurulum: pip install segmentation-models-pytorch"
            ) from exc
        smp = _smp
    return smp

# Mevcut projedeki attention modüllerini import et
try:
    from archaeo_detect import (
        ChannelAttention,
        SpatialAttention,
        CBAM,
        AttentionWrapper,
    )
except ImportError:
    print("UYARI: archaeo_detect.py'den attention modülleri import edilemedi.")
    print("Attention modülleri bu dosyada tanımlanacak.")
    
    # Fallback tanımlamalar
    class ChannelAttention(nn.Module):
        def __init__(self, in_channels: int, reduction: int = 4):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            reduced = max(in_channels // reduction, 1)
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, reduced, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced, in_channels, 1, bias=False)
            )
        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            return x * torch.sigmoid(avg_out + max_out)
    
    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
    
    class CBAM(nn.Module):
        def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
            super().__init__()
            self.channel_attention = ChannelAttention(in_channels, reduction)
            self.spatial_attention = SpatialAttention(kernel_size)
        def forward(self, x):
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            return x
    
    class AttentionWrapper(nn.Module):
        def __init__(self, base_model, in_channels: int, reduction: int = 4):
            super().__init__()
            self.input_attention = CBAM(in_channels, reduction=reduction)
            self.base_model = base_model
        def forward(self, x):
            x = self.input_attention(x)
            return self.base_model(x)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("training")

# ==================== CONFIG ====================
# IDE uzerinden dogrudan "Run" ettiginizde bu degerler kullanilir.
# Komut satirindan verilen argumanlar her zaman bu degerleri ezer.
CONFIG: dict[str, object] = {
    # data:
    # Egitim veri kok dizini.
    # Beklenen yapi: train/images, train/masks, val/images, val/masks
    "data": "training_data",

    # arch:
    # Segmentasyon model ailesi (segmentation_models_pytorch).
    # Secim; hiz, bellek kullanimi ve kaliteyi birlikte etkiler.
    # Unet | UnetPlusPlus | DeepLabV3Plus | FPN | PSPNet | MAnet | Linknet
    "arch": "Unet",

    # encoder:
    # BackBone/encoder secimi.
    # Daha buyuk encoder genelde daha guclu temsil ama daha yuksek maliyet demektir.
    # ornekler: resnet34 | resnet50 | efficientnet-b3 | mobilenet_v2 | densenet121
    "encoder": "resnet50",

    # no_attention:
    # True ise giristeki CBAM attention katmani kapatilir.
    # False iken attention aktiftir.
    "no_attention": False,

    # epochs:
    # Toplam egitim epoch sayisi.
    "epochs": 50,

    # batch_size:
    # Batch boyutu.
    # Yuksek deger daha stabil gradient verebilir; daha fazla VRAM ister.
    "batch_size": 8,

    # lr:
    # Ogrenme orani (AdamW).
    # Cok yuksek olursa kararsizlik, cok dusuk olursa yavas ogrenme gorulebilir.
    "lr": 1e-4,

    # loss:
    # Optimize edilen kayip fonksiyonu.
    # bce | dice | combined | focal
    # Not: balance_mode/pos_weight ayarlari sadece bce ve combined icin etkilidir.
    "loss": "combined",

    # balance_mode:
    # Sinif dengesizligi modu (BCE tarafinda agirliklandirma davranisi):
    # - none   : Agirliklandirma kapali (pos_weight=1.0 gibi davranir)
    # - auto   : Train maskelerinden piksel bazli negatif/pozitif orani hesaplanir
    #            ve bu orana gore otomatik pos_weight secilir
    # - manual : pos_weight degeri dogrudan kullanilir
    # Not: Yalnizca bce/combined icin etkilidir.
    "balance_mode": None,             #"auto",

    # pos_weight:
    # Manual modda BCE pozitif sinif agirligi.
    # Pozitif sinif azsa degeri artirmak recall'i destekleyebilir.
    "pos_weight": 1.0,

    # max_auto_pos_weight:
    # Auto modda hesaplanan pos_weight icin ust sinir (clip).
    # Asiri buyuk agirliklarin egitimi bozmasini onler.
    "max_auto_pos_weight": 100.0,

    # patience:
    # Erken durdurma sabri (iyilesme olmayan epoch sayisi).
    "patience": 10,

    # metric_threshold:
    # Egitim/dogrulama metriklerini (IoU/F1/Precision/Recall) hesaplamak icin
    # kullanilan olasilik esigi.
    "metric_threshold": 0.5,

    # val_threshold_sweep:
    # Dogrulamada farkli esikleri tarayip en iyi IoU ve esigi raporlar.
    # Bu ayar sadece metrik/raporlama icindir; loss'u etkilemez.
    "val_threshold_sweep": True,
    "val_threshold_min": 0.1,
    "val_threshold_max": 0.9,
    "val_threshold_step": 0.05,

    # workers:
    # DataLoader worker sayisi.
    # CPU cekirdek sayisi ve disk hizina gore ayarlanir.
    "workers": 4,

    # no_amp:
    # True ise mixed precision (AMP) kapatilir.
    # CUDA'da genelde AMP acik kullanmak hiz/VRAM acisindan avantajlidir.
    "no_amp": False,

    # output:
    # Checkpoint cikti dizini.
    "output": "checkpoints",

    # save_every_epoch:
    # True ise her epoch sonunda ek checkpoint kaydeder.
    # Disk kullanimi artar ama geri analiz kolaylasir.
    "save_every_epoch": True,

    # epoch_dir:
    # save_every_epoch aktifken, epoch checkpoint'lerinin tutulacagi alt klasor adi.
    "epoch_dir": "epochs",

    # seed:
    # Rastgelelik sabiti.
    # Deneyleri tekrar edilebilir kilmak icin sabit tutulur.
    "seed": 42,

    # allow_all_negative:
    # True ise tum maskeler negatif olsa bile egitim zorla devam eder.
    # Veri hatasini erken yakalamak icin normalde False onerilir.
    "allow_all_negative": False,

    # train_neg_to_pos_ratio:
    # Train setinde negatif ornekleri pozitiflere gore oransal olarak alt-ornekle.
    # None: kapali (tum train ornekleri kullanilir)
    # 1.0: negatif ~= pozitif
    # 0.5: negatif ~= pozitifin yarisi
    "train_neg_to_pos_ratio": 2, #None,

    # train_neg_sample_seed:
    # Oran bazli negatif alt-orneklemede rastgelelik tohumu.
    "train_neg_sample_seed": 42,

    # val_keep_ratio:
    # Val hedef boyutu train-secilen ornek sayisina gore hesaplanir:
    # hedef_val ~= round(train_secilen * val_keep_ratio)
    # 1.0: val hedefi train-secilen kadar
    # 0.5: val hedefi train-secilenin yarisi
    # Not: hedef, val toplamindan buyuk olamaz.c
    "val_keep_ratio": 0.5,

    # val_sample_seed:
    # Val alt-ornekleme rastgelelik tohumu.
    "val_sample_seed": 42,
}
# ===============================================


# ==============================================================================
# DATASET
# ==============================================================================

class ArchaeologyDataset(Dataset):
    """
    12 kanallı arkeolojik alan tespiti veri seti.
    
    egitim_verisi_olusturma.py tarafından oluşturulan .npy dosyalarını okur.
    """
    
    def __init__(
        self,
        data_dir: Path,
        augment: bool = True,
        file_format: str = "npz",
    ):
        """
        Args:
            data_dir: train/ veya val/ dizini
            augment: Veri artırma uygula
            file_format: "npy" veya "npz"
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.augment = augment
        self.file_format = file_format
        
        # Dosyaları listele
        pattern = f"*.{file_format}"
        self.image_files = sorted(self.images_dir.glob(pattern))
        
        if len(self.image_files) == 0:
            raise ValueError(f"Hiç görüntü dosyası bulunamadı: {self.images_dir}")
        
        LOGGER.info(f"Dataset yüklendi: {len(self.image_files)} örnek ({data_dir})")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")
        
        # Dosyaları oku
        if self.file_format == "npy":
            image = np.load(img_path)  # (C, H, W)
            mask = np.load(mask_path)  # (H, W)
        else:  # npz
            image = np.load(img_path)["image"]
            mask = np.load(mask_path)["mask"]

        # Legacy veriyle uyumluluk: maskeyi zorunlu 0/1 yap.
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mask = np.where(np.isfinite(mask) & (mask > 0), 1.0, 0.0).astype(np.float32)
        
        # Veri artırma
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Tensor'a çevir
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        
        return image, mask
    
    def _augment(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Basit veri artırma işlemleri."""
        
        # Yatay flip (%50)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Dikey flip (%50)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        
        # 90° rotasyonlar (%75)
        if np.random.random() > 0.25:
            k = np.random.randint(1, 4)  # 90, 180, 270 derece
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
        
        return image, mask


# ==============================================================================
# LOSS FONKSİYONLARI
# ==============================================================================

class DiceLoss(nn.Module):
    """Dice Loss - Segmentasyon için yaygın kullanılan kayıp fonksiyonu."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice


class BCELoss(nn.Module):
    """BCE loss with optional positive class weight for class imbalance."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError(f"Geçersiz pos_weight değeri: {pos_weight}")
        self.pos_weight = float(pos_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight_tensor: Optional[torch.Tensor] = None
        if self.pos_weight != 1.0:
            pos_weight_tensor = torch.tensor(
                [self.pos_weight],
                dtype=pred.dtype,
                device=pred.device,
            )
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=pos_weight_tensor,
        )


class CombinedLoss(nn.Module):
    """BCE + Dice Loss kombinasyonu."""
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss - Dengesiz sınıflar için etkili."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


# ==============================================================================
# METRİKLER
# ==============================================================================

def calculate_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Segmentasyon metriklerini hesaplar."""

    tp, fp, fn, tn = _compute_confusion_counts(pred, target, threshold)
    return _metrics_from_confusion_counts(tp=tp, fp=fp, fn=fn, tn=tn)


def _compute_confusion_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Tuple[float, float, float, float]:
    """Logit tahminlerden TP/FP/FN/TN piksel sayÄ±larÄ±nÄ± Ã§Ä±kar."""
    pred_binary = (torch.sigmoid(pred) > float(threshold)).float()
    target_binary = (target > 0.5).float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    tp = float((pred_flat * target_flat).sum().item())
    fp = float((pred_flat * (1.0 - target_flat)).sum().item())
    fn = float(((1.0 - pred_flat) * target_flat).sum().item())
    tn = float(((1.0 - pred_flat) * (1.0 - target_flat)).sum().item())
    return tp, fp, fn, tn


def _metrics_from_confusion_counts(
    *,
    tp: float,
    fp: float,
    fn: float,
    tn: float,
) -> Dict[str, float]:
    """TP/FP/FN/TN deÄŸerlerinden metrikleri hesapla."""
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def _build_threshold_sweep(
    *,
    metric_threshold: float,
    min_threshold: float,
    max_threshold: float,
    step: float,
    enabled: bool,
) -> Optional[Tuple[float, ...]]:
    """Validation iÃ§in kullanÄ±lacak eÅŸik tarama listesini oluÅŸtur."""
    if not enabled:
        return None
    if step <= 0:
        return (float(metric_threshold),)
    if min_threshold > max_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    thresholds = np.arange(min_threshold, max_threshold + (0.5 * step), step, dtype=np.float32)
    values = {
        round(float(np.clip(th, 0.0, 1.0)), 6)
        for th in thresholds
    }
    values.add(round(float(np.clip(metric_threshold, 0.0, 1.0)), 6))
    return tuple(sorted(values))


# ==============================================================================
# EĞİTİM FONKSİYONLARI
# ==============================================================================

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu."""
    
    # Veri
    data_dir: Path = field(default_factory=lambda: Path("training_data"))
    # Model
    arch: str = "Unet"
    encoder: str = "resnet34"
    in_channels: int = 12
    enable_attention: bool = True
    attention_reduction: int = 4
    
    # Eğitim
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    
    # Loss
    loss_type: str = "combined"  # "bce", "dice", "combined", "focal"
    balance_mode: str = "auto"  # "none", "auto", "manual"
    pos_weight: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "plateau", "step"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Metric / validation
    metric_threshold: float = 0.5
    val_threshold_sweep: bool = True
    val_threshold_min: float = 0.1
    val_threshold_max: float = 0.9
    val_threshold_step: float = 0.05
    
    # Diğer
    num_workers: int = 4
    use_amp: bool = True  # Mixed precision
    seed: int = 42
    train_neg_to_pos_ratio: Optional[float] = None
    train_neg_sample_seed: int = 42
    val_keep_ratio: float = 1.0
    val_sample_seed: int = 42
    
    # Çıktı
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every_epoch: bool = True
    epoch_dir: str = "epochs"


def _detect_split_file_format(images_dir: Path) -> Optional[str]:
    """Detect file format for a split directory and reject mixed extensions."""
    has_npz = any(images_dir.glob("*.npz"))
    has_npy = any(images_dir.glob("*.npy"))

    if has_npz and has_npy:
        raise ValueError(
            f"Ayni klasorde hem .npz hem .npy bulundu: {images_dir}. "
            "Tek bir format kullanin."
        )
    if has_npz:
        return "npz"
    if has_npy:
        return "npy"
    return None


def _infer_file_format(data_dir: Path) -> str:
    """Egitim verisindeki dosya formatini (npz/npy) tespit eder."""
    train_images = data_dir / "train" / "images"
    val_images = data_dir / "val" / "images"

    train_format = _detect_split_file_format(train_images)
    val_format = _detect_split_file_format(val_images)

    if train_format is None:
        raise ValueError(
            f"Desteklenen egitim dosyasi bulunamadi: {train_images} (*.npz / *.npy)"
        )
    if val_format is None:
        raise ValueError(
            f"Desteklenen dogrulama dosyasi bulunamadi: {val_images} (*.npz / *.npy)"
        )
    if train_format != val_format:
        raise ValueError(
            "Train ve val klasorlerinde farkli dosya formati bulundu: "
            f"train={train_format}, val={val_format}"
        )

    return train_format


def _load_mask_array(mask_path: Path, file_format: str) -> np.ndarray:
    """Maske dosyasını npy/npz formatından okur."""
    if file_format == "npy":
        return np.load(mask_path)

    with np.load(mask_path) as packed:
        if "mask" not in packed:
            raise ValueError(f"'mask' anahtarı bulunamadı: {mask_path}")
        return packed["mask"]


def _count_positive_mask_files(mask_dir: Path, file_format: str) -> Tuple[int, int]:
    """
    Maske dosyalarında en az bir pozitif piksel içeren tile sayısını döndür.

    Returns:
        (toplam_tile_sayisi, pozitif_tile_sayisi)
    """
    mask_files = sorted(mask_dir.glob(f"*.{file_format}"))
    positive_files = 0

    for mask_path in mask_files:
        mask = _load_mask_array(mask_path, file_format)

        if np.any(mask > 0):
            positive_files += 1

    return len(mask_files), positive_files


def _count_positive_mask_pixels(mask_dir: Path, file_format: str) -> Tuple[int, int]:
    """
    Maske dosyalarındaki toplam piksel ve pozitif piksel sayısını döndürür.

    Returns:
        (toplam_piksel, pozitif_piksel)
    """
    mask_files = sorted(mask_dir.glob(f"*.{file_format}"))
    total_pixels = 0
    positive_pixels = 0

    for mask_path in mask_files:
        mask = _load_mask_array(mask_path, file_format)
        total_pixels += int(mask.size)
        positive_pixels += int(np.count_nonzero(mask > 0))

    return total_pixels, positive_pixels


def _select_train_indices_by_neg_pos_ratio(
    train_dataset: ArchaeologyDataset,
    neg_to_pos_ratio: Optional[float],
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    """Train setinde negatif tile'lari pozitiflere gore alt-ornekle."""
    ratio: Optional[float] = None
    if neg_to_pos_ratio is not None:
        ratio = float(neg_to_pos_ratio)
        if ratio < 0:
            raise ValueError(
                f"train_neg_to_pos_ratio None veya >= 0 olmalı, verilen: {neg_to_pos_ratio}"
            )
    if int(seed) < 0:
        raise ValueError(f"train_neg_sample_seed negatif olamaz, verilen: {seed}")

    positive_indices: List[int] = []
    negative_indices: List[int] = []

    for idx, img_path in enumerate(train_dataset.image_files):
        mask_path = train_dataset.masks_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")
        mask = _load_mask_array(mask_path, train_dataset.file_format)
        if np.any(mask > 0):
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)

    if ratio is None:
        target_negative_keep = len(negative_indices)
    else:
        target_negative_keep = int(round(len(positive_indices) * ratio))
        target_negative_keep = max(0, min(target_negative_keep, len(negative_indices)))

    if target_negative_keep >= len(negative_indices):
        selected_negative_indices = list(negative_indices)
    elif target_negative_keep <= 0:
        selected_negative_indices = []
    else:
        rng = np.random.RandomState(int(seed))
        selected_pos = rng.choice(
            len(negative_indices),
            size=target_negative_keep,
            replace=False,
        )
        selected_negative_indices = sorted(negative_indices[int(i)] for i in selected_pos)

    selected_indices = sorted(positive_indices + selected_negative_indices)
    stats = {
        "total_samples": len(train_dataset),
        "positive_samples": len(positive_indices),
        "negative_samples": len(negative_indices),
        "target_negative_keep": int(target_negative_keep),
        "selected_positive_samples": len(positive_indices),
        "selected_negative_samples": len(selected_negative_indices),
        "selected_total_samples": len(selected_indices),
    }
    return selected_indices, stats


def _select_indices_by_keep_ratio(
    total_samples: int,
    keep_ratio: float,
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    """Toplam orneklerden belirtilen oranda alt-ornek sec."""
    if total_samples <= 0:
        raise ValueError(f"total_samples pozitif olmalı, verilen: {total_samples}")

    ratio = float(keep_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"keep_ratio 0-1 aralığında olmalı (0 hariç), verilen: {keep_ratio}")
    if int(seed) < 0:
        raise ValueError(f"sample_seed negatif olamaz, verilen: {seed}")

    if ratio >= 1.0:
        selected_indices = list(range(total_samples))
    else:
        target_keep = int(round(total_samples * ratio))
        target_keep = max(1, min(total_samples, target_keep))
        rng = np.random.RandomState(int(seed))
        selected = rng.choice(total_samples, size=target_keep, replace=False)
        selected_indices = sorted(int(i) for i in selected)

    stats = {
        "total_samples": int(total_samples),
        "selected_total_samples": int(len(selected_indices)),
    }
    return selected_indices, stats


def _compute_val_target_samples(
    train_selected_samples: int,
    val_total_samples: int,
    val_keep_ratio: float,
) -> int:
    """Val hedef ornek sayisini train-secilen ornek sayisina gore hesapla."""
    if train_selected_samples <= 0:
        raise ValueError(
            f"train_selected_samples pozitif olmalı, verilen: {train_selected_samples}"
        )
    if val_total_samples <= 0:
        raise ValueError(f"val_total_samples pozitif olmalı, verilen: {val_total_samples}")
    if not 0.0 < float(val_keep_ratio) <= 1.0:
        raise ValueError(
            f"val_keep_ratio 0-1 aralığında olmalı (0 hariç), verilen: {val_keep_ratio}"
        )

    target = int(round(int(train_selected_samples) * float(val_keep_ratio)))
    target = max(1, min(int(val_total_samples), int(target)))
    return int(target)


def create_model(config: TrainingConfig) -> nn.Module:
    """Model oluşturur."""

    smp_lib = _require_smp()

    if not hasattr(smp_lib, config.arch):
        raise ValueError(f"Mimari bulunamadı: {config.arch}")

    model_cls = getattr(smp_lib, config.arch)
    
    # Base model
    base_model = model_cls(
        encoder_name=config.encoder,
        encoder_weights="imagenet",  # RGB kanalları için
        in_channels=config.in_channels,
        classes=1,
        activation=None,
    )
    
    # Attention ekle
    if config.enable_attention:
        model = AttentionWrapper(
            base_model, 
            in_channels=config.in_channels,
            reduction=config.attention_reduction,
        )
        LOGGER.info(f"CBAM Attention eklendi (reduction={config.attention_reduction})")
    else:
        model = base_model
    
    return model


def create_loss(config: TrainingConfig) -> nn.Module:
    """Kayıp fonksiyonu oluşturur."""
    
    if config.loss_type == "bce":
        return BCELoss(pos_weight=config.pos_weight)
    elif config.loss_type == "dice":
        if config.balance_mode != "none":
            LOGGER.warning(
                "balance_mode=%s, ancak dice loss için pos_weight kullanılmaz.",
                config.balance_mode,
            )
        return DiceLoss()
    elif config.loss_type == "combined":
        return CombinedLoss(pos_weight=config.pos_weight)
    elif config.loss_type == "focal":
        if config.balance_mode != "none":
            LOGGER.warning(
                "balance_mode=%s, ancak focal loss için pos_weight kullanılmaz.",
                config.balance_mode,
            )
        return FocalLoss()
    else:
        raise ValueError(f"Bilinmeyen loss tipi: {config.loss_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Learning rate scheduler oluşturur."""
    
    if config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:
        return None


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    metric_threshold: float,
) -> Tuple[float, Dict[str, float]]:
    """Bir epoch eğitim yapar."""
    
    model.train()
    total_loss = 0.0
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    
    pbar = tqdm(train_loader, desc="Eğitim", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Metrikler (global piksel confusion sayıları ile biriktirilir)
        with torch.no_grad():
            tp, fp, fn, tn = _compute_confusion_counts(
                outputs,
                masks,
                threshold=metric_threshold,
            )
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["tn"] += tn
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    n = len(train_loader)
    avg_loss = total_loss / n
    avg_metrics = _metrics_from_confusion_counts(
        tp=totals["tp"],
        fp=totals["fp"],
        fn=totals["fn"],
        tn=totals["tn"],
    )
    
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_threshold: float = 0.5,
    sweep_thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, float]]]:
    """Doğrulama yapar."""
    
    model.eval()
    total_loss = 0.0
    fixed_totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    sweep_totals: Optional[Dict[float, Dict[str, float]]] = None
    if sweep_thresholds:
        sweep_totals = {
            float(th): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
            for th in sweep_thresholds
        }
    
    for images, masks in tqdm(val_loader, desc="Doğrulama", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()

        tp, fp, fn, tn = _compute_confusion_counts(
            outputs,
            masks,
            threshold=metric_threshold,
        )
        fixed_totals["tp"] += tp
        fixed_totals["fp"] += fp
        fixed_totals["fn"] += fn
        fixed_totals["tn"] += tn

        if sweep_totals is not None:
            for th, totals in sweep_totals.items():
                stp, sfp, sfn, stn = _compute_confusion_counts(
                    outputs,
                    masks,
                    threshold=th,
                )
                totals["tp"] += stp
                totals["fp"] += sfp
                totals["fn"] += sfn
                totals["tn"] += stn
    
    n = len(val_loader)
    avg_loss = total_loss / n
    avg_metrics = _metrics_from_confusion_counts(
        tp=fixed_totals["tp"],
        fp=fixed_totals["fp"],
        fn=fixed_totals["fn"],
        tn=fixed_totals["tn"],
    )

    sweep_summary: Optional[Dict[str, float]] = None
    if sweep_totals is not None and len(sweep_totals) > 0:
        best_threshold = float(metric_threshold)
        best_metrics = dict(avg_metrics)
        for th, totals in sweep_totals.items():
            metrics = _metrics_from_confusion_counts(
                tp=totals["tp"],
                fp=totals["fp"],
                fn=totals["fn"],
                tn=totals["tn"],
            )
            if metrics["iou"] > best_metrics["iou"] + 1e-12:
                best_metrics = metrics
                best_threshold = float(th)

        sweep_summary = {
            "best_threshold": float(best_threshold),
            "best_iou": float(best_metrics["iou"]),
            "best_precision": float(best_metrics["precision"]),
            "best_recall": float(best_metrics["recall"]),
            "best_f1": float(best_metrics["f1"]),
            "fixed_iou": float(avg_metrics["iou"]),
        }
    
    return avg_loss, avg_metrics, sweep_summary


def train(config: TrainingConfig) -> Path:
    """
    Model eğitimi ana fonksiyonu.
    
    Returns:
        En iyi model checkpoint'unun yolu
    """
    
    # Seed ayarla
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not 0.0 < float(config.val_keep_ratio) <= 1.0:
        raise ValueError(
            f"val_keep_ratio 0-1 aralığında olmalı (0 hariç), verilen: {config.val_keep_ratio}"
        )
    if int(config.val_sample_seed) < 0:
        raise ValueError(f"val_sample_seed negatif olamaz, verilen: {config.val_sample_seed}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Cihaz: {device}")
    
    if device.type == "cuda":
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
        LOGGER.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Çıktı dizini
    config.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_dir: Optional[Path] = None
    if config.save_every_epoch:
        epoch_checkpoint_dir = config.output_dir / config.epoch_dir
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset ve DataLoader
    file_format = _infer_file_format(config.data_dir)
    train_dataset = ArchaeologyDataset(
        config.data_dir / "train",
        augment=True,
        file_format=file_format,
    )
    val_dataset = ArchaeologyDataset(
        config.data_dir / "val",
        augment=False,
        file_format=file_format,
    )
    
    train_loader_dataset: Dataset = train_dataset
    train_sampling_stats: Optional[Dict[str, int]] = None
    if config.train_neg_to_pos_ratio is not None:
        selected_indices, train_sampling_stats = _select_train_indices_by_neg_pos_ratio(
            train_dataset=train_dataset,
            neg_to_pos_ratio=config.train_neg_to_pos_ratio,
            seed=config.train_neg_sample_seed,
        )
        if len(selected_indices) == 0:
            raise ValueError(
                "Train set alt-ornekleme sonrasi bos kaldi. "
                "--train-neg-to-pos-ratio degerini artirin veya devre disi birakin."
            )
        train_loader_dataset = Subset(train_dataset, selected_indices)

    val_loader_dataset: Dataset = val_dataset
    val_sampling_stats: Optional[Dict[str, int]] = None
    train_selected_count = len(train_loader_dataset)
    val_target_samples = _compute_val_target_samples(
        train_selected_samples=train_selected_count,
        val_total_samples=len(val_dataset),
        val_keep_ratio=config.val_keep_ratio,
    )
    if val_target_samples < len(val_dataset):
        val_keep_ratio_on_val = float(val_target_samples) / float(len(val_dataset))
        val_indices, val_sampling_stats = _select_indices_by_keep_ratio(
            total_samples=len(val_dataset),
            keep_ratio=val_keep_ratio_on_val,
            seed=config.val_sample_seed,
        )
        val_sampling_stats["target_samples"] = int(val_target_samples)
        val_sampling_stats["reference_train_samples"] = int(train_selected_count)
        val_loader_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_loader_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_loader_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    if train_sampling_stats is None and val_sampling_stats is None:
        LOGGER.info(f"E?itim: {len(train_dataset)} ?rnek, Do?rulama: {len(val_dataset)} ?rnek")
    else:
        train_count = (
            int(train_sampling_stats["selected_total_samples"])
            if train_sampling_stats is not None
            else len(train_dataset)
        )
        val_count = (
            int(val_sampling_stats["selected_total_samples"])
            if val_sampling_stats is not None
            else len(val_dataset)
        )
        LOGGER.info(
            "E?itim: %d/%d ?rnek | Do?rulama: %d/%d ?rnek",
            train_count,
            len(train_dataset),
            val_count,
            len(val_dataset),
        )
        if train_sampling_stats is not None:
            LOGGER.info(
                "  Train alt-ornekleme -> pozitif=%d, negatif=%d, secilen_negatif=%d, oran=%.3f",
                int(train_sampling_stats["positive_samples"]),
                int(train_sampling_stats["negative_samples"]),
                int(train_sampling_stats["selected_negative_samples"]),
                float(config.train_neg_to_pos_ratio),
            )
        if val_sampling_stats is not None:
            LOGGER.info(
                "  Val alt-ornekleme -> secilen=%d/%d, hedef=%d (train_ref=%d, oran=%.3f, seed=%d)",
                int(val_sampling_stats["selected_total_samples"]),
                len(val_dataset),
                int(val_sampling_stats.get("target_samples", 0)),
                int(val_sampling_stats.get("reference_train_samples", train_selected_count)),
                float(config.val_keep_ratio),
                int(config.val_sample_seed),
            )

    # Model
    model = create_model(config)
    model = model.to(device)
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Parametre sayısı: {total_params:,} (eğitilebilir: {trainable_params:,})")
    
    # Loss, Optimizer, Scheduler
    criterion = create_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == "cuda" else None
    
    # Eğitim döngüsü
    best_val_loss = float("inf")
    best_val_iou = float("-inf")
    patience_counter = 0
    best_model_path: Optional[Path] = None
    best_loss_model_path: Optional[Path] = None
    best_iou_model_path: Optional[Path] = None

    model_name = f"{config.arch}_{config.encoder}_{config.in_channels}ch"
    if config.enable_attention:
        model_name += "_attention"
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "val_iou_fixed": [],
        "val_iou_best": [],
        "val_iou_best_threshold": [],
        "lr": [],
    }
    sweep_thresholds = _build_threshold_sweep(
        metric_threshold=config.metric_threshold,
        min_threshold=config.val_threshold_min,
        max_threshold=config.val_threshold_max,
        step=config.val_threshold_step,
        enabled=config.val_threshold_sweep,
    )
    
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EĞİTİM BAŞLIYOR")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Model: {config.arch} + {config.encoder}")
    LOGGER.info(f"Girdi kanalları: {config.in_channels}")
    LOGGER.info(f"Attention: {'Aktif' if config.enable_attention else 'Kapalı'}")
    LOGGER.info(f"Loss: {config.loss_type}")
    LOGGER.info(
        f"Class balance: {config.balance_mode} (pos_weight={config.pos_weight:.4f})"
    )
    if config.train_neg_to_pos_ratio is None:
        LOGGER.info("Train negatif alt-ornekleme: kapalÄ±")
    else:
        LOGGER.info(
            "Train negatif alt-ornekleme: neg/pos=%.3f (seed=%d)",
            float(config.train_neg_to_pos_ratio),
            int(config.train_neg_sample_seed),
        )
    LOGGER.info(
        "Val alt-ornekleme: hedef=round(train_secilen*%.3f), max=val_toplam (seed=%d)",
        float(config.val_keep_ratio),
        int(config.val_sample_seed),
    )
    LOGGER.info(f"Epochs: {config.epochs}")
    LOGGER.info(f"Batch size: {config.batch_size}")
    LOGGER.info(f"Learning rate: {config.lr}")
    LOGGER.info(
        "Metric threshold: %.2f | Val sweep: %s",
        config.metric_threshold,
        "aktif" if sweep_thresholds else "kapalı",
    )
    if sweep_thresholds:
        LOGGER.info(
            "Val threshold sweep aralığı: %.2f-%.2f (adım=%.2f, %d eşik)",
            config.val_threshold_min,
            config.val_threshold_max,
            config.val_threshold_step,
            len(sweep_thresholds),
        )
    LOGGER.info(f"Mixed precision: {'Aktif' if config.use_amp else 'Kapalı'}")
    LOGGER.info(f"Checkpoint dizini: {config.output_dir.resolve()}")
    if epoch_checkpoint_dir is not None:
        LOGGER.info(f"Epoch checkpoint dizini: {epoch_checkpoint_dir.resolve()}")
    LOGGER.info("=" * 60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Eğitim
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config.use_amp,
            config.metric_threshold,
        )
        
        # Doğrulama
        val_loss, val_metrics, val_sweep = validate(
            model,
            val_loader,
            criterion,
            device,
            metric_threshold=config.metric_threshold,
            sweep_thresholds=sweep_thresholds,
        )
        val_iou_fixed = float(val_metrics["iou"])
        val_iou_selection = val_iou_fixed
        val_iou_selection_threshold = float(config.metric_threshold)
        if val_sweep is not None:
            val_iou_selection = float(val_sweep["best_iou"])
            val_iou_selection_threshold = float(val_sweep["best_threshold"])
        
        # Scheduler güncelle
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Geçmiş kaydet
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_iou_selection)
        history["val_iou_fixed"].append(val_iou_fixed)
        history["val_iou_best"].append(val_iou_selection)
        history["val_iou_best_threshold"].append(val_iou_selection_threshold)
        history["lr"].append(current_lr)
        
        # Log
        epoch_time = time.time() - epoch_start
        if val_sweep is None:
            LOGGER.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU@{config.metric_threshold:.2f}: {val_iou_fixed:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Süre: {epoch_time:.1f}s"
            )
        else:
            LOGGER.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU@{config.metric_threshold:.2f}: {val_iou_fixed:.4f} | "
                f"Val IoU(best): {val_iou_selection:.4f} @th={val_iou_selection_threshold:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Süre: {epoch_time:.1f}s"
            )
        
        # En iyi model kontrolü
        val_iou = float(val_iou_selection)
        loss_improved = val_loss < best_val_loss - config.min_delta
        iou_improved = np.isfinite(val_iou) and (val_iou > best_val_iou)
        improved = loss_improved or iou_improved

        if loss_improved:
            best_val_loss = val_loss
        if iou_improved:
            best_val_iou = val_iou

        epoch_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_iou": float(train_metrics["iou"]),
            "val_loss": val_loss,
            "val_iou": val_iou_fixed,
            "val_iou_selection": val_iou_selection,
            "val_iou_threshold": float(config.metric_threshold),
            "val_iou_selection_threshold": val_iou_selection_threshold,
            "lr": current_lr,
            "config": {
                "arch": config.arch,
                "encoder": config.encoder,
                "in_channels": config.in_channels,
                "enable_attention": config.enable_attention,
                "attention_reduction": config.attention_reduction,
                "loss_type": config.loss_type,
                "balance_mode": config.balance_mode,
                "pos_weight": config.pos_weight,
                "train_neg_to_pos_ratio": config.train_neg_to_pos_ratio,
                "train_neg_sample_seed": config.train_neg_sample_seed,
                "val_keep_ratio": config.val_keep_ratio,
                "val_sample_seed": config.val_sample_seed,
                "metric_threshold": config.metric_threshold,
                "val_threshold_sweep": config.val_threshold_sweep,
                "val_threshold_min": config.val_threshold_min,
                "val_threshold_max": config.val_threshold_max,
                "val_threshold_step": config.val_threshold_step,
            },
        }
        if val_sweep is not None:
            epoch_checkpoint["val_iou_best"] = float(val_sweep["best_iou"])
            epoch_checkpoint["val_iou_best_threshold"] = float(val_sweep["best_threshold"])
        if epoch_checkpoint_dir is not None:
            epoch_model_path = epoch_checkpoint_dir / f"epoch_{epoch+1:03d}_{model_name}.pth"
            torch.save(epoch_checkpoint, epoch_model_path)

        if improved:
            patience_counter = 0

            if loss_improved:
                best_loss_model_path = config.output_dir / f"best_loss_{model_name}.pth"
                loss_checkpoint = dict(epoch_checkpoint)
                loss_checkpoint["best_metric"] = "val_loss"
                torch.save(loss_checkpoint, best_loss_model_path)
                LOGGER.info(f"  → En iyi loss modeli kaydedildi: {best_loss_model_path.name}")

            if iou_improved:
                best_iou_model_path = config.output_dir / f"best_iou_{model_name}.pth"
                iou_checkpoint = dict(epoch_checkpoint)
                iou_checkpoint["best_metric"] = "val_iou_selection"
                torch.save(iou_checkpoint, best_iou_model_path)
                LOGGER.info(f"  → En iyi IoU modeli kaydedildi: {best_iou_model_path.name}")

                # Backward-compatible default path used by downstream scripts/docs.
                best_model_path = config.output_dir / f"best_{model_name}.pth"
                torch.save(iou_checkpoint, best_model_path)
                LOGGER.info(f"  → Varsayılan en iyi model güncellendi: {best_model_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                LOGGER.info(f"\n⚠ Erken durdurma: {config.patience} epoch boyunca iyileşme yok")
                break
    
    total_time = time.time() - start_time
    
    # Eğitim geçmişini kaydet
    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Sonuç raporu
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EĞİTİM TAMAMLANDI")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Toplam süre: {total_time/60:.1f} dakika")
    LOGGER.info(f"En iyi Val Loss: {best_val_loss:.4f}")
    LOGGER.info(f"En iyi Val IoU (selection): {best_val_iou:.4f}")
    LOGGER.info(f"En iyi loss modeli: {best_loss_model_path}")
    LOGGER.info(f"En iyi IoU modeli: {best_iou_model_path}")
    LOGGER.info(f"Varsayılan en iyi model: {best_model_path}")
    LOGGER.info(f"Eğitim geçmişi: {history_path}")
    if epoch_checkpoint_dir is not None:
        LOGGER.info(f"Epoch modelleri: {epoch_checkpoint_dir}")
    LOGGER.info("=" * 60)

    if best_model_path is None:
        if best_iou_model_path is not None:
            best_model_path = best_iou_model_path
        elif best_loss_model_path is not None:
            best_model_path = best_loss_model_path
        else:
            raise RuntimeError("Hiç checkpoint kaydedilmedi; eğitim çıktıları oluşturulamadı.")

    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="12 kanallı arkeolojik tespit modeli eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Veri
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=str(CONFIG["data"]),
        help="Eğitim verisi dizini (egitim_verisi_olusturma.py çıktısı)",
    )
    
    # Model
    parser.add_argument(
        "--arch", "-a",
        type=str,
        default=str(CONFIG["arch"]),
        choices=["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN", "PSPNet", "MAnet", "Linknet"],
        help="Model mimarisi",
    )
    parser.add_argument(
        "--encoder", "-e",
        type=str,
        default=str(CONFIG["encoder"]),
        help="Encoder (resnet34, resnet50, efficientnet-b3, vb.)",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        default=bool(CONFIG["no_attention"]),
        help="CBAM Attention kullanma",
    )
    
    # Eğitim
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(CONFIG["epochs"]),
        help="Epoch sayısı",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=int(CONFIG["batch_size"]),
        help="Batch boyutu",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(CONFIG["lr"]),
        help="Learning rate",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=str(CONFIG["loss"]),
        choices=["bce", "dice", "combined", "focal"],
        help=(
            "Kayıp fonksiyonu. "
            "Class balance ayarları (balance_mode/pos_weight) sadece bce ve combined loss için etkilidir."
        ),
    )
    parser.add_argument(
        "--balance-mode",
        type=str,
        default=str(CONFIG["balance_mode"]),
        choices=["none", "auto", "manual"],
        help=(
            "Sınıf dengesizliği modu (yalnızca BCE/combined). "
            "none: kapalı, auto: train piksel oranından otomatik pos_weight, "
            "manual: --pos-weight değerini kullan."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=float(CONFIG["pos_weight"]),
        help=(
            "Manual modda pozitif sınıf ağırlığı (BCE/combined). "
            "1.0 ağırlıksız, daha yüksek değerler pozitif sınıf hatalarını daha çok cezalandırır."
        ),
    )
    parser.add_argument(
        "--max-auto-pos-weight",
        type=float,
        default=float(CONFIG["max_auto_pos_weight"]),
        help=(
            "Auto modda hesaplanan pos_weight için üst sınır (clip). "
            "Aşırı büyük ağırlıkların eğitimi dengesizleştirmesini engeller."
        ),
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=float(CONFIG["metric_threshold"]),
        help=(
            "Metrik hesaplama eşiği (IoU/F1/Precision/Recall için). "
            "0.5 klasik seçimdir; model düşük olasılık üretiyorsa daha düşük deneyin."
        ),
    )
    parser.add_argument(
        "--val-threshold-sweep",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["val_threshold_sweep"]),
        help=(
            "Doğrulamada birden fazla eşik tarayıp en iyi IoU eşiğini raporla."
        ),
    )
    parser.add_argument(
        "--val-threshold-min",
        type=float,
        default=float(CONFIG["val_threshold_min"]),
        help="Validation eşik taraması alt sınırı.",
    )
    parser.add_argument(
        "--val-threshold-max",
        type=float,
        default=float(CONFIG["val_threshold_max"]),
        help="Validation eşik taraması üst sınırı.",
    )
    parser.add_argument(
        "--val-threshold-step",
        type=float,
        default=float(CONFIG["val_threshold_step"]),
        help="Validation eşik taraması adımı.",
    )
    
    # Diğer
    parser.add_argument(
        "--patience",
        type=int,
        default=int(CONFIG["patience"]),
        help="Early stopping patience",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(CONFIG["workers"]),
        help="DataLoader worker sayısı",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=bool(CONFIG["no_amp"]),
        help="Mixed precision kullanma",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(CONFIG["output"]),
        help="Checkpoint dizini",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(CONFIG["seed"]),
        help="Random seed",
    )
    parser.add_argument(
        "--allow-all-negative",
        action="store_true",
        default=bool(CONFIG["allow_all_negative"]),
        help="Pozitif etiket içermeyen veri setleriyle eğitime devam et (önerilmez)",
    )
    
    parser.add_argument(
        "--train-neg-to-pos-ratio",
        type=float,
        default=(
            None
            if CONFIG["train_neg_to_pos_ratio"] is None
            else float(CONFIG["train_neg_to_pos_ratio"])
        ),
        help=(
            "Train setinde negatif tile alt-ornekleme oranı. "
            "1.0: negatif~=pozitif, 0.5: negatif~=pozitifin yarısı. "
            "Argüman verilmezse kapalı."
        ),
    )
    parser.add_argument(
        "--train-neg-sample-seed",
        type=int,
        default=int(CONFIG["train_neg_sample_seed"]),
        help="Train negatif alt-ornekleme rastgelelik tohumu.",
    )
    parser.add_argument(
        "--val-keep-ratio",
        type=float,
        default=float(CONFIG["val_keep_ratio"]),
        help="Val hedef oranı (train-secilene gore). 0.5 = val hedefi train-secilenin yarısı.",
    )
    parser.add_argument(
        "--val-sample-seed",
        type=int,
        default=int(CONFIG["val_sample_seed"]),
        help="Val alt-ornekleme rastgelelik tohumu.",
    )

    args = parser.parse_args()
    
    # Veri dizini kontrolü
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"HATA: Veri dizini bulunamadı: {data_dir}")
        print("\nÖnce egitim_verisi_olusturma.py ile veri oluşturun:")
        print(f"  python egitim_verisi_olusturma.py --input kesif_alani.tif --mask ground_truth.tif --output {data_dir}")
        sys.exit(1)
    
    if not (data_dir / "train" / "images").exists():
        print(f"HATA: Eğitim verisi bulunamadı: {data_dir / 'train' / 'images'}")
        print("\negitim_verisi_olusturma.py çıktısını kontrol edin.")
        sys.exit(1)
    
    # Metadata'dan kanal sayısını oku
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        in_channels = metadata.get("num_channels", 12)
        LOGGER.info(f"Metadata'dan kanal sayısı okundu: {in_channels}")
    else:
        in_channels = 12
        LOGGER.warning(f"Metadata bulunamadı, varsayılan kanal sayısı kullanılıyor: {in_channels}")

    # Etiket dağılımını doğrula (tamamı negatif veri sessizce eğitime girmesin)
    file_format = _infer_file_format(data_dir)
    train_total, train_positive = _count_positive_mask_files(data_dir / "train" / "masks", file_format)
    val_total, val_positive = _count_positive_mask_files(data_dir / "val" / "masks", file_format)
    LOGGER.info(
        "Maske dağılımı | train: %d/%d pozitif tile | val: %d/%d pozitif tile",
        train_positive, train_total, val_positive, val_total,
    )

    total_positive = train_positive + val_positive
    if total_positive == 0 and not args.allow_all_negative:
        print("HATA: Eğitim verisinde pozitif etiket bulunamadı (tüm maskeler 0).")
        print("Ground truth maskesini ve egitim_verisi_olusturma.py çıktısını kontrol edin.")
        print("Bilerek tamamen negatif veriyle eğitim yapmak isterseniz: --allow-all-negative")
        sys.exit(1)
    if total_positive == 0 and args.allow_all_negative:
        LOGGER.warning(
            "Pozitif etiket bulunamadı, ancak --allow-all-negative nedeniyle eğitim devam edecek."
        )

    train_total_pixels, train_positive_pixels = _count_positive_mask_pixels(
        data_dir / "train" / "masks",
        file_format,
    )
    train_positive_ratio = (
        train_positive_pixels / train_total_pixels if train_total_pixels > 0 else 0.0
    )
    LOGGER.info(
        "Train piksel dağılımı | pozitif: %d/%d (%.6f%%)",
        train_positive_pixels,
        train_total_pixels,
        100.0 * train_positive_ratio,
    )

    if args.max_auto_pos_weight <= 0:
        print("HATA: --max-auto-pos-weight 0'dan büyük olmalı.")
        sys.exit(1)

    if args.pos_weight <= 0:
        print("HATA: --pos-weight 0'dan büyük olmalı.")
        sys.exit(1)

    if not 0.0 < args.metric_threshold < 1.0:
        print("HATA: --metric-threshold 0 ile 1 arasında olmalı (uçlar hariç).")
        sys.exit(1)

    if args.val_threshold_step <= 0:
        print("HATA: --val-threshold-step 0'dan büyük olmalı.")
        sys.exit(1)

    if args.val_threshold_min < 0 or args.val_threshold_max > 1:
        print("HATA: --val-threshold-min/max 0-1 aralığında olmalı.")
        sys.exit(1)

    if args.train_neg_to_pos_ratio is not None and args.train_neg_to_pos_ratio < 0:
        print("HATA: --train-neg-to-pos-ratio None veya 0'dan büyük/eşit olmalı.")
        sys.exit(1)

    if args.train_neg_sample_seed < 0:
        print("HATA: --train-neg-sample-seed negatif olamaz.")
        sys.exit(1)

    if not 0.0 < args.val_keep_ratio <= 1.0:
        print("HATA: --val-keep-ratio 0-1 aralığında olmalı (0 hariç).")
        sys.exit(1)

    if args.val_sample_seed < 0:
        print("HATA: --val-sample-seed negatif olamaz.")
        sys.exit(1)

    resolved_pos_weight = 1.0
    if args.balance_mode == "manual":
        resolved_pos_weight = float(args.pos_weight)
        LOGGER.info("Manual class balance aktif: pos_weight=%.4f", resolved_pos_weight)
    elif args.balance_mode == "auto":
        if train_positive_pixels == 0:
            resolved_pos_weight = 1.0
            LOGGER.warning(
                "Auto class balance hesaplanamadı (train pozitif piksel yok), pos_weight=1.0 kullanılacak."
            )
        else:
            train_negative_pixels = train_total_pixels - train_positive_pixels
            # BCE pos_weight klasik yaklaşım: negatif/pozitif piksel oranı.
            # Pozitif sınıf azaldıkça ağırlık artar.
            raw_pos_weight = train_negative_pixels / train_positive_pixels
            resolved_pos_weight = float(
                np.clip(raw_pos_weight, 1.0, float(args.max_auto_pos_weight))
            )
            if raw_pos_weight != resolved_pos_weight:
                LOGGER.info(
                    "Auto class balance: ham pos_weight=%.4f, kırpılmış pos_weight=%.4f (max=%.4f)",
                    raw_pos_weight,
                    resolved_pos_weight,
                    float(args.max_auto_pos_weight),
                )
            else:
                LOGGER.info("Auto class balance: pos_weight=%.4f", resolved_pos_weight)
    else:
        LOGGER.info("Class balance kapalı: pos_weight=1.0")
    
    # Config oluştur
    config = TrainingConfig(
        data_dir=data_dir,
        arch=args.arch,
        encoder=args.encoder,
        in_channels=in_channels,
        enable_attention=not args.no_attention,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss,
        balance_mode=args.balance_mode,
        pos_weight=resolved_pos_weight,
        patience=args.patience,
        metric_threshold=float(args.metric_threshold),
        val_threshold_sweep=bool(args.val_threshold_sweep),
        val_threshold_min=float(args.val_threshold_min),
        val_threshold_max=float(args.val_threshold_max),
        val_threshold_step=float(args.val_threshold_step),
        num_workers=args.workers,
        use_amp=not args.no_amp,
        seed=args.seed,
        train_neg_to_pos_ratio=(
            None
            if args.train_neg_to_pos_ratio is None
            else float(args.train_neg_to_pos_ratio)
        ),
        train_neg_sample_seed=int(args.train_neg_sample_seed),
        val_keep_ratio=float(args.val_keep_ratio),
        val_sample_seed=int(args.val_sample_seed),
        output_dir=Path(args.output),
        save_every_epoch=bool(CONFIG["save_every_epoch"]),
        epoch_dir=str(CONFIG["epoch_dir"]),
    )
    
    # Eğitimi başlat
    try:
        best_model = train(config)
        
        print("\n" + "=" * 60)
        print("✓ EĞİTİM TAMAMLANDI!")
        print("=" * 60)
        print("\nEğitilmiş modeli kullanmak için config.yaml'da:")
        print(f"  weights: \"{best_model}\"")
        print("  zero_shot_imagenet: false")
        print("\nVeya komut satırından:")
        print(f"  python archaeo_detect.py --weights {best_model}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Eğitim kullanıcı tarafından durduruldu.")
    except Exception as e:
        LOGGER.error(f"Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
