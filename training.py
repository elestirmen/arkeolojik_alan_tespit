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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    # Egitim veri dizini (train/ ve val/ klasorlerini icermeli)
    "data": "training_data",
    # Segmentasyon modeli (SMP):
    # Unet | UnetPlusPlus | DeepLabV3Plus | FPN | PSPNet | MAnet | Linknet
    "arch": "Unet",
    # Encoder omurgasi (ornekler):
    # resnet34 | resnet50 | efficientnet-b3 | mobilenet_v2 | densenet121
    "encoder": "resnet50",
    # True ise CBAM attention varsayilan olarak kapatilir
    "no_attention": False,
    # Egitim suresi
    "epochs": 50,
    # Batch boyutu
    "batch_size": 8,
    # Ogrenme orani
    "lr": 1e-4,
    # Kayip fonksiyonu: bce | dice | combined | focal
    # Not: class balance ayarlari (balance_mode/pos_weight) sadece bce ve combined
    # kayiplarinda etkilidir.
    "loss": "combined",
    # Sinif dengesizligi ayari:
    # - none   : Agirliklandirma kapali (pos_weight=1.0 gibi davranir)
    # - auto   : Train maskelerinden piksel bazli negatif/pozitif orani hesaplanir
    #            ve bu orana gore otomatik pos_weight secilir
    # - manual : pos_weight degeri dogrudan kullanilir
    # Not: Yalnizca bce ve combined loss tiplerinde etkilidir.
    "balance_mode": "auto",
    # Manual modunda BCE positive class agirligi.
    # 1.0 = agirlik yok, >1.0 = pozitif sinif hatalari daha fazla cezalandirilir.
    # Cok dengesiz veri setlerinde 20-300 arasi degerler pratikte ise yarayabilir.
    "pos_weight": 1.0,
    # Auto modunda hesaplanan pos_weight icin ust sinir (clip).
    # Asiri buyuk agirliklarin egitimi kararsizlastirmasini engeller.
    "max_auto_pos_weight": 100.0,
    # Erken durdurma sabri
    "patience": 10,
    # DataLoader worker sayisi
    "workers": 4,
    # True ise AMP (mixed precision) varsayilan olarak kapatilir
    "no_amp": False,
    # Checkpoint cikti dizini
    "output": "checkpoints",
    # Her epoch sonunda da checkpoint kaydet
    "save_every_epoch": True,
    # Epoch checkpoint'lerinin alt dizini
    "epoch_dir": "epochs",
    # Rastgelelik sabiti
    "seed": 42,
    # Tum maskeler negatif olsa da egitime devam et
    "allow_all_negative": False,
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
        
        # Dosyaları oku
        if self.file_format == "npy":
            image = np.load(img_path)  # (C, H, W)
            mask = np.load(mask_path)  # (H, W)
        else:  # npz
            image = np.load(img_path)["image"]
            mask = np.load(mask_path)["mask"]
        
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
    
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # True/False Positives/Negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Metrikler
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
    balance_mode: str = "none"  # "none", "auto", "manual"
    pos_weight: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "plateau", "step"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Diğer
    num_workers: int = 4
    use_amp: bool = True  # Mixed precision
    seed: int = 42
    
    # Çıktı
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every_epoch: bool = True
    epoch_dir: str = "epochs"


def _infer_file_format(data_dir: Path) -> str:
    """Eğitim verisindeki dosya formatını (npz/npy) tespit eder."""
    train_images = data_dir / "train" / "images"
    if any(train_images.glob("*.npz")):
        return "npz"
    if any(train_images.glob("*.npy")):
        return "npy"
    raise ValueError(f"Desteklenen eğitim dosyası bulunamadı: {train_images} (*.npz / *.npy)")


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
) -> Tuple[float, Dict[str, float]]:
    """Bir epoch eğitim yapar."""
    
    model.train()
    total_loss = 0.0
    all_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "accuracy": 0}
    
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
        
        # Metrikler
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k] += v
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    n = len(train_loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Doğrulama yapar."""
    
    model.eval()
    total_loss = 0.0
    all_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "accuracy": 0}
    
    for images, masks in tqdm(val_loader, desc="Doğrulama", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        
        metrics = calculate_metrics(outputs, masks)
        for k, v in metrics.items():
            all_metrics[k] += v
    
    n = len(val_loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


def train(config: TrainingConfig) -> Path:
    """
    Model eğitimi ana fonksiyonu.
    
    Returns:
        En iyi model checkpoint'unun yolu
    """
    
    # Seed ayarla
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    LOGGER.info(f"Eğitim: {len(train_dataset)} örnek, Doğrulama: {len(val_dataset)} örnek")
    
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
        "lr": [],
    }
    
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
    LOGGER.info(f"Epochs: {config.epochs}")
    LOGGER.info(f"Batch size: {config.batch_size}")
    LOGGER.info(f"Learning rate: {config.lr}")
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
            model, train_loader, criterion, optimizer, device, scaler, config.use_amp
        )
        
        # Doğrulama
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
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
        history["val_iou"].append(val_metrics["iou"])
        history["lr"].append(current_lr)
        
        # Log
        epoch_time = time.time() - epoch_start
        LOGGER.info(
            f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Süre: {epoch_time:.1f}s"
        )
        
        # En iyi model kontrolü
        val_iou = float(val_metrics["iou"])
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
            "val_iou": val_iou,
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
            },
        }
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
                iou_checkpoint["best_metric"] = "val_iou"
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
    LOGGER.info(f"En iyi Val IoU: {best_val_iou:.4f}")
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
        num_workers=args.workers,
        use_amp=not args.no_amp,
        seed=args.seed,
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
