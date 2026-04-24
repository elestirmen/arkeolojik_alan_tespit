#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arkeolojik Alan Tespiti - 5 Kanallı U-Net Eğitim Scripti

Bu script, egitim_verisi_olusturma.py ile oluşturulan 5 kanallı tile'ları
kullanarak U-Net modelini eğitir.

Özellikler:
    - 5 kanallı girdi desteği (RGB + SVF + SLRM)
    - CBAM Attention modülü (opsiyonel)
    - Mixed precision training (AMP)
    - Erken durdurma (Early stopping)
    - Model checkpoint'leri
    - Detaylı eğitim logları
    - TensorBoard desteği (opsiyonel)

Kullanım:
    python training.py --data workspace/training_data --epochs 50 --encoder resnet34
    
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
import csv
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from archeo_shared.console import configure_utf8_console

configure_utf8_console()

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from archeo_shared.channels import METADATA_SCHEMA_VERSION, MODEL_CHANNEL_NAMES
from archeo_shared.modeling import AttentionWrapper, CBAM, ChannelAttention, SpatialAttention

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
    "data": "workspace/training_data_topo5_merged",

    # task:
    # segmentation       : Piksel maskesi ogrenilir
    # tile_classification: Her tile icin tek skor/etiket ogrenilir
    "task": "tile_classification",

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
    "no_attention": True,

    # epochs:
    # Toplam egitim epoch sayisi.
    "epochs": 50,

    # batch_size:
    # Batch boyutu.
    # Yuksek deger daha stabil gradient verebilir; daha fazla VRAM ister.
    "batch_size": 32,

    # lr:
    # Ogrenme orani (AdamW).
    # Cok yuksek olursa kararsizlik, cok dusuk olursa yavas ogrenme gorulebilir.
    "lr": 2e-4,

    # loss:
    # Optimize edilen kayip fonksiyonu.
    # bce      : Piksel/batch bazli stabil temel kayip. Tile classification icin
    #            en guvenli baslangic secenegidir. balance_mode/pos_weight ile calisir.
    # dice     : Tahmin-mask overlap'ini dogrudan iter. Kucuk/seyrek pozitif
    #            maskelerde segmentation icin faydali olabilir. Tile classification'ta desteklenmez.
    # combined : BCE + Dice. Hem stabil optimizasyon hem de daha iyi overlap dengesi
    #            verir; segmentation icin genelde en pratik secenektir.
    # focal    : Kolay negatifleri bastirip zor orneklere daha cok odaklanir.
    #            Kuvvetli sinif dengesizliginde ozellikle tile classification'ta yararlidir.
    # Not: balance_mode/pos_weight ayarlari sadece bce ve combined icin etkilidir.
    "loss": "bce",

    # balance_mode:
    # Sinif dengesizligi modu (BCE tarafinda agirliklandirma davranisi):
    # - none   : Agirliklandirma kapali (pos_weight=1.0 gibi davranir)
    # - auto   : Train maskelerinden piksel bazli negatif/pozitif orani hesaplanir
    #            ve bu orana gore otomatik pos_weight secilir
    # - manual : pos_weight degeri dogrudan kullanilir
    # Not: Yalnizca bce/combined icin etkilidir.
    "balance_mode": "auto",

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
    "patience": 20,

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

    # monitor_channel_importance:
    # Egitim sirasinda kanal (band) onemini olc ve raporla.
    "monitor_channel_importance": True,

    # channel_importance_max_batches:
    # Gradient fallback modunda kac batch uzerinden ortalama alinacagi.
    # 0: tum batch'ler.
    "channel_importance_max_batches": 12,

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
    "output": "workspace/checkpoints",

    # publish_active:
    # True ise egitim sonunda aktif IDE modeli sabit bir klasore kopyalanir.
    "publish_active": True,

    # active_dir:
    # IDE/inference tarafinin okuyacagi aktif model klasoru.
    "active_dir": "workspace/checkpoints/active",

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
    "train_neg_to_pos_ratio": None,

    # train_neg_sample_seed:
    # Oran bazli negatif alt-orneklemede rastgelelik tohumu.
    "train_neg_sample_seed": 42,

    # val_keep_ratio:
    # Val hedef boyutu train-secilen ornek sayisina gore hesaplanir:
    # hedef_val ~= round(train_secilen * val_keep_ratio)
    # 1.0: val hedefi train-secilen kadar
    # 0.5: val hedefi train-secilenin yarisi
    # Not: hedef, val toplamindan buyuk olamaz.c
    "val_keep_ratio": 1.0,

    # val_sample_seed:
    # Val alt-ornekleme rastgelelik tohumu.
    "val_sample_seed": 42,

    # tile_label_min_positive_ratio:
    # tile_classification modunda bir tile'in pozitif sayilmasi icin gereken
    # minimum pozitif piksel orani. 0.0 => maskede en az bir pozitif piksel
    # varsa tile pozitiftir.
    "tile_label_min_positive_ratio": 0.02,

    # deterministic_rotate_step_deg:
    # Train augment icin deterministik donme aci adimi.
    # 0.0: kapali, 30.0: [0,30,...,330] olacak sekilde 12x gorunum.
    "deterministic_rotate_step_deg": 30.0,

    # scale_augment:
    # Egitim sirasinda rastgele olcek degisikligi (zoom in/out) uygula.
    # Model farkli boyutlardaki yapilari tanir hale gelir.
    "scale_augment": True,

    # scale_augment_range:
    # Olcek araliginin alt ve ust siniri.
    # 0.7 = %30 zoom out, 1.3 = %30 zoom in.
    "scale_augment_min": 0.7,
    "scale_augment_max": 1.3,

    # use_fpn_classifier:
    # True ise TileClassifier encoder'in tum katmanlarindan feature toplar (FPN-style).
    # False ise sadece son katmani kullanir (eski davranis).
    "use_fpn_classifier": True,
}
# ===============================================


# ==============================================================================
# DATASET
# ==============================================================================

def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Maskeyi legacy/veri tipi farklarina karsi 0/1 float32 formatina getir."""
    return np.where(np.isfinite(mask) & (mask > 0), 1.0, 0.0).astype(np.float32)


def _mask_positive_ratio(mask: np.ndarray) -> float:
    """Bir maskede pozitif piksel oranini hesapla."""
    mask_bin = _mask_to_binary(mask)
    total_pixels = int(mask_bin.size)
    if total_pixels <= 0:
        return 0.0
    positive_pixels = int(np.count_nonzero(mask_bin))
    return float(positive_pixels / total_pixels)


def _mask_to_tile_label(
    mask: np.ndarray,
    min_positive_ratio: float = 0.0,
) -> float:
    """
    Piksel maskesini tek bir tile etiketine indirger.

    `min_positive_ratio=0.0` iken "maskede en az bir pozitif piksel varsa pozitif"
    mantigi kullanilir.
    """
    positive_ratio = _mask_positive_ratio(mask)
    if positive_ratio <= 0.0:
        return 0.0
    threshold = float(min_positive_ratio)
    if threshold <= 0.0:
        return 1.0
    return 1.0 if positive_ratio >= threshold else 0.0


CLASS_LABEL_NEGATIVE = "Negative"
CLASS_LABEL_POSITIVE = "Positive"
CLASS_LABELS: Tuple[str, str] = (
    CLASS_LABEL_NEGATIVE,
    CLASS_LABEL_POSITIVE,
)


def _resolve_class_dir(split_dir: Path, label_name: str) -> Path:
    candidate = split_dir / label_name
    if candidate.exists():
        return candidate
    if not split_dir.exists():
        return candidate
    wanted = str(label_name).strip().lower()
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.strip().lower() == wanted:
            return child
    return candidate


def _has_classification_folder_layout(split_dir: Path) -> bool:
    return any(_resolve_class_dir(split_dir, label).exists() for label in CLASS_LABELS)


def _detect_split_layout(
    split_dir: Path,
    task_type: Optional[str] = None,
) -> str:
    paired = (split_dir / "images").exists() and (split_dir / "masks").exists()
    class_layout = _has_classification_folder_layout(split_dir)

    if paired and class_layout:
        raise ValueError(
            f"Ayni split icinde hem images/masks hem Positive/Negative duzeni bulundu: {split_dir}"
        )
    if paired:
        return "paired"
    if class_layout:
        normalized_task = None if task_type is None else str(task_type).strip().lower()
        if normalized_task == "segmentation":
            raise ValueError(
                "Positive/Negative klasor duzeni yalnizca tile_classification icin desteklenir."
            )
        return "classification_folders"
    raise ValueError(
        "Desteklenen split duzeni bulunamadi. Beklenen yapilardan biri:\n"
        f"- {split_dir / 'images'} + {split_dir / 'masks'}\n"
        f"- {_resolve_class_dir(split_dir, CLASS_LABEL_POSITIVE)} / {_resolve_class_dir(split_dir, CLASS_LABEL_NEGATIVE)}"
    )


def _detect_dataset_layout(
    data_dir: Path,
    task_type: Optional[str] = None,
) -> str:
    train_split = Path(data_dir) / "train"
    val_split = Path(data_dir) / "val"

    train_layout = _detect_split_layout(train_split, task_type)
    val_layout = _detect_split_layout(val_split, task_type)
    if train_layout != val_layout:
        raise ValueError(
            "Train ve val splitleri farkli veri duzeni kullaniyor "
            f"(train={train_layout}, val={val_layout})."
        )
    return train_layout


def _detect_file_format_in_dirs(directories: Sequence[Path]) -> Optional[str]:
    found_formats: set[str] = set()
    checked_dirs: List[Path] = []
    for directory in directories:
        checked_dirs.append(directory)
        has_npz = directory.exists() and any(directory.glob("*.npz"))
        has_npy = directory.exists() and any(directory.glob("*.npy"))
        if has_npz and has_npy:
            raise ValueError(
                f"Ayni klasorde hem .npz hem .npy bulundu: {directory}. "
                "Tek bir format kullanin."
            )
        if has_npz:
            found_formats.add("npz")
        if has_npy:
            found_formats.add("npy")
    if len(found_formats) > 1:
        raise ValueError(
            "Farkli klasorlerde farkli tile formatlari bulundu: "
            f"{sorted(found_formats)} | dizinler={checked_dirs}"
        )
    if not found_formats:
        return None
    return next(iter(found_formats))


def _count_positive_tiles_from_class_dirs(
    split_dir: Path,
    file_format: str,
) -> Tuple[int, int]:
    positive_dir = _resolve_class_dir(split_dir, CLASS_LABEL_POSITIVE)
    negative_dir = _resolve_class_dir(split_dir, CLASS_LABEL_NEGATIVE)
    positive = len(list(positive_dir.glob(f"*.{file_format}"))) if positive_dir.exists() else 0
    negative = len(list(negative_dir.glob(f"*.{file_format}"))) if negative_dir.exists() else 0
    return positive + negative, positive

class ArchaeologyDataset(Dataset):
    """
    5 kanallı arkeolojik alan tespiti veri seti.
    
    egitim_verisi_olusturma.py tarafından oluşturulan .npy dosyalarını okur.
    """
    
    def __init__(
        self,
        data_dir: Path,
        augment: bool = True,
        file_format: str = "npz",
        task_type: str = "segmentation",
        tile_label_min_positive_ratio: float = 0.0,
        deterministic_rotate_step_deg: float = 0.0,
        scale_augment: bool = False,
        scale_augment_min: float = 0.7,
        scale_augment_max: float = 1.3,
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
        self.task_type = str(task_type).strip().lower()
        self.tile_label_min_positive_ratio = float(tile_label_min_positive_ratio)
        self.scale_augment = bool(scale_augment) and self.augment
        self.scale_augment_min = float(scale_augment_min)
        self.scale_augment_max = float(scale_augment_max)
        self.deterministic_rotate_step_deg = float(deterministic_rotate_step_deg)
        if not 0.0 <= self.deterministic_rotate_step_deg < 360.0:
            raise ValueError(
                "deterministic_rotate_step_deg 0-360 araliginda olmali "
                f"(360 haric), verilen: {deterministic_rotate_step_deg}"
            )
        if self.task_type not in {"segmentation", "tile_classification"}:
            raise ValueError(f"Desteklenmeyen task_type: {task_type}")
        self.tile_labels: Optional[List[float]] = None
        self.dataset_layout = _detect_split_layout(self.data_dir, self.task_type)
        self.class_dirs: Dict[str, Path] = {
            label: _resolve_class_dir(self.data_dir, label)
            for label in CLASS_LABELS
        }
        self._use_deterministic_rotation = bool(
            self.augment and self.deterministic_rotate_step_deg > 0.0
        )
        self._rotation_angles: Tuple[float, ...] = (
            self._build_rotation_angles(self.deterministic_rotate_step_deg)
            if self._use_deterministic_rotation
            else (0.0,)
        )
        self._augmentation_multiplier = max(1, len(self._rotation_angles))
        if self._use_deterministic_rotation:
            rotation_count = 360.0 / self.deterministic_rotate_step_deg
            if not np.isclose(rotation_count, round(rotation_count), atol=1e-6):
                LOGGER.warning(
                    "Deterministik rotasyon adimi 360'i tam bolmuyor (step=%.4f). "
                    "Aci listesi 360'tan once sonlanacak.",
                    self.deterministic_rotate_step_deg,
                )
        
        # Dosyaları listele
        pattern = f"*.{file_format}"
        if self.dataset_layout == "classification_folders":
            if self.task_type != "tile_classification":
                raise ValueError(
                    "Positive/Negative klasor düzeni yalnızca tile_classification için kullanılabilir."
                )
            positive_files = (
                sorted(self.class_dirs[CLASS_LABEL_POSITIVE].glob(pattern))
                if self.class_dirs[CLASS_LABEL_POSITIVE].exists()
                else []
            )
            negative_files = (
                sorted(self.class_dirs[CLASS_LABEL_NEGATIVE].glob(pattern))
                if self.class_dirs[CLASS_LABEL_NEGATIVE].exists()
                else []
            )
            self.image_files = positive_files + negative_files
            if len(self.image_files) == 0:
                raise ValueError(
                    "Hiç tile bulunamadı. Beklenen klasörler: "
                    f"{self.class_dirs[CLASS_LABEL_POSITIVE]} / "
                    f"{self.class_dirs[CLASS_LABEL_NEGATIVE]}"
                )
            self.tile_labels = [1.0] * len(positive_files) + [0.0] * len(negative_files)
        else:
            self.image_files = sorted(self.images_dir.glob(pattern))
            if len(self.image_files) == 0:
                raise ValueError(f"Hiç görüntü dosyası bulunamadı: {self.images_dir}")

        if self.task_type == "tile_classification" and self.dataset_layout == "paired":
            manifest_labels = _load_tile_label_manifest(self.data_dir)
            labels: List[float] = []
            missing_manifest_label = False
            for img_path in self.image_files:
                tile_name = img_path.stem
                manifest_label = None if manifest_labels is None else manifest_labels.get(tile_name)
                if manifest_label is not None:
                    labels.append(float(manifest_label))
                    continue

                missing_manifest_label = True
                mask_path = self.masks_dir / img_path.name
                if not mask_path.exists():
                    raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")
                if self.file_format == "npy":
                    mask = np.load(mask_path)
                else:
                    with np.load(mask_path) as packed:
                        if "mask" not in packed:
                            raise ValueError(f"'mask' anahtari bulunamadi: {mask_path}")
                        mask = packed["mask"]
                labels.append(
                    _mask_to_tile_label(
                        mask,
                        min_positive_ratio=self.tile_label_min_positive_ratio,
                    )
                )
            self.tile_labels = labels
            if manifest_labels is not None and not missing_manifest_label:
                LOGGER.info(
                    "Tile label manifesti kullaniliyor: %s",
                    self.data_dir.parent / "tile_labels.csv",
                )

        base_count = len(self.image_files)
        total_count = len(self)
        if self._use_deterministic_rotation and self._augmentation_multiplier > 1:
            LOGGER.info(
                "Dataset yuklendi: %d temel ornek, %d aci ile %d ornek (%s, layout=%s)",
                base_count,
                self._augmentation_multiplier,
                total_count,
                data_dir,
                self.dataset_layout,
            )
        else:
            LOGGER.info(
                "Dataset yuklendi: %d ornek (%s, layout=%s)",
                total_count,
                data_dir,
                self.dataset_layout,
            )
    
    def __len__(self) -> int:
        return len(self.image_files) * self._augmentation_multiplier

    @staticmethod
    def _build_rotation_angles(step_deg: float) -> Tuple[float, ...]:
        if step_deg <= 0.0:
            return (0.0,)
        angles = np.arange(0.0, 360.0, step_deg, dtype=np.float32)
        if angles.size == 0:
            return (0.0,)
        return tuple(float(angle) for angle in angles.tolist())

    def _resolve_sample_index(self, idx: int) -> Tuple[int, float]:
        total_samples = len(self)
        if idx < 0 or idx >= total_samples:
            raise IndexError(f"Dataset index out of range: {idx}")

        if self._augmentation_multiplier <= 1:
            return idx, 0.0

        base_idx = idx // self._augmentation_multiplier
        angle_idx = idx % self._augmentation_multiplier
        return base_idx, self._rotation_angles[angle_idx]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx, rotation_angle = self._resolve_sample_index(idx)
        img_path = self.image_files[base_idx]

        if self.file_format == "npy":
            image = np.load(img_path)  # (C, H, W)
        else:  # npz
            image = np.load(img_path)["image"]

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.task_type == "tile_classification":
            if self.augment:
                if self._use_deterministic_rotation:
                    image = self._apply_rotation_image(image, rotation_angle)
                    image = self._apply_random_flips_image(image)
                else:
                    image = self._augment_image(image)
            image = torch.from_numpy(image.copy()).float()
            if self.tile_labels is None:
                raise RuntimeError("tile_labels hazir degil")
            label = np.array([self.tile_labels[base_idx]], dtype=np.float32)
            label = torch.from_numpy(label).float()
            return image, label

        mask_path = self.masks_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")

        if self.file_format == "npy":
            mask = np.load(mask_path)  # (H, W)
        else:  # npz
            mask = np.load(mask_path)["mask"]

        mask = _mask_to_binary(mask)
        
        # Veri artırma
        if self.augment:
            if self._use_deterministic_rotation:
                image, mask = self._apply_rotation_pair(image, mask, rotation_angle)
                image, mask = self._apply_random_flips_pair(image, mask)
            else:
                image, mask = self._augment(image, mask)
        
        # Tensor'a çevir
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        
        return image, mask

    def _apply_rotation_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate CHW image with fixed output shape."""
        if abs(float(angle_deg)) < 1e-8:
            return image

        rotated = ndimage.rotate(
            image,
            angle=float(angle_deg),
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return np.asarray(rotated, dtype=np.float32)

    def _apply_rotation_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image+mask pair (bilinear image / nearest-neighbor mask)."""
        if abs(float(angle_deg)) < 1e-8:
            return image, mask

        rotated_image = self._apply_rotation_image(image, angle_deg)
        rotated_mask = ndimage.rotate(
            mask,
            angle=float(angle_deg),
            reshape=False,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return rotated_image, _mask_to_binary(rotated_mask)

    def _apply_random_flips_image(self, image: np.ndarray) -> np.ndarray:
        """Deterministic rotation ile birlikte rastgele yatay/dikey flip uygula."""

        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()

        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()

        return image

    def _apply_random_flips_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic rotation ile birlikte image+mask flip uygula."""

        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        return image, mask

    @staticmethod
    def _scale_augment_image(image: np.ndarray, scale: float) -> np.ndarray:
        """Rastgele zoom in/out uygula. image: (C, H, W)."""
        if abs(scale - 1.0) < 1e-4:
            return image
        C, H, W = image.shape
        new_h, new_w = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        zoom_factors = (1.0, new_h / H, new_w / W)
        scaled = ndimage.zoom(image, zoom_factors, order=1, mode="nearest")
        sC, sH, sW = scaled.shape
        out = np.zeros_like(image)
        if sH >= H and sW >= W:
            r0 = (sH - H) // 2
            c0 = (sW - W) // 2
            out = scaled[:, r0:r0 + H, c0:c0 + W].copy()
        else:
            r0 = (H - sH) // 2
            c0 = (W - sW) // 2
            out[:, r0:r0 + sH, c0:c0 + sW] = scaled
        return out.astype(np.float32)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Tile-level classification icin goruntu augmentasyonu."""

        if self.scale_augment:
            s = np.random.uniform(self.scale_augment_min, self.scale_augment_max)
            image = self._scale_augment_image(image, s)

        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()

        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()

        if np.random.random() > 0.25:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()

        return image

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
    # Overlap'i dogrudan optimize eder.
    # Pozitif alan cok kucukse segmentation tarafinda BCE'ye gore daha faydali olabilir.
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
    # En temel ve genelde en stabil binary kayip secenegi.
    # pos_weight ile pozitif sinif daha agir cezalandirilabilir.
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
    # BCE'nin stabilitesini ve Dice'in overlap odagini birlestirir.
    # Segmentasyon icin pratik ve dengeli bir varsayilan alternatiftir.
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
    # Kolay negatiflerin etkisini azaltir, zor orneklere odaklanir.
    # Kuvvetli sinif dengesizliginde ozellikle tile classification'ta yararli olabilir.
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


def _resolve_channel_names(channel_names: Sequence[str], in_channels: int) -> List[str]:
    """Return a fixed-length, human-readable channel name list."""
    names = [str(name) for name in channel_names]
    if len(names) < in_channels:
        names.extend(f"ch_{idx+1}" for idx in range(len(names), in_channels))
    return names[:in_channels]


def _find_first_channel_attention_module(model: nn.Module) -> Optional[ChannelAttention]:
    """Find first ChannelAttention module in model tree."""
    for module in model.modules():
        if isinstance(module, ChannelAttention):
            return module
    return None


class ChannelImportanceTracker:
    """
    Track per-channel importance during training.

    Priority:
    1) CBAM ChannelAttention output (if present)
    2) Gradient*input fallback
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        in_channels: int,
        channel_names: Sequence[str],
        enabled: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.in_channels = int(in_channels)
        self.channel_names = _resolve_channel_names(channel_names, self.in_channels)
        self.mode = "disabled"
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._sum = np.zeros(self.in_channels, dtype=np.float64)
        self._count = 0

        if not self.enabled:
            return

        channel_attention = _find_first_channel_attention_module(model)
        if channel_attention is not None:
            self.mode = "attention"
            self._hook_handle = channel_attention.register_forward_hook(self._forward_hook)
            LOGGER.info("Band onemi monitoru: CBAM ChannelAttention modu aktif.")
        else:
            self.mode = "gradient"
            LOGGER.info("Band onemi monitoru: CBAM bulunamadi, gradient fallback aktif.")

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self) -> None:
        self._sum.fill(0.0)
        self._count = 0

    def is_gradient_mode(self) -> bool:
        return self.enabled and self.mode == "gradient"

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> None:
        if not self.enabled or self.mode != "attention":
            return
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            return
        if not (
            hasattr(module, "avg_pool")
            and hasattr(module, "max_pool")
            and hasattr(module, "fc")
        ):
            return

        with torch.no_grad():
            avg_out = module.fc(module.avg_pool(x))
            max_out = module.fc(module.max_pool(x))
            attention = torch.sigmoid(avg_out + max_out)
            scores = (
                attention.detach().float().mean(dim=(0, 2, 3)).cpu().numpy()
            )
        self._accumulate(scores)

    def accumulate_gradient(self, images: torch.Tensor, scale: float = 1.0) -> None:
        if not self.enabled or self.mode != "gradient":
            return
        grad = images.grad
        if grad is None:
            return
        with torch.no_grad():
            scores = (
                grad.detach().float().abs() * images.detach().float().abs()
            ).mean(dim=(0, 2, 3)).cpu().numpy()
        if np.isfinite(scale) and scale > 0:
            scores = scores / float(scale)
        self._accumulate(scores)

    def _accumulate(self, scores: np.ndarray) -> None:
        if scores.ndim != 1:
            return
        if scores.shape[0] != self.in_channels:
            min_len = min(scores.shape[0], self.in_channels)
            self._sum[:min_len] += scores[:min_len].astype(np.float64)
        else:
            self._sum += scores.astype(np.float64)
        self._count += 1

    def summarize(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._count <= 0:
            return None

        avg_scores = (self._sum / float(self._count)).astype(np.float64)
        if self.mode == "gradient":
            denom = float(np.sum(avg_scores))
            if denom > 0:
                avg_scores = avg_scores / denom

        ranking = sorted(
            [
                {"channel": name, "score": float(score)}
                for name, score in zip(self.channel_names, avg_scores)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )
        top = ranking[0] if ranking else {"channel": "n/a", "score": 0.0}
        return {
            "mode": self.mode,
            "samples": int(self._count),
            "scores": {item["channel"]: float(item["score"]) for item in ranking},
            "ranking": ranking,
            "top_channel": str(top["channel"]),
            "top_score": float(top["score"]),
        }


# ==============================================================================
# EĞİTİM FONKSİYONLARI
# ==============================================================================

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu."""
    
    # Veri
    data_dir: Path = field(default_factory=lambda: Path("workspace/training_data"))
    task_type: str = "segmentation"
    tile_label_min_positive_ratio: float = 0.0
    # Model
    arch: str = "Unet"
    encoder: str = "resnet34"
    in_channels: int = 5
    channel_names: Tuple[str, ...] = field(default_factory=lambda: MODEL_CHANNEL_NAMES)
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

    # Band importance monitoring
    monitor_channel_importance: bool = True
    channel_importance_max_batches: int = 12
    
    # Diğer
    num_workers: int = 4
    use_amp: bool = True  # Mixed precision
    seed: int = 42
    train_neg_to_pos_ratio: Optional[float] = None
    train_neg_sample_seed: int = 42
    val_keep_ratio: float = 1.0
    val_sample_seed: int = 42
    auto_val_from_train: bool = False
    auto_val_ratio: float = 0.2
    auto_val_reason: str = ""
    deterministic_rotate_step_deg: float = 0.0

    # Scale augmentation
    scale_augment: bool = True
    scale_augment_min: float = 0.7
    scale_augment_max: float = 1.3

    # FPN-style multi-level classifier
    use_fpn_classifier: bool = True

    # Çıktı
    output_dir: Path = field(default_factory=lambda: Path("workspace/checkpoints"))
    save_every_epoch: bool = True
    epoch_dir: str = "epochs"
    publish_active: bool = True
    active_dir: Path = field(default_factory=lambda: Path("workspace/checkpoints") / "active")
    source_metadata_path: Optional[Path] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)


def _build_training_metadata_payload(config: TrainingConfig) -> Dict[str, Any]:
    """Merge dataset metadata with the training config used for the checkpoint."""
    payload = dict(config.source_metadata)
    payload.update(
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "task_type": str(config.task_type).strip().lower(),
            "arch": _resolved_model_arch_name(config),
            "encoder": str(config.encoder),
            "in_channels": int(config.in_channels),
            "channel_names": list(config.channel_names),
            "data_dir": str(config.data_dir),
            "auto_val_from_train": bool(config.auto_val_from_train),
            "auto_val_ratio": float(config.auto_val_ratio),
            "auto_val_reason": str(config.auto_val_reason),
        }
    )
    return payload


def _publish_active_artifacts(
    *, config: TrainingConfig, best_model_path: Path
) -> Dict[str, Path]:
    """Publish the latest trained model and metadata for IDE-first inference."""
    active_dir = Path(config.active_dir)
    active_dir.mkdir(parents=True, exist_ok=True)

    model_target = active_dir / "model.pth"
    shutil.copy2(best_model_path, model_target)

    metadata_target = active_dir / "training_metadata.json"
    metadata_payload = _build_training_metadata_payload(config)
    metadata_target.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_target = active_dir / "published_from.json"
    manifest = {
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_checkpoint": str(best_model_path),
        "source_training_metadata": (
            str(config.source_metadata_path) if config.source_metadata_path else None
        ),
        "task_type": str(config.task_type).strip().lower(),
        "arch": _resolved_model_arch_name(config),
        "encoder": str(config.encoder),
        "in_channels": int(config.in_channels),
        "channel_names": list(config.channel_names),
        "tile_size": metadata_payload.get("tile_size"),
        "overlap": metadata_payload.get("overlap"),
        "bands": metadata_payload.get("bands"),
        "data_dir": str(config.data_dir),
    }
    manifest_target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "model": model_target,
        "training_metadata": metadata_target,
        "manifest": manifest_target,
    }


def _detect_split_file_format(images_dir: Path) -> Optional[str]:
    """Detect file format for a split directory and reject mixed extensions."""
    return _detect_file_format_in_dirs([images_dir])


def _infer_file_format(data_dir: Path, *, allow_missing_val: bool = False) -> str:
    """Egitim verisindeki dosya formatini (npz/npy) tespit eder."""
    layout = _detect_dataset_layout(data_dir)
    if layout == "classification_folders":
        train_dirs = [_resolve_class_dir(data_dir / "train", label) for label in CLASS_LABELS]
        val_dirs = [_resolve_class_dir(data_dir / "val", label) for label in CLASS_LABELS]
        train_format = _detect_file_format_in_dirs(train_dirs)
        val_format = _detect_file_format_in_dirs(val_dirs)
        train_target = "train/Positive|Negative"
        val_target = "val/Positive|Negative"
    else:
        train_images = data_dir / "train" / "images"
        val_images = data_dir / "val" / "images"
        train_format = _detect_split_file_format(train_images)
        val_format = _detect_split_file_format(val_images)
        train_target = str(train_images)
        val_target = str(val_images)

    if train_format is None:
        raise ValueError(
            f"Desteklenen egitim dosyasi bulunamadi: {train_target} (*.npz / *.npy)"
        )
    if val_format is None and not allow_missing_val:
        raise ValueError(
            f"Desteklenen dogrulama dosyasi bulunamadi: {val_target} (*.npz / *.npy)"
        )
    if val_format is not None and train_format != val_format:
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


def _load_tile_label_manifest(data_dir: Path) -> Optional[Dict[str, float]]:
    """Kok dizindeki tile_labels.csv manifestinden split'e ait label'lari oku."""
    root_dir = data_dir.parent
    manifest_path = root_dir / "tile_labels.csv"
    split_name = data_dir.name.strip().lower()
    if not manifest_path.exists():
        return None

    labels: Dict[str, float] = {}
    with open(manifest_path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row_split = str(row.get("split", "")).strip().lower()
            tile_name = str(row.get("tile_name", "")).strip()
            if row_split != split_name or not tile_name:
                continue
            try:
                labels[tile_name] = float(row.get("tile_label", "0"))
            except (TypeError, ValueError):
                continue

    return labels or None


def _count_positive_tile_labels_from_manifest(data_dir: Path) -> Optional[Tuple[int, int]]:
    """Varsa manifestten toplam ve pozitif tile sayisini dondur."""
    labels = _load_tile_label_manifest(data_dir)
    if labels is None:
        return None
    total = len(labels)
    positive = sum(1 for value in labels.values() if float(value) > 0.5)
    return total, int(positive)


def _resolve_classification_folder_split_counts(
    split_dir: Path,
    file_format: str,
) -> Tuple[Tuple[int, int], str]:
    """Classification split count'larinda fiziksel klasorleri kaynak dogrulama olarak kullan."""
    manifest_counts = _count_positive_tile_labels_from_manifest(split_dir)
    dir_counts = _count_positive_tiles_from_class_dirs(split_dir, file_format)
    if manifest_counts is not None and manifest_counts != dir_counts:
        LOGGER.warning(
            "tile_labels.csv ile %s splitindeki Positive/Negative klasor sayilari uyusmuyor "
            "(manifest: %d/%d pozitif, klasor: %d/%d pozitif). Fiziksel klasor sayilari kullanilacak.",
            split_dir.name,
            int(manifest_counts[1]),
            int(manifest_counts[0]),
            int(dir_counts[1]),
            int(dir_counts[0]),
        )
        return dir_counts, "class_dirs"
    if manifest_counts is not None:
        return manifest_counts, "manifest"
    return dir_counts, "class_dirs"


def _count_positive_mask_files(
    mask_dir: Path,
    file_format: str,
    tile_label_min_positive_ratio: float = 0.0,
) -> Tuple[int, int]:
    """
    Maske dosyalarında en az bir pozitif piksel içeren tile sayısını döndür.

    Returns:
        (toplam_tile_sayisi, pozitif_tile_sayisi)
    """
    mask_files = sorted(mask_dir.glob(f"*.{file_format}"))
    positive_files = 0

    for mask_path in mask_files:
        mask = _load_mask_array(mask_path, file_format)

        if _mask_to_tile_label(
            mask,
            min_positive_ratio=tile_label_min_positive_ratio,
        ) > 0.5:
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
        mask = _mask_to_binary(_load_mask_array(mask_path, file_format))
        total_pixels += int(mask.size)
        positive_pixels += int(np.count_nonzero(mask > 0))

    return total_pixels, positive_pixels


def _select_train_indices_by_neg_pos_ratio(
    train_dataset: ArchaeologyDataset,
    neg_to_pos_ratio: Optional[float],
    seed: int,
    allowed_base_indices: Optional[Sequence[int]] = None,
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
    allowed_base_set = (
        None if allowed_base_indices is None else {int(idx) for idx in allowed_base_indices}
    )
    if allowed_base_set is not None and len(allowed_base_set) == 0:
        raise ValueError("allowed_base_indices bos olamaz.")

    positive_indices: List[int] = []
    negative_indices: List[int] = []

    base_positive_cache: Dict[int, bool] = {}
    for idx in range(len(train_dataset)):
        base_idx, _ = train_dataset._resolve_sample_index(idx)
        if allowed_base_set is not None and int(base_idx) not in allowed_base_set:
            continue
        cached_positive = base_positive_cache.get(base_idx)
        if cached_positive is not None:
            is_positive = cached_positive
        else:
            if (
                train_dataset.task_type == "tile_classification"
                and train_dataset.tile_labels is not None
            ):
                is_positive = float(train_dataset.tile_labels[base_idx]) > 0.5
            else:
                img_path = train_dataset.image_files[base_idx]
                mask_path = train_dataset.masks_dir / img_path.name
                if not mask_path.exists():
                    raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")
                mask = _load_mask_array(mask_path, train_dataset.file_format)
                is_positive = np.any(mask > 0)
            base_positive_cache[base_idx] = bool(is_positive)

        if is_positive:
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
        "total_samples": len(positive_indices) + len(negative_indices),
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


def _resolve_auto_val_holdout_ratio(
    source_metadata: Dict[str, Any],
    *,
    default_ratio: float = 0.2,
) -> float:
    """Qt export metadata'sindaki train/val oranindan holdout oranini cikar."""
    fallback = float(default_ratio)
    if not 0.0 < fallback < 1.0:
        raise ValueError(
            f"default_ratio 0-1 araliginda olmali (uclar haric), verilen: {default_ratio}"
        )
    if not isinstance(source_metadata, dict):
        return fallback

    try:
        train_ratio = float(source_metadata.get("train_ratio", 0.0))
        val_ratio = float(source_metadata.get("val_ratio", 0.0))
    except (TypeError, ValueError):
        return fallback

    total_ratio = train_ratio + val_ratio
    if train_ratio <= 0.0 or val_ratio <= 0.0 or total_ratio <= 0.0:
        return fallback

    holdout_ratio = val_ratio / total_ratio
    if not 0.0 < holdout_ratio < 1.0:
        return fallback
    return float(holdout_ratio)


def _build_auto_val_holdout_indices(
    tile_labels: Sequence[float],
    *,
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Train icindeki explicit tile etiketlerinden stratified validation holdout uret."""
    if not 0.0 < float(holdout_ratio) < 1.0:
        raise ValueError(
            f"holdout_ratio 0-1 araliginda olmali (uclar haric), verilen: {holdout_ratio}"
        )
    if int(seed) < 0:
        raise ValueError(f"seed negatif olamaz, verilen: {seed}")

    labels = [float(value) for value in tile_labels]
    total_samples = len(labels)
    if total_samples < 2:
        raise ValueError(
            "Otomatik validation holdout icin en az 2 tile gerekir."
        )

    positive_indices = [idx for idx, value in enumerate(labels) if value > 0.5]
    negative_indices = [idx for idx, value in enumerate(labels) if value <= 0.5]
    if not positive_indices:
        raise ValueError(
            "Otomatik validation holdout icin train split icinde en az bir pozitif tile gerekir."
        )
    if len(positive_indices) < 2:
        raise ValueError(
            "Pozitifsiz validation split yerine train'den holdout uretmek icin "
            "en az 2 pozitif tile gerekir."
        )

    rng = np.random.RandomState(int(seed))

    def _split_class_indices(indices: Sequence[int]) -> Tuple[List[int], List[int]]:
        raw_indices = [int(idx) for idx in indices]
        if not raw_indices:
            return [], []
        if len(raw_indices) == 1:
            return list(raw_indices), []

        target_holdout = int(round(len(raw_indices) * float(holdout_ratio)))
        target_holdout = max(1, min(len(raw_indices) - 1, target_holdout))
        selected = rng.choice(raw_indices, size=target_holdout, replace=False)
        holdout_set = {int(idx) for idx in selected.tolist()}
        holdout_indices = sorted(holdout_set)
        train_indices = sorted(idx for idx in raw_indices if idx not in holdout_set)
        return train_indices, holdout_indices

    train_positive, val_positive = _split_class_indices(positive_indices)
    train_negative, val_negative = _split_class_indices(negative_indices)

    train_indices = sorted(train_positive + train_negative)
    val_indices = sorted(val_positive + val_negative)

    if not train_indices:
        raise ValueError("Otomatik validation holdout sonrasi train split bos kaldi.")
    if not val_indices:
        raise ValueError("Otomatik validation holdout validation split olusturamadi.")

    stats = {
        "total_samples": int(total_samples),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "positive_samples": int(len(positive_indices)),
        "negative_samples": int(len(negative_indices)),
        "train_positive_samples": int(len(train_positive)),
        "train_negative_samples": int(len(train_negative)),
        "val_positive_samples": int(len(val_positive)),
        "val_negative_samples": int(len(val_negative)),
    }
    return train_indices, val_indices, stats


def _build_auto_val_holdout_from_dataset(
    train_dataset: ArchaeologyDataset,
    *,
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Train dataset'ten base tile etiketlerini cikarip stratified val holdout uret."""
    base_labels: List[float] = []
    if (
        train_dataset.task_type == "tile_classification"
        and train_dataset.tile_labels is not None
    ):
        base_labels = [float(value) for value in train_dataset.tile_labels]
    else:
        for img_path in train_dataset.image_files:
            mask_path = train_dataset.masks_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Maske dosyasi bulunamadi: {mask_path}")
            mask = _load_mask_array(mask_path, train_dataset.file_format)
            base_labels.append(
                _mask_to_tile_label(
                    mask,
                    min_positive_ratio=float(train_dataset.tile_label_min_positive_ratio),
                )
            )

    return _build_auto_val_holdout_indices(
        base_labels,
        holdout_ratio=holdout_ratio,
        seed=seed,
    )


def _expand_base_indices_to_dataset_indices(
    dataset: ArchaeologyDataset,
    base_indices: Sequence[int],
) -> List[int]:
    """Augment carpani olan dataset icin base indeksleri sample indekslerine acar."""
    if dataset._augmentation_multiplier <= 1:
        return sorted(int(idx) for idx in base_indices)

    selected_base = sorted({int(idx) for idx in base_indices})
    expanded: List[int] = []
    multiplier = int(dataset._augmentation_multiplier)
    for base_idx in selected_base:
        start = int(base_idx) * multiplier
        expanded.extend(range(start, start + multiplier))
    return expanded


class TileClassifier(nn.Module):
    """SMP encoder tabanli tile-level binary classifier.

    use_fpn=True ise encoder'in tum katmanlarindan feature toplar
    (FPN-style multi-level aggregation). False ise sadece son katman.
    """

    def __init__(
        self,
        *,
        encoder_name: str,
        in_channels: int,
        dropout: float = 0.2,
        use_fpn: bool = False,
    ):
        super().__init__()
        smp_lib = _require_smp()
        self.encoder = smp_lib.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights="imagenet",
        )
        encoder_channels = list(getattr(self.encoder, "out_channels", []))
        if not encoder_channels:
            raise ValueError(f"Encoder cikis kanallari okunamadi: {encoder_name}")

        self.use_fpn = bool(use_fpn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        if self.use_fpn and len(encoder_channels) > 1:
            fpn_ch = sum(int(c) for c in encoder_channels[1:])
            self.classifier = nn.Linear(fpn_ch, 1)
        else:
            self.use_fpn = False
            self.classifier = nn.Linear(int(encoder_channels[-1]), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if not isinstance(features, (list, tuple)):
            features = [features]

        if self.use_fpn and len(features) > 1:
            pooled = [self.pool(f).flatten(1) for f in features[1:]]
            x = torch.cat(pooled, dim=1)
        else:
            x = self.pool(features[-1]).flatten(1)

        x = self.dropout(x)
        return self.classifier(x)


def _resolved_model_arch_name(config: TrainingConfig) -> str:
    """Checkpoint/model adlarinda kullanilacak gercek model ailesi adi."""
    if str(config.task_type).strip().lower() == "tile_classification":
        return "TileClassifier"
    return str(config.arch)


def create_model(config: TrainingConfig) -> nn.Module:
    """Model oluşturur."""

    task_type = str(config.task_type).strip().lower()

    if task_type == "tile_classification":
        base_model = TileClassifier(
            encoder_name=config.encoder,
            in_channels=config.in_channels,
            use_fpn=config.use_fpn_classifier,
        )
    else:
        smp_lib = _require_smp()
        if not hasattr(smp_lib, config.arch):
            raise ValueError(f"Mimari bulunamadı: {config.arch}")

        model_cls = getattr(smp_lib, config.arch)
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
    # Pratik secim ozeti:
    # - tile_classification: genelde bce, dengesizlik cok gucluyse focal
    # - segmentation: bce en stabil baslangic, dice overlap odakli,
    #   combined ise ikisi arasinda dengeli bir secim
    """Kayıp fonksiyonu oluşturur."""

    task_type = str(config.task_type).strip().lower()

    if task_type == "tile_classification" and config.loss_type in {"dice", "combined"}:
        raise ValueError(
            "tile_classification icin sadece 'bce' veya 'focal' loss kullanilabilir."
        )

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
    channel_importance_tracker: Optional[ChannelImportanceTracker] = None,
    channel_importance_max_batches: int = 0,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, Any]]]:
    """Bir epoch eğitim yapar."""
    
    model.train()
    total_loss = 0.0
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    if channel_importance_tracker is not None:
        channel_importance_tracker.reset()
    
    pbar = tqdm(train_loader, desc="Eğitim", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        capture_grad_importance = (
            channel_importance_tracker is not None
            and channel_importance_tracker.is_gradient_mode()
            and (
                int(channel_importance_max_batches) <= 0
                or batch_idx < int(channel_importance_max_batches)
            )
        )
        if capture_grad_importance:
            images.requires_grad_(True)
        
        optimizer.zero_grad()
        grad_scale = 1.0
        
        # Mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if use_amp and scaler is not None:
            grad_scale = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if capture_grad_importance and channel_importance_tracker is not None:
            channel_importance_tracker.accumulate_gradient(images, scale=grad_scale)
        
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
    importance_summary = (
        channel_importance_tracker.summarize()
        if channel_importance_tracker is not None
        else None
    )
    
    return avg_loss, avg_metrics, importance_summary


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

    task_type = str(config.task_type).strip().lower()
    if task_type not in {"segmentation", "tile_classification"}:
        raise ValueError(f"Desteklenmeyen task_type: {config.task_type}")
    if not 0.0 <= float(config.tile_label_min_positive_ratio) <= 1.0:
        raise ValueError(
            "tile_label_min_positive_ratio 0-1 araliginda olmali, "
            f"verilen: {config.tile_label_min_positive_ratio}"
        )

    if not 0.0 < float(config.val_keep_ratio) <= 1.0:
        raise ValueError(
            f"val_keep_ratio 0-1 aralığında olmalı (0 hariç), verilen: {config.val_keep_ratio}"
        )
    if int(config.val_sample_seed) < 0:
        raise ValueError(f"val_sample_seed negatif olamaz, verilen: {config.val_sample_seed}")
    if config.auto_val_from_train and not 0.0 < float(config.auto_val_ratio) < 1.0:
        raise ValueError(
            "auto_val_ratio 0-1 araliginda olmali (uclar haric), "
            f"verilen: {config.auto_val_ratio}"
        )
    if not 0.0 <= float(config.deterministic_rotate_step_deg) < 360.0:
        raise ValueError(
            "deterministic_rotate_step_deg 0-360 araliginda olmali "
            f"(360 haric), verilen: {config.deterministic_rotate_step_deg}"
        )
    
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
    file_format = _infer_file_format(
        config.data_dir,
        allow_missing_val=bool(config.auto_val_from_train),
    )
    train_dataset = ArchaeologyDataset(
        config.data_dir / "train",
        augment=True,
        file_format=file_format,
        task_type=config.task_type,
        tile_label_min_positive_ratio=config.tile_label_min_positive_ratio,
        deterministic_rotate_step_deg=float(config.deterministic_rotate_step_deg),
        scale_augment=config.scale_augment,
        scale_augment_min=config.scale_augment_min,
        scale_augment_max=config.scale_augment_max,
    )
    auto_val_holdout_stats: Optional[Dict[str, int]] = None
    train_base_indices_for_holdout: Optional[List[int]] = None
    if config.auto_val_from_train:
        train_base_indices_for_holdout, val_base_indices, auto_val_holdout_stats = (
            _build_auto_val_holdout_from_dataset(
                train_dataset,
                holdout_ratio=float(config.auto_val_ratio),
                seed=int(config.val_sample_seed),
            )
        )
        val_source_dataset = ArchaeologyDataset(
            config.data_dir / "train",
            augment=False,
            file_format=file_format,
            task_type=config.task_type,
            tile_label_min_positive_ratio=config.tile_label_min_positive_ratio,
            deterministic_rotate_step_deg=0.0,
        )
        val_dataset: Dataset = Subset(val_source_dataset, val_base_indices)
    else:
        val_dataset = ArchaeologyDataset(
            config.data_dir / "val",
            augment=False,
            file_format=file_format,
            task_type=config.task_type,
            tile_label_min_positive_ratio=config.tile_label_min_positive_ratio,
            deterministic_rotate_step_deg=0.0,
        )
    if train_dataset._use_deterministic_rotation:
        LOGGER.info(
            "Deterministik train rotasyonu aktif: adim=%.2f°, carpim=%dx, rastgele flip=acik",
            float(config.deterministic_rotate_step_deg),
            int(train_dataset._augmentation_multiplier),
        )
    
    train_loader_dataset: Dataset = train_dataset
    if train_base_indices_for_holdout is not None:
        train_indices = _expand_base_indices_to_dataset_indices(
            train_dataset,
            train_base_indices_for_holdout,
        )
        train_loader_dataset = Subset(train_dataset, train_indices)
    train_sampling_stats: Optional[Dict[str, int]] = None
    if config.train_neg_to_pos_ratio is not None:
        selected_indices, train_sampling_stats = _select_train_indices_by_neg_pos_ratio(
            train_dataset=train_dataset,
            neg_to_pos_ratio=config.train_neg_to_pos_ratio,
            seed=config.train_neg_sample_seed,
            allowed_base_indices=train_base_indices_for_holdout,
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
    if (
        train_sampling_stats is None
        and val_sampling_stats is None
        and auto_val_holdout_stats is None
    ):
        LOGGER.info(
            f"Eğitim: {len(train_loader_dataset)} örnek, Doğrulama: {len(val_dataset)} örnek"
        )
    else:
        train_count = (
            int(train_sampling_stats["selected_total_samples"])
            if train_sampling_stats is not None
            else len(train_loader_dataset)
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
        if auto_val_holdout_stats is not None:
            LOGGER.info(
                "  Otomatik val holdout -> train=%d (poz=%d, neg=%d) | val=%d (poz=%d, neg=%d) | oran=%.3f",
                int(auto_val_holdout_stats["train_samples"]),
                int(auto_val_holdout_stats["train_positive_samples"]),
                int(auto_val_holdout_stats["train_negative_samples"]),
                int(auto_val_holdout_stats["val_samples"]),
                int(auto_val_holdout_stats["val_positive_samples"]),
                int(auto_val_holdout_stats["val_negative_samples"]),
                float(config.auto_val_ratio),
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
    channel_importance_tracker = ChannelImportanceTracker(
        model=model,
        in_channels=config.in_channels,
        channel_names=config.channel_names,
        enabled=bool(config.monitor_channel_importance),
    )
    
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

    model_arch_name = _resolved_model_arch_name(config)
    model_name = f"{model_arch_name}_{config.encoder}_{config.in_channels}ch"
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
        "channel_importance_train": [],
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
    LOGGER.info(f"Task: {config.task_type}")
    LOGGER.info(f"Model: {model_arch_name} + {config.encoder}")
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
    if config.auto_val_from_train:
        LOGGER.info(
            "Validation holdout kaynagi: train split icinden stratified ayrim (oran=%.3f, neden=%s)",
            float(config.auto_val_ratio),
            str(config.auto_val_reason or "otomatik"),
        )
    LOGGER.info(f"Epochs: {config.epochs}")
    LOGGER.info(f"Batch size: {config.batch_size}")
    LOGGER.info(f"Learning rate: {config.lr}")
    if config.monitor_channel_importance:
        if channel_importance_tracker.mode == "attention":
            LOGGER.info("Band onemi takibi: aktif (mode=attention, train tum batch'ler).")
        elif channel_importance_tracker.mode == "gradient":
            max_batches = int(config.channel_importance_max_batches)
            max_batches_info = "tum batch'ler" if max_batches <= 0 else f"ilk {max_batches} batch"
            LOGGER.info("Band onemi takibi: aktif (mode=gradient, %s).", max_batches_info)
        else:
            LOGGER.info("Band onemi takibi: devre disi (uygun modul bulunamadi).")
    else:
        LOGGER.info("Band onemi takibi: kapali.")
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
        train_loss, train_metrics, train_channel_importance = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config.use_amp,
            config.metric_threshold,
            channel_importance_tracker=channel_importance_tracker,
            channel_importance_max_batches=int(config.channel_importance_max_batches),
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
        history["channel_importance_train"].append(train_channel_importance)
        
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
        
        if train_channel_importance is not None:
            top_items = list(train_channel_importance.get("ranking", []))[:3]
            top_text = ", ".join(
                f"{item.get('channel')}={float(item.get('score', 0.0)):.4f}"
                for item in top_items
            )
            LOGGER.info(
                "Train band onemi (%s, n=%d): %s",
                str(train_channel_importance.get("mode", "n/a")),
                int(train_channel_importance.get("samples", 0)),
                top_text if top_text else "yeterli veri yok",
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
            "train_channel_importance": train_channel_importance,
            "config": {
                "schema_version": METADATA_SCHEMA_VERSION,
                "arch": model_arch_name,
                "encoder": config.encoder,
                "in_channels": config.in_channels,
                "channel_names": list(config.channel_names),
                "task_type": config.task_type,
                "tile_size": config.source_metadata.get("tile_size"),
                "overlap": config.source_metadata.get("overlap"),
                "bands": config.source_metadata.get("bands"),
                "input_file": config.source_metadata.get("input_file"),
                "mask_file": config.source_metadata.get("mask_file"),
                "tile_label_min_positive_ratio": config.tile_label_min_positive_ratio,
                "enable_attention": config.enable_attention,
                "attention_reduction": config.attention_reduction,
                "loss_type": config.loss_type,
                "balance_mode": config.balance_mode,
                "pos_weight": config.pos_weight,
                "train_neg_to_pos_ratio": config.train_neg_to_pos_ratio,
                "train_neg_sample_seed": config.train_neg_sample_seed,
                "val_keep_ratio": config.val_keep_ratio,
                "val_sample_seed": config.val_sample_seed,
                "auto_val_from_train": config.auto_val_from_train,
                "auto_val_ratio": config.auto_val_ratio,
                "auto_val_reason": config.auto_val_reason,
                "metric_threshold": config.metric_threshold,
                "val_threshold_sweep": config.val_threshold_sweep,
                "val_threshold_min": config.val_threshold_min,
                "val_threshold_max": config.val_threshold_max,
                "val_threshold_step": config.val_threshold_step,
                "monitor_channel_importance": config.monitor_channel_importance,
                "channel_importance_max_batches": config.channel_importance_max_batches,
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
    channel_importance_tracker.close()
    
    # Eğitim geçmişini kaydet
    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    channel_history_path: Optional[Path] = None
    if config.monitor_channel_importance:
        channel_history_path = config.output_dir / "channel_importance_history.json"
        with open(channel_history_path, "w") as f:
            json.dump(history["channel_importance_train"], f, indent=2)
    
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
    if channel_history_path is not None:
        LOGGER.info("Band onemi gecmisi: %s", channel_history_path)
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

    if config.publish_active:
        published = _publish_active_artifacts(config=config, best_model_path=best_model_path)
        LOGGER.info("Aktif model yayinlandi: %s", published["model"])
        LOGGER.info("Aktif training metadata: %s", published["training_metadata"])
        LOGGER.info("Aktif publish manifesti: %s", published["manifest"])

    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="5 kanallı arkeolojik tespit modeli eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Veri
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=str(CONFIG["data"]),
        help="Eğitim verisi dizini (egitim_verisi_olusturma.py çıktısı)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=str(CONFIG["task"]),
        choices=["segmentation", "tile_classification"],
        help=(
            "Egitim gorevi. segmentation: piksel maskesi ogren; "
            "tile_classification: tile icin tek olasilik/etiket ogren."
        ),
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
    # Loss secimi kisaca:
    # - bce: en stabil temel secenek
    # - dice: overlap odakli, segmentation icin
    # - combined: BCE + Dice dengesi
    # - focal: kuvvetli dengesizlikte zor orneklere odaklanir
    parser.add_argument(
        "--loss",
        type=str,
        default=str(CONFIG["loss"]),
        choices=["bce", "dice", "combined", "focal"],
        help=(
            # bce: stabil temel secenek
            # dice: overlap odakli segmentation
            # combined: BCE + Dice dengesi
            # focal: zor/dengesiz orneklere odaklanir
            "Kayıp fonksiyonu. "
            "Class balance ayarları (balance_mode/pos_weight) sadece bce ve combined loss için etkilidir."
        ),
    )
    parser.add_argument(
        "--balance-mode",
        type=str,
        default=(
            "none"
            if CONFIG["balance_mode"] is None
            else str(CONFIG["balance_mode"])
        ),
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
        "--publish-active",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["publish_active"]),
        help="Egitim sonunda aktif IDE modelini sabit klasore kopyala.",
    )
    parser.add_argument(
        "--active-dir",
        type=str,
        default=str(CONFIG["active_dir"]),
        help="Aktif model ve training metadata yayin klasoru.",
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
    parser.add_argument(
        "--tile-label-min-positive-ratio",
        type=float,
        default=float(CONFIG["tile_label_min_positive_ratio"]),
        help=(
            "tile_classification modunda pozitif tile etiketi icin gereken "
            "minimum pozitif piksel orani. 0.0 ise herhangi bir pozitif piksel yeterlidir."
        ),
    )
    parser.add_argument(
        "--deterministic-rotate-step-deg",
        type=float,
        default=float(CONFIG["deterministic_rotate_step_deg"]),
        help=(
            "Train augment icin sabit donme aci adimi (derece). "
            "0: kapali, 30: 0..330 arasi 12 farkli aci."
        ),
    )
    parser.add_argument(
        "--scale-augment",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["scale_augment"]),
        help="Egitimde rastgele zoom in/out augmentasyonu.",
    )
    parser.add_argument("--scale-augment-min", type=float, default=float(CONFIG["scale_augment_min"]),
                        help="Scale augment min (0.7 = %%30 zoom out).")
    parser.add_argument("--scale-augment-max", type=float, default=float(CONFIG["scale_augment_max"]),
                        help="Scale augment max (1.3 = %%30 zoom in).")
    parser.add_argument(
        "--use-fpn-classifier",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["use_fpn_classifier"]),
        help="TileClassifier icin FPN-style multi-level feature aggregation.",
    )
    parser.add_argument(
        "--monitor-channel-importance",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["monitor_channel_importance"]),
        help=(
            "Egitim sirasinda kanal (band) onemini olc ve raporla. "
            "CBAM varsa attention skorlarini, yoksa gradient fallback kullanir."
        ),
    )
    parser.add_argument(
        "--channel-importance-max-batches",
        type=int,
        default=int(CONFIG["channel_importance_max_batches"]),
        help=(
            "Gradient fallback modunda train'de kac batch uzerinden ortalama alinacagi. "
            "0 ise tum batch'ler kullanilir."
        ),
    )

    args = parser.parse_args()
    
    # Veri dizini kontrolü
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"HATA: Veri dizini bulunamadı: {data_dir}")
        print("\nÖnce veri hazırlama adımını çalıştırın:")
        print(f"  python egitim_verisi_olusturma.py --input kesif_alani.tif --mask ground_truth.tif --output {data_dir}")
        print(
            "  veya  python prepare_tile_classification_dataset.py "
            f"--pair kesif_alani.tif ground_truth.tif --output-dir {data_dir}"
        )
        sys.exit(1)

    try:
        data_layout = _detect_dataset_layout(data_dir, args.task)
    except ValueError as exc:
        print(f"HATA: Eğitim veri düzeni algılanamadı:\n{exc}")
        sys.exit(1)

    if data_layout == "paired":
        expected_hint = data_dir / "train" / "images"
    else:
        expected_hint = data_dir / "train"

    if not expected_hint.exists():
        print(f"HATA: Eğitim verisi bulunamadı: {expected_hint}")
        print("\nVeri hazırlama çıktısını kontrol edin.")
        sys.exit(1)
    
    # Metadata'dan kanal sayısını oku
    metadata_path = data_dir / "metadata.json"
    channel_names = MODEL_CHANNEL_NAMES
    source_metadata: Dict[str, Any] = {}
    source_metadata_path: Optional[Path] = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            source_metadata = json.load(f)
        source_metadata_path = metadata_path
        in_channels = source_metadata.get("num_channels", 5)
        LOGGER.info(f"Metadata'dan kanal sayısı okundu: {in_channels}")
        raw_channel_names = source_metadata.get("channel_names")
        if isinstance(raw_channel_names, list) and raw_channel_names:
            channel_names = tuple(str(x) for x in raw_channel_names)
    else:
        in_channels = 5
        LOGGER.warning(f"Metadata bulunamadı, varsayılan kanal sayısı kullanılıyor: {in_channels}")

    auto_val_ratio = _resolve_auto_val_holdout_ratio(source_metadata)

    if not 0.0 <= float(args.tile_label_min_positive_ratio) <= 1.0:
        print("HATA: --tile-label-min-positive-ratio 0 ile 1 arasında olmalı.")
        sys.exit(1)

    if not 0.0 <= float(args.deterministic_rotate_step_deg) < 360.0:
        print("HATA: --deterministic-rotate-step-deg 0 ile 360 arasinda olmali (360 haric).")
        sys.exit(1)

    if args.task == "tile_classification" and args.loss in {"dice", "combined"}:
        print("HATA: tile_classification için --loss yalnızca 'bce' veya 'focal' olabilir.")
        sys.exit(1)

    # Etiket dağılımını doğrula (tamamı negatif veri sessizce eğitime girmesin)
    allow_missing_val = bool(
        (args.task == "tile_classification" and data_layout == "classification_folders")
        or data_layout == "paired"
    )
    file_format = _infer_file_format(data_dir, allow_missing_val=allow_missing_val)
    manifest_train_counts = None
    manifest_val_counts = None
    class_folder_train_counts = None
    class_folder_val_counts = None
    class_folder_count_source = None
    if args.task == "tile_classification":
        if data_layout == "classification_folders":
            class_folder_train_counts, train_count_source = (
                _resolve_classification_folder_split_counts(
                    data_dir / "train",
                    file_format,
                )
            )
            class_folder_val_counts, val_count_source = (
                _resolve_classification_folder_split_counts(
                    data_dir / "val",
                    file_format,
                )
            )
            class_folder_count_source = (
                "manifest"
                if train_count_source == "manifest" and val_count_source == "manifest"
                else "class_dirs"
            )
        else:
            manifest_train_counts = _count_positive_tile_labels_from_manifest(data_dir / "train")
            manifest_val_counts = _count_positive_tile_labels_from_manifest(data_dir / "val")

    if class_folder_train_counts is not None:
        train_total, train_positive = class_folder_train_counts
    elif manifest_train_counts is not None:
        train_total, train_positive = manifest_train_counts
    elif data_layout == "classification_folders" and args.task == "tile_classification":
        train_total, train_positive = _count_positive_tiles_from_class_dirs(
            data_dir / "train",
            file_format,
        )
    else:
        train_total, train_positive = _count_positive_mask_files(
            data_dir / "train" / "masks",
            file_format,
            tile_label_min_positive_ratio=float(args.tile_label_min_positive_ratio),
        )

    if class_folder_val_counts is not None:
        val_total, val_positive = class_folder_val_counts
    elif manifest_val_counts is not None:
        val_total, val_positive = manifest_val_counts
    elif data_layout == "classification_folders" and args.task == "tile_classification":
        val_total, val_positive = _count_positive_tiles_from_class_dirs(
            data_dir / "val",
            file_format,
        )
    else:
        val_total, val_positive = _count_positive_mask_files(
            data_dir / "val" / "masks",
            file_format,
            tile_label_min_positive_ratio=float(args.tile_label_min_positive_ratio),
        )
    if args.task == "tile_classification":
        if class_folder_count_source == "manifest":
            LOGGER.info("Tile label sayaclari tile_labels.csv manifestinden okundu.")
        elif data_layout == "classification_folders":
            LOGGER.info("Tile etiket sayaclari Positive/Negative klasorlerinden okundu.")
        elif manifest_train_counts is not None and manifest_val_counts is not None:
            LOGGER.info("Tile label sayaclari tile_labels.csv manifestinden okundu.")
        LOGGER.info(
            "Tile etiket dağılımı | train: %d/%d pozitif tile | val: %d/%d pozitif tile",
            train_positive, train_total, val_positive, val_total,
        )
    else:
        LOGGER.info(
            "Maske dağılımı | train: %d/%d pozitif tile | val: %d/%d pozitif tile",
            train_positive, train_total, val_positive, val_total,
        )

    auto_val_from_train = False
    auto_val_reason = ""
    if data_layout == "classification_folders":
        if val_total <= 0:
            auto_val_from_train = True
            auto_val_reason = "val split boş"
        elif val_positive <= 0:
            auto_val_from_train = True
            auto_val_reason = "val splitte pozitif tile yok"
    elif data_layout == "paired" and val_total <= 0:
        auto_val_from_train = True
        auto_val_reason = "val split boş"

    if auto_val_from_train:
        LOGGER.warning(
            "Otomatik validation holdout etkinleştirildi: %s. "
            "Train split içinden stratified val ayrımı yapılacak (oran=%.3f).",
            auto_val_reason,
            auto_val_ratio,
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

    train_total_pixels = 0
    train_positive_pixels = 0
    if args.task == "segmentation":
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

    if args.channel_importance_max_batches < 0:
        print("HATA: --channel-importance-max-batches negatif olamaz.")
        sys.exit(1)

    resolved_pos_weight = 1.0
    if args.balance_mode == "manual":
        resolved_pos_weight = float(args.pos_weight)
        LOGGER.info("Manual class balance aktif: pos_weight=%.4f", resolved_pos_weight)
    elif args.balance_mode == "auto":
        if args.task == "tile_classification":
            if train_positive == 0:
                resolved_pos_weight = 1.0
                LOGGER.warning(
                    "Auto class balance hesaplanamadı (train pozitif tile yok), pos_weight=1.0 kullanılacak."
                )
            else:
                train_negative_tiles = train_total - train_positive
                raw_pos_weight = train_negative_tiles / train_positive
                resolved_pos_weight = float(
                    np.clip(raw_pos_weight, 1.0, float(args.max_auto_pos_weight))
                )
                if raw_pos_weight != resolved_pos_weight:
                    LOGGER.info(
                        "Auto class balance (tile) : ham pos_weight=%.4f, kırpılmış pos_weight=%.4f (max=%.4f)",
                        raw_pos_weight,
                        resolved_pos_weight,
                        float(args.max_auto_pos_weight),
                    )
                else:
                    LOGGER.info("Auto class balance (tile): pos_weight=%.4f", resolved_pos_weight)
        else:
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
        task_type=args.task,
        tile_label_min_positive_ratio=float(args.tile_label_min_positive_ratio),
        arch=args.arch,
        encoder=args.encoder,
        in_channels=in_channels,
        channel_names=channel_names,
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
        monitor_channel_importance=bool(args.monitor_channel_importance),
        channel_importance_max_batches=int(args.channel_importance_max_batches),
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
        auto_val_from_train=bool(auto_val_from_train),
        auto_val_ratio=float(auto_val_ratio),
        auto_val_reason=str(auto_val_reason),
        deterministic_rotate_step_deg=float(args.deterministic_rotate_step_deg),
        scale_augment=bool(args.scale_augment),
        scale_augment_min=float(args.scale_augment_min),
        scale_augment_max=float(args.scale_augment_max),
        use_fpn_classifier=bool(args.use_fpn_classifier),
        output_dir=Path(args.output),
        save_every_epoch=bool(CONFIG["save_every_epoch"]),
        epoch_dir=str(CONFIG["epoch_dir"]),
        publish_active=bool(args.publish_active),
        active_dir=Path(args.active_dir),
        source_metadata_path=source_metadata_path,
        source_metadata=source_metadata,
    )
    
    # Eğitimi başlat
    try:
        best_model = train(config)
        
        print("\n" + "=" * 60)
        print("EGITIM TAMAMLANDI!")
        print("=" * 60)
        print("\nEğitilmiş modeli kullanmak için config.yaml veya config.local.yaml içinde:")
        print(f"  weights: \"{best_model}\"")
        print("  zero_shot_imagenet: false")
        if config.task_type == "tile_classification":
            print("  dl_task: \"tile_classification\"")
        print("\nVeya komut satırından:")
        if config.task_type == "tile_classification":
            print(f"  python archaeo_detect.py --weights {best_model} --dl-task tile_classification")
        else:
            print(f"  python archaeo_detect.py --weights {best_model}")
        if config.publish_active:
            print("\nIDE trained-only profili icin aktif artefact'lar guncellendi:")
            print(f"  weights: \"{Path(config.active_dir) / 'model.pth'}\"")
            print(f"  training_metadata: \"{Path(config.active_dir) / 'training_metadata.json'}\"")
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

