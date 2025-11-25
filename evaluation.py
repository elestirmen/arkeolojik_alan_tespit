"""
Arkeolojik Alan Tespiti - Değerlendirme Metrikleri

Bu modül tespit sonuçlarını ground truth ile karşılaştırmak için
çeşitli metrikler sağlar.

Kullanım:
    from evaluation import evaluate_predictions, compute_metrics

    # Raster ile değerlendirme
    metrics = evaluate_predictions(
        prediction_path="output_mask.tif",
        ground_truth_path="ground_truth.tif"
    )
    print(f"IoU: {metrics['iou']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")

    # NumPy array ile değerlendirme
    metrics = compute_metrics(prediction_array, ground_truth_array)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:
    gpd = None

LOGGER = logging.getLogger("archaeo_detect.evaluation")


@dataclass
class EvaluationMetrics:
    """Değerlendirme metriklerini tutan sınıf."""
    
    # Temel sayımlar
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Türetilmiş metrikler
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float  # Intersection over Union (Jaccard Index)
    dice: float  # Dice Coefficient (F1 ile aynı)
    specificity: float
    
    # Ek bilgiler
    total_pixels: int
    positive_pixels_pred: int
    positive_pixels_gt: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Metrikleri dictionary olarak döndür."""
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "dice": self.dice,
            "specificity": self.specificity,
            "total_pixels": self.total_pixels,
            "positive_pixels_pred": self.positive_pixels_pred,
            "positive_pixels_gt": self.positive_pixels_gt,
        }
    
    def summary(self) -> str:
        """İnsan okunabilir özet döndür."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DEĞERLENDİRME METRİKLERİ                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Accuracy (Doğruluk)     : {self.accuracy:>8.4f}  ({self.accuracy*100:>6.2f}%)         ║
║  Precision (Kesinlik)    : {self.precision:>8.4f}  ({self.precision*100:>6.2f}%)         ║
║  Recall (Duyarlılık)     : {self.recall:>8.4f}  ({self.recall*100:>6.2f}%)         ║
║  F1 Score                : {self.f1:>8.4f}  ({self.f1*100:>6.2f}%)         ║
║  IoU (Jaccard)           : {self.iou:>8.4f}  ({self.iou*100:>6.2f}%)         ║
║  Dice Coefficient        : {self.dice:>8.4f}  ({self.dice*100:>6.2f}%)         ║
║  Specificity (Özgüllük)  : {self.specificity:>8.4f}  ({self.specificity*100:>6.2f}%)         ║
╠══════════════════════════════════════════════════════════════════╣
║  Confusion Matrix:                                               ║
║    TP: {self.true_positives:>10}    FP: {self.false_positives:>10}                     ║
║    FN: {self.false_negatives:>10}    TN: {self.true_negatives:>10}                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Toplam Piksel      : {self.total_pixels:>12}                              ║
║  Tahmin Pozitif     : {self.positive_pixels_pred:>12}                              ║
║  Ground Truth Pozitif: {self.positive_pixels_gt:>12}                              ║
╚══════════════════════════════════════════════════════════════════╝
"""


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """
    İkili maske tahminleri için değerlendirme metriklerini hesapla.
    
    Args:
        prediction: Tahmin maskesi (0/1 veya boolean)
        ground_truth: Ground truth maskesi (0/1 veya boolean)
        mask: Opsiyonel geçerlilik maskesi (True = değerlendir)
    
    Returns:
        EvaluationMetrics: Hesaplanan tüm metrikler
    
    Example:
        >>> pred = np.array([[1, 0], [0, 1]])
        >>> gt = np.array([[1, 1], [0, 0]])
        >>> metrics = compute_metrics(pred, gt)
        >>> print(f"IoU: {metrics.iou:.3f}")
    """
    # Binary'ye çevir
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    
    # Şekil kontrolü
    if pred.shape != gt.shape:
        raise ValueError(
            f"Tahmin ve ground truth şekilleri eşleşmiyor: "
            f"{pred.shape} vs {gt.shape}"
        )
    
    # Maske uygula
    if mask is not None:
        if mask.shape != pred.shape:
            raise ValueError(f"Maske şekli eşleşmiyor: {mask.shape} vs {pred.shape}")
        valid = mask.astype(bool)
        pred = pred[valid]
        gt = gt[valid]
    
    # Temel sayımlar
    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    
    total = tp + tn + fp + fn
    
    # Güvenli bölme için epsilon
    eps = 1e-8
    
    # Metrik hesaplamaları
    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    
    # F1 ve IoU
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)  # Dice = F1 for binary
    
    return EvaluationMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        iou=float(iou),
        dice=float(dice),
        specificity=float(specificity),
        total_pixels=total,
        positive_pixels_pred=tp + fp,
        positive_pixels_gt=tp + fn,
    )


def compute_metrics_probabilistic(
    prediction_prob: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Olasılık haritası için çoklu eşikte metrik hesapla.
    
    Args:
        prediction_prob: Olasılık haritası (0-1 arası)
        ground_truth: Ground truth maskesi (0/1)
        thresholds: Değerlendirilecek eşik değerleri
    
    Returns:
        Dictionary: Her eşik için metrikler ve özet istatistikler
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    results = {
        "thresholds": thresholds.tolist(),
        "metrics_per_threshold": [],
        "best_threshold": None,
        "best_f1": 0.0,
        "best_iou": 0.0,
    }
    
    best_f1 = 0.0
    best_iou = 0.0
    best_th_f1 = 0.5
    best_th_iou = 0.5
    
    for th in thresholds:
        pred_binary = (prediction_prob >= th).astype(np.uint8)
        metrics = compute_metrics(pred_binary, ground_truth)
        
        results["metrics_per_threshold"].append({
            "threshold": float(th),
            "f1": metrics.f1,
            "iou": metrics.iou,
            "precision": metrics.precision,
            "recall": metrics.recall,
        })
        
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_th_f1 = float(th)
        
        if metrics.iou > best_iou:
            best_iou = metrics.iou
            best_th_iou = float(th)
    
    results["best_threshold_f1"] = best_th_f1
    results["best_f1"] = best_f1
    results["best_threshold_iou"] = best_th_iou
    results["best_iou"] = best_iou
    
    return results


def evaluate_predictions(
    prediction_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    band: int = 1,
    threshold: Optional[float] = None,
) -> EvaluationMetrics:
    """
    Raster dosyalarından tahmin ve ground truth'u oku ve değerlendir.
    
    Args:
        prediction_path: Tahmin raster dosyası yolu
        ground_truth_path: Ground truth raster dosyası yolu
        band: Okunacak bant numarası (1-tabanlı)
        threshold: Olasılık haritası ise uygulanacak eşik
    
    Returns:
        EvaluationMetrics: Hesaplanan metrikler
    """
    if rasterio is None:
        raise ImportError("rasterio gerekli: pip install rasterio")
    
    prediction_path = Path(prediction_path)
    ground_truth_path = Path(ground_truth_path)
    
    if not prediction_path.exists():
        raise FileNotFoundError(f"Tahmin dosyası bulunamadı: {prediction_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth dosyası bulunamadı: {ground_truth_path}")
    
    with rasterio.open(prediction_path) as src_pred:
        prediction = src_pred.read(band)
    
    with rasterio.open(ground_truth_path) as src_gt:
        ground_truth = src_gt.read(band)
    
    # Şekil kontrolü
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Raster boyutları eşleşmiyor: {prediction.shape} vs {ground_truth.shape}"
        )
    
    # Eşikleme (olasılık haritası ise)
    if threshold is not None:
        prediction = (prediction >= threshold).astype(np.uint8)
    
    # Ground truth'u binary'ye çevir
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    return compute_metrics(prediction, ground_truth)


def evaluate_vectors(
    prediction_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    rasterize_resolution: float = 1.0,
) -> EvaluationMetrics:
    """
    Vektör dosyalarını (GeoPackage) değerlendir.
    
    Args:
        prediction_path: Tahmin vektör dosyası (gpkg)
        ground_truth_path: Ground truth vektör dosyası (gpkg)
        rasterize_resolution: Rasterleştirme çözünürlüğü (metre)
    
    Returns:
        EvaluationMetrics: Hesaplanan metrikler
    """
    if gpd is None:
        raise ImportError("geopandas gerekli: pip install geopandas")
    
    pred_gdf = gpd.read_file(prediction_path)
    gt_gdf = gpd.read_file(ground_truth_path)
    
    # CRS kontrolü
    if pred_gdf.crs != gt_gdf.crs:
        gt_gdf = gt_gdf.to_crs(pred_gdf.crs)
    
    # Bounding box hesapla
    total_bounds = np.array([
        min(pred_gdf.total_bounds[0], gt_gdf.total_bounds[0]),
        min(pred_gdf.total_bounds[1], gt_gdf.total_bounds[1]),
        max(pred_gdf.total_bounds[2], gt_gdf.total_bounds[2]),
        max(pred_gdf.total_bounds[3], gt_gdf.total_bounds[3]),
    ])
    
    # Rasterleştir
    width = int((total_bounds[2] - total_bounds[0]) / rasterize_resolution)
    height = int((total_bounds[3] - total_bounds[1]) / rasterize_resolution)
    
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    
    transform = from_bounds(*total_bounds, width, height)
    
    pred_shapes = [(geom, 1) for geom in pred_gdf.geometry]
    gt_shapes = [(geom, 1) for geom in gt_gdf.geometry]
    
    pred_raster = rasterize(pred_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    gt_raster = rasterize(gt_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    
    return compute_metrics(pred_raster, gt_raster)


def compute_pr_curve(
    prediction_prob: np.ndarray,
    ground_truth: np.ndarray,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precision-Recall eğrisi hesapla.
    
    Args:
        prediction_prob: Olasılık haritası (0-1)
        ground_truth: Ground truth maskesi
        num_points: Eğri noktası sayısı
    
    Returns:
        Tuple: (precision_array, recall_array, thresholds)
    """
    thresholds = np.linspace(0.0, 1.0, num_points)
    precisions = []
    recalls = []
    
    gt = ground_truth.astype(bool).ravel()
    prob = prediction_prob.ravel()
    
    for th in thresholds:
        pred = prob >= th
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds


def compute_auc(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Area Under PR Curve (AUC-PR) hesapla.
    
    Args:
        precisions: Precision değerleri
        recalls: Recall değerleri
    
    Returns:
        float: AUC-PR değeri
    """
    # Recall'a göre sırala
    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    
    # Trapezoid kuralı ile alan hesapla
    auc = np.trapz(sorted_precisions, sorted_recalls)
    return float(auc)


# ============================================================================
# CLI Arayüzü
# ============================================================================

def main():
    """Komut satırı arayüzü."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Arkeolojik alan tespiti sonuçlarını değerlendir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prediction", "-p",
        type=str,
        required=True,
        help="Tahmin dosyası yolu (raster veya vektör)",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        required=True,
        help="Ground truth dosyası yolu",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Olasılık haritası için eşik değeri",
    )
    parser.add_argument(
        "--band", "-b",
        type=int,
        default=1,
        help="Okunacak bant numarası",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Sonuçları JSON olarak kaydet",
    )
    
    args = parser.parse_args()
    
    # Dosya türüne göre değerlendir
    pred_path = Path(args.prediction)
    
    if pred_path.suffix.lower() in (".gpkg", ".shp", ".geojson"):
        metrics = evaluate_vectors(args.prediction, args.ground_truth)
    else:
        metrics = evaluate_predictions(
            args.prediction,
            args.ground_truth,
            band=args.band,
            threshold=args.threshold,
        )
    
    # Sonuçları göster
    print(metrics.summary())
    
    # JSON kaydet (opsiyonel)
    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()

