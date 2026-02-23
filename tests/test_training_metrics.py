from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import (
    ArchaeologyDataset,
    BCELoss,
    _compute_val_target_samples,
    _select_indices_by_keep_ratio,
    _select_train_indices_by_neg_pos_ratio,
    validate,
)


class _LogitFromInputModel(nn.Module):
    """Use first input channel as a proxy mask to generate deterministic logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x==1 -> logit -0.4 (p~0.40), x==0 -> logit -1.0 (p~0.27)
        return x[:, :1, :, :] * 0.6 - 1.0


def _build_loader_with_positive_and_empty(batch_size: int = 1) -> DataLoader:
    masks = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    masks[0, 0, 2:6, 2:6] = 1.0
    images = masks.clone()
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_validate_uses_global_confusion_counts_not_batch_average() -> None:
    model = _LogitFromInputModel()
    loader = _build_loader_with_positive_and_empty(batch_size=1)
    criterion = BCELoss()

    _loss, metrics, sweep = validate(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        metric_threshold=0.3,
        sweep_thresholds=None,
    )

    assert sweep is None
    assert metrics["iou"] > 0.99
    assert metrics["precision"] > 0.99
    assert metrics["recall"] > 0.99


def test_validate_threshold_sweep_finds_better_threshold() -> None:
    model = _LogitFromInputModel()
    loader = _build_loader_with_positive_and_empty(batch_size=2)
    criterion = BCELoss()

    _loss, fixed_metrics, sweep = validate(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        metric_threshold=0.5,
        sweep_thresholds=(0.3, 0.5),
    )

    assert fixed_metrics["iou"] < 0.05
    assert sweep is not None
    assert sweep["best_iou"] > 0.99
    assert abs(sweep["best_threshold"] - 0.3) < 1e-6


def _write_npz_tile(split_dir: Path, tile_name: str, positive: bool) -> None:
    image = np.zeros((12, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    if positive:
        mask[2:4, 2:4] = 1
    np.savez_compressed(split_dir / "images" / tile_name, image=image)
    np.savez_compressed(split_dir / "masks" / tile_name, mask=mask)


def _build_npz_train_dataset(tmp_path: Path, positive_count: int, negative_count: int) -> ArchaeologyDataset:
    split_dir = tmp_path / "train"
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "masks").mkdir(parents=True, exist_ok=True)

    for idx in range(positive_count):
        _write_npz_tile(split_dir, f"pos_{idx}.npz", positive=True)
    for idx in range(negative_count):
        _write_npz_tile(split_dir, f"neg_{idx}.npz", positive=False)

    return ArchaeologyDataset(split_dir, augment=False, file_format="npz")


def test_train_negative_ratio_sampling_keeps_all_positive_and_target_negative(tmp_path: Path) -> None:
    dataset = _build_npz_train_dataset(tmp_path, positive_count=2, negative_count=5)

    selected_indices, stats = _select_train_indices_by_neg_pos_ratio(
        train_dataset=dataset,
        neg_to_pos_ratio=1.0,
        seed=42,
    )

    assert stats["positive_samples"] == 2
    assert stats["negative_samples"] == 5
    assert stats["target_negative_keep"] == 2
    assert stats["selected_negative_samples"] == 2
    assert stats["selected_total_samples"] == 4
    selected_names = {dataset.image_files[i].name for i in selected_indices}
    assert {"pos_0.npz", "pos_1.npz"}.issubset(selected_names)
    assert len([name for name in selected_names if name.startswith("neg_")]) == 2


def test_train_negative_ratio_sampling_zero_ratio_keeps_only_positive(tmp_path: Path) -> None:
    dataset = _build_npz_train_dataset(tmp_path, positive_count=3, negative_count=4)

    selected_indices, stats = _select_train_indices_by_neg_pos_ratio(
        train_dataset=dataset,
        neg_to_pos_ratio=0.0,
        seed=42,
    )

    assert stats["target_negative_keep"] == 0
    assert stats["selected_negative_samples"] == 0
    assert stats["selected_total_samples"] == 3
    selected_names = {dataset.image_files[i].name for i in selected_indices}
    assert all(name.startswith("pos_") for name in selected_names)


def test_train_negative_ratio_sampling_rejects_invalid_values(tmp_path: Path) -> None:
    dataset = _build_npz_train_dataset(tmp_path, positive_count=1, negative_count=1)

    with pytest.raises(ValueError, match="train_neg_to_pos_ratio"):
        _select_train_indices_by_neg_pos_ratio(
            train_dataset=dataset,
            neg_to_pos_ratio=-0.1,
            seed=42,
        )

    with pytest.raises(ValueError, match="train_neg_sample_seed"):
        _select_train_indices_by_neg_pos_ratio(
            train_dataset=dataset,
            neg_to_pos_ratio=1.0,
            seed=-1,
        )


def test_select_indices_by_keep_ratio_half() -> None:
    selected_indices, stats = _select_indices_by_keep_ratio(
        total_samples=10,
        keep_ratio=0.5,
        seed=42,
    )
    assert len(selected_indices) == 5
    assert stats["selected_total_samples"] == 5
    assert stats["total_samples"] == 10


def test_select_indices_by_keep_ratio_keeps_at_least_one() -> None:
    selected_indices, stats = _select_indices_by_keep_ratio(
        total_samples=3,
        keep_ratio=0.1,
        seed=42,
    )
    assert len(selected_indices) == 1
    assert stats["selected_total_samples"] == 1


def test_select_indices_by_keep_ratio_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        _select_indices_by_keep_ratio(
            total_samples=10,
            keep_ratio=0.0,
            seed=42,
        )
    with pytest.raises(ValueError, match="sample_seed"):
        _select_indices_by_keep_ratio(
            total_samples=10,
            keep_ratio=0.5,
            seed=-1,
        )


def test_compute_val_target_samples_uses_train_reference() -> None:
    target = _compute_val_target_samples(
        train_selected_samples=300,
        val_total_samples=1000,
        val_keep_ratio=0.5,
    )
    assert target == 150


def test_compute_val_target_samples_clips_to_val_total() -> None:
    target = _compute_val_target_samples(
        train_selected_samples=5000,
        val_total_samples=1200,
        val_keep_ratio=1.0,
    )
    assert target == 1200


def test_compute_val_target_samples_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="train_selected_samples"):
        _compute_val_target_samples(
            train_selected_samples=0,
            val_total_samples=100,
            val_keep_ratio=0.5,
        )
    with pytest.raises(ValueError, match="val_keep_ratio"):
        _compute_val_target_samples(
            train_selected_samples=100,
            val_total_samples=100,
            val_keep_ratio=0.0,
        )
