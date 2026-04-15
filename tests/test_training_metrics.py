from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import (
    ArchaeologyDataset,
    BCELoss,
    _build_auto_val_holdout_from_dataset,
    _build_auto_val_holdout_indices,
    _compute_val_target_samples,
    _infer_file_format,
    _resolve_classification_folder_split_counts,
    _resolve_auto_val_holdout_ratio,
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


def _build_class_folder_train_dataset(
    tmp_path: Path,
    positive_count: int,
    negative_count: int,
) -> ArchaeologyDataset:
    split_dir = tmp_path / "train"
    (split_dir / "Positive").mkdir(parents=True, exist_ok=True)
    (split_dir / "Negative").mkdir(parents=True, exist_ok=True)

    image = np.zeros((12, 8, 8), dtype=np.float32)
    for idx in range(positive_count):
        np.savez_compressed(split_dir / "Positive" / f"pos_{idx}.npz", image=image)
    for idx in range(negative_count):
        np.savez_compressed(split_dir / "Negative" / f"neg_{idx}.npz", image=image)

    return ArchaeologyDataset(
        split_dir,
        augment=False,
        file_format="npz",
        task_type="tile_classification",
    )


def test_classification_folder_layout_loads_labels_without_masks(tmp_path: Path) -> None:
    dataset = _build_class_folder_train_dataset(tmp_path, positive_count=2, negative_count=1)

    assert len(dataset) == 3
    labels = [float(dataset[idx][1].item()) for idx in range(len(dataset))]
    assert labels.count(1.0) == 2
    assert labels.count(0.0) == 1
    assert dataset.tile_labels == [1.0, 1.0, 0.0]


def test_infer_file_format_allows_empty_val_for_classification_layout(tmp_path: Path) -> None:
    for rel in [
        "train/Positive",
        "train/Negative",
        "val/Positive",
        "val/Negative",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    image = np.zeros((12, 8, 8), dtype=np.float32)
    np.savez_compressed(tmp_path / "train/Positive" / "pos_0.npz", image=image)
    np.savez_compressed(tmp_path / "train/Negative" / "neg_0.npz", image=image)

    assert _infer_file_format(tmp_path, allow_missing_val=True) == "npz"


def test_resolve_classification_folder_split_counts_prefers_real_files_over_stale_manifest(
    tmp_path: Path,
) -> None:
    for rel in [
        "train/Positive",
        "train/Negative",
        "val/Positive",
        "val/Negative",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    image = np.zeros((12, 8, 8), dtype=np.float32)
    np.savez_compressed(tmp_path / "train/Positive" / "pos_0.npz", image=image)
    np.savez_compressed(tmp_path / "train/Negative" / "neg_0.npz", image=image)

    with open(tmp_path / "tile_labels.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["tile_name", "split", "tile_label", "positive_ratio"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_name": "pos_0",
                "split": "train",
                "tile_label": "1",
                "positive_ratio": "1.0",
            }
        )
        writer.writerow(
            {
                "tile_name": "neg_0",
                "split": "train",
                "tile_label": "0",
                "positive_ratio": "0.0",
            }
        )
        writer.writerow(
            {
                "tile_name": "ghost_val_pos",
                "split": "val",
                "tile_label": "1",
                "positive_ratio": "1.0",
            }
        )

    train_counts, train_source = _resolve_classification_folder_split_counts(
        tmp_path / "train",
        "npz",
    )
    val_counts, val_source = _resolve_classification_folder_split_counts(
        tmp_path / "val",
        "npz",
    )

    assert train_counts == (1, 1)
    assert train_source == "manifest"
    assert val_counts == (0, 0)
    assert val_source == "class_dirs"


def test_auto_val_holdout_indices_are_stratified() -> None:
    train_indices, val_indices, stats = _build_auto_val_holdout_indices(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        holdout_ratio=0.33,
        seed=42,
    )

    assert set(train_indices).isdisjoint(val_indices)
    assert sorted(train_indices + val_indices) == list(range(6))
    assert stats["train_positive_samples"] >= 1
    assert stats["val_positive_samples"] >= 1
    assert stats["val_negative_samples"] >= 1


def test_auto_val_holdout_from_segmentation_dataset_is_stratified(
    tmp_path: Path,
) -> None:
    dataset = _build_npz_train_dataset(tmp_path, positive_count=3, negative_count=3)

    train_indices, val_indices, stats = _build_auto_val_holdout_from_dataset(
        dataset,
        holdout_ratio=0.33,
        seed=42,
    )

    assert set(train_indices).isdisjoint(val_indices)
    assert sorted(train_indices + val_indices) == list(range(6))
    assert stats["train_positive_samples"] >= 1
    assert stats["val_positive_samples"] >= 1
    assert stats["val_negative_samples"] >= 1


def test_auto_val_holdout_requires_two_positive_tiles() -> None:
    with pytest.raises(ValueError, match="en az 2 pozitif tile"):
        _build_auto_val_holdout_indices(
            [1.0, 0.0, 0.0],
            holdout_ratio=0.2,
            seed=42,
        )


def test_resolve_auto_val_holdout_ratio_uses_metadata_when_available() -> None:
    ratio = _resolve_auto_val_holdout_ratio(
        {
            "train_ratio": 0.8,
            "val_ratio": 0.2,
        }
    )

    assert abs(ratio - 0.2) < 1e-9


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


def test_train_negative_ratio_sampling_respects_allowed_base_indices(tmp_path: Path) -> None:
    dataset = _build_npz_train_dataset(tmp_path, positive_count=2, negative_count=3)
    allowed_names = {dataset.image_files[i].name for i in [0, 2, 4]}

    selected_indices, stats = _select_train_indices_by_neg_pos_ratio(
        train_dataset=dataset,
        neg_to_pos_ratio=None,
        seed=42,
        allowed_base_indices=[0, 2, 4],
    )

    assert stats["total_samples"] == 3
    selected_names = {dataset.image_files[i].name for i in selected_indices}
    assert selected_names == allowed_names


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
