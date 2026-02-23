from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import BCELoss, validate


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
