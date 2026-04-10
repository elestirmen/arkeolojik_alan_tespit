import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from archeo_shared.channels import MODEL_CHANNEL_NAMES, METADATA_SCHEMA_VERSION
from training import TrainingConfig, _publish_active_artifacts


def test_training_help_returns_zero_exit_code():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "training.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"--help failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout.decode('utf-8', errors='replace')}\n"
        f"stderr:\n{result.stderr.decode('utf-8', errors='replace')}"
    )
    assert b"usage:" in result.stdout.lower()


def test_training_fails_when_all_masks_are_negative(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "training.py"

    # Minimal dataset layout
    for rel in [
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    image = np.random.rand(12, 32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)

    np.savez_compressed(tmp_path / "train/images/tile_0.npz", image=image)
    np.savez_compressed(tmp_path / "train/masks/tile_0.npz", mask=mask)
    np.savez_compressed(tmp_path / "val/images/tile_0.npz", image=image)
    np.savez_compressed(tmp_path / "val/masks/tile_0.npz", mask=mask)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data",
            str(tmp_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )

    combined_output = (
        result.stdout.decode("utf-8", errors="replace")
        + "\n"
        + result.stderr.decode("utf-8", errors="replace")
    ).lower()
    assert result.returncode != 0
    assert "pozitif etiket bulunamad" in combined_output


def test_training_npy_format_is_detected_and_validated(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "training.py"

    for rel in [
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    image = np.random.rand(12, 32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)

    np.save(tmp_path / "train/images/tile_0.npy", image)
    np.save(tmp_path / "train/masks/tile_0.npy", mask)
    np.save(tmp_path / "val/images/tile_0.npy", image)
    np.save(tmp_path / "val/masks/tile_0.npy", mask)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data",
            str(tmp_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )

    combined_output = (
        result.stdout.decode("utf-8", errors="replace")
        + "\n"
        + result.stderr.decode("utf-8", errors="replace")
    ).lower()
    assert result.returncode != 0
    assert "pozitif etiket bulunamad" in combined_output


def test_training_fails_when_class_folder_dataset_has_no_positive_tiles(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "training.py"

    for rel in [
        "train/Negative",
        "val/Negative",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    image = np.random.rand(12, 32, 32).astype(np.float32)
    np.savez_compressed(tmp_path / "train/Negative/tile_0.npz", image=image)
    np.savez_compressed(tmp_path / "val/Negative/tile_0.npz", image=image)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data",
            str(tmp_path),
            "--task",
            "tile_classification",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )

    combined_output = (
        result.stdout.decode("utf-8", errors="replace")
        + "\n"
        + result.stderr.decode("utf-8", errors="replace")
    ).lower()
    assert result.returncode != 0
    assert "pozitif etiket bulunamad" in combined_output


def test_publish_active_artifacts_writes_model_metadata_and_manifest(tmp_path):
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()
    metadata_path = data_dir / "metadata.json"
    metadata_payload = {
        "tile_size": 256,
        "overlap": 64,
        "bands": "1,2,3,4,5",
        "input_file": "input.tif",
        "mask_file": "mask.tif",
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    best_model_path = tmp_path / "best_model.pth"
    best_model_path.write_bytes(b"dummy-checkpoint")

    config = TrainingConfig(
        data_dir=data_dir,
        task_type="tile_classification",
        encoder="resnet50",
        in_channels=12,
        channel_names=tuple(MODEL_CHANNEL_NAMES),
        active_dir=tmp_path / "active",
        source_metadata_path=metadata_path,
        source_metadata=metadata_payload,
    )

    published = _publish_active_artifacts(config=config, best_model_path=best_model_path)

    assert published["model"].exists()
    assert published["training_metadata"].exists()
    assert published["manifest"].exists()
    assert published["model"].read_bytes() == b"dummy-checkpoint"

    training_metadata = json.loads(published["training_metadata"].read_text(encoding="utf-8"))
    assert training_metadata["schema_version"] == METADATA_SCHEMA_VERSION
    assert training_metadata["task_type"] == "tile_classification"
    assert training_metadata["tile_size"] == 256
    assert training_metadata["overlap"] == 64
    assert training_metadata["channel_names"] == list(MODEL_CHANNEL_NAMES)

    manifest = json.loads(published["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_checkpoint"] == str(best_model_path)
    assert manifest["source_training_metadata"] == str(metadata_path)
    assert manifest["tile_size"] == 256
    assert manifest["overlap"] == 64
    assert manifest["bands"] == "1,2,3,4,5"
