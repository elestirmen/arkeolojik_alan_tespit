import subprocess
import sys
from pathlib import Path

import numpy as np


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
