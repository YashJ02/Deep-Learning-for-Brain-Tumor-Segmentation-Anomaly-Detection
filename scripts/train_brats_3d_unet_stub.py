# -----yash jain------
"""Compatibility wrapper that points to the full training pipeline."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    print("The training stub has been replaced by the full CUDA pipeline.")
    print("Run the following commands from project root:")
    print("1) python scripts/prepare_brats_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData")
    print("2) python scripts/train_brats_3d_unet.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --amp")
    print("3) python scripts/evaluate_brats_3d_unet.py --checkpoint models/checkpoints/best.pt")
    print(f"Project root: {project_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
