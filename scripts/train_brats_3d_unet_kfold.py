# -----yash jain------
"""Local launcher for running 5-fold BraTS training sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_brats_3d_unet.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-fold training using train_brats_3d_unet.py")
    parser.add_argument("--fold-root", type=Path, default=Path("data") / "splits" / "folds")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("models") / "kfold")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Fold indices to run (default: all found folds).")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-latest", action="store_true", help="Resume each fold from its latest.pt if present.")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _discover_folds(fold_root: Path) -> List[int]:
    fold_indices: List[int] = []
    for item in sorted(fold_root.glob("fold_*")):
        if item.is_dir() and item.name.startswith("fold_"):
            suffix = item.name.split("fold_", maxsplit=1)[1]
            if suffix.isdigit():
                fold_indices.append(int(suffix))
    if not fold_indices:
        raise FileNotFoundError(f"No fold directories found under: {fold_root}")
    return fold_indices


def _train_command(args: argparse.Namespace, fold_index: int, fold_root: Path, checkpoint_root: Path) -> List[str]:
    fold_dir = fold_root / f"fold_{fold_index}"
    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing train/val CSV for fold {fold_index} in {fold_dir}")

    fold_checkpoint_dir = checkpoint_root / f"fold_{fold_index}"

    command = [
        str(args.python_executable),
        str(TRAIN_SCRIPT),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--checkpoint-dir",
        str(fold_checkpoint_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--ce-weight",
        str(args.ce_weight),
        "--target-shape",
        str(args.target_shape[0]),
        str(args.target_shape[1]),
        str(args.target_shape[2]),
        "--base-channels",
        str(args.base_channels),
        "--seed",
        str(args.seed + fold_index),
        "--device",
        args.device,
    ]

    if args.amp:
        command.append("--amp")

    latest_checkpoint = fold_checkpoint_dir / "latest.pt"
    if args.resume_latest and latest_checkpoint.exists():
        command.extend(["--resume", str(latest_checkpoint)])

    return command


def main() -> int:
    args = parse_args()

    fold_root = _resolve_path(args.fold_root)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    fold_indices = sorted(args.folds) if args.folds else _discover_folds(fold_root)

    print("Starting local K-fold training launcher")
    print(f"Fold root: {fold_root}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Folds: {fold_indices}")

    for fold_index in fold_indices:
        command = _train_command(args, fold_index, fold_root, checkpoint_root)
        print(f"\nRunning fold_{fold_index}")
        print("Command:")
        print(" ".join(command))

        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Fold {fold_index} failed with exit code {completed.returncode}")

    print("All requested folds completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
