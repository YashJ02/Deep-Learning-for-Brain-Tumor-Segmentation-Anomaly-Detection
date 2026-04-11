# -----yash jain------
"""Generate deterministic K-fold CSV splits for BraTS training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import discover_brats_cases, kfold_cases, write_split_csv


def autodetect_data_root(data_dir: Path) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    candidates = [path for path in data_dir.iterdir() if path.is_dir()]
    for candidate in sorted(candidates):
        if any(candidate.rglob("*_seg.nii*")):
            return candidate

    raise FileNotFoundError(
        "Could not detect a BraTS training directory automatically. "
        "Pass --data-root explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BraTS K-fold train/validation splits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="BraTS dataset root (folder containing case directories).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "splits" / "folds",
        help="Folder where fold subdirectories will be written.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.data_root is None:
        default_root = PROJECT_ROOT / "data" / "MICCAI_BraTS2020_TrainingData"
        data_root = default_root if default_root.exists() else autodetect_data_root(PROJECT_ROOT / "data")
    else:
        data_root = args.data_root.resolve()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_brats_cases(data_root)
    folds = kfold_cases(records, n_splits=args.n_splits, seed=args.seed)

    write_split_csv(records, output_dir / "all.csv")

    for fold_index, (train_records, val_records) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_split_csv(train_records, fold_dir / "train.csv")
        write_split_csv(val_records, fold_dir / "val.csv")

    print("BraTS K-fold split generation complete")
    print(f"Dataset root: {data_root}")
    print(f"Total cases: {len(records)}")
    print(f"Number of folds: {args.n_splits}")
    print(f"Output folder: {output_dir}")

    for fold_index, (train_records, val_records) in enumerate(folds):
        print(
            f"fold_{fold_index}: train={len(train_records)} val={len(val_records)} "
            f"files=({output_dir / f'fold_{fold_index}' / 'train.csv'}, {output_dir / f'fold_{fold_index}' / 'val.csv'})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
