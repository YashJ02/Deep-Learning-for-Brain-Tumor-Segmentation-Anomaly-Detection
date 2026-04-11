"""Create train/validation CSV splits for BraTS training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import discover_brats_cases, split_cases, write_split_csv


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
    parser = argparse.ArgumentParser(description="Prepare BraTS train/validation splits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="BraTS dataset root (folder containing case directories).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "splits",
        help="Folder where split CSV files will be written.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio in range (0,1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    if args.data_root is None:
        default_root = project_root / "data" / "MICCAI_BraTS2020_TrainingData"
        data_root = default_root if default_root.exists() else autodetect_data_root(project_root / "data")
    else:
        data_root = args.data_root.resolve()

    records = discover_brats_cases(data_root)
    train_records, val_records = split_cases(records, val_ratio=args.val_ratio, seed=args.seed)

    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_split_csv(records, output_dir / "all.csv")
    write_split_csv(train_records, output_dir / "train.csv")
    write_split_csv(val_records, output_dir / "val.csv")

    print("BraTS split generation complete")
    print(f"Dataset root: {data_root}")
    print(f"Total cases: {len(records)}")
    print(f"Train cases: {len(train_records)}")
    print(f"Validation cases: {len(val_records)}")
    print(f"CSV output folder: {output_dir}")
    print("Generated files: all.csv, train.csv, val.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
