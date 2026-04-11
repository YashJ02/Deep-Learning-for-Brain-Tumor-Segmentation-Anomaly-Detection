"""Download BraTS 2015 dataset via kagglehub with clear progress logs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import kagglehub


DATASET_ID = "andrewmvd/brain-tumor-segmentation-in-mri-brats-2015"


def format_seconds(total_seconds: float) -> str:
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def print_step(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def summarize_download(path: str) -> None:
    root = Path(path)
    if not root.exists():
        print_step(f"Downloaded path was returned, but it does not exist: {root}")
        return

    files = [p for p in root.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    size_mb = total_size / (1024 * 1024)

    print_step(f"Path to dataset files: {root}")
    print_step(f"Total files found: {len(files)}")
    print_step(f"Total size on disk: {size_mb:.2f} MB")

    preview_count = min(10, len(files))
    if preview_count:
        print_step("Sample files:")
        for item in files[:preview_count]:
            rel = item.relative_to(root)
            print(f"  - {rel}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BraTS 2015 dataset from Kaggle using kagglehub.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists in cache.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Download directly to this folder instead of KaggleHub cache.",
    )
    args = parser.parse_args()

    print_step("Initializing KaggleHub downloader")
    print_step(f"Dataset: {DATASET_ID}")
    if args.output_dir:
        print_step(f"Output directory: {Path(args.output_dir).resolve()}")

    if not os.environ.get("KAGGLE_USERNAME") and not os.environ.get("KAGGLE_KEY"):
        print_step(
            "Kaggle credentials not found in environment. If download fails, set KAGGLE_USERNAME and KAGGLE_KEY or configure kaggle.json."
        )

    start_time = time.perf_counter()

    try:
        print_step("Starting dataset download...")
        dataset_path = kagglehub.dataset_download(
            DATASET_ID,
            force_download=args.force,
            output_dir=args.output_dir,
        )
        elapsed = time.perf_counter() - start_time
        print_step(f"Download completed in {format_seconds(elapsed)}")
        summarize_download(dataset_path)
        return 0
    except Exception as exc:  # pragma: no cover
        elapsed = time.perf_counter() - start_time
        print_step(f"Download failed after {format_seconds(elapsed)}")
        print_step(f"Error: {exc}")
        print_step("How to fix authentication:")
        print("  1. Generate API token at https://www.kaggle.com/settings", flush=True)
        print("  2. Save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json", flush=True)
        print("  3. Or set env vars KAGGLE_USERNAME and KAGGLE_KEY", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
