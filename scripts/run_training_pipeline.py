"""One-command orchestrator for BraTS training workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _has_seg_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.rglob("*_seg.nii*"))


def autodetect_data_root(search_root: Path) -> Path | None:
    if not search_root.exists() or not search_root.is_dir():
        return None

    # First pass: look for directories that directly contain case folders.
    candidates = [search_root, *sorted(path for path in search_root.iterdir() if path.is_dir())]
    for candidate in candidates:
        for child in sorted(path for path in candidate.iterdir() if path.is_dir()):
            if any(child.glob("*_seg.nii*")):
                return candidate

    # Fallback: infer root from the first discovered segmentation file.
    first_seg = next(search_root.rglob("*_seg.nii*"), None)
    if first_seg is None:
        return None
    case_dir = first_seg.parent
    if case_dir.parent.exists():
        return case_dir.parent
    return case_dir


def run_command(command: list[str]) -> None:
    pretty = " ".join(shlex.quote(part) for part in command)
    log(f"Running: {pretty}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete BraTS training pipelines from one script."
    )

    parser.add_argument(
        "--pipeline",
        choices=["deep", "kfold", "all"],
        default="all",
        help="deep: one deep-model checkpoint, kfold: 5-fold only, all: both",
    )

    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run child scripts.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data") / "MICCAI_BraTS2020_TrainingData",
        help="BraTS dataset root with case folders.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the dataset before split generation and training.",
    )
    parser.add_argument(
        "--download-dataset-id",
        type=str,
        default="awsaf49/brats20-dataset-training-validation",
        help="Kaggle dataset id used by download_brats_dataset.py.",
    )
    parser.add_argument(
        "--download-output-dir",
        type=Path,
        default=Path("data"),
        help="Output folder for dataset download when --download-data is enabled.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download when --download-data is enabled.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data") / "splits",
        help="Directory for train/val split CSVs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models") / "checkpoints",
        help="Directory for deep-model checkpoints.",
    )
    parser.add_argument(
        "--kfold-checkpoint-root",
        type=Path,
        default=Path("models") / "kfold",
        help="Directory for k-fold checkpoints.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for evaluation JSON reports.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="*", default=None)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision where supported (default: true).",
    )

    parser.add_argument(
        "--resume-latest-kfold",
        action="store_true",
        help="Resume each selected fold from latest.pt if present.",
    )
    parser.add_argument(
        "--skip-deep-eval",
        action="store_true",
        help="Skip evaluate_brats_3d_unet.py step.",
    )
    parser.add_argument(
        "--skip-ensemble-eval",
        action="store_true",
        help="Skip evaluate_brats_3d_unet_ensemble.py step.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    python_executable = _resolve(args.python_executable)
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")

    configured_data_root = _resolve(args.data_root)
    download_output_dir = _resolve(args.download_output_dir)

    if args.download_data:
        download_output_dir.mkdir(parents=True, exist_ok=True)
        download_command = [
            str(python_executable),
            str(SCRIPTS_DIR / "download_brats_dataset.py"),
            "--dataset-id",
            args.download_dataset_id,
            "--output-dir",
            str(download_output_dir),
        ]
        if args.force_download:
            download_command.append("--force")
        run_command(download_command)

    data_root = configured_data_root
    if not _has_seg_files(data_root):
        detection_search_roots = [download_output_dir, configured_data_root.parent, PROJECT_ROOT / "data"]
        deduped_search_roots: list[Path] = []
        seen = set()
        for root in detection_search_roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key in seen:
                continue
            seen.add(key)
            deduped_search_roots.append(root)

        detected_root: Path | None = None
        for search_root in deduped_search_roots:
            detected_root = autodetect_data_root(search_root)
            if detected_root is not None and _has_seg_files(detected_root):
                break

        if detected_root is not None and _has_seg_files(detected_root):
            data_root = detected_root.resolve()
            log(
                "Configured --data-root does not contain complete BraTS files. "
                f"Using auto-detected dataset root: {data_root}"
            )

    splits_dir = _resolve(args.splits_dir)
    checkpoint_dir = _resolve(args.checkpoint_dir)
    kfold_checkpoint_root = _resolve(args.kfold_checkpoint_root)
    report_dir = _resolve(args.report_dir)

    if not data_root.exists() or not _has_seg_files(data_root):
        raise FileNotFoundError(
            "BraTS dataset root not found or incomplete. Expected case folders with all modalities and *_seg files under: "
            f"{data_root}. Use --data-root or enable --download-data."
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    kfold_checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    log("Starting one-command training pipeline")
    log(f"Pipeline mode: {args.pipeline}")
    log(f"Data root: {data_root}")

    # Always generate train/val splits. They are used by deep-model training and ensemble eval.
    run_command(
        [
            str(python_executable),
            str(SCRIPTS_DIR / "prepare_brats_dataset.py"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(splits_dir),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
    )

    if args.pipeline in {"deep", "all"}:
        train_command = [
            str(python_executable),
            str(SCRIPTS_DIR / "train_brats_3d_unet.py"),
            "--train-csv",
            str(splits_dir / "train.csv"),
            "--val-csv",
            str(splits_dir / "val.csv"),
            "--checkpoint-dir",
            str(checkpoint_dir),
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
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.amp:
            train_command.append("--amp")

        run_command(train_command)

        if not args.skip_deep_eval:
            run_command(
                [
                    str(python_executable),
                    str(SCRIPTS_DIR / "evaluate_brats_3d_unet.py"),
                    "--csv",
                    str(splits_dir / "val.csv"),
                    "--checkpoint",
                    str(checkpoint_dir / "best.pt"),
                    "--report-dir",
                    str(report_dir),
                    "--device",
                    args.device,
                    "--ce-weight",
                    str(args.ce_weight),
                    "--target-shape",
                    str(args.target_shape[0]),
                    str(args.target_shape[1]),
                    str(args.target_shape[2]),
                ]
            )

    if args.pipeline in {"kfold", "all"}:
        folds_dir = splits_dir / "folds"

        run_command(
            [
                str(python_executable),
                str(SCRIPTS_DIR / "prepare_brats_kfold_dataset.py"),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(folds_dir),
                "--n-splits",
                str(args.n_splits),
                "--seed",
                str(args.seed),
            ]
        )

        kfold_command = [
            str(python_executable),
            str(SCRIPTS_DIR / "train_brats_3d_unet_kfold.py"),
            "--fold-root",
            str(folds_dir),
            "--checkpoint-root",
            str(kfold_checkpoint_root),
            "--python-executable",
            str(python_executable),
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
            str(args.seed),
            "--device",
            args.device,
        ]

        if args.folds:
            kfold_command.extend(["--folds", *[str(index) for index in args.folds]])
        if args.amp:
            kfold_command.append("--amp")
        if args.resume_latest_kfold:
            kfold_command.append("--resume-latest")

        run_command(kfold_command)

        if not args.skip_ensemble_eval:
            run_command(
                [
                    str(python_executable),
                    str(SCRIPTS_DIR / "evaluate_brats_3d_unet_ensemble.py"),
                    "--csv",
                    str(splits_dir / "val.csv"),
                    "--checkpoint-glob",
                    str(kfold_checkpoint_root / "fold_*" / "best.pt"),
                    "--report-dir",
                    str(report_dir),
                    "--threshold",
                    str(args.threshold),
                    "--device",
                    args.device,
                ]
            )

    log("Pipeline completed successfully")
    log(f"Deep-model checkpoints: {checkpoint_dir}")
    log(f"K-fold checkpoints: {kfold_checkpoint_root}")
    log(f"Reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
