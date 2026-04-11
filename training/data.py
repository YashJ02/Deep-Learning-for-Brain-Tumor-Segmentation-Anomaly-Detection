"""Dataset discovery and split utilities for BraTS data."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_MODALITY_SUFFIX = {
    "flair": "_flair",
    "t1": "_t1",
    "t1ce": "_t1ce",
    "t2": "_t2",
    "seg": "_seg",
}


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _resolve_nifti_path(case_dir: Path, stem_with_suffix: str) -> Path | None:
    for extension in (".nii.gz", ".nii"):
        candidate = case_dir / f"{stem_with_suffix}{extension}"
        if candidate.exists():
            return candidate
    return None


def discover_brats_cases(dataset_root: Path | str) -> List[Dict[str, str]]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    seg_files = sorted(dataset_root.rglob("*_seg.nii*"))
    if not seg_files:
        raise FileNotFoundError(f"No BraTS segmentation files found under: {dataset_root}")

    records: List[Dict[str, str]] = []
    for seg_path in seg_files:
        stem = _strip_nii_suffix(seg_path.name)
        prefix = stem[:-4] if stem.endswith("_seg") else stem
        case_dir = seg_path.parent

        row: Dict[str, str] = {"case_id": case_dir.name}
        missing_key = None

        for key, suffix in _MODALITY_SUFFIX.items():
            candidate = _resolve_nifti_path(case_dir, f"{prefix}{suffix}")
            if candidate is None:
                missing_key = key
                break
            row[key] = str(candidate.resolve())

        if missing_key is None:
            records.append(row)

    if not records:
        raise RuntimeError("No complete BraTS cases found (missing one or more modalities).")

    records.sort(key=lambda item: item["case_id"])
    return records


def write_split_csv(records: Sequence[Dict[str, str]], output_csv: Path | str) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["case_id", "flair", "t1", "t1ce", "t2", "seg"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})


def read_split_csv(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = [dict(row) for row in reader]

    if not records:
        raise RuntimeError(f"Split CSV is empty: {path}")
    return records


def split_cases(
    records: Sequence[Dict[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("Need at least 2 cases to create train/val split")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    val_count = min(max(val_count, 1), len(shuffled) - 1)

    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    return train_records, val_records


def kfold_cases(
    records: Sequence[Dict[str, str]],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(records) < n_splits:
        raise ValueError(f"Need at least {n_splits} records for {n_splits}-fold split")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    fold_sizes = [len(shuffled) // n_splits] * n_splits
    for idx in range(len(shuffled) % n_splits):
        fold_sizes[idx] += 1

    folds: List[List[Dict[str, str]]] = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(shuffled[start:end])
        start = end

    result: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = []
    for fold_index in range(n_splits):
        val_records = folds[fold_index]
        train_records: List[Dict[str, str]] = []
        for idx, fold_records in enumerate(folds):
            if idx != fold_index:
                train_records.extend(fold_records)
        result.append((train_records, val_records))

    return result


def summarize_cases(records: Iterable[Dict[str, str]]) -> Dict[str, int]:
    count = 0
    for _ in records:
        count += 1
    return {"cases": count}
