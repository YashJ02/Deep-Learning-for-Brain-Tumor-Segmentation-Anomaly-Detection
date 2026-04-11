# NeuroScope 3D: End-to-End Brain Tumor Segmentation Pipeline

NeuroScope 3D is a full MRI tumor segmentation project built for BraTS-style NIfTI data.
It includes:

1. FastAPI backend + browser UI for interactive 3D visualization.
2. Baseline and deep-learning segmentation engines (binary + multiclass).
3. Deep-model and 5-fold training workflows.
4. Ensemble evaluation and prediction utilities.
5. HPC Slurm templates (Northeastern Explorer defaults).

## Quick Start (One Command)

### 1) Run full training pipeline from one script

```bash
python scripts/run_training_pipeline.py --pipeline all --task multiclass --amp
```

This command can run:

1. Train/val split generation.
2. Deep-model training.
3. Deep-model evaluation.
4. K-fold split generation.
5. K-fold training launcher.
6. Ensemble evaluation.

### 2) Run project showcase from one script

```bash
python scripts/run_showcase.py
```

This command:

1. Starts the FastAPI app.
2. Waits for health check.
3. Opens the browser to `http://localhost:8000`.

## MRI Modality Guide (BraTS)

BraTS case folders include four MRI modalities plus segmentation labels:

1. `t1`: anatomical baseline detail.
2. `t1ce`: T1 with contrast enhancement; often highlights enhancing tumor core.
3. `t2`: fluid-sensitive; edema often appears bright.
4. `flair`: suppresses CSF signal; edema/infiltrative regions become easier to see.
5. `seg`: voxel-wise tumor labels.

BraTS segmentation labels are typically encoded as:

1. `0`: background.
2. `1`: necrotic/non-enhancing tumor core.
3. `2`: peritumoral edema.
4. `4`: enhancing tumor.

This project supports two training targets:

1. `binary`: converts labels to `seg > 0` (`tumor` vs `non-tumor`).
2. `multiclass`: predicts BraTS regions separately (`0/1/2/4`, mapped internally to class indices `0/1/2/3`).

Default training mode in launchers is `multiclass`.

## Expected Dataset Layout

Each case directory should contain all five files:

```text
data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/
  BraTS20_Training_001_flair.nii or .nii.gz
  BraTS20_Training_001_t1.nii or .nii.gz
  BraTS20_Training_001_t1ce.nii or .nii.gz
  BraTS20_Training_001_t2.nii or .nii.gz
  BraTS20_Training_001_seg.nii or .nii.gz
```

Auto-detection scripts search under `data/` for folders containing `*_seg.nii*`.

## Data Files and What They Mean

### Core data folders

1. `data/MICCAI_BraTS2020_TrainingData/`: main training dataset root.
2. `data/MICCAI_BraTS_2019_Data_Training/`: optional alternate dataset root.
3. `data/MICCAI_BraTS_2018_Data_Training/`: optional alternate dataset root.
4. `data/.complete/`: marker content from dataset download tooling.

### Split CSV files

1. `data/splits/all.csv`: all discovered complete cases.
2. `data/splits/train.csv`: training subset for deep-model run.
3. `data/splits/val.csv`: validation subset for deep-model run.
4. `data/splits/train_smoke.csv`: tiny split for smoke tests.
5. `data/splits/val_smoke.csv`: tiny validation split for smoke tests.

CSV columns are:

1. `case_id`
2. `flair`
3. `t1`
4. `t1ce`
5. `t2`
6. `seg`

### K-fold split CSV files

1. `data/splits/folds/all.csv`: all cases used for fold generation.
2. `data/splits/folds/fold_0/train.csv` and `val.csv`
3. `data/splits/folds/fold_1/train.csv` and `val.csv`
4. `data/splits/folds/fold_2/train.csv` and `val.csv`
5. `data/splits/folds/fold_3/train.csv` and `val.csv`
6. `data/splits/folds/fold_4/train.csv` and `val.csv`

## Output Artifacts and Their Purpose

### Model artifacts

1. `models/checkpoints/best.pt`: best deep-model checkpoint by validation Dice.
2. `models/checkpoints/latest.pt`: most recent deep-model checkpoint.
3. `models/checkpoints/history.json`: epoch-by-epoch training history.
4. `models/brats_3d_unet_best.pt`: exported copy of best deep-model checkpoint.
5. `models/kfold/fold_<k>/best.pt`: best checkpoint for fold `k`.
6. `models/kfold_smoke/fold_<k>/*`: smoke-test fold checkpoints.
7. `models/checkpoints_smoke/*`: smoke-test deep-model artifacts.

### Prediction artifacts

1. `models/predictions/<name>_mask.nii.gz`: deep-model predicted mask.
2. `models/predictions/<name>_ensemble_mask.nii.gz`: ensemble predicted mask.

When task is `multiclass`, prediction outputs store BraTS labels (`0/1/2/4`) in the mask file.

### Report artifacts

1. `reports/eval_<timestamp>.json`: deep-model evaluation summary and per-case metrics.
2. `reports/eval_ensemble_<timestamp>.json`: ensemble evaluation summary and per-case metrics.

## Complete Repository Guide (File-by-File)

### Root

1. `README.md`: this documentation.
2. `requirements.txt`: runtime dependencies (API, preprocessing, NIfTI tools).
3. `requirements-train.txt`: runtime + PyTorch + tqdm training stack.
4. `.gitignore`: git exclusions.

### backend/app/

1. `backend/app/main.py`: FastAPI app, routes, request parsing, static frontend mount.
2. `backend/app/segmentation.py`: baseline segmentation, deep/ensemble engine selection, checkpoint discovery.
3. `backend/app/mesh.py`: marching-cubes mesh extraction and mesh downsampling.
4. `backend/app/metrics.py`: volumetric measurement calculation from predicted mask.
5. `backend/app/__init__.py`: package marker.

### frontend/

1. `frontend/index.html`: page structure and controls.
2. `frontend/styles.css`: page styling.
3. `frontend/app.js`: API calls, fold selector behavior, metrics rendering, 3D Plotly viewer.

### training/

1. `training/model.py`: 3D U-Net architecture and parameter counting helper.
2. `training/torch_dataset.py`: BraTS PyTorch dataset loader, normalization, resize, augmentation.
3. `training/losses.py`: binary BCE+Dice and multiclass CE+Dice losses.
4. `training/metrics.py`: binary and multiclass Dice/IoU metrics from logits.
5. `training/inference.py`: checkpoint loading, deep-model and ensemble inference.
6. `training/data.py`: case discovery, CSV read/write, random split, k-fold split.
7. `training/utils.py`: seed, directory, JSON, timestamp, device helpers.
8. `training/__init__.py`: package marker.

### scripts/

1. `scripts/download_brats_dataset.py`: download dataset bundle via kagglehub.
2. `scripts/prepare_brats_dataset.py`: generate all.csv, train.csv, val.csv.
3. `scripts/prepare_brats_kfold_dataset.py`: generate deterministic fold CSVs.
4. `scripts/train_brats_3d_unet.py`: train one 3D U-Net checkpoint (`--task binary|multiclass`).
5. `scripts/train_brats_3d_unet_kfold.py`: local launcher to train multiple folds.
6. `scripts/evaluate_brats_3d_unet.py`: evaluate one checkpoint on a CSV split (auto task detection).
7. `scripts/evaluate_brats_3d_unet_ensemble.py`: evaluate fold ensemble on a CSV split (supports multiclass).
8. `scripts/predict_brats_3d_unet.py`: infer one volume with one checkpoint (supports multiclass labels).
9. `scripts/predict_brats_3d_unet_ensemble.py`: infer one volume with multiple fold checkpoints.
10. `scripts/train_brats_3d_unet_stub.py`: compatibility helper that prints migration commands.
11. `scripts/run_training_pipeline.py`: one-command orchestrator for deep/kfold/all training flows.
12. `scripts/run_showcase.py`: one-command showcase launcher for backend + browser.

### configs/

1. `configs/train_brats_3d.example.args`: example argument file for deep-model training.

### hpc/

1. `hpc/README.md`: HPC script notes.
2. `hpc/slurm_train_3d_unet.sh`: deep-model training job.
3. `hpc/slurm_eval_3d_unet.sh`: deep-model evaluation job.
4. `hpc/slurm_train_3d_unet_kfold_array.sh`: 5-fold array training job.
5. `hpc/slurm_eval_ensemble_3d_unet.sh`: ensemble evaluation job.

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

PyTorch CUDA note:

1. On many systems, install torch wheels from the official PyTorch selector first.
2. Then install the remaining project requirements.

## Main Workflows

### A) Full pipeline in one command

```bash
python scripts/run_training_pipeline.py --pipeline all --task multiclass --amp
```

Useful options:

1. `--pipeline deep`: run only deep-model flow.
2. `--pipeline kfold`: run only k-fold flow.
3. `--task binary`: switch to binary target.
4. `--no-amp`: disable mixed precision.
5. `--skip-deep-eval`: skip deep-model evaluation.
6. `--skip-ensemble-eval`: skip ensemble evaluation.
7. `--folds 0 1`: train selected folds only.

### B) Manual deep-model commands

```bash
python scripts/prepare_brats_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits --val-ratio 0.2 --seed 42
python scripts/train_brats_3d_unet.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --task multiclass --epochs 120 --batch-size 1 --num-workers 8 --modality t1ce --target-shape 128 128 128 --amp
python scripts/evaluate_brats_3d_unet.py --csv data/splits/val.csv --checkpoint models/checkpoints/best.pt --task auto --device auto
```

### C) Manual 5-fold + ensemble commands

```bash
python scripts/prepare_brats_kfold_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits/folds --n-splits 5 --seed 42
python scripts/train_brats_3d_unet_kfold.py --fold-root data/splits/folds --checkpoint-root models/kfold --task multiclass --epochs 120 --batch-size 1 --num-workers 8 --modality t1ce --target-shape 128 128 128 --amp
python scripts/evaluate_brats_3d_unet_ensemble.py --csv data/splits/val.csv --checkpoint-glob "models/kfold/fold_*/best.pt" --modality t1ce --threshold 0.5 --device auto
```

### D) Showcase run in one command

```bash
python scripts/run_showcase.py
```

Useful options:

1. `--port 8001`: run on a different port.
2. `--no-open-browser`: do not launch browser automatically.
3. `--no-reload`: disable autoreload.
4. `--force-new-server`: ignore existing running instance and start a new one.

## Web API

### GET /api/health

Returns service health.

### GET /api/checkpoints

Returns checkpoint inventory:

1. deep checkpoint path + existence.
2. fold checkpoint list and count.

### POST /api/segment

Form fields:

1. `flair_file`: required `.nii` or `.nii.gz` file.
2. `t1_file`: required `.nii` or `.nii.gz` file.
3. `t1ce_file`: required `.nii` or `.nii.gz` file.
4. `t2_file`: required `.nii` or `.nii.gz` file.
5. `engine`: `all`, `auto`, `deep`, `ensemble`, `baseline`.
6. `threshold`: probability threshold in `(0, 1)`.
7. `ensemble_folds`: comma-separated fold indices (optional; used by `ensemble`/`all`).

Response includes:

1. input metadata.
2. inference metadata.
3. aggregate tumor metrics (`metrics`).
4. class-wise metrics (`class_metrics`) for labels `1/2/4` when multiclass predictions are available.
5. `mesh`: aggregate tumor mesh.
6. `brain_mesh`: brain surface mesh.
7. `class_meshes`: class-colored meshes for necrotic core, edema, and enhancing tumor (multiclass mode).

## 3D Viewer Color Convention

Current UI rendering:

1. brain mesh: blue.
2. aggregate tumor mesh: warm color (binary fallback).
3. multiclass overlays:
4. label `1` (necrotic/non-enhancing core): orange.
5. label `2` (edema): green.
6. label `4` (enhancing tumor): red.

## HPC (Slurm) Usage

Templates are preconfigured for Northeastern Explorer style defaults:

1. partition: `gpu`
2. module stack: `explorer anaconda3/2024.06 cuda/12.1.1`
3. account placeholder: `your_nurc_project` (replace before submit)

Submit examples:

```bash
sbatch hpc/slurm_train_3d_unet.sh
sbatch hpc/slurm_eval_3d_unet.sh
sbatch hpc/slurm_train_3d_unet_kfold_array.sh
sbatch hpc/slurm_eval_ensemble_3d_unet.sh
```

## Troubleshooting

1. `No complete BraTS cases found`: verify each case has all modality files plus `_seg`.
2. `Deep-model checkpoint was not found`: train first or set the correct checkpoint path.
3. `No ensemble checkpoints available`: ensure files exist under `models/kfold/fold_*/best.pt`.
4. Out-of-memory on GPU: keep `--batch-size 1`, reduce `--target-shape`, or disable extra jobs.
5. Mixed-task ensemble error: ensure all checkpoints in the ensemble are trained with the same `--task` mode.

## Disclaimer

For research and education only. Not a clinical diagnostic tool.

## BraTS Citation

If you publish with BraTS data, cite official BraTS papers.
