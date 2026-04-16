<!-- -----yash jain------ -->
# NeuroScope 3D: End-to-End Brain Tumor Segmentation Pipeline

NeuroScope 3D is a full MRI tumor segmentation project built for BraTS-style NIfTI data.
It includes:

1. FastAPI backend + browser UI for interactive 3D visualization.
2. Baseline and deep-learning segmentation engines (multimodal multiclass).
3. Deep-model and 5-fold training workflows.
4. Ensemble evaluation and prediction utilities.
5. HPC Slurm templates (Northeastern Explorer defaults).

## Training Results

Trained on BraTS 2020 (368 cases: 294 train / 74 validation) using a 3D U-Net (5.6M parameters) on an NVIDIA H200 GPU with mixed precision.

### Single Deep Model

| Metric | Score |
|--------|-------|
| Mean Dice | 0.7178 (std=0.1309) |
| Mean IoU | 0.5918 (std=0.1351) |
| Training Time | 2.13 hours |

### 5-Fold Ensemble

| Metric | Score |
|--------|-------|
| **Mean Dice** | **0.9146 (std=0.0395)** |
| **Mean IoU** | **0.8449 (std=0.0641)** |
| Training Time | ~2 hours per fold |

### Per-Class Ensemble Performance

| Tumor Region | Dice | IoU |
|--------------|------|-----|
| Necrotic/Non-Enhancing Core (label 1) | 0.7587 | 0.6411 |
| Peritumoral Edema (label 2) | 0.8255 | 0.7127 |
| Enhancing Tumor (label 4) | 0.7707 | 0.6665 |

Training environment: Python 3.12.4, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA H200.

## Quick Start (One Command)

### 1) Run full training pipeline from one script

```bash
python scripts/run_training_pipeline.py --pipeline all --amp
```

From a fresh machine (download + split + train + evaluate in one go):

```bash
python scripts/run_training_pipeline.py --download-data --download-dataset-id awsaf49/brats20-dataset-training-validation --pipeline deep --amp
```

This command downloads the dataset into `data/`, auto-detects the BraTS root, prepares splits, trains, and writes model checkpoints.

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

This project enforces one training target and input mode:

1. Multimodal input: all four MRI modalities are used together as 4 channels (`flair`, `t1`, `t1ce`, `t2`).
2. Multiclass output: predicts BraTS regions (`0/1/2/4`, mapped internally to class indices `0/1/2/3`).

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
6. `models/archive/legacy_single_modality/*`: archived legacy single-modality checkpoints.
7. `models/archive/legacy_predictions/*`: archived smoke prediction outputs.

### Prediction artifacts

1. `models/predictions/<name>_mask.nii.gz`: deep-model predicted mask.
2. `models/predictions/<name>_ensemble_mask.nii.gz`: ensemble predicted mask.

When task is `multiclass`, prediction outputs store BraTS labels (`0/1/2/4`) in the mask file.

### Report artifacts

1. `reports/eval_<timestamp>.json`: deep-model evaluation summary and per-case metrics.
2. `reports/eval_ensemble_<timestamp>.json`: ensemble evaluation summary and per-case metrics.
3. `reports/baseline_summary_<timestamp>.json`: consolidated baseline comparison for deep and ensemble runs.
4. `reports/e2e_validation_<timestamp>.json`: CLI vs API end-to-end consistency validation.
5. `reports/archive/*`: archived legacy or superseded reports.

Current evaluation reports include reproducibility metadata:

1. run configuration arguments.
2. split fingerprint (`sha256`, row count).
3. environment snapshot (Python, torch, CUDA).
4. git commit hash when available.

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
2. `training/torch_dataset.py`: multimodal BraTS PyTorch dataset loader, normalization, resize, augmentation.
3. `training/losses.py`: multiclass CE+Dice losses.
4. `training/metrics.py`: multiclass Dice/IoU metrics from logits.
5. `training/inference.py`: checkpoint loading, deep-model and ensemble inference.
6. `training/data.py`: case discovery, CSV read/write, random split, k-fold split.
7. `training/utils.py`: seed, directory, JSON, timestamp, device helpers.
8. `training/__init__.py`: package marker.

### scripts/

1. `scripts/download_brats_dataset.py`: download dataset bundle via kagglehub.
2. `scripts/prepare_brats_dataset.py`: generate all.csv, train.csv, val.csv.
3. `scripts/prepare_brats_kfold_dataset.py`: generate deterministic fold CSVs.
4. `scripts/train_brats_3d_unet.py`: train one multimodal multiclass 3D U-Net checkpoint.
5. `scripts/train_brats_3d_unet_kfold.py`: local launcher to train multiple folds.
6. `scripts/evaluate_brats_3d_unet.py`: evaluate one multimodal multiclass checkpoint on a CSV split.
7. `scripts/evaluate_brats_3d_unet_ensemble.py`: evaluate multimodal multiclass fold ensemble on a CSV split.
8. `scripts/predict_brats_3d_unet.py`: infer one case from four modality files with one checkpoint.
9. `scripts/predict_brats_3d_unet_ensemble.py`: infer one case from four modality files with multiple fold checkpoints.
10. `scripts/train_brats_3d_unet_stub.py`: compatibility helper that prints migration commands.
11. `scripts/run_training_pipeline.py`: one-command orchestrator for deep/kfold/all training flows.
12. `scripts/run_showcase.py`: one-command showcase launcher for backend + browser.

## Migration Notes (Multimodal Multiclass-Only)

This repository has been migrated to multimodal multiclass-only behavior across training, evaluation, prediction, and API inference.

1. Single-modality inference/training paths are removed.
2. Binary task mode is removed.
3. Inputs must include all four MRI modalities (`flair`, `t1`, `t1ce`, `t2`).
4. Labels are handled as BraTS multiclass (`0/1/2/4`) with internal class indices (`0/1/2/3`).
5. Legacy single-modality checkpoints are archived under `models/archive/legacy_single_modality/`.
6. If you have older custom checkpoints, retrain them with `in_channels=4` and multiclass outputs before reuse.

### configs/

1. `configs/train_brats_3d.example.args`: example argument file for deep-model training.

### hpc/

1. `hpc/README.md`: HPC script notes.
2. `hpc/slurm_train_3d_unet.sh`: deep-model training job.
3. `hpc/slurm_eval_3d_unet.sh`: deep-model evaluation job.
4. `hpc/slurm_train_3d_unet_kfold_array.sh`: 5-fold array training job.
5. `hpc/slurm_eval_ensemble_3d_unet.sh`: ensemble evaluation job.
6. `hpc/slurm_download_data.sh`: dataset download job.

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
python scripts/run_training_pipeline.py --pipeline all --amp
```

One-command from scratch (download first):

```bash
python scripts/run_training_pipeline.py --download-data --download-dataset-id awsaf49/brats20-dataset-training-validation --pipeline deep --amp
```

Useful options:

1. `--pipeline deep`: run only deep-model flow.
2. `--pipeline kfold`: run only k-fold flow.
3. `--no-amp`: disable mixed precision.
4. `--skip-deep-eval`: skip deep-model evaluation.
5. `--skip-ensemble-eval`: skip ensemble evaluation.
6. `--folds 0 1`: train selected folds only.
7. `--download-data`: fetch data before running the rest of the pipeline.
8. `--download-dataset-id <owner/dataset>`: choose which Kaggle dataset to pull.
9. `--download-output-dir <path>`: where download output is stored (default `data/`).
10. `--force-download`: force re-download even if cache/output already exists.

### B) Manual deep-model commands

```bash
python scripts/prepare_brats_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits --val-ratio 0.2 --seed 42
python scripts/train_brats_3d_unet.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 120 --batch-size 1 --num-workers 8 --target-shape 128 128 128 --amp
python scripts/evaluate_brats_3d_unet.py --csv data/splits/val.csv --checkpoint models/checkpoints/best.pt --device auto
```

### C) Manual 5-fold + ensemble commands

```bash
python scripts/prepare_brats_kfold_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits/folds --n-splits 5 --seed 42
python scripts/train_brats_3d_unet_kfold.py --fold-root data/splits/folds --checkpoint-root models/kfold --epochs 120 --batch-size 1 --num-workers 8 --target-shape 128 128 128 --amp
python scripts/evaluate_brats_3d_unet_ensemble.py --csv data/splits/val.csv --checkpoint-glob "models/kfold/fold_*/best.pt" --threshold 0.5 --device auto
```

### D) Copy-paste prediction examples (multimodal only)

Deep checkpoint prediction:

```bash
python scripts/predict_brats_3d_unet.py --flair data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz --t1 data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz --t1ce data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz --t2 data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz --checkpoint models/checkpoints/best.pt --output models/predictions/BraTS20_Training_001_mask.nii.gz --device auto
```

Ensemble prediction:

```bash
python scripts/predict_brats_3d_unet_ensemble.py --flair data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz --t1 data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz --t1ce data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz --t2 data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz --checkpoint-glob "models/kfold/fold_*/best.pt" --output models/predictions/BraTS20_Training_001_ensemble_mask.nii.gz --device auto
```

API prediction example:

```bash
curl -X POST "http://127.0.0.1:8000/api/segment" -F "flair_file=@data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz" -F "t1_file=@data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz" -F "t1ce_file=@data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz" -F "t2_file=@data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz" -F "engine=deep" -F "threshold=0.5"
```

### E) Showcase run in one command

```bash
python scripts/run_showcase.py
```

Useful options:

1. `--port 8001`: run on a different port.
2. `--no-open-browser`: do not launch browser automatically.
3. `--no-reload`: disable autoreload.
4. `--force-new-server`: ignore existing running instance and start a new one.

## Quality Checks

Local verification commands:

```bash
python -m compileall backend training scripts tests
python -m ruff check backend training scripts tests --select E9,F63,F7,F82
python -m pytest -q
```

Automated checks run on pull requests via `.github/workflows/ci.yml`.

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
2. aggregate tumor mesh: warm color.
3. multiclass overlays:
4. label `1` (necrotic/non-enhancing core): orange.
5. label `2` (edema): green.
6. label `4` (enhancing tumor): red.

## HPC (Slurm) Usage

Templates are preconfigured for Northeastern Explorer defaults:

1. partition: `gpu`
2. module stack: `explorer anaconda3/2024.06 cuda/12.8.0`
3. account placeholder: `your_nurc_project` (replace before submit)
4. GPU: H200 (specify `--gres=gpu:h200:1` for H200 nodes)

### Setup on HPC

```bash
# Clone repo and download data
git clone https://github.com/YashJ02/Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection.git
cd Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection

# Set your NURC account in all scripts
sed -i 's/your_nurc_project/YOUR_ACCOUNT/g' hpc/slurm_*.sh

# Download dataset (submit as a job, not on login node)
sbatch hpc/slurm_download_data.sh

# Fix nested directory after download
mv data/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/* data/MICCAI_BraTS2020_TrainingData/
rm -rf data/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData
```

### PyTorch CUDA Compatibility

The H200 nodes on Explorer use CUDA driver 12.8. Install PyTorch with the matching CUDA toolkit:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Submit jobs

```bash
sbatch hpc/slurm_train_3d_unet.sh
sbatch hpc/slurm_eval_3d_unet.sh
sbatch hpc/slurm_train_3d_unet_kfold_array.sh
sbatch hpc/slurm_eval_ensemble_3d_unet.sh
sbatch hpc/slurm_download_data.sh
```

### Multi-GPU Support

The training script supports `torch.nn.DataParallel` automatically. When multiple GPUs are available, it wraps the model and distributes batches across devices. Checkpoints are saved without the `module.` prefix for single-GPU compatibility.

## Troubleshooting

1. `No complete BraTS cases found`: verify each case has all modality files plus `_seg`.
2. `Deep-model checkpoint was not found`: train first or set the correct checkpoint path.
3. `No ensemble checkpoints available`: ensure files exist under `models/kfold/fold_*/best.pt`.
4. Out-of-memory on GPU: keep `--batch-size 1`, reduce `--target-shape`, or disable extra jobs.
5. Incompatible ensemble checkpoint error: ensure all selected checkpoints are multimodal multiclass (`in_channels=4`, multiclass outputs).

## Disclaimer

For research and education only. Not a clinical diagnostic tool.

## BraTS Citation

If you publish with BraTS data, cite official BraTS papers.
