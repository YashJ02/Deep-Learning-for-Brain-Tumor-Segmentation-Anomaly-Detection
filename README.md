# NeuroScope 3D: CUDA Brain Tumor Segmentation Pipeline

This project is a full 3D MRI tumor segmentation pipeline for BraTS-style NIfTI data.

It includes:

1. FastAPI web application with 3D mesh visualization.
2. Dataset split generation for BraTS folders.
3. CUDA-ready 3D U-Net training scripts (single model + 5-fold launcher).
4. Validation and report generation.
5. Single-volume prediction CLI (single model + fold-ensemble).
6. HPC Slurm templates for Northeastern Explorer, including job arrays.

## Repository Layout

- `backend/app/main.py`: API routes and web UI hosting.
- `backend/app/segmentation.py`: baseline segmentation + deep checkpoint auto-switch.
- `backend/app/metrics.py`: volumetric metrics from predicted mask.
- `backend/app/mesh.py`: marching-cubes mesh extraction for viewer.
- `frontend/`: browser UI (upload + metrics + interactive 3D mesh).
- `training/model.py`: 3D U-Net architecture.
- `training/torch_dataset.py`: PyTorch dataset loader for BraTS volumes.
- `training/losses.py`: BCE + Dice training loss.
- `training/metrics.py`: Dice and IoU metrics.
- `training/inference.py`: checkpoint loading and 3D inference utilities.
- `scripts/prepare_brats_dataset.py`: generate `all.csv`, `train.csv`, `val.csv`.
- `scripts/prepare_brats_kfold_dataset.py`: generate fold CSV files under `data/splits/folds/`.
- `scripts/train_brats_3d_unet.py`: main CUDA training entrypoint.
- `scripts/train_brats_3d_unet_kfold.py`: local multi-fold launcher.
- `scripts/evaluate_brats_3d_unet.py`: validation report script.
- `scripts/predict_brats_3d_unet.py`: prediction for one input NIfTI file.
- `scripts/evaluate_brats_3d_unet_ensemble.py`: evaluate fold-ensemble predictions.
- `scripts/predict_brats_3d_unet_ensemble.py`: ensemble prediction for one input NIfTI file.
- `hpc/`: Slurm templates and usage notes.

## Installation

Base runtime dependencies (API + preprocessing):

```bash
pip install -r requirements.txt
```

Training dependencies (PyTorch + tqdm):

```bash
pip install -r requirements-train.txt
```

### CUDA PyTorch Note

On many systems you should install PyTorch using the official CUDA wheel command from:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

If needed, install `torch`, `torchvision`, and `torchaudio` first via that command, then install the rest:

```bash
pip install -r requirements.txt
pip install tqdm
```

## Dataset Download

Download BraTS bundle directly into `data/`:

```bash
python scripts/download_brats_dataset.py --output-dir data --force
```

Expected training folder example:

- `data/MICCAI_BraTS2020_TrainingData`

## Single-Model Training Pipeline

### 1) Build train/validation split CSV files

```bash
python scripts/prepare_brats_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits --val-ratio 0.2 --seed 42
```

### 2) Train 3D U-Net (CUDA)

```bash
python scripts/train_brats_3d_unet.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 120 --batch-size 1 --num-workers 8 --modality t1ce --target-shape 128 128 128 --amp
```

Outputs:

- `models/checkpoints/best.pt`
- `models/checkpoints/latest.pt`
- `models/checkpoints/history.json`
- `models/brats_3d_unet_best.pt`

### 3) Evaluate best checkpoint

```bash
python scripts/evaluate_brats_3d_unet.py --csv data/splits/val.csv --checkpoint models/checkpoints/best.pt --device auto
```

Output report:

- `reports/eval_*.json`

### 4) Predict one patient volume

```bash
python scripts/predict_brats_3d_unet.py --input data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii --checkpoint models/checkpoints/best.pt --modality-index 0 --threshold 0.5
```

Default output:

- `models/predictions/<input_name>_mask.nii.gz`

## 5-Fold Training + Ensemble Pipeline

### 1) Build 5-fold CSV split files

```bash
python scripts/prepare_brats_kfold_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits/folds --n-splits 5 --seed 42
```

### 2) Run local 5-fold launcher

```bash
python scripts/train_brats_3d_unet_kfold.py --fold-root data/splits/folds --checkpoint-root models/kfold --epochs 120 --batch-size 1 --num-workers 8 --modality t1ce --target-shape 128 128 128 --amp
```

Output checkpoints:

- `models/kfold/fold_0/best.pt`
- `models/kfold/fold_1/best.pt`
- `models/kfold/fold_2/best.pt`
- `models/kfold/fold_3/best.pt`
- `models/kfold/fold_4/best.pt`

### 3) Evaluate fold ensemble (stronger final Dice)

```bash
python scripts/evaluate_brats_3d_unet_ensemble.py --csv data/splits/val.csv --checkpoint-glob "models/kfold/fold_*/best.pt" --modality t1ce --threshold 0.5 --device auto
```

Output report:

- `reports/eval_ensemble_*.json`

### 4) Predict with fold ensemble

```bash
python scripts/predict_brats_3d_unet_ensemble.py --input data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii --checkpoint-glob "models/kfold/fold_*/best.pt" --modality-index 0 --threshold 0.5
```

## Run the Web App

Start server:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- [http://localhost:8000](http://localhost:8000)

API route:

- `POST /api/segment`

Form fields:

- `file`: `.nii` or `.nii.gz`
- `modality_index`: integer for 4D volumes
- `engine`: `auto`, `deep`, `ensemble`, `baseline`
- `threshold`: float in range `(0, 1)`

Inference behavior:

1. `auto`: uses fold ensemble if available, then single deep checkpoint, then baseline.
2. `deep`: requires a single checkpoint at `models/checkpoints/best.pt`.
3. `ensemble`: requires one or more fold checkpoints under `models/kfold/fold_*/best.pt`.
4. `baseline`: uses non-DL threshold+morphology pipeline.

Web UI enhancements:

1. The sidebar now loads fold checkpoint inventory from `/api/checkpoints`.
2. You can select which fold checkpoints are used for ensemble/auto mode.
3. Fold selector controls include `Select all`, `Clear all`, and `Refresh inventory`.
4. The metrics panel now includes ensemble confidence summary (`probability_mean`, `probability_max`, fold indices, and checkpoint count).

## HPC (Slurm) Usage

Templates:

- `hpc/slurm_train_3d_unet.sh`
- `hpc/slurm_eval_3d_unet.sh`
- `hpc/slurm_train_3d_unet_kfold_array.sh`
- `hpc/slurm_eval_ensemble_3d_unet.sh`

Submit training:

```bash
sbatch hpc/slurm_train_3d_unet.sh
```

Submit evaluation:

```bash
sbatch hpc/slurm_eval_3d_unet.sh
```

Submit 5-fold job array:

```bash
sbatch hpc/slurm_train_3d_unet_kfold_array.sh
```

Submit ensemble evaluation:

```bash
sbatch hpc/slurm_eval_ensemble_3d_unet.sh
```

The Slurm templates are prefilled for Northeastern Explorer defaults:

1. `--partition=gpu`
2. `module load explorer anaconda3/2024.06 cuda/12.1.1`
3. `--account=your_nurc_project` (replace with your actual allocation)

## Practical Notes

1. For 16 GB GPUs, keep batch size at `1` with `128^3` target shape.
2. For large GPUs (H100), increase workers and evaluate larger patch/target sizes.
3. Save checkpoints frequently and keep `best.pt` for deployment.
4. For strongest research results, run multiple folds on HPC and ensemble outputs.

## Disclaimer

This project is for research and educational use only. It is not a clinical diagnosis tool.

## BraTS Citation

If you use BraTS data in research, cite official BraTS publications:

1. Menze et al., IEEE TMI 2015, DOI: 10.1109/TMI.2014.2377694.
2. Bakas et al., Scientific Data 2017, DOI: 10.1038/sdata.2017.117.
3. Bakas et al., arXiv:1811.02629 (2018).
