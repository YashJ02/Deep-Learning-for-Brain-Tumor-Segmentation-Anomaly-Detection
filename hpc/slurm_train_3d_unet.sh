#!/bin/bash
#SBATCH --job-name=brats3d-train
#SBATCH --partition=gpu
#SBATCH --account=your_nurc_project
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection}
cd "$PROJECT_DIR"

mkdir -p logs reports models/checkpoints data/splits

# Northeastern Explorer module stack (NURC docs).
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt

python scripts/prepare_brats_dataset.py \
  --data-root data/MICCAI_BraTS2020_TrainingData \
  --output-dir data/splits \
  --val-ratio 0.2 \
  --seed 42

python scripts/train_brats_3d_unet.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --epochs 150 \
  --batch-size 1 \
  --num-workers 8 \
  --target-shape 128 128 128 \
  --amp
