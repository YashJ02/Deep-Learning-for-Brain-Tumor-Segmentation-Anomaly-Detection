#!/bin/bash
# -----yash jain------
#SBATCH --job-name=brats3d-ens-eval
#SBATCH --partition=gpu
#SBATCH --account=your_nurc_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ensemble_eval_%j.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection}
cd "$PROJECT_DIR"

mkdir -p logs reports

module purge
module load explorer anaconda3/2024.06 cuda/12.8.0

source .venv/bin/activate

python scripts/evaluate_brats_3d_unet_ensemble.py \
  --csv data/splits/val.csv \
  --checkpoint-glob "models/kfold/fold_*/best.pt" \
  --threshold 0.5 \
  --device auto
