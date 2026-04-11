<!-- -----yash jain------ -->
# HPC Job Templates (Northeastern Explorer)

This folder contains Slurm templates tailored to Northeastern Explorer docs.

Applied defaults in templates:

1. Partition: `gpu`
2. Account directive: `#SBATCH --account=your_nurc_project`
3. Module stack: `module load explorer anaconda3/2024.06 cuda/12.1.1`
4. Training scripts in this folder run multimodal multiclass mode by default (4-channel input, BraTS multiclass output).

Update `your_nurc_project` in each script to your allocation account.

## Deep-model training job

Submit:

```bash
sbatch hpc/slurm_train_3d_unet.sh
```

Script: `hpc/slurm_train_3d_unet.sh`

## Deep-model evaluation job

Submit:

```bash
sbatch hpc/slurm_eval_3d_unet.sh
```

Script: `hpc/slurm_eval_3d_unet.sh`

## 5-fold training job array

Submit:

```bash
sbatch hpc/slurm_train_3d_unet_kfold_array.sh
```

Script: `hpc/slurm_train_3d_unet_kfold_array.sh`

Details:

1. Uses `#SBATCH --array=0-4%5` (one task per fold).
2. Creates fold CSVs if missing using `scripts/prepare_brats_kfold_dataset.py`.
3. Trains fold checkpoints at `models/kfold/fold_<id>/best.pt`.

## Ensemble evaluation job

Submit:

```bash
sbatch hpc/slurm_eval_ensemble_3d_unet.sh
```

Script: `hpc/slurm_eval_ensemble_3d_unet.sh`

Details:

1. Loads all fold checkpoints from `models/kfold/fold_*/best.pt`.
2. Runs ensemble evaluation via `scripts/evaluate_brats_3d_unet_ensemble.py`.
3. Writes report JSON to `reports/`.

## Practical notes

1. `gpu` partition allows one GPU per job by default; use `multigpu` only if your project has access.
2. Job arrays on Explorer support concurrency throttling with `%` and a max concurrent array count of 50 per account.
3. If you need H200 specifically, change `--gres=gpu:1` to `--gres=gpu:h200:1`.
