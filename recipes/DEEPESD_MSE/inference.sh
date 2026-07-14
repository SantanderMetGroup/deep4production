#!/bin/bash
# Run inference for the DEEPESD_MSE run. Reads inference.yaml from this directory and
# writes predictions under ./outputs/predictions/. Requires a trained
# checkpoint under ./outputs/models/ (produced by train.sh).
#
# The recipe's run_ID must equal this directory's name (DEEPESD_MSE) and its
# output_dir must be this directory's parent (id_dir = output_dir/run_ID).
#
#SBATCH --job-name=deepesd_mse_infer
#SBATCH --output=./inference.out
#SBATCH --error=./inference.err
##SBATCH --partition=your_partition
##SBATCH --gres=gpu:1
##SBATCH --time=12:00:00

set -e
cd "$(dirname "$0")"

# --- Activate your environment (edit for your setup) ------------------------
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate d4p
# module load pixi && pixi shell

d4p-downscale ./inference.yaml
