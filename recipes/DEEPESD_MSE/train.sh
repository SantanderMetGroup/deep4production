#!/bin/bash
# Train the DEEPESD_MSE run. This script lives in the run directory (id_dir) alongside
# train.yaml / inference.yaml; d4p-train writes all outputs under ./outputs/.
#
# The recipe's run_ID must equal this directory's name (DEEPESD_MSE) and its
# output_dir must be this directory's parent, so id_dir = output_dir/run_ID
# resolves back to here.
#
#SBATCH --job-name=deepesd_mse_train
#SBATCH --output=./train.out
#SBATCH --error=./train.err
##SBATCH --partition=your_partition
##SBATCH --gres=gpu:1
##SBATCH --time=72:00:00

set -e
cd "$(dirname "$0")"

# --- Activate your environment (edit for your setup) ------------------------
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate d4p
# module load pixi && pixi shell

d4p-train ./train.yaml
