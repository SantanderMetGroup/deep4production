#!/bin/bash
# Run gradient input-attribution (d4p-explain) for the SONG_UNET_DET_ASYM run. Reads
# explain.yaml from this directory and writes attribution maps under
# ./outputs/xai/. Requires a trained checkpoint under ./outputs/models/.
#
# The recipe's run_ID must equal this directory's name (SONG_UNET_DET_ASYM) and its
# output_dir must be this directory's parent (id_dir = output_dir/run_ID).
#
#SBATCH --job-name=song_unet_det_asym_explain
#SBATCH --output=./explain.out
#SBATCH --error=./explain.err
##SBATCH --partition=your_partition
##SBATCH --gres=gpu:1
##SBATCH --time=02:00:00

set -e
cd "$(dirname "$0")"

# --- Activate your environment (edit for your setup) ------------------------
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate d4p
# module load pixi && pixi shell

d4p-explain ./explain.yaml
