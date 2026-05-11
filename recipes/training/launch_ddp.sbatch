#!/bin/bash
# ============================================================================
# Multi-GPU / multi-node DDP launcher for deep4production
# ----------------------------------------------------------------------------
# Usage:
#   sbatch --nodes=$NUM_NODES --ntasks-per-node=$GPUS_PER_NODE \
#          --gres=gpu:$GPUS_PER_NODE launch_ddp.sbatch path/to/recipe.yaml
#
# The recipe's ``hardware`` block must match the SLURM allocation:
#   hardware:
#     num_nodes: $NUM_NODES
#     gpus_per_node: $GPUS_PER_NODE
#
# ``srun`` spawns one task per GPU; each task imports d4p, reads SLURM_PROCID
# / SLURM_LOCALID, joins the NCCL process group, and trains a shard of the
# dataset. Rank 0 owns checkpoints / MLflow.
# ============================================================================

#SBATCH --job-name=d4p-ddp
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Override --nodes / --ntasks-per-node / --gres on the sbatch CLI to match
# the recipe's hardware block.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch launch_ddp.sbatch path/to/recipe.yaml" >&2
    exit 1
fi
RECIPE="$1"

# Pick a free-ish port deterministically from the job id so concurrent jobs
# don't collide on the default 29500.
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))

# One task per GPU; each task discovers its rank via SLURM_PROCID.
srun --kill-on-bad-exit=1 d4p-train "$RECIPE"
