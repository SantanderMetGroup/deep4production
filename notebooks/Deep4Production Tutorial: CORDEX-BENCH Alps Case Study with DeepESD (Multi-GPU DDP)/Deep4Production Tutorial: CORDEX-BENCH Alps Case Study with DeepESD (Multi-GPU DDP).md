# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with DeepESD (Multi-GPU DDP)

This tutorial picks up where the single-GPU [DeepESD tutorial](../Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) leaves off and shows how to scale the **same** DeepESD recipe to multiple GPUs — both across the GPUs of a single node and across multiple nodes — using PyTorch **DistributedDataParallel (DDP)** under SLURM.

The dataset preparation (`d4p-create`) and inference (`d4p-downscale`) steps are unchanged; only the training step is parallelized. We will:

1. Recall how DDP works conceptually and what the global batch size becomes.
2. Show the new `hardware` section in the training YAML.
3. Launch the training with `sbatch` / `srun` on SLURM.
4. Verify the run scaled correctly (loss curves, per-rank logs, MLflow).

---

## 1. What changes when we move from 1 → N GPUs?

`deep4production` uses standard **PyTorch DistributedDataParallel**:

* Each GPU runs an **identical copy** of the model with its own optimizer state.
* The training dataset is sharded across ranks by a `DistributedSampler`, so each GPU sees a different subset of samples per epoch.
* After every backward pass, gradients are **all-reduced** across ranks so every replica steps in lockstep. The mathematical effect is equivalent to training with a larger batch size.

Concretely, if the recipe has `dataloader.batch_size: 64` and we launch with `num_nodes=2` and `gpus_per_node=4`:

* per-GPU local batch: **64 samples**
* global batch (effective): **64 × 2 × 4 = 512 samples per optimizer step**
* dataset sharding: each GPU iterates over ⌈N/8⌉ samples per epoch.

Because the global batch grows linearly with the number of ranks, you typically want to scale the **learning rate** with the world size (the classic *linear LR scaling rule*) — e.g. multiply `optimizer_params.lr` by `num_nodes * gpus_per_node` as a starting point.

Only **rank 0** writes checkpoints, talks to MLflow, and emits per-epoch log lines. All other ranks are silent on stdout/stderr so SLURM logs stay readable.

---

## 2. The `hardware` section in the training YAML

The training YAML now accepts an optional `hardware` block. **Omitting it (or leaving the defaults at 1) keeps the single-GPU behavior from the original tutorial — nothing else changes.**

```yaml
##### GENERAL INFO #####
run_ID: deepesd_ddp
output_dir: ./outputs
overwrite: true

##### HARDWARE / DATA-PARALLEL CONFIGURATION (optional) #####
# When num_nodes * gpus_per_node > 1, the trainer:
#   - wraps the model in DistributedDataParallel
#   - swaps the DataLoader's sampler for DistributedSampler
#   - all-reduces train/val losses each epoch
#   - gates checkpoints / MLflow on rank 0
hardware:
  num_nodes: 2
  gpus_per_node: 4
  # master_port: 29500  # optional; set if the default port collides on your cluster

##### TRAINING DATA CONFIGURATION (uses pre-computed zarr files) #####
data:
  load_in_memory: true
  training_period: [1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1976, 1977, 1978, 1979, 1980]
  validation_period: [1967, 1975]

  predictors:
    paths:
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    variables: [u_850, u_700, u_500, v_850, v_700, v_500, t_850, t_700, t_500, q_850, q_700, q_500, z_850, z_700, z_500]
    normalizer:
      path_reference: ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
      default: mean_std
      q_850: std
      q_700: std
      q_500: std
    transform_to_2D: True

  predictands:
    paths:
      - ./AI_ready_datasets/files/RCM_1961-1980.zarr
    variables:
      - pr
    normalizer: null
    transform_to_2D: True

##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 64      # per-GPU local batch — global batch is batch_size * world_size
  shuffle: true       # handed to DistributedSampler; epoch order is re-seeded per epoch
  num_workers: 4

##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: DeepESD_BerGamma_DDP
  loss_params:
    name: NLLBerGammaLoss
    module: deep4production.deep.loss
    kwargs:
      threshold: 0.999
      ignore_nans: True
  model_params:
    name: DeepESD
    module: deep4production.deep.models.cnn.DeepESD
    kwargs:
      x_shape: [15, 16, 16]
      y_shape: [1, 128, 128]
      f_shape: [1, 128, 128]
      filters: [50, 25, 10]
      kernel_size: 3
      loss_function_name: NLLBerGammaLoss
  training_params:
    num_epochs: 1000
    patience_early_stopping: 30
    optimizer_params:
      lr: 0.0008          # 0.0001 × 8 (linear LR scaling for world_size=8)
    # ddp_find_unused_parameters: false   # default; flip to true if a forward pass skips parameters
```

That is the **only** change to the recipe. The model, dataset, normalizer, loss, and MLflow blocks are identical to the single-GPU version.

---

## 3. Launching on SLURM

The repo ships a ready-to-use sbatch helper at `recipes/training/launch_ddp.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=d4p-ddp
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
RECIPE="$1"

# Pick a per-job rendezvous port so concurrent jobs don't collide
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))

# One task per GPU; each task discovers its rank via SLURM_PROCID
srun --kill-on-bad-exit=1 d4p-train "$RECIPE"
```

Launch the run from `./example/`:

```bash
sbatch \
  --nodes=2 \
  --ntasks-per-node=4 \
  --gres=gpu:4 \
  --cpus-per-task=8 \
  ../recipes/training/launch_ddp.sbatch \
  ./training/configs/deepesd_ddp.yaml
```

The arguments must match the recipe:

| sbatch flag             | Meaning                          | Must equal recipe                |
|-------------------------|----------------------------------|----------------------------------|
| `--nodes=2`             | Number of nodes                  | `hardware.num_nodes`             |
| `--ntasks-per-node=4`   | Tasks (= GPUs) per node          | `hardware.gpus_per_node`         |
| `--gres=gpu:4`          | GPUs requested per node          | `hardware.gpus_per_node`         |
| `--cpus-per-task`       | CPU cores per dataloader worker  | choose ≥ `dataloader.num_workers`|

`d4p-train` reads `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, and `SLURM_NODELIST`, derives `MASTER_ADDR` from `scontrol show hostnames`, and calls `torch.distributed.init_process_group("nccl", ...)`. No other changes are needed; **the framework auto-detects whether it was launched under SLURM**, and falls back to `RANK`/`LOCAL_RANK`/`WORLD_SIZE` for `torchrun` users.

If `num_nodes * gpus_per_node == 1` (the default), `init_distributed()` short-circuits before touching `dist`, so single-GPU and CPU training paths remain unchanged.

---

## 4. What rank 0 prints

The log on the rank-0 task looks like this (the others are silent):

```
✻ d4p train: starting
✻ DDP initialized: backend=nccl world_size=8 (num_nodes=2 × gpus_per_node=4) ...
✻ Distributed training: rank=0/8 local_rank=0 device=cuda:0
✻ AMP enabled (dtype=torch.bfloat16, scaler=off)
✻ Model parameters: 2,317,651 (8.84 MB)
✻ Dataloaders ready (DDP: world_size=8)
✻ Starting training for 1000 epochs on CUDA:0 (DDP, world_size=8)
✻ [hh:mm:ss] Epoch 0000 | Step       143 | Time:  4.21s | LR: 8.00e-04 | Train Loss: 1.92341 | Val Loss: 1.61288 | 💾 model saved (best)
...
```

`Step` here counts **per-rank** optimizer steps. With a global batch of `batch_size * world_size`, fewer steps per epoch means each step makes more progress — set the LR accordingly.

---

## 5. Checkpoints and inference are unchanged

`save_model` is called only on rank 0 and **always saves the unwrapped (DDP-stripped) module**. That means the resulting `*_best.pt` file is byte-identical in structure to a single-GPU checkpoint, so `d4p-downscale` and `load_model` continue to work without modification.

```bash
d4p-downscale ./inference/configs/deepesd.yaml
```

If you trained on DDP and want to resume on a single GPU later, just omit the `hardware` block (or set both to 1) and point `saving_params.resume_checkpoint` at the saved file.

---

## 6. Quick sanity checks before going to many GPUs

A good rule of thumb is to **always confirm a 2-GPU run on one node first** before requesting a full multi-node allocation. On 2 GPUs you can verify:

1. The per-epoch log line is printed once (rank-0 only).
2. Train/val losses are roughly comparable to the single-GPU run when the LR is scaled.
3. The best-model checkpoint loads back into `d4p-downscale`.

For a 2-GPU one-node sanity run, the YAML and sbatch are:

```yaml
hardware:
  num_nodes: 1
  gpus_per_node: 2
```

```bash
sbatch --nodes=1 --ntasks-per-node=2 --gres=gpu:2 ../recipes/training/launch_ddp.sbatch ./training/configs/deepesd_ddp.yaml
```

Once that passes, move to the full configuration.

---

## 7. Troubleshooting

| Symptom                                                | Likely cause                                          | Fix                                                                              |
|--------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------------------|
| `RuntimeError: ... local_rank=X but only Y devices ...`| `--gres=gpu:N` smaller than `--ntasks-per-node=N`     | Match `--gres=gpu:N` to `gpus_per_node`.                                         |
| Job hangs at the first epoch                           | Firewall blocks `MASTER_PORT`                         | Set `hardware.master_port` (or `export MASTER_PORT=…`) to a permitted port.      |
| Loss diverges vs single-GPU                            | Effective batch grew without LR change                | Multiply `optimizer_params.lr` by `world_size` (linear scaling).                 |
| `find_unused_parameters` warning                       | A forward pass leaves some parameters un-touched      | Set `model_info.training_params.ddp_find_unused_parameters: true`.               |
| Non-rank-0 logs flood the SLURM file                   | `setup_logging` ran before DDP init                   | Verify you're using the shipped `d4p-train` CLI (it silences non-zero ranks).    |

---

## 8. Summary

| Step                | Single-GPU                    | Multi-GPU DDP                            |
|---------------------|-------------------------------|------------------------------------------|
| Recipe YAML         | unchanged                     | adds `hardware` block                    |
| Launcher            | `d4p-train recipe.yaml`       | `sbatch launch_ddp.sbatch recipe.yaml`   |
| Effective batch     | `batch_size`                  | `batch_size × num_nodes × gpus_per_node` |
| Checkpoints         | `*_best.pt`                   | identical (`unwrap` happens at save)     |
| Inference           | `d4p-downscale recipe.yaml`   | identical                                |

You now have a fully reproducible multi-GPU, multi-node training pipeline for DeepESD on the CORDEX-BENCH Alps domain, driven by the same YAML-first workflow as single-GPU `deep4production`.
