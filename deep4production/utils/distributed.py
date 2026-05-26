"""
SLURM-aware DistributedDataParallel (DDP) helpers for deep4production.

Reads launcher env vars (SLURM first, torchrun fallback) and initializes
``torch.distributed``. Provides small helpers for rank queries, all-reduce
of loss tensors, and unwrapping ``DistributedDataParallel`` modules.

Typical recipe ``hardware`` block::

    hardware:
      num_nodes: 4
      gpus_per_node: 4

Launch (SLURM)::

    srun --nodes=4 --ntasks-per-node=4 --gres=gpu:4 d4p-train recipe.yaml

When ``num_nodes * gpus_per_node <= 1`` initialization is skipped and the
trainer runs in plain single-process mode (CPU or single GPU).
"""

import os
import socket
import subprocess
import logging
import torch
import torch.distributed as dist

from deep4production.utils.log import get_logger

log = get_logger("distributed")


# ----------------------------------------------------------------------------
def _resolve_master_addr_from_slurm() -> str:
    """Pick the first host in ``SLURM_NODELIST`` as the rendezvous address."""
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return socket.gethostname()
    try:
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        ).splitlines()
        return hostnames[0] if hostnames else socket.gethostname()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return socket.gethostname()


# ----------------------------------------------------------------------------
def init_distributed(hardware: dict | None = None) -> dict:
    """
    Initialize ``torch.distributed`` based on the recipe ``hardware`` block.

    Parameters
    ----------
    hardware : dict, optional
        ``{"num_nodes": int, "gpus_per_node": int, "master_port": int}``.
        Missing keys default to 1 (no DDP) and 29500 respectively.

    Returns
    -------
    dict
        ``{rank, local_rank, world_size, device, distributed, num_nodes,
        gpus_per_node}`` — usable by the trainer regardless of whether DDP
        was actually started.
    """
    hardware = hardware or {}
    num_nodes = int(hardware.get("num_nodes", 1))
    gpus_per_node = int(hardware.get("gpus_per_node", 1))
    world_expected = max(1, num_nodes * gpus_per_node)

    # --- Single-process fast path ---
    if world_expected <= 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "distributed": False,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
        }

    # --- Detect launcher env (SLURM first, then torchrun) ---
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", world_expected))
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = _resolve_master_addr_from_slurm()
        os.environ.setdefault("MASTER_PORT", str(hardware.get("master_port", 29500)))
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError(
            f"hardware requested {world_expected} workers "
            f"(num_nodes={num_nodes} × gpus_per_node={gpus_per_node}) but no "
            "launcher env vars are set. Launch via `srun` (SLURM) or "
            "`torchrun` so SLURM_PROCID or RANK is defined."
        )

    if world_size != world_expected:
        log.warning(
            "Launcher world_size=%d differs from recipe hardware "
            "(num_nodes=%d × gpus_per_node=%d = %d). Using launcher value.",
            world_size,
            num_nodes,
            gpus_per_node,
            world_expected,
        )

    # --- Bind local GPU and start the process group ---
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA but torch.cuda.is_available() is False.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"local_rank={local_rank} but only "
            f"{torch.cuda.device_count()} CUDA devices visible on this node."
        )
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if rank == 0:
        log.info(
            "DDP initialized: backend=%s world_size=%d (num_nodes=%d × "
            "gpus_per_node=%d) MASTER_ADDR=%s MASTER_PORT=%s",
            backend,
            world_size,
            num_nodes,
            gpus_per_node,
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
        )

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": f"cuda:{local_rank}",
        "distributed": True,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
    }


# ----------------------------------------------------------------------------
def cleanup_distributed() -> None:
    """Destroy the process group if one was created. Idempotent."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ----------------------------------------------------------------------------
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


# ----------------------------------------------------------------------------
def barrier() -> None:
    """No-op outside DDP; blocks until all ranks arrive otherwise."""
    if is_distributed():
        dist.barrier()


# ----------------------------------------------------------------------------
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar/loss tensor across ranks and divide by world_size.

    Operates in place; also returns the tensor. No-op outside DDP.
    """
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# ----------------------------------------------------------------------------
def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from a DDP wrapper (or the model itself)."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ----------------------------------------------------------------------------
def silence_non_main_ranks() -> None:
    """Set the ``d4p`` logger to ERROR on non-zero ranks so SLURM logs stay
    readable. Call once after ``init_distributed``."""
    if is_distributed() and not is_main_process():
        logging.getLogger("d4p").setLevel(logging.ERROR)
