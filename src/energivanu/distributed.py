import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist

_BACKEND = "nccl" if torch.cuda.is_available() else "gloo"


def setup(backend: Optional[str] = None, timeout_minutes: int = 30):
    """Initialize the distributed process group.

    Reads environment variables set by ``torchrun``:
      - ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``

    Falls back to single-process no-op if ``RANK`` is not set.
    """
    if "RANK" not in os.environ:
        return False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        return False
    dist.init_process_group(
        backend=backend or _BACKEND,
        init_method="env://",
        timeout=torch.distributed.default_pg_options.timeout if hasattr(torch.distributed, 'default_pg_options') else None,
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    return True


def cleanup():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return dist.is_initialized() and get_world_size() > 1


@contextmanager
def main_process_first():
    """Context manager that runs the enclosed block only on the main process
    (rank 0), with a barrier so all processes wait before proceeding."""
    if is_distributed():
        dist.barrier()
    try:
        yield
    finally:
        if is_distributed():
            dist.barrier()


def save_checkpoint(state: dict, path: str, only_on_main: bool = True):
    """Save checkpoint only on rank 0 (or all ranks if ``only_on_main=False``)."""
    if only_on_main and not is_main_process():
        return
    torch.save(state, path)


def broadcast_state_dict(model: torch.nn.Module, src: int = 0):
    """Broadcast model state from rank ``src`` to all other ranks.
    Useful when loading a checkpoint only on rank 0."""
    if is_distributed():
        for param in model.parameters():
            dist.broadcast(param.data, src=src)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the mean of ``tensor`` across all processes."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def get_device() -> torch.device:
    """Get the correct device for the current process."""
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device("cuda")
    return torch.device("cpu")
