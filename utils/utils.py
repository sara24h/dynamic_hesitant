# dynamic_hesitant/utils/utils.py

import os
import random
import numpy as np
import torch
import torch.distributed as dist

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds set to: {seed}")

def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size
 
def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
