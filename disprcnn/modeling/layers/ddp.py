import torch
import torch.nn as nn
from .norm.syncbatchnorm import SyncBatchNorm

class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    def _passing_sync_batchnorm_handle(self, module_copies):
        for dev_idx, module in enumerate(module_copies):
            for layer in module.modules():
                if isinstance(layer, SyncBatchNorm):
                    assert self.is_cuda, "SyncBatchNorm layers only work with CUDA modules"
                    layer._specify_ddp_gpu_num(
                        len(self.device_ids) if self.device_ids else 1)
