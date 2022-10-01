import numpy as np
import torch
import pycocotools.mask as mask_utils

from disprcnn.structures.segmentation_mask import SegmentationMask

bitmasks = torch.load("tmp/masks.pth")
_, height, width = bitmasks.shape
tmp = np.ascontiguousarray(bitmasks.byte().cpu().numpy().transpose(1, 2, 0))
rlemasks = mask_utils.encode(np.asfortranarray(tmp))

decoded = mask_utils.decode(rlemasks)
m = SegmentationMask(rlemasks, (width, height), mode="mask")
ms = m.get_mask_tensor()
# print()
refs = []
for bm in bitmasks:
    refs.append(np.asfortranarray(bm.byte().cpu().numpy()))
refs = np.stack(refs, axis=-1)
print()
