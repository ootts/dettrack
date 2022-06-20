import numpy as np
import os.path as osp
from warnings import warn

import zarr
from .label_2 import load_label_2
from .image_info import load_image_info


def load_kins_mask_2(kittiroot, split, imgid):
    assert split == 'training'
    path = osp.join(kittiroot, 'object', split, 'kins_mask_2/%06d.zarr' % imgid)
    if not osp.exists(path):
        warn(path + ' not exists. return zeros')
        labels = load_label_2(kittiroot, split, imgid)
        H, W, _ = load_image_info(kittiroot, split, imgid)
        mask = np.zeros((len(labels), H, W)).astype(np.uint8)
    else:
        mask = zarr.load(path)
    return mask
