import matplotlib.pyplot as plt
import numpy as np
import imageio
import pdb

import cv2
import torch
import os
import os.path as osp

import tqdm
from PIL import Image
from disprcnn.structures.bounding_box import BoxList
from tqdm import trange

from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation

from disprcnn.registry import VISUALIZERS
from disprcnn.utils.comm import get_rank
from disprcnn.utils.plt_utils import COLORS


@VISUALIZERS.register('drcnn')
class DrcnnVisualizer:
    def __init__(self, cfg):
        self.total_cfg = cfg
        self.cfg = cfg.model.drcnn

    def __call__(self, *args, **kwargs):
        vis_dir = osp.join(self.total_cfg.output_dir, 'visualization', self.total_cfg.datasets.test)
        os.makedirs(vis_dir, exist_ok=True)
        os.system(f'rm {vis_dir}/*')
        outputs, trainer = args
        ds = trainer.valid_dl.dataset
        for i in trange(min(len(outputs), self.cfg.nvis)):
            dps = ds[i]
            imgid = dps['imgid'] if 'imgid' in dps else i
            left_img = dps['original_images']['left']
            right_img = dps['original_images']['right']
            left_result: BoxList = outputs[i]['left']
            right_result = outputs[i]['right']
            left_result.plot(left_img, show=False)
            plt.savefig(osp.join(vis_dir, f'{imgid:06d}_left.png'))
            right_result.plot(right_img, show=False)
            plt.savefig(osp.join(vis_dir, f'{imgid:06d}_right.png'))
