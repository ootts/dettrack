import imageio
import numpy as np
import pdb

import cv2
import torch
import os
import os.path as osp

import tqdm
from PIL import Image
from tqdm import trange

from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation, postprocess

from disprcnn.registry import VISUALIZERS
from disprcnn.utils.comm import get_rank
from disprcnn.utils.plt_utils import COLORS


@VISUALIZERS.register('yolact_tracking')
class YolactTrackingVisualizer:
    def __init__(self, cfg):
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact_tracking

    def __call__(self, *args, **kwargs):
        vis_dir = osp.join(self.total_cfg.output_dir, 'visualization', self.total_cfg.datasets.test)
        os.makedirs(vis_dir, exist_ok=True)
        os.system(f'rm {vis_dir}/*png')
        outputs, trainer = args
        ds = trainer.valid_dl.dataset
        last_seqid = -1
        displays = []
        for i in trange(len(outputs)):
            dps = ds[i]
            imgid = dps['imgid']
            seqid = dps['seq']
            img = dps['image']
            height = dps['height']
            width = dps['width']
            if seqid != last_seqid:
                if len(displays) != 0:
                    imageio.mimsave(osp.join(vis_dir, f"{last_seqid:04d}.mp4"), displays)
                displays = []
                last_seqid = seqid
            img_numpy = self.prep_display(outputs[i], img, height, width, seqid, imgid)
            displays.append(img_numpy)
        if len(displays) != 0:
            imageio.mimsave(osp.join(vis_dir, f"{last_seqid:04d}.mp4"), displays)

    def prep_display(self, pred, img, h, w, seqid, imgid):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        img_numpy = undo_image_transformation(img, w, h,
                                              [103.94, 116.78, 123.68],
                                              [57.38, 57.12, 58.40])
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.0
        font_thickness = 2

        cv2.putText(img_numpy, f"{seqid}:{imgid}", (0, 0), font_face, font_scale, (255, 255, 255), font_thickness,
                    cv2.LINE_AA)
        from matplotlib import colors as mcolors
        colors = list(mcolors.BASE_COLORS.values())
        for i, box in enumerate(pred.convert('xyxy').bbox.long().tolist()):
            x1, y1, x2, y2 = box
            trackid = pred.get_field("trackids")[i].item()
            color = colors[trackid % len(colors)]
            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 3)

            text_pt = (x1, y1 - 3)
            cv2.putText(img_numpy, str(trackid), text_pt, font_face, font_scale, color, font_thickness,
                        cv2.LINE_AA)
        img_numpy = (np.clip(img_numpy, 0, 1) * 255).astype(np.uint8)
        return img_numpy
