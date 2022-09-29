import numpy as np
import imageio
import pdb

import cv2
import torch
import os
import os.path as osp

import tqdm
from PIL import Image
from tqdm import trange

from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation

from disprcnn.registry import VISUALIZERS
from disprcnn.utils.comm import get_rank
from disprcnn.utils.plt_utils import COLORS


@VISUALIZERS.register('yolact')
class YolactVisualizer:
    def __init__(self, cfg):
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact

    def __call__(self, *args, **kwargs):
        vis_dir = osp.join(self.total_cfg.output_dir, 'visualization', self.total_cfg.datasets.test)
        os.makedirs(vis_dir, exist_ok=True)
        os.system(f'rm {vis_dir}/*')
        outputs, trainer = args
        ds = trainer.valid_dl.dataset
        imgs = []
        for i in trange(min(len(outputs), self.cfg.nvis)):
            # if 0 < self.cfg.nvis < i: break
            dps = ds[i]
            imgid = dps['imgid'] if 'imgid' in dps else i
            dets_out = outputs[i]
            img = dps['image']
            height = dps['height']
            width = dps['width']
            img_numpy = self.prep_display([dets_out], img, height, width)
            Image.fromarray(img_numpy).save(osp.join(vis_dir, f'{imgid:06d}.png'))
            imgs.append(img_numpy)
        maxh = max([i.shape[0] for i in imgs])
        maxw = max([i.shape[1] for i in imgs])
        ims = np.zeros([len(imgs), maxh, maxw, 3], dtype=np.uint8)
        for i in range(len(imgs)):
            im = imgs[i]
            ims[i, :im.shape[0], :im.shape[1], :] = im
        imageio.mimwrite(osp.join(vis_dir, "video.mp4"), ims, fps=1)

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h,
                                                  [103.94, 116.78, 123.68],
                                                  [57.38, 57.12, 58.40])
            img_gpu = torch.Tensor(img_numpy)
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        if dets_out[0]['detection'] is None:
            return (img_gpu * 255).byte().cpu().numpy()
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=self.cfg.score_threshold)

        idx = t[1].argsort(0, descending=True)[:self.cfg.top_k]

        if self.cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.cfg.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.cfg.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            # global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            # if on_gpu is not None and color_idx in color_cache[on_gpu]:
            #     return color_cache[on_gpu][color_idx]
            # else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            color = torch.Tensor(color).float() / 255.
            if on_gpu is not None:
                color = color.to(on_gpu)
                # color_cache[on_gpu][color_idx] = color
            return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        # if self.cfg.display_masks and self.cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
            dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return img_numpy

        if True:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j).tolist()
                score = scores[j]

                # if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                # if args.display_text:
                _class = self.cfg.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score)

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

        return img_numpy
