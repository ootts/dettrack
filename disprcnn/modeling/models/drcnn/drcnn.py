import numpy as np
import pytorch_ssim
import torch
from disprcnn.structures.segmentation_mask import SegmentationMask

from disprcnn.structures.bounding_box import BoxList

from disprcnn.modeling.models.yolact.layers.output_utils import postprocess

from disprcnn.modeling.models.yolact.yolact import Yolact
from torch import nn


class DRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.drcnn
        self.dbg = cfg.dbg is True

        if self.cfg.yolact_on:
            self.yolact = Yolact(cfg)
            ckpt = torch.load(self.cfg.pretrained_yolact, 'cpu')
            ckpt = {k.lstrip("model."): v for k, v in ckpt['model'].items()}
            self.yolact.load_state_dict(ckpt)
        if self.cfg.idispnet_on:
            raise NotImplementedError()
            print()
        if self.cfg.detector_3d_on:
            raise NotImplementedError()
            print()

    def forward(self, dps):
        assert self.yolact.training is False
        with torch.no_grad():
            preds_left = self.yolact({'image': dps['images']['left']})
            preds_right = self.yolact({'image': dps['images']['right']})
        h, w, _ = dps['original_images']['left'][0].shape
        left_boxes = self.decode_yolact_preds(preds_left, h, w)
        right_boxes = self.decode_yolact_preds(preds_right, h, w, add_mask=False)
        left_boxes, right_boxes = self.match_lp_rp(left_boxes, right_boxes,
                                                   dps['original_images']['left'][0],
                                                   dps['original_images']['right'][0])
        left_boxes.add_field('imgid', dps['imgid'][0].item())
        right_boxes.add_field('imgid', dps['imgid'][0].item())
        outputs = {'left': left_boxes, 'right': right_boxes}
        if self.dbg:
            left_boxes.plot(dps['original_images']['left'][0], show=True)
            right_boxes.plot(dps['original_images']['right'][0], show=True)
        loss_dict = {}
        return outputs, loss_dict

    def decode_yolact_preds(self, preds, h, w, add_mask=True):
        if preds[0]['detection'] is not None:
            preds = postprocess(preds, w, h, to_long=False)
            labels, scores, box2d, masks = preds
        else:
            box2d = torch.empty([0, 4], dtype=torch.float, device='cuda')
            labels = torch.empty([0], dtype=torch.long, device='cuda')
            scores = torch.empty([0], dtype=torch.float, device='cuda')
            masks = torch.empty([0, h, w], dtype=torch.long, device='cuda')
        boxlist = BoxList(box2d, (w, h))
        boxlist.add_field("labels", labels + 1)
        boxlist.add_field("scores", scores)
        if add_mask:
            keep = masks.sum(1).sum(1) > 20
            boxlist = boxlist[keep]
            masks = masks[keep]
            masks = SegmentationMask(masks, (w, h), mode='mask').convert("poly")
            boxlist.add_map("masks", masks)
        return boxlist

    def match_lp_rp(self, lp, rp, img2, img3):
        # lp = maskrcnn_to_disprcnn(lp)
        # rp = maskrcnn_to_disprcnn(rp)
        W, H = lp.size
        lboxes = lp.bbox.round().long().tolist()
        rboxes = rp.bbox.round().long().tolist()
        ssims = torch.zeros((len(lboxes), len(rboxes)))
        for i in range(len(lboxes)):
            x1, y1, x2, y2 = lboxes[i]
            for j in range(len(rboxes)):
                x1p, y1p, x2p, y2p = rboxes[j]
                # adaptive thresh
                hmean = (y2 - y1 + y2p - y1p) / 2
                center_disp_expectation = hmean * self.cfg.ssim_coef + self.cfg.ssim_intercept
                cd = (x1 + x2 - x1p - x2p) / 2
                if abs(cd - center_disp_expectation) < 3 * self.cfg.ssim_std:
                    w = max(x2 - x1, x2p - x1p)
                    h = max(y2 - y1, y2p - y1p)
                    w = min(min(w, W - x1, ), W - x1p)
                    h = min(min(h, H - y1, ), H - y1p)
                    lroi = img2[y1:y1 + h, x1:x1 + w, :].permute(2, 0, 1)[None] / 255.0
                    rroi = img3[y1p:y1p + h, x1p:x1p + w, :].permute(2, 0, 1)[None] / 255.0
                    s = pytorch_ssim.ssim(lroi, rroi)
                else:
                    s = -10
                ssims[i, j] = s
        if len(lboxes) <= len(rboxes):
            num = ssims.shape[0]
        else:
            num = ssims.shape[1]
        lidx, ridx = [], []
        for _ in range(num):
            tmp = torch.argmax(ssims).item()
            row, col = tmp // ssims.shape[1], tmp % ssims.shape[1]
            if ssims[row, col] > 0:
                lidx.append(row)
                ridx.append(col)
            ssims[row] = ssims[row].clamp(max=0)
            ssims[:, col] = ssims[:, col].clamp(max=0)
        lp = lp[lidx]
        rp = rp[ridx]
        return lp, rp
