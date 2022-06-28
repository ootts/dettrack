import numpy as np
import pytorch_ssim
import torch

from disprcnn.modeling.models.psmnet.inference import DisparityMapProcessor
from disprcnn.structures.disparity import DisparityMap

from disprcnn.modeling.layers import ROIAlign
from torchvision.transforms import transforms

from disprcnn.modeling.models.psmnet.stackhourglass import PSMNet
from disprcnn.structures.segmentation_mask import SegmentationMask

from disprcnn.structures.bounding_box import BoxList

from disprcnn.modeling.models.yolact.layers.output_utils import postprocess

from disprcnn.modeling.models.yolact.yolact import Yolact
from torch import nn

from disprcnn.utils.stereo_utils import expand_box_to_integer
from disprcnn.utils.vis3d_ext import Vis3D


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
            self.idispnet = PSMNet(cfg)
            self.roi_align = ROIAlign((self.idispnet.input_size, self.idispnet.input_size), 1.0, 0)
        if self.cfg.detector_3d_on:
            raise NotImplementedError()
            print()

    def forward(self, dps):
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="drcnn_forward",
            # auto_increase=,
            enable=self.dbg,
        )
        vis3d.set_scene_id(dps['global_step'])
        ##############  ↓ Step 1: 2D  ↓  ##############
        assert self.yolact.training is False
        with torch.no_grad():
            preds_left = self.yolact({'image': dps['images']['left']})
            preds_right = self.yolact({'image': dps['images']['right']})
        h, w, _ = dps['original_images']['left'][0].shape
        left_result = self.decode_yolact_preds(preds_left, h, w)
        right_result = self.decode_yolact_preds(preds_right, h, w, add_mask=False)
        left_result, right_result = self.match_lp_rp(left_result, right_result,
                                                     dps['original_images']['left'][0],
                                                     dps['original_images']['right'][0])
        left_result.add_field('imgid', dps['imgid'][0].item())
        right_result.add_field('imgid', dps['imgid'][0].item())
        if self.dbg:
            left_result.plot(dps['original_images']['left'][0], show=True)
            right_result.plot(dps['original_images']['right'][0], show=True)
        ##############  ↓ Step 2: idispnet  ↓  ##############
        left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps = self.prepare_idispnet_input(dps,
                                                                                                    left_result,
                                                                                                    right_result)
        if len(left_roi_images) > 0:
            disp_output = self.idispnet({'left': left_roi_images, 'right': right_roi_images})
        else:
            disp_output = torch.zeros((0, self.idispnet.input_size, self.idispnet.input_size)).cuda()
        left_result.add_field('disparity', disp_output)
        self.vis_roi_disp(dps, left_result, right_result, vis3d)
        ##############  ↓ Step 3: 3D detector  ↓  ##############
        # todo
        ##############  ↓ Step 4: return  ↓  ##############
        outputs = {'left': left_result, 'right': right_result}

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

    def prepare_idispnet_input(self, dps, left_result, right_result):
        assert dps['original_images']['left'].shape[0] == 1
        img_left = dps['original_images']['left'][0]
        img_right = dps['original_images']['right'][0]
        rois_left = []
        rois_right = []
        fxus, x1s, x1ps, x2s, x2ps = [], [], [], [], []
        # for i in range(bsz):
        left_target = dps['targets']['left'][0]
        calib = left_target.get_field('calib')
        fxus.extend([calib.stereo_fuxbaseline for _ in range(len(left_result))])
        mask_preds_per_img = left_result.get_map('masks').get_mask_tensor(squeeze=False)
        for j, (leftbox, rightbox, mask_pred) in enumerate(zip(left_result.bbox.tolist(),
                                                               right_result.bbox.tolist(),
                                                               mask_preds_per_img)):
            # 1 align left box and right box
            x1, y1, x2, y2 = expand_box_to_integer(leftbox)
            x1p, _, x2p, _ = expand_box_to_integer(rightbox)
            x1 = max(0, x1)
            x1p = max(0, x1p)
            y1 = max(0, y1)
            y2 = min(y2, left_result.height - 1)
            x2 = min(x2, left_result.width - 1)
            x2p = min(x2p, left_result.width - 1)
            max_width = max(x2 - x1, x2p - x1p)
            allow_extend_width = min(left_result.width - x1, left_result.width - x1p)
            max_width = min(max_width, allow_extend_width)
            rois_left.append([0, x1, y1, x1 + max_width, y2])
            rois_right.append([0, x1p, y1, x1p + max_width, y2])
            x1s.append(x1)
            x1ps.append(x1p)
            x2s.append(x1 + max_width)
            x2ps.append(x1p + max_width)
        left_roi_images = self.crop_and_transform_roi_img(img_left.permute(2, 0, 1)[None], rois_left)
        right_roi_images = self.crop_and_transform_roi_img(img_right.permute(2, 0, 1)[None], rois_right)
        if len(left_roi_images) != 0:
            x1s = torch.tensor(x1s).cuda()
            x1ps = torch.tensor(x1ps).cuda()
            x2s = torch.tensor(x2s).cuda()
            x2ps = torch.tensor(x2ps).cuda()
            fxus = torch.tensor(fxus).cuda()
        else:
            left_roi_images = torch.empty((0, 3, self.idispnet.input_size, self.idispnet.input_size)).cuda()
            right_roi_images = torch.empty((0, 3, self.idispnet.input_size, self.idispnet.input_size)).cuda()
        return left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps

    def crop_and_transform_roi_img(self, im, rois_for_image_crop):
        rois_for_image_crop = torch.as_tensor(rois_for_image_crop, dtype=torch.float32, device=im.device)
        im = self.roi_align(im.float() / 255.0, rois_for_image_crop)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=im.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=im.device)
        im.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return im

    def vis_roi_disp(self, dps, left_result, right_result, vis3d):
        if vis3d.enable:
            dmp = DisparityMapProcessor()
            disparity_map = dmp(left_result, right_result)
            target = dps['targets']['left'][0]
            calib = target.get_field('calib').calib
            pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data)
            vis3d.add_point_cloud(pts_rect)
            vis3d.add_image(dps['original_images']['left'][0].cpu().numpy())
            vis3d.add_box3dlist(target.get_field('box3d'))
            print()
