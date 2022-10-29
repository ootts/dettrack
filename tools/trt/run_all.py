import imageio
import torch
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import os.path as osp
import glob
import os

from PIL import Image
from torchvision.ops import RoIAlign

from disprcnn.data import make_data_loader
from disprcnn.engine.defaults import setup
from disprcnn.engine.defaults import default_argument_parser
from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.trt.idispnet_inference import IDispnetInference
from disprcnn.trt.yolact_inference import YolactInference
from disprcnn.trt.yolact_tracking_head_inference import YolactTrackingHeadInference
from disprcnn.utils.pytorch_ssim import ssim
from disprcnn.utils.stereo_utils import expand_box_to_integer_torch
from disprcnn.utils.timer import EvalTime


class TotalInference:
    def __init__(self):
        parser = default_argument_parser()
        args = parser.parse_args()
        args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
        cfg = setup(args)
        self.cfg = cfg

        valid_ds = make_data_loader(cfg).dataset

        data0 = valid_ds[0]
        calib = data0['targets']['left'].extra_fields['calib']
        self.calib = calib

        yolact_cfg = cfg.model.yolact
        detector = Detect(yolact_cfg, yolact_cfg.num_classes, bkg_label=0, top_k=yolact_cfg.nms_top_k,
                          conf_thresh=yolact_cfg.nms_conf_thresh, nms_thresh=yolact_cfg.nms_thresh)

        self.yolact_inf = YolactInference(osp.join(cfg.trt.convert_to_trt.output_path, "yolact.engine"),
                                          detector)

        self.yolact_tracking_head_inf = YolactTrackingHeadInference(
            osp.join(cfg.trt.convert_to_trt.output_path, "yolact_tracking_head.engine"))

        self.idispnet_inf = IDispnetInference(
            osp.join(cfg.trt.convert_to_trt.output_path, "idispnet.engine"))

        self.roi_align = RoIAlign((112, 112), 1.0, 0)

    def infer(self, input_file1, input_file2):
        evaltime = EvalTime()
        left = cv2.cvtColor(cv2.imread(input_file1), cv2.COLOR_BGR2GRAY)
        left = np.repeat(left[:, :, None], 3, axis=2)
        right = cv2.cvtColor(cv2.imread(input_file2), cv2.COLOR_BGR2GRAY)
        right = np.repeat(right[:, :, None], 3, axis=2)
        height, width, _ = left.shape
        original_left_img = torch.from_numpy(left).cuda()[None]
        original_right_img = torch.from_numpy(right).cuda()[None]
        evaltime('begin')
        left_preds, right_preds, left_feat, right_feat = self.yolact_inf.detect(input_file1, input_file2)
        evaltime('2d detection')

        left_pred, right_pred = self.yolact_tracking_head_inf.track(
            left_preds, left_feat, right_preds, right_feat, width, height)
        evaltime('track')
        left_result, right_result = self.match_lp_rp(left_pred, right_pred,
                                                     original_left_img[0],
                                                     original_right_img[0])
        evaltime('match lr')
        idispnet_prep = self.prepare_idispnet_input(original_left_img, original_right_img,
                                                    left_result, right_result)
        left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps = idispnet_prep
        evaltime('idispnet prep')
        if len(left_roi_images) > 0:
            self.idispnet_inf.infer(left_roi_images, right_roi_images)
            disp_output = self.idispnet_inf.cuda_outputs['output']
        else:
            disp_output = torch.zeros((0, 112, 112)).cuda()
        left_result.add_field('disparity', disp_output)
        evaltime('idispnet forward')

    def match_lp_rp(self, lp, rp, img2, img3):
        W, H = lp.size
        lboxes = lp.bbox.round().long().tolist()
        # llabels = lp.get_field('labels').long().tolist() # onnx fails.
        rboxes = rp.bbox.round().long().tolist()
        ssims = torch.zeros((len(lboxes), len(rboxes)))
        for i in range(len(lboxes)):
            x1, y1, x2, y2 = lboxes[i]
            ssim_coef = self.cfg.model.drcnn.ssim_coefs[0]
            ssim_intercept = self.cfg.model.drcnn.ssim_intercepts[0]
            ssim_std = self.cfg.model.drcnn.ssim_stds[0]
            for j in range(len(rboxes)):
                x1p, y1p, x2p, y2p = rboxes[j]
                # adaptive thresh
                hmean = (y2 - y1 + y2p - y1p) / 2
                center_disp_expectation = hmean * ssim_coef + ssim_intercept
                cd = (x1 + x2 - x1p - x2p) / 2
                if abs(cd - center_disp_expectation) < 3 * ssim_std:
                    w = max(x2 - x1, x2p - x1p)
                    h = max(y2 - y1, y2p - y1p)
                    w = min(min(w, W - x1, ), W - x1p)
                    h = min(min(h, H - y1, ), H - y1p)
                    lroi = img2[y1:y1 + h, x1:x1 + w, :].permute(2, 0, 1)[None] / 255.0
                    rroi = img3[y1p:y1p + h, x1p:x1p + w, :].permute(2, 0, 1)[None] / 255.0
                    s = ssim(lroi, rroi)
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

    def prepare_idispnet_input(self, original_left_image, original_right_image,
                               left_result, right_result):
        # assert dps['original_images']['left'].shape[0] == 1
        img_left = original_left_image[0]
        img_right = original_right_image[0]
        rois_left = []
        rois_right = []
        x1s, x1ps, x2s, x2ps = [], [], [], []
        calib = self.calib
        fxus = torch.tensor([calib.stereo_fuxbaseline for _ in range(len(left_result))]).float().cuda()
        mask_preds_per_img = left_result.get_map('masks').get_mask_tensor(squeeze=False)
        for j, (leftbox, rightbox, mask_pred) in enumerate(zip(left_result.bbox,  # .tolist() causes onnx error
                                                               right_result.bbox,
                                                               mask_preds_per_img)):
            # 1 align left box and right box
            leftbox = expand_box_to_integer_torch(leftbox)
            rightbox = expand_box_to_integer_torch(rightbox)
            leftbox[0] = torch.clamp(leftbox[0], min=0)
            rightbox[0] = torch.clamp(rightbox[0], min=0)
            leftbox[1] = torch.clamp(leftbox[1], min=0)
            leftbox[3] = torch.clamp(leftbox[3], max=left_result.height - 1)
            leftbox[2] = torch.clamp(leftbox[2], max=left_result.width - 1)
            rightbox[2] = torch.clamp(rightbox[2], max=left_result.width - 1)
            max_width = max(leftbox[2] - leftbox[0], rightbox[2] - rightbox[0])
            allow_extend_width = min(left_result.width - leftbox[0], left_result.width - rightbox[0])
            max_width = min(max_width, allow_extend_width)
            leftbox[2] = leftbox[0] + max_width
            rightbox[2] = rightbox[0] + max_width
            rightbox[1] = leftbox[1]
            rightbox[3] = leftbox[3]
            rois_left.append(torch.cat([torch.tensor([0]).long().cuda(), leftbox]))
            rois_right.append(torch.cat([torch.tensor([0]).long().cuda(), rightbox]))
            x1s.append(leftbox[0])
            x1ps.append(rightbox[0])
            x2s.append(leftbox[2])
            x2ps.append(rightbox[2])

        left_roi_images = self.crop_and_transform_roi_img(img_left.permute(2, 0, 1)[None], torch.stack(rois_left))
        right_roi_images = self.crop_and_transform_roi_img(img_right.permute(2, 0, 1)[None], torch.stack(rois_right))
        if len(left_roi_images) != 0:
            x1s = torch.stack(x1s)
            x1ps = torch.stack(x1ps)
            x2s = torch.stack(x2s)
            x2ps = torch.stack(x2ps)
        else:
            left_roi_images = torch.empty((0, 3, 112, 112)).cuda()
            right_roi_images = torch.empty((0, 3, 112, 112)).cuda()
        return left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps

    def crop_and_transform_roi_img(self, im, rois_for_image_crop):
        im = self.roi_align(im.float() / 255.0, rois_for_image_crop.float())
        mean = torch.tensor([0.485, 0.456, 0.406]).float().cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).float().cuda()
        im.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return im

    def destroy(self):
        self.yolact_inf.destory()


def main():
    data2_dir = "data/kitti/tracking/training/image_02/0001/"
    data3_dir = "data/kitti/tracking/training/image_03/0001/"
    input_file1s = sorted(glob.glob(osp.join(data2_dir, "*.png")))
    input_file2s = sorted(glob.glob(osp.join(data3_dir, "*.png")))
    assert len(input_file1s) == len(input_file2s)

    inference = TotalInference()

    for i in range(len(input_file1s)):
        inference.infer(input_file1s[i], input_file2s[i])

    inference.destroy()


if __name__ == '__main__':
    main()
