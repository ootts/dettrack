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
from disprcnn.modeling.models.pointpillars.submodules import PointPillarsScatter
from disprcnn.modeling.models.psmnet.inference import DisparityMapProcessor
from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.structures.calib import Calib
from disprcnn.trt.idispnet_inference import IDispnetInference
from disprcnn.trt.pointpillars_part1_inference import PointPillarsPart1Inference
from disprcnn.trt.pointpillars_part2_inference import PointPillarsPart2Inference
from disprcnn.trt.yolact_inference import YolactInference
from disprcnn.trt.yolact_tracking_head_inference import YolactTrackingHeadInference
from disprcnn.utils.pn_utils import to_tensor, to_array
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
from disprcnn.utils.ppp_utils.box_np_ops import sparse_sum_for_anchors_mask, fused_get_anchors_area, \
    rbbox2d_to_near_bbox
from disprcnn.utils.ppp_utils.target_assigner import build_target_assigner
from disprcnn.utils.ppp_utils.voxel_generator import build_voxel_generator
from disprcnn.utils.pytorch_ssim import ssim
from disprcnn.utils.stereo_utils import expand_box_to_integer_torch
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.utils_3d import matrix_3x4_to_4x4


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

        # self.pointpillars_inf = PointPillarsPart1Inference(
        #     osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars.engine"))
        #
        # self.pointpillars_part2_inf = PointPillarsPart2Inference(
        #     osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars_part2.engine"))
        #
        # self.voxel_generator = build_voxel_generator(self.cfg.voxel_generator)
        # box_coder = GroundBox3dCoderTorch()
        # self.target_assigner = build_target_assigner(self.cfg.model.pointpillars.target_assigner, box_coder)
        # feature_map_size = [1, 248, 216]
        # ret = self.target_assigner.generate_anchors(feature_map_size)  # [352, 400]
        # anchors = torch.from_numpy(ret["anchors"]).cuda()
        # anchors = anchors.reshape([-1, 7])
        # matched_thresholds = torch.from_numpy(ret["matched_thresholds"]).cuda()
        # unmatched_thresholds = torch.from_numpy(ret["unmatched_thresholds"]).cuda()
        # anchors_bv = rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        # self.anchor_cache = {
        #     "anchors": anchors,
        #     "anchors_bv": anchors_bv,
        #     "matched_thresholds": matched_thresholds,
        #     "unmatched_thresholds": unmatched_thresholds,
        # }
        # self.middle_feature_extractor = PointPillarsScatter(output_shape=[1, 1, 496, 432, 64],
        #                                                     num_input_features=64)

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
            disp_output = self.idispnet_inf.predict_idisp(left_roi_images, right_roi_images)
        else:
            disp_output = torch.zeros((0, 112, 112)).cuda()
        left_result.add_field('disparity', disp_output)
        evaltime('idispnet forward')
        # pp_input = self.prepare_pointpillars_input(left_result, right_result, width, height)
        # self.pointpillars_inf.infer(pp_input['voxels'], pp_input['num_points'], pp_input['coordinates'])
        # voxel_features = self.pointpillars_inf.cuda_outputs['output'].clone()
        # spatial_features = self.middle_feature_extractor(voxel_features, pp_input['coordinates'],
        #                                                  pp_input["anchors"].shape[0])
        # pp_output = self.pointpillars_part2_inf.predict(spatial_features, pp_input['anchors'],
        #                                                 pp_input['rect'], pp_input['Trv2c'], pp_input['P2'],
        #                                                 pp_input['anchors_mask'])
        print()

    #     anchors, rect, Trv2c, P2, anchors_mask

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

    def prepare_pointpillars_input(self, left_result, right_result, width, height):
        dmp = DisparityMapProcessor()
        voxel_generator = self.voxel_generator
        disparity_map = dmp(left_result, right_result)
        calib = self.calib
        calib = Calib(calib, (calib.width, calib.height), 'cuda')
        pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data)
        keep = (pts_rect[:, 0] > -20) & (pts_rect[:, 0] < 20) & \
               (pts_rect[:, 1] > -3) & (pts_rect[:, 1] < 3) \
               & (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
        # keep = (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
        pts_rect = pts_rect[keep]
        points = calib.rect_to_lidar(pts_rect)
        rect = torch.eye(4).cuda().float()
        rect[:3, :3] = to_tensor(calib.R0, torch.float, 'cuda')
        Trv2c = torch.eye(4).cuda().float()
        Trv2c[:3, :4] = to_tensor(calib.V2C, torch.float, 'cuda')
        # augmentation
        # if self.cfg.detector_3d.shuffle_points:
        #     perm = np.random.permutation(points.shape[0])
        #     points = points[perm]
        points = torch.cat([points, torch.full_like(points[:, 0:1], 0.5)], dim=1)
        voxel_size = voxel_generator.voxel_size
        pc_range = voxel_generator.point_cloud_range
        grid_size = voxel_generator.grid_size
        p = to_array(points)
        voxels, coordinates, num_points = voxel_generator.generate(p,
                                                                   self.cfg.model.drcnn.detector_3d.max_number_of_voxels)

        coordinates = torch.from_numpy(coordinates).long().cuda()
        voxels = torch.from_numpy(voxels).cuda()
        num_points = torch.from_numpy(num_points).long().cuda()
        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'rect': rect[None],
            'Trv2c': Trv2c[None],
            'P2': to_tensor(matrix_3x4_to_4x4(calib.P2), torch.float, 'cuda')[None],
        }
        anchor_cache = self.anchor_cache
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        # matched_thresholds = anchor_cache["matched_thresholds"]
        # unmatched_thresholds = anchor_cache["unmatched_thresholds"]
        example["anchors"] = anchors[None]
        # anchors_mask = None
        anchor_area_threshold = self.cfg.model.drcnn.detector_3d.anchor_area_threshold
        if anchor_area_threshold >= 0:
            coors = coordinates
            dense_voxel_map = sparse_sum_for_anchors_mask(to_array(coors), tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = fused_get_anchors_area(dense_voxel_map, to_array(anchors_bv),
                                                  voxel_size, pc_range, grid_size)
            anchors_mask = torch.from_numpy(anchors_area > anchor_area_threshold).cuda()
            example['anchors_mask'] = anchors_mask[None]
        example['coordinates'] = torch.cat([torch.full_like(coordinates[:, 0:1], 0), example['coordinates']], dim=1)
        # example['image_idx'] = [torch.tensor(left_result.extra_fields['imgid']).cuda()]
        example['calib'] = calib
        example['width'] = width
        example['height'] = height
        return example

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
