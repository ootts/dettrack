import os

import imageio
import tensorrt as trt
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pycuda.driver as cuda
# import pycuda.autoinit
import os.path as osp
import glob

import yaml
from PIL import Image
from torchvision.ops import RoIAlign

from disprcnn.data import make_data_loader
from disprcnn.engine.defaults import setup
from disprcnn.engine.defaults import default_argument_parser
from disprcnn.modeling.models.pointpillars.submodules import PointPillarsScatter
from disprcnn.modeling.models.psmnet.inference import DisparityMapProcessor
from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.boxlist_ops import boxlist_iou
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
from disprcnn.utils.pytorch_ssim import ssim, SSIM
from disprcnn.utils.stereo_utils import expand_box_to_integer_torch
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.utils_3d import matrix_3x4_to_4x4
from disprcnn.utils.vis3d_ext import Vis3D


def read_calib(raw_dir):
    from dl_ext.vision_ext.datasets.kitti.structures import Calibration
    with open(osp.join(raw_dir, "calib.txt")) as f:
        lines = f.readlines()
    calibs = {}
    for line in lines:
        k = line.split()[0]
        nums = list(map(float, line.split()[1:]))
        if len(nums) == 12:
            nums = torch.tensor(nums).reshape(3, 4).float()
        elif len(nums) == 9:
            nums = torch.tensor(nums).reshape(3, 3).float()
        else:
            raise RuntimeError()
        if k == 'R_rect':
            k = 'R0_rect'
        elif k == 'Tr_velo_cam':
            k = 'Tr_velo_to_cam'
        elif k == 'Tr_imu_velo':
            k = 'Tr_imu_to_velo'
        k = k.strip(":")
        calibs[k] = nums.numpy()
    img = Image.open(glob.glob(osp.join("data/real/left/*png"))[0])
    calib = Calibration(calibs, [img.width, img.height])
    return calib


def load_cam_attrs(cam_path):
    resize_factor = 2.0
    cam_attrs = yaml.full_load(open(cam_path))
    camera_matrix = np.array(
        [[cam_attrs['projection_parameters']['fx'] * resize_factor, 0,
          cam_attrs['projection_parameters']['cx'] * resize_factor],
         [0, cam_attrs['projection_parameters']['fy'] * resize_factor,
          cam_attrs['projection_parameters']['cy'] * resize_factor],
         [0, 0, 1]])
    dist_coeffs = np.array([cam_attrs['distortion_parameters']['k1'],
                            cam_attrs['distortion_parameters']['k2'],
                            cam_attrs['distortion_parameters']['p1'],
                            cam_attrs['distortion_parameters']['p2'],
                            0])
    return camera_matrix, dist_coeffs


def prepare_stereo_rectify(raw_dir):
    # raw_dir = "data/real/"
    left_path = sorted(glob.glob(osp.join(raw_dir, "left/*.png")))[0]

    left = imageio.imread(left_path)
    H, W = left.shape
    left_K, left_dist = load_cam_attrs(osp.join(raw_dir, "cam0_small.yaml"))
    right_K, right_dist = load_cam_attrs(osp.join(raw_dir, "cam1_small.yaml"))

    R2 = np.array([9.9999e-01, 1.3479e-04, 4.7469e-03, 1.2004e-01,
                   -1.2457e-04, 1, -2.1531e-03, -7.4797e-04,
                   -4.7472e-03, 2.1524e-03, 9.9999e-01, 5.4342e-04,
                   0, 0, 0, 1]).reshape(4, 4)
    R2 = np.linalg.inv(R2)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_K, left_dist, right_K, right_dist,
                                                                      (W, H), R2[:3, :3], R2[:3, 3],
                                                                      flags=cv2.CALIB_ZERO_DISPARITY)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, (W, H), cv2.CV_32FC1)

    return P1, P2, map1x, map1y, map2x, map2y


class TotalInference:
    def __init__(self, raw_dir):
        parser = default_argument_parser()
        args = parser.parse_args()
        args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_real_demo.yaml'
        cfg = setup(args)
        self.cfg = cfg
        self.dbg = cfg.dbg

        calib = read_calib(raw_dir)
        # self.calib = Calib(calib)
        self.calib = Calib(calib, (calib.width, calib.height), 'cuda')

        yolact_cfg = cfg.model.yolact
        detector = Detect(yolact_cfg, yolact_cfg.num_classes, bkg_label=0, top_k=yolact_cfg.nms_top_k,
                          conf_thresh=yolact_cfg.nms_conf_thresh, nms_thresh=yolact_cfg.nms_thresh)
        postfix = ".engine"
        if cfg.trt.convert_to_trt.fp16:
            postfix = "-fp16" + postfix

        self.yolact_inf = YolactInference(osp.join(cfg.trt.convert_to_trt.output_path, f"yolact{postfix}"),
                                          detector)

        self.yolact_tracking_head_inf = YolactTrackingHeadInference(
            osp.join(cfg.trt.convert_to_trt.output_path, f"yolact_tracking_head{postfix}"))

        self.idispnet_inf = IDispnetInference(
            osp.join(cfg.trt.convert_to_trt.output_path, f"idispnet{postfix}"))
        self.roi_align = RoIAlign((112, 112), 1.0, 0)

        self.pointpillars_inf = PointPillarsPart1Inference(
            osp.join(cfg.trt.convert_to_trt.output_path, f"pointpillars{postfix}"))

        self.pointpillars_part2_inf = PointPillarsPart2Inference(
            osp.join(cfg.trt.convert_to_trt.output_path, f"pointpillars_part2{postfix}"))

        self.voxel_generator = build_voxel_generator(self.cfg.voxel_generator)
        box_coder = GroundBox3dCoderTorch()
        self.target_assigner = build_target_assigner(self.cfg.model.pointpillars.target_assigner, box_coder)
        feature_map_size = [1, 248, 216]
        ret = self.target_assigner.generate_anchors(feature_map_size)  # [352, 400]
        anchors = torch.from_numpy(ret["anchors"]).cuda()
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = torch.from_numpy(ret["matched_thresholds"]).cuda()
        unmatched_thresholds = torch.from_numpy(ret["unmatched_thresholds"]).cuda()
        anchors_bv = rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        self.anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self.middle_feature_extractor = PointPillarsScatter(output_shape=[1, 1, 496, 432, 64],
                                                            num_input_features=64)
        self.evaltime = EvalTime()
        self.ssim = SSIM().cuda()
        self.stereo_rectify_params = prepare_stereo_rectify(raw_dir)  # P1, P2, map1x, map1y, map2x, map2y

    def stereo_rectify(self, input_file1, input_file2):
        left = imageio.imread(input_file1)
        P1, P2, map1x, map1y, map2x, map2y = self.stereo_rectify_params
        right = imageio.imread(input_file2)
        left_remap = cv2.remap(left, map1x, map1y,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
        right_remap = cv2.remap(right, map2x, map2y,
                                interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        return left_remap, right_remap

    def infer(self, input_file1, input_file2):
        left, right = self.stereo_rectify(input_file1, input_file2)
        # left = cv2.cvtColor(cv2.imread(input_file1), cv2.COLOR_BGR2GRAY)
        left = np.repeat(left[:, :, None], 3, axis=2)
        # right = cv2.cvtColor(cv2.imread(input_file2), cv2.COLOR_BGR2GRAY)
        right = np.repeat(right[:, :, None], 3, axis=2)
        height, width, _ = left.shape
        original_left_img = torch.from_numpy(left).cuda()[None]
        original_right_img = torch.from_numpy(right).cuda()[None]
        self.evaltime('begin')
        left_preds, right_preds, left_feat, right_feat = self.yolact_inf.detect(input_file1, input_file2)
        # evaltime('yolact:detect')

        left_result, right_result = self.yolact_tracking_head_inf.track(
            left_preds, left_feat, right_preds, right_feat, width, height)
        # evaltime('track')
        self.evaltime("")
        if len(left_result) > 0 or len(right_result) > 0:
            left_result, right_result = self.match_lp_rp(left_result, right_result,
                                                         original_left_img[0],
                                                         original_right_img[0])
            self.evaltime('match lr')
        idispnet_prep = self.prepare_idispnet_input(original_left_img, original_right_img,
                                                    left_result, right_result)
        left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps = idispnet_prep
        self.evaltime('idispnet:prep input')
        if len(left_roi_images) > 0:
            disp_output = self.idispnet_inf.predict_idisp(left_roi_images, right_roi_images)
        else:
            disp_output = torch.zeros((0, 112, 112)).cuda()
        left_result.add_field('disparity', disp_output)
        if len(left_result) > 0:
            self.evaltime("")
            pp_input = self.prepare_pointpillars_input(left_result, right_result, width, height)
            self.evaltime('pointpillars:prepare input')
            if pp_input['voxels'].shape[0] > 0:
                cuda_outputs = self.pointpillars_inf.infer(pp_input['voxels'], pp_input['num_points'],
                                                           pp_input['coordinates'])
                self.evaltime('pointpillars:infer')
                voxel_features = cuda_outputs['output']
                spatial_features = self.middle_feature_extractor(voxel_features, pp_input['coordinates'],
                                                                 pp_input["anchors"].shape[0])
                self.evaltime('pointpillars:middle_feature_extractor')
                pp_output = self.pointpillars_part2_inf.detect_3d_bbox(spatial_features, pp_input['anchors'],
                                                                       pp_input['rect'], pp_input['Trv2c'],
                                                                       pp_input['P2'],
                                                                       pp_input['anchors_mask'], width, height)
            else:
                pp_output = None
        self.evaltime("")
        if len(left_result) == 0 or pp_output is None:
            box3d = Box3DList(torch.empty([0, 7]), "xyzhwl_ry")
            left_result.add_field('box3d', box3d)
        elif len(pp_output['left']) == 0:
            box3d = Box3DList(torch.ones([0, 7], dtype=torch.float, device='cuda'), "xyzhwl_ry")
            left_result.add_field('box3d', box3d)
        else:
            iou = boxlist_iou(left_result, pp_output['left'])

            maxiou, maxiouidx = iou.max(1)
            keep = maxiou > 0.5
            box3d = pp_output['left'].get_field('box3d')[maxiouidx]
            left_result.add_field('box3d', box3d)
            # masks = left_result.PixelWise_map['masks'].convert('mask')
            # left_result.PixelWise_map['masks'] = masks
            left_result = left_result[keep]
            # left_result.PixelWise_map['masks'] = left_result.PixelWise_map['masks'].convert(
            #     self.cfg.mask_mode)
            right_result = right_result[keep]
        self.evaltime("pipeline: postprocess")
        if self.dbg:
            self.vis_final_result(left_result, right_result, self.calib, original_left_img, original_right_img)

    def match_lp_rp(self, lp, rp, img2, img3):
        # self.evaltime("")
        W, H = lp.size
        lboxes = lp.bbox.round().long()
        rboxes = rp.bbox.round().long()
        ssims = torch.zeros((len(lboxes), len(rboxes)))
        ssim_coef = self.cfg.model.drcnn.ssim_coefs[0]
        ssim_intercept = self.cfg.model.drcnn.ssim_intercepts[0]
        ssim_std = self.cfg.model.drcnn.ssim_stds[0]
        M, N = lboxes.shape[0], rboxes.shape[0]
        lboxes_exp = lboxes.unsqueeze(1).repeat(1, N, 1)
        rboxes_exp = rboxes.unsqueeze(0).repeat(M, 1, 1)
        hmeans = (lboxes_exp[:, :, 3] - lboxes_exp[:, :, 1] + rboxes_exp[:, :, 3] - rboxes_exp[:, :, 1]) / 2
        center_disp_expectations = hmeans * ssim_coef + ssim_intercept
        cds = (lboxes_exp[:, :, 0] + lboxes_exp[:, :, 2] - rboxes_exp[:, :, 0] - rboxes_exp[:, :, 2]) / 2
        valid = (cds - center_disp_expectations).abs() < 3 * ssim_std
        nzs = valid.nonzero()
        lrois, rrois = [], []
        sls, srs = lboxes[nzs[:, 0]], rboxes[nzs[:, 1]]
        w = torch.max(sls[:, 2] - sls[:, 0], srs[:, 2] - srs[:, 0])
        h = torch.max(sls[:, 3] - sls[:, 1], srs[:, 3] - srs[:, 1])
        ws = torch.min(torch.min(w, W - sls[:, 0]), W - srs[:, 0])
        hs = torch.min(torch.min(h, H - sls[:, 1]), H - srs[:, 1])
        # self.evaltime("match_lr: prepare")
        for nz, w, h in zip(nzs, ws, hs):
            i, j = nz
            x1, y1, x2, y2 = lboxes[i]
            x1p, y1p, x2p, y2p = rboxes[j]
            lroi = img2[y1:y1 + h, x1:x1 + w, :].permute(2, 0, 1)[None] / 255.0
            rroi = img3[y1p:y1p + h, x1p:x1p + w, :].permute(2, 0, 1)[None] / 255.0
            lrois.append(lroi)
            rrois.append(rroi)
        # self.evaltime("match_lr: crop loop")
        for nz, lroi, rroi in zip(nzs, lrois, rrois):
            i, j = nz
            s = self.ssim(lroi, rroi)
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
        # self.evaltime("match_lr: ready to return")
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
        if len(rois_left) > 0:
            left_roi_images = self.crop_and_transform_roi_img(img_left.permute(2, 0, 1)[None], torch.stack(rois_left))
            right_roi_images = self.crop_and_transform_roi_img(img_right.permute(2, 0, 1)[None],
                                                               torch.stack(rois_right))
            # if len(left_roi_images) != 0:
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
        # evaltime = self.evaltime
        # evaltime('')
        dmp = DisparityMapProcessor()
        voxel_generator = self.voxel_generator
        disparity_map = dmp(left_result, right_result)
        # evaltime('dmp')
        calib = self.calib
        pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data)
        keep = (pts_rect[:, 0] > -20) & (pts_rect[:, 0] < 20) & \
               (pts_rect[:, 1] > -3) & (pts_rect[:, 1] < 3) \
               & (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
        # evaltime('to points')
        pts_rect = pts_rect[keep]
        points = calib.rect_to_lidar(pts_rect)
        rect = torch.eye(4).cuda().float()
        rect[:3, :3] = to_tensor(calib.R0, torch.float, 'cuda')
        Trv2c = torch.eye(4).cuda().float()
        Trv2c[:3, :4] = to_tensor(calib.V2C, torch.float, 'cuda')
        points = torch.cat([points, torch.full_like(points[:, 0:1], 0.5)], dim=1)
        voxel_size = voxel_generator.voxel_size
        pc_range = voxel_generator.point_cloud_range
        grid_size = voxel_generator.grid_size
        # evaltime('points cat')

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
        # evaltime('init example')
        anchor_cache = self.anchor_cache
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        example["anchors"] = anchors[None]
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
        example['calib'] = calib
        example['width'] = width
        example['height'] = height
        # evaltime('return example')
        return example

    def vis_final_result(self, left_result, right_result, calib, left_original_images, right_original_images):
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="drcnn_vis_final_result_trt",
            # auto_increase=False,
            enable=self.dbg,
        )
        # target = dps['targets']['left'][0]
        # calib = target.get_field('calib').calib
        # if self.cfg.detector_3d_on:
        dmp = DisparityMapProcessor()
        disparity_map = dmp(left_result, right_result)
        pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data)
        vis3d.add_point_cloud(pts_rect)
        vis3d.add_boxes(left_result.get_field('box3d').convert('corners').bbox_3d, name='pred')
        # vis3d.add_boxes(target.get_field('box3d').convert('corners').bbox_3d, name='gt')
        left_result.plot(left_original_images[0], show=False, calib=calib, ignore_2d_when_3d_exists=True,
                         class_names=['Car'],
                         draw_mask=False)
        outpath = osp.join(vis3d.out_folder, vis3d.sequence_name, f'{vis3d.scene_id:05d}', 'images', 'left_result.png')
        os.makedirs(osp.dirname(outpath))
        plt.savefig(outpath)
        outpath = osp.join(vis3d.out_folder, vis3d.sequence_name, f'{vis3d.scene_id:05d}', 'images', 'right_result.png')
        right_result.plot(right_original_images[0], show=False)
        plt.savefig(outpath)
        print()

    def destroy(self):
        self.yolact_inf.destory()
