import loguru
import numpy as np
import pytorch_ssim
import torch

from disprcnn.data.datasets.kitti_velodyne import random_flip, global_rotation, global_scaling_v2, \
    global_translate, \
    filter_gt_box_outside_range
from disprcnn.modeling.models.pointpillars.pointpillars import PointPillars
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.utils.pn_utils import to_array
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
from disprcnn.utils.ppp_utils.box_np_ops import rbbox2d_to_near_bbox, limit_period, sparse_sum_for_anchors_mask, \
    fused_get_anchors_area
from disprcnn.utils.ppp_utils.box_torch_ops import lidar_to_camera, box_camera_to_lidar, box_lidar_to_camera
from disprcnn.utils.ppp_utils.target_assigner import build_target_assigner
from disprcnn.utils.ppp_utils.voxel_generator import build_voxel_generator
from disprcnn.utils.timer import EvalTime

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
from disprcnn.utils.utils_3d import matrix_3x4_to_4x4
from disprcnn.utils.vis3d_ext import Vis3D
from disprcnn.utils.averagemeter import AverageMeter


class DRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.drcnn
        self.dbg = cfg.dbg is True
        self.evaltime = EvalTime(disable=not cfg.evaltime, do_print=False)
        # self.detector2d_timer = Timer(ignore_first_n=20)
        # self.idispnet_timer = Timer(ignore_first_n=20)
        if self.cfg.yolact_on:
            self.yolact = Yolact(cfg)
            ckpt = torch.load(self.cfg.pretrained_yolact, 'cpu')
            ckpt = {k.lstrip("model."): v for k, v in ckpt['model'].items()}
            self.yolact.load_state_dict(ckpt)
        if self.cfg.idispnet_on:
            self.idispnet = PSMNet(cfg)
            self.roi_align = ROIAlign((self.idispnet.input_size, self.idispnet.input_size), 1.0, 0)
        if self.cfg.detector_3d_on:
            self.pointpillars = PointPillars(cfg)
            self.voxel_generator = build_voxel_generator(self.total_cfg.voxel_generator)
            box_coder = GroundBox3dCoderTorch()
            self.target_assigner = build_target_assigner(self.total_cfg.model.pointpillars.target_assigner, box_coder)
            feature_map_size = self.cfg.detector_3d.feature_map_size
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
        self.time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.decode_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.match_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.idispnet_prep_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.idispnet_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.pp_prep_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)
        self.pp_time_meter = AverageMeter(ignore_first=20, enable=self.total_cfg.evaltime)

    def forward(self, dps):
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="drcnn_forward",
            # auto_increase=,
            enable=self.dbg,
        )
        vis3d.set_scene_id(dps['global_step'])
        loss_dict, outputs = {}, {}
        ##############  ↓ Step 1: 2D  ↓  ##############
        if 'predictions' in dps:
            left_result = dps['predictions']['left'][0]
            right_result = dps['predictions']['right'][0]
        else:
            assert self.yolact.training is False
            self.evaltime('begin')
            with torch.no_grad():
                preds_left = self.yolact({'image': dps['images']['left']})
                preds_right = self.yolact({'image': dps['images']['right']})

            h, w, _ = dps['original_images']['left'][0].shape
            self.time_meter.update(self.evaltime('yolact forward'))
            if self.total_cfg.evaltime: print('forward', self.time_meter.avg)
            left_result = self.decode_yolact_preds(preds_left, h, w, retvalid=self.cfg.retvalid)
            right_result = self.decode_yolact_preds(preds_right, h, w, add_mask=False)
            self.decode_time_meter.update(self.evaltime('decode'))
            if self.total_cfg.evaltime: print('decode', self.decode_time_meter.avg)
            left_result, right_result = self.match_lp_rp(left_result, right_result,
                                                         dps['original_images']['left'][0],
                                                         dps['original_images']['right'][0])
            self.match_time_meter.update(self.evaltime('match'))
            if self.total_cfg.evaltime: print('match', self.match_time_meter.avg)
            left_result.add_field('imgid', dps['imgid'][0].item())
            right_result.add_field('imgid', dps['imgid'][0].item())
            if self.dbg:
                left_result.plot(dps['original_images']['left'][0], show=True)
                right_result.plot(dps['original_images']['right'][0], show=True)
            ##############  ↓ Step 2: idispnet  ↓  ##############
            if self.cfg.idispnet_on:
                assert self.idispnet.training is False
                with torch.no_grad():
                    self.evaltime('idispnet begin')
                    left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps = self.prepare_idispnet_input(dps,
                                                                                                                left_result,
                                                                                                                right_result)
                    self.idispnet_prep_time_meter.update(self.evaltime('idispnet prep'))
                    if self.total_cfg.evaltime: print('idispnet prep', self.idispnet_prep_time_meter.avg)
                    if len(left_roi_images) > 0:
                        disp_output = self.idispnet({'left': left_roi_images, 'right': right_roi_images})
                    else:
                        disp_output = torch.zeros((0, self.idispnet.input_size, self.idispnet.input_size)).cuda()
                    left_result.add_field('disparity', disp_output)
                    self.idispnet_time_meter.update(self.evaltime('idispnet end'))
                    if self.total_cfg.evaltime: print('idispnet', self.idispnet_time_meter.avg)
                    self.vis_roi_disp(dps, left_result, right_result, vis3d)
        ##############  ↓ Step 3: 3D detector  ↓  ##############
        if self.cfg.detector_3d_on:
            self.evaltime('pp begin')
            pp_input = self.prepare_pointpillars_input(dps, left_result, right_result)
            self.pp_prep_time_meter.update(self.evaltime('pp prep'))
            if self.total_cfg.evaltime: print('pp prep', self.pp_prep_time_meter.avg)
            pp_output, pp_loss_dict = self.pointpillars(pp_input)
            self.pp_time_meter.update(self.evaltime('pp forward'))
            if self.total_cfg.evaltime: print('pp forward', self.pp_time_meter.avg)
            loss_dict.update(pp_loss_dict)
            if self.pointpillars.training:
                metrics = pp_output['ret']
                outputs['metrics'] = {k.replace("@", "_"): torch.tensor([v]).float().cuda() for k, v in metrics.items()}
        # print()
        ##############  ↓ Step 4: return  ↓  ##############
        outputs.update({'left': left_result, 'right': right_result})
        return outputs, loss_dict

    def decode_yolact_preds(self, preds, h, w, add_mask=True, retvalid=False):
        evaltime = EvalTime(disable=True)
        evaltime('')
        if preds[0]['detection'] is not None:
            preds = postprocess(preds, w, h, to_long=False)
            labels, scores, box2d, masks = preds
        else:
            box2d = torch.empty([0, 4], dtype=torch.float, device='cuda')
            labels = torch.empty([0], dtype=torch.long, device='cuda')
            scores = torch.empty([0], dtype=torch.float, device='cuda')
            masks = torch.empty([0, h, w], dtype=torch.long, device='cuda')
        evaltime('postprocess')
        boxlist = BoxList(box2d, (w, h))
        boxlist.add_field("labels", labels + 1)
        boxlist.add_field("scores", scores)
        if add_mask:
            keep = masks.sum(1).sum(1) > 20
            boxlist = boxlist[keep]
            masks = masks[keep]
            if retvalid:
                vec, masks = SegmentationMask(masks, (w, h), mode='mask').convert(self.cfg.mask_mode, retvalid=retvalid)
                if vec is not None and not vec.all():
                    boxlist = boxlist[vec]
            else:
                masks = SegmentationMask(masks, (w, h), mode='mask').convert(self.cfg.mask_mode, retvalid=retvalid)
            boxlist.add_map("masks", masks)
        evaltime('masks')
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

    def prepare_pointpillars_input(self, dps, left_result, right_result):
        # todo: fix this function
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="prepare_pointpillars_input",
            # auto_increase=,
            enable=self.dbg,
        )
        evaltime = EvalTime(disable=True)
        evaltime('')
        dmp = DisparityMapProcessor()
        voxel_generator = self.voxel_generator

        disparity_map = dmp(left_result, right_result)
        evaltime('disp pp')
        target = dps['targets']['left'][0]
        calib = target.get_field('calib').calib
        if self.cfg.detector_3d.use_lidar:
            loguru.logger.warning("use lidar!!")
            pts_rect = dps['lidar'][0]
        else:
            pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data)
        vis3d.add_point_cloud(pts_rect, name='pts_rect')
        keep = (pts_rect[:, 0] > -20) & (pts_rect[:, 0] < 20) & \
               (pts_rect[:, 1] > -3) & (pts_rect[:, 1] < 3) \
               & (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
        vis3d.add_point_cloud(pts_rect[keep], name='pts_rect_keep')
        points = calib.rect_to_lidar(pts_rect)
        rect = torch.eye(4).cuda().float()
        rect[:3, :3] = torch.from_numpy(calib.R0).cuda()
        Trv2c = torch.eye(4).cuda().float()
        Trv2c[:3, :4] = torch.from_numpy(calib.V2C).cuda()
        gt_boxes = box_camera_to_lidar(target.get_field('box3d').convert('xyzhwl_ry').bbox_3d[:, [0, 1, 2, 5, 3, 4, 6]],
                                       rect, Trv2c)
        evaltime('pp1')
        # augmentation
        if self.cfg.detector_3d.aug_on is True and self.pointpillars.training:
            aug_cfg = self.cfg.detector_3d.aug
            gt_boxes, points = random_flip(gt_boxes, points)
            gt_boxes, points = global_rotation(gt_boxes, points, rotation=aug_cfg.global_rotation_noise)
            gt_boxes, points = global_scaling_v2(gt_boxes, points, *aug_cfg.global_scaling_noise)

            # Global translation
            global_loc_noise_std = (0.2, 0.2, 0.2)
            gt_boxes, points = global_translate(gt_boxes, points, global_loc_noise_std)

            bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(to_array(gt_boxes), bv_range)
            gt_boxes = gt_boxes[mask]
            # gt_classes = gt_classes[mask]

        # limit rad to [-pi, pi]
        gt_boxes[:, 6] = limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)
        evaltime('pp1.1')
        vis3d.add_point_cloud(lidar_to_camera(points[:, :3], rect, Trv2c))
        box3d = box_lidar_to_camera(gt_boxes, rect, Trv2c)
        box3d = Box3DList(box3d[:, [0, 1, 2, 4, 5, 3, 6]].float(), "xyzhwl_ry")
        vis3d.add_box3dlist(box3d)
        evaltime('pp1.2')
        perm = np.random.permutation(points.shape[0])
        points = points[perm]
        # perm = torch.randperm(points.shape[0])
        # points = points[perm]
        evaltime('pp1.3')
        voxel_size = voxel_generator.voxel_size
        pc_range = voxel_generator.point_cloud_range
        grid_size = voxel_generator.grid_size
        evaltime('pp1.4')
        # [352, 400]
        evaltime('pp2')
        p = to_array(points)
        # evaltime('pp2.0.5')
        voxels, coordinates, num_points = voxel_generator.generate(p, self.cfg.detector_3d.max_number_of_voxels)
        # evaltime('pp2.1')

        coordinates = torch.from_numpy(coordinates).long().cuda()
        # evaltime('pp2.2')
        voxels = torch.from_numpy(voxels).cuda()
        num_points = torch.from_numpy(num_points).long().cuda()
        vis3d.add_point_cloud(coordinates * torch.tensor(voxel_size.tolist()[::-1]).float().cuda()[None],
                              name='coordinates')
        vis3d.add_point_cloud(voxels[..., :3].reshape(-1, 3), name='voxels')
        vis3d.add_point_cloud(points, name='points')
        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'rect': rect[None],
            'Trv2c': Trv2c[None],
            'P2': torch.from_numpy(matrix_3x4_to_4x4(calib.P2)).float().cuda()[None],
        }
        evaltime('pp3')
        anchor_cache = self.anchor_cache
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
        example["anchors"] = anchors[None]
        anchors_mask = None
        anchor_area_threshold = self.cfg.detector_3d.anchor_area_threshold
        if anchor_area_threshold >= 0:
            coors = coordinates
            dense_voxel_map = sparse_sum_for_anchors_mask(to_array(coors, int), tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = fused_get_anchors_area(dense_voxel_map, to_array(anchors_bv),
                                                  voxel_size, pc_range, grid_size)
            anchors_mask = torch.from_numpy(anchors_area > anchor_area_threshold).cuda()
            example['anchors_mask'] = anchors_mask[None]
        evaltime('pp4')
        if self.pointpillars.training:
            gt_classes = torch.ones([gt_boxes.shape[0]]).cuda().long()
            targets_dict = self.target_assigner.assign(
                to_array(anchors), to_array(gt_boxes), to_array(anchors_mask, bool),
                gt_classes=to_array(gt_classes, int),
                matched_thresholds=to_array(matched_thresholds),
                unmatched_thresholds=to_array(unmatched_thresholds))
            example.update({
                'labels': torch.from_numpy(targets_dict['labels']).long().cuda()[None],
                'reg_targets': torch.from_numpy(targets_dict['bbox_targets']).float().cuda()[None],
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).float().cuda()[None],
            })
            pos_keep = targets_dict['labels'] == 1
            box3d = box_lidar_to_camera(anchors[pos_keep], rect, Trv2c)
            box3d = Box3DList(box3d[:, [0, 1, 2, 4, 5, 3, 6]].float(), "xyzhwl_ry")
            vis3d.add_box3dlist(box3d, name='pos_anchors')
        evaltime('pp5')
        example['coordinates'] = torch.cat([example['coordinates'], torch.full_like(coordinates[:, 0:1], 0)], dim=1)
        example['image_idx'] = [torch.tensor(left_result.extra_fields['imgid'])]
        return example
