import os.path as osp
from abc import abstractmethod

import numpy as np
import torch.nn.functional as F
import torch

from disprcnn.structures.bounding_box import BoxList
from dl_ext.vision_ext.datasets.kitti.io import load_calib, load_velodyne, load_image_2, load_label_2, load_image_info

from torch import nn

from disprcnn.modeling.models.pointpillars import metrics, ops, utils
from disprcnn.modeling.models.pointpillars.submodules import PillarFeatureNet, PointPillarsScatter, RPN
from disprcnn.modeling.models.pointpillars.utils import one_hot
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
from disprcnn.utils.ppp_utils.losses import WeightedSmoothL1LocalizationLoss, SigmoidFocalClassificationLoss
from disprcnn.utils.ppp_utils.target_assigner import build_target_assigner
from disprcnn.utils.ppp_utils.voxel_generator import build_voxel_generator
from disprcnn.utils.vis3d_ext import Vis3D


class PointPillars(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.pointpillars
        voxel_generator_cfg = self.total_cfg.voxel_generator

        self.num_class = self.cfg.num_classes
        # self.use_rotate_nms = self.cfg.use_rotate_nms
        self.multiclass_nms = self.cfg.multiclass_nms
        self.nms_score_threshold = self.cfg.nms_score_threshold
        self.nms_pre_max_size = self.cfg.nms_pre_max_size
        self.nms_post_max_size = self.cfg.nms_post_max_size
        self.nms_iou_threshold = self.cfg.nms_iou_threshold
        self.use_sigmoid_score = self.cfg.use_sigmoid_score
        self.encode_background_as_zeros = self.cfg.encode_background_as_zeros
        # self.use_sparse_rpn = False
        # self.use_direction_classifier = self.cfg.use_direction_classifier
        self.use_bev = self.cfg.use_bev
        self.total_forward_time = 0.0
        self.total_postprocess_time = 0.0
        self.total_inference_count = 0
        self.num_input_features = self.cfg.num_point_features
        self.lidar_only = self.cfg.lidar_only
        self.box_coder = GroundBox3dCoderTorch()
        self.pos_cls_weight = self.cfg.pos_class_weight
        self.neg_cls_weight = self.cfg.neg_class_weight
        self.encode_rad_error_by_sin = self.cfg.encode_rad_error_by_sin
        self.loss_norm_type = self.cfg.loss_norm_type
        self.dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self.loc_loss_ftor = WeightedSmoothL1LocalizationLoss(sigma=self.cfg.localization_loss.sigma,
                                                              code_weights=self.cfg.localization_loss.code_weight)
        self.cls_loss_ftor = SigmoidFocalClassificationLoss(self.cfg.classification_loss.gamma,
                                                            self.cfg.classification_loss.alpha)
        self.cls_loss_weight = self.cfg.classification_weight
        self.loc_loss_weight = self.cfg.localization_weight
        self._direction_loss_weight = self.cfg.direction_loss_weight

        self.voxel_feature_extractor = PillarFeatureNet(
            self.cfg.num_point_features,
            use_norm=True,
            num_filters=self.cfg.voxel_feature_extractor.num_filters,
            with_distance=self.cfg.voxel_feature_extractor.with_distance,
            voxel_size=voxel_generator_cfg.voxel_size,
            pc_range=voxel_generator_cfg.point_cloud_range
        )
        self.voxel_generator = build_voxel_generator(voxel_generator_cfg)
        self.target_assigner = build_target_assigner(self.cfg.target_assigner, self.box_coder)

        grid_size = self.voxel_generator.grid_size

        vfe_num_filters = list(self.cfg.voxel_feature_extractor.num_filters)
        output_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]

        self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                            num_input_features=vfe_num_filters[-1])
        num_rpn_input_filters = self.middle_feature_extractor.nchannels

        self.rpn = RPN(
            use_norm=True,
            num_class=self.cfg.num_classes,
            layer_nums=self.cfg.rpn.layer_nums,
            layer_strides=self.cfg.rpn.layer_strides,
            num_filters=self.cfg.rpn.num_filters,
            upsample_strides=self.cfg.rpn.upsample_strides,
            num_upsample_filters=self.cfg.rpn.num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=self.target_assigner.num_anchors_per_location,
            encode_background_as_zeros=self.cfg.encode_background_as_zeros,
            # use_direction_classifier=self.cfg.use_direction_classifier,
            use_bev=self.cfg.use_bev,
            box_code_size=self.target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=self.cfg.encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=self.cfg.use_sigmoid_score,
            encode_background_as_zeros=self.cfg.encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.dbg = self.total_cfg.dbg

        if self.cfg.pretrained_model != "":
            ckpt = torch.load(self.cfg.pretrained_model, 'cpu')
            self.load_state_dict(ckpt['model'])

    def forward(self, dps):
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="pointpillars_forward",
            enable=self.dbg,
        )
        voxels = dps["voxels"]
        num_points = dps["num_points"]
        coors = dps["coordinates"]
        batch_anchors = dps["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
        preds_dict = self.rpn(spatial_features)
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        if self.training:
            labels = dps['labels']
            reg_targets = dps['reg_targets']
            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self.pos_cls_weight,
                neg_cls_weight=self.neg_cls_weight,
                dtype=voxels.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self.loc_loss_ftor,
                self.cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self.num_class,
                encode_rad_error_by_sin=self.encode_rad_error_by_sin,
                encode_background_as_zeros=self.encode_background_as_zeros,
                box_code_size=self.box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self.loc_loss_weight
            cls_pos_loss, cls_neg_loss = get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self.pos_cls_weight
            cls_neg_loss /= self.neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self.cls_loss_weight
            loss_dict = {'loc_loss_reduced': loc_loss_reduced,
                         'cls_loss_reduced': cls_loss_reduced}

            loss = loc_loss_reduced + cls_loss_reduced

            dir_targets = get_direction_target(dps['anchors'],
                                               reg_targets)
            dir_logits = preds_dict["dir_cls_preds"].view(
                batch_size_dev, -1, 2)
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_ftor(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev

            loss += dir_loss * self._direction_loss_weight

            loss_dict['dir_loss'] = dir_loss * self.cfg.direction_loss_weight

            outputs = {}
            metrics = {}  # todo
            metrics['loss'] = loss
            metrics['cls_loss'] = cls_loss
            metrics['loc_loss'] = loc_loss
            metrics['cls_pos_loss'] = cls_pos_loss
            metrics['cls_neg_loss'] = cls_neg_loss
            metrics['cls_preds'] = cls_preds
            metrics['dir_loss_reduced'] = dir_loss
            metrics['cls_loss_reduced'] = cls_loss_reduced
            metrics['loc_loss_reduced'] = loc_loss_reduced
            metrics['cared'] = cared

            outputs['ret'] = self.update_metrics(cls_loss_reduced, loc_loss_reduced, cls_preds, labels, cared)

            outputs['metrics'] = metrics
            return outputs, loss_dict
        else:
            pred_dict = self.predict(dps, preds_dict)
            score_thresh = 0.05
            score = pred_dict[0]['scores']
            keep = score > score_thresh
            score = score[keep]
            box3d = pred_dict[0]['box3d_camera']
            box3d = box3d[:, [0, 1, 2, 4, 5, 3, 6]][keep]
            box2d = pred_dict[0]['bbox'][keep]
            labels = pred_dict[0]['label_preds'][keep] + 1
            KITTIROOT = osp.expanduser('~/Datasets/kitti')
            imgid = dps['image_idx'][0].item()
            if 'width' in dps and 'height' in dps:
                h, w = dps['height'], dps['width']
            else:
                h, w, _ = load_image_info(KITTIROOT, 'training', imgid)
            result = BoxList(box2d, (w, h))
            box3d = Box3DList(box3d, "xyzhwl_ry")
            result.add_field("box3d", box3d)
            result.add_field("labels", labels)
            result.add_field("scores", score)
            result.add_field("imgid", imgid)
            if self.dbg:
                calib = load_calib(KITTIROOT, 'training', imgid)
                lidar = load_velodyne(KITTIROOT, 'training', imgid)[:, :3]
                img2 = load_image_2(KITTIROOT, 'training', imgid)
                vis3d.add_image(img2, name=f"{imgid:06d}")
                lidar = calib.lidar_to_rect(lidar)
                vis3d.add_point_cloud(lidar)
                vis_keep = (result.get_field('scores') > self.cfg.vis_threshold)
                box3d_vis = box3d[vis_keep]
                if len(box3d_vis) > 0:
                    vis3d.add_boxes(box3d_vis.convert("corners").bbox_3d.reshape(-1, 8, 3), name='pred')
                labels = load_label_2(KITTIROOT, 'training', imgid, ['Pedestrian'])
                gt_box3d = []
                for label in labels:
                    gt_box3d.append([label.x, label.y, label.z, label.h, label.w, label.l, label.ry])
                if len(gt_box3d) > 0:
                    gt_box3d = Box3DList(gt_box3d, 'xyzhwl_ry')
                    vis3d.add_boxes(gt_box3d.convert('corners').bbox_3d.reshape(-1, 8, 3), name='gt')
                print()
            output = {'left': result, 'right': result}
            return output, {}

    def predict(self, example, preds_dict):
        # t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        self.total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        # self.total_forward_time += time.time() - t
        # t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self.box_coder.code_size)
        num_class_with_bg = self.num_class
        if not self.encode_background_as_zeros:
            num_class_with_bg = self.num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds,
                                                      batch_anchors)

        batch_dir_preds = preds_dict["dir_cls_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        # else:
        #     batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            # if self.use_direction_classifier:
            if a_mask is not None:
                dir_preds = dir_preds[a_mask.bool()]
            # print(dir_preds.shape)
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self.encode_background_as_zeros:
                # this don't support softmax
                assert self.use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self.use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            # assert not self.use_rotate_nms
            # nms_func = box_torch_ops.rotate_nms
            # else:
            nms_func = ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self.multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = ops.corner_to_standup_nd(
                    box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self.num_class,
                    pre_max_size=self.nms_pre_max_size,
                    post_max_size=self.nms_post_max_size,
                    iou_threshold=self.nms_iou_threshold,
                    score_thresh=self.nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        # if self.use_direction_classifier:
                        selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    # if self.use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self.nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self.nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self.nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        # if self.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    box_preds_corners = ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = ops.corner_to_standup_nd(
                        box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self.nms_pre_max_size,
                        post_max_size=self.nms_post_max_size,
                        iou_threshold=self.nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    # if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                # if self.use_direction_classifier:
                dir_labels = selected_dir_labels
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                box_preds[..., -1] += torch.where(
                    opp_labels, torch.tensor(np.pi).cuda().float(),
                    torch.tensor(0.0).cuda().float())
                # box_preds[..., -1] += (
                #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
                # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
                # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
                # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
                # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": torch.empty([0, 4], dtype=torch.float, device='cuda'),
                    "box3d_camera": torch.empty([0, 7], dtype=torch.float, device='cuda'),
                    "box3d_lidar": torch.empty([0, 7], dtype=torch.float, device='cuda'),
                    "scores": torch.empty([0, ], dtype=torch.float, device='cuda'),
                    "label_preds": torch.empty([0, ], dtype=torch.float, device='cuda'),
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self.num_class
        if not self.encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh * 100)}"] = float(prec[i])
            ret[f"rec@{int(thresh * 100)}"] = float(recall[i])
        return ret


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         # loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    # if loss_norm_type == LossNormType.NormByNumExamples:
    #     num_examples = cared.type(dtype).sum(1, keepdim=True)
    #     num_examples = torch.clamp(num_examples, min=1.0)
    #     cls_weights /= num_examples
    #     bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
    #     reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    # elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
    pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
    reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    return cls_weights, reg_weights, cared


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = utils.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


class WeightedSoftmaxClassificationLoss:
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

        Args:
          logit_scale: When this value is high, the prediction is "diffused" and
                       when this value is low, the prediction is made peakier.
                       (default 1.0)

        """
        self.logit_scale = logit_scale

    def __call__(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(
            prediction_tensor, self.logit_scale)
        per_row_cross_ent = (softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)  # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduce=False)
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
