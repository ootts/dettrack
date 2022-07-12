import torch
import pathlib
import pickle

import numba
import numpy as np
from torch.utils.data import Dataset

from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.utils.ppp_utils import preprocess as prep, kitti_common as kitti
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
from disprcnn.utils.ppp_utils.box_np_ops import limit_period, rbbox2d_to_near_bbox, center_to_corner_box2d, \
    center_to_corner_box3d, minmax_to_corner_2d, rotation_points_single_angle, corner_to_surfaces_3d_jit, \
    box2d_to_corner_jit, corner_to_standup_nd_jit, points_in_rbbox, sparse_sum_for_anchors_mask, fused_get_anchors_area, \
    box_camera_to_lidar, lidar_to_camera, box_lidar_to_camera
from disprcnn.utils.ppp_utils.geometry import points_in_convex_polygon_jit, points_in_convex_polygon_3d_jit
from disprcnn.utils.ppp_utils.preprocess import DataBasePreprocessor
from disprcnn.utils.ppp_utils.sample_ops import DataBaseSamplerV2
from disprcnn.utils.ppp_utils.target_assigner import build_target_assigner
from disprcnn.utils.ppp_utils.voxel_generator import build_voxel_generator
from disprcnn.utils.vis3d_ext import Vis3D


class KittiVelodyneDataset(Dataset):
    def __init__(self, cfg, split, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.kitti_velodyne
        self.split = split
        info_path = self.cfg.info_path % split
        num_point_features = self.cfg.num_point_features
        box_coder = GroundBox3dCoderTorch()  # todo??
        target_assigner = build_target_assigner(self.total_cfg.model.pointpillars.target_assigner, box_coder)  # todo?
        feature_map_size = self.cfg.feature_map_size
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self.kitti_infos = infos
        self.num_point_features = num_point_features
        # generate anchors cache
        ret = target_assigner.generate_anchors(feature_map_size)  # [352, 400]
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self.db_sampler = build_db_sampler(self.cfg.db_sampler)
        self.voxel_generator = build_voxel_generator(self.total_cfg.voxel_generator)
        self.anchor_cache = anchor_cache
        self.target_assigner = target_assigner

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, idx):
        info = self.kitti_infos[idx]
        root_path = self.cfg.root_path
        v_path = pathlib.Path(root_path) / info['velodyne_path']
        v_path = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name

        points = np.fromfile(str(v_path), dtype=np.float32).reshape([-1, self.num_point_features])
        image_idx = info['image_idx']
        rect = info['calib/R0_rect'].astype(np.float32)
        Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib/P2'].astype(np.float32)

        input_dict = {
            'points': points,
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            'image_idx': image_idx,
            'image_path': info['img_path'],
        }

        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = kitti.remove_dontcare(annos)
            loc = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            difficulty = annos["difficulty"]
            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'difficulty': difficulty,
            })
            if 'group_ids' in annos:
                input_dict['group_ids'] = annos["group_ids"]
        example = self.prep_pointcloud(input_dict=input_dict)
        example["image_idx"] = image_idx
        example["image_shape"] = input_dict["image_shape"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        example.pop('num_voxels')
        return example

    def prep_pointcloud(self, input_dict):
        root_path = self.cfg.root_path
        class_names = list(self.cfg.class_names)
        voxel_generator = self.voxel_generator
        target_assigner = self.target_assigner
        training = 'train' in self.split
        max_voxels = self.cfg.max_number_of_voxels
        shuffle_points = self.cfg.shuffle_points
        gt_rotation_noise = list(self.cfg.groundtruth_rotation_uniform_noise)
        gt_loc_noise_std = list(self.cfg.groundtruth_localization_noise_std)
        global_rotation_noise = list(self.cfg.global_rotation_uniform_noise)
        global_scaling_noise = list(self.cfg.global_scaling_uniform_noise)
        global_loc_noise_std = (0.2, 0.2, 0.2)
        global_random_rot_range = list(self.cfg.global_random_rotation_range_per_object)
        db_sampler = self.db_sampler
        without_reflectivity = self.cfg.without_reflectivity
        anchor_area_threshold = self.cfg.anchor_area_threshold
        remove_points_after_sample = self.cfg.remove_points_after_sample
        anchor_cache = self.anchor_cache
        num_point_features = self.num_point_features
        # #####
        points = input_dict["points"]
        if training:
            gt_boxes = input_dict["gt_boxes"]
            gt_names = input_dict["gt_names"]
        rect = input_dict["rect"]
        Trv2c = input_dict["Trv2c"]
        P2 = input_dict["P2"]
        vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="kitti_velodyne_loader_prep",
            # auto_increase=,
            enable=self.total_cfg.dbg,
        )
        if training:
            selected = kitti.drop_arrays_by_name(gt_names, ["DontCare"])
            gt_boxes = gt_boxes[selected]
            gt_names = gt_names[selected]

            gt_boxes = box_camera_to_lidar(gt_boxes, rect, Trv2c)
            gt_boxes_mask = np.array([n in class_names for n in gt_names], dtype=np.bool_)
            assert db_sampler is not None
            sampled_dict = db_sampler.sample_all(root_path, gt_boxes, gt_names, num_point_features, False,
                                                 rect=rect, Trv2c=Trv2c, P2=P2)
            # if self.total_cfg.dbg:

            vis3d.add_point_cloud(lidar_to_camera(sampled_dict['points'][:, :3], rect, Trv2c))
            box3d = box_lidar_to_camera(sampled_dict['gt_boxes'], rect, Trv2c)
            box3d = Box3DList(torch.from_numpy(box3d[:, [0, 1, 2, 4, 5, 3, 6]]).float(), "xyzhwl_ry")
            vis3d.add_box3dlist(box3d)
            # xyz_lidar, w, l, h, r
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                if remove_points_after_sample:  # not entered
                    points = remove_points_in_boxes(points, sampled_gt_boxes)

                points = np.concatenate([sampled_points, points], axis=0)

                vis3d.add_point_cloud(lidar_to_camera(points[:, :3], rect, Trv2c))
                box3d = box_lidar_to_camera(gt_boxes, rect, Trv2c)
                box3d = Box3DList(torch.from_numpy(box3d[:, [0, 1, 2, 4, 5, 3, 6]]).float(), "xyzhwl_ry")
                vis3d.add_box3dlist(box3d)
            if without_reflectivity:  # not entered
                used_point_axes = list(range(num_point_features))
                used_point_axes.pop(3)
                points = points[:, used_point_axes]
            noise_per_object_v3_(
                gt_boxes,
                points,
                gt_boxes_mask,
                rotation_perturb=gt_rotation_noise,
                center_noise_std=gt_loc_noise_std,
                global_random_rot_range=global_random_rot_range,
                num_try=100)
            vis3d.add_point_cloud(lidar_to_camera(points[:, :3], rect, Trv2c))
            box3d = box_lidar_to_camera(gt_boxes, rect, Trv2c)
            box3d = Box3DList(torch.from_numpy(box3d[:, [0, 1, 2, 4, 5, 3, 6]]).float(), "xyzhwl_ry")
            vis3d.add_box3dlist(box3d)
            # should remove unrelated objects after noise per object
            gt_boxes = gt_boxes[gt_boxes_mask]
            gt_names = gt_names[gt_boxes_mask]
            gt_classes = np.array([class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

            vis3d.add_point_cloud(lidar_to_camera(points[:, :3], rect, Trv2c))
            box3d = box_lidar_to_camera(gt_boxes, rect, Trv2c)
            box3d = Box3DList(torch.from_numpy(box3d[:, [0, 1, 2, 4, 5, 3, 6]]).float(), "xyzhwl_ry")
            vis3d.add_box3dlist(box3d)

            gt_boxes, points = random_flip(gt_boxes, points)
            gt_boxes, points = global_rotation(gt_boxes, points, rotation=global_rotation_noise)
            gt_boxes, points = global_scaling_v2(gt_boxes, points, *global_scaling_noise)

            # Global translation
            gt_boxes, points = global_translate(gt_boxes, points, global_loc_noise_std)

            bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(gt_boxes, bv_range)
            gt_boxes = gt_boxes[mask]
            gt_classes = gt_classes[mask]

            # limit rad to [-pi, pi]
            gt_boxes[:, 6] = limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

            vis3d.add_point_cloud(lidar_to_camera(points[:, :3], rect, Trv2c))
            box3d = box_lidar_to_camera(gt_boxes, rect, Trv2c)
            box3d = Box3DList(torch.from_numpy(box3d[:, [0, 1, 2, 4, 5, 3, 6]]).float(), "xyzhwl_ry")
            vis3d.add_box3dlist(box3d)
        if shuffle_points:
            # shuffle is a little slow.
            np.random.shuffle(points)

        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = voxel_generator.voxel_size
        pc_range = voxel_generator.point_cloud_range
        grid_size = voxel_generator.grid_size
        # [352, 400]

        voxels, coordinates, num_points = voxel_generator.generate(points, max_voxels)

        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
        }
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
        example["anchors"] = anchors
        anchors_mask = None
        if anchor_area_threshold >= 0:
            coors = coordinates
            dense_voxel_map = sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > anchor_area_threshold
            example['anchors_mask'] = anchors_mask
        if training:
            targets_dict = target_assigner.assign(
                anchors, gt_boxes, anchors_mask, gt_classes=gt_classes,
                matched_thresholds=matched_thresholds, unmatched_thresholds=unmatched_thresholds)
            example.update({
                'labels': targets_dict['labels'],
                'reg_targets': targets_dict['bbox_targets'],
                'reg_weights': targets_dict['bbox_outside_weights'],
            })
        return example


def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = center_to_corner_box2d(gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, 6])
    bounding_box = minmax_to_corner_2d(np.asarray(limit_range)[None])
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)


def global_translate(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    noise_translate = np.array([np.random.normal(0, noise_translate_std[0], 1),
                                np.random.normal(0, noise_translate_std[1], 1),
                                np.random.normal(0, noise_translate_std[0], 1)]).T
    if isinstance(points, torch.Tensor):
        nt = torch.from_numpy(noise_translate).float().to(points.device)
        points[:, :3] += nt
    else:
        points[:, :3] += noise_translate
    if isinstance(points, torch.Tensor):
        nt = torch.from_numpy(noise_translate).float().to(points.device)
        gt_boxes[:, :3] += nt
    else:
        gt_boxes[:, :3] += noise_translate

    return gt_boxes, points


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = rotation_points_single_angle(points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = rotation_points_single_angle(gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def noise_per_object_v3_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=100,
                         # group_ids=None
                         ):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])

    origin = [0.5, 0.5, 0]
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)
    if not enable_grot:
        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                       valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises,
                                           rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                    C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                    C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                        boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                        boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                            qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                            qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises,
                      global_rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[0, 2:
                                                    4] * corners_norm @ rot_mat_T + current_box[0, :
                                                                                                   2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask


def _select_transform(transform, indices):
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def remove_points_in_boxes(points, boxes):
    masks = points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def build_db_sampler(sampler_config):
    cfg = sampler_config
    groups = cfg.sample_groups
    prepors = [
        build_db_preprocess(c)
        for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [{g["key"]: g["value"]} for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler


def build_db_preprocess(config):
    # prep_type = db_prep_config.WhichOneof('database_preprocessing_step')
    prep_type = config["name"]
    if prep_type == 'filter_by_difficulty':
        # cfg = db_prep_config.filter_by_difficulty
        return prep.DBFilterByDifficulty(config["removed_difficulties"])
    elif prep_type == 'filter_by_min_num_points':
        return prep.DBFilterByMinNumPoint({config["key"]: config["value"]})
    else:
        raise ValueError("unknown database prep type")
