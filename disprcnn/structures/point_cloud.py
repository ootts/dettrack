# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import copy
import numpy as np
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class PointCloud(object):
    def __init__(self, pts, mode, size, calib=None):
        """

        :param pts: input pts, should be 2d or 3d, with intensity or without intensity
        :param mode: one of ['ref', 'velo', 'rect', 'image']
        :param calib: calibration matrix w.r.t. point cloud
        """
        assert mode in ['ref', 'velo', 'rect', 'image', 'imu'], \
            "mode should be one of ['ref', 'velo', 'rect', 'image', 'imu']"

        self.device = pts.device if isinstance(pts, torch.Tensor) else torch.device("cpu")
        pts = torch.as_tensor(pts, dtype=torch.float, device=self.device)

        self.pts = pts
        self.mode = mode
        self.calibration = calib
        self.size = size
        self.feature = None

    def project_to(self, mode):
        """
        :param mode: one of ['ref', 'velo', 'rect', 'image', 'imu']
        :return: projected point cloud
        """
        assert mode in ['ref', 'velo', 'rect', 'image', 'imu'], \
            "mode should be one of ['ref', 'velo', 'rect', 'image', 'imu']"
        if self.calibration is None:
            raise RuntimeError("no calibration matrix is specified")
        if self.mode == mode:
            return self
        if self.mode is not 'image' and mode is not 'image':
            pts_3d = self.pts[:, :3]
            pts_3d = getattr(self.calibration, "project_{}_to_{}".format(self.mode, mode))(pts_3d)
            pts = torch.cat((pts_3d, self.pts[:, 3:]), dim=1)
        elif mode is 'image':
            pts_3d = self.pts[:, :3]
            if self.mode != 'rect':
                pts_3d = getattr(self.calibration, "project_{}_to_rect".format(self.mode))(pts_3d)
            pts_2d = self.calibration.project_rect_to_image(pts_3d)
            pts = torch.cat((pts_2d, self.pts[:, 3:]), dim=1)
        else:
            assert self.pts.shape[-1] == 4, "image mode point cloud should be [u, v, z, intensity]"
            pts_2d = self.pts[:, :3]
            pts_3d_rect = self.calibration.project_image_to_rect(pts_2d)
            if self.mode != 'rect':
                pts_3d = getattr(self.calibration, "project_{}_to_{}".format('rect', mode))(pts_3d_rect)
            pts = torch.cat((pts_3d, self.pts[:, -1].view(-1, 1)), dim=1)

        return PointCloud(pts, mode, self.size, self.calibration)

    def affine_transform(self, rot, t):
        """
        :param rot: 3x3 matrix
        :param t: 3x1 matrix
        :return: transformed point clouds
        """
        assert self.mode != 'image', "affine_transform can only manipulate 3d pts"

        t = t.view(-1)

        assert rot.shape[0] == 3 and rot.shape[1] == 3, \
            "shape of rotation matrix R should be 3x3"
        assert t.size()[0] == 3, \
            "shape of translation matrix t should be 3x1 or 3"

        affine_matrix = torch.zeros(3, 4, device=self.device)
        affine_matrix[:3, :3] = rot
        affine_matrix[:, -1] = t
        pts_3d = self.pts[:, :3]
        pts_3d_hom = self.calibration.cart2hom(pts_3d)
        pts_3d = torch.t(torch.matmul(affine_matrix, torch.t(pts_3d_hom)))
        pts_3d = torch.cat((pts_3d, self.pts[:, 3:]), dim=1)

        return PointCloud(pts_3d, self.mode, self.size, self.calibration)

    def filter_bbox_2d(self, xmin, ymin, xmax, ymax, clip_distance=2.0, reduce_by_range=False):
        pts = self.project_to('image')
        pts_2d = pts.pts
        fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
                   (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
        fov_inds = fov_inds & (pts_2d[:, 2] > clip_distance)
        if reduce_by_range:
            x_range, y_range, z_range = self.pts.new([[-40., 40. ],[ -1., 3. ], [0.,  70.4]])
            pts_x, pts_y, pts_z = self.pts[:, 0], self.pts[:, 1], self.pts[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = fov_inds & range_flag
            return pts, pts_valid_flag
        else:
            return pts, fov_inds

    def get_depth_map(self, xmin, ymin, xmax, ymax, clip_distance=2.0):
        pts_2d, fov_inds = self.filter_bbox_2d(xmin, ymin, xmax, ymax, clip_distance)
        pts_2d = pts_2d[fov_inds]
        depth_map = torch.zeros((ymax - ymin, xmax - xmin), device=self.device, dtype=torch.float)
        depth_map[pts_2d[:, 1].long(), pts_2d[:, 0].long()] = pts_2d[:, 2]
        return depth_map

    def get_intensity_map(self, xmin, ymin, xmax, ymax, clip_distance=2.0):
        assert self.__with_intensity is True, \
            "no intensity info"
        pts_2d, fov_inds = self.filter_bbox_2d(xmin, ymin, xmax, ymax, clip_distance)
        pts_2d = pts_2d[fov_inds]
        intensity_map = torch.zeros((ymax - ymin, xmax - xmin), device=self.device, dtype=torch.float)
        intensity_map[pts_2d[:, 1].long(), pts_2d[:, 0].long()] = self[fov_inds][:, -1]
        return intensity_map

    def filter_bbox_3d(self, box_list_3d):
        assert self.mode != 'image', "filter_bbox_3d can only manipulate 3d pts"
        box_list_3d = box_list_3d.convert('corners').bbox_3d.view(-1, 8, 3).to(self.device)
        index_list = []
        num_passed_list = []
        cs_list = []

        for bbox_3d in box_list_3d:
            point = copy.deepcopy(self.pts[:, :3])
            v45 = bbox_3d[5] - bbox_3d[4]
            v40 = bbox_3d[0] - bbox_3d[4]
            v47 = bbox_3d[7] - bbox_3d[4]
            point -= bbox_3d[4]
            m0 = torch.matmul(point, v45)
            m1 = torch.matmul(point, v40)
            m2 = torch.matmul(point, v47)

            cs = []
            for m, v in zip([m0, m1, m2], [v45, v40, v47]):
                c0 = 0 < m
                c1 = m < torch.matmul(v, v)
                c = c0 & c1
                cs.append(c)
            cs = cs[0] & cs[1] & cs[2]
            passed_inds = torch.nonzero(cs).squeeze(1)

            index_list.append(passed_inds)
            num_passed_list.append(torch.sum(cs))
            cs_list.append(cs)

        return num_passed_list, index_list, cs_list

    def filter_sample(self, batch_size_per_image, positive_fraction, matched_idxs):
        from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
            BalancedPositiveNegativeSampler
        )
        sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        pos_idx, neg_idx = sampler(matched_idxs)
        return PointCloud(self[pos_idx], self.mode, self.size, self.calibration), \
               PointCloud(self[neg_idx], self.mode, self.size, self.calibration)

    def pointrcnn_sample(self, sample_num, sample_depth=0x3f3f3f3f):
        np.random.seed(666)
        if sample_num < self.pts.shape[0]:
            pts_depth = self.pts[:, 2]
            pts_near_flag = pts_depth < sample_depth
            far_idxs_choice = torch.nonzero(pts_near_flag == 0).squeeze(1).cpu().numpy()
            near_idxs = torch.nonzero(pts_near_flag == 1).squeeze(1).cpu().numpy()
            # TODO stable
            near_idxs_choice = np.random.choice(near_idxs, sample_num - far_idxs_choice.shape[0], replace=False)
            # near_idxs_choice = near_idxs[:sample_num - far_idxs_choice.shape[0]]
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, self.pts.shape[0], dtype=np.int32)
            if sample_num > self.pts.shape[0]:
                extra_choice = np.random.choice(choice, sample_num - self.pts.shape[0], replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        return PointCloud(self.pts[choice, :], self.mode, self.size, self.calibration)

    def __getitem__(self, item):
        return PointCloud(self.pts[item], self.mode, self.size, self.calibration[item])

    def __len__(self):
        return self.pts.shape[0]

    def transpose(self, method):
        """`
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        pts = self.pts

        if method == FLIP_LEFT_RIGHT:
            pts[:, 0] = -pts[:, 0]

        if method == FLIP_TOP_BOTTOM:
            pts[:, 1] = -pts[:, 1]

        pointcloud = PointCloud(pts, self.mode, self.size, self.calibration)
        return pointcloud

    # Tensor-like methods
    def to(self, device):
        pointcloud = PointCloud(self.pts.to(device), self.mode, self.size, self.calibration)
        return pointcloud

    def resize(self, size):
        dst_w, dst_h = size
        src_w, src_h = self.size
        if self.calibration is not None:
            self.calibration.P[0, :] = self.calibration.P[0, :] * (dst_w / src_w)
            self.calibration.P[1, :] = self.calibration.P[1, :] * (dst_h / src_h)
        return PointCloud(self.pts, self.mode, size, self.calibration)

    @staticmethod
    def rotate_pc_along_y(pc, rot_angle):
        """
        params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
        params rot_angle: rad scalar
        Output pc: updated pc with XYZ rotated
        """
        cosval = np.cos(rot_angle)
        sinval = np.sin(rot_angle)
        rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
        rotmat = torch.from_numpy(rotmat).to(pc.device).float()
        pc[:, [0, 2]] = torch.matmul(pc[:, [0, 2]], torch.t(rotmat))
        return pc

    def rotate(self, angle):
        return PointCloud(self.rotate_pc_along_y(self.pts, angle), self.mode, self.size, self.calibration)

    def scale(self, scale):
        pts = self.pts[:,:3] * scale
        pts = torch.cat([pts, self.pts[:,-1].view(-1,1)], dim=1)
        return PointCloud(pts, self.mode, self.size, self.calibration)

    def generate_rpn_training_labels(self, box3d):
        gt_boxes3d = box3d.convert('xyzhwl_ry')
        gt_corners = box3d.convert('corners')
        # TODO change for different dataset
        # extend_gt_boxes3d = box3d.enlarge_box3d(0.06)
        extend_gt_boxes3d = box3d.enlarge_box3d(0.2)

        extend_box_corners = extend_gt_boxes3d.convert('corners')
        pts_rect = self.pts[:, :3]
        cls_label = torch.zeros((pts_rect.shape[0])).to(self.device)
        reg_label = torch.zeros((pts_rect.shape[0], 7)).to(self.device)  # dx, dy, dz, ry, h, w, l
        for k in range(gt_boxes3d.bbox_3d.shape[0]):
            _, _, cs = self.filter_bbox_3d(gt_corners[k])
            cs = cs[0]
            fg_pts_rect = pts_rect[cs]
            cls_label[cs] = 1

            _, _, enlarged_cs = self.filter_bbox_3d(extend_box_corners[k])
            enlarged_cs = enlarged_cs[0]
            # enlarge the bbox3d, ignore nearby points
            ignore_flag = ~(enlarged_cs == cs)
            cls_label[ignore_flag] = -1
            gt_box3d = gt_boxes3d[k].bbox_3d[0]
            # pixel offset of object center
            center3d = gt_box3d[0:3].clone()  # (x, y, z)
            center3d[1] -= gt_box3d[3] / 2
            reg_label[cs, 0:3] = center3d.unsqueeze(0) - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[cs, 3] = gt_box3d[3]  # h
            reg_label[cs, 4] = gt_box3d[4]  # w
            reg_label[cs, 5] = gt_box3d[5]  # l
            reg_label[cs, 6] = gt_box3d[6]  # ry
        return cls_label, reg_label

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_point={}, ".format(len(self))
        s += "point_cloud_mode={}, ".format(self.mode)
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += ")"
        return s

