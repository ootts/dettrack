# https://medium.com/test-ttile/kitti-3d-object-detection-dataset-d78a762b5a4
import numpy as np
import torch

from nds.utils.kitti_utils import inverse_rigid_trans


class KITTICalib:
    def __init__(self, calibs):
        self.P0 = calibs['P0']  # 3 x 4
        self.P1 = calibs['P1']  # 3 x 4
        self.P2 = calibs['P2']  # 3 x 4
        self.P3 = calibs['P3']  # 3 x 4
        self.R0 = calibs['R0_rect']  # 3 x 3
        self.V2C = calibs['Tr_velo_to_cam']  # 3 x 4
        self.I2V = calibs['Tr_imu_to_velo']  # 3 x 4
        self.V2I = inverse_rigid_trans(self.I2V)
        self.C2V = inverse_rigid_trans(self.V2C)

    @property
    def cu(self):
        return self.P2[0, 2]

    @property
    def cv(self):
        return self.P2[1, 2]

    @property
    def fu(self):
        return self.P2[0, 0]

    @property
    def fv(self):
        return self.P2[1, 1]

    @property
    def tx(self):
        return self.P2[0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[1, 3] / (-self.fv)

    @property
    def stereo_baseline(self):
        return self.P2[0, 3] - self.P3[0, 3]

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        if isinstance(pts, np.ndarray):
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        else:
            pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1))), dim=1)
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        # pts_rect = pts_lidar_hom @ self.V2C.T @ self.R0.T
        if isinstance(pts_lidar_hom, np.ndarray):
            pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        else:
            device = pts_lidar_hom.device
            pts_rect = pts_lidar_hom @ torch.tensor(self.V2C).float().t().to(device=device) @ torch.tensor(
                self.R0).float().t().to(device=device)
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        if isinstance(pts_rect, np.ndarray):
            pts_2d_hom = pts_rect_hom @ self.P2.T
        else:
            pts_2d_hom = pts_rect_hom @ torch.from_numpy(self.P2.T).float().to(device=pts_rect.device)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return: pts_rect:(N, 3)
        """
        if isinstance(depth_rect, np.ndarray):
            x = ((u - self.cu) * depth_rect) / self.fu + self.tx
            y = ((v - self.cv) * depth_rect) / self.fv + self.ty
            pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        else:
            x = ((u.float() - self.cu) * depth_rect.float()) / self.fu + self.tx
            y = ((v.float() - self.cv) * depth_rect.float()) / self.fv + self.ty
            pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return: pts_rect(H*W, 3), x_idxs(N), y_idxs(N)
        """
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range)
        # x_range = np.arange(0, depth_map.shape[1])
        # y_range = np.arange(0, depth_map.shape[0])
        # x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        # x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def disparity_map_to_rect(self, disparity_map, epsilon=1e-6):
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return self.depthmap_to_rect(depth_map)

    def disparity_map_to_depth_map(self, disparity_map, epsilon=1e-6):
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return depth_map

    def depth_map_to_disparity_map(self, depth_map, epsilon=1e-6):
        disparity_map = self.stereo_baseline / (depth_map + epsilon)
        return disparity_map

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

    def cam0_to_cam2(self, p):
        # TODO understand it
        t = np.array([self.tx, self.ty, 0])
        # p[:, 0] = p[:, 0] - self.tx
        # p[:, 1] = p[:, 1] - self.ty
        return p - t[None, :]

    def crop_and_resize(self, box, dst_size, return_intrinsics_only=True):
        """

        @param box:
        @param dst_size: int
        @param return_intrinsics_only:
        @return:
        """
        if not return_intrinsics_only:
            raise NotImplementedError()
        x1, y1, x2, y2 = box.tolist()
        # crop
        fu = self.fu
        fv = self.fv
        cu = self.cu
        cv = self.cv
        cu = cu - x1
        cv = cv - y1
        crop_width = x2 - x1
        crop_height = y2 - y1
        fu = fu / crop_width * dst_size
        fv = fv / crop_height * dst_size
        cu = cu / crop_width * dst_size
        cv = cv / crop_height * dst_size
        return torch.tensor([fu, fv, cu, cv]).float()
