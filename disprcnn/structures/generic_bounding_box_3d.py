# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from disprcnn.structures.point_cloud import PointCloud
import numpy as np

# from disprcnn.utils.eval_time import EvalTime

# get_time = EvalTime()
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class GeBox3DList(object):
    def __init__(self, bbox_3d, mode="xyzhwl_rxryrz"):
        self.device = bbox_3d.device if isinstance(bbox_3d, torch.Tensor) else torch.device("cpu")
        bbox_3d = torch.as_tensor(bbox_3d, dtype=torch.float32, device=self.device)

        if bbox_3d.ndimension() == 1:
            bbox_3d = bbox_3d.unsqueeze(0)
        if bbox_3d.ndimension() == 3:
            bbox_3d = bbox_3d.view(-1, 24)
        if bbox_3d.ndimension() != 2:
            raise ValueError(
                "bbox_3d should have 2 dimensions, got {} {}".format(bbox_3d.ndimension(), bbox_3d.size())
            )
        # if (mode == "ry_lhwxyz" or mode == "alpha_lhwxyz" or mode == "xyzhwl_ry") and bbox_3d.size(-1) != 7:
        #     raise ValueError(
        #         "last dimenion of bbox_3d in the ry_lhwxyz mode should have a "
        #         "size of 7, got {}".format(bbox_3d.size(-1))
        #     )
        if mode == "corners" and bbox_3d.size(-1) != 24:
            raise ValueError(
                "last dimenion of bbox_3d in the corners mode should have a "
                "size of 24, got {}".format(bbox_3d.size(-1))
            )

        if mode not in ("xyzhwl_rxryrz", "corners"):
            raise ValueError("mode should be 'xyzhwl_rxryrz' or 'corners'")

        self.bbox_3d = bbox_3d
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def __getitem__(self, item):
        bbox_3d = GeBox3DList(self.bbox_3d[item], self.mode)
        return bbox_3d

    def __len__(self):
        return self.bbox_3d.shape[0]

    def convert(self, mode):
        if mode not in ("xyzhwl_rxryrz", "corners"):
            raise ValueError("mode should be 'xyzhwl_rxryrz' or 'corners'")

        if mode == self.mode:
            return self

        corners = self._split_into_corners()
        if mode == "corners":
            bbox_3d = torch.cat(corners, dim=-1)
            box_3d_list = GeBox3DList(bbox_3d, mode=mode)
        elif mode == "xyzhwl_rxryrz":
            # todo
            dif = (corners[3] - corners[0])
            if self.frame == "velodyne":
                ry = torch.atan2(dif[:, 1], dif[:, 0]).view(-1, 1)
            else:
                ry = -(torch.atan2(dif[:, 2], dif[:, 0])).view(-1, 1)
            xyz = ((corners[7] + corners[0]) / 2).view(-1, 3)
            l = torch.norm((corners[0] - corners[3]), dim=1).view(-1, 1)
            h = torch.norm((corners[0] - corners[1]), dim=1).view(-1, 1)
            w = torch.norm((corners[0] - corners[4]), dim=1).view(-1, 1)
            if mode == "xyzhwl_ry":
                bbox_3d = torch.cat((xyz, h, w, l, ry), dim=-1)
            else:
                bbox_3d = torch.cat((ry, l, h, w, xyz), dim=-1)

            box_3d_list = GeBox3DList(bbox_3d, mode=mode)

        return box_3d_list

    def _split_into_corners(self):
        """
        :return: a list of 8 tensor, with shape of (N, 3), each row is a point of corners
        """
        if self.mode == "corners":
            corners = self.bbox_3d.split(3, dim=-1)
            return corners

        elif self.mode == "xyzhwl_rxryrz":
            if self.mode == "xyzhwl_ry":
                x, y, z, h, w, l, ry = self.bbox_3d.split(1, dim=-1)
            else:
                ry, l, h, w, x, y, z = self.bbox_3d.split(1, dim=-1)
            # get_time('generate tensor')
            zero_col = torch.zeros(ry.shape).to(self.device)
            ones_col = torch.ones(ry.shape).to(self.device)
            # get_time('zero ones over')
            # get_time(str(w.shape))
            half_w = w / 2
            ne_half_w = - half_w
            half_l = l / 2
            ne_half_l = -half_l
            cos_ry = torch.cos(ry)
            sin_ry = torch.sin(ry)
            ne_sin_ry = -sin_ry
            # get_time('compute_over')
            y_corners = torch.cat((half_w, half_w, half_w, half_w, ne_half_w, ne_half_w, ne_half_w, ne_half_w), dim=1)
            # get_time('y_over')
            z_corners = torch.cat((zero_col, h, h, zero_col, zero_col, h, h, zero_col), dim=1)
            # get_time('l_over')
            x_corners = torch.cat((ne_half_l, ne_half_l, half_l, half_l, ne_half_l, ne_half_l, half_l, half_l), dim=1)
            # get_time('cat over')
            corners_obj = (torch.stack((x_corners, y_corners, z_corners), dim=1))
            # get_time('stack over')

            R = torch.cat([cos_ry, ne_sin_ry, zero_col, sin_ry, cos_ry, zero_col,
                           zero_col, zero_col, ones_col], dim=1)
            # get_time('cat again')

            R = R.view(-1, 3, 3)
            # get_time('box3d convert')
            corners_cam = torch.matmul(R, corners_obj) + torch.cat((x, y, z), dim=-1).view(-1, 3, 1)
            corners_cam = corners_cam.transpose(1, 2).reshape(-1, 24)
            return corners_cam.split(3, dim=-1)


        else:
            raise RuntimeError("Should not be here")

    def _split_into_ry_lhwxyz(self):
        box_3d_list = self.convert("ry_lhwxyz")
        return box_3d_list.box_3d.split(1, dim=-1)

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        pts_3d = pts_3d.view(-1, 3)
        ones = torch.ones(pts_3d.shape[0]).unsqueeze(1)
        pts_3d = torch.cat([pts_3d, ones], dim=1)
        return pts_3d

    def project_world_to_rect(self, boxes_world, pose):
        pts_3d_world = self.cart2hom(boxes_world)  # nx4
        box3d_rect = torch.inverse(pose) @ pts_3d_world.transpose(0, 1)
        return box3d_rect.transpose(0, 1)[:, :3]

    def project_rect_to_world(self, boxes_rect, pose):
        pts_3d_rect = self.cart2hom(boxes_rect)  # nx4
        box3d_world = pose @ pts_3d_rect.transpose(0, 1)
        return box3d_world.transpose(0, 1)[:, :3]

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        return self

    # Tensor-like methods
    def to(self, device):
        bbox_3d = GeBox3DList(self.bbox_3d.to(device), self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox_3d.add_field(k, v)
        return bbox_3d

    def resize(self, size, *args, **kwargs):
        # TODO make sure image size is not changed before finished this part
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        return self

    def rotate(self, angle, gt_alpha):
        return self

    def scale(self, scale):
        return self

    def crop(self, box):
        return self

    def clip_to_image(self, remove_empty=True):
        return self

    def area(self):
        bbox_3d = self.bbox_3d
        if self.mode == "corners":
            corners = self._split_into_corners()
            l = torch.norm((corners[0] - corners[4]), dim=1).reshape(-1, 1)
            h = torch.norm((corners[0] - corners[1]), dim=1).reshape(-1, 1)
            w = torch.norm((corners[0] - corners[3]), dim=1).reshape(-1, 1)
            area = l * h * w
        else:
            _, l, h, w, _, _, _ = self._split_into_ry_lhwxyz()
            area = l * h * w

        return area

    def project_to_2d(self, P):
        """
        project the 3d bbox to camera plane using the intrinsic
        :param P: the intrinsic and extrinsics of camera. shape: (3, 4)
        :return bbox_2d: the projected bbox in camera plane, shape: (N, 8, 2)
        """
        box_3d_list = self.convert("corners")
        bbox_3d = box_3d_list.bbox_3d
        bbox_3d = bbox_3d.view(-1, 8, 3)

        n = bbox_3d.shape[0]
        ones = torch.ones((n, 8, 1)).to(self.bbox_3d.device)
        bbox_3d = torch.cat([bbox_3d, ones], dim=-1)
        bbox_2d = torch.matmul(P, bbox_3d.permute(0, 2, 1)).permute(0, 2, 1)

        # (N, 8, 2)
        valid = bbox_2d[:, :, 2] > 0.5

        bbox_2d = torch.stack((bbox_2d[:, :, 0] / bbox_2d[:, :, 2], bbox_2d[:, :, 1] / bbox_2d[:, :, 2]), dim=2)
        return bbox_2d, valid

    def enlarge_box3d(self, extra_width):
        boxes3d = self.convert("xyzhwl_ry").bbox_3d
        large_boxes3d = boxes3d.clone()
        large_boxes3d[:, 3:6] += extra_width * 2
        large_boxes3d[:, 1] += extra_width
        box_3d_list = Box3DList(large_boxes3d, mode="xyzhwl_ry", frame=self.frame).convert(self.mode)
        return box_3d_list

    @staticmethod
    def get_faces():
        return [[2, 3, 7, 6],
                [2, 3, 0, 1],
                [6, 7, 4, 5],
                [0, 1, 5, 4],
                [0, 4, 5, 1],
                [5, 6, 2, 1]]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes_3d={}, ".format(len(self))
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    # euler = (-0.06113487224401537, -0.010398221352184765, 0.35926017719345693)
    bbox_3d = torch.tensor([[-39.5482, 1.0015, 72.5878],
                            [-39.5280, -0.9614, 72.4898],
                            [-44.6086, -1.0026, 72.2687],
                            [-44.6288, 0.9603, 72.3666],
                            [-39.6350, 0.9002, 74.5999],
                            [-39.6148, -1.0627, 74.5020],
                            [-44.6954, -1.1039, 74.2808],
                            [-44.7155, 0.8590, 74.3788]], dtype=torch.float32)

    bbox_3d = bbox_3d.view(1, -1)

    box_3d_list = Box3DList(bbox_3d, (1280, 720))
    box_3d_list = box_3d_list.convert("ry_lhwxyz")
    # print("-----------convert to ry_lhwxyz-----------")
    # print("after convert: {}, annotation: {}".format(box_3d_list.bbox_3d[0, 0], euler[2]))
    # print("dif: {}".format(torch.norm(box_3d_list.bbox_3d[0, 0] - euler[2])))

    box_3d_list = box_3d_list.convert("corners")
    print("-----------convert to corners-----------")
    print("after convert: {}".format(box_3d_list.bbox_3d))
    print("annotation: {}".format(bbox_3d))
    print("dif: {}".format(torch.norm(box_3d_list.bbox_3d - bbox_3d)))
