import numpy as np
import torch

from disprcnn.utils.ppp_utils.box_torch_ops import rotation_2d


def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    from disprcnn.utils.ppp_utils.non_max_suppression import nms_gpu
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = torch.cat([bboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    from disprcnn.modeling.models.pointpillars.utils import torch_to_np_dtype
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def multiclass_nms(nms_func,
                   boxes,
                   scores,
                   num_class,
                   pre_max_size=None,
                   post_max_size=None,
                   score_thresh=0.0,
                   iou_threshold=0.5):
    # only output [selected] * num_class, please slice by your self
    selected_per_class = []
    assert len(boxes.shape) == 3, "bbox must have shape [N, num_cls, 7]"
    assert len(scores.shape) == 2, "score must have shape [N, num_cls]"
    num_class = scores.shape[1]
    if not (boxes.shape[1] == scores.shape[1] or boxes.shape[1] == 1):
        raise ValueError('second dimension of boxes must be either 1 or equal '
                         'to the second dimension of scores')
    num_boxes = boxes.shape[0]
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]
    boxes_ids = (range(num_classes)
                 if boxes.shape[1] > 1 else [0] * num_classes)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
        # for class_idx in range(1, num_class):
        class_scores = scores[:, class_idx]
        class_boxes = boxes[:, boxes_idx]
        if score_thresh > 0.0:
            class_scores_keep = torch.nonzero(class_scores >= score_thresh)
            if class_scores_keep.shape[0] != 0:
                class_scores_keep = class_scores_keep[:, 0]
            else:
                selected_per_class.append(None)
                continue
            class_scores = class_scores[class_scores_keep]
        if class_scores.shape[0] != 0:
            if score_thresh > 0.0:
                class_boxes = class_boxes[class_scores_keep]
            keep = nms_func(class_boxes, class_scores, pre_max_size,
                            post_max_size, iou_threshold)
            if keep is not None:
                if score_thresh > 0.0:
                    selected_per_class.append(class_scores_keep[keep])
                else:
                    selected_per_class.append(keep)
            else:
                selected_per_class.append(None)
        else:
            selected_per_class.append(None)
    return selected_per_class


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def lidar_to_camera(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    from torch import stack as tstack
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([
            tstack([rot_cos, zeros, -rot_sin]),
            tstack([zeros, ones, zeros]),
            tstack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([
            tstack([rot_cos, -rot_sin, zeros]),
            tstack([rot_sin, rot_cos, zeros]),
            tstack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tstack([
            tstack([zeros, rot_cos, -rot_sin]),
            tstack([zeros, rot_sin, rot_cos]),
            tstack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res
