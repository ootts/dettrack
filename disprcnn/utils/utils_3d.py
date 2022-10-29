import open3d as o3d
import cv2
# import pytorch3d.transforms
# import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
# from monotrack.utils.vision_ext.datasets.kitti.load import Calibration
import tqdm
import trimesh
from dl_ext.primitive import safe_zip
from dl_ext.vision_ext.datasets.kitti.structures import Calibration
from multipledispatch import dispatch
from packaging import version
from scipy.spatial.distance import cdist
from torchvision.ops import nms

from disprcnn.utils.pn_utils import padded_stack, to_tensor, to_array
from disprcnn.utils.tsdf_fusion_python import get_view_frustum, TSDFVolume
from disprcnn.utils.utils_2d import bilinear_sampler
from disprcnn.utils.vis3d_ext import Vis3D


# from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import ball_query as ball_query_pointnet2


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = Calibration.cart_to_hom(pts)
    # pts = np.hstack((pts, np.ones((pts.shape[0], 1))))  # 4XN
    pts = pts @ pose.T  # 4xN
    pts = Calibration.hom_to_cart(pts)
    return pts


@dispatch(np.ndarray, np.ndarray)
def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = Calibration.cart_to_hom(pts)
    pts = pts @ pose.T
    pts = Calibration.hom_to_cart(pts)
    return pts


@dispatch(torch.Tensor, torch.Tensor)
def canonical_to_camera(pts, pose):
    pts = Calibration.cart_to_hom(pts)
    pts = pts @ pose.t()
    pts = Calibration.hom_to_cart(pts)
    return pts


transform_points = canonical_to_camera


# def canonical_to_camera(pts, pose, calib=None):
#     """
#     :param pts: Nx3
#     :param pose: 4x4
#     :param calib:Calibration
#     :return:
#     """
#     # P2 = calib.P2  # 3x4
#     pts = pts.T  # 3xN
#     pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
#     p = pose @ pts  # 4xN
#     if calib is None:
#         p[0:3] /= p[3:]
#         p = p[0:3]
#     else:
#         p = calib.P2 @ p
#         p[0:2] /= p[2:]
#         p = p[0:2]
#     p = p.T
#     return p


def camera_to_canonical(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :return:
    """
    if isinstance(pts, np.ndarray) and isinstance(pose, np.ndarray):
        pts = pts.T  # 3xN
        pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
        p = np.linalg.inv(pose) @ pts  # 4xN
        p[0:3] /= p[3:]
        p = p[0:3]
        p = p.T
        return p
    else:
        pts = Calibration.cart_to_hom(pts)
        pts = pts @ torch.inverse(pose).t()
        pts = Calibration.hom_to_cart(pts)
        return pts


# def cam0_to_cam2(p, calib):
#     """
#     :param p: Nx3
#     :param calib:
#     :return:
#     """
#     P2 = calib.P2
#     fu = P2[0, 0]
#     b = -P2[0, 3] / fu
#     p[:, 0] = p[:, 0] - b
#     return p


def xyzr_to_pose4x4(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def xyzr_to_pose4x4_torch(x, y, z, r):
    if isinstance(x, torch.Tensor):
        pose = torch.eye(4, device=x.device, dtype=torch.float)
        pose[0, 0] = torch.cos(r)
        pose[0, 2] = torch.sin(r)
        pose[2, 0] = -torch.sin(r)
        pose[2, 2] = torch.cos(r)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose
    else:
        return torch.from_numpy(xyzr_to_pose4x4_np(x, y, z, r)).float()


from multipledispatch import dispatch


@dispatch(np.ndarray)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = np.arctan2(sin, cos)
    return x, y, z, angle


@dispatch(torch.Tensor)
def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = torch.atan2(sin, cos)
    return x, y, z, angle


def camera_coordinate_to_world_coordinate(pts_in_camera, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_camera: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_camera, np.ndarray):
        pts_hom = np.hstack((pts_in_camera, np.ones((pts_in_camera.shape[0], 1), dtype=np.float32)))
        pts_world = pts_hom @ cam_pose.T
    else:
        ones = torch.ones((pts_in_camera.shape[0], 1), dtype=torch.float32, device=pts_in_camera.device)
        pts_hom = torch.cat((pts_in_camera, ones), dim=1)
        cam_pose = torch.tensor(cam_pose).float().to(device=pts_in_camera.device)
        pts_world = pts_hom @ cam_pose.t()
    pts_world = pts_world[:, :3] / pts_world[:, 3:]
    return pts_world


def world_coordinate_to_camera_coordinate(pts_in_world, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_world: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_world, np.ndarray):
        cam_pose_inv = np.linalg.inv(cam_pose)
        pts_hom = np.hstack((pts_in_world, np.ones((pts_in_world.shape[0], 1), dtype=np.float32)))
        pts_cam = pts_hom @ cam_pose_inv.T
    else:
        cam_pose = cam_pose.float().to(device=pts_in_world.device)
        cam_pose_inv = torch.inverse(cam_pose)
        ones = torch.ones((pts_in_world.shape[0], 1), dtype=torch.float32, device=pts_in_world.device)
        pts_hom = torch.cat((pts_in_world, ones), dim=1)
        pts_cam = pts_hom @ cam_pose_inv.t()
    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]
    return pts_cam


# def canonical_coordinate_to_rect(canonical_pts, object_pose):
#     """
#     :param canonical_pts: n,3
#     :param object_pose: 4x4 [R t \\ 0 1]
#     :return:
#     """
#     n, _ = canonical_pts.shape
#     if isinstance(canonical_pts, np.ndarray):
#         p = np.hstack(canonical_pts, np.ones((n, 1)))
#         p = p @ object_pose.T
#     else:
#         p = torch.cat((canonical_pts, torch.ones((n, 1), dtype=canonical_pts.dtype, device=canonical_pts.device)), 1)
#         p = p @ object_pose.t()
#     p = p[:, :3] / p[:, 3:]
#     return p
#
#
# def nocs_coordinate_to_canonical_coordinate()


def xyzr_to_pose4x4_np(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def canonical_to_camera_np(pts, pose, calib=None):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib: KITTICalib
    :return:
    """
    pts = pts.T  # 3xN
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
    p = pose @ pts  # 4xN
    if calib is None:
        p[0:3] /= p[3:]
        p = p[0:3]
    else:
        p = calib.P2 @ p
        p[0:2] /= p[2:]
        p = p[0:2]
    p = p.T
    return p


def rotx_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([ones, zeros, zeros,
                    zeros, c, -s,
                    zeros, s, c])
    return rot.reshape((-1, 3, 3))


def roty_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    # TODO validate
    rot = np.stack([c, zeros, s,
                    zeros, ones, zeros,
                    -s, zeros, c])
    return rot.reshape((-1, 3, 3))


def roty_torch(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    # if a.shape[-1] != 1:
    #     a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    # TODO validate
    rot = torch.stack([c, zeros, s,
                       zeros, ones, zeros,
                       -s, zeros, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotz_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(np.float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, -s, zeros,
                    s, c, zeros,
                    zeros, zeros, ones])
    return rot.reshape((-1, 3, 3))


def rotz_torch(a):
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    if a.shape[-1] != 1:
        a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, -s, zeros,
                       s, c, zeros,
                       zeros, zeros, ones], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotx(t):
    """
    Rotation along the x-axis.
    :param t: tensor of (N, 1) or (N), or float, or int
              angle
    :return: tensor of (N, 3, 3)
             rotation matrix
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t])
    if t.shape[-1] != 1:
        t = t[..., None]
    t = t.type(torch.float)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(t)
    c = torch.cos(t)
    s = torch.sin(t)
    rot = torch.stack([ones, zeros, zeros,
                       zeros, c, -s,
                       zeros, s, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def matrix_3x4_to_4x4(a):
    if len(a.shape) == 2:
        assert a.shape == (3, 4)
    else:
        assert len(a.shape) == 3
        assert a.shape[1:] == (3, 4)
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            ones = np.array([[0, 0, 0, 1]])
            return np.vstack((a, ones))
        else:
            ones = np.array([[0, 0, 0, 1]])[None].repeat(a.shape[0], axis=0)
            return np.concatenate((a, ones), axis=1)
    else:
        ones = torch.tensor([[0, 0, 0, 1]]).float().to(device=a.device)
        if a.ndim == 3:
            ones = ones[None].repeat(a.shape[0], 1, 1)
            ret = torch.cat((a, ones), dim=1)
        else:
            ret = torch.cat((a, ones), dim=0)
        return ret


def matrix_3x3_to_4x4(a):
    assert a.shape == (3, 3)
    if isinstance(a, np.ndarray):
        ret = np.eye(4)
    else:
        ret = torch.eye(4).float().to(a.device)
    ret[:3, :3] = a
    return ret


def radian_to_negpi_to_pi(x):
    if isinstance(x, torch.Tensor):
        return x - 2 * torch.floor((x / np.pi + 1) / 2) * np.pi
    else:
        return x - 2 * np.floor((x / np.pi + 1) / 2) * np.pi


def filter_bbox_3d(bbox_3d, point):
    """
    @param bbox_3d: corners (8,3)
    @param point: (?,3)
    @return:
    """
    v45 = bbox_3d[5] - bbox_3d[4]
    v40 = bbox_3d[0] - bbox_3d[4]
    v47 = bbox_3d[7] - bbox_3d[4]
    # point -= bbox_3d[4]
    point = point - bbox_3d[4]
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
    num_passed = torch.sum(cs)
    return num_passed, passed_inds, cs


def triangulate_static_points(projmat1, projmat2, projpoints1, projpoints2) -> np.ndarray:
    """
    triangulate points python version
    :param projmat1: 3x4
    :param projmat2: 3x4
    :param projpoints1: 2xn
    :param projpoints2: 2xn
    :return: points4D: 4xn
    """
    num_points = projpoints1.shape[1]
    assert projpoints2.shape[1] == num_points
    points4D = []
    for i in range(num_points):
        x, y = projpoints1[:, i]
        xp, yp = projpoints2[:, i]
        A = np.stack([x * projmat1[2] - projmat1[0],
                      y * projmat1[2] - projmat1[1],
                      xp * projmat2[2] - projmat2[0],
                      yp * projmat2[2] - projmat2[1]], axis=0)
        U, S, V = np.linalg.svd(A)
        points4D.append(V[3, 0:4])
    points4D = np.stack(points4D).T
    return points4D


def subsample_mask_by_grid(pc_rect):
    def filter_mask(pc_rect):
        """Return index of points that lies within the region defined below."""
        valid_inds = (pc_rect[:, 2] < 80) * \
                     (pc_rect[:, 2] > 1) * \
                     (pc_rect[:, 0] < 40) * \
                     (pc_rect[:, 0] >= -40) * \
                     (pc_rect[:, 1] < 2.5) * \
                     (pc_rect[:, 1] >= -1)
        return valid_inds

    GRID_SIZE = 0.1
    index_field_sample = np.full(
        (35, int(80 / 0.1), int(80 / 0.1)), -1, dtype=np.int32)

    N = pc_rect.shape[0]
    perm = np.random.permutation(pc_rect.shape[0])
    pc_rect = pc_rect[perm]

    range_filter = filter_mask(pc_rect)
    pc_rect = pc_rect[range_filter]

    pc_rect_quantized = np.floor(pc_rect[:, :3] / GRID_SIZE).astype(np.int32)
    pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
                              + int(80 / GRID_SIZE / 2)
    pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + int(1 / GRID_SIZE)

    index_field = index_field_sample.copy()

    index_field[pc_rect_quantized[:, 1],
                pc_rect_quantized[:, 2], pc_rect_quantized[:, 0]] = np.arange(pc_rect.shape[0])
    mask = np.zeros(perm.shape, dtype=np.bool)
    mask[perm[range_filter][index_field[index_field >= 0]]] = 1
    return mask


def img_to_rect(fu, fv, cu, cv, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    # x = ((u - cu) * depth_rect) / fu
    # y = ((v - cv) * depth_rect) / fv
    # pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect


def depth_to_rect(fu, fv, cu, cv, depth_map, ray_mode=False, select_coords=None):
    """

    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param depth_map:
    :param ray_mode: whether values in depth_map are Z or norm
    :return:
    """
    if len(depth_map.shape) == 2:
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
    else:
        x_idxs = select_coords[:, 1].float()
        y_idxs = select_coords[:, 0].float()
        depth = depth_map
    if ray_mode is True:
        if isinstance(depth, torch.Tensor):
            depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                    (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
        else:
            depth = depth / (((x_idxs - cu) / fu) ** 2 + (
                    (y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    pts_rect = img_to_rect(fu, fv, cu, cv, x_idxs, y_idxs, depth)
    # if ray_mode is True:
    #     todo check this
    # pts_rect[:, 2] = (pts_rect[:, 2] ** 2 - pts_rect[:, 0] ** 2 - pts_rect[:, 1] ** 2) ** 0.5
    return pts_rect


def depth_norm_to_depth_z(K, depth_map):
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if isinstance(depth_map, np.ndarray):
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    else:
        x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
        y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
        y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing='ij')
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    depth = depth_map[y_idxs, x_idxs]

    if isinstance(depth, torch.Tensor):
        depth = depth / (((x_idxs.float() - cu.float()) / fu.float()) ** 2 + (
                (y_idxs.float() - cv.float()) / fv.float()) ** 2 + 1) ** 0.5
    else:
        depth = depth / (((x_idxs - cu) / fu) ** 2 + ((y_idxs - cv) / fv) ** 2 + 1) ** 0.5
    return depth.reshape(depth_map.shape)


def rect_to_img(fu, fv, cu, cv, pts_rect):
    if isinstance(pts_rect, np.ndarray):
        K = np.array([[fu, 0, cu],
                      [0, fv, cv],
                      [0, 0, 1]])
        pts_2d_hom = pts_rect @ K.T
        pts_img = Calibration.hom_to_cart(pts_2d_hom)
    else:
        device = pts_rect.device
        P2 = torch.tensor([[fu, 0, cu],
                           [0, fv, cv],
                           [0, 0, 1]], dtype=torch.float, device=device)
        pts_2d_hom = pts_rect @ P2.t()
        pts_img = Calibration.hom_to_cart(pts_2d_hom)
    return pts_img


def backproject_flow3d_torch(flow2d, depth0, depth1, intrinsics, campose0, campose1):
    """ compute 3D flow from 2D flow + depth change """
    # raise NotImplementedError()

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(),
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    # X1 = depth1 * ((x1 - cx) / fx)
    # Y1 = depth1 * ((y1 - cy) / fy)
    # Z1 = depth1

    grid = torch.stack([x1, y1], dim=-1)[None]
    grid[:, :, :, 0] = grid[:, :, :, 0] / (wd - 1)
    grid[:, :, :, 1] = grid[:, :, :, 1] / (ht - 1)
    grid = grid * 2 - 1
    depth1_interp = torch.nn.functional.grid_sample(
        depth1[None, None],
        grid,
        mode='bilinear'
    )[0, 0]

    X1 = depth1_interp * ((x1 - cx) / fx)
    Y1 = depth1_interp * ((y1 - cy) / fy)
    Z1 = depth1_interp  # todo: or interpolated depth?

    pts0_cam = torch.stack([X0, Y0, Z0], dim=-1)
    pts1_cam = torch.stack([X1, Y1, Z1], dim=-1)

    # pts0_world = camera_coordinate_to_world_coordinate(pts0_cam, campose0)
    # pts1_world = camera_coordinate_to_world_coordinate(pts1_cam, campose1)

    # flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    flow3d = pts1_cam - pts0_cam
    return flow3d, pts0_cam, pts1_cam


def backproject_flow3d_np(flow2d, depth0, depth1, intrinsics0, intrinsics1, campose0, campose1, occlusion,
                          max_norm=0.5, interpolation_mode='nearest'):
    """
    compute 3D flow from 2D flow + depth change
    :param flow2d:
    :param depth0:
    :param depth1:
    :param intrinsics0:
    :param intrinsics1:
    :param campose0:
    :param campose1:
    :param occlusion: 0,255
    :param max_norm:
    :return:
    """

    ht, wd = flow2d.shape[0:2]

    fx0, fy0, cx0, cy0 = intrinsics0[0, 0], intrinsics0[1, 1], intrinsics0[0, 2], intrinsics0[1, 2]
    fx1, fy1, cx1, cy1 = intrinsics1[0, 0], intrinsics1[1, 1], intrinsics1[0, 2], intrinsics1[1, 2]

    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx0) / fx0)
    Y0 = depth0 * ((y0 - cy0) / fy0)
    Z0 = depth0

    # bilinear sample
    grid = torch.from_numpy(np.stack([x1, y1], axis=-1)[None]).float()
    grid[:, :, :, 0] = grid[:, :, :, 0] / (wd - 1)
    grid[:, :, :, 1] = grid[:, :, :, 1] / (ht - 1)
    grid = grid * 2 - 1
    depth1_interp = torch.nn.functional.grid_sample(
        torch.from_numpy(depth1)[None, None],
        grid,
        mode=interpolation_mode
    )[0, 0].numpy()

    X1 = depth1_interp * ((x1 - cx1) / fx1)
    Y1 = depth1_interp * ((y1 - cy1) / fy1)
    Z1 = depth1_interp  # todo: or interpolated depth?

    pts0_cam = np.stack([X0, Y0, Z0], axis=-1).reshape(-1, 3)
    pts1_cam = np.stack([X1, Y1, Z1], axis=-1).reshape(-1, 3)
    posenorm = np.linalg.inv(campose0)
    campose0, campose1 = posenorm @ campose0, posenorm @ campose1
    pts0_world = camera_coordinate_to_world_coordinate(pts0_cam, campose0)
    pts1_world = camera_coordinate_to_world_coordinate(pts1_cam, campose1)

    # flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    flow3d = pts1_world - pts0_world
    # with Vis3D(
    #         xyz_pattern=('x', '-y', '-z'),
    #         out_folder="dbg"
    # ) as vis3d:
    #     vis3d.set_scene_id(0)
    #     vis3d.add_point_cloud(pts0_world)
    #     vis3d.add_point_cloud(pts1_world)
    #     print()
    flow3d = flow3d.reshape(ht, wd, 3)
    norm = np.linalg.norm(flow3d, axis=-1)
    ncc = occlusion != 255
    flow3dncc = flow3d * ncc[:, :, None]
    normncc = norm * ncc
    # plt.imshow(normncc, 'jet')
    # plt.show()
    normmask = normncc < max_norm
    flow3dncc = flow3dncc * normmask[:, :, None]
    return flow3dncc, normmask


def optical_flow_to_voxel_matching(st0, st1, optical_flow, voxel_size, fu, fv, cu, cv, target_points):
    """
    todo: add mutual nearest neighbor
    :param st0:
    :param st1:
    :param optical_flow:
    :param voxel_size:
    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param target_points: 3,H,W
    :return:
    """
    import torchsparse.nn as spnn

    ptsimg = rect_to_img(fu, fv, cu, cv, st0.C * voxel_size)
    sampled_optical_flow = bilinear_sampler(torch.from_numpy(optical_flow).float()[None],
                                            torch.from_numpy(ptsimg).float()[None, None]
                                            ).numpy()[0][:, 0, :].transpose(1, 0)  # N,2
    ptsimgp = ptsimg + sampled_optical_flow
    pointsp = target_points
    sampled_pointsp = bilinear_sampler(torch.from_numpy(pointsp).float()[None],
                                       torch.from_numpy(ptsimgp).float()[None, None]
                                       ).numpy()[0, :, 0, :].transpose(1, 0)
    rounded_coord = np.round(sampled_pointsp / voxel_size).astype(np.int32)
    # inds, invs = sparse_quantize(rounded_coord, sampled_pointsp,
    #                              return_index=True,
    #                              return_invs=True)
    # voxel_pc = rounded_coord[inds]
    # voxel_pc = rounded_coord[inds[invs]]
    voxel_pc = rounded_coord
    source_coord = np.hstack([voxel_pc, np.zeros((voxel_pc.shape[0], 1))])
    target_coord = np.hstack([st1.C, np.zeros((st1.C.shape[0], 1))])
    source_hash = spnn.functional.sphash(torch.floor(torch.from_numpy(source_coord).float()).int())
    target_hash = spnn.functional.sphash(torch.floor(torch.from_numpy(target_coord).float()).int())
    idx_query = spnn.functional.sphashquery(source_hash, target_hash).numpy()
    with Vis3D(
            # xyz_pattern=('x', 'y', 'z'),
            # out_folder="dbg",
            sequence='optical_flow_to_voxel_matching',
            auto_increase=True
    ) as vis3d:
        # vis3d.set_scene_id(0)
        vis3d.add_point_cloud(source_coord[:, :3] * voxel_size, name='source_coord')
        vis3d.add_point_cloud(target_coord[:, :3] * voxel_size, name='target_coord')
        print()
    # idx_query[197]=201,then voxel_pc[197]==st1.C[201]
    valid_src_idxs = (idx_query != -1).nonzero()[0]
    valid_tgts = idx_query[valid_src_idxs]
    matching = -np.ones(source_hash.shape[0], dtype=np.int)
    matching[valid_src_idxs] = valid_tgts
    # todo: too sparse. ~10%
    return matching


def optical_flow_to_voxel_matching_nn(st0, st1, optical_flow, voxel_size, fu, fv, cu, cv, target_points):
    """
    todo: add mutual nearest neighbor
    :param st0:
    :param st1:
    :param optical_flow:
    :param voxel_size:
    :param fu:
    :param fv:
    :param cu:
    :param cv:
    :param target_points: 3,H,W
    :return:
    """
    ptsimg = rect_to_img(fu, fv, cu, cv, st0.C * voxel_size)
    sampled_optical_flow = bilinear_sampler(torch.from_numpy(optical_flow).float()[None],
                                            torch.from_numpy(ptsimg).float()[None, None]
                                            ).numpy()[0][:, 0, :].transpose(1, 0)  # N,2
    ptsimgp = ptsimg + sampled_optical_flow
    pointsp = target_points
    sampled_pointsp = bilinear_sampler(torch.from_numpy(pointsp).float()[None],
                                       torch.from_numpy(ptsimgp).float()[None, None]
                                       ).numpy()[0, :, 0, :].transpose(1, 0)
    rounded_coord = np.round(sampled_pointsp / voxel_size).astype(np.int32)
    # inds, invs = sparse_quantize(rounded_coord, sampled_pointsp,
    #                              return_index=True,
    #                              return_invs=True)
    # voxel_pc = rounded_coord[inds]
    # voxel_pc = rounded_coord[inds[invs]]
    voxel_pc = rounded_coord
    # source_coord = np.hstack([voxel_pc, np.zeros((voxel_pc.shape[0], 1))])
    # target_coord = np.hstack([st1.C, np.zeros((st1.C.shape[0], 1))])
    dists = cdist(voxel_pc * voxel_size, st1.C * voxel_size)
    min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)

    min1v = np.min(dists, axis=1)
    min1f = min2[min1v < 3]  # todo tune the threshold?
    xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    matches = np.intersect1d(min1f, xx)

    missing1 = np.setdiff1d(np.arange(voxel_pc.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(voxel_pc)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)

    # valid_src_idxs = (idx_query != -1).nonzero()[0]
    # valid_tgts = idx_query[valid_src_idxs]
    matching = -np.ones(voxel_pc.shape[0], dtype=np.int)
    matching[MN[:, 0]] = MN[:, 1]
    return matching


def scene_flow_image_to_voxel_flow(st0, pixelloc_to_voxelidx, scene_flow):
    voxel_idxs = pixelloc_to_voxelidx.reshape(-1)[pixelloc_to_voxelidx.reshape(-1) != -1]
    valid_scene_flow = scene_flow.reshape(3, -1)[:, pixelloc_to_voxelidx.reshape(-1) != -1]
    # warpped_coords1 = coords0.reshape(-1, 3)[pixelloc_to_voxelidx.reshape(-1) != -1] + valid_scene_flow.T
    # warpped_st0C = st0.C[voxel_idxs] * voxel_size + valid_scene_flow.T
    voxel_flow = np.zeros((len(st0.C), 3))
    voxel_flow[voxel_idxs] = valid_scene_flow.T
    return voxel_flow


def scene_flow_to_voxel_flow(st0, mapidx, scene_flow):
    voxel_flow = np.zeros((len(st0.C), 3))
    voxel_flow[mapidx] = scene_flow
    return voxel_flow


def scene_flow_to_voxel_flow_gpu(st0, mapidx, scene_flow):
    voxel_flow = torch.zeros((len(st0.C), 3), dtype=torch.float, device='cuda')
    voxel_flow[mapidx] = scene_flow
    return voxel_flow


def scene_flow_image_to_voxel_matching(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx,
                                       coord_img0, coord_img1, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    try:
        return scene_flow_image_to_voxel_matching_gpu(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx,
                                                      coord_img0, coord_img1, dbg)
    except RuntimeError as e:
        return scene_flow_image_to_voxel_matching_np(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx, coord_img0,
                                                     coord_img1, dbg)


def scene_flow_image_to_voxel_matching_gpu(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx,
                                           coord_img0, coord_img1, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    voxel_flow = scene_flow_image_to_voxel_flow(st0, pixelloc_to_voxelidx, scene_flow)
    nonvalid_mask = np.all(voxel_flow == 0, axis=1)
    voxel_flow[nonvalid_mask, :] = 1000
    warpped_st1C = st0.C + voxel_flow / voxel_size
    dists = torch.cdist(warpped_st1C.float().cuda(), st1.C.float().cuda())
    min1 = torch.argmin(dists, dim=0)
    min2 = torch.argmin(dists, dim=1)

    min1v = torch.min(dists, dim=1).values
    min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    xx = torch.where(min2[min1] == torch.arange(min1.shape[0], device='cuda'))[0]

    min1f = min1f.cpu().numpy()
    xx = xx.cpu().numpy()
    min1 = min1.cpu().numpy()
    matches = np.intersect1d(min1f, xx)

    missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        with Vis3D(
                # xyz_pattern=('x', '-y', '-z'),
                # out_folder="dbg",
                sequence='scene_flow_to_voxel_matching',
                auto_increase=True
        ) as vis3d:
            # vis3d.set_scene_id(0)
            vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
            vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
            vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
            vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                                  MN, 0.1, name='matching')
            # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
            # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
            print()
    return MN, MN2, MN3


def scene_flow_image_to_voxel_matching_np(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx,
                                          coord_img0, coord_img1, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    voxel_flow = scene_flow_image_to_voxel_flow(st0, pixelloc_to_voxelidx, scene_flow)
    nonvalid_mask = np.all(voxel_flow == 0, axis=1)
    voxel_flow[nonvalid_mask, :] = 1000
    warpped_st1C = st0.C + voxel_flow / voxel_size
    # st1c = st1.C * voxel_size

    dists = cdist(warpped_st1C, st1.C)
    min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)

    min1v = np.min(dists, axis=1)
    min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    matches = np.intersect1d(min1f, xx)

    missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        with Vis3D(
                # xyz_pattern=('x', '-y', '-z'),
                # out_folder="dbg",
                sequence='scene_flow_to_voxel_matching',
                auto_increase=True
        ) as vis3d:
            # vis3d.set_scene_id(0)
            vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
            vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
            vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
            vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                                  MN, 0.1, name='matching')
            # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
            # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
            print()
    return MN, MN2, MN3


def barycoord_to_scene_flow(start_points, bc0, ptf0, vertices, faces):
    faces = faces[ptf0]
    verts = vertices[faces.reshape(-1)].reshape(-1, 3, 3)
    end_points = (bc0[:, :, None] * verts).sum(1)
    scene_flow = end_points - start_points
    # scene_flow = torch.from_numpy(scene_flow).float()
    return scene_flow


def scene_flow_to_voxel_matching(st0, st1, pt_tensor0, pt_tensor1, scene_flow, voxel_size, dbg=False):
    try:
        return scene_flow_to_voxel_matching_gpu(st0, st1, pt_tensor0, pt_tensor1, scene_flow, voxel_size, dbg)
    except RuntimeError:
        return scene_flow_to_voxel_matching_np(st0, st1, pt_tensor0, pt_tensor1, scene_flow, voxel_size, dbg)


def scene_flow_to_voxel_matching_gpu(st0, st1, pt_tensor0, pt_tensor1, scene_flow, voxel_size, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    st0 = st0.cuda()
    st1 = st1.cuda()
    pt_tensor0 = pt_tensor0.cuda()
    pt_tensor1 = pt_tensor1.cuda()
    scene_flow = scene_flow.cuda()
    mapidx = pt_tensor0.additional_features['idx_query'][1]
    voxel_flow = scene_flow_to_voxel_flow_gpu(st0, mapidx, scene_flow)
    nonvalid_mask = torch.all(voxel_flow == 0, dim=1)
    voxel_flow[nonvalid_mask, :] = 1000
    warpped_st1C = st0.C + voxel_flow / voxel_size
    # st1c = st1.C * voxel_size

    dists = torch.cdist(warpped_st1C, st1.C.float())
    min1 = torch.argmin(dists, dim=0)
    min2 = torch.argmin(dists, dim=1)

    min1v = torch.min(dists, dim=1).values
    min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    xx = torch.where(min2[min1] == torch.arange(min1.shape[0], device='cuda'))[0]
    min1f = min1f.cpu().numpy()
    xx = xx.cpu().numpy()
    matches = np.intersect1d(min1f, xx)
    min1 = min1.cpu().numpy()
    missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        vis3d = Vis3D(
            # xyz_pattern=('x', '-y', '-z'),
            # out_folder="dbg",
            sequence='scene_flow_to_voxel_matching',
            # auto_increase=True
        )
        vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
        vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
        vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
        vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                              MN, 0.1, name='matching')
        # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
        # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
        print()
    return MN, MN2, MN3


def scene_flow_to_voxel_matching_np(st0, st1, pt_tensor0, pt_tensor1, scene_flow, voxel_size, dbg=False):
    """

        :param st0:
        :param st1:
        :param scene_flow:
        :param voxel_size:
        :return:
        """
    mapidx = pt_tensor0.additional_features['idx_query'][1]
    voxel_flow = scene_flow_to_voxel_flow(st0, mapidx, scene_flow)
    nonvalid_mask = np.all(voxel_flow == 0, axis=1)
    voxel_flow[nonvalid_mask, :] = 1000
    warpped_st1C = st0.C + voxel_flow / voxel_size
    # st1c = st1.C * voxel_size

    dists = cdist(warpped_st1C, st1.C)
    min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)

    min1v = np.min(dists, axis=1)
    min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    matches = np.intersect1d(min1f, xx)

    missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        vis3d = Vis3D(
            # xyz_pattern=('x', '-y', '-z'),
            # out_folder="dbg",
            sequence='scene_flow_to_voxel_matching',
            # auto_increase=True
        )
        vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
        vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
        vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
        vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                              MN, 0.1, name='matching')
        # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
        # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
        print()
    return MN, MN2, MN3


def inv_scene_flow_to_inv_voxel_flow(st0, st1, pt_tensor0, pt_tensor1, inv_scene_flow, voxel_size, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    st0 = st0.cuda()
    st1 = st1.cuda()
    pt_tensor0 = pt_tensor0.cuda()
    pt_tensor1 = pt_tensor1.cuda()
    inv_scene_flow = inv_scene_flow.cuda()
    mapidx = pt_tensor1.additional_features['idx_query'][1]
    voxel_flow = scene_flow_to_voxel_flow_gpu(st1, mapidx, inv_scene_flow)
    nonvalid_mask = torch.all(voxel_flow == 0, dim=1)
    voxel_flow[nonvalid_mask, :] = 1000
    warpped_st1C = st0.C + voxel_flow / voxel_size
    # st1c = st1.C * voxel_size

    dists = torch.cdist(warpped_st1C, st1.C.float())
    min1 = torch.argmin(dists, dim=0)
    min2 = torch.argmin(dists, dim=1)

    min1v = torch.min(dists, dim=1).values
    min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    xx = torch.where(min2[min1] == torch.arange(min1.shape[0], device='cuda'))[0]
    min1f = min1f.cpu().numpy()
    xx = xx.cpu().numpy()
    matches = np.intersect1d(min1f, xx)
    min1 = min1.cpu().numpy()
    missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        vis3d = Vis3D(
            # xyz_pattern=('x', '-y', '-z'),
            # out_folder="dbg",
            sequence='scene_flow_to_voxel_matching',
            # auto_increase=True
        )
        vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
        vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
        vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
        vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                              MN, 0.1, name='matching')
        # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
        # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
        print()
    return MN, MN2, MN3


def scene_flow_to_voxel_matching_single_dir(st0, st1, scene_flow, voxel_size, pixelloc_to_voxelidx,
                                            coord_img0, coord_img1, dbg=False):
    """

    :param st0:
    :param st1:
    :param scene_flow:
    :param voxel_size:
    :return:
    """
    voxel_idxs = pixelloc_to_voxelidx.reshape(-1)[pixelloc_to_voxelidx.reshape(-1) != -1]
    valid_scene_flow = scene_flow.reshape(3, -1)[:, pixelloc_to_voxelidx.reshape(-1) != -1]
    # warpped_coords1 = coords0.reshape(-1, 3)[pixelloc_to_voxelidx.reshape(-1) != -1] + valid_scene_flow.T
    # warpped_st0C = st0.C[voxel_idxs] * voxel_size + valid_scene_flow.T
    voxel_flow = np.zeros((len(st0.C), 3))
    voxel_flow[voxel_idxs] = valid_scene_flow.T
    warpped_st1C = st0.C + voxel_flow / voxel_size
    # st1c = st1.C * voxel_size

    dists = cdist(warpped_st1C, st1.C)
    # min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)
    min_dists = np.min(dists, axis=1)
    keep = min_dists < 3 / voxel_size
    a = np.arange(st0.C.shape[0])
    a = a[keep]
    b = min2[keep]

    missing1 = a[~keep]
    missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), min2)

    # min1v = np.min(dists, axis=1)
    # min1f = min2[min1v < 3 / voxel_size]  # todo tune the threshold?
    # xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    # matches = np.intersect1d(min1f, xx)
    #
    # missing1 = np.setdiff1d(np.arange(warpped_st1C.shape[0]), min1[matches])
    # missing2 = np.setdiff1d(np.arange(st1.C.shape[0]), matches)

    MN = np.concatenate([a[np.newaxis, :], b[np.newaxis, :]]).T
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(st1.C)) * np.ones((1, len(missing1)), dtype=np.int64)]).T
    MN3 = np.concatenate(
        [(len(warpped_st1C)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]]).T
    all_matches = np.concatenate([MN, MN2, MN3], axis=0)
    if dbg:
        with Vis3D(
                # xyz_pattern=('x', '-y', '-z'),
                # out_folder="dbg",
                sequence='scene_flow_to_voxel_matching_single_dir',
                auto_increase=True
        ) as vis3d:
            # vis3d.set_scene_id(0)
            vis3d.add_point_cloud(st0.C * voxel_size, name='st0')
            vis3d.add_point_cloud(warpped_st1C * voxel_size, name='warped')
            vis3d.add_point_cloud(st1.C * voxel_size, name='st1')
            vis3d.add_3d_matching(st0.C * voxel_size, st1.C * voxel_size,
                                  MN, 0.1, name='matching')
            # vis3d.add_image(PIL.Image.fromarray((255 * img0).astype(np.uint8)))
            # vis3d.add_image(PIL.Image.fromarray((255 * img1).astype(np.uint8)))
            print()
    return MN, MN2, MN3


# def trilinear_sampler(input, coords, mode='trilinear', mask=False) -> torch.Tensor:
#     """
#     Wrapper for grid_sample, uses pixel coordinates
#     :param tensor: B,C,H,W,D
#     :param coords: B,M,3:(x,y,z)
#     :param mode:
#     :param mask:
#     :return:
#     """
#     H, W, D = input.shape[-3:]
#     xgrid, ygrid, zgrid = coords.split([1, 1, 1], dim=-1)
#     xgrid = 2 * xgrid / (W - 1) - 1
#     ygrid = 2 * ygrid / (H - 1) - 1
#     zgrid = 2 * zgrid / (D - 1) - 1
#
#     grid = torch.cat([xgrid, ygrid, zgrid], dim=-1)
#     F.interpolate(input,)
#     output = F.grid_sample(input, grid, mode=mode)
#
#     if mask:
#         mask = (xgrid > -1) & (ygrid > -1) & (zgrid > -1) & (xgrid < 1) & (ygrid < 1) & (zgrid < 1)
#         return output, mask.float()
#
#     return output
#
#

def _sample_at_integer_locs(input_feats, index_tensor):
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert index_tensor.ndimension() == 4, 'index_tensor should be of shape [B,H,W,3]'
    # first sample pixel locations using nearest neighbour interpolation
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = index_tensor.shape[1], index_tensor.shape[2]

    xy_grid = index_tensor[..., 0:2]
    xy_grid[..., 0] = xy_grid[..., 0] - ((width - 1.0) / 2.0)
    xy_grid[..., 0] = xy_grid[..., 0] / ((width - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] - ((height - 1.0) / 2.0)
    xy_grid[..., 1] = xy_grid[..., 1] / ((height - 1.0) / 2.0)
    xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)
    sampled_in_2d = F.grid_sample(input=input_feats.view(batch_size, num_chans * num_d, height, width),
                                  grid=xy_grid, mode='nearest', align_corners=False).view(batch_size, num_chans, num_d,
                                                                                          grid_height,
                                                                                          grid_width)
    z_grid = index_tensor[..., 2].view(batch_size, 1, 1, grid_height, grid_width)
    z_grid = z_grid.long().clamp(min=0, max=num_d - 1)
    z_grid = z_grid.expand(batch_size, num_chans, 1, grid_height, grid_width)
    sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
    return sampled_in_3d


def trilinear_interpolation(input_feats, sampling_grid):
    """
    interploate value in 3D volume
    :param input_feats: [B,C,D,H,W]
    :param sampling_grid: [B,H,W,3] unscaled coordinates
    :return:
    """
    assert input_feats.ndimension() == 5, 'input_feats should be of shape [B,F,D,H,W]'
    assert sampling_grid.ndimension() == 4, 'sampling_grid should be of shape [B,H,W,3]'
    batch_size, num_chans, num_d, height, width = input_feats.shape
    grid_height, grid_width = sampling_grid.shape[1], sampling_grid.shape[2]
    # make sure sampling grid lies between -1, 1
    sampling_grid[..., 0] = 2 * sampling_grid[..., 0] / (num_d - 1) - 1
    sampling_grid[..., 1] = 2 * sampling_grid[..., 1] / (height - 1) - 1
    sampling_grid[..., 2] = 2 * sampling_grid[..., 2] / (width - 1) - 1
    sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
    # map to 0,1
    sampling_grid = (sampling_grid + 1) / 2.0
    # Scale grid to floating point pixel locations
    scaling_factor = torch.FloatTensor([width - 1.0, height - 1.0, num_d - 1.0]).to(input_feats.device).view(1, 1,
                                                                                                             1, 3)
    sampling_grid = scaling_factor * sampling_grid
    # Now sampling grid is between [0, w-1; 0,h-1; 0,d-1]
    x, y, z = torch.split(sampling_grid, split_size_or_sections=1, dim=3)
    x_0, y_0, z_0 = torch.split(sampling_grid.floor(), split_size_or_sections=1, dim=3)
    x_1, y_1, z_1 = x_0 + 1.0, y_0 + 1.0, z_0 + 1.0
    u, v, w = x - x_0, y - y_0, z - z_0
    u, v, w = map(lambda x: x.view(batch_size, 1, grid_height, grid_width).expand(
        batch_size, num_chans, grid_height, grid_width), [u, v, w])
    c_000 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
    c_001 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
    c_010 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
    c_011 = _sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
    c_100 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
    c_101 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
    c_110 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
    c_111 = _sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))
    c_xyz = (1.0 - u) * (1.0 - v) * (1.0 - w) * c_000 + \
            (1.0 - u) * (1.0 - v) * w * c_001 + \
            (1.0 - u) * v * (1.0 - w) * c_010 + \
            (1.0 - u) * v * w * c_011 + \
            u * (1.0 - v) * (1.0 - w) * c_100 + \
            u * (1.0 - v) * w * c_101 + \
            u * v * (1.0 - w) * c_110 + \
            u * v * w * c_111
    return c_xyz


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, random_sample=False):
    from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import ball_query as ball_query_pointnet2
    batch_size = xyz.shape[0]
    if random_sample:
        torch.manual_seed(0)
        # assert xyz.shape[0] == 1  # todo: batch_size>1?
        if batch_size == 1:
            perm = torch.randperm(xyz.shape[1], device=xyz.device)
            xyz_perm = xyz[:, perm, :]
            idxs = ball_query_pointnet2(radius, nsample, xyz_perm, new_xyz)
            neg_mask = idxs < 0
            idxs_init = perm[idxs.reshape(-1).long()].reshape_as(idxs)
            idxs_init[neg_mask] = -1
            idxs = idxs_init
        else:
            batch_idxs = []
            for bi in range(batch_size):
                subxyz = xyz[bi]
                keep = subxyz[:, -1] != -1000
                subxyz = subxyz[keep]
                sub_newxyz = new_xyz[bi]
                keep = sub_newxyz[:, -1] != -1000
                sub_newxyz = sub_newxyz[keep]
                perm = torch.randperm(subxyz.shape[0], device=xyz.device)
                xyz_perm = subxyz[perm, :]
                idxs = ball_query_pointnet2(radius, nsample, xyz_perm[None], sub_newxyz[None])
                neg_mask = idxs < 0
                idxs_init = perm[idxs.reshape(-1).long()].reshape_as(idxs)
                idxs_init[neg_mask] = -1
                idxs = idxs_init
                batch_idxs.append(idxs[0])
            batch_idxs = padded_stack(batch_idxs)
            idxs = batch_idxs
    else:
        if batch_size == 1:
            idxs = ball_query_pointnet2(radius, nsample, xyz, new_xyz)
        else:
            raise NotImplementedError()
    return idxs


def cube_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, random_sample=False):
    from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import cube_query as cube_query_pointnet2
    batch_size = xyz.shape[0]
    if random_sample:
        torch.manual_seed(0)
        # assert xyz.shape[0] == 1  # todo: batch_size>1?
        if batch_size == 1:
            perm = torch.randperm(xyz.shape[1], device=xyz.device)
            xyz_perm = xyz[:, perm, :]
            idxs = cube_query_pointnet2(radius, nsample, xyz_perm, new_xyz)
            neg_mask = idxs < 0
            idxs_init = perm[idxs.reshape(-1).long()].reshape_as(idxs)
            idxs_init[neg_mask] = -1
            idxs = idxs_init
        else:
            batch_idxs = []
            for bi in range(batch_size):
                subxyz = xyz[bi]
                keep = subxyz[:, -1] != -1000
                subxyz = subxyz[keep]
                sub_newxyz = new_xyz[bi]
                keep = sub_newxyz[:, -1] != -1000
                sub_newxyz = sub_newxyz[keep]
                perm = torch.randperm(subxyz.shape[0], device=xyz.device)
                xyz_perm = subxyz[perm, :]
                idxs = cube_query_pointnet2(radius, nsample, xyz_perm[None], sub_newxyz[None])
                neg_mask = idxs < 0
                idxs_init = perm[idxs.reshape(-1).long()].reshape_as(idxs)
                idxs_init[neg_mask] = -1
                idxs = idxs_init
                batch_idxs.append(idxs[0])
            batch_idxs = padded_stack(batch_idxs)
            idxs = batch_idxs
    else:
        if batch_size == 1:
            idxs = cube_query_pointnet2(radius, nsample, xyz, new_xyz)
        else:
            raise NotImplementedError()
    return idxs


def transformation_matrix_to_sceneflow(pt0, pt1, matrix, dbg=False):
    try:
        return transformation_matrix_to_sceneflow_cuda(pt0, pt1, matrix, dbg)
    except RuntimeError:
        print(f'cuda oom. switch cpu,{pt0.shape}, {pt1.shape}')
        return transformation_matrix_to_sceneflow_np(pt0, pt1, matrix, dbg)


def transformation_matrix_to_sceneflow_cuda(pt0, pt1, matrix, dbg=False):
    vis3d = Vis3D(
        sequence="transformation_matrix_to_sceneflow",
        auto_increase=True,
        enable=dbg,
    )
    pt0 = torch.from_numpy(pt0).cuda().float()
    pt1 = torch.from_numpy(pt1).cuda().float()
    matrix = torch.from_numpy(matrix).cuda().float()
    warped_pts = Calibration.hom_to_cart(Calibration.cart_to_hom(pt0) @ matrix.T)
    vis3d.add_point_cloud(pt0, name='pt0')
    vis3d.add_point_cloud(pt1, name='pt1')
    vis3d.add_point_cloud(warped_pts, name='warped')
    dists = torch.cdist(warped_pts, pt1)
    min_dists = torch.min(dists, dim=1).values
    keep = min_dists < 0.01  # todo: need to tune threshold?
    vis3d.add_point_cloud(warped_pts[keep], name='warped_keep')
    sceneflow = torch.zeros_like(pt0)
    sceneflow[keep] = warped_pts[keep] - pt0[keep]
    sceneflow = sceneflow.cpu().numpy()
    return sceneflow


def transformation_matrix_to_sceneflow_np(pt0, pt1, matrix, dbg=False):
    # dbg = True
    vis3d = Vis3D(
        sequence="transformation_matrix_to_sceneflow",
        auto_increase=True,
        enable=dbg,
    )
    matrix = matrix
    warped_pts = Calibration.hom_to_cart(Calibration.cart_to_hom(pt0) @ matrix.T)
    vis3d.add_point_cloud(pt0, name='pt0')
    vis3d.add_point_cloud(pt1, name='pt1')
    vis3d.add_point_cloud(warped_pts, name='warped')
    dists = cdist(warped_pts, pt1)
    min_dists = np.min(dists, axis=1)
    keep = min_dists < 0.01  # todo: need to tune threshold?
    vis3d.add_point_cloud(warped_pts[keep], name='warped_keep')
    sceneflow = np.zeros_like(pt0)
    sceneflow[keep] = warped_pts[keep] - pt0[keep]
    return sceneflow


def create_center_radius(center=np.array([0, 0, 0]), dist=5., angle_z=30, nrad=180, start=0.):
    RTs = []
    center = np.array(center).reshape(3, 1)
    thetas = np.linspace(start, start + 2 * np.pi, nrad)
    angle_z = np.deg2rad(angle_z)
    radius = dist * np.cos(angle_z)
    height = dist * np.sin(angle_z)
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        center_ = np.array([radius * ct, radius * st, height]).reshape(3, 1)
        center_[0] += center[0, 0]
        center_[1] += center[1, 0]
        R = np.array([
            [-st, ct, 0],
            [0, 0, -1],
            [-ct, -st, 0]
        ])
        Rotx = cv2.Rodrigues(angle_z * np.array([1., 0., 0.]))[0]
        R = Rotx @ R
        T = - R @ center_
        center_ = - R.T @ T
        RT = np.hstack([R, T])
        RTs.append(RT)
    return np.stack(RTs)


def tsdf_fusion_get_vol_bnds(depths, obj_poses, cam_intr, voxel_length):
    vol_bnds = np.zeros((3, 2))
    # obj_poses = obj_poses @ np.linalg.inv(obj_poses[index])
    for depth_im, op in safe_zip(depths, obj_poses):
        cam_pose = np.linalg.inv(op)
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    vol_bnds[:, 0] = np.floor(vol_bnds[:, 0] / voxel_length) * voxel_length
    vol_bnds[:, 1] = np.ceil(vol_bnds[:, 1] / voxel_length) * voxel_length
    return vol_bnds


def tsdf_fusion_python_api(depths, obj_poses, cam_intr, voxel_length, colors=None, dbg=False):
    """

    :param depths: N,H,W
    :param obj_poses: N,4,4
    :param cam_intr: 3,3
    :param voxel_length: float
    :param colors: None or N,H,W,3 of uint8
    :param dbg:
    :return:
    """
    vis3d = Vis3D(
        # xyz_pattern=('x', 'y', 'z'),
        # out_folder="dbg",
        sequence="tsdf_fusion_python_api",
        auto_increase=True,
        enable=dbg
    )
    vol_bnds = tsdf_fusion_get_vol_bnds(depths, obj_poses, cam_intr, voxel_length)
    if colors is None:
        colors = np.zeros([len(depths), depths[0].shape[0], depths[0].shape[1], 3], dtype=np.uint8)
    volume = TSDFVolume(vol_bnds, voxel_size=voxel_length)
    for color, depth, op in tqdm.tqdm(safe_zip(colors, depths, obj_poses), total=len(obj_poses), leave=False):
        volume.integrate(color, depth, cam_intr, np.linalg.inv(op), obs_weight=1.)
        # if dbg:
        #     vis3d.add_mesh(volume.get_mesh(True))

    mesh_mc = volume.get_mesh(return_trimesh=True)
    vis3d.add_mesh(mesh_mc)
    bound_coord = (vol_bnds / voxel_length).round().astype(int)
    return mesh_mc, volume, bound_coord


def open3d_tsdf_fusion_api(depths, obj_poses, K, voxel_length, colors=None):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=3 * voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    if colors is None:
        colors = np.zeros([len(depths), depths[0].shape[0], depths[0].shape[1], 3], dtype=np.uint8)
    for i in tqdm.tqdm(range(len(colors))):
        H, W, _ = colors[0].shape
        pose = np.linalg.inv(obj_poses[i])
        rgb = o3d.geometry.Image(colors[i])
        depth_pred = o3d.geometry.Image(depths[i].astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                           vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)[:, :3])
    return mesh


def open3d_icp_api(pts0, pts1, thresh, init_Rt=np.eye(4)):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: mx3
    :param thresh: float
    :param init_Rt: 4x4
    :return:
    """
    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0.copy())
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1.copy())
    if version.parse(o3d.__version__) < version.parse('0.10.0'):
        result = o3d.registration.registration_icp(
            pcd0, pcd1, thresh, init_Rt)
    else:
        result = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, thresh, init_Rt)
    return result.transformation


def open3d_ransac_api(pts0, pts1, thresh):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: mx3
    :param thresh: float
    :param init_Rt: 4x4
    :return:
    """
    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    corres = np.arange(pts0.shape[0])[:, None].repeat(2, axis=1)
    corres = o3d.utility.Vector2iVector(corres)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd0, pcd1, corres, thresh)
    return result.transformation


def open3d_colored_icp_api(src_pc, tgt_pc, src_color, tgt_color, init_tsfm=np.eye(4)):
    if isinstance(src_pc, trimesh.Trimesh):
        src_pc = src_pc.vertices
    if isinstance(src_pc, torch.Tensor):
        src_pc = src_pc.cpu().numpy()
    if isinstance(tgt_pc, trimesh.Trimesh):
        tgt_pc = tgt_pc.vertices
    if isinstance(tgt_pc, torch.Tensor):
        tgt_pc = tgt_pc.cpu().numpy()
    # if normalize_scale:
    #     scaling = 1.0 / (src_pc.max(0) - src_pc.min(0)).max()
    #     src_pc = src_pc * scaling
    # scaling = 1.0 / (tgt_pc.max(0) - tgt_pc.min(0)).max()
    # tgt_pc = tgt_pc * scaling
    # if normalize_position:
    #     src_pc = src_pc - src_pc.min(0)
    #     tgt_pc = tgt_pc - tgt_pc.min(0)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_pc.copy())
    source.colors = o3d.utility.Vector3dVector(src_color.copy())

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tgt_pc.copy())
    target.colors = o3d.utility.Vector3dVector(tgt_color.copy())

    # voxel_radius = [0.04, 0.02, 0.01]
    # voxel_radius = [0.02, 0.01]
    voxel_radius = [0.01]
    # voxel_radius = [0.08, 0.04, 0.02]
    # voxel_radius = [0.16, 0.08, 0.04]
    # max_iter = [50, 30, 14]
    max_iter = [14]
    current_transformation = init_tsfm
    print("3. Colored point cloud registration")
    results_icp = []
    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        results_icp.append(result_icp)
        # print(result_icp)
    return results_icp


def cpa_pytorch3d_api(pts0, pts1):
    """
    R*pts0+t=pts1
    :param pts0: nx3
    :param pts1: nx3
    :return: 4x4
    """
    from pytorch3d.ops import corresponding_points_alignment
    pts0 = torch.from_numpy(pts0).float().cuda()
    pts1 = torch.from_numpy(pts1).float().cuda()
    cpa_res = corresponding_points_alignment(pts0[None], pts1[None])
    R = cpa_res.R[0].T
    t = cpa_res.T[0]
    Rt = torch.cat([R, t.reshape(3, 1)], dim=1)
    pose = matrix_3x4_to_4x4(Rt)
    return pose.cpu().numpy()


@dispatch(np.ndarray)
def interp_pose(poses):
    import pytorch3d.transforms
    """

    :param poses: np.ndarray N,4,4
    :return:
    """
    N = len(poses)
    nN = N * 2 - 1
    newposes = np.zeros([nN, 4, 4])
    newposes[::2, :, :] = poses
    a = poses[:-1]
    b = poses[1:]
    aa = pytorch3d.transforms.se3_log_map(torch.from_numpy(a.transpose(0, 2, 1)))
    bb = pytorch3d.transforms.se3_log_map(torch.from_numpy(b.transpose(0, 2, 1)))
    cc = (aa + bb) / 2
    c = pytorch3d.transforms.se3_exp_map(cc).numpy().transpose(0, 2, 1)
    newposes[1::2, :, :] = c
    return newposes


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
    R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
    R_new = R_b @ R_a
    t_new = (R_b @ t_a + t_b)[..., 0]
    pose_new = torch.cat([R_new, t_new[..., None]], dim=-1)
    pose_new = matrix_3x4_to_4x4(pose_new)
    return pose_new


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    if R1.ndim == 2 and R2.ndim == 2:
        R_diff = R1[:3, :3] @ R2[:3, :3].T
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
    else:
        R_diff = R1[..., :3, :3] @ R2.transpose(-2, -1)[..., :3, :3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def pose_distance(pred, gt, eps=1e-7, align=False):
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.empty([0]), torch.empty([0])
    if pred.ndim == 2 and gt.ndim == 2:
        pred = pred[None]
        gt = gt[None]
    if align:
        gt = gt @ gt[0].inverse()[None]
        pred = pred @ pred[0].inverse()[None]
    R_error = rotation_distance(pred, gt, eps)
    t_error = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
    return R_error, t_error


def project_to_img(pts_cam, K, shape):
    tmp = torch.zeros(shape)
    K = K.cpu()
    pts_img = rect_to_img(K[0, 0], K[1, 1], K[0, 2], K[1, 2], pts_cam.cpu())
    tmp[pts_img[:, 1].long(), pts_img[:, 0].long()] = 1
    return tmp


def chamfer_distance(pts0, pts1, color0=None, color1=None, use_gpu=True):
    if use_gpu:
        from disprcnn.utils.chamfer3D import dist_chamfer_3D
        chamLoss = dist_chamfer_3D.chamfer_3DDist()
        points1 = to_tensor(pts0).cuda().float()[None]
        points2 = to_tensor(pts1).cuda().float()[None]
        # points1 = torch.rand(32, 1000, 3).cuda()
        # points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
        dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
        loss = dist1.mean() + dist2.mean()
        if color0 is not None and color1 is not None:
            color0 = to_tensor(color0).cuda()
            color1 = to_tensor(color1).cuda()
            if color0.max() > 1:
                color0 = color0.float() / 255.0
            if color1.max() > 1:
                color1 = color1.float() / 255.0
            idx1 = idx1[0]
            idx2 = idx2[0]
            l1 = ((color0 - color1[idx1.long()]) ** 2).mean()
            l2 = ((color1 - color0[idx2.long()]) ** 2).mean()
            loss = loss + l1 + l2
        return loss.item()
    else:
        pts0 = to_tensor(pts0)
        pts1 = to_tensor(pts1)

        def square_distance(src, dst):
            return torch.sum((src[:, None, :] - dst[None, :, :]) ** 2, dim=-1)

        dist_src = torch.min(square_distance(pts0, pts1), dim=-1)
        dist_ref = torch.min(square_distance(pts1, pts0), dim=-1)
        chamfer_dist = torch.mean(dist_src[0]) + torch.mean(dist_ref[0])
        if color0 is not None or color1 is not None:
            raise NotImplementedError()
        return chamfer_dist.item()


def open3d_plane_segment_api(pts, distance_threshold, ransac_n=3, num_iterations=1000):
    pts = to_array(pts)
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd0.segment_plane(distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    return plane_model, inliers


def pytorch_nms(bboxes,
                scores,
                pre_max_size=None,
                post_max_size=None,
                iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    if len(bboxes) == 0:
        keep = torch.tensor([], dtype=torch.long).cuda()
    else:
        m = bboxes.min()
        bboxes = bboxes + m
        ret = nms(bboxes, scores, iou_threshold)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        # keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()
