""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
from PIL import Image
import cv2
import os
import torch

class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        # modified to due with calib format of kitti tracking
        if 'Tr_velo_to_cam' in calibs.keys():
            self.V2C = calibs['Tr_velo_to_cam']
        elif 'Tr_velo_cam' in calibs.keys():
            self.V2C = calibs['Tr_velo_cam']
        else:
            raise ValueError("Required key for Tr_velo_to_cam not "
                             "found in calib. Get keys:{}".format(calibs.keys()))
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)

        # Rigid transform from imu coord to Velodyne coord
        # modified to due with calib format of kitti tracking
        if 'Tr_imu_to_velo' in calibs.keys():
            self.I2V = calibs['Tr_imu_to_velo']
        elif 'Tr_imu_velo' in calibs.keys():
            self.I2V = calibs['Tr_imu_velo']
        else:
            raise ValueError("Required key for Tr_imu_to_velo not "
                             "found in calib. Get keys:{}".format(calibs.keys()))
        self.I2V = np.reshape(self.I2V, [3, 4])
        self.V2I = inverse_rigid_trans(self.I2V)

        # Rotation from reference camera coord to rect camera coord
        # modified to due with calib format of kitti tracking
        if 'R0_rect' in calibs.keys():
            self.R0 = calibs['R0_rect']
        elif 'R_rect' in calibs.keys():
            self.R0 = calibs['R_rect']
        else:
            raise ValueError("Required key for R_rect not "
                             "found in calib. Get keys:{}".format(calibs.keys()))

        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                # This part due with the special format of calib files in KITTI-tracking
                if ':' not in line:
                    keywords = ['R_rect', 'Tr_velo_cam', 'Tr_imu_velo']
                    for keyword in keywords:
                        if keyword in line:
                            key = keyword
                            value = line.split(keyword)[1]
                else:
                    key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        imu2velo = self.read_calib_file(os.path.join(calib_root_dir, 'calib_imu_to_velo.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        Tr_imu_to_velo = np.zeros((3, 4))
        Tr_imu_to_velo[0:3, 0:3] = np.reshape(imu2velo['R'], [3, 3])
        Tr_imu_to_velo[:, 3] = imu2velo['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['Tr_imu_to_velo'] = np.reshape(Tr_imu_to_velo, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        if isinstance(pts_3d, np.ndarray):
            n = pts_3d.shape[0]
            pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
            return pts_3d_hom
        else:
            n = pts_3d.shape[0]
            pts_3d_hom = torch.cat((pts_3d, torch.ones((n, 1), dtype=torch.float, device=pts_3d.device)), dim=1)
            return pts_3d_hom
    # ===========================
    # ------- 3d to 3d ----------
    # ===========================

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        if isinstance(pts_3d_velo, np.ndarray):
            return np.dot(pts_3d_velo, np.transpose(self.V2C))
        else:
            return torch.matmul(pts_3d_velo, torch.t(torch.from_numpy(self.V2C).float().to(pts_3d_velo.device)))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        if isinstance(pts_3d_ref, np.ndarray):
            return np.dot(pts_3d_ref, np.transpose(self.C2V))
        else:
            C2V = torch.from_numpy(self.C2V).float().to(pts_3d_ref.device)
            return torch.matmul(pts_3d_ref, torch.t(C2V))

    def project_imu_to_velo(self, pts_3d_imu):
        pts_3d_imu = self.cart2hom(pts_3d_imu)  # nx4
        if isinstance(pts_3d_imu, np.ndarray):
            return np.dot(pts_3d_imu, np.transpose(self.I2V))
        else:
            return pts_3d_imu @ torch.from_numpy(self.I2V).to(pts_3d_imu.device).float().T

    def project_velo_to_imu(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        if isinstance(pts_3d_velo, np.ndarray):
            return np.dot(pts_3d_velo, np.transpose(self.V2I))
        else:
            return pts_3d_velo @ torch.from_numpy(self.V2I).to(pts_3d_velo.device).float().T

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        if isinstance(pts_3d_rect, np.ndarray):
            return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
        else:
            r0_inv = torch.inverse(torch.from_numpy(self.R0).float())
            return torch.t(torch.matmul(r0_inv.to(pts_3d_rect.device), torch.t(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        if isinstance(pts_3d_ref, np.ndarray):
            return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
        else:
            r0 = torch.from_numpy(self.R0).float().to(pts_3d_ref.device)
            return torch.t(torch.matmul(r0, torch.t(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def project_rect_to_imu(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_velo = self.project_rect_to_velo(pts_3d_rect)
        return self.project_velo_to_imu(pts_3d_velo)

    def project_imu_to_rect(self, pts_3d_imu):
        pts_3d_velo = self.project_imu_to_velo(pts_3d_imu)
        return self.project_velo_to_rect(pts_3d_velo)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        if isinstance(pts_3d_rect, np.ndarray):
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
            pts_2d[:, 0] /= pts_3d_rect[:, 2]
            pts_2d[:, 1] /= pts_3d_rect[:, 2]
            return pts_2d
        else:
            p = torch.from_numpy(self.P).float().to(pts_3d_rect.device)
            pts_2d = torch.matmul(pts_3d_rect, torch.t(p))  # nx3
            pts_2d[:, 0] /= pts_3d_rect[:, 2]
            pts_2d[:, 1] /= pts_3d_rect[:, 2]
            return pts_2d

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        if isinstance(uv_depth, np.ndarray):
            pts_3d_rect = np.zeros((n, 3))
        else:
            pts_3d_rect = torch.zeros((n, 3), device=uv_depth.device).float()
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def transpose(self, method):
        return self

    def resize(self, size, *args, **kwargs):
        return self

    def to(self, device):
        return self

    def __getitem__(self, ind):
        return self


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    return Image.open(img_filename).convert("RGB")


def load_depth(img_filename):
    depth = np.array(Image.open(img_filename)) * 0.001
    depth = Image.fromarray(depth)
    return depth


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0];
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1];
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2];
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.CV_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.CV_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.CV_AA)
    return image


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo