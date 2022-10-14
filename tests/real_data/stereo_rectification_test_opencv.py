import os.path as osp

import cv2
import imageio
import numpy as np
import yaml

from disprcnn.utils.plt_utils import stereo_images_grid


def load_cam_attrs(cam_path, resize_factor=2.0):
    cam_attrs = yaml.load(open(cam_path))
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


def main():
    raw_dir = "data/real/raw"
    imgid = '007865'
    left_path = f"data/real/raw/left/frame{imgid}.png"
    right_path = f"data/real/raw/right/frame{imgid}.png"

    resize_factor = 1.0
    left = imageio.imread(left_path)
    right = imageio.imread(right_path)
    size = int(left.shape[1] * resize_factor), int(left.shape[0] * resize_factor)
    left = cv2.resize(left, size)
    right = cv2.resize(right, size)
    H, W = left.shape
    left_K, left_dist = load_cam_attrs(osp.join(raw_dir, "cam0_small.yaml"), resize_factor * 2)
    right_K, right_dist = load_cam_attrs(osp.join(raw_dir, "cam1_small.yaml"), resize_factor * 2)

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

    left_remap = cv2.remap(left, map1x, map1y,
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    right_remap = cv2.remap(right, map2x, map2y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    stereo_images_grid(left, right, title='before')
    stereo_images_grid(left_remap, right_remap, title='after')


if __name__ == '__main__':
    main()
