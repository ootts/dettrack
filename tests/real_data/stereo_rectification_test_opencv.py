from math import floor

import matplotlib.pyplot as plt
import cv2
import os.path as osp

import imageio
import numpy as np
import yaml


def load_cam_attrs(cam_path):
    cam_attrs = yaml.load(open(cam_path))
    camera_matrix = np.array(
        [[cam_attrs['projection_parameters']['fx'] * 2, 0, cam_attrs['projection_parameters']['cx'] * 2],
         [0, cam_attrs['projection_parameters']['fy'] * 2, cam_attrs['projection_parameters']['cy'] * 2],
         [0, 0, 1]])
    dist_coeffs = np.array([cam_attrs['distortion_parameters']['k1'],
                            cam_attrs['distortion_parameters']['k2'],
                            cam_attrs['distortion_parameters']['p1'],
                            cam_attrs['distortion_parameters']['p2'],
                            0])
    return camera_matrix, dist_coeffs


def main():
    raw_dir = "data/real/raw"
    left_path = "data/real/raw/left/frame007865.png"
    right_path = "data/real/raw/right/frame007865.png"
    left_img = cv2.resize(imageio.imread(left_path), (640, 360))
    right_img = cv2.resize(imageio.imread(right_path), (640, 360))
    H, W = left_img.shape
    left_K, left_dist = load_cam_attrs(osp.join(raw_dir, "cam0_small.yaml"))
    right_K, right_dist = load_cam_attrs(osp.join(raw_dir, "cam1_small.yaml"))

    R2 = np.array([9.9999e-01, 1.3479e-04, 4.7469e-03, 1.2004e-01,
                   -1.2457e-04, 1, -2.1531e-03, -7.4797e-04,
                   -4.7472e-03, 2.1524e-03, 9.9999e-01, 5.4342e-04,
                   0, 0, 0, 1]).reshape(4, 4)
    # R2 = np.linalg.inv(R2)
    fx2, fy2, cx2, cy2 = right_K[0, 0], right_K[1, 1], right_K[0, 2], right_K[1, 2]
    k21, k22, p21, p22, k23 = right_dist.tolist()
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_K, left_dist, right_K, right_dist,
                                                                      (W, H), R2[:3, :3], R2[:3, 3],
                                                                      flags=cv2.CALIB_ZERO_DISPARITY)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, (W, H), cv2.CV_32FC1)

    left_new_img = cv2.remap(left_img, map1x, map1y,
                             interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    right_new_img = cv2.remap(right_img, map2x, map2y,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(left_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(right_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.subplot(1, 2, 1)
    plt.imshow(left_new_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(right_new_img, cmap='gray', vmin=0, vmax=255)
    plt.show()

    tmp = left_new_img / 255.0 / 2 + right_new_img / 255.0 / 2
    plt.imshow(tmp, cmap='gray', vmin=0, vmax=1.0)
    plt.show()


if __name__ == '__main__':
    main()
