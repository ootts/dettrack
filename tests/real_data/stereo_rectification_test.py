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
    left_path = "data/real/raw/left/frame007860.png"
    right_path = "data/real/raw/right/frame007860.png"
    left_img = cv2.resize(imageio.imread(left_path), (640, 360))
    right_img = cv2.resize(imageio.imread(right_path), (640, 360))
    H, W = left_img.shape
    left_K, left_dist = load_cam_attrs(osp.join(raw_dir, "cam0_small.yaml"))
    right_K, right_dist = load_cam_attrs(osp.join(raw_dir, "cam1_small.yaml"))
    K = (left_K + right_K) / 2
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fx1, fy1, cx1, cy1 = left_K[0, 0], left_K[1, 1], left_K[0, 2], left_K[1, 2]
    k11, k12, p11, p12, k13 = left_dist.tolist()
    f1 = 1
    gamma = 0
    left_new_img = np.zeros([H, W])
    for x in range(W):
        for y in range(H):
            yy = (y - cy) / fy
            xx = (x - cx - gamma * yy) / fx
            xx = xx / f1
            yy = yy / f1
            r = xx * xx + yy * yy
            xxx = xx * (1 + k11 * r + k12 * r ** 2 + k13 * r ** 3) + 2 * p11 * xx * yy + p12 * (r + 2 * xx ** 2)
            yyy = yy * (1 + k11 * r + k12 * r ** 2 + k13 * r ** 3) + 2 * p12 * xx * yy + p11 * (r + 2 * yy ** 2)
            xxxx = xxx * fx1 + cx1
            yyyy = yyy * fy1 + cy1
            if 0 < xxxx < W and 0 < yyyy < H:
                h = yyyy
                w = xxxx
                tmp1 = (np.floor(w + 1) - w) * (np.floor(h + 1) - h) * left_img[
                    int(np.floor(h - 1)), int(np.floor(w - 1))]
                tmp2 = (np.floor(w + 1) - w) * (h - np.floor(h)) * left_img[int(np.floor(h)), int(np.floor(w - 1))]
                tmp3 = (w - np.floor(w)) * (np.floor(h + 1) - h) * left_img[int(np.floor(h - 1)), int(np.floor(w))]
                tmp4 = (w - np.floor(w)) * (h - np.floor(h)) * left_img[int(np.floor(h)), int(np.floor(w))]
                left_new_img[y, x] = tmp1 + tmp2 + tmp3 + tmp4
    # plt.subplot(1, 2, 1)
    # plt.imshow(left_img, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(1, 2, 2)
    # plt.imshow(left_new_img, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    f2 = 1
    gama = 0
    R2 = np.array([9.9999e-01, 1.3479e-04, 4.7469e-03, 1.2004e-01,
                   -1.2457e-04, 1, -2.1531e-03, -7.4797e-04,
                   -4.7472e-03, 2.1524e-03, 9.9999e-01, 5.4342e-04,
                   0, 0, 0, 1]).reshape(4, 4)[:3, :3]
    fx2, fy2, cx2, cy2 = right_K[0, 0], right_K[1, 1], right_K[0, 2], right_K[1, 2]
    k21, k22, p21, p22, k23 = right_dist.tolist()
    right_new_img = np.zeros([H, W])
    for x in range(W):
        for y in range(H):
            yy = (y - cy) / fy
            xx = (x - cx - gama * yy) / fx
            pos = np.array([xx, yy, f2])
            pos = np.linalg.inv(R2) @ pos
            xx = pos[0] / pos[2]
            yy = pos[1] / pos[2]
            r = xx ** 2 + yy ** 2
            xxx = xx * (1 + k21 * r + k22 * r ** 2 + k23 * r ** 3) + 2 * p21 * xx * yy + p22 * (r + 2 * xx ** 2)
            yyy = yy * (1 + k21 * r + k22 * r ** 2 + k23 * r ** 3) + 2 * p22 * xx * yy + p21 * (r + 2 * yy ** 2)
            xxxx = xxx * fx2 + cx2
            yyyy = yyy * fy2 + cy2
            if xxxx > 0 and xxxx <= W - 1 and yyyy > 0 and yyyy <= H - 1:
                h = yyyy
                w = xxxx
                tmp1 = (floor(w + 1) - w) * (floor(h + 1) - h) * right_img[floor(h - 1), floor(w)]
                tmp2 = (floor(w + 1) - w) * (h - floor(h)) * right_img[floor(h), floor(w - 1)]
                tmp3 = (w - floor(w)) * (floor(h + 1) - h) * right_img[floor(h - 1), floor(w)]
                tmp4 = (w - floor(w)) * (h - floor(h)) * right_img[floor(h), floor(w)]
                right_new_img[y, x] = tmp1 + tmp2 + tmp3 + tmp4
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
