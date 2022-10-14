from multiprocessing import Pool

import tqdm
import glob
import os
import os.path as osp

import cv2
import imageio
import numpy as np
import yaml

from disprcnn.utils.plt_utils import stereo_images_grid


def load_cam_attrs(cam_path):
    resize_factor = 2.0
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


def stereo_rectify(raw_dir):
    # left_path = osp.join(raw_dir, f"left/frame}.png")
    left_path = sorted(glob.glob(osp.join(raw_dir, 'left/*.png')))[0]

    left = imageio.imread(left_path)
    H, W = left.shape
    left_K, left_dist = load_cam_attrs(osp.join(raw_dir, "cam0_small.yaml"))
    right_K, right_dist = load_cam_attrs(osp.join(raw_dir, "cam1_small.yaml"))

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

    return P1, P2, map1x, map1y, map2x, map2y


def process_img(data):
    left_path, right_path, map1x, map1y, map2x, map2y, left_out_path, right_out_path = data
    left = imageio.imread(left_path)
    right = imageio.imread(right_path)
    left_remap = cv2.remap(left, map1x, map1y,
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    right_remap = cv2.remap(right, map2x, map2y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    # stereo_images_grid(left, right, title='before')
    # stereo_images_grid(left_remap, right_remap, title='after')
    imageio.imwrite(left_out_path, left_remap)
    imageio.imwrite(right_out_path, right_remap)


def main():
    raw_dir = "data/real/raw"
    processed_dir = 'data/real/processed'
    seq_id = "0000"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(osp.join(processed_dir, "calib"), exist_ok=True)
    os.makedirs(osp.join(processed_dir, "image_02", seq_id), exist_ok=True)
    os.makedirs(osp.join(processed_dir, "image_03", seq_id), exist_ok=True)

    frameids = [f.split("/")[-1].rstrip(".png") for f in sorted(glob.glob(osp.join(raw_dir, "left/*png")))]

    P1, P2, map1x, map1y, map2x, map2y = stereo_rectify(raw_dir)

    with open(osp.join(processed_dir, f"calib/{seq_id}.txt"), "w") as f:
        s = "P0: " + " ".join(map(str, P1.reshape(-1).tolist())) + "\n"
        s += "P1: " + " ".join(map(str, P1.reshape(-1).tolist())) + "\n"
        s += "P2: " + " ".join(map(str, P1.reshape(-1).tolist())) + "\n"
        s += "P3: " + " ".join(map(str, P2.reshape(-1).tolist())) + "\n"
        s += "R_rect 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01\nTr_velo_cam 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01\nTr_imu_velo 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01\n"
        f.write(s)
    imgid = 0
    datas = []
    for frameid in tqdm.tqdm(frameids):
        left_path = osp.join(raw_dir, "left", frameid + ".png")
        right_path = osp.join(raw_dir, "right", frameid + ".png")
        left_out_path = osp.join(processed_dir, f"image_02/{seq_id}/{imgid:06d}.png")
        right_out_path = osp.join(processed_dir, f"image_03/{seq_id}/{imgid:06d}.png")
        data = [left_path, right_path, map1x, map1y, map2x, map2y, left_out_path, right_out_path]
        datas.append(data)
        # stereo_images_grid(left, right, title='before')
        # stereo_images_grid(left_remap, right_remap, title='after')
        imgid += 1

    with Pool() as p:
        results = list(tqdm.tqdm(p.imap(process_img, datas), total=len(datas)))


if __name__ == '__main__':
    main()
