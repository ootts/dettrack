import matplotlib.pyplot as plt
import tqdm
import glob
import os

import numpy as np
import os.path as osp
import imageio
import cv2
import yaml


# path = osp.join(raw_dir, "left/frame007850.png")
# img = imageio.imread(path)

def undistort(img_path, cam_path):
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
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramatrix)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    newcameramatrix[0, 2] -= x
    newcameramatrix[1, 2] -= y
    # plt.subplot(2, 1, 1)
    # plt.imshow(img)
    # plt.subplot(2, 1, 2)
    # plt.imshow(dst)
    # plt.show()
    # print()
    return dst, newcameramatrix


def main():
    raw_dir = "data/real/raw"
    car_dir = "data/real/car/0000"

    min_id = 7860

    os.makedirs(osp.join(car_dir, "image_02"), exist_ok=True)
    os.makedirs(osp.join(car_dir, "image_03"), exist_ok=True)
    os.makedirs(osp.join(car_dir, "calib"), exist_ok=True)
    new_imgs = {}
    for viewid, view in enumerate(['left', 'right']):
        img_paths = sorted(glob.glob(osp.join(raw_dir, f"{view}/*.png")))
        cam_path = osp.join(raw_dir, f"cam{viewid}_small.yaml")
        for img_path in img_paths:
            imgid = int(img_path.split("/")[-1].lstrip("frame").rstrip(".png"))
            if imgid >= min_id:
                new_img, K = undistort(img_path, cam_path)
                if imgid not in new_imgs:
                    new_imgs[imgid] = {}
                new_imgs[imgid][view] = new_img
    minh = min(new_imgs[min_id]['left'].shape[0], new_imgs[min_id]['right'].shape[0])
    minw = min(new_imgs[min_id]['left'].shape[1], new_imgs[min_id]['right'].shape[1])
    plt.subplot(1, 2, 1)
    plt.imshow(new_imgs[min_id]['left'][:minh, :minw])
    plt.subplot(1, 2, 2)
    plt.imshow(new_imgs[min_id]['right'][:minh, :minw])
    plt.show()
    # newimgs[viewid]
    # print()
    # img_paths = sorted(glob.glob(osp.join(raw_dir, "right/*.png")))
    # cam_path = osp.join(raw_dir, "cam0_small.yaml")
    # for img_path in img_paths:
    #     if int(img_path.split("/")[-1].lstrip("frame").rstrip(".png")) >= min_id:
    #         new_img = undistort(img_path, left_cam_path)
    #         print()


if __name__ == '__main__':
    main()
