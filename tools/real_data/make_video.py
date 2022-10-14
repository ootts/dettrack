import tqdm
import glob
import os

import numpy as np
import os.path as osp
import imageio
import cv2

imgs = []
data_dir = "data/real/left"
paths = sorted(glob.glob(osp.join(data_dir, "*.png")))
for i, path in enumerate(tqdm.tqdm(paths)):
    img = imageio.imread(path)
    cv2.putText(img, f'{i}:{path.split("/")[-1]}',
                (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    imgs.append(img)
imageio.mimsave(osp.join(data_dir, f'../left.mp4'), imgs)
