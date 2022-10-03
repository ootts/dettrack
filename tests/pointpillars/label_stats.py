import numpy as np
import tqdm
import os.path as osp
from dl_ext.vision_ext.datasets.kitti.io import load_label_2

kitti_root = osp.expanduser("~/Datasets/kitti")
sizes = []
for i in tqdm.trange(7481):
    labels = load_label_2(kitti_root, 'training', i, ['Pedestrian'])
    for label in labels:
        sizes.append([label.w, label.l, label.h])
sizes = np.array(sizes)
print()
# [1.62858987 3.88395449 1.52608343]
# [0.66018944 0.84228438 1.76070649]
