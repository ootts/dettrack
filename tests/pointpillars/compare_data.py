import numpy as np
import os.path as osp
import pickle


def main():
    ref_dir = 'data/kitti_second'
    my_dir = 'data/pointpillars/kitti_second'
    with open(osp.join(ref_dir, 'kitti_infos_train.pkl'), 'rb') as f:
        kitti_infos_train_ref = pickle.load(f)
    with open(osp.join(my_dir, 'kitti_infos_train.pkl'), 'rb') as f:
        kitti_infos_train_my = pickle.load(f)
    with open(osp.join(ref_dir, "kitti_dbinfos_train.pkl"), 'rb') as f:
        kitti_dbinfos_train_ref = pickle.load(f)
    with open(osp.join(my_dir, "kitti_dbinfos_train.pkl"), 'rb') as f:
        kitti_dbinfos_train_my = pickle.load(f)
    lidar_ref = np.fromfile(osp.join(ref_dir, "training/velodyne_reduced/000003.bin"), np.float32).reshape(-1, 4)
    lidar_my = np.fromfile(osp.join(my_dir, "training/velodyne_reduced/000003.bin"), np.float32).reshape(-1, 4)
    print()


if __name__ == '__main__':
    main()
