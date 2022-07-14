import os
import os.path as osp
import copy
import pathlib
import pickle

import fire
import numpy as np
import tqdm

from disprcnn.utils.ppp_utils import box_np_ops
from skimage import io as imgio

from disprcnn.utils.ppp_utils import kitti_common as kitti


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False):
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in tqdm.tqdm(kitti_infos):
        v_path = info['velodyne_path']
        if not osp.exists(v_path):
            v_path = osp.join(data_path, v_path)
        save_filename = osp.join(save_path, v_path.split("/")[-1])
        if osp.exists(save_filename):
            continue
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, info["img_shape"])

        # if save_path is None:
        #     save_filename = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
        #     save_filename = str(v_path) + '_reduced'
        # if back:
        #     save_filename += "_back"
        # else:
        if back:
            save_filename += "_back"
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


def create_reduced_point_cloud(data_path,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    if train_info_path is None:
        train_info_path = pathlib.Path(data_path) / 'kitti_infos_train.pkl'
    if val_info_path is None:
        val_info_path = pathlib.Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = pathlib.Path(data_path) / 'kitti_infos_test.pkl'

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, test_info_path, save_path, back=True)


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/pointpillars/lidar/preprocess.yaml'
    cfg = setup(args)

    save_dir = cfg.model.pointpillars.preprocess.save_dir
    velodyne_reduced_dir = cfg.model.pointpillars.preprocess.velodyne_reduced_dir
    assert velodyne_reduced_dir != ""
    os.makedirs(velodyne_reduced_dir, exist_ok=True)
    create_reduced_point_cloud(save_dir, save_path=velodyne_reduced_dir)


if __name__ == '__main__':
    main()
