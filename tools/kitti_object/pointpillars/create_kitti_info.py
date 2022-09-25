import os
import os.path as osp
import pathlib
import pickle

import numpy as np
import tqdm

from disprcnn.utils.ppp_utils import box_np_ops
from disprcnn.utils.ppp_utils import kitti_common as kitti


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = [int(l) for l in f.read().splitlines()]
    return lines


def _calculate_num_points_in_gt(data_path, infos, relative_path, remove_outside=True, num_features=4):
    for info in tqdm.tqdm(infos):
        if relative_path and not osp.exists(info["velodyne_path"]):
            v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
        else:
            v_path = info["velodyne_path"]
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = info['calib/R0_rect']
        Trv2c = info['calib/Tr_velo_to_cam']
        P2 = info['calib/P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, info["img_shape"])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(data_path,
                           save_dir,
                           velodyne_template,
                           create_trainval=False,
                           relative_path=True):
    split_set_dir = 'data/kitti/object/split_set/'
    train_img_ids = _read_imageset_file(osp.join(split_set_dir, "train.txt"))
    val_img_ids = _read_imageset_file(osp.join(split_set_dir, "val.txt"))
    trainval_img_ids = _read_imageset_file(osp.join(split_set_dir, "trainval.txt"))
    test_img_ids = _read_imageset_file(osp.join(split_set_dir, "test.txt"))

    filename = osp.join(save_dir, 'kitti_infos_train.pkl')
    if not osp.exists(filename):
        kitti_infos_train = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=train_img_ids,
            relative_path=relative_path,
            velodyne_template=velodyne_template)

        _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
    else:
        with open(filename, 'rb') as f:
            kitti_infos_train = pickle.load(open(filename, "rb"))
    filename = osp.join(save_dir, 'kitti_infos_val.pkl')
    if not osp.exists(filename):
        kitti_infos_val = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=val_img_ids,
            relative_path=relative_path,
            velodyne_template=velodyne_template)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)

        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
    else:
        with open(filename, 'rb') as f:
            kitti_infos_val = pickle.load(open(filename, "rb"))
    """
    if create_trainval:
        kitti_infos_trainval = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=trainval_img_ids,
            relative_path=relative_path)
        filename = save_path / 'kitti_infos_trainval.pkl'
        print(f"Kitti info trainval file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_trainval, f)
    """
    filename = osp.join(save_dir, 'kitti_infos_trainval.pkl')

    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    filename = osp.join(save_dir, 'kitti_infos_test.pkl')
    if not osp.exists(filename):
        kitti_infos_test = kitti.get_kitti_image_info(
            data_path,
            training=False,
            label_info=False,
            velodyne=True,
            calib=True,
            image_ids=test_img_ids,
            relative_path=relative_path,
            velodyne_template=velodyne_template
        )
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    parser = default_argument_parser()
    args = parser.parse_args()
    # args.config_file = 'configs/pointpillars/disprcnn_112/preprocess.yaml'
    cfg = setup(args)

    save_dir = cfg.model.pointpillars.preprocess.save_dir
    assert save_dir != ""
    velodyne_template = cfg.model.pointpillars.preprocess.velodyne_template
    os.makedirs(save_dir, exist_ok=True)
    create_kitti_info_file('data/kitti/object', save_dir, velodyne_template)


if __name__ == '__main__':
    main()
