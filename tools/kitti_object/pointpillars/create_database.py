import os
import os.path as osp
import pathlib
import pickle

import numpy as np
import tqdm

from disprcnn.utils.ppp_utils import box_np_ops
from disprcnn.utils.ppp_utils import kitti_common as kitti


def create_groundtruth_database(data_path,
                                # info_path=None,
                                used_classes=None,
                                # database_save_path=None,
                                # db_info_save_path=None,
                                relative_path=True,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None):
    root_path = data_path
    # if info_path is None:
    info_path = osp.join(root_path, 'kitti_infos_train.pkl')
    # if database_save_path is None:
    database_save_path = osp.join(root_path, 'gt_database')
    # else:
    #     database_save_path = pathlib.Path(database_save_path)
    # if db_info_save_path is None:
    db_info_save_path = osp.join(root_path, "kitti_dbinfos_train.pkl")
    os.makedirs(database_save_path, exist_ok=True)
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    all_db_infos = {}
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = []
    group_counter = 0
    for info in tqdm.tqdm(kitti_infos):
        velodyne_path = info['velodyne_path']
        if relative_path and not osp.exists(info['velodyne_path']):
            velodyne_path = osp.join(root_path, velodyne_path)
        num_features = 4
        if 'pointcloud_num_features' in info:
            num_features = info['pointcloud_num_features']
        points = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        image_idx = info["image_idx"]
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        if not lidar_only:
            points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2, info["img_shape"])

        annos = info["annos"]
        names = annos["name"]
        bboxes = annos["bbox"]
        difficulty = annos["difficulty"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)
        rbbox_cam = kitti.anno_to_rbboxes(annos)[:num_obj]
        rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
        if bev_only:  # set z and h to limits
            assert coors_range is not None
            rbbox_lidar[:, 2] = coors_range[2]
            rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]

        group_dict = {}
        group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(bboxes.shape[0], dtype=np.int64)
        point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{gt_idxes[i]}.bin"
            filepath = osp.join(database_save_path, filename)
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= rbbox_lidar[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
            if names[i] in used_classes:
                if relative_path:
                    db_path = osp.join(database_save_path, filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": gt_idxes[i],
                    "box3d_lidar": rbbox_lidar[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }

                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                all_db_infos[names[i]].append(db_info)
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    parser = default_argument_parser()
    args = parser.parse_args()
    # args.config_file = 'configs/pointpillars/disprcnn_112/preprocess.yaml'
    cfg = setup(args)

    save_dir = cfg.model.pointpillars.preprocess.save_dir
    create_groundtruth_database(save_dir)


if __name__ == '__main__':
    main()
