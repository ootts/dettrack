import numpy as np
import os

import torch
import os.path as osp

import tqdm
from dl_ext.vision_ext.datasets.kitti.io import load_velodyne

from disprcnn.modeling.models.psmnet.inference import DisparityMapProcessor
from disprcnn.utils.vis3d_ext import Vis3D


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data.build import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/idispnet/kittiobj/kittiobj_resizegray.yaml'
    cfg = setup(args, freeze=False)
    cfg.dbg = False

    ds = make_data_loader(cfg, False).dataset
    pred_path = osp.join(cfg.output_dir, 'inference', cfg.datasets.test, 'predictions.pth')
    preds = torch.load(pred_path, 'cpu')
    dmp = DisparityMapProcessor()
    assert len(preds) == len(ds)
    output_dir = 'data/disprcnn/pseudo_lidar/training/velodyne'
    os.makedirs(output_dir, exist_ok=True)
    vis3d = Vis3D(
        xyz_pattern=('x', '-y', '-z'),
        out_folder="dbg",
        sequence="convert_preds_to_pseudo_lidar",
        # auto_increase=,
        enable=cfg.dbg is True,
    )
    for i, (pred, dps) in enumerate(tqdm.tqdm(zip(preds, ds), total=len(preds))):
        vis3d.set_scene_id(i)
        imgid = pred['left'].extra_fields['imgid']
        disparity_map = dmp(pred['left'], pred['right'])
        calib = dps['targets']['left'].get_field('calib')
        pts_rect, _, _ = calib.disparity_map_to_rect(disparity_map.data.cpu())
        keep = (pts_rect[:, 0] > -20) & (pts_rect[:, 0] < 20) & \
               (pts_rect[:, 1] > -3) & (pts_rect[:, 1] < 3) \
               & (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
        pts_rect = pts_rect[keep]
        vis3d.add_point_cloud(pts_rect, name='pts_rect')
        pts_lidar = calib.rect_to_lidar(pts_rect)
        pts_lidar = torch.cat([pts_lidar, torch.full_like(pts_lidar[:, 0:1], 0.5)], dim=1)
        pts_lidar = pts_lidar.numpy().astype(np.float32)
        vis3d.add_point_cloud(pts_lidar[:, :3], name='pts_lidar')
        lidar = load_velodyne('data/kitti', 'training', imgid)
        vis3d.add_point_cloud(lidar[:, :3], name='lidar')
        vis3d.add_point_cloud(calib.lidar_to_rect(lidar[:, :3]), name='pts_rect_gt')
        with open(osp.join(output_dir, f"{imgid:06d}.bin"), 'w') as f:
            pts_lidar.tofile(f)


if __name__ == '__main__':
    main()
