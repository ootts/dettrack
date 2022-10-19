import torch
import imageio
import numpy as np
from disprcnn.utils import utils_3d

from disprcnn.utils.psmnet_vanilla_api import psmnet_vanilla_api
from disprcnn.utils.vis3d_ext import Vis3D


def main():
    vis3d = Vis3D(
        xyz_pattern=('x', '-y', '-z'),
        out_folder="dbg",
        sequence="psmnet_vanilla_spi",
        # auto_increase=,
        # enable=,
    )
    predictions = torch.load(
        'models/drcnn/kitti_tracking/pointpillars_112_600x300_real_demo/inference/realtrackingstereo_demo/predictions.pth',
        'cpu')
    masks = [pred['left'].PixelWise_map['masks'].get_mask_tensor() for pred in predictions]
    id_min = 1573

    P2 = np.array([[944.3854374830397, 0.0, 632.9827575683594, -113.36724556782147],
                   [0.0, 944.3854374830397, 346.6133842468262, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    K = P2[:3, :3]
    fuxb = -P2[0, 3]
    pretrained_model = "../PSMNet/pretrained_model_KITTI2015.tar"
    for i in range(6):
        imgid = id_min + i
        left_path = f"data/real/processed/image_02/0000/{imgid:06d}.png"
        right_path = f"data/real/processed/image_03/0000/{imgid:06d}.png"

        disp = psmnet_vanilla_api(left_path, right_path, pretrained_model)
        depth = fuxb / disp
        mask = masks[i]
        depth = depth * mask.cpu().numpy()
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)

        left_rgb = imageio.imread(left_path)
        if len(left_rgb.shape) == 2:
            left_rgb = np.repeat(left_rgb[:, :, None], 3, axis=-1)
        left_rgb = left_rgb.reshape(-1, 3)
        keep = pts_rect[:, 2] > 0
        vis3d.add_point_cloud(pts_rect[keep], left_rgb[keep])
        vis3d.increase_scene_id()


if __name__ == '__main__':
    main()
