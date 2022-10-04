import torch

from disprcnn.structures.calib import Calib
from disprcnn.utils.vis3d_ext import Vis3D

vis3d = Vis3D(
    xyz_pattern=('x', 'y', 'z'),
    out_folder="dbg",
    sequence="compare_drcnn_and_ppp2",
    # auto_increase=,
    # enable=,
)
calib_ref, disp_ref = torch.load('tmp/dbg_ref.pth')
calib_src, disp_src = torch.load('tmp/dbg_src.pth')
print()
pts_rect, _, _ = calib_ref.disparity_map_to_rect(disp_ref.data.cpu())
keep = (pts_rect[:, 0] > -20) & (pts_rect[:, 0] < 20) & \
       (pts_rect[:, 1] > -3) & (pts_rect[:, 1] < 3) \
       & (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
pts_rect = pts_rect[keep]
# src
calib_src = Calib(calib_src, (calib_src.width, calib_src.height))
pts_rect_src, _, _ = calib_src.disparity_map_to_rect(disp_src.data)
keep_src = (pts_rect_src[:, 0] > -20) & (pts_rect_src[:, 0] < 20) & \
           (pts_rect_src[:, 1] > -3) & (pts_rect_src[:, 1] < 3) \
           & (pts_rect_src[:, 2] > 0) & (pts_rect_src[:, 2] < 80)
# keep = (pts_rect[:, 2] > 0) & (pts_rect[:, 2] < 80)
pts_rect_src = pts_rect_src[keep_src]
print()
i, j = 153, 1045
