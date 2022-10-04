import torch

from disprcnn.utils.vis3d_ext import Vis3D

vis3d = Vis3D(
    xyz_pattern=('x', 'y', 'z'),
    out_folder="dbg",
    sequence="compare_drcnn_and_ppp",
    # auto_increase=,
    # enable=,
)
# ref = torch.load('tmp/points_ref.pth')
# src = torch.load('tmp/points.pth')
# vis3d.add_point_cloud(ref[:, :3], name='ref')
# vis3d.add_point_cloud(src[:, :3], name='src')

ref = torch.load('tmp/example_ref.pth')
src = torch.load('tmp/example_src.pth')
print()
