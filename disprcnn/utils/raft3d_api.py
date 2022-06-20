import imageio
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image
from dl_ext.timer import EvalTime
from easydict import EasyDict
from tqdm import tqdm

from disprcnn.modeling.models.raft3d.raft3d import RAFT3D
import disprcnn.modeling.models.raft3d.projective_ops as pops
from disprcnn.utils.flow_utils import read_gen
from disprcnn.utils.utils_3d import depth_to_rect
from disprcnn.utils.vis3d_ext import Vis3D


class RAFT3DAPI:
    model = None

    @classmethod
    def getinstance(cls, args=None):
        """
        :param args:
        parser.add_argument('--model', help="restore checkpoint", default='models/raft/raft-things.pth')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--split', default='train', type=str)
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        :return:
        """
        if RAFT3DAPI.model is None:
            RAFT3DAPI.model = setup_model(args)
        return RAFT3DAPI.model


def setup_model(args):
    model = RAFT3D(args)
    ckpt = torch.load(args.model)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    model.cuda()
    model.eval()
    return model


def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[-2:]  # 375 1242
    pad_ht = ht % 8  # 1
    pad_wd = wd % 8  # 6

    img = F.pad(img, [0, pad_wd, 0, pad_ht], mode='replicate')
    return img, {'pad_ht': pad_ht, 'pad_wd': pad_wd}


def prepare_images_and_depths(image1, image2, depth1, depth2, normalize=True, DEPTH_SCALE=1.0):
    """ padding, normalization, and scaling """

    image1, pd = pad8(image1)
    image2, _ = pad8(image2)
    depth1, _ = pad8(depth1)
    depth2, _ = pad8(depth2)

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    if normalize:
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)

    return image1, image2, depth1, depth2, pd


def normalize_image(image):
    image = image[:, [2, 1, 0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image / 255.0).sub_(mean[:, None, None]).div_(std[:, None, None])


@torch.no_grad()
def raft3d_api(img1_path, img2_path, depth1, depth2, intrinsics, DEPTH_SCALE=1.0, args=None):
    """

    :param img1_path:
    :param img2_path:
    :param depth1: h,w
    :param depth2: h,w
    :param intrinsics: fx,fy,cx,cy list of float
    :param args:
    :return:
    """
    if args is None:
        args = EasyDict({'model': "models/raft3d/raft3d.pth",
                         "network": "raft3d.raft3d"})
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    depth1init = torch.tensor(depth1).float().cuda().unsqueeze(0).float()
    depth2init = torch.tensor(depth2).float().cuda().unsqueeze(0).float()
    image1init = torch.from_numpy(img1).permute(2, 0, 1).float().cuda().unsqueeze(0)
    image2init = torch.from_numpy(img2).permute(2, 0, 1).float().cuda().unsqueeze(0)
    intrinsics = torch.tensor(intrinsics).cuda().unsqueeze(0).float()

    image1, image2, depth1, depth2, pd = prepare_images_and_depths(image1init, image2init, depth1init, depth2init,
                                                                   DEPTH_SCALE=DEPTH_SCALE)

    model = RAFT3DAPI.getinstance(args)
    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics)
    if pd['pad_ht'] != 0:
        flow3d = flow3d[:, :-pd['pad_ht'], :, :]
    if pd['pad_wd'] != 0:
        flow3d = flow3d[:, :, :-pd['pad_wd'], :]
    return flow3d


def main():
    fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
    img1_path = 'data/raft3d_assets/image1.png'
    img2_path = 'data/raft3d_assets/image2.png'
    disp1 = read_gen('data/raft3d_assets/disp1.pfm')
    disp2 = read_gen('data/raft3d_assets/disp2.pfm')
    depth1 = fx / disp1
    depth2 = fx / disp2
    flow3d = raft3d_api(img1_path, img2_path, depth1, depth2, intrinsics=[fx, fy, cx, cy], DEPTH_SCALE=0.2)
    img1 = imageio.imread(img1_path)
    img2 = imageio.imread(img2_path)
    pts1 = depth_to_rect(fx, fy, cx, cy, depth1)
    pts2 = depth_to_rect(fx, fy, cx, cy, depth2)
    vis3d = Vis3D(
        xyz_pattern=('x', '-y', '-z'),
        out_folder="dbg",
        sequence="raft3d_demo",
        # auto_increase=,
        # enable=,
    )
    vis3d.add_point_cloud(pts1, colors=img1.reshape(-1, 3), name='pts1', sample=0.05)
    vis3d.add_point_cloud(pts2, colors=img2.reshape(-1, 3), name='pts2', sample=0.05)
    vis3d.add_flow_3d(pts1, flow3d, sample=0.01, name='flow')


if __name__ == '__main__':
    main()
#
