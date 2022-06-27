import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from dl_ext.vision_ext import imagenet_normalize
from torch.utils.data import Dataset

from disprcnn.structures.segmentation_mask import SegmentationMask


class KITTIRoiDatasetRA(Dataset):
    def __init__(self, cfg, split, transforms=None, ds_len=-1):
        """
        :param split:
        :param resolution: width, height
        :param maxdisp:
        :param mindisp:
        """
        # assert split in ['train', 'val']
        self.split = split
        self.cfg = cfg.dataset.kittiroi
        self.root = self.cfg.root
        self.maxdisp = self.cfg.maxdisp
        self.mindisp = self.cfg.mindisp

        self.leftimgdir = os.path.join(self.root, self.split, 'image/left')
        self.rightimgdir = os.path.join(self.root, self.split, 'image/right')
        self.maskdir = os.path.join(self.root, self.split, 'mask')
        self.labeldir = os.path.join(self.root, self.split, 'label')
        self.disparitydir = os.path.join(self.root, self.split, 'disparity')
        # ts = [imagenet_normalize]
        self.transform = imagenet_normalize
        if ds_len < 0:
            self.length = len(os.listdir(self.leftimgdir))
        else:
            self.length = ds_len
        # self.sample_disp = sample_disp
        # self.perturb_disp = perturb_disp
        # self.sample = sample_disp < 1 or perturb_disp > 0
        print('using dataset of length', self.length)

    def __getitem__(self, index):
        images = self.get_image(index)
        targets = self.get_target(index)
        inputs = {**images, **targets}
        return inputs, targets

    def get_image(self, index):
        leftimg = self.get_left_img(index)
        rightimg = self.get_right_img(index)
        # transforms
        if self.transform is not None:
            leftimg = self.transform(leftimg)
            rightimg = self.transform(rightimg)
        return {'left': leftimg, 'right': rightimg}

    def get_target(self, index):
        disparity = self.get_disparity(index)
        mask = self.get_mask(index).get_mask_tensor()
        # if self.sample_disp < 1.0 and mask.sum() > 10:
        #     torch.random.manual_seed(index)
        #     mask1d = mask.reshape(-1)
        #     nz = mask1d.nonzero()[:, 0]
        #     cis_to_set0 = np.random.choice(nz.cpu().numpy(), int(nz.shape[0] * (1 - self.sample_disp)), replace=False)
        #     mask1d[cis_to_set0] = 0
        #     mask = mask1d.reshape_as(mask)
        # if self.perturb_disp > 0.0:
        #     torch.random.manual_seed(index)
        #     flag = 1 if torch.randn(1) > 0.5 else -1
        #     disparity = disparity * (1 + flag * self.perturb_disp)
        mask = mask & (disparity < self.maxdisp).byte() & (disparity > self.mindisp).byte()
        # if mask.sum() == 0:
        #     print('mask = 0')
        # x1_minus_x1p = self.get_x1_minux_x1p(index)
        label = self.get_label(index)
        targets = {**label, 'mask': mask, 'disparity': disparity.data}
        return targets

    def get_left_img(self, index):
        leftimg = zarr.load(osp.join(self.leftimgdir, str(index) + '.zarr'))
        leftimg = torch.from_numpy(leftimg)
        return leftimg

    def get_right_img(self, index):
        rightimg = zarr.load(osp.join(self.rightimgdir, str(index) + '.zarr'))
        rightimg = torch.from_numpy(rightimg)
        return rightimg

    def get_disparity(self, index):
        disparity = torch.from_numpy(zarr.load(osp.join(self.disparitydir, str(index) + '.zarr')))
        return disparity

    def get_mask(self, index):
        mask: SegmentationMask = self.get_label(index)['mask']
        return mask

    def get_label(self, index):
        return pickle.load(open(os.path.join(self.labeldir, str(index) + '.pkl'), 'rb'))

    def get_x1_minux_x1p(self, index):
        return self.get_label(index)['x1-x1p']

    # def get_box3d(self, index):
    #     return self.get_label(index)['box3d']

    def __len__(self):
        return self.length

    # def _check_resolution(self, r):
    #     if isinstance(r, int):
    #         r = (r, r)
    #     if isinstance(r, (tuple, list)):
    #         assert len(r) == 2
    #         assert isinstance(r[0], int) and isinstance(r[1], int)
    #     return r


# def check_coarse_depth_quality(fuxb, x1, x2, x1p, x2p, depth, mask):
#     # coarse_depth = fuxb / ((x1 + x2) / 2 - (x1p + x2p) / 2)
#     coarse_depth = fuxb / (x1 - x1p + 1e-6)
#     mask = mask & (depth < coarse_depth + 12).byte() & (depth > coarse_depth - 12).byte()
#     return mask


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/idispnet/kitti.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg, is_train=False).dataset
    print()


if __name__ == '__main__':
    main()
