import glob
import time

import numpy as np
import os
import pickle

import cv2
import os.path as osp
import torch
from PIL import Image
from disprcnn.structures.bounding_box_3d import Box3DList

from disprcnn.structures.calib import Calib

from disprcnn.structures.bounding_box import BoxList
from tqdm import tqdm


class RealTrackingStereoDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "car",
        'pedestrian',
        'dontcare'
    )

    def __init__(self, cfg, root, split, transforms=None, remove_ignore=True, ds_len=-1):
        """

        :param cfg:
        :param root: data/real/processed
        :param split:
        :param transforms:
        :param remove_ignore:
        :param ds_len:
        """
        self.root = root
        self.split = split
        self.gray = cfg.dataset.kitti_tracking_stereo.use_gray
        cls = RealTrackingStereoDataset.CLASSES
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        # make cache or read cached annotation
        # assert split == 'val'
        self.seqs = [0]

        self.pairs = self.make_pairs()

        # self.annotations = self.read_annotations()
        self.infos = self.read_info()
        if ds_len > 0:
            self.pairs = self.pairs[:ds_len]

        print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        seq, imgid = self.pairs[index]
        imgs = self.get_image(seq, imgid)
        height, width, _ = imgs['left'].shape

        left_targets = BoxList(torch.empty([0, 4]), (1, 1))
        left_targets.add_field("calib", Calib(self.get_calibration(seq, imgid), (width, height)))
        left_targets.add_field("box3d", Box3DList(torch.empty([0, 7]), "xyzhwl_ry"))
        # left_targets.add_field("lidar", )
        right_targets = BoxList(torch.empty([0, 4]), (1, 1))
        targets = {'left': left_targets, 'right': right_targets}
        assert self.transforms is not None

        tsfmed_left_img = self.transforms({'image': imgs['left']})['image']
        tsfmed_right_img = self.transforms({'image': imgs['right']})['image']
        dps = {
            'original_images': {'left': np.ascontiguousarray(imgs['left'][:, :, ::-1]),
                                'right': np.ascontiguousarray(imgs['right'][:, :, ::-1])},
            'images': {'left': torch.from_numpy(tsfmed_left_img).permute(2, 0, 1),
                       'right': torch.from_numpy(tsfmed_right_img).permute(2, 0, 1)},
            "targets": targets,
            'height': height,
            'width': width,
            'index': index,
            'seq': seq,
            'imgid': imgid
        }

        return dps

    def make_pairs(self):
        all_pairs = []
        # split = 'training' if not is_testing_split(self.split) else 'testing'
        for seq in self.seqs:
            nimgs = len(os.listdir(osp.join(self.root, 'image_02', f"{seq:04d}")))
            pairs = list(zip([seq] * nimgs, range(1573, nimgs)))  # todo
            all_pairs.extend(pairs)
        return all_pairs

    def get_image(self, seq, frameid):
        left_img = cv2.imread(osp.join(self.root, f'image_02/{seq:04d}/{frameid:06d}.png'))
        right_img = cv2.imread(osp.join(self.root, f'image_03/{seq:04d}/{frameid:06d}.png'))
        if self.gray:
            Lgray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            left_img[..., 0] = Lgray
            left_img[..., 1] = Lgray
            left_img[..., 2] = Lgray

            Rgray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            right_img[..., 0] = Rgray
            right_img[..., 1] = Rgray
            right_img[..., 2] = Rgray
        imgs = {'left': left_img, 'right': right_img}
        return imgs

    def __len__(self):
        return len(self.pairs)

    def get_img_info(self, seq, frameid):
        return self.infos[seq][frameid]

    def read_info(self):
        # split = 'training' if not is_testing_split(self.split) else 'testing'
        infopath = os.path.join(self.root, f'infos_{self.split}.pkl')
        if not os.path.exists(infopath):
            all_infos = {}
            for seq in self.seqs:
                infos = []
                for path in sorted(glob.glob(osp.join(self.root, f"image_02/{seq:04d}/*.png"))):
                    img = Image.open(path)
                    infos.append({"height": img.height, "width": img.width, 'size': img.size})
                all_infos[seq] = infos
            pickle.dump(all_infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                all_infos = pickle.load(f)
        return all_infos

    def get_calibration(self, seq, frameid):
        from dl_ext.vision_ext.datasets.kitti.structures import Calibration
        with open(osp.join(self.root, f"calib/{seq:04d}.txt")) as f:
            lines = f.readlines()
        calibs = {}
        for line in lines:
            k = line.split()[0]
            nums = list(map(float, line.split()[1:]))
            if len(nums) == 12:
                nums = torch.tensor(nums).reshape(3, 4).float()
            elif len(nums) == 9:
                nums = torch.tensor(nums).reshape(3, 3).float()
            else:
                raise RuntimeError()
            if k == 'R_rect':
                k = 'R0_rect'
            elif k == 'Tr_velo_cam':
                k = 'Tr_velo_to_cam'
            elif k == 'Tr_imu_velo':
                k = 'Tr_imu_to_velo'
            k = k.strip(":")
            calibs[k] = nums.numpy()
        info = self.get_img_info(seq, frameid)
        calib = Calibration(calibs, [info['width'], info['height']])
        return calib
