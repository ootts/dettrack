import os
import random

import numpy as np
import os.path as osp
import pickle
from warnings import warn

import cv2
import torch
import torch.utils.data
import zarr
from PIL import Image
from dl_ext.primitive import safe_zip

from disprcnn.data.datasets.coco import COCOAnnotationTransform
from dl_ext.vision_ext.datasets.kitti.io import *
from tqdm import tqdm

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask


class RealDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, root, split, transforms=None, remove_ignore=True, ds_len=-1):
        # todo: fix shape_prior_base.
        """
        :param root: 'data/real/processed/0000/
        :param split: ['train','val']
        :param transforms:
        :param filter_empty:
        :param offline_2d_predictions_path:
        """
        self.root = root
        self.split = split
        self.cfg = cfg.dataset.real
        cls = self.cfg.classes
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        # make cache or read cached annotation
        # self.annotations = self.read_annotations()
        self.infos = self.read_info()
        nimgs = len(os.listdir(osp.join(self.root, "image_02/0000")))
        id_min, id_max = self.cfg.id_min, self.cfg.id_max
        if id_max < 0: id_max = nimgs
        self.ids = list(map(lambda x: f"{x:06d}", range(id_min, id_max)))
        # 0000: 1573,-1 for car, 801/835,847 for pedestrian
        if ds_len > 0:
            self.ids = self.ids[:ds_len]
        print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        img = self.get_image(index)
        imgid = int(self.ids[index])
        # print("imgid=0!!!")
        # imgid = 0
        height, width, _ = img.shape
        num_crowds = 0

        targets = self.get_ground_truth(index)
        img = self.transforms({'image': img,
                               'masks': np.zeros((1, height, width), dtype=np.float),
                               'boxes': np.array([[0.0, 0.0, 1.0, 1.0]]),
                               'labels': {'num_crowds': 0, 'labels': np.array([0])}})['image']
        target = np.array([[0.0, 0, 1, 1, 1]])
        masks = np.zeros((1, height, width), dtype=np.float)

        targets = BoxList(target[:, :4], (1, 1))
        targets.add_field("labels", torch.from_numpy(target[:, 4]))
        targets.add_field("masks", torch.from_numpy(masks).float())

        dps = {
            "image": torch.from_numpy(img).permute(2, 0, 1),
            "target": targets,
            'height': height,
            'width': width,
            'num_crowds': num_crowds,
            'imgid': imgid,
            'index': index
        }
        return dps

    def get_image(self, index):
        img_id = self.ids[index]
        left_img = cv2.imread(osp.join(self.root, "image_02/0000", img_id + '.png'))
        if self.cfg.use_gray:
            Lgray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            left_img[..., 0] = Lgray
            left_img[..., 1] = Lgray
            left_img[..., 2] = Lgray
        return left_img

    def get_ground_truth(self, index):
        img_id = self.ids[index]
        fakebox = torch.tensor([[0, 0, 0, 0]])
        info = self.get_img_info(index)
        height, width = info['height'], info['width']
        # left target
        left_target = BoxList(fakebox, (width, height), mode="xyxy")
        left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
        left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
        left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
        left_target.add_field('masks', self.get_mask(index))
        # left_target.add_field('kins_masks', self.get_kins_mask(index))
        # left_target.add_map('disparity', self.get_disparity(index))
        left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
        # right target
        right_target = BoxList(fakebox, (width, height), mode="xyxy")
        target = {'left': left_target, 'right': right_target}
        return target

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        return self.infos[int(img_id)]

    # def map_class_id_to_class_name(self, class_id):
    #     return KITTIObjectDataset.CLASSES[class_id]

    def read_info(self):
        # split = 'training' if self.split != 'test' else 'testing'
        # split = 'training' if not is_testing_split(self.split) else 'testing'
        infopath = os.path.join(self.root, f'infos.pkl')
        if not os.path.exists(infopath):
            infos = []
            # total = 7481 if self.split != 'test' else 7518
            # total = 7481 if not is_testing_split(self.split) else 7518
            total = len(os.listdir(osp.join(self.root, "image_02/0000")))
            for i in tqdm(range(total)):
                img = Image.open(osp.join(self.root, f"image_02/0000/{i:06d}.png"))
                infos.append({"height": img.height, "width": img.width, 'size': img.size})
            pickle.dump(infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                infos = pickle.load(f)
        return infos

    def get_mask(self, index):
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']

        mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        return mask

    def get_calibration(self, index):
        from dl_ext.vision_ext.datasets.kitti.structures import Calibration
        with open(osp.join(self.root, f"calib/0000.txt")) as f:
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
        W, H = 1280, 720
        calib = Calibration(calibs, [W, H])
        return calib
