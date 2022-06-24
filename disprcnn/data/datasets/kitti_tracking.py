import glob

import numpy as np
import os
import pickle

import cv2
import os.path as osp
import torch
from PIL import Image
from disprcnn.structures.calib import Calib

from disprcnn.structures.bounding_box import BoxList
from tqdm import tqdm


class KITTITrackingDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "car",
        'dontcare'
    )
    NUM_TRAINING = 20
    NUM_TRAIN = 9
    NUM_VAL = 11
    NUM_TESTING = -1  # todo: not used

    def __init__(self, cfg, root, split, transforms=None, remove_ignore=True, ds_len=-1):
        # todo: fix shape_prior_base.
        """
        :param root: '.../kitti/
        :param split: ['train','val']
        :param transforms:
        """
        self.root = root
        self.split = split
        cls = KITTITrackingDataset.CLASSES
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        # make cache or read cached annotation
        if split == 'train':
            self.seqs = [0, 2, 3, 4, 5, 7, 9, 11, 17]
        elif split == 'val':
            self.seqs = [1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]
        else:
            raise NotImplementedError()

        self.pairs = self.make_pairs()

        self.annotations = self.read_annotations()
        self.infos = self.read_info()
        if ds_len > 0:
            self.pairs = self.pairs[:ds_len]
        if self.split == 'train':
            self.pairs = self.filter_empty()

        print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        if self.split == 'train':
            seq, imgid0, imgid1 = self.pairs[index]
            img0 = self.get_image(seq, imgid0)
            img1 = self.get_image(seq, imgid1)
            height, width, _ = img0.shape

            targets0 = self.get_ground_truth(seq, imgid0)
            targets1 = self.get_ground_truth(seq, imgid1)
            targets0 = targets0[targets0.get_field('labels') == 1]  # remove not cars
            targets1 = targets1[targets1.get_field('labels') == 1]  # remove not cars

            assert self.transforms is not None

            tsfmed_img0 = self.transforms({'image': img0})['image']
            tsfmed_img1 = self.transforms({'image': img1})['image']

            dps = {
                'image0': torch.from_numpy(tsfmed_img0).permute(2, 0, 1),
                'image1': torch.from_numpy(tsfmed_img1).permute(2, 0, 1),
                "target0": targets0,
                "target1": targets1,
                'height': height,
                'width': width,
                'index': index
            }
        elif self.split == 'val':
            seq, imgid = self.pairs[index]
            img = self.get_image(seq, imgid)
            height, width, _ = img.shape

            targets = self.get_ground_truth(seq, imgid)
            targets = targets[targets.get_field('labels') == 1]  # remove not cars

            assert self.transforms is not None

            tsfmed_img = self.transforms({'image': img})['image']

            dps = {
                'image': torch.from_numpy(tsfmed_img).permute(2, 0, 1),
                "target": targets,
                'height': height,
                'width': width,
                'index': index,
                'seq': seq,
                'imgid': imgid
            }
        else:
            raise NotImplementedError()

        return dps

    def filter_empty(self):
        non_empty_pairs = []
        for pair in self.pairs:
            seq, f0, f1 = pair
            labels = self.annotations[seq]['labels']
            keys = labels.keys()
            if f0 in keys and f1 in keys and labels[f0].sum() > 0 and labels[f1].sum() > 0:
                non_empty_pairs.append([seq, f0, f1])
        return non_empty_pairs

    def make_pairs(self):
        all_pairs = []
        split = 'training' if not is_testing_split(self.split) else 'testing'
        for seq in self.seqs:
            nimgs = len(os.listdir(osp.join(self.root, 'tracking', split, 'image_02', f"{seq:04d}")))
            if self.split == 'train':
                pairs = list(zip([seq] * (nimgs - 1), range(0, nimgs - 1), range(1, nimgs)))
            elif self.split == 'val':
                pairs = list(zip([seq] * nimgs, range(0, nimgs)))
            else:
                raise NotImplementedError()
            all_pairs.extend(pairs)
        return all_pairs

    def get_image(self, seq, frameid):
        split = 'training' if not is_testing_split(self.split) else 'testing'
        left_img = cv2.imread(osp.join(self.root, 'tracking', split, f'image_02/{seq:04d}/{frameid:06d}.png'))
        return left_img

    def get_ground_truth(self, seq, frameid):
        if not is_testing_split(self.split):
            labels = self.annotations[seq]['labels'][frameid]
            boxes = self.annotations[seq]['boxes'][frameid]
            trackids = self.annotations[seq]['trackids'][frameid]
            info = self.get_img_info(seq, frameid)
            height, width = info['height'], info['width']
            # left target
            target = BoxList(boxes, (width, height), mode="xyxy")
            target.add_field("labels", labels)
            target.add_field("trackids", trackids)
            target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(target), 1))
            target.add_field('calib', Calib(self.get_calibration(seq, frameid), (width, height)))
            # left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            # left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            target = target.clip_to_image(remove_empty=True)
            return target
        else:
            # todo: not used
            fakebox = torch.tensor([[0, 0, 0, 0]])
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            target = BoxList(fakebox, (width, height), mode="xyxy")
            target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(target), 1))
            target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            target.add_field('index', torch.full((len(target), 1), index, dtype=torch.long))
            target.add_field('masks', self.get_mask(index))
            # left_target.add_field('kins_masks', self.get_kins_mask(index))
            target.add_map('disparity', self.get_disparity(index))
            target.add_field('imgid', torch.full((len(target), 1), int(img_id), dtype=torch.long))
            # right target
            right_target = BoxList(fakebox, (width, height), mode="xyxy")
            target = {'left': target, 'right': right_target}
            return target

    def __len__(self):
        return len(self.pairs)

    def get_img_info(self, seq, frameid):
        return self.infos[seq][frameid]

    def read_annotations(self):
        if is_testing_split(self.split):
            return {'left': [], 'right': []}
        annodir = os.path.join(self.root, f"tracking/training/label_02")
        anno_cache_path = os.path.join(annodir, f'annotations_{self.split}.pkl')
        if os.path.exists(anno_cache_path):
            with open(anno_cache_path, 'rb') as f:
                annotations = pickle.load(f)
        else:
            print('generating', anno_cache_path)
            annotations = {}
            for seq in tqdm(self.seqs):
                with open(osp.join(self.root, f"tracking/training/label_02/{seq:04d}.txt")) as f:
                    lines = f.read().splitlines()
                bbox_per_seq = {}
                labels_per_seq = {}
                trackids_per_seq = {}
                for line in lines:
                    frame, trackid, cls_str, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = line.split()
                    if cls_str != 'DontCare':
                        frame = int(frame)
                        trackid = int(trackid)
                        cls_str = cls_str.lower().strip()
                        cls_str = 'car' if cls_str in ['car', 'van'] else '__background__'
                        cls = self.class_to_ind[cls_str]
                        if frame not in bbox_per_seq.keys():
                            bbox_per_seq[frame] = []
                            labels_per_seq[frame] = []
                            trackids_per_seq[frame] = []
                        bbox_per_seq[frame].append([float(x1), float(y1), float(x2), float(y2)])
                        labels_per_seq[frame].append(cls)
                        trackids_per_seq[frame].append(trackid)
                # assert list(bbox_per_seq.keys()) == list(range(len(bbox_per_seq)))
                bbox_per_seq = {k: torch.tensor(v).float() for k, v in bbox_per_seq.items()}
                labels_per_seq = {k: torch.tensor(v).long() for k, v in labels_per_seq.items()}
                trackids_per_seq = {k: torch.tensor(v).long() for k, v in trackids_per_seq.items()}
                with open(osp.join(self.root, f"tracking/training/calib/{seq:04d}.txt")) as f:
                    line = f.read().splitlines()[2]
                    P2 = torch.tensor(list(map(float, line.split()[1:]))).reshape(3, 4).float()
                annotations[seq] = ({'labels': labels_per_seq,
                                     'boxes': bbox_per_seq,
                                     'trackids': trackids_per_seq,
                                     'P2': P2
                                     })
            pickle.dump(annotations, open(anno_cache_path, 'wb'))
        return annotations

    def read_info(self):
        split = 'training' if not is_testing_split(self.split) else 'testing'
        infopath = os.path.join(self.root, f'tracking/{split}/infos_{self.split}.pkl')
        if not os.path.exists(infopath):
            all_infos = {}
            for seq in self.seqs:
                infos = []
                for path in sorted(glob.glob(osp.join(self.root, "tracking", split, f"image_02/{seq:04d}/*.png"))):
                    img = Image.open(path)
                    infos.append({"height": img.height, "width": img.width, 'size': img.size})
                all_infos[seq] = infos
            pickle.dump(all_infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                all_infos = pickle.load(f)
        return all_infos

    def get_calibration(self, seq, frameid):
        split = 'training' if not is_testing_split(self.split) else 'testing'
        from dl_ext.vision_ext.datasets.kitti.structures import Calibration
        with open(osp.join(self.root, "tracking", split, f"calib/{seq:04d}.txt")) as f:
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
            calibs[k] = nums
        info = self.get_img_info(seq, frameid)
        calib = Calibration(calibs, [info['width'], info['height']])
        return calib

    def remove_ignore_cars(self, l, r):
        if len(l) == 0 and len(r) == 0:
            return l, r

        heights = l.heights / l.height * l.get_field('image_size')[0, 1]
        # print(heights)
        truncations = l.get_field('truncation').tolist()
        occlusions = l.get_field('occlusion').tolist()
        keep = []
        levels = []
        for i, (height, truncation, occlusion) in enumerate(zip(heights, truncations, occlusions)):
            # print(height,truncation,occlusion)
            if height >= 40 and truncation <= 0.15 and occlusion <= 0:
                keep.append(i)
                levels.append(1)
            elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
                keep.append(i)
                levels.append(2)
            elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
                keep.append(i)
                levels.append(3)
        l = l[keep]
        r = r[keep]
        l.add_field('levels', torch.tensor(levels))
        return l, r

    def is_testing_split(self):
        return is_testing_split(self.split)


def is_testing_split(split):
    return split in ['test', 'testmini', 'test1', 'test2']


def main():
    ds = KITTITrackingDataset(None, 'data/kitti', 'val', None, )
    d = ds[0]


if __name__ == '__main__':
    main()
