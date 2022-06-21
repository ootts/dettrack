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
from disprcnn.data.datasets.coco import COCOAnnotationTransform
from dl_ext.vision_ext.datasets.kitti.io import *
from tqdm import tqdm

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask


# from disprcnn.utils.stereo_utils import align_left_right_targets

# from disprcnn.utils.timer import Timer

#
# kitti_timer = Timer()
# img_timer = Timer()
# target_timer = Timer()
# post_process = Timer()
# transform_timer = Timer()
# PRINTTIME = False


# align = Timer()
# get_pred = Timer()


class KITTIKinsDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "car",
        'dontcare'
    )
    NUM_TRAINING = 7481
    NUM_TRAIN = 3712
    NUM_VAL = 3769
    NUM_TESTING = 7518

    def __init__(self, cfg, root, split, transforms=None, remove_ignore=True, ds_len=-1):
        # todo: fix shape_prior_base.
        """
        :param root: '.../kitti/
        :param split: ['train','val']
        :param transforms:
        :param filter_empty:
        :param offline_2d_predictions_path:
        """
        self.root = root
        self.split = split
        cls = KITTIKinsDataset.CLASSES
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        # make cache or read cached annotation
        self.annotations = self.read_annotations()
        self.infos = self.read_info()
        self._imgsetpath = os.path.join(self.root, "object/split_set/%s_set.txt")

        with open(self._imgsetpath % self.split) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        if hasattr(self, 'o2dpreds'):
            assert len(self.ids) == len(self.o2dpreds['left'])
        self.truncations_list, self.occlusions_list = self.get_truncations_occluded_list()

        print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        img = self.get_image(index)
        targets = self.get_ground_truth(index)
        if not is_testing_split(self.split):
            labels = targets.get_field('labels')
            targets = targets[labels == 1]  # remove not cars
            # if self.split == 'val' and self.remove_ignore:
            #     raise NotImplementedError()
            masks = targets.get_field('kins_masks').get_mask_tensor(squeeze=False).numpy().astype(np.uint8)
            # adapt to yolact style
            tgts = []
            height, width, _ = img.shape
            scale = torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            for bbox in targets.bbox:
                tgt = bbox / scale
                tgts.append(tgt.tolist() + [0])
            num_crowds = 0
            if self.transforms is not None:
                if len(tgts) > 0:
                    tgts = np.array(tgts)
                    dps = {'image': img, 'masks': masks, 'boxes': tgts[:, :4],
                           'labels': {'num_crowds': num_crowds,
                                      'labels': tgts[:, 4]}}
                    # img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    #                                            {'num_crowds': num_crowds, 'labels': target[:, 4]})
                    dps = self.transforms(dps)
                    img = dps['image']
                    masks = dps['masks']
                    boxes = dps['boxes']
                    labels = dps['labels']
                    # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                    num_crowds = labels['num_crowds']
                    labels = labels['labels']

                    target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                else:
                    return self.__getitem__(random.randint(0, len(self.ids) - 1))
                    # img = self.transforms({'image': img,
                    #                        'masks': np.zeros((1, height, width), dtype=np.float),
                    #                        'boxes': np.array([[0.0, 0.0, 1.0, 1.0]]),
                    #                        'labels': {'num_crowds': 0, 'labels': np.array([0])}})['image']
                    # masks = None
                    # target = None
            # if target.shape[0] == 0:
            #     print('Warning: Augmentation output an example with no ground truth. Resampling...')

        dps = {
            'image': torch.from_numpy(img).permute(2, 0, 1),
            'target': target,
            'masks': masks,
            'height': height,
            'width': width,
            'num_crowds': num_crowds
        }
        return dps

    def get_image(self, index):
        img_id = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        left_img = cv2.imread(os.path.join(self.root, 'object', split, 'image_2', img_id + '.png'))
        return left_img

    def get_ground_truth(self, index):
        img_id = self.ids[index]
        if not is_testing_split(self.split):
            left_annotation = self.annotations['left'][int(img_id)]
            # right_annotation = self.annotations['right'][int(img_id)]
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(left_annotation["boxes"], (width, height), mode="xyxy")
            left_target.add_field("labels", left_annotation["labels"])
            # left_target.add_field("alphas", left_annotation['alphas'])
            # boxes_3d = Box3DList(left_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            # left_target.add_field("box3d", boxes_3d)
            # left_target.add_map('disparity', self.get_disparity(index))
            # left_target.add_map('disparity_fg', self.get_disparity_fg(index))
            # left_target.add_field('masks', self.get_mask(index))
            left_target.add_field('kins_masks', self.get_kins_mask(index, ))
            # left_target.add_field('kins_a_masks', self.get_kins_a_mask(index))
            # left_target.add_field('truncation', torch.tensor(self.truncations_list[int(img_id)]))
            # left_target.add_field('occlusion', torch.tensor(self.occlusions_list[int(img_id)]))
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            left_target = left_target.clip_to_image(remove_empty=True)
            # right target
            # right_target = BoxList(right_annotation["boxes"], (width, height), mode="xyxy")
            # right_target.add_field("labels", right_annotation["labels"])
            # right_target.add_field("alphas", right_annotation['alphas'])
            # boxes_3d = Box3DList(right_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            # right_target.add_field("box3d", boxes_3d)
            # right_target = right_target.clip_to_image(remove_empty=True)
            # target = {'left': left_target, 'right': right_target}
            return left_target
        else:
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
            left_target.add_map('disparity', self.get_disparity(index))
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

    def map_class_id_to_class_name(self, class_id):
        return KITTIObjectDataset.CLASSES[class_id]

    def read_annotations(self):
        double_view_annotations = {}
        # if self.split == 'test':
        if is_testing_split(self.split):
            return {'left': [], 'right': []}
        for view in [2, 3]:
            annodir = os.path.join(self.root, f"object/training/label_{view}")
            anno_cache_path = os.path.join(annodir, 'annotations.pkl')
            if os.path.exists(anno_cache_path):
                annotations = pickle.load(open(anno_cache_path, 'rb'))
            else:
                print('generating', anno_cache_path)
                annotations = []
                for i in tqdm(range(7481)):
                    if view == 2:
                        anno_per_img = load_label_2(self.root, 'training', i)
                    else:
                        anno_per_img = load_label_3(self.root, 'training', i)
                    num_objs = len(anno_per_img)
                    label = np.zeros((num_objs), dtype=np.int32)
                    boxes = np.zeros((num_objs, 4), dtype=np.float32)
                    boxes_3d = np.zeros((num_objs, 7), dtype=np.float32)
                    alphas = np.zeros((num_objs), dtype=np.float32)
                    ix = 0
                    for anno in anno_per_img:
                        cls, truncated, occluded, alpha, x1, \
                        y1, x2, y2, h, w, l, x, y, z, ry = anno.cls.name, anno.truncated, anno.occluded, anno.alpha, anno.x1, anno.y1, anno.x2, anno.y2, \
                                                           anno.h, anno.w, anno.l, \
                                                           anno.x, anno.y, anno.z, anno.ry
                        cls_str = cls.lower().strip()
                        if self.split == 'training':
                            # regard car and van as positive
                            cls_str = 'car' if cls_str in ['car', 'van'] else '__background__'
                        else:  # val
                            # return 'dontcare' in validation phase
                            if cls_str != 'car':
                                cls_str = '__background__'
                        cls = self.class_to_ind[cls_str]
                        label[ix] = cls
                        alphas[ix] = float(alpha)
                        boxes[ix, :] = [float(x1), float(y1), float(x2), float(y2)]
                        boxes_3d[ix, :] = [ry, l, h, w, x, y, z]
                        ix += 1
                    label = label[:ix]
                    alphas = alphas[:ix]
                    boxes = boxes[:ix, :]
                    boxes_3d = boxes_3d[:ix, :]
                    P2 = load_calib(self.root, 'training', i).P2
                    annotations.append({'labels': torch.tensor(label),
                                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                                        'boxes_3d': torch.tensor(boxes_3d),
                                        'alphas': torch.tensor(alphas),
                                        'P2': torch.tensor(P2).float(),
                                        })
                pickle.dump(annotations, open(anno_cache_path, 'wb'))
            if view == 2:
                double_view_annotations['left'] = annotations
            else:
                double_view_annotations['right'] = annotations
        return double_view_annotations

    def read_info(self):
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        infopath = os.path.join(self.root,
                                f'object/{split}/infos.pkl')
        if not os.path.exists(infopath):
            infos = []
            # total = 7481 if self.split != 'test' else 7518
            total = 7481 if not is_testing_split(self.split) else 7518
            for i in tqdm(range(total)):
                img = load_image_2(self.root, split, i)
                # img = Image.fromarray(img2)
                infos.append({"height": img.height, "width": img.width, 'size': img.size})
            pickle.dump(infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                infos = pickle.load(f)
        return infos

    def get_truncations_occluded_list(self):
        # if self.split == 'test':
        if is_testing_split(self.split):
            return [], []
        annodir = os.path.join(self.root, f"object/training/label_2")
        truncations_occluded_cache_path = os.path.join(annodir, 'truncations_occluded.pkl')
        if os.path.exists(truncations_occluded_cache_path):
            truncations_list, occluded_list = pickle.load(open(truncations_occluded_cache_path, 'rb'))
        else:
            truncations_list, occluded_list = [], []
            print('generating', truncations_occluded_cache_path)
            for i in tqdm(range(7481)):
                anno_per_img = load_label_2(self.root, 'training', i)
                truncations_list_per_img = []
                occluded_list_per_img = []
                for anno in anno_per_img:
                    truncated, occluded = float(anno.truncated), float(anno.occluded)
                    truncations_list_per_img.append(truncated)
                    occluded_list_per_img.append(occluded)
                truncations_list.append(truncations_list_per_img)
                occluded_list.append(occluded_list_per_img)
            pickle.dump([truncations_list, occluded_list],
                        open(truncations_occluded_cache_path, 'wb'))
        return truncations_list, occluded_list

    def get_offline_prediction(self, index):
        # imgid = self.ids[index]
        # pred = pickle.load(open(os.path.join(
        #     self.offline_2d_predictions_dir, str(imgid) + '.pkl'), 'rb'))
        # lp, rp = pred['left'], pred['right']
        lpmem, rpmem = self.o2dpreds['left'][index], self.o2dpreds['right'][index]
        return lpmem, rpmem

    def get_kins_mask(self, index, ):
        # try:
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']
        if split == 'training':
            path = os.path.join(self.root, 'object', split, 'kins_mask_2', imgid + '.zarr')
            if osp.exists(path):
                mask = zarr.load(path) != 0
                mask = SegmentationMask(mask, (width, height), mode='mask')
            else:
                mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        else:
            mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        # except Exception as e:
        #     mask = self.get_mask(index)
        return mask

    # def get_kins_a_mask(self, index):
    #     try:
    #         imgid = self.ids[index]
    #         # split = 'training' if self.split != 'test' else 'testing'
    #         split = 'training' if not is_testing_split(self.split) else 'testing'
    #         imginfo = self.get_img_info(index)
    #         width = imginfo['width']
    #         height = imginfo['height']
    #         if split == 'training':
    #             mask = zarr.load(
    #                 os.path.join(self.root, 'object', split, 'kins_a_mask_2', imgid + '.zarr')) != 0
    #             mask = SegmentationMask(mask, (width, height), mode='mask')
    #         else:
    #             mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
    #     except Exception as e:
    #         mask = self.get_mask(index)
    #     return mask

    def get_mask(self, index):
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']
        if split == 'training':
            path = os.path.join(self.root, 'object', split, self.shape_prior_base, 'mask_2', imgid + '.zarr')
            assert osp.exists(path)
            mask = zarr.load(path) != 0
            # else:
            #     warn(f'{path} not exists! using 0 mask')
            #     mask = np.zeros((height, width), dtype=np.bool)
            mask = SegmentationMask(mask, (width, height), mode='mask')
        else:
            mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        return mask

    def get_disparity(self, index):
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        if split == 'training':
            path = os.path.join(self.root, 'object', split,
                                self.shape_prior_base, 'disparity_2',
                                imgid + '.png')
            assert osp.exists(path), path
            disp = cv2.imread(path, 2).astype(np.float32) / 256
            disp = DisparityMap(disp)
        else:
            imginfo = self.get_img_info(index)
            width = imginfo['width']
            height = imginfo['height']
            disp = DisparityMap(np.ones((height, width)))
        return disp

    def get_disparity_fg(self, index):
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        if split == 'training':
            path = os.path.join(self.root, 'object', split,
                                self.shape_prior_base, 'disparity_fg_2',
                                imgid + '.png')
            assert osp.exists(path), path
            disp = cv2.imread(path, 2).astype(np.float32) / 256
            disp = DisparityMap(disp)
        else:
            imginfo = self.get_img_info(index)
            width = imginfo['width']
            height = imginfo['height']
            disp = DisparityMap(np.ones((height, width)))
        return disp

    def get_shape(self, index):
        # raise NotImplementedError()
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'test'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        path = os.path.join(self.root, 'object', split,
                            self.shape_prior_base, 'shape_prior_results_bin_z/',
                            imgid + '.bin')
        z = np.fromfile(path, dtype=np.float64)
        z = z.reshape((-1, 5))
        z = torch.from_numpy(z).float()
        return z

    def get_mesh_iou(self, index):
        imgid = self.ids[index]
        meshiou = torch.tensor(self.meshious[int(imgid)]).reshape(-1, 1)
        return meshiou

    def prepare_mesh_iou(self):
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        if split == 'testing':
            return None
        path = os.path.join(self.root, 'object', split,
                            self.shape_prior_base, 'meshiou/meshiou.pkl')
        if os.path.exists(path):
            meshious = pickle.load(open(path, 'rb'))
        else:
            print('no meshiou found.')
            meshious = None

        return meshious

    def get_calibration(self, index):
        imgid = self.ids[index]
        # split = 'training' if self.split != 'test' else 'testing'
        split = 'training' if not is_testing_split(self.split) else 'testing'
        calib = load_calib(self.root, split, imgid)
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
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/yolact/kitti/resnet50.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg).dataset
    d = ds[4]
    # ds = Linemod(cfg, data_dir, scene, 'trainval', 1, 19)


if __name__ == '__main__':
    main()
