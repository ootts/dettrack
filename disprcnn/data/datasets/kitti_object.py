import os

import numpy as np
import os.path as osp
import pickle
from warnings import warn

import cv2
import torch
import torch.utils.data
import zarr
from PIL import Image
from dl_ext.vision_ext.datasets.kitti.io import *
from tqdm import tqdm

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.stereo_utils import align_left_right_targets


class KITTIObjectDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "car",
        'dontcare'
    )
    NUM_TRAINING = 7481
    NUM_TRAIN = 3712
    NUM_VAL = 3769
    NUM_TESTING = 7518

    def __init__(self, cfg, root, split, transforms=None, ds_len=-1):
        self.root = root
        self.split = split
        self.cfg = cfg.dataset.kitti_object
        cls = KITTIObjectDataset.CLASSES
        self.remove_ignore = self.cfg.remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        self.shape_prior_base = self.cfg.shape_prior_base
        offline_2d_predictions_path = self.cfg.offline_2d_predictions_path
        filter_empty = self.cfg.filter_empty
        # make cache or read cached annotation
        self.annotations = self.read_annotations()
        self.infos = self.read_info()
        self._imgsetpath = os.path.join(self.root, "object/split_set/%s_set.txt")
        if offline_2d_predictions_path != '':
            o2ppath = offline_2d_predictions_path % split
            if is_testing_split(self.split):
                s = o2ppath.split('/')[-2]
                s = '_'.join(s.split('_')[:2])
                o2ppath = '/'.join(o2ppath.split('/')[:-2] + [s] + [o2ppath.split('/')[-1]])
            o2p_dir = o2ppath.rstrip(".pth")
            if not osp.exists(o2p_dir):
                o2dpreds = torch.load(o2ppath, 'cpu')
                os.makedirs(o2p_dir, exist_ok=True)
                for pred in o2dpreds:
                    imgid = pred['left'].extra_fields['imgid']
                    with  open(osp.join(o2p_dir, f"{imgid:06d}.pkl"), "wb") as f:
                        pickle.dump(pred, f)
            self.o2p_dir = o2p_dir
        else:
            self.o2p_dir = ""
        with open(self._imgsetpath % self.split) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        # if hasattr(self, 'o2dpreds'):
        #     assert len(self.ids) == len(self.o2dpreds)
        if filter_empty:
            ids = []
            for i, id in enumerate(self.ids):
                if self.annotations['left'][int(id)]['labels'].sum() != 0:
                    ids.append(id)
            self.ids = ids
        self.truncations_list, self.occlusions_list = self.get_truncations_occluded_list()

        # self.offline_2d_predictions_dir = offline_2d_predictions_path
        if ds_len > 0:
            self.ids = self.ids[:ds_len]
        print('using dataset of length', self.__len__())

    def __getitem__(self, index):
        imgs = self.get_image(index)
        targets = self.get_ground_truth(index)
        dps = {}
        if self.transforms is not None:
            tsfmed_left_img = self.transforms({'image': imgs['left']})['image']
            tsfmed_right_img = self.transforms({'image': imgs['right']})['image']
            dps['images'] = {'left': torch.from_numpy(tsfmed_left_img).permute(2, 0, 1),
                             'right': torch.from_numpy(tsfmed_right_img).permute(2, 0, 1)}
        dps['original_images'] = {'left': np.ascontiguousarray(imgs['left'][:, :, ::-1]),
                                  'right': np.ascontiguousarray(imgs['right'][:, :, ::-1])}
        if not is_testing_split(self.split):
            for view in ['left', 'right']:
                labels = targets[view].get_field('labels')
                targets[view] = targets[view][labels == 1]  # remove not cars
            l, r = align_left_right_targets(targets['left'], targets['right'], thresh=0.15)
            if self.split == 'val' and self.remove_ignore:
                l, r = self.remove_ignore_cars(l, r)
            targets['left'] = l
            targets['right'] = r
        dps['targets'] = targets
        if self.o2p_dir != '':
            lp, rp = self.get_offline_prediction(index)
            lp = lp.resize(targets['left'].size)
            rp = rp.resize(targets['right'].size)
            dps['predictions'] = {'left': lp, 'right': rp}
        dps['imgid'] = int(self.ids[index])
        return dps

    def get_image(self, index):
        img_id = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        left_img = cv2.imread(os.path.join(self.root, 'object', split, 'image_2', img_id + '.png'))
        right_img = cv2.imread(os.path.join(self.root, 'object', split, 'image_3', img_id + '.png'))
        if self.cfg.use_gray:
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

    def get_ground_truth(self, index):
        img_id = self.ids[index]
        if not is_testing_split(self.split):
            left_annotation = self.annotations['left'][int(img_id)]
            right_annotation = self.annotations['right'][int(img_id)]
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(left_annotation["boxes"], (width, height), mode="xyxy")
            left_target.add_field("labels", left_annotation["labels"])
            boxes_3d = Box3DList(left_annotation["boxes_3d"], mode='ry_lhwxyz')
            left_target.add_field("box3d", boxes_3d)
            left_target.add_map('disparity', self.get_disparity(index))
            left_target.add_field('masks', self.get_mask(index))
            left_target.add_field('kins_masks', self.get_kins_mask(index, ))
            left_target.add_field('truncation', torch.tensor(self.truncations_list[int(img_id)]))
            left_target.add_field('occlusion', torch.tensor(self.occlusions_list[int(img_id)]))
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            left_target = left_target.clip_to_image(remove_empty=True)
            # right target
            right_target = BoxList(right_annotation["boxes"], (width, height), mode="xyxy")
            right_target.add_field("labels", right_annotation["labels"])
            # right_target.add_field("alphas", right_annotation['alphas'])
            # boxes_3d = Box3DList(right_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            # right_target.add_field("box3d", boxes_3d)
            right_target = right_target.clip_to_image(remove_empty=True)
            target = {'left': left_target, 'right': right_target}
            return target
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
        # if self.split == 'val':
        #     print('VAL 4!')
        # return 1
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
                with open(anno_cache_path, 'rb') as f:
                    annotations = pickle.load(f)
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
            with open(truncations_occluded_cache_path, 'rb') as f:
                truncations_list, occluded_list = pickle.load(f)
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
        imgid = self.ids[index]
        with open(osp.join(self.o2p_dir, f"{imgid}.pkl"), "rb") as f:
            pred = pickle.load(f)
        lpmem, rpmem = pred['left'], pred['right']
        return lpmem, rpmem

    def get_kins_mask(self, index, ):
        imgid = self.ids[index]
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

    def get_mask(self, index):
        imgid = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']
        if split == 'training':
            path = os.path.join(self.root, 'object', split, self.shape_prior_base, 'mask_2', imgid + '.zarr')
            assert osp.exists(path)
            mask = zarr.load(path) != 0
            mask = SegmentationMask(mask, (width, height), mode='mask')
        else:
            mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        return mask

    def get_disparity(self, index):
        imgid = self.ids[index]
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
    # return split in ['test', 'testmini', 'test1', 'test2', 'val']
    return split in ['test', 'testmini', 'test1', 'test2']


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_object/resnet50.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg).dataset
    d = ds[4]


if __name__ == '__main__':
    main()
