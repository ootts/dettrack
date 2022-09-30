import cv2
import os.path as osp
import random

import torch
import numpy as np
from torch.utils import data

from disprcnn.structures.bounding_box import BoxList

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


def get_label_map():
    # if cfg.dataset.label_map is None:
    #     return {x + 1: x + 1 for x in range(len(cfg.dataset.class_names))}
    # else:
    #     return cfg.dataset.label_map
    return COCO_LABEL_MAP


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res


class COCODetection(data.Dataset):
    def __init__(self, cfg, data_dir, split='train', transforms=None,
                 has_gt=True,
                 ds_len=-1):
        # Do this here because we have too many things named COCO
        self.cfg = cfg.dataset.coco
        from pycocotools.coco import COCO

        image_dir = osp.join(data_dir, split + "2017")
        self.root = image_dir
        if self.cfg.class_only != "":
            class_only = "_" + self.cfg.class_only
        else:
            class_only = ""
        info_file = osp.join(data_dir, f"annotations/instances{class_only}_{split}2017.json")
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transforms
        self.target_transform = COCOAnnotationTransform()

        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)
        # return 30

    def __getitem__(self, index):
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape
        if self.cfg.use_gray:
            Lgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[..., 0] = Lgray
            img[..., 1] = Lgray
            img[..., 2] = Lgray

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                dps = {'image': img, 'masks': masks, 'boxes': target[:, :4],
                       'labels': {'num_crowds': num_crowds,
                                  'labels': target[:, 4]}}
                # img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                #                                            {'num_crowds': num_crowds, 'labels': target[:, 4]})
                dps = self.transform(dps)
                img = dps['image']
                masks = dps['masks']
                boxes = dps['boxes']
                labels = dps['labels']
                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img = self.transform({'image': img,
                                      'masks': np.zeros((1, height, width), dtype=np.float),
                                      'boxes': np.array([[0, 0, 1, 1]]),
                                      'labels': {'num_crowds': 0, 'labels': np.array([0])}})['image']
                masks = None
                target = None

        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.__getitem__(random.randint(0, len(self.ids) - 1))
        # boxlist = BoxList(target[:, 0:4], (1, 1))
        boxlist = BoxList(target[:, :4], (1, 1))
        boxlist.add_field("labels", torch.tensor(target[:, -1]).long())
        boxlist.add_field("masks", torch.from_numpy(masks).long())
        is_last_frame = index == len(self) - 1
        dps = {
            'image': torch.from_numpy(img).permute(2, 0, 1),
            'target': boxlist,
            # 'masks': masks,
            'height': height,
            'width': width,
            'num_crowds': num_crowds,
            'imgid': img_id,
            'index': index,
            'is_last_frame': is_last_frame

        }
        # print(f"index {index}, is_last_frame {is_last_frame}")
        return dps


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/yolact/coco/resnet50_ped_gray.yaml'
    cfg = setup(args)
    ds = make_data_loader(cfg).dataset
    d = ds[0]
    boxlist: BoxList = d['target']
    boxlist.plot(d['image'].permute(1, 2, 0), show=True, draw_mask=True)
    print()


if __name__ == '__main__':
    main()
