from collections import OrderedDict

import torch
import loguru
import numpy as np
import tqdm

from disprcnn.modeling.models.yolact.layers.box_utils import jaccard, mask_iou
from disprcnn.registry import EVALUATORS
import os

from disprcnn.structures.apdataobject import APDataObject
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.timer import EvalTime


@EVALUATORS.register("yolact")
class YolactEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.iou_thresholds = [x / 100 for x in range(50, 100, 5)]
        self.class_names = self.cfg.model.yolact.class_names
        self.ap_data = {
            'box': [[APDataObject() for _ in self.class_names] for _ in self.iou_thresholds],
            'mask': [[APDataObject() for _ in self.class_names] for _ in self.iou_thresholds]
        }

    def __call__(self, predictions, trainer):
        # class_names = list(trainer.cfg.model.yolact.class_names)

        ds = trainer.valid_dl.dataset
        assert len(predictions) == len(ds)
        et = EvalTime()
        for i in tqdm.trange(len(predictions), desc='evaling'):
            # for pred, dps in tqdm.tqdm(zip(predictions, ds), total=len(ds), desc="evaling"):
            pred = predictions[i]
            et('')
            dps = ds[i]
            et('loader')
            gt = torch.cat([dps['target'].bbox, dps['target'].get_field('labels')[:, None]], dim=1)
            self.prep_metrics(pred, dps['image'], gt,
                              dps['target'].get_field('masks'),
                              dps['height'],
                              dps['width'],
                              dps['num_crowds'],
                              dps['imgid'],
                              None
                              )
            et('prep')
        self.calc_map()

    def prep_metrics(self, result, img, gt, gt_masks, h, w, num_crowd, image_id, detections):
        """ Returns a list of APs for this image, with each element being for a class  """
        gt_boxes = gt[:, :4]
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h
        gt_classes = gt[:, 4].long().tolist()
        gt_masks = gt_masks.view(-1, h * w)

        if num_crowd > 0:
            split = lambda x: (x[-num_crowd:], x[:-num_crowd])
            crowd_boxes, gt_boxes = split(gt_boxes)
            crowd_masks, gt_masks = split(gt_masks)
            crowd_classes, gt_classes = split(gt_classes)

        if len(result) == 0:
            return

        classes = result.get_field("labels").tolist()
        scores = result.get_field("scores").tolist()
        masks = SegmentationMask(result.get_field("masks"), (w, h), mode="mask")
        masks = masks.get_mask_tensor(squeeze=False)
        boxes = result.bbox
        box_scores = scores
        mask_scores = scores
        masks = masks.reshape(-1, h * w).cuda()
        boxes = boxes.cuda()

        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks.cuda())
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float().cuda())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks.cuda(), iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float().cuda(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

        for _class in set(classes + gt_classes):
            ap_per_iou = []
            num_gt_for_class = sum([1 for x in gt_classes if x == _class])

            for iouIdx in range(len(self.iou_thresholds)):
                iou_threshold = self.iou_thresholds[iouIdx]

                for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                    gt_used = [False] * len(gt_classes)

                    ap_obj = self.ap_data[iou_type][iouIdx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class)

                    for i in indices:
                        if classes[i] != _class:
                            continue

                        max_iou_found = iou_threshold
                        max_match_idx = -1
                        for j in range(num_gt):
                            if gt_used[j] or gt_classes[j] != _class:
                                continue

                            iou = iou_func(i, j)

                            if iou > max_iou_found:
                                max_iou_found = iou
                                max_match_idx = j

                        if max_match_idx >= 0:
                            gt_used[max_match_idx] = True
                            ap_obj.push(score_func(i), True)
                        else:
                            # If the detection matches a crowd, we can just ignore it
                            matched_crowd = False

                            if num_crowd > 0:
                                for j in range(len(crowd_classes)):
                                    if crowd_classes[j] != _class:
                                        continue

                                    iou = crowd_func(i, j)

                                    if iou > iou_threshold:
                                        matched_crowd = True
                                        break

                            # All this crowd code so that we can make sure that our eval code gives the
                            # same result as COCOEval. There aren't even that many crowd annotations to
                            # begin with, but accuracy is of the utmost importance.
                            if not matched_crowd:
                                ap_obj.push(score_func(i), False)

    def calc_map(self):
        print('Calculating mAP...')
        aps = [{'box': [], 'mask': []} for _ in self.iou_thresholds]

        for _class in range(len(self.class_names)):
            for iou_idx in range(len(self.iou_thresholds)):
                for iou_type in ('box', 'mask'):
                    ap_obj = self.ap_data[iou_type][iou_idx][_class]

                    if not ap_obj.is_empty():
                        aps[iou_idx][iou_type].append(ap_obj.get_ap())

        all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

        # Looking back at it, this code is really hard to read :/
        for iou_type in ('box', 'mask'):
            all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
            for i, threshold in enumerate(self.iou_thresholds):
                mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
                all_maps[iou_type][int(threshold * 100)] = mAP
            all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

        self.print_maps(all_maps)

        # Put in a prettier format so we can serialize it to json during training
        all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
        return all_maps

    def print_maps(self, all_maps):
        # Warning: hacky
        make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
        make_sep = lambda n: ('-------+' * n)

        print()
        print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
        print(make_sep(len(all_maps['box']) + 1))
        for iou_type in ('box', 'mask'):
            print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
        print(make_sep(len(all_maps['box']) + 1))
        print()


def _mask_iou(mask1, mask2, iscrowd=False):
    ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()
