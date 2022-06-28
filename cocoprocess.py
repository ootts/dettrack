import cv2
import os.path as osp
import random

import torch
import numpy as np
from torch.utils import data
from pycocotools.coco import COCO

def main():
    data_dir = "data/coco"
    split = "train"
    image_dir = osp.join(data_dir, split + "2017")
    root = image_dir
    info_file = osp.join(data_dir, f"annotations/instances_{split}2017.json")
    coco = COCO(info_file)
    ids = list(coco.imgToAnns.keys())
    has_gt = True
    if len( ids) == 0 or not has_gt:
         ids = list( coco.imgs.keys())
    ann_ids = coco.getAnnIds(ids)
    anns = coco.loadAnns(ann_ids)
    print(anns[0])
    



if __name__ == "__main__":
    main()