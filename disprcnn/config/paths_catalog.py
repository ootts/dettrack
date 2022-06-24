import os.path as osp
import os

import re

import loguru
from dl_ext.primitive import safe_zip


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('~/Datasets')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)

    @staticmethod
    def get(name: str):
        if name.startswith('coco2017'):  # nerf_blender_lego_train
            return get_coco2017(name)
        elif name.startswith("kittikins"):
            return get_kittikins(name)
        elif name.startswith("kittitracking"):
            return get_kittitracking(name)
        raise RuntimeError("Dataset not available: {}".format(name))


def get_coco2017(name):
    split = name.split("_")[1]
    return dict(
        factory='COCODetection',
        args={'data_dir': 'data/coco',
              'split': split,
              }
    )


def get_kittikins(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 100
    return dict(
        factory='KITTIKinsDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )


def get_kittitracking(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 100
    return dict(
        factory='KITTITrackingDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )
