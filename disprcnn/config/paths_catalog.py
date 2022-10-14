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
        elif name.startswith("graykittikins"):
            return get_graykittikins(name)
        elif name.startswith("kittitrackingstereo"):
            return get_kittitrackingstereo(name)
        elif name.startswith("kittitracking"):
            return get_kittitracking(name)
        elif name.startswith("kittiobj"):
            return get_kittiobj(name)
        elif name.startswith("kittiroi"):
            return get_kittiroi(name)
        elif name.startswith("kittivelodyne"):
            return get_kittivelodyne(name)
        elif name.startswith("realtrackingstereo"):
            return get_realtrackingstereo(name)

        raise RuntimeError("Dataset not available: {}".format(name))


def get_graykittikins(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 100
    return dict(
        factory='GrayKITTIKinsDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )


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
        ds_len = 700
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
        ds_len = 700
        # ds_len = 10
    return dict(
        factory='KITTITrackingDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )


def get_kittitrackingstereo(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 700
        # ds_len = 10
    return dict(
        factory='KITTITrackingStereoDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )


def get_kittiobj(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 100
        # ds_len = 10
    return dict(
        factory='KITTIObjectDataset',
        args={'root': 'data/kitti',
              'split': split,
              'ds_len': ds_len
              }
    )


def get_kittivelodyne(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 100
    return dict(
        factory='KittiVelodyneDataset',
        args={'split': split,
              'ds_len': ds_len
              }
    )


def get_kittiroi(name):
    split = name.split("_")[1]
    ds_len = -1
    if split == 'valmini':
        split = 'val'
        ds_len = 12
    if split == 'trainmini':
        split = 'train'
        ds_len = 12
    return dict(
        factory='KITTIRoiDatasetRA',
        args={'split': split,
              'ds_len': ds_len
              }
    )


def get_realtrackingstereo(name):
    split = name.split("_")[1]
    ds_len = -1
    return dict(
        factory='RealTrackingStereoDataset',
        args={'root': 'data/real/processed',
              'split': split,
              'ds_len': ds_len
              }
    )

