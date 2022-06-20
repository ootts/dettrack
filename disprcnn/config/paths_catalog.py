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
        if name.startswith('nerf_blender'):  # nerf_blender_lego_train
            return get_nerf_blender(name)
        elif name.startswith('nerf_llff'):  # nerf_blender_lego_train
            return get_nerf_llff(name)
        elif 'epic' in name:
            return get_epic(name)
        elif 'clevr' in name:
            return get_clevr(name)
        elif 'kitti2' in name:
            return get_kitti2(name)
        elif 'kitti' in name:
            return get_kitti(name)
        elif 'linemodmv' in name:
            return get_linemodmv(name)
        elif 'linemodmco' in name:
            return get_linemodmco(name)
        elif 'linemodmc' in name:
            return get_linemodmc(name)
        elif 'linemodvideo' in name:
            return get_linemodvideo(name)
        elif 'linemodmo' in name:
            return get_linemodmo(name)
        elif 'linemodssbarf' in name:
            return get_linemodssbarf(name)
        elif 'linemodssmo' in name:
            return get_linemodssmo(name)
        elif 'linemodssmv' in name:
            return get_linemodssmv(name)
        elif 'linemodss' in name:
            return get_linemodss(name)
        elif 'linemod' in name:
            return get_linemod(name)
        elif 'dtubarf' in name:
            return get_dtubarf(name)
        elif 'dtu' in name:
            return get_dtu(name)
        elif 'cater' in name:
            return get_cater(name)
        elif 'realsense' in name:
            return get_realsense(name)
        elif 'gripper_kinect' in name:
            return get_gripper_kinect(name)
        elif 'gripper' in name:
            return get_gripper(name)
        elif 'mesh' in name:
            return get_mesh(name)
        elif 'kinectrobot' in name:
            return get_kinectrobot(name)
        elif 'kinect' in name:
            return get_kinect(name)
        raise RuntimeError("Dataset not available: {}".format(name))


def get_nerf_blender(name):
    # nerf_blender_lego_ds2_train
    # scene = name.split("_")[2]
    _, _, scene, ds, split = name.split('_')
    skip_dict = {'train': 1, 'val': 8, 'test': 8, 'render': 1}
    skip = skip_dict[split]
    return dict(
        factory='NerfBlenderDataset',
        args={'data_dir': 'data/nerf_synthetic',
              'scene': scene,
              'split': split,
              'downscale': int(ds[2:]),
              'skip': skip,
              'ds_len': -1,
              }
    )


def get_nerf_llff(name):
    # nerf_llff_fern_ds8_train
    _, _, scene, ds, split = name.split('_')
    return dict(
        factory='NerfLLFFDataset',
        args={'data_dir': 'data/nerf_llff_data',
              'scene': scene,
              'split': split,
              'downscale': int(ds[2:]),
              # 'skip': skip,
              'ds_len': -1,
              }
    )


def get_clevr(name):
    if 'train' in name:
        split = 'train'
    elif 'val' in name:
        split = 'val'
    else:
        raise NotImplementedError()
    if 'mmini' in name:
        ds_len = 1
    elif 'mini' in name:
        ds_len = 100
    else:
        ds_len = -1
    return dict(
        factory='ClevrDataset',
        args={'data_dir': osp.expanduser('~/Datasets/CLEVR_v1.0'),
              'split': split,
              'max_n_objects': 6,
              'ds_len': ds_len,
              }
    )


def get_kitti2(name):
    data_dir = osp.expanduser('~/Datasets/kitti/tracking/training')
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='KittiNsg2Dataset',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor,
              }
    )


def get_kitti(name):
    data_dir = osp.expanduser('~/Datasets/kitti/tracking/training')
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='KittiNsgDataset',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor,
              }
    )


def get_linemod(name):
    data_dir = 'data/blenderproc/physics_positioning/'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='Linemod',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_linemodss(name):
    data_dir = 'data/blenderproc/physics_positioning_ss'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodSs',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_linemodssmv(name):
    data_dir = 'data/blenderproc/physics_positioning_ss'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodSsMv',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_linemodssmo(name):
    data_dir = 'data/blenderproc/physics_positioning_mo'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodSsMo',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_realsense(name):
    data_dir = 'data/realsense'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='Realsense',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_kinect(name):
    data_dir = 'data/kinect'
    s1, s2, s3, s4, firstend, split, training_factor = name.split('_')[1:]
    scene = '_'.join([s1, s2, s3, s4])
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='Kinect',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_kinectrobot(name):
    data_dir = 'data/kinect'
    s1, s2, s3, s4, skip, firstend, split, training_factor = name.split('_')[1:]
    scene = '_'.join([s1, s2, s3, s4])
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    skip = int(skip[4:])
    return dict(
        factory='KinectRobot',
        args={'data_dir': data_dir,
              'scene': scene,
              'skip': skip,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_gripper_kinect(name):
    data_dir = 'data/kinect/2022_0607_1519_16/processed'
    tf = int(name.split("_")[-1][2:])
    return dict(
        factory='GripperKinectDataset',
        args={'data_dir': data_dir,
              "downscale": tf
              }
    )


def get_gripper(name):
    return dict(
        factory='GripperDataset',
        args={}
    )


def get_mesh(name):
    return dict(
        factory='MeshDataset',
        args={}
    )


def get_linemodssbarf(name):
    data_dir = 'data/blenderproc/physics_positioning_ss'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodSsBarf',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_linemodmo(name):
    data_dir = 'data/blenderproc/physics_positioning_mo/'
    scene, firstend, split, training_factor = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodMo',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )


def get_linemodmv(name):
    data_dir = 'data/blenderproc/physics_positioning_mv/'
    # scene, split, training_factor, views, maxframe = name.split('_')[1:6]
    scene, startend, split, training_factor, views = name.split('_')[1:6]
    training_factor = float(training_factor[2:])
    start, end = map(int, startend.split('to'))
    views = int(views[:-5])
    return dict(
        factory='LinemodMV',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'start': start,
              'end': end,
              'views': views,
              # 'first_frame': first_frame,
              # 'last_frame': last_frame,
              'resize_factor': training_factor,
              }
    )


def get_linemodmco(name):
    data_dir = 'data/blenderproc/physics_positioning_mco/'
    # scene, split, training_factor, views, maxframe = name.split('_')[1:6]
    scene, startend, training_factor = name.split('_')[1:6]
    training_factor = float(training_factor[2:])
    start, end = map(int, startend.split('to'))
    # views = int(views[:-5])
    return dict(
        factory='LinemodMCO',
        args={'data_dir': data_dir,
              'scene': scene,
              # 'split': split,
              'start': start,
              'end': end,
              # 'views': views,
              # 'first_frame': first_frame,
              # 'last_frame': last_frame,
              'resize_factor': training_factor,
              }
    )


def get_linemodmc(name):
    data_dir = 'data/blenderproc/physics_positioning_mc/'
    # scene, split, training_factor, views, maxframe = name.split('_')[1:6]
    scene, startend, split, training_factor, views = name.split('_')[1:6]
    training_factor = float(training_factor[2:])
    start, end = map(int, startend.split('to'))
    views = int(views[:-5])
    return dict(
        factory='LinemodMC',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'start': start,
              'end': end,
              'views': views,
              # 'first_frame': first_frame,
              # 'last_frame': last_frame,
              'resize_factor': training_factor,
              }
    )


def get_dtu(name):
    # dtu_scan65_trainval_ds4_sr3
    # data_dir, split, downscale=1.0, scale_radius=-1, transforms=None, ds_len=-1
    _, scanid, split, ds, sr = name.split('_')
    data_dir = f'data/DTU/{scanid}'
    ds = float(ds[2:])
    sr = float(sr[2:])
    return dict(
        factory='DTUDataset',
        args={'data_dir': data_dir,
              'split': split,
              'downscale': ds,
              'scale_radius': sr,
              }
    )


def get_dtubarf(name):
    # dtu_scan65_trainval_ds4_sr3
    # data_dir, split, downscale=1.0, scale_radius=-1, transforms=None, ds_len=-1
    _, scanid, split, ds, sr = name.split('_')
    data_dir = f'data/DTU/{scanid}'
    ds = float(ds[2:])
    sr = float(sr[2:])
    return dict(
        factory='DTUBarfDataset',
        args={'data_dir': data_dir,
              'split': split,
              'downscale': ds,
              'scale_radius': sr,
              }
    )


def get_cater(name):
    # cater_nf6_train
    # data_dir = osp.expanduser('~/Datasets/CATER/all_actions_cameramotion')
    data_dir = 'data/CATER'
    nframes, split = name.split('_')[1:3]
    nframes = int(nframes[2:])
    if 'mmini' in name:
        ds_len = 1
    elif 'mini' in name:
        ds_len = 100
    else:
        ds_len = -1
    return dict(
        factory='CaterDataset',
        args={'data_dir': data_dir,
              'nframes': nframes,
              'split': split,
              'ds_len': ds_len
              }
    )


def get_linemodvideo(name):
    data_dir = 'data/blenderproc/physics_positioning/'
    scene, firstend, split = name.split('_')[1:]
    first_frame, last_frame = map(int, firstend.split('to'))
    # training_factor = float(training_factor[2:])
    return dict(
        factory='LinemodVideo',
        args={'data_dir': data_dir,
              'scene': scene,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              # 'training_factor': training_factor, todo
              }
    )


def get_epic(name):
    # epic_P01_01_train
    data_dir = osp.expanduser('~/Datasets/EPIC-Diff')
    _, a, b, split = name.split('_')
    vid = a + "_" + b
    # root, vid, split
    return dict(
        factory='EPICDiff',
        args={'root': data_dir,
              'vid': vid,
              'split': split,
              }
    )
