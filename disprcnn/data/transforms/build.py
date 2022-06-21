from . import transforms as T
from .transforms import *


def build_transforms(cfg, is_train=True):
    ts = []
    transforms = cfg.input.transforms
    if not is_train and len(cfg.input.transforms_test) > 0:
        transforms = cfg.input.transforms_test
    for transform in transforms:
        ts.append(build_transform(transform))
    transform = T.Compose(ts)
    return transform


def build_transform(t):
    if t['name'] == 'ConvertFromInts':
        return ConvertFromInts()
    elif t['name'] == 'ToAbsoluteCoords':
        return ToAbsoluteCoords()
    elif t['name'] == 'PhotometricDistort':
        return PhotometricDistort()
    elif t['name'] == 'Expand':
        return Expand(t['mean'])
    elif t['name'] == 'RandomSampleCrop':
        return RandomSampleCrop()
    elif t['name'] == 'RandomMirror':
        return RandomMirror()
    elif t['name'] == 'Resize':
        return Resize(t['max_size'], t['preserve_aspect_ratio'],
                      t['discard_box_width'], t['discard_box_height'], t['resize_gt'])
    elif t['name'] == 'Pad':
        return Pad(t['width'], t['height'], t['mean'])
    elif t['name'] == 'ToPercentCoords':
        return ToPercentCoords()
    elif t['name'] == 'PrepareMasks':
        return PrepareMasks(t['mask_size'], t['use_gt_bboxes'])
    elif t['name'] == 'BackboneTransform':
        return BackboneTransform(t['mean'], t['std'], t['in_channel_order'],
                                 t['channel_order'], t['normalize'], t['subtract_means'],
                                 t['to_float'])
    elif t['name'] == 'CenterCrop':
        return CenterCrop(t['width'], t['height'])
    else:
        raise NotImplementedError()
