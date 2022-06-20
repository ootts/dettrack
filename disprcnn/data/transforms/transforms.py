import random

import numpy as np
import torch
import torchvision
from dl_ext.primitive import safe_zip
from dl_ext.timer import EvalTime
from torchvision.transforms import functional as F

from disprcnn.utils.pn_utils import stack


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, target=None, **kwargs):
        if target is None:
            for t in self.transforms:
                inputs = t(inputs, **kwargs)
            return inputs
        else:
            for t in self.transforms:
                inputs, target = t(inputs, target, **kwargs)
            return inputs, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

    def get_voxel_sizes(self):
        for t in self.transforms:
            if hasattr(t, 'voxel_sizes'):
                return getattr(t, 'voxel_sizes')
        return None


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, image, target=None, **kwargs):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        if hasattr(target, 'resize'):
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None, **kwargs):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None and hasattr(target, 'transpose'):
                target = target.transpose(0)
        if target is None:
            return image
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target=None, **kwargs):
        image = self.color_jitter(image)
        if target is None:
            return image
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None, **kwargs):
        image = F.to_tensor(image)
        if target is None:
            return image
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, **kwargs):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class Voxelization(object):
    def __init__(self, voxel_sizes, feature):
        if isinstance(voxel_sizes, float):
            voxel_sizes = [voxel_sizes]
        self.voxel_sizes = voxel_sizes
        self.feature = feature  # tsdf/color

    def __call__(self, inputs, targets=None, **kwargs):
        sample0 = inputs['sample0']
        sample1 = inputs['sample1']
        for voxel_size in self.voxel_sizes:
            sample0 = self.voxelize_one_sample(sample0, voxel_size)
            sample1 = self.voxelize_one_sample(sample1, voxel_size)
        sample0['pixelloc_to_voxelidx'] = np.stack(sample0['pixelloc_to_voxelidxs'])
        sample0.pop('pixelloc_to_voxelidxs')
        sample1['pixelloc_to_voxelidx'] = np.stack(sample1['pixelloc_to_voxelidxs'])
        sample1.pop('pixelloc_to_voxelidxs')
        inputs['sample0'] = sample0
        inputs['sample1'] = sample1
        if targets is None:
            return inputs
        else:
            raise NotImplementedError()
            voxel_flow3d = self.voxelize_3d_flow(inputs['sample0']['coord'], targets['flow3d'])
            targets['voxel_flow3d'] = voxel_flow3d
            return inputs, targets

    def voxelize_one_sample(self, sample, voxel_size):
        coord, feat = sample['coord'], sample[self.feature]
        rounded_coord = np.round(coord / voxel_size).astype(np.int32)
        inds, invs = sparse_quantize(rounded_coord, feat,
                                     return_index=True,
                                     return_invs=True)
        voxel_pc = rounded_coord[inds]
        voxel_feat = feat[inds]
        if 'voxel_pc' not in sample:
            sample['voxel_pc'] = []
        if 'voxel_feat' not in sample:
            sample['voxel_feat'] = []
        sample['voxel_pc'].append(voxel_pc)
        sample['voxel_feat'].append(voxel_feat)
        if 'pixelloc_to_voxelidxs' not in sample:
            sample['pixelloc_to_voxelidxs'] = []
        if 'pixelloc_to_voxelidx' in sample:
            pixelloc_to_voxelidx = sample['pixelloc_to_voxelidx'].copy().reshape(-1)
            pixelloc_to_voxelidx[pixelloc_to_voxelidx != -1] = invs
            sample['pixelloc_to_voxelidxs'].append(pixelloc_to_voxelidx.reshape(sample['pixelloc_to_voxelidx'].shape))
        return sample

    def voxelize_3d_flow(self, coord, flow3d):
        rounded_coord = np.round(coord / self.voxel_size).astype(np.int32)
        inds = sparse_quantize(rounded_coord, flow3d,
                               return_index=True,
                               return_invs=True)[0]
        voxel_pc = rounded_coord[inds]
        voxel_flow3d = flow3d[inds]
        voxel_flow3d = voxel_flow3d / self.voxel_size
        return voxel_flow3d


class ToSparseTensor(object):
    def __call__(self, inputs, targets=None, **kwargs):
        inputs['sample0'] = self.to_sparse_tensor_for_one_sample(inputs['sample0'])
        inputs['sample1'] = self.to_sparse_tensor_for_one_sample(inputs['sample1'])
        if targets is None:
            return inputs
        else:
            return inputs, targets

    def to_sparse_tensor_for_one_sample(self, sample):
        voxel_feats = sample['voxel_feat']
        voxel_pcs = sample['voxel_pc']
        if 'sparse_tensor' not in sample:
            sample['sparse_tensor'] = []
        for voxel_feat, voxel_pc in safe_zip(voxel_feats, voxel_pcs):
            sparse_tensor = ts.SparseTensor(torch.as_tensor(voxel_feat),
                                            torch.as_tensor(voxel_pc))
            sample['sparse_tensor'].append(sparse_tensor)
        return sample


class ToPointTensor(object):
    def __init__(self, feature):
        self.feature = feature

    def __call__(self, inputs, targets=None, **kwargs):
        evaltime = EvalTime(disable=True)
        evaltime('')
        inputs['sample0'] = self.to_point_tensor_for_one_sample(inputs['sample0'])
        if 'sample1' in inputs:
            inputs['sample1'] = self.to_point_tensor_for_one_sample(inputs['sample1'])
        evaltime('to point tensor')
        if targets is None:
            return inputs
        else:
            return inputs, targets

    def to_point_tensor_for_one_sample(self, sample):
        if self.feature == 'one':
            point_feats = np.ones((sample['coord'].shape[0], 1), dtype=np.float32)
        elif self.feature == 'one+coord':
            point_feats = np.ones((sample['coord'].shape[0], 1), dtype=np.float32)
            point_feats = np.concatenate((point_feats, sample['coord']), axis=1)
        else:
            point_feats = sample[self.feature]
        point_pcs = sample['coord']
        sample['point_tensor'] = ts.PointTensor(torch.as_tensor(point_feats),
                                                torch.as_tensor(point_pcs))
        return sample


class SpvcnnVoxelization(object):
    def __init__(self, voxel_sizes, late=False):
        if isinstance(voxel_sizes, float):
            voxel_sizes = [voxel_sizes]
        self.voxel_sizes = voxel_sizes
        # self.feature = feature  # one/tsdf
        self.late = late

    def __call__(self, inputs, targets=None, **kwargs):
        if not self.late:
            evaltime = EvalTime(disable=True)
            evaltime('')
            trans_sample1 = 'sample1' in inputs
            sample0 = inputs['sample0']
            if trans_sample1: sample1 = inputs['sample1']
            for vi, voxel_size in enumerate(self.voxel_sizes):
                sample0 = self.voxelize_one_sample(sample0, voxel_size, vi)
                if trans_sample1:
                    sample1 = self.voxelize_one_sample(sample1, voxel_size, vi)
            evaltime('voxelize samples')
            sample0['pixelloc_to_voxelidx'] = stack(sample0['pixelloc_to_voxelidxs'])
            sample0.pop('pixelloc_to_voxelidxs')
            if isinstance(sample0['pixelloc_to_voxelidx'], list) and len(sample0['pixelloc_to_voxelidx']) == 0:
                sample0.pop('pixelloc_to_voxelidx')
            if trans_sample1:
                sample1['pixelloc_to_voxelidx'] = stack(sample1['pixelloc_to_voxelidxs'])
                sample1.pop('pixelloc_to_voxelidxs')
                if isinstance(sample1['pixelloc_to_voxelidx'], list) and len(sample1['pixelloc_to_voxelidx']) == 0:
                    sample1.pop('pixelloc_to_voxelidx')
            inputs['sample0'] = sample0
            if trans_sample1: inputs['sample1'] = sample1
            evaltime('spvcnn voxelization')
        if targets is None:
            return inputs
        else:
            raise NotImplementedError()

    def voxelize_one_sample(self, sample, voxel_size, i):
        from disprcnn.utils.ts_utils import pad_batch_idx, remove_batch_idx

        if 'point_tensors' in sample and 'sparse_tensors' in sample and len(sample['point_tensors']) > i and len(
                sample['sparse_tensors']) > i:
            z = sample['point_tensors'][0]
            x = sample['sparse_tensors'][0]
        else:
            z = pad_batch_idx(sample['point_tensor'].clone())
            x, z = initial_voxelize_cpu(z, 1.0, voxel_size)
            x = remove_batch_idx(x)
            z = remove_batch_idx(z)
            if 'sparse_tensors' not in sample:
                sample['sparse_tensors'] = []
            if 'point_tensors' not in sample:
                sample['point_tensors'] = []
            sample['sparse_tensors'].append(x)
            sample['point_tensors'].append(z)

        invs = z.additional_features['idx_query'][1]
        if 'pixelloc_to_voxelidxs' not in sample:
            sample['pixelloc_to_voxelidxs'] = []
        if 'pixelloc_to_voxelidx' in sample:
            pixelloc_to_voxelidx = sample['pixelloc_to_voxelidx'].copy().reshape(-1)
            pixelloc_to_voxelidx[pixelloc_to_voxelidx != -1] = invs
            sample['pixelloc_to_voxelidxs'].append(pixelloc_to_voxelidx.reshape(sample['pixelloc_to_voxelidx'].shape))
        return sample


class ClipRange(object):
    def __init__(self, range):
        """
        Keep point cloud only in specified range.
        :param range: xmin,ymin,zmin,xmax,ymax,zmax
        """
        self.range = range

    def __call__(self, inputs, targets=None, **kwargs):
        range = kwargs.get('range', self.range)
        coord0 = inputs['sample0']['coord']
        keep0 = np.logical_and.reduce([
            coord0[:, 0] > range[0],
            coord0[:, 1] > range[1],
            coord0[:, 2] > range[2],
            coord0[:, 0] < range[3],
            coord0[:, 1] < range[4],
            coord0[:, 2] < range[5]
        ])
        inputs['sample0']['coord'] = inputs['sample0']['coord'][keep0]
        inputs['sample0']['color'] = inputs['sample0']['color'][keep0]
        inputs['sample0']['pixelloc_to_voxelidx'].fill(-1)
        inputs['sample0']['pixelloc_to_voxelidx'].reshape(-1)[keep0.nonzero()[0]] = np.arange(
            keep0.nonzero()[0].shape[0])
        if targets is not None:
            targets['flow3d'] = targets['flow3d'][keep0]
        coord1 = inputs['sample1']['coord']
        keep1 = np.logical_and.reduce([
            coord1[:, 0] > range[0],
            coord1[:, 1] > range[1],
            coord1[:, 2] > range[2],
            coord1[:, 0] < range[3],
            coord1[:, 1] < range[4],
            coord1[:, 2] < range[5]
        ])
        inputs['sample1']['coord'] = inputs['sample1']['coord'][keep1]
        inputs['sample1']['color'] = inputs['sample1']['color'][keep1]
        inputs['sample1']['pixelloc_to_voxelidx'].fill(-1)
        inputs['sample1']['pixelloc_to_voxelidx'].reshape(-1)[keep1.nonzero()[0]] = np.arange(
            keep1.nonzero()[0].shape[0])

        if targets is None:
            return inputs
        else:
            return inputs, targets


class CenterCrop:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, inputs, targets=None, **kwargs):
        oh, ow = inputs['image'].shape[:2]
        ch = oh // 2
        cw = ow // 2
        sh = ch - self.height // 2
        eh = sh + self.height
        sw = cw - self.width // 2
        ew = sw + self.width
        inputs['image'] = inputs['image'][sh:eh, sw:ew]
        inputs['depth'] = inputs['depth'][sh:eh, sw:ew]
        inputs['mask'] = inputs['mask'][sh:eh, sw:ew]
        inputs['forward_flow'] = inputs['forward_flow'][sh:eh, sw:ew]
        inputs['backward_flow'] = inputs['backward_flow'][sh:eh, sw:ew]
        inputs['H'] = self.height
        inputs['W'] = self.width
        inputs['K'][0, 2] = inputs['K'][0, 2] - sw
        inputs['K'][1, 2] = inputs['K'][1, 2] - sh
        return inputs
