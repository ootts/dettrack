import cv2
from numpy import random
from math import sqrt

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

    def __call__(self, dps):
        for t in self.transforms:
            dps = t(dps)
        return dps

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


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


MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, dps):
        image, masks, boxes, labels = dps['image'], dps['masks'], dps['boxes'], dps['labels']
        if random.randint(2):
            return dps

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros(
            (masks.shape[0], int(height * ratio), int(width * ratio)),
            dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height),
        int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        dps['image'] = image
        dps['masks'] = masks
        dps['boxes'] = boxes
        dps['labels'] = labels
        return dps


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, dps):
        image, masks, boxes, labels = dps['image'], dps['masks'], dps['boxes'], dps['labels']
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return dps

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]
                dps['image'] = current_image
                dps['masks'] = current_masks
                dps['boxes'] = current_boxes
                dps['labels'] = current_labels
                return dps


class ToAbsoluteCoords(object):
    def __call__(self, dps):
        height, width, channels = dps['image'].shape
        dps['boxes'][:, 0] *= width
        dps['boxes'][:, 2] *= width
        dps['boxes'][:, 1] *= height
        dps['boxes'][:, 3] *= height

        return dps


class ConvertFromInts(object):
    def __call__(self, dps):
        image = dps['image']
        dps['image'] = image.astype(np.float32)
        return dps


class ToPercentCoords(object):
    def __call__(self, dps):
        height, width, channels = dps['image'].shape
        dps['boxes'][:, 0] /= width
        dps['boxes'][:, 2] /= width
        dps['boxes'][:, 1] /= height
        dps['boxes'][:, 3] /= height

        return dps


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """

    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, dps):
        image, masks, boxes, labels = dps['image'], dps['masks'], dps['boxes'], dps['labels']
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks
        dps['image'] = expand_image
        dps['masks'] = masks
        dps['boxes'] = boxes
        dps['labels'] = labels
        return dps


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, max_size, preserve_aspect_ratio, discard_box_width, discard_box_height, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = max_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.discard_box_width = discard_box_width
        self.discard_box_height = discard_box_height

    def __call__(self, dps):
        image = dps['image']
        img_h, img_w, _ = image.shape

        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt and 'masks' in dps:
            # Act like each object is a color channel
            masks = dps['masks'].transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))
            dps['masks'] = masks
            # Scale bounding boxes (which are currently absolute coordinates)
            if 'boxes' in dps:
                dps['boxes'][:, [0, 2]] *= (width / img_w)
                dps['boxes'][:, [1, 3]] *= (height / img_h)
        if 'boxes' in dps:
            # Discard boxes that are smaller than we'd like
            w = dps['boxes'][:, 2] - dps['boxes'][:, 0]
            h = dps['boxes'][:, 3] - dps['boxes'][:, 1]

            keep = (w > self.discard_box_width) * (h > self.discard_box_height)
            dps['masks'] = dps['masks'][keep]
            dps['boxes'] = dps['boxes'][keep]
            dps['labels']['labels'] = dps['labels']['labels'][keep]
            dps['labels']['num_crowds'] = (dps['labels']['labels'] < 0).sum()
        dps['image'] = image
        return dps


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, dps):
        im = dps['image'].copy()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        dps['image'] = self.rand_light_noise(im)
        return dps


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size. This
    function doesn't do anything if cfg.use_gt_bboxes is False.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, dps):
        image, masks, boxes, labels = dps['image'], dps['masks'], dps['boxes'], dps['labels']
        if not self.use_gt_bboxes:
            return dps

        height, width, _ = image.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            # +1 So that if y1=10.6 and y2=10.9 we still have a bounding box
            cropped_mask = masks[i, y1:(y2 + 1), x1:(x2 + 1)]
            scaled_mask = cv2.resize(cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)

        # Binarize
        new_masks[new_masks > 0.5] = 1
        new_masks[new_masks <= 0.5] = 0
        dps['image'] = image
        dps['masks'] = new_masks
        dps['boxes'] = boxes
        dps['labels'] = labels
        return image, new_masks, boxes, labels


class RandomMirror(object):
    def __call__(self, dps):
        _, width, _ = dps['image'].shape
        if random.randint(2):
            dps['image'] = dps['image'][:, ::-1]
            dps['masks'] = dps['masks'][:, :, ::-1]
            dps['boxes'][:, 0::2] = width - dps['boxes'][:, 2::-2]
        return dps


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """

    def __init__(self, mean, std, in_channel_order, channel_order, normalize, subtract_means, to_float):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.in_channel_order = in_channel_order
        self.channel_order = channel_order
        self.normalize = normalize
        self.subtract_means = subtract_means
        self.to_float = to_float

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in channel_order]

    def __call__(self, dps):
        image = dps['image']

        image = image.astype(np.float32)

        if self.normalize:
            image = (image - self.mean) / self.std
        elif self.subtract_means:
            image = (image - self.mean)
        elif self.to_float:
            image = image / 255

        image = image[:, :, self.channel_permutation]
        image = image.astype(np.float32)
        dps['image'] = image
        return dps
