from abc import ABCMeta, abstractmethod

import numpy as np

# from second.core import region_similarity
from disprcnn.utils.ppp_utils import box_np_ops


class RegionSimilarityCalculator(object):
    """Abstract base class for 2d region similarity calculator."""
    __metaclass__ = ABCMeta

    def compare(self, boxes1, boxes2):
        """Computes matrix of pairwise similarity between BoxLists.

        This op (to be overriden) computes a measure of pairwise similarity between
        the boxes in the given BoxLists. Higher values indicate more similarity.

        Note that this method simply measures similarity and does not explicitly
        perform a matching.

        Args:
          boxes1: [N, 5] [x,y,w,l,r] tensor.
          boxes2: [M, 5] [x,y,w,l,r] tensor.

        Returns:
          a (float32) tensor of shape [N, M] with pairwise similarity score.
        """
        return self._compare(boxes1, boxes2)

    @abstractmethod
    def _compare(self, boxes1, boxes2):
        pass


class NearestIouSimilarity(RegionSimilarityCalculator):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def _compare(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(boxes2)
        ret = box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret


def build_similarity_calculator():
    return NearestIouSimilarity()
