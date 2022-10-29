import numpy as np
import torch

from disprcnn.modeling.models.pointpillars import ops
from torchvision.ops.boxes import nms


def pytorch_nms(bboxes,
                scores,
                pre_max_size=None,
                post_max_size=None,
                iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    if len(bboxes) == 0:
        keep = torch.tensor([], dtype=torch.long).cuda()
    else:
        m = bboxes.min()
        bboxes = bboxes + m
        ret = nms(bboxes, scores, iou_threshold)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        # keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()


def main():
    bboxes, scores = torch.load('tmp/nms_data.pth')
    keep_numba = ops.nms(bboxes, scores, 1000, 300, 0.5)
    keep_pytorch = pytorch_nms(bboxes, scores, 1000, 300, 0.5)
    print()


if __name__ == '__main__':
    main()
