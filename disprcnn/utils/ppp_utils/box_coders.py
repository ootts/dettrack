from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

# from second.core.box_coders import GroundBox3dCoder, BevBoxCoder

from disprcnn.utils.ppp_utils import box_np_ops, box_torch_ops


class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 8 if self.vec_encode else 7

    def _encode(self, boxes, anchors):
        return box_np_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        return box_np_ops.second_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)


class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode, self.linear_dim)

# class BevBoxCoderTorch(BevBoxCoder):
#     def encode_torch(self, boxes, anchors):
#         anchors = anchors[..., [0, 1, 3, 4, 6]]
#         boxes = boxes[..., [0, 1, 3, 4, 6]]
#         return box_torch_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)
#
#     def decode_torch(self, encodings, anchors):
#         anchors = anchors[..., [0, 1, 3, 4, 6]]
#         ret = box_torch_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
#         z_fixed = torch.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype, device=ret.device)
#         h_fixed = torch.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype, device=ret.device)
#         return torch.cat([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], dim=-1)
#
#
