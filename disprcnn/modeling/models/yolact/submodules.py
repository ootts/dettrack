from collections import defaultdict
from itertools import product
from math import sqrt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
from disprcnn.modeling.models.yolact.utils.functions import make_net

# from data.config import cfg, mask_type
# from utils import timer
# from utils.functions import MovingAverage, make_net
# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
# torch.cuda.current_device()

use_jit = torch.cuda.device_count() <= 1
# if not use_jit:
#     print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


prior_cache = defaultdict(lambda: None)


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, cfg, num_heads, mask_dim, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1],
                 parent=None,
                 index=0):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.mask_dim = mask_dim  # Defined by Yolact
        self.num_priors = sum(len(x) * len(scales) for x in aspect_ratios)
        self.parent = [parent]  # Don't include this in the state dict
        self.index = index
        self.num_heads = num_heads  # Defined by Yolact

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == 1:
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim

        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)

            if cfg.use_mask_scoring:
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs,
                                            **cfg.head_layer_params)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]

            if cfg.mask_type == 1 and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x, _tmp_img_w, _tmp_img_h):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        conv_h = x.size(2)
        conv_w = x.size(3)

        # if self.cfg.extra_head_net is not None:
        x = src.upfeature(x)

        # if self.cfg.use_prediction_module:
        #     # The two branches of PM design (c)
        #     a = src.block(x)
        #
        #     b = src.conv(x)
        #     b = src.bn(b)
        #     b = F.relu(b)
        #
        #     # TODO: Possibly switch this out for a product
        #     x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if self.cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if self.cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if self.cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.cfg.num_instance_coeffs)

            # See box_utils.decode for an explanation of this
        if self.cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if self.cfg.eval_mask_branch:
            if self.cfg.mask_type == 0:
                mask = torch.sigmoid(mask)
            elif self.cfg.mask_type == 1:
                mask = torch.tanh(mask)

                if self.cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        if self.cfg.mask_proto_split_prototypes_by_head and self.cfg.mask_type == 1:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim),
                         mode='constant', value=0)

        priors = self.make_priors(conv_h, conv_w, x.device, _tmp_img_w, _tmp_img_h)

        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}

        if self.cfg.use_mask_scoring:
            preds['score'] = score

        if self.cfg.use_instance_coeff:
            preds['inst'] = inst

        return preds

    def make_priors(self, conv_h, conv_w, device, _tmp_img_w, _tmp_img_h):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        if self.last_img_size != (_tmp_img_w, _tmp_img_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for ars in self.aspect_ratios:
                    for scale in self.scales:
                        for ar in ars:
                            if not self.cfg.backbone.preapply_sqrt:
                                ar = sqrt(ar)

                            if self.cfg.backbone.use_pixel_scales:
                                w = scale * ar / self.cfg.max_size
                                h = scale / ar / self.cfg.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h

                            # This is for backward compatability with a bug where I made everything square by accident
                            if self.cfg.backbone.use_square_anchors:
                                h = w

                            prior_data += [x, y, w, h]

            self.priors = torch.tensor(prior_data, device=device).view(-1, 4).detach()
            self.priors.requires_grad = False
            self.last_img_size = (_tmp_img_w, _tmp_img_h)
            self.last_conv_size = (conv_w, conv_h)
            prior_cache[size] = None
        elif self.priors.device != device:
            # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
            if prior_cache[size] is None:
                prior_cache[size] = {}

            if device not in prior_cache[size]:
                prior_cache[size][device] = self.priors.to(device)

            self.priors = prior_cache[size][device]

        return self.priors


class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, cfg, in_channels):
        super().__init__()

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.num_downsample = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers = cfg.fpn.relu_pred_layers

    @script_method_wrapper
    def forward(self, convouts: List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)

            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out


class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_channels = 1
        last_layer = [(self.cfg.num_classes - 1, 1, {})]
        maskiou_net = [(8, 3, {'stride': 2}), (16, 3, {'stride': 2}), (32, 3, {'stride': 2}), (64, 3, {'stride': 2}),
                       (128, 3, {'stride': 2})]
        self.maskiou_net, _ = make_net(input_channels, maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p
