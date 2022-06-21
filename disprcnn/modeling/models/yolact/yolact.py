import cv2
import torch
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation, postprocess
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

# from data.config import cfg, mask_type
from disprcnn.modeling.models.yolact.layers import Detect, MultiBoxLoss
from disprcnn.modeling.models.yolact.backbone import construct_backbone

import torch.backends.cudnn as cudnn

# from utils import timer
# from utils.functions import MovingAverage, make_net

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
# torch.cuda.current_device()

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
from disprcnn.modeling.models.yolact.utils.functions import make_net
from disprcnn.utils.plt_utils import COLORS

use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

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

        if self.cfg.extra_head_net is not None:
            x = src.upfeature(x)

        if self.cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)

            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)

            # TODO: Possibly switch this out for a product
            x = a + b

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
        self.maskiou_net, _ = make_net(input_channels, self.cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p


class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact
        self.backbone = construct_backbone(self.cfg.backbone)

        # todo: init backbone weights

        if self.cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if self.cfg.mask_type == 0:  # todo:???
            mask_dim = cfg.mask_size ** 2
        elif self.cfg.mask_type == 1:
            if self.cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(self.cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = self.cfg.mask_proto_src

            if self.proto_src is None:
                in_channels = 3
            elif self.cfg.fpn is not None:
                in_channels = self.cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, mask_dim = make_net(in_channels, self.cfg.mask_proto_net,
                                                include_last_relu=False)

            if self.cfg.mask_proto_bias:
                mask_dim += 1
        self.mask_dim = mask_dim
        self.selected_layers = self.cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if self.cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()

        if self.cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN(self.cfg, [src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + self.cfg.fpn.num_downsample))
            src_channels = [self.cfg.fpn.num_features] * len(self.selected_layers)

        self.prediction_layers = nn.ModuleList()
        self.num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if self.cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(self.cfg, self.num_heads, mask_dim,
                                    src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios=self.cfg.backbone.pred_aspect_ratios[idx],
                                    scales=self.cfg.backbone.pred_scales[idx],
                                    parent=parent,
                                    index=idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        if self.cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], self.cfg.num_classes - 1)

        if self.cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], self.cfg.num_classes - 1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(self.cfg, self.cfg.num_classes, bkg_label=0, top_k=self.cfg.nms_top_k,
                             conf_thresh=self.cfg.nms_conf_thresh, nms_thresh=self.cfg.nms_thresh)
        if self.cfg.pretrained_backbone != '':
            self.init_weights(self.cfg.pretrained_backbone)
        self.freeze_bn(True)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if self.cfg.fpn is not None and int(key.split('.')[2]) >= self.cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                            all_in(module.__dict__['_constants_set'], conv_constants)
                            and all_in(conv_constants, module.__dict__['_constants_set']))

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if self.cfg.use_focal_loss and 'conf_layer' in name:
                        if not self.cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0] = np.log(
                                (1 - self.cfg.focal_loss_init_pi) / self.cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0] = -np.log(
                                self.cfg.focal_loss_init_pi / (1 - self.cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log(
                                (1 - self.cfg.focal_loss_init_pi) / self.cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

        if self.cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, dps):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        x = dps['image']
        _, _, img_h, img_w = x.size()
        _tmp_img_h = img_h
        _tmp_img_w = img_w

        outs = self.backbone(x)

        if self.cfg.fpn is not None:
            # with timer.env('fpn'):
            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [outs[i] for i in self.cfg.backbone.selected_layers]
            outs = self.fpn(outs)

        proto_out = None
        if self.cfg.mask_type == 1 and self.cfg.eval_mask_branch:
            proto_x = x if self.proto_src is None else outs[self.proto_src]

            if self.num_grids > 0:
                grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                proto_x = torch.cat([proto_x, grids], dim=1)

            proto_out = self.proto_net(proto_x)
            proto_out = F.relu(proto_out, inplace=True)

            if self.cfg.mask_proto_prototypes_as_features:
                # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                proto_downsampled = proto_out.clone()

                if self.cfg.mask_proto_prototypes_as_features_no_grad:
                    proto_downsampled = proto_out.detach()

            # Move the features last so the multiplication is easy
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

            if self.cfg.mask_proto_bias:
                bias_shape = [x for x in proto_out.size()]
                bias_shape[-1] = 1
                proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}

        if self.cfg.use_mask_scoring:
            pred_outs['score'] = []

        if self.cfg.use_instance_coeff:
            pred_outs['inst'] = []

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]

            if self.cfg.mask_type == 1 and self.cfg.mask_proto_prototypes_as_features:
                # Scale the prototypes down to the current prediction layer's size and add it as inputs
                proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear',
                                                  align_corners=False)
                pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

            # A hack for the way dataparallel works
            if self.cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                pred_layer.parent = [self.prediction_layers[0]]

            p = pred_layer(pred_x, _tmp_img_w, _tmp_img_h)

            for k, v in p.items():
                pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:
            # For the extra loss functions
            if self.cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if self.cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
        else:
            if self.cfg.use_mask_scoring:
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            if self.cfg.use_focal_loss:
                if self.cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if self.cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif self.cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:

                if self.cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])

                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                                                  * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)

                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            return self.detect(pred_outs, self)


class YolactWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact
        self.model = Yolact(cfg)
        self.criterion = MultiBoxLoss(cfg.model.yolact)

    def forward(self, dps):
        outputs = self.model(dps)
        targets = dps['targets']
        masks = dps['masks']
        num_crowds = dps['num_crowds']
        if not self.training:
            img_numpy = self.prep_display(outputs, dps['image'][0], dps['height'][0].item(), dps['width'][0].item())
            import matplotlib.pyplot as plt
            plt.imshow(img_numpy)
            plt.show()
            losses = {}
        else:
            losses = self.criterion(outputs, targets, masks, num_crowds, self.model.mask_dim)
        return outputs, losses

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h,
                                                  [103.94, 116.78, 123.68],
                                                  [57.38, 57.12, 58.40])
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        # save = cfg.rescore_bbox
        # rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=self.cfg.score_threshold)
        # cfg.rescore_bbox = save

        # with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:self.cfg.top_k]

        if self.cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.cfg.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.cfg.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            # global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            # if on_gpu is not None and color_idx in color_cache[on_gpu]:
            #     return color_cache[on_gpu][color_idx]
            # else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                # color_cache[on_gpu][color_idx] = color
            return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        # if self.cfg.display_masks and self.cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
            dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return img_numpy

        if True:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                # if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                # if args.display_text:
                _class = self.cfg.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score)

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

        return img_numpy
