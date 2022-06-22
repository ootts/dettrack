import cv2
import numpy as np

from disprcnn.modeling.models.yolact.backbone import construct_backbone
# from data.config import cfg, mask_type
from disprcnn.modeling.models.yolact.layers import Detect, MultiBoxLoss
from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation, postprocess
# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
from disprcnn.modeling.models.yolact.submodules import *
from disprcnn.utils.plt_utils import COLORS, show


class Yolact(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact
        self.backbone = construct_backbone(self.cfg.backbone)

        if self.cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if self.cfg.mask_type == 0:
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

        if self.cfg.track_head.on:
            self.track_head = None

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
            if self.total_cfg.dbg:
                img_numpy = self.prep_display(outputs, dps['image'][0], dps['height'][0].item(), dps['width'][0].item())
                show(img_numpy)
                print()
            for o, idx in zip(outputs, dps['index'].tolist()):
                if o['detection'] is None:
                    o['detection'] = {}
                o['detection']['index'] = idx
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
