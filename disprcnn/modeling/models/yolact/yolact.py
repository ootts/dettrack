from collections import OrderedDict

import cv2
import numpy as np
import torch

from disprcnn.modeling.models.yolact.backbone import construct_backbone
# from data.config import cfg, mask_type
from disprcnn.modeling.models.yolact.layers import Detect, MultiBoxLoss
from disprcnn.modeling.models.yolact.layers.output_utils import undo_image_transformation
# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
from disprcnn.modeling.models.yolact.submodules import *
from disprcnn.structures.apdataobject import APDataObject
from disprcnn.structures.bounding_box import BoxList
from disprcnn.utils.plt_utils import COLORS, show
from disprcnn.modeling.models.yolact.layers.box_utils import crop, sanitize_coordinates, mask_iou, jaccard
from disprcnn.utils.timer import EvalTime


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
            self.maskiou_net = FastMaskIoUNet(self.cfg)

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

    def forward(self, dps, return_features=False):
        x = dps['image']
        _, _, img_h, img_w = x.size()
        _tmp_img_h = img_h
        _tmp_img_w = img_w

        outs = self.backbone(x)

        if self.cfg.fpn is not None:
            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [outs[i] for i in self.cfg.backbone.selected_layers]
            outs = self.fpn(outs)

        proto_x = x if self.proto_src is None else outs[self.proto_src]

        if self.num_grids > 0:
            grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
            proto_x = torch.cat([proto_x, grids], dim=1)

        proto_out = self.proto_net(proto_x)
        proto_out = F.relu(proto_out, inplace=True)

        # Move the features last so the multiplication is easy
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]

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
            assert not self.cfg.use_mask_scoring
            # pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            assert not self.cfg.use_focal_loss
            #     if self.cfg.use_sigmoid_focal_loss:
            #         # Note: even though conf[0] exists, this mode doesn't train it so don't use it
            #         pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
            #         if self.cfg.use_mask_scoring:
            #             pred_outs['conf'] *= pred_outs['score']
            #     elif self.cfg.use_objectness_score:
            #         # See focal_loss_sigmoid in multibox_loss.py for details
            #         objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
            #         pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
            #         pred_outs['conf'][:, :, 0] = 1 - objectness
            #     else:
            #         pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            # else:
            assert not self.cfg.use_objectness_score
            #     objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
            #     pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
            #                                   * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
            # else:
            pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            rets = self.detect(pred_outs, self)
            if return_features:
                rets = rets, outs
            return rets

    def forward_onnx(self, image):
        x = image
        _, _, img_h, img_w = x.size()
        _tmp_img_h = img_h
        _tmp_img_w = img_w

        outs = self.backbone(x)

        if self.cfg.fpn is not None:
            # Use backbone.selected_layers because we overwrote self.selected_layers
            outs = [outs[i] for i in self.cfg.backbone.selected_layers]
            outs = self.fpn(outs)

        proto_x = x if self.proto_src is None else outs[self.proto_src]

        if self.num_grids > 0:
            grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
            proto_x = torch.cat([proto_x, grids], dim=1)

        proto_out = self.proto_net(proto_x)
        proto_out = F.relu(proto_out, inplace=True)

        # Move the features last so the multiplication is easy
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]

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

        assert not self.training
        assert not self.cfg.use_mask_scoring
        # pred_outs['score'] = torch.sigmoid(pred_outs['score'])

        assert not self.cfg.use_focal_loss
        assert not self.cfg.use_objectness_score
        pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
        unified_out = torch.zeros([image.shape[0], 11481, 4 + 2 + 32 + 4 + 32 + 75], device='cuda', dtype=torch.float32)
        unified_out[:, :, :4] = pred_outs['loc']
        unified_out[:, :, 4:4 + 2] = pred_outs['conf']
        unified_out[:, :, 4 + 2:4 + 2 + 32] = pred_outs['mask']
        unified_out[:, :, 4 + 2 + 32:4 + 2 + 32 + 4] = pred_outs['priors'][None].repeat(image.shape[0], 1, 1)
        proto_out = pred_outs['proto'].reshape(image.shape[0], -1, 32)
        unified_out[:, :proto_out.shape[1], 4 + 2 + 32 + 4:4 + 2 + 32 + 4 + 32] = proto_out
        feat_out = outs[0]
        feat_out = feat_out.reshape(feat_out.shape[0], -1, feat_out.shape[-1])
        unified_out[:, :feat_out.shape[1], 4 + 2 + 32 + 4 + 32:] = feat_out

        return unified_out


class YolactWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact
        self.model = Yolact(cfg)
        self.criterion = MultiBoxLoss(cfg.model.yolact)
        # for evaluation
        self.iou_thresholds = [x / 100 for x in range(50, 100, 5)]
        self.ap_data = {
            'box': [[APDataObject() for _ in self.cfg.class_names] for _ in self.iou_thresholds],
            'mask': [[APDataObject() for _ in self.cfg.class_names] for _ in self.iou_thresholds]
        }
        # detections = Detections()

    def forward(self, dps):
        outputs = self.model(dps)
        if not self.training:
            results = []
            bsz = dps['width'].shape[0]
            for bi in range(bsz):
                w, h = dps['width'][bi].item(), dps['height'][bi].item()
                classes, scores, boxes, masks = self.postprocess(outputs[bi], w, h,
                                                                 score_threshold=0.05,
                                                                 to_long=False)
                result = BoxList(boxes, (w, h))
                result.add_field("labels", classes)
                result.add_field("scores", scores)
                result.add_field("masks", masks)
                results.append(result)
                if self.total_cfg.dbg:
                    img_numpy = self.prep_display(result, dps['image'][bi], h, w)
                    show(img_numpy)
                    print()
                elif self.total_cfg.model.meta_architecture == 'Yolact':
                    # et('begin')
                    gt = torch.cat([dps['target'][bi].bbox, dps['target'][bi].get_field('labels')[:, None]], dim=1)
                    self.prep_metrics(self.ap_data, result, dps['image'][bi], gt,
                                      dps['target'][bi].get_field('masks'),
                                      dps['height'][bi].item(),
                                      dps['width'][bi].item(),
                                      dps['num_crowds'][bi].item(),
                                      dps['imgid'][bi].item(),
                                      None
                                      )
                    if 'is_last_frame' in dps and dps['is_last_frame'][bi].item() is True:
                        self.calc_map()
            return results, {}
        else:
            targets = [torch.cat([boxlist.bbox, boxlist.get_field('labels').reshape(-1, 1)
                                  ], dim=1) for boxlist in dps['target']]
            masks = [boxlist.get_field('masks').float() for boxlist in dps['target']]
            num_crowds = dps['num_crowds']
            losses = self.criterion(self.model, outputs, targets, masks, num_crowds, self.model.mask_dim)
            return outputs, losses

    def prep_display(self, result, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
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
        save = self.cfg.rescore_bbox
        # self.cfg.defrost()
        # self.cfg.rescore_bbox = True
        # t = self.postprocess(dets_out, w, h, visualize_lincomb=False,
        #                      crop_masks=True,
        #                      score_threshold=self.cfg.score_threshold)
        # self.cfg.rescore_bbox = save
        # classes, scores, boxes, masks
        classes = result.get_field("labels")
        scores = result.get_field("scores")
        masks = result.get_field("masks")
        boxes = result.bbox
        idx = scores.argsort(0, descending=True)[:self.cfg.top_k]

        if self.cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = masks[idx]
        # classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        classes = classes[idx].cpu().numpy()
        scores = scores[idx].cpu().numpy()
        boxes = boxes[idx].cpu().numpy()
        num_dets_to_consider = min(self.cfg.top_k, classes.shape[0])
        if num_dets_to_consider == 0:
            return (img_gpu * 255).byte().cpu().numpy()
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
            return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        # if self.cfg.display_masks and self.cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider][..., None]

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

        # if num_dets_to_consider == 0:
        #     return img_numpy

        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = map(int, boxes[j, :])
            color = get_color(j)
            score = scores[j]

            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

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

    def postprocess(self, det_output, w, h, interpolation_mode='bilinear',
                    crop_masks=True, score_threshold=0, to_long=True):
        """
        Postprocesses the output of Yolact on testing mode into a format that makes sense,
        accounting for all the possible configuration settings.

        Args:
            - det_output: The lost of dicts that Detect outputs.
            - w: The real with of the image.
            - h: The real height of the image.
            - batch_idx: If you have multiple images for this batch, the image's index in the batch.
            - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

        Returns 4 torch Tensors (in the following order):
            - classes [num_det]: The class idx for each detection.
            - scores  [num_det]: The confidence score for each detection.
            - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
            - masks   [num_det, h, w]: Full image masks for each detection.
        """

        dets = det_output
        dets = dets['detection']

        if dets is None:
            # classes, scores, boxes, masks
            return torch.empty([0, ]), torch.empty([0, ]), torch.empty([0, 4]), torch.empty([0, 1, 1])
            # return [torch.Tensor([[]])] * 4  # Warning, this is 4 copies of the same thing

        if score_threshold > 0:
            keep = dets['score'] > score_threshold

            for k in dets:
                if k != 'proto':
                    dets[k] = dets[k][keep]

            if dets['score'].size(0) == 0:
                return [torch.Tensor()] * 4

        # Actually extract everything from dets now
        classes = dets['class']
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']

        # if cfg.mask_type == 1 and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']

        # Test flag, do not upvote
        # if cfg.mask_proto_debug:
        #     np.save('scripts/proto.npy', proto_data.cpu().numpy())

        # if visualize_lincomb:
        #     display_lincomb(proto_data, masks)

        masks = proto_data @ masks.t()
        masks = torch.sigmoid(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        if self.cfg.use_maskiou:
            with torch.no_grad():
                maskiou_p = self.model.maskiou_net(masks.unsqueeze(1))
                maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
                if self.cfg.rescore_mask:
                    if self.cfg.rescore_bbox:
                        scores = scores * maskiou_p
                    else:
                        scores = [scores, scores * maskiou_p]

        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
        if to_long:
            boxes = boxes.long()

        return classes, scores, boxes, masks

    def prep_metrics(self, ap_data, result, img, gt, gt_masks, h, w, num_crowd, image_id, detections):
        """ Returns a list of APs for this image, with each element being for a class  """
        gt_boxes = gt[:, :4]
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h
        gt_classes = gt[:, 4].long().tolist()
        gt_masks = gt_masks.view(-1, h * w)

        if num_crowd > 0:
            split = lambda x: (x[-num_crowd:], x[:-num_crowd])
            crowd_boxes, gt_boxes = split(gt_boxes)
            crowd_masks, gt_masks = split(gt_masks)
            crowd_classes, gt_classes = split(gt_classes)

        # classes, scores, boxes, masks = self.postprocess(dets, w, h, crop_masks=True,
        #                                                  score_threshold=0)

        if len(result) == 0:
            return

        classes = result.get_field("labels").tolist()
        scores = result.get_field("scores").tolist()
        masks = result.get_field("masks")
        boxes = result.bbox
        # if isinstance(scores, list):
        #     box_scores = list(scores[0].cpu().numpy().astype(float))
        #     mask_scores = list(scores[1].cpu().numpy().astype(float))
        # else:
        #     scores = list(scores.cpu().numpy().astype(float))
        box_scores = scores
        mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

        for _class in set(classes + gt_classes):
            ap_per_iou = []
            num_gt_for_class = sum([1 for x in gt_classes if x == _class])

            for iouIdx in range(len(self.iou_thresholds)):
                iou_threshold = self.iou_thresholds[iouIdx]

                for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                    gt_used = [False] * len(gt_classes)

                    ap_obj = ap_data[iou_type][iouIdx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class)

                    for i in indices:
                        if classes[i] != _class:
                            continue

                        max_iou_found = iou_threshold
                        max_match_idx = -1
                        for j in range(num_gt):
                            if gt_used[j] or gt_classes[j] != _class:
                                continue

                            iou = iou_func(i, j)

                            if iou > max_iou_found:
                                max_iou_found = iou
                                max_match_idx = j

                        if max_match_idx >= 0:
                            gt_used[max_match_idx] = True
                            ap_obj.push(score_func(i), True)
                        else:
                            # If the detection matches a crowd, we can just ignore it
                            matched_crowd = False

                            if num_crowd > 0:
                                for j in range(len(crowd_classes)):
                                    if crowd_classes[j] != _class:
                                        continue

                                    iou = crowd_func(i, j)

                                    if iou > iou_threshold:
                                        matched_crowd = True
                                        break

                            # All this crowd code so that we can make sure that our eval code gives the
                            # same result as COCOEval. There aren't even that many crowd annotations to
                            # begin with, but accuracy is of the utmost importance.
                            if not matched_crowd:
                                ap_obj.push(score_func(i), False)

    def calc_map(self):
        print('Calculating mAP...')
        aps = [{'box': [], 'mask': []} for _ in self.iou_thresholds]

        for _class in range(len(self.cfg.class_names)):
            for iou_idx in range(len(self.iou_thresholds)):
                for iou_type in ('box', 'mask'):
                    ap_obj = self.ap_data[iou_type][iou_idx][_class]

                    if not ap_obj.is_empty():
                        aps[iou_idx][iou_type].append(ap_obj.get_ap())

        all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

        # Looking back at it, this code is really hard to read :/
        for iou_type in ('box', 'mask'):
            all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
            for i, threshold in enumerate(self.iou_thresholds):
                mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
                all_maps[iou_type][int(threshold * 100)] = mAP
            all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

        self.print_maps(all_maps)

        # Put in a prettier format so we can serialize it to json during training
        all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
        return all_maps

    def print_maps(self, all_maps):
        # Warning: hacky
        make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
        make_sep = lambda n: ('-------+' * n)

        print()
        print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
        print(make_sep(len(all_maps['box']) + 1))
        for iou_type in ('box', 'mask'):
            print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
        print(make_sep(len(all_maps['box']) + 1))
        print()


def _mask_iou(mask1, mask2, iscrowd=False):
    ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()
